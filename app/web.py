"""FastAPI dashboard + JSON API (LAN only, token required).

Auth: `X-Token: <DASHBOARD_TOKEN>` or `Authorization: Bearer <token>` header (scripts, curl), or a cookie
set by opening `/?token=<token>` once in the browser (the token is then removed from the URL).
"""
from __future__ import annotations

import asyncio
import datetime as dt
import re
import secrets
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import summary

if TYPE_CHECKING:
    from .main import App

HERE = Path(__file__).parent
COOKIE = "pet_tracker_token"
CLIP_RE = re.compile(r"^[\w.-]+\.mp4$")
MJPEG_FPS = 5


def _placeholder(text: str, w: int, h: int) -> bytes:
    img = np.full((h, w, 3), 32, np.uint8)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.putText(img, text, ((w - tw) // 2, (h + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2,
                cv2.LINE_AA)
    return cv2.imencode(".jpg", img)[1].tobytes()


def create_app(core: "App") -> FastAPI:
    cfg = core.cfg
    token = cfg.dashboard_token
    api = FastAPI(title="pet-tracker", docs_url=None, redoc_url=None, openapi_url=None)
    templates = Jinja2Templates(directory=str(HERE / "templates"))
    api.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")
    off_jpg = _placeholder("Camera off", cfg.frame_width, cfg.frame_height)
    wait_jpg = _placeholder("Connecting...", cfg.frame_width, cfg.frame_height)

    def valid(candidate: str | None) -> bool:
        return bool(token) and bool(candidate) and secrets.compare_digest(candidate.encode(), token.encode())

    def presented(request: Request) -> str | None:
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            return auth[7:].strip()
        return request.headers.get("x-token") or request.cookies.get(COOKIE)

    def require_auth(request: Request) -> None:
        if not token:
            raise HTTPException(503, "DASHBOARD_TOKEN is not set in .env")
        if not valid(presented(request)):
            raise HTTPException(401, "missing or wrong token")

    auth = [Depends(require_auth)]

    def parse_day(date: str | None) -> dt.date:
        if not date:
            return core.today()
        try:
            return dt.date.fromisoformat(date)
        except ValueError:
            raise HTTPException(400, "date must be YYYY-MM-DD")

    async def body_flag(request: Request, current: bool) -> bool:
        """JSON {"enabled": bool} / {"tracking": bool}, or an empty body / ?toggle=1 to flip."""
        if request.query_params.get("toggle"):
            return not current
        raw = await request.body()
        if not raw.strip():
            return not current
        try:
            data = await request.json()
        except ValueError:
            raise HTTPException(400, "body must be JSON")
        value = data.get("enabled", data.get("tracking", data.get("value"))) if isinstance(data, dict) else data
        if not isinstance(value, bool):
            raise HTTPException(400, 'expected {"enabled": true|false}')
        return value

    # --- pages -------------------------------------------------------------------------------------
    @api.get("/healthz", include_in_schema=False)
    def healthz():
        return {"ok": True}

    @api.get("/", response_class=HTMLResponse)
    def index(request: Request):
        q = request.query_params.get("token")
        if q is not None:
            if not valid(q):
                return templates.TemplateResponse(request, "login.html", {"error": True}, status_code=401)
            resp = RedirectResponse("/", status_code=303)
            resp.set_cookie(COOKIE, q, max_age=365 * 86400, httponly=True, samesite="strict")
            return resp
        if not token:
            return HTMLResponse("Set DASHBOARD_TOKEN in .env and restart.", status_code=503)
        if not valid(presented(request)):
            return templates.TemplateResponse(request, "login.html", {"error": False}, status_code=401)
        return templates.TemplateResponse(request, "index.html", {"dog_name": cfg.dog_name})

    @api.post("/logout", dependencies=auth)
    def logout():
        resp = RedirectResponse("/", status_code=303)
        resp.delete_cookie(COOKIE)
        return resp

    # --- live view ---------------------------------------------------------------------------------
    def current_jpeg() -> bytes:
        if not core.enabled:
            return off_jpg
        return core.annotated_jpeg() or wait_jpg

    @api.get("/snapshot.jpg", dependencies=auth)
    def snapshot():
        return Response(current_jpeg(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @api.get("/stream.mjpg", dependencies=auth)
    async def stream(request: Request):
        async def frames():
            boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
            while not await request.is_disconnected():
                yield boundary + current_jpeg() + b"\r\n"
                await asyncio.sleep(1 / MJPEG_FPS)

        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame",
                                 headers={"Cache-Control": "no-store"})

    # --- API ---------------------------------------------------------------------------------------
    @api.get("/api/state", dependencies=auth)
    def get_state():
        return core.status()

    @api.post("/api/enabled", dependencies=auth)
    async def post_enabled(request: Request):
        value = await body_flag(request, core.enabled)
        source = request.headers.get("x-source", "dashboard")[:20]
        await asyncio.to_thread(core.set_enabled, value, source)
        return core.status()

    @api.post("/api/tracking", dependencies=auth)
    async def post_tracking(request: Request):
        value = await body_flag(request, core.tracking)
        core.set_tracking(value, request.headers.get("x-source", "dashboard")[:20])
        return core.status()

    @api.get("/api/timeline", dependencies=auth)
    def timeline(date: str | None = None):
        day = parse_day(date)
        stats = summary.day_stats(core.store, day, core.tz)
        start, end = summary.day_bounds(day, core.tz)
        stats.update(day_start=start, day_end=end)
        return stats

    @api.get("/api/barks", dependencies=auth)
    def barks(date: str | None = None):
        start, end = summary.day_bounds(parse_day(date), core.tz)
        events = core.store.events(start, end, "bark")
        for e in events:
            e["clip_url"] = f"/clips/{e['clip']}" if e["clip"] else None
        return JSONResponse(events)

    @api.get("/clips/{name}", dependencies=auth)
    def clip(name: str):
        if not CLIP_RE.match(name):
            raise HTTPException(404)
        path = (cfg.clips_dir / name).resolve()
        if path.parent != cfg.clips_dir.resolve() or not path.is_file():
            raise HTTPException(404)
        return FileResponse(path, media_type="video/mp4")  # supports Range requests (needed by iOS Safari)

    return api
