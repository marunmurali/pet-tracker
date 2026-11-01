"""Minimal Telegram Bot API client: long-polling commands + sending messages, photos and videos.

Only messages from TELEGRAM_CHAT_ID are obeyed; everything else is ignored.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Callable

import requests

log = logging.getLogger(__name__)

HELP = (
    "/on – start watching\n"
    "/off – stop watching (camera stream is closed)\n"
    "/status – current state\n"
    "/snap – photo now\n"
    "/clip – record a short clip now\n"
    "/summary – today so far\n"
    "/help – this list"
)

# Called with the command name (without '/') and the argument text; returns a reply text or None.
CommandHandler = Callable[[str, str], "str | None"]


class TelegramBot:
    def __init__(self, token: str, chat_id: str, session: requests.Session | None = None):
        self.token = token
        self.chat_id = str(chat_id)
        self.http = session or requests.Session()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._offset = 0
        self.handler: CommandHandler | None = None

    @property
    def configured(self) -> bool:
        return bool(self.token and self.chat_id)

    def _url(self, method: str) -> str:
        return f"https://api.telegram.org/bot{self.token}/{method}"

    def _call(self, method: str, timeout: float = 30, files=None, **params):
        try:
            r = self.http.post(self._url(method), data=params, files=files, timeout=timeout)
            body = r.json()
        except (requests.RequestException, ValueError) as e:
            # never log the exception text: requests includes the URL, which contains the bot token
            log.warning("telegram %s failed: %s", method, type(e).__name__)
            return None
        if not body.get("ok"):
            log.warning("telegram %s error: %s", method, body.get("description"))
            return None
        return body["result"]

    # --- sending -----------------------------------------------------------------------------------
    def send_message(self, text: str) -> None:
        if self.configured:
            self._call("sendMessage", chat_id=self.chat_id, text=text)

    def send_photo(self, jpeg: bytes, caption: str = "") -> None:
        if self.configured:
            self._call("sendPhoto", timeout=60, files={"photo": ("snap.jpg", jpeg, "image/jpeg")},
                       chat_id=self.chat_id, caption=caption)

    def send_video(self, path: Path, caption: str = "") -> None:
        if not self.configured:
            return
        with open(path, "rb") as f:
            self._call("sendVideo", timeout=120, files={"video": (path.name, f, "video/mp4")},
                       chat_id=self.chat_id, caption=caption, supports_streaming="true")

    def send_photos(self, paths: list[Path], caption: str = "") -> None:
        """Up to 10 photos as one album; the caption goes on the first."""
        if not self.configured or not paths:
            return
        paths = paths[:10]
        if len(paths) == 1:
            self.send_photo(paths[0].read_bytes(), caption)
            return
        media = [{"type": "photo", "media": f"attach://p{i}", **({"caption": caption} if i == 0 else {})}
                 for i in range(len(paths))]
        files = {f"p{i}": (p.name, p.read_bytes(), "image/jpeg") for i, p in enumerate(paths)}
        self._call("sendMediaGroup", timeout=120, files=files, chat_id=self.chat_id, media=json.dumps(media))

    # --- receiving ---------------------------------------------------------------------------------
    def start(self, handler: CommandHandler) -> None:
        self.handler = handler
        if not self.configured:
            log.info("telegram not configured; bot disabled")
            return
        self._call("setMyCommands", commands=json.dumps([
            {"command": line.split(" ")[0][1:], "description": line.split("– ", 1)[1]}
            for line in HELP.splitlines()]))
        self._thread = threading.Thread(target=self._poll, name="telegram", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _poll(self) -> None:
        while not self._stop.is_set():
            updates = self._get_updates()
            if updates is None:
                self._stop.wait(5)
                continue
            for u in updates:
                self._offset = max(self._offset, u["update_id"] + 1)
                self.handle_update(u)

    def _get_updates(self):
        try:
            r = self.http.post(self._url("getUpdates"), timeout=40,
                               data={"offset": self._offset, "timeout": 30, "allowed_updates": '["message"]'})
            body = r.json()
        except (requests.RequestException, ValueError) as e:
            log.warning("telegram getUpdates failed: %s", type(e).__name__)
            return None
        return body.get("result", []) if body.get("ok") else None

    def handle_update(self, update: dict) -> None:
        msg = update.get("message") or {}
        if str(msg.get("chat", {}).get("id")) != self.chat_id:
            return  # not our chat: ignore silently
        text = (msg.get("text") or "").strip()
        if not text.startswith("/"):
            return
        cmd, _, arg = text[1:].partition(" ")
        cmd = cmd.split("@")[0].lower()  # "/status@MyBot" in groups
        if cmd in ("help", "start"):
            self.send_message(HELP)
            return
        if not self.handler:
            return
        try:
            reply = self.handler(cmd, arg.strip())
        except Exception:
            log.exception("telegram command /%s failed", cmd)
            reply = f"/{cmd} failed, see logs"
        if reply:
            self.send_message(reply)
