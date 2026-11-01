from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

from app.config import Config
from app.store import Store
from app.telegram_bot import TelegramBot
from app.web import create_app

TOKEN = "s3cret-token"


class FakeApp:
    """The slice of app.main.App that the web layer uses."""

    def __init__(self, tmp: Path, token=TOKEN):
        self.cfg = Config(dashboard_token=token, data_dir=tmp, dog_name="Rex")
        self.cfg.clips_dir.mkdir(parents=True)
        self.store = Store(self.cfg.db_path)
        self.tz = ZoneInfo("UTC")
        self.enabled, self.tracking = True, True
        self.calls = []

    def today(self):
        import datetime as dt
        return dt.datetime.now(self.tz).date()

    def status(self):
        return {"enabled": self.enabled, "tracking": self.tracking}

    def set_enabled(self, on, source):
        self.calls.append(("enabled", on, source))
        self.enabled = on

    def set_tracking(self, on, source):
        self.tracking = on

    def annotated_jpeg(self):
        return b"\xff\xd8fake\xff\xd9"


@pytest.fixture
def fake(tmp_path):
    return FakeApp(tmp_path)


@pytest.fixture
def client(fake):
    return TestClient(create_app(fake))


def test_api_requires_token(client):
    assert client.get("/api/state").status_code == 401
    assert client.get("/api/state", headers={"X-Token": "wrong"}).status_code == 401
    assert client.get("/api/state", headers={"X-Token": TOKEN}).json()["enabled"] is True
    assert client.get("/api/state", headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 200
    assert client.get("/snapshot.jpg").status_code == 401
    assert client.get("/healthz").status_code == 200


def test_no_token_configured_means_locked(tmp_path):
    c = TestClient(create_app(FakeApp(tmp_path, token="")))
    assert c.get("/api/state", headers={"X-Token": ""}).status_code == 503
    assert c.get("/").status_code == 503


def test_browser_login_sets_cookie_and_strips_token(client):
    assert client.get("/").status_code == 401  # login page
    r = client.get(f"/?token={TOKEN}", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/"
    assert "httponly" in r.headers["set-cookie"].lower()
    assert client.get("/").status_code == 200
    assert client.get("/api/state").status_code == 200


def test_api_toggle_and_explicit_set(client, fake):
    h = {"X-Token": TOKEN, "X-Source": "script"}
    assert client.post("/api/enabled", headers=h).json()["enabled"] is False  # empty body toggles
    assert client.post("/api/enabled", headers=h, json={"enabled": True}).json()["enabled"] is True
    assert client.post("/api/enabled", headers=h, json={"enabled": "yes"}).status_code == 400
    assert fake.calls == [("enabled", False, "script"), ("enabled", True, "script")]


def test_clip_path_traversal_blocked(client, fake):
    (fake.cfg.clips_dir / "a.mp4").write_bytes(b"x" * 100)
    h = {"X-Token": TOKEN}
    assert client.get("/clips/a.mp4", headers=h).status_code == 200
    assert client.get("/clips/a.mp4", headers={**h, "Range": "bytes=0-9"}).status_code == 206
    assert client.get("/clips/..%2Fpet-tracker.db", headers=h).status_code == 404
    assert client.get("/clips/missing.mp4", headers=h).status_code == 404


def test_timeline_and_barks(client, fake):
    import time
    fake.store.add_event("bark", time.time(), {"score": 0.8}, clip="x.mp4")
    h = {"X-Token": TOKEN}
    barks = client.get("/api/barks", headers=h).json()
    assert barks[0]["clip_url"] == "/clips/x.mp4"
    assert client.get("/api/timeline", headers=h).json()["barks"] == 1
    assert client.get("/api/timeline?date=nope", headers=h).status_code == 400


# --- telegram --------------------------------------------------------------------------------------

class FakeSession:
    def __init__(self):
        self.posts = []

    def post(self, url, data=None, files=None, timeout=None):
        self.posts.append((url.rsplit("/", 1)[1], data))

        class R:
            def json(self):
                return {"ok": True, "result": {}}
        return R()


def msg(chat_id, text):
    return {"update_id": 1, "message": {"chat": {"id": chat_id}, "text": text}}


def test_bot_only_obeys_configured_chat():
    sess = FakeSession()
    bot = TelegramBot("tok", "42", session=sess)
    got = []
    bot.handler = lambda cmd, arg: got.append(cmd) or "ok"
    bot.handle_update(msg(999, "/off"))
    assert got == [] and sess.posts == []
    bot.handle_update(msg(42, "/off"))
    bot.handle_update(msg(42, "/status@DogBot"))
    assert got == ["off", "status"]
    assert [p[0] for p in sess.posts] == ["sendMessage", "sendMessage"]


def test_bot_help():
    sess = FakeSession()
    bot = TelegramBot("tok", "42", session=sess)
    bot.handle_update(msg(42, "/help"))
    assert "/clip" in sess.posts[0][1]["text"]
