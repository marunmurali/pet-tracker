"""Settings loaded from .env (and the process environment) into one frozen dataclass."""
from __future__ import annotations

import os
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


def _str(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _int(name: str, default: int) -> int:
    v = _str(name)
    return int(v) if v else default


def _float(name: str, default: float) -> float:
    v = _str(name)
    return float(v) if v else default


def _bool(name: str, default: bool) -> bool:
    v = _str(name).lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


def parse_zones(spec: str) -> dict[str, tuple[float, float, float, float]]:
    """'bed:0.1,0.5,0.4,0.9; bowl:0.7,0.7,0.9,0.95' -> {name: (x1, y1, x2, y2)} in 0..1 frame coords."""
    zones: dict[str, tuple[float, float, float, float]] = {}
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        name, _, coords = part.partition(":")
        x1, y1, x2, y2 = (float(c) for c in coords.split(","))
        zones[name.strip()] = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
    return zones


@dataclass(frozen=True)
class Config:
    # Camera
    cam_host: str = ""
    cam_user: str = ""
    cam_pass: str = field(default="", repr=False)
    cam_stream: str = "stream2"
    cam_source: str = ""  # optional file/URL that replaces the RTSP stream (offline testing)
    onvif_port: int = 2020
    tapo_cloud_password: str = field(default="", repr=False)
    ptz_backend: str = "onvif"  # onvif | pytapo | none
    ptz_home_preset: str = ""
    ptz_invert_pan: bool = False
    ptz_invert_tilt: bool = False

    # Stream / processing
    frame_width: int = 640
    frame_height: int = 360
    stream_fps: int = 10
    detect_fps: float = 6.0

    # Telegram
    telegram_bot_token: str = field(default="", repr=False)
    telegram_chat_id: str = ""

    # Behaviour
    dog_name: str = "Buddy"
    tz: str = "UTC"
    yolo_imgsz: int = 320
    yolo_model: str = "yolo11n.onnx"
    dog_conf_threshold: float = 0.45
    bark_threshold: float = 0.5
    bark_alert_cooldown_s: float = 120
    clip_pre_s: float = 5
    clip_post_s: float = 10
    daily_summary_time: str = "21:00"
    tracking_enabled: bool = True
    lost_timeout_s: float = 8
    sweep_on_lost: bool = True
    sweep_max_s: float = 120
    sweep_min_gap_s: float = 300
    zones: dict = field(default_factory=dict)
    snapshot_interval_min: float = 30
    training_capture: bool = True
    training_interval_s: float = 10

    # Dashboard
    dashboard_port: int = 8000
    dashboard_token: str = field(default="", repr=False)

    # Paths
    data_dir: Path = ROOT / "data"
    models_dir: Path = ROOT / "models"

    @property
    def rtsp_url(self) -> str:
        q = lambda s: urllib.parse.quote(s, safe="")  # noqa: E731
        return f"rtsp://{q(self.cam_user)}:{q(self.cam_pass)}@{self.cam_host}:554/{self.cam_stream}"

    @property
    def source(self) -> str:
        return self.cam_source or self.rtsp_url

    @property
    def dataset_dir(self) -> Path:
        return self.data_dir / "dataset"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "pet-tracker.db"

    @property
    def clips_dir(self) -> Path:
        return self.data_dir / "clips"

    @property
    def snapshots_dir(self) -> Path:
        return self.data_dir / "snapshots"


def load_config(env_file: Path | None = ROOT / ".env") -> Config:
    if env_file is not None and env_file.exists():
        load_dotenv(env_file, override=False)
    cam_source = _str("CAM_SOURCE")
    if cam_source and "://" not in cam_source and not Path(cam_source).is_absolute():
        cam_source = str(ROOT / cam_source)
    return Config(
        cam_host=_str("CAM_HOST"),
        cam_user=_str("CAM_USER"),
        cam_pass=_str("CAM_PASS"),
        cam_stream=_str("CAM_STREAM", "stream2"),
        cam_source=cam_source,
        onvif_port=_int("ONVIF_PORT", 2020),
        tapo_cloud_password=_str("TAPO_CLOUD_PASSWORD"),
        ptz_backend=_str("PTZ_BACKEND", "onvif").lower(),
        ptz_home_preset=_str("PTZ_HOME_PRESET"),
        ptz_invert_pan=_bool("PTZ_INVERT_PAN", False),
        ptz_invert_tilt=_bool("PTZ_INVERT_TILT", False),
        frame_width=_int("FRAME_WIDTH", 640),
        frame_height=_int("FRAME_HEIGHT", 360),
        stream_fps=_int("STREAM_FPS", 10),
        detect_fps=_float("DETECT_FPS", 6.0),
        telegram_bot_token=_str("TELEGRAM_BOT_TOKEN"),
        telegram_chat_id=_str("TELEGRAM_CHAT_ID"),
        dog_name=_str("DOG_NAME", "Buddy"),
        tz=_str("TZ", "UTC"),
        yolo_imgsz=_int("YOLO_IMGSZ", 320),
        yolo_model=_str("YOLO_MODEL", "yolo11n.onnx"),
        dog_conf_threshold=_float("DOG_CONF_THRESHOLD", 0.45),
        bark_threshold=_float("BARK_THRESHOLD", 0.5),
        bark_alert_cooldown_s=_float("BARK_ALERT_COOLDOWN_S", 120),
        clip_pre_s=_float("CLIP_PRE_S", 5),
        clip_post_s=_float("CLIP_POST_S", 10),
        daily_summary_time=_str("DAILY_SUMMARY_TIME", "21:00"),
        tracking_enabled=_bool("TRACKING_ENABLED", True),
        lost_timeout_s=_float("LOST_TIMEOUT_S", 8),
        sweep_on_lost=_bool("SWEEP_ON_LOST", True),
        sweep_max_s=_float("SWEEP_MAX_S", 120),
        sweep_min_gap_s=_float("SWEEP_MIN_GAP_S", 300),
        zones=parse_zones(_str("ZONES")),
        snapshot_interval_min=_float("SNAPSHOT_INTERVAL_MIN", 30),
        training_capture=_bool("TRAINING_CAPTURE", True),
        training_interval_s=_float("TRAINING_INTERVAL_S", 10),
        dashboard_port=_int("DASHBOARD_PORT", 8000),
        dashboard_token=_str("DASHBOARD_TOKEN"),
    )
