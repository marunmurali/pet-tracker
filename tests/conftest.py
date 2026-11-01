import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODELS = ROOT / "models"
SAMPLES = ROOT / "data" / "samples"


def have_models(*names: str) -> bool:
    return all((MODELS / n).exists() for n in names)


@pytest.fixture(scope="session")
def synthetic_video(tmp_path_factory) -> Path:
    """4 s test pattern at 15 fps with a 440 Hz tone, encoded like the camera (H.264 + G.711)."""
    path = tmp_path_factory.mktemp("media") / "synthetic.mkv"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y",
         "-f", "lavfi", "-i", "testsrc=size=1280x720:rate=15:duration=4",
         "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=8000:duration=4",
         "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "pcm_alaw", str(path)],
        check=True)
    return path
