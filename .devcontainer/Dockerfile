# Multi-arch base (amd64 + arm64), so the same container runs on a PC and a Raspberry Pi 5.
FROM mcr.microsoft.com/devcontainers/python:1-3.11-bookworm

# ffmpeg: pulls the Tapo RTSP stream (video + audio) and writes clips.
# sqlite3: handy for inspecting the event database by hand.
# The base image may include Yarn's APT source, whose signing key is stale.
# Node is installed by the devcontainer feature, so Yarn's APT repo is not needed.
RUN rm -f /etc/apt/sources.list.d/yarn.list /etc/apt/sources.list.d/yarn.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg sqlite3 tzdata curl \
    && rm -rf /var/lib/apt/lists/*

# Runtime Python deps are baked into the image so rebuilds are fast and
# the always-on service can reuse this exact image later.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt
