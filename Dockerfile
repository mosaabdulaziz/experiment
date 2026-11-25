# ── Stage 1: build the React frontend ──
FROM node:22-slim AS frontend-builder

WORKDIR /frontend
COPY frontend/package*.json ./
# npm workspaces (frontend/package.json "workspaces"): npm ci installs a
# member's dependencies only if that member's package.json is present, so each
# one is copied before it. Copying only the top-level file (as before
# 2026-09-24) left the products' dependencies out, and the iris product failed
# to type-check. A new product under src/products/ needs a line here.
COPY frontend/src/shared/package.json ./src/shared/
COPY frontend/src/products/attendance/package.json ./src/products/attendance/
COPY frontend/src/products/barber/package.json ./src/products/barber/
COPY frontend/src/products/iris/package.json ./src/products/iris/
COPY frontend/src/products/fall_detection/package.json ./src/products/fall_detection/
COPY frontend/src/products/safety/package.json ./src/products/safety/
COPY frontend/src/products/concealment_detection/package.json ./src/products/concealment_detection/
RUN npm ci

COPY frontend/ ./
RUN npm run build

# ── Stage 2: Python app ──
FROM python:3.11-slim@sha256:90744cff8f32887f075c47d747a173ff333e9e98801667af93c357fa9f5e28ff

# System dependencies for OpenCV headless + PyTorch CPU (ultralytics);
# ffmpeg is required by app/clip_builder.py (concat+trim of recorded segments
# into FOUND preview clips) — it shells out to the ffmpeg/ffprobe binaries directly.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv, pinned by version and digest (DEPENDENCIES.md, "uv")
COPY --from=ghcr.io/astral-sh/uv:0.13.6@sha256:af22a52f5eafef3d78e551599d405c02df61919d18028f0cea8eda93bc9986d0 /uv /uvx /usr/local/bin/

# Install Python deps — swap opencv-python for headless (no display needed).
# torch/torchvision are pre-installed from the CPU-only index BEFORE the main
# install: ultralytics' bare `torch>=1.8.0` requirement would otherwise resolve
# to the CUDA build on Linux (~2.5 GB of nvidia-* wheels) even though this image
# only ever runs CPU inference. The CPU wheels are ~200 MB and satisfy the
# constraint, so the uv install below skips torch entirely. Pinned to the root
# lock's versions (uv.lock, group app-image; tests/test_locks.py checks they agree).
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu
# uv's wheel cache is persisted across builds via a BuildKit cache mount, so a
# rebuild (requirements.txt change, base-image digest drift, etc.) reuses the
# already-downloaded wheels instead of re-fetching every package.
RUN --mount=type=cache,target=/root/.cache/uv \
    sed 's/^opencv-python/opencv-python-headless/' requirements.txt \
    | uv pip install --system -r /dev/stdin

# Pre-download YOLO weights so the image is self-contained and doesn't
# need internet on first request. Lands in /app/yolo11n.pt via WORKDIR.
RUN python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"

# InsightFace runtime deps (CPU mode, used by /api/faces/register). Pinned to the root lock's versions (uv.lock, group
# app-image; tests/test_locks.py checks they agree).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system insightface==2.0 onnxruntime==1.30.0

# The InsightFace buffalo_l pack is NOT in the image: /api/faces/register loads the repo's DVC copy,
# perception/models/insightface/models/buffalo_l (docker-compose mounts ./perception at /app/perception;
# perception/insightface_pack.py). A missing pack answers 503 with the `dvc pull` to run; nothing is downloaded.

# Copy backend source
COPY app/ ./app/
COPY main.py config.yaml ./
# Only the yaml config and the GPU-free sources.py/recorder_kind.py helpers, not the rest of
# perception/pipeline/ (GPU/DeepStream-only code that doesn't belong in this image). sidecar.yaml is read
# by app/server.py to convert zone coordinates using the sidecar's mux_width/mux_height; sources.py's
# resolve_sources() is imported by app/server.py to map stream ids to the sidecar's numeric pipeline
# indices when generating nvdsanalytics_config.txt.
# recorder_kind.py is imported by app/server.py (mixed-fleet startup warnings) and
# app/stream_manager.py (which streams resolve alert clips via smart-record), so it
# must ship here too — omitting it crash-loops this container on startup.
COPY perception/__init__.py ./perception/
COPY perception/pipeline/sidecar.yaml ./perception/pipeline/sidecar.yaml
COPY perception/engine/__init__.py perception/engine/sources.py perception/engine/recorder_kind.py ./perception/engine/
# app/zones.py reads the proximity module's per-zone defaults and checks by path (the zone editor).
COPY perception/modules/proximity/zone_defaults.json ./perception/modules/proximity/zone_defaults.json
# The safety product's app-side code (products/safety: its API, settings, Telegram, camera health, and the config it
# shares with the host controller), imported by app/server.py only when IRIS_PRODUCT names safety.
COPY products/safety/config.py products/safety/service.py products/safety/api.py products/safety/notifier.py products/safety/camera_health.py ./products/safety/

# Copy built frontend from stage 1 — submodule's vite.config.ts outputs to dist/
COPY --from=frontend-builder /frontend/dist ./static

RUN mkdir -p data/uploads

EXPOSE 8000

CMD ["python", "main.py"]
