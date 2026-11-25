# Concealment Detection — Deployment Guide

Concise guide to deploying **Concealment Review** (`products/concealment_detection`) alongside the IRIS platform (`go2rtc`, `redis`, `minio`, `iris`, and `deepstream-sidecar-concealment`).

---

## 1. Architecture & Components

```text
RTSP Camera ─▶ go2rtc ─▶ deepstream-sidecar-concealment (GPU) ─▶ Redis
                               │  pose (25 Hz) + dets (person/bag)      │
                               ├─▶ iris:meta:<cam> ──────────────────────▶ concealment service (:8005)
                               └─▶ iris:jpeg:<cam> ─▶ iris (:18000) ─────▶ Frontend UI (:5180 / static)
```

| Component | Where it runs | Default Port / Endpoint |
|---|---|---|
| Shared infra (`redis`, `minio`, `go2rtc`, `iris`) | Docker Compose | `:6379`, `:19000`, `:8554` / `:1984`, `:18000` |
| DeepStream sidecar (`deepstream-sidecar-concealment`) | Docker Compose profile `concealment` (GPU) | Telemetry: `:9107/metrics` |
| Review & detection service (`products/concealment_detection`) | Host Python (`uv` venv) | REST API: `:8005` (`/status`, `/docs`) |
| Dashboard (`Concealment Review` tab) | Vite dev (`:5180`) or built into `iris` (`:18000`) | `http://localhost:5180/?product=concealment_detection#/` |

---

## 2. Prerequisites

On the GPU deployment host (Linux with Docker + NVIDIA Container Toolkit):

1. Required tools: `docker` (with Compose v2 plugin), `just`, `uv`, `python3`, `node`, `ffmpeg`, `nvidia-smi`.
2. Environment file:
   ```bash
   just bootstrap   # creates .env from .env.example, installs frontend node_modules, checks tools/data
   ```
   Set `RTSP_USER` / `RTSP_PASS` (for real RTSP cameras) and adjust port overrides in `.env` if running multiple checkouts on the same host (see `.env.example`).
3. DVC model weights required by `products/concealment_detection/sidecar.yaml`:
   - `yolo11x.onnx` (PGIE — person & bags: `person`, `handbag`, `backpack`, `suitcase`)
   - `yolo26x-pose.onnx` (SGIE — 17-keypoint pose at `interval=1` / 25 Hz)
   ```bash
   just data-pull models-fall
   ```
   *(First container startup compiles TensorRT `.engine` files next to the ONNX checkpoints; allow a few minutes once.)*

---

## 3. Quick Start (Recommended)

Run everything from the repository root with `just`:

```bash
# 1. Verify tools, Python venv, config/zones, DVC models, and camera exclusivity
just concealment-check

# 2. Start shared infra (if not up), the :8005 review service, GPU sidecar, and UI
just concealment-up
```

`just concealment-up` starts services in safe order:
1. Syncs `products/concealment_detection/.venv` (`uv sync`).
2. Validates `concealment_detection.yaml`, `zones.yaml`, camera matching, DVC models, and refuses if another live `deepstream-sidecar-*` container already holds the same camera(s).
3. Brings up `redis`, `minio`, `go2rtc`, and `iris` (for `/stream/<camera>/frame` MJPEG frames).
4. Starts `products/concealment_detection/main.py` on port `8005` and verifies `/status` before touching the GPU sidecar.
5. Starts `deepstream-sidecar-concealment` (`--profile concealment`) and waits for `pipeline PLAYING`.
6. Starts the Vite frontend (`:5180`) if not already running and prints the dashboard URL.

### Useful Operational Commands

| Action | Command |
|---|---|
| Run preflight checks only | `just concealment-check` |
| Start concealment stack | `just concealment-up` |
| Stop concealment service + sidecar (keeps shared infra up) | `just concealment-down` |
| Stop entire stack (all containers + host processes) | `just down` |
| Tail Python review service logs | `just concealment-log` |
| Tail DeepStream sidecar container logs | `just concealment-sidecar-log` |
| Run unit & API tests (no GPU or Redis needed) | `just concealment-test` |

---

## 4. Manual Docker Compose Commands

If running without `just`:

```bash
# 1. Build images (first time or after Dockerfile/frontend changes)
docker compose build iris deepstream-sidecar-concealment

# 2. Start shared services + GPU sidecar (name services explicitly!)
docker compose --profile concealment up -d redis minio iris go2rtc deepstream-sidecar-concealment

# 3. Start the review backend on the host
cd products/concealment_detection
uv sync
REDIS_URL=redis://localhost:6379/0 uv run python main.py
```

> **Note:** Always pass explicit service names (`redis minio iris go2rtc deepstream-sidecar-concealment`) when using `--profile concealment` with `docker compose up`, or Compose will also start the default profile-less `deepstream-sidecar` and fail on duplicate camera ownership.

---

## 5. Configuration & Zones

- **Product config**: [`concealment_detection.yaml`](../concealment_detection.yaml) (override path via `CONCEALMENT_CONFIG`)
- **Shelf zones**: [`zones.yaml`](../zones.yaml) (override path via `CONCEALMENT_ZONES_PATH`; every declared camera needs either normalized `zones` or explicit `zone_free: true`)
- **Sidecar pipeline overlay**: [`sidecar.yaml`](../sidecar.yaml) (override path via `CONCEALMENT_SIDECAR_CONFIG`)
- **Environment overrides**:
  - `CONCEALMENT_BACKEND_PORT` (default `8005`)
  - `SIDECAR_CONCEALMENT_TELEMETRY_PORT` (default `9107`)
  - `REDIS_URL` (default `redis://localhost:6379/0`)
  - `CONCEALMENT_DATA_DIR` (SQLite `concealment_detection.db` directory, default `products/concealment_detection/data`)

---

## 6. Post-Deploy Verification

1. **Service health**: `curl -sf http://localhost:8005/status` -> `"running": true` and all configured cameras `"camera_ok": true`.
2. **Sidecar metrics**: `curl -sf http://localhost:9107/metrics | head`.
3. **Live frame preview**: `curl -I http://localhost:18000/stream/cam-1/frame` returns `200 OK`.
4. **UI check**: Open `http://localhost:5180/?product=concealment_detection#/` (or add `IRIS_PRODUCT=concealment_detection` in `.env` to scope the switcher menu). High-tier alarms render in red (`VERIFY · HIGH`), medium-tier alarms in amber (`VERIFY`), and all verdicts remain non-accusatory (`acknowledged`, `concern_noted`, `dismissed`).
