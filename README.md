# IRIS
AI-powered surveillance camera system that combines local YOLO object detection with a Vision Language Model to monitor a live camera feed or video file, detect safety-relevant events, and stream observations to a browser in real time.

Backed by a **FastAPI** server and a **React 19 + Vite + TypeScript** frontend with a glass/spatial dark UI.

---

## Documentation map

| If you want to... | read |
|---|---|
| understand the GPU sidecar (engine + modules), with diagrams | [`perception/README.md`](perception/README.md), then [`perception/ARCHITECTURE.md`](perception/ARCHITECTURE.md) |
| add a capability or a model | [`perception/modules/README.md`](perception/modules/README.md) |
| prove a change is safe, or bring a product up to check it | [`perception/TESTING.md`](perception/TESTING.md) |
| plan fall detection, PPE or proximity | [`perception/NEXT_MODULES.md`](perception/NEXT_MODULES.md) |
| run the stack | [`RUNNING.md`](RUNNING.md), [`JUST_GUIDE.md`](JUST_GUIDE.md) |
| find a file | [`FILES.md`](FILES.md) |
| version data and model weights | [`DATA.md`](DATA.md) |
| train, evaluate, register or roll back a model; read a startup manifest error | [`MLFLOW.md`](MLFLOW.md) |
| work on a product | [`products/attendance/RUNBOOK.md`](products/attendance/RUNBOOK.md), [`products/barber/README.md`](products/barber/README.md), [`products/multicam/README.md`](products/multicam/README.md) |
| send empty-shelf alerts to iris-platform | [Shelf -> iris-platform](#shelf---iris-platform) below, [`perception/modules/shelf_void/README.md`](perception/modules/shelf_void/README.md) |

---

## Features

### Monitoring
- **Two perception backends** — `local` (in-process OpenCV capture + YOLO, single camera) or `deepstream` (a separate GPU sidecar that batches N streams, runs YOLO + tracking, and publishes detections + annotated JPEGs to Redis for a multi-stream video wall). The DeepStream preview has two modes: `pipeline_preview: true` (default — boxes + JPEGs baked in the sidecar) or `false` (inference-only — the website plays each source's **raw stream** over WebRTC via go2rtc and overlays detections from `iris:meta`, freeing the GPU of all preview encoding)
- **Proactive / Chat mode toggle** — switch between proactive analysis and chat from the header; Chat mode pauses the proactive VLM cycles (the camera feed and chat buffers stay warm) so the model is dedicated to the conversation. A chat turn also auto-pauses analysis for its duration
- **Multi-frame batch analysis** — sends short clips (configurable frame count + overlap) to the VLM for temporal reasoning across motion events
- **Local YOLO enrichment** — YOLO runs on every frame to detect and annotate vehicles, pedestrians, and traffic signals before the VLM sees them
- **Annotated live stream** — bounding boxes overlaid on the MJPEG feed in the browser, driven by a dedicated annotation loop
- **Multi-event output** — a single clip can produce multiple independent events (e.g. an alert and a comment simultaneously)
- **Embedding-based deduplication** — cosine similarity on DashScope or local embeddings suppresses repetitive observations without blocking genuinely new ones
- **Personas** — switchable AI roles (`traffic`, `indoor_security`) with independently editable instructions
- **Conversational chat** — ask questions across every live stream with the cross-camera event log and analytics as context; answers stream token-by-token over SSE, optional event-DB tool calls, and a backward history sweep when an object isn't in the current frames. Sessions persist across turns
- **Video file support** — upload and stream `.mp4 / .avi / .mkv / .mov / .webm`; pause, resume, and loop
- **Multi-language output** — responses in English or Arabic

### UI (React frontend)
- **Landing page** (`/#/`) — branded hero with animated mesh-gradient background, glass feature cards, live backend-health badge
- **Dashboard** (`/#/dashboard`) — live MJPEG feed, real-time event log over WebSocket, conversational chat panel, collapsible monitoring-instructions editor, and a header mode toggle (Proactive / Chat) with an "analysis paused" indicator
- **Multi-stream video wall** (deepstream backend) — adaptive mosaic of every discovered stream with per-camera event filtering and an expandable tile view
- **Analytics** (`/#/analytics`) — stats cards plus Recharts panels for events-per-hour, alert ratio, top tags, and an activity timeline
- **Event filtering & search** — type chips (all / comment / alert), free-text search across messages and tags, time-range filter
- **Glass / spatial design** — frosted panels with backdrop-blur, soft elevation, animated accent glow, fully responsive

### Engineering
- **Type-safe API + WebSocket client** — all endpoint shapes and WS messages are typed; discriminated-union event handler
- **Zustand stores** — camera, event, chat, and config state slices with stable references
- **Single-port deploy** — `npm run build` writes to `static/`; FastAPI serves the SPA + APIs on `:8000`
- **Multi-stage Dockerfile** — Node build stage compiles the frontend, Python stage runs the server; host doesn't need Node
- **Debug outputs** — annotated frame snapshots, per-cycle YOLO logs, full prompt log, and analyzer response log

---

## Requirements

- Python 3.11+
- Node 22+ (only to build the frontend; not needed at runtime once `static/` is built — Docker handles this for you)
- A [DashScope](https://dashscope-intl.aliyuncs.com) API key (`QWEN_API_KEY`)
- A webcam **or** a video file (RTSP/HTTP URLs also work)
- `ultralytics` for YOLO enrichment (installed via `requirements.txt`)
- **For the `deepstream` multi-stream backend only:** Redis and an NVIDIA GPU + the DeepStream 9.0 container (run via Docker Compose). The `local` single-camera backend needs none of these.

---

## Setup

```bash
# 1. Clone the repo (with submodule)
git clone --recurse-submodules <repo-url>
cd surveillance-camera-sys
# If you already cloned without --recurse-submodules:
# git submodule update --init

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux / macOS

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Configure API key
cp .env.example .env
# Edit .env and set QWEN_API_KEY=<your-key>

# 5. Build the frontend (requires Node 22+)
cd frontend
npm install
npm run build       # outputs to ../static/
cd ..

# 6. Run
python main.py
```

Open **http://localhost:8000** in your browser.

### Frontend development

The frontend lives in `frontend/` — a standalone repository (submodule) with React 19 + Vite + TypeScript. While iterating on the UI, run the Vite dev server with HMR alongside the Python backend:

```bash
# Terminal 1
python main.py             # FastAPI on :8000

# Terminal 2
cd frontend
npm run dev                # Vite on :5173 with HMR; proxies /api, /ws, /stream to :8000
```

Open **http://localhost:5173** during development. For production / Docker, `npm run build` writes the compiled assets into `dist/`, which the multi-stage Dockerfile copies to `static/` for FastAPI to serve.

The frontend repo is at `ssh://git@gitlab.urca.ai:2222/pyai/iris/iris-frontend.git` — you can develop and version it independently.

---

## Docker

```bash
# Build and run
docker compose up --build

# Detached
docker compose up -d
```

`docker-compose.yml` defines five core services: **`iris`** (the FastAPI app), **`redis`** (the perception data/control plane), **`minio`** (object store for recorded segments), **`deepstream-sidecar`** (the GPU perception pipeline), and **`go2rtc`** (media server for raw WebRTC previews when `pipeline_preview: false`), plus an optional **monitoring stack** — `prometheus`, `grafana`, `dcgm-exporter`, `node-exporter` (see [`monitoring/README.md`](monitoring/README.md)). For the `local` single-camera backend you only need the `iris` service — `/dev/video0` is passed through for webcam access (remove the `devices` entry when using RTSP or a video file). For the `deepstream` multi-stream backend, bring up the core five (Redis + an NVIDIA GPU required) and set `perception.backend: deepstream` with `cv.enabled: false`.

**Fresh clone — what you need before `docker compose up --build` works end-to-end:**

- The frontend is a **git submodule** — clone with `--recurse-submodules` (or run `git submodule update --init --recursive`) before building, or the Docker build has nothing to compile.
- **`.env` is optional** — compose boots without it using MINIO defaults and no `QWEN_API_KEY` (chat/analysis degrade until you `cp .env.example .env` and fill it in).
- The **first boot builds the TRT engines** (INT8 pgie ≈ 1 min incl. calibration, face SGIE ≈ 20 s) from the ONNX weights; every later boot deserializes from cache. The weights are versioned with DVC, not git: run `just data-mount && just data-pull` (or `just bootstrap`) first -- see [DATA.md](DATA.md).
- The shipped `perception/pipeline/sidecar.yaml` sources sample `file://` test videos that are **not** committed (`data/uploads/` is gitignored; the clips are DVC pointers, `just data-pull samples-attendance`, see [DATA.md](DATA.md)) — point `sources:` at your own files/URLs or upload videos through the UI.
- **RTSP camera credentials are not committed** and the shipped `perception/pipeline/sidecar.yaml` has no RTSP sources (they're unreachable on a fresh network) — it runs the sample file sources only. To add LAN cameras, regenerate the config with `python3 scripts/live_streams.py run --n 6` (RTSP is added by default; the tool writes `${RTSP_USER}`/`${RTSP_PASS}` placeholders that the sidecar expands from its environment). Set `RTSP_USER`/`RTSP_PASS` in `.env` (docker-compose forwards them) — unset values leave the placeholders in place and those sources fail visibly at connect time.
- The sidecar container always sees the repo at `/workspace/iris` regardless of the checkout directory's name (compose mounts `./:/workspace/iris`), so every config path in the repo is stable across machines.
- **`go2rtc` serves raw previews** (`pipeline_preview: false`): `test_streamer/stream_recordings.py` publishes the sample uploads to its RTSP ingest as `raw-<cam-id>`, go2rtc derives `<cam-id>` (original codec + on-demand H.264 transcode for browsers), the sidecar consumes `rtsp://go2rtc:8554/<cam-id>`, and the website plays `/api/webrtc?src=<cam-id>`. The pinned ICE candidates in `go2rtc.yaml` (`127.0.0.1:8555` + your LAN IP) must be reachable from the browser — adjust the LAN entry for your network.
- Host prerequisites: an NVIDIA GPU + the NVIDIA Container Toolkit, and the `video` group (used by the iris container for webcam access).

---

## Shelf -> iris-platform

The shelf preset (`shelf_void`, `stability: tracker`: the accepted shelf-demo's tracker, person pause and one situation
per shelf region) writes `shelf_empty` / `shelf_restocked` alerts to `iris:alert:<camera>` (`iris:alert:<camera>@<role>`
when the sidecar runs with `role_scoped_keys`). The **bridge**
(`products/shelf_iris_bridge`, `python -m shelf_iris_bridge`) sends them to iris-platform as an AI backend
over RabbitMQ (iris-platform `docs/ai-integration.md`):

```
DeepStream sidecar --iris:alert:<key id>--> app (plain XREAD: still gets every alert)
                                       \--> bridge (own consumer group) --RabbitMQ iris.ai--> iris-platform
```

| iris-platform message | when |
|---|---|
| `catalog.snapshot` (`shelf-void`, `PUBLISH_CATALOG`), then `state.sync.request` | every (re)connect |
| `pipeline.status` `synced` | an upsert whose cameras are all in the mapping file (or a remove) |
| `pipeline.status` `failed`, detail naming the unmapped cameras | any camera of the upsert missing from the mapping file |
| `event.raised` (`shelf.empty_spot`, confidence = spot stability) | `shelf_empty` of a mapped region |
| `event.cleared` | `shelf_restocked` of a mapped region; also `paused` / `camera_removed` for a region left open |

One open IRIS alert per pipeline x camera x region: `correlationKey` is `shelf-void:<cameraId>:<region>`. Detection
settings (confirm, clear, confidence, person pause, regions) come from the sidecar preset, not from the IRIS
pipeline's `config` / `zonePolygon` (the status detail says so).

**Which streams** (`products/shelf_iris_bridge/keys.py`): the bridge never names the stream itself. It loads the
sidecar's own preset (`SIDECAR_CONFIG`, `extends:` followed), resolves the role as the engine does (`IRIS_SIDECAR_ROLE`
if set, else the yaml's `sidecar_role`, then that role's `role_overrides`) and names each mapped camera's stream with
the engine's `stream_key_id` (`perception/engine/sources.py`, what `EngineServices.alert` writes to):

| sidecar `role_scoped_keys` | streams read | mapping lookup |
|---|---|---|
| off (default) | `iris:alert:<camera>` | `<camera>` |
| on (needs a role) | `iris:alert:<camera>@<role>` | `<camera>` (the `@<role>` suffix is stripped) |

The startup log line `alert streams` gives the mode, the role and every stream read. `role_scoped_keys` without a role,
or a camera id containing `@`, refuses startup as it does the sidecar's.

**Run** (the shelf stack, e.g. `just shelf-live-sim VIDEO=/path/to/test.mp4`, which uses
`configs/presets/shelf/live-test.yaml`, cameras `cam-shelf-test` with regions R1/R2):

```bash
cp products/shelf_iris_bridge/.env.example products/shelf_iris_bridge/.env      # then set AMQP_URL (git-ignored)
cp products/shelf_iris_bridge/mapping.example.yaml scratch/shelf-iris-mapping.yaml  # then put the iris-platform ids
source scripts/shelf_live_env.sh
# SIDECAR_CONFIG: the sidecar's own preset (shelf-live-sim's CONFIG=, in $SHELF_LIVE_DIR/current.env); required
SIDECAR_CONFIG=configs/presets/shelf/live-test.yaml SHELF_IRIS_MAPPING=./scratch/shelf-iris-mapping.yaml \
    shelf_compose up -d --build shelf-iris-bridge
shelf_compose logs -f shelf-iris-bridge        # JSON lines; never the AMQP password or camera credentials
```

**Mapping file** (`SHELF_IRIS_MAPPING`, mounted at `/config/shelf-iris-mapping.yaml`): each DeepStream camera (the
source id of the preset) and shelf region (the `roi` name: `R1`, `R2`, ... for a `roi.json` list) maps to the
iris-platform `pipelineId` and `cameraId` it reports to. Several regions may report to one pipeline x camera.

```yaml
cam-shelf-test:
  R1: {pipelineId: <iris-platform pipeline id>, cameraId: <iris-platform camera id>}
  R2: {pipelineId: <another pipeline id>, cameraId: <iris-platform camera id>}
```

**Environment** (`products/shelf_iris_bridge/.env`, from `.env.example`; placeholders only in git): `AMQP_URL`
(required; the broker user iris-platform lists in `AI_PUBLISHER_USERS`; `amqps://` in production), `AMQP_USER`,
`TARGET` (`cloud` | `edge.<slug>`), `AMQP_QUEUE`, `PUBLISH_CATALOG` (`false` when another cloud AI backend publishes
the catalog), `CONFIDENCE_MODE` (`stability` | `detector`), `CONSUMER_GROUP` (default `shelf-iris-bridge`). Compose sets
`REDIS_URL` (`SHELF_BRIDGE_REDIS_URL`, default the stack's `redis`), `MAPPING_FILE`, `SIDECAR_CONFIG` (from your shell,
as for the sidecar; empty = the bridge refuses to start), `IRIS_SIDECAR_ROLE` (passed through from your shell to both
the sidecar and the bridge, so they agree; unset = the preset's `sidecar_role`; a value in the bridge's `.env` is
ignored) and mounts `configs/` and `perception/` read-only, so the
preset and the key helper are the checkout's own.

- **Run either this bridge or the shelf-demo service against one iris-platform, never both**: they publish the same
  catalog id and consume the same queue, so each would get half the upserts and answer the other's pipelines.
- A **local** iris-platform binds RabbitMQ to the host's `127.0.0.1`; join its network instead with a compose
  override (`services: {shelf-iris-bridge: {networks: [default, iris]}}`, `networks: {iris: {external: true, name:
  iris_default}}`) and `AMQP_URL=amqp://...@rabbitmq:5672/iris`.
- Check: `pipeline.status` in the bridge log, the alert in iris-platform, and `iris.platform.inbound.dlq` empty.

---

## Configuration

All runtime tuning lives in `config.yaml` — no code changes needed.

### Perception backend

| Key | Default | Description |
|---|---|---|
| `perception.backend` | `local` | `local` (in-process camera + YOLO, single camera) \| `deepstream` (consume the Redis sidecar; multi-stream) |
| `perception.redis_url` | `redis://redis:6379/0` | Redis data/control plane; use `redis://localhost:6379/0` for a native run |
| `perception.queue_depth` | `3` | Per-stream VLM request queue depth (drop-oldest when full) |
| `perception.discovery_interval_seconds` | `5` | How often IRIS rescans `iris:status:*` for live streams |

> Set `cv.enabled: false` when using the `deepstream` backend — the sidecar already runs YOLO. See [`ARCHITECTURE.md`](ARCHITECTURE.md) and `CLAUDE.md` for the full multi-stream design.

### Camera & Sampling

| Key | Default | Description |
|---|---|---|
| `camera.source` | `0` | Device index, RTSP/HTTP URL, or local file path |
| `sampling.interval_seconds` | `4` | How often a batch is sent for analysis |
| `sampling.frames_per_batch` | `7` | Frames per analysis call |
| `sampling.overlap_frames` | `2` | Frames shared between consecutive batches |
| `sampling.jpeg_quality` | `80` | JPEG quality sent to the API |

### Model

| Key | Default | Description |
|---|---|---|
| `api.backend` | `cloud` | `cloud` (DashScope) \| `local` (any OpenAI-compatible server — vLLM, Ollama, LMStudio) |
| `api.model` | `qwen3.6-35b-a3b` | Cloud vision model; `qwen3-vl-32b-instruct` and `qwen3.6-plus` also supported |
| `api.max_tokens` | `600` | Max tokens in the model response |
| `api.enable_thinking` | `false` | Set `false` for qwen3; omit entirely for qwen2-vl |
| `api.timeout_seconds` | `20` | API call timeout |
| `api.local_base_url` | `http://localhost:8001/v1` | Local inference server URL (when `backend: local`) |
| `api.local_visual_format` | `image_sequence` | `image_sequence` (N `image_url` blocks) \| `video_url` (frames as one mp4 data-URI for video-native models) |

### Monitoring

| Key | Default | Description |
|---|---|---|
| `monitoring.persona` | `traffic` | Startup persona: `traffic` \| `indoor_security` |
| `monitoring.mode` | `proactive` | Startup working mode: `proactive` (run analysis) \| `chat` (analysis paused) |
| `monitoring.output_language` | `english` | Response language: `english` \| `arabic` |
| `monitoring.cooldown_seconds` | `8` | Minimum seconds between COMMENT events |
| `monitoring.alert_cooldown_seconds` | `4` | Minimum seconds between ALERT events |
| `monitoring.max_history` | `20` | Recent events fetched (oldest-first) for the prompt's history block, before per-severity bucketing |
| `monitoring.history_per_severity` | `5` | Max history lines rendered per severity tier (critical/warning/info) |
| `monitoring.pending_queue_depth` | `2` | Max queued analysis calls while one is running |

### Deduplication

| Key | Default | Description |
|---|---|---|
| `dedup.enabled` | `true` | Toggle embedding-based duplicate suppression |
| `dedup.threshold` | `0.93` | Cosine similarity threshold for COMMENTs |
| `dedup.alert_threshold` | `0.97` | Cosine similarity threshold for ALERTs |
| `dedup.window_seconds` | `10` | Only compare against embeddings within this window |
| `dedup.backend` | `api` | `api` (DashScope) \| `local` (fastembed, `pip install fastembed`) |

### CV / YOLO

| Key | Default | Description |
|---|---|---|
| `cv.enabled` | `true` | Enable YOLO enrichment |
| `cv.yolo_model` | `yolo11n.pt` | Model downloaded automatically on first use |
| `cv.confidence` | `0.40` | Minimum detection confidence |
| `cv.annotate_frames` | `true` | Draw boxes on frames sent to the VLM |
| `cv.annotate_stream` | `false` | Draw boxes on the live MJPEG stream in the browser |
| `cv.stream_annotation_fps` | `20` | Stream annotation rate (lower = less CPU) |
| `cv.inference_width` | `640` | Resize frames before YOLO inference; `0` = native |

---

## How It Works

```
Camera (thread)
  ├── Raw frames → _frame_buffer (downsampled)
  └── Raw frames → stream annotation loop
                       └── YOLO → annotated frames → browser stream
                                → _annotated_buffer (sampled at analysis cadence)

Scheduler (every interval_seconds)        ← local backend; gated by mode (skips while in Chat mode or a chat turn is in flight)
  ├── grab_annotated_batch()  ← reuses stream YOLO output (no double inference)
  │   or grab_frame_batch() + process_batch()  ← fallback when stream annotation off
  ├── analyzer.analyze()  → list of events [{type, message, tags}, ...]
  └── for each event: cooldown → dedup → save → WebSocket broadcast → browser

Browser (React SPA)
  ├── useWebSocket  ─► event/chat/config stores (Zustand)
  ├── /#/           Landing page (health badge)
  ├── /#/dashboard  CameraFeed + EventLog/ChatPanel + SystemPromptPanel
  └── /#/analytics  Stats + Recharts (events/hour, alert ratio, top tags, timeline)
```

In the **`deepstream`** backend the DeepStream sidecar owns capture + YOLO + tracking for N streams and publishes per-source detections and annotated JPEGs to Redis; a `StreamManager` discovers the streams, runs one producer per stream (also mode-gated) feeding a single fair VLM worker, and tags each event with its `camera`.

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full component diagram, data flows, and design decisions.

---

## Debug Files

All written to `data/` and cleared by the **Clear Events** button in the UI.

| File | Contents |
|---|---|
| `events.db` | All saved events (SQLite) |
| `cv_debug.json` | Per-cycle YOLO detection counts and frame offsets |
| `debug_frame.jpg` | Middle annotated frame of the last analysis batch |
| `prompt_log.json` | Full system prompt + user text for every `analyze()` call |
| `response_log.json` | Parsed analyzer output arrays with `[hh:mm:ss-hh:mm:ss]` timestamps |
