# IRIS — Repository File Guide

A one-line description of every file in the repo, grouped by directory. See `README.md` for setup/usage and `CLAUDE.md` / `ARCHITECTURE.md` for the full architecture.

## Root

| File | Description |
|---|---|
| `.dockerignore` | Files/dirs excluded from the Docker build context. |
| `.env.example` | Template for `.env`; shows the `QWEN_API_KEY` variable to set. |
| `.gitignore` | Git-ignored paths (venv, node_modules, data, build artifacts, etc.). |
| `ARCHITECTURE.md` | Deep-dive system architecture doc: component diagram, data flows, API endpoint table. |
| `CLAUDE.md` | Reference for Claude Code: setup, full config reference, architecture, module responsibilities. |
| `Dockerfile` | Multi-stage build — compiles the React frontend, then builds the Python FastAPI image. |
| `README.md` | Project overview, features, setup, Docker, and configuration reference. |
| `config.yaml` | Single source of truth for all runtime-tunable settings (camera, sampling, perception, api, monitoring, dedup, chat, cv, etc.). |
| `docker-compose.yml` | Defines the core services (`iris`, `redis`, `minio`, `deepstream-sidecar`, `go2rtc`) plus the optional monitoring stack (`prometheus`, `grafana`, `dcgm-exporter`, `node-exporter`). |
| `main.py` | Entry point — loads config/env, validates the API key, builds the FastAPI app, runs uvicorn. |
| `package-lock.json` | Empty stub npm lockfile at repo root (no root-level Node package; the real one is `frontend/package-lock.json`). |
| `requirements.txt` | Python runtime dependencies. |
| `requirements-dev.txt` | Python dev/test dependencies (pytest, etc.). |

## `app/` — FastAPI backend

| File | Description |
|---|---|
| `app/__init__.py` | Empty package marker. |
| `app/analyzer.py` | `Analyzer` — all VLM calls (`analyze`/`chat`/`chat_multi`), prompt building, response parsing, debug logging. |
| `app/camera.py` | `Camera` — threaded OpenCV capture, frame buffering, MJPEG streaming, file playback pause/resume. |
| `app/chat_frame_buffer.py` | `ChatFrameBuffer` — slow-sampled ring buffer of `(offset, jpeg)` per stream for chat history. |
| `app/chat_search.py` | Pure helpers for the backward chat "search" sweep: FOUND/NOT_FOUND marker parsing, timestamp clamping, clip-stream selection. |
| `app/chat_select.py` | Pure helpers for picking which buffered frames go into a normal chat turn. |
| `app/chat_sessions.py` | `ChatSessionStore` — in-memory chat session state, turn history, TTL pruning. |
| `app/clip_store.py` | `ClipStore` — in-memory store of FOUND search-result preview clips (frames + meta), TTL-pruned. |
| `app/cv_enricher.py` | `CVEnricher` — YOLO object detection/annotation per batch; builds `CVBatchContext` metadata for the VLM prompt. |
| `app/event_tools.py` | `query_events` tool schema/implementation used for chat function-calling against the event DB. |
| `app/events.py` | `EventStore` — `aiosqlite`-backed event log (save/query/analytics), auto-purge by retention. |
| `app/local_chat_context.py` | `LocalChatContext` — single-camera chat-context provider (slow sampler + history sweep) for the local backend. |
| `app/log_setup.py` | `configure_logging()` — applies the config's logging level to the root logger. |
| `app/metrics.py` | `MetricsRegistry` — pure in-memory per-stream analysis timing/queue metrics + status-table rendering. |
| `app/mode.py` | Shared proactive/chat mode gate (`analysis_enabled`) plus the `chat_in_flight` auto-pause counter. |
| `app/perception_source.py` | `PerceptionSource` — Redis-backed `Camera` stand-in for one DeepStream sidecar stream. |
| `app/personas.py` | `PERSONAS` dict — role + default instructions for the `traffic` and `indoor_security` personas. |
| `app/scheduler.py` | `Scheduler` — local-backend asyncio loop driving the proactive analysis cycle (cooldown, dedup, save, broadcast). |
| `app/server.py` | `create_app()` — FastAPI app factory, all HTTP/WebSocket routes, lifespan task wiring. |
| `app/similarity.py` | `CommentFilter` — embedding-based cosine-similarity duplicate suppression for events. |
| `app/stream_manager.py` | `StreamManager` — DeepStream-backend multi-stream orchestrator (discovery, producers, fair VLM worker). |

## `perception/` — the DeepStream perception sidecar

Start with [`perception/README.md`](perception/README.md) (reading order) and [`perception/ARCHITECTURE.md`](perception/ARCHITECTURE.md)
(the engine, the modules and how they interact, with diagrams).

| Path | Description |
|---|---|
| `perception/engine/` | The use-case-free DeepStream engine, run as `python3 -m perception.engine.main <yaml>`: `gst_build.py` (pipeline), `probe.py` (per-frame hot path), `frame.py` (`FrameView`/`ObjectView`/`FrameOutput`), `modules.py` (the module contract, loader, `EngineServices`), `publisher.py` (preview drawing, JPEG, Redis), `objects.py` + `nvds.py` (the only pyds readers), `gie_gate.py` (run a model on fewer frames/objects), `recording.py`/`svr*.py`/`segment_index.py` (recording and alert clips), `source_recovery.py`/`flight.py` (recovery), `sources.py`/`config.py` (config, roles), `telemetry.py`, `control.py`, `requirements.txt` |
| `perception/modules/<name>/` | One folder per capability, switched on by `modules:` in a product yaml: `zones`, `fire_alerts`, `box_stabilizer`, `person_boxes`, `person_masks`, `face_detect`, `face_recognition`, `customer_faces`, `appearance`, `plates`, `mtmc`, and `_template` to copy. See [`perception/modules/README.md`](perception/modules/README.md). Each may hold a `requirements.txt` |
| `perception/lib/` | Helpers shared by modules and the engine that import no GStreamer: `versioned.py` (app-edited settings with reload and acknowledgement), `overcrowding_dedup.py`, `imaging.py` |
| `perception/models/` | One folder per model family: nvinfer configs, ONNX files, and `nvdsinfer_custom_impl_*/` parsers (built into the image by `build_parsers.sh`; `parser_flags` holds per-parser `make` variables) |
| `perception/pipeline/sidecar.yaml` | The shared engine settings. Names no product, no module and no camera; a product overlay (`products/*/sidecar.yaml`) adds them |
| `perception/pipeline/Dockerfile` | The sidecar image: targets `slim` (default) and `mtmc` (torch, Re-ID, VLM clients) |
| `perception/pipeline/*.py` | Diagnostic and capture tools (`register_face.py`, `run_diag.py`, `capture_*`, ...), not part of the running engine |
| `perception/ARCHITECTURE.md`, `NEXT_MODULES.md`, `TESTING.md` | Architecture with diagrams; what fall/PPE/proximity need; how every change is verified |
| `tools/engine_golden*.sh`, `engine_golden.py` | The golden replay: record what the sidecar writes to Redis on a recorded clip and diff two runs |
| `tools/engine_gpu_budget.sh`, `.py` | Per-model GPU cost of a config |

## `monitoring/` — Prometheus/Grafana/DCGM stack

| File | Description |
|---|---|
| `monitoring/prometheus.yml` | Prometheus scrape config: `deepstream-sidecar:9100` (per shard), `dcgm-exporter:9400`, `node-exporter:9100`. |
| `monitoring/grafana/provisioning/datasources/prometheus.yml` | Provisioned Prometheus datasource (auto-created on first Grafana boot). |
| `monitoring/grafana/provisioning/dashboards/dashboards.yml` | Provisioning file that loads the IRIS dashboard folder. |
| `monitoring/grafana/dashboards/iris.json` | The "IRIS — pipeline & GPU" dashboard: DCGM GPU panels (util, NVDEC decode util, memory, temp, power) + sidecar pipeline panels (stage busy%, GPU-element busy%, per-call ms, per-cam fps, drops, mux queue, SGIE mode). |
| `monitoring/README.md` | How to bring up the stack, enable the sidecar metrics, and reach each service. |

## `tests/` — pytest suite

| File | Description |
|---|---|
| `tests/__init__.py` | Empty package marker. |
| `tests/test_analyzer_analytics.py` | Tests `Analyzer._format_analytics` formatting of the analytics summary block. |
| `tests/test_analyzer_chat.py` | Tests `Analyzer` chat history formatting (camera tagging, prompt assembly). |
| `tests/test_analyzer_logging.py` | Tests `Analyzer`'s per-call logging/metrics recording. |
| `tests/test_chat_auto_pause.py` | Tests that proactive analysis auto-pauses while a chat turn is in flight. |
| `tests/test_chat_context.py` | Tests chat frame-selection helpers (`_stride_pick`, `select_chat_frames`). |
| `tests/test_chat_debug.py` | Tests the per-mode chat debug dump (prompt + frames) and image-payload elision. |
| `tests/test_chat_endpoint_sse.py` | Tests the `/api/chat` SSE endpoint (`status`/`token`/`done`/`error` events). |
| `tests/test_chat_frame_buffer.py` | Tests `ChatFrameBuffer` append/snapshot behavior. |
| `tests/test_chat_multi_stream.py` | Tests `Analyzer.chat_multi` across multiple camera streams. |
| `tests/test_chat_search.py` | Tests the backward search-sweep helpers (marker parsing, timestamp clamping, stream selection). |
| `tests/test_chat_select.py` | Tests `recent_strided` frame-selection logic for a normal chat turn. |
| `tests/test_chat_sessions.py` | Tests `ChatSessionStore` session/turn/sweep-cursor persistence. |
| `tests/test_clip_encode.py` | Tests mp4 encoding of preview-clip frames in `Analyzer`. |
| `tests/test_clip_store.py` | Tests `ClipStore` add/get/TTL-prune behavior. |
| `tests/test_event_tools.py` | Tests `resolve_relative()` date-keyword resolution for the `query_events` tool. |
| `tests/test_events_analytics.py` | Tests `EventStore.analytics_summary()` aggregation. |
| `tests/test_events_camera.py` | Tests `EventStore` camera-column tagging/filtering. |
| `tests/test_events_query.py` | Tests `EventStore.query()` filtering (time range, type, camera). |
| `tests/test_local_chat_context.py` | Tests `LocalChatContext`'s slow sampler and chat-context gathering. |
| `tests/test_logging_config.py` | Tests `configure_logging()` applies the configured level. |
| `tests/test_metrics.py` | Tests `MetricsRegistry` recording and table rendering. |
| `tests/test_metrics_task.py` | Tests the background metrics-logging loop task. |
| `tests/test_mode.py` | Tests `analysis_enabled()` mode-gate logic. |
| `tests/test_mode_endpoint.py` | Tests the `/api/mode` endpoint. |
| `tests/test_perception_source.py` | Tests `PerceptionSource` reading Redis-backed stream data (via `fakeredis`). |
| `tests/test_personas.py` | Tests persona instruction content (e.g. right-turn-on-red exception present). |
| `tests/test_scheduler_gate.py` | Tests `Scheduler` skips cycles when the camera isn't running/paused or is mode-gated. |
| `tests/test_scheduler_obs.py` | Tests `Scheduler`'s pending-queue-depth observability accessor. |
| `tests/test_sidecar_sources.py` | Tests `sidecar.sources.resolve_sources`/`make_stream_state` config helpers. |
| `tests/test_stream_manager.py` | Tests `StreamManager`'s `pick_source`/`enqueue_drop_oldest`/`should_emit` pure helpers. |
| `tests/test_stream_manager_chat.py` | Tests `StreamManager`'s multi-stream chat-context gathering. |
| `tests/test_stream_manager_obs.py` | Tests `StreamManager` observability (queue/pending state) via `fakeredis`. |
| `tests/test_think_stripper.py` | Tests `Analyzer._ThinkStripper` incremental `<think>` block stripping. |

## `frontend/` — React/TypeScript SPA (builds into `static/`)

### Config

| File | Description |
|---|---|
| `frontend/.prettierrc` | Prettier formatting rules (no semicolons, single quotes, 100-char width). |
| `frontend/eslint.config.js` | ESLint flat config for TS/React with hooks and refresh plugins. |
| `frontend/index.html` | Vite HTML entry point; loads fonts and mounts the React root. |
| `frontend/package.json` | npm manifest listing scripts and dependencies (React 19, Vite, Zustand, TanStack Query, Recharts). |
| `frontend/package-lock.json` | npm lockfile for the frontend package. |
| `frontend/public/favicon.svg` | SVG favicon icon (IRIS eye/wave mark). |
| `frontend/tsconfig.json` | Root TypeScript project-references config pointing to the app/node configs. |
| `frontend/tsconfig.app.json` | TypeScript config for the app source (strict, bundler resolution, `@/*` path alias). |
| `frontend/tsconfig.node.json` | TypeScript config for Vite's own config file. |
| `frontend/vite.config.ts` | Vite build config: React plugin, `@` alias, output to `../static`, dev API/WS proxy. |

### `frontend/src/` — application code

| File | Description |
|---|---|
| `src/App.tsx` | Top-level router: landing page vs. app-shell-wrapped dashboard/analytics routes. |
| `src/main.tsx` | React app bootstrap: mounts root with `QueryClient` and `HashRouter` providers. |
| `src/vite-env.d.ts` | Vite client type reference for TS. |
| `src/api/camera.ts` | API client for camera start/stop/pause/switch/upload endpoints. |
| `src/api/chat.ts` | Streams `/api/chat` SSE responses and exposes session deletion. |
| `src/api/client.ts` | Shared fetch helpers (GET/POST/DELETE) with a typed `ApiError`. |
| `src/api/config.ts` | API client for prompt, persona, language, and mode endpoints. |
| `src/api/events.ts` | API client to list and clear stored events. |
| `src/api/streams.ts` | API client for multi-stream camera listing and pipeline start/stop. |
| `src/components/analytics/aggregations.ts` | Pure functions computing hourly/timeline/tag/ratio stats from events. |
| `src/components/camera/CameraControls.tsx` (+ `.module.css`) | Buttons to start/stop/pause camera, switch source, and upload video. |
| `src/components/camera/CameraFeed.tsx` (+ `.module.css`) | Single-camera MJPEG viewer with offset badge, alert overlay, and controls. |
| `src/components/chat/ChatPanel.tsx` (+ `.module.css`) | Chat UI: message stream, normal/search mode toggle, SSE token rendering. |
| `src/components/chat/ClipPreviewModal.tsx` (+ `.module.css`) | Modal that plays a looping MJPEG preview of a search-result clip. |
| `src/components/events/EventCard.tsx` (+ `.module.css`) | Renders one event (type badge, camera, offsets, timestamp, tags). |
| `src/components/events/EventFilters.tsx` (+ `.module.css`) | Type/time-range/text filter controls for the event log. |
| `src/components/events/EventLog.tsx` (+ `.module.css`) | Scrollable, auto-scrolling list of event cards with empty states. |
| `src/components/events/RightPanel.tsx` (+ `.module.css`) | Right sidebar toggling between the proactive event log and the chat panel. |
| `src/components/layout/AlertFlash.tsx` (+ `.module.css`) | Brief full-screen flash effect triggered by a new alert event. |
| `src/components/layout/AppShell.tsx` (+ `.module.css`) | Wraps routed pages with the persistent Header. |
| `src/components/layout/Header.tsx` (+ `.module.css`) | Top nav bar: brand, live/status badges, mode toggle, page links, event count. |
| `src/components/layout/ModeToggle.tsx` (+ `.module.css`) | Proactive/chat mode switch button synced with the backend. |
| `src/components/prompt/SystemPromptPanel.tsx` (+ `.module.css`) | Editable monitoring-instructions panel with persona/language selectors. |
| `src/components/streams/ExpandedStream.tsx` (+ `.module.css`) | Modal detail view of one stream's feed plus its own event log. |
| `src/components/streams/StreamGrid.tsx` (+ `.module.css`) | Multi-stream video wall grid with pipeline start/stop control. |
| `src/components/streams/StreamTile.tsx` (+ `.module.css`) | One clickable stream thumbnail tile with live status and alert flash. |
| `src/components/streams/layout.ts` | Pure helpers computing mosaic row/column layout for the stream grid. |
| `src/components/ui/BrandLogo.tsx` | Reusable IRIS logo SVG component. |
| `src/components/ui/icons.tsx` | Shared inline SVG icon components used across the UI. |
| `src/components/upload/UploadOverlay.tsx` (+ `.module.css`) | Modal showing upload/processing progress and errors for video uploads. |
| `src/hooks/useDebounce.ts` | Generic hook debouncing a value by a delay. |
| `src/hooks/useElementSize.ts` | Hook tracking an element's size via `ResizeObserver`. |
| `src/hooks/useEventFilters.ts` | Hook managing type/time/text filter state and the filtered event list. |
| `src/hooks/useEventsForCamera.ts` | Hook selecting events for one camera (or all) from the event store. |
| `src/hooks/useStreamTime.ts` | Hook polling `/api/camera/time` for the current file playback offset. |
| `src/hooks/useStreams.ts` | Hook polling `/api/cameras` to sync the streams store and detect backend mode. |
| `src/hooks/useWebSocket.ts` | Hook managing the app's WebSocket connection, reconnection, and event/mode dispatch. |
| `src/lib/format.ts` | Date/time and stream-offset formatting helpers. |
| `src/pages/AnalyticsPage.tsx` (+ `.module.css`) | Analytics dashboard with Recharts stat cards, timeline, and tag charts. |
| `src/pages/DashboardPage.tsx` (+ `.module.css`) | Main dashboard page switching between the legacy feed and the multi-stream grid. |
| `src/pages/LandingPage.tsx` (+ `.module.css`) | Marketing landing page with backend health check and nav CTAs. |
| `src/stores/cameraStore.ts` | Zustand store for single-camera running/paused/source/device-list state. |
| `src/stores/chatStore.ts` | Zustand store for chat messages, streaming state, session id, and search mode. |
| `src/stores/configStore.ts` | Zustand store for active persona, personas list, language, and prompt text. |
| `src/stores/eventStore.ts` | Zustand store for the event list, WS status, and last-alert tracking (global + per-camera). |
| `src/stores/modeStore.ts` | Zustand store for the proactive/chat working mode. |
| `src/stores/streamsStore.ts` | Zustand store for discovered streams, expanded overlay id, and pipeline desired state. |
| `src/styles/buttons.css` | Shared `.iris-btn`/`.iris-select` utility class styles. |
| `src/styles/globals.css` | Global resets, background mesh/noise, scrollbars, and shared keyframe animations. |
| `src/styles/tokens.css` | CSS custom-property design tokens (colors, glass, elevation, radius, fonts). |
| `src/types/camera.ts` | TypeScript types for camera/stream status and API responses. |
| `src/types/chat.ts` | TypeScript types for chat messages, clips, and streaming handlers. |
| `src/types/config.ts` | TypeScript types for language, mode, and persona config responses. |
| `src/types/event.ts` | TypeScript types for the `IrisEvent` shape. |
| `src/types/ws.ts` | TypeScript discriminated-union types for WebSocket messages. |

## `static/` — built frontend output (generated by `npm run build`)

| File | Description |
|---|---|
| `static/index.html` | Built SPA entry HTML served by FastAPI with SPA fallback. |
| `static/favicon.svg` | Favicon copied from `frontend/public`. |
| `static/assets/*.js` / `static/assets/*.css` | Compiled, content-hashed JS/CSS bundle — regenerated (new hashes) on every frontend build. |

## `data/` — runtime data (mostly gitignored)

| File | Description |
|---|---|
| `data/.gitkeep` | Keeps the empty `data/` directory tracked in git. |
| `data/events.db` | SQLite event database (created at runtime). |
| `data/cv_debug.json`, `data/prompt_log.json`, `data/response_log.json` | Rolling debug logs written by the CV enricher and analyzer. |
| `data/debug_frame.jpg` | Latest YOLO-annotated debug frame, overwritten each cycle. |
| `data/analysis_calls/`, `data/chat_calls/`, `data/description_calls/` | Rolling per-call debug dumps (frames + prompts) for analysis and chat. |
| `data/uploads/` | User-uploaded video files streamed as camera sources. |
