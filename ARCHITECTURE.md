# IRIS — System Architecture

## Overview

IRIS is a FastAPI-based AI surveillance application that captures frames from a webcam or video file, enriches them with local YOLO object detection, sends batches to a Vision Language Model (Alibaba's DashScope), and streams observations to a browser UI in real time over WebSocket.

The browser UI is a **React 19 + Vite + TypeScript** SPA living in `frontend/`. It is compiled to plain HTML/JS/CSS into `static/`, which FastAPI serves on the same port as the APIs and WebSocket — production runs as a single service.

---

## Input Sources

| Source | Description |
|---|---|
| Uploaded Video | `.mp4`, `.avi`, `.mkv`, `.mov`, `.webm` uploaded via the UI; loops on EOF |
| Webcam | Device index (e.g. `0`) configured in `config.yaml` |
| RTSP / HTTP | Any URL string accepted by `cv2.VideoCapture` |

---

## Component Diagram

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                              INPUT SOURCES                                   ║
╠══════════════╦══════════════════╦════════════════════════════════════════════╣
║  Webcam      ║  Uploaded Video  ║  RTSP / HTTP stream                        ║
║  device idx  ║  .mp4 .avi etc   ║  (URL in config.yaml or via upload)        ║
╚══════════════╩══════════════════╩════════════════════════════════════════════╝
                           │
                           ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                        app/camera.py  —  Camera                              ║
║                                                                              ║
║  • _capture_loop (background thread, ~30 fps)                                ║
║    - reads frames via cv2.VideoCapture                                       ║
║    - JPEG-encodes each frame (quality + optional CLAHE + max_width resize)   ║
║    - writes to _latest_frame (for stream display and chat)                   ║
║    - downsamples into _frame_buffer deque at buffer_interval cadence         ║
║      (buffer_interval = interval_seconds / (frames_per_batch - overlap))     ║
║  • Pause / resume (threading.Event) for video file playback                  ║
║  • _annotated_buffer: populated by stream annotator at buffer_interval;      ║
║    lets _cycle() skip redundant YOLO when stream annotation is enabled       ║
║  • Video files: pre-reads first batch into _first_batch before thread starts ║
║                                                                              ║
║  Consumer APIs:                                                              ║
║    grab_frame()            → latest raw JPEG (chat, stream annotator)        ║
║    grab_frame_batch()      → (frames[], offsets[]) with overlap prepended    ║
║    grab_annotated_batch()  → (frames[], detections[], offsets[]) or None     ║
║    push_annotated_frame()  → enqueues annotated frame at buffer_interval     ║
║    grab_display_frame()    → annotated frame if available, else raw          ║
║    generate_mjpeg()        → async MJPEG generator (~25 fps to browser)      ║
╚══════════════════════════════════════════════════════════════════════════════╝
       │                    │                        │
       │ grab_frame_batch() │ grab_display_frame()   │ grab_frame()
       │ or                 │ (~25 fps)              │ (stream annotator)
       │ grab_annotated_    │                        │
       │ batch()            ▼                        ▼
       │            ╔═══════════════╗   ╔════════════════════════════════════╗
       │            ║  GET /stream  ║   ║  _stream_annotation_loop           ║
       │            ║  (server.py)  ║   ║  (server.py, asyncio task)         ║
       │            ║               ║   ║                                    ║
       │            ║  Streaming    ║   ║  Dedicated ThreadPoolExecutor      ║
       │            ║  Response     ║   ║  (max_workers=1, isolated from     ║
       │            ║  MJPEG        ║   ║   batch YOLO pool)                 ║
       │            ║  → browser    ║   ║                                    ║
       │            ╚═══════════════╝   ║  1. grab_frame() → raw JPEG        ║
       │                                ║  2. enricher.annotate_single()     ║
       │                                ║     → (annotated JPEG, detections) ║
       │                                ║  3. camera.set_annotated_frame()   ║
       │                                ║     → display via grab_display_    ║
       │                                ║       frame() in MJPEG stream      ║
       │                                ║  4. camera.push_annotated_frame()  ║
       │                                ║     → feeds _annotated_buffer for  ║
       │                                ║       scheduler reuse              ║
       │                                ╚════════════════════════════════════╝
       ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                      app/scheduler.py  —  Scheduler                          ║
║                                                                              ║
║  asyncio loop, fires every interval_seconds (default 4s)                     ║
║  Skips tick if camera is not running or is paused.                           ║
║  Pending queue (depth configurable): queues fired ticks while analysis runs; ║
║  drops ticks beyond pending_queue_depth.                                     ║
║                                                                              ║
║  _cycle():                                                                   ║
║  1. grab_annotated_batch() → if non-None: reuse stream YOLO output           ║
║     else: grab_frame_batch() → run process_batch() in thread executor        ║
║  2. store.get_recent()    — fetch last N events as history context           ║
║  3. analyzer.analyze()    — send batch to VLM → list[{type,message,tags}]    ║
║  4. For each returned event:                                                 ║
║       a. _should_emit()   — cooldown check per type                          ║
║       b. comment_filter.is_duplicate() — embedding cosine similarity check   ║
║       c. store.save()     — persist to SQLite                                ║
║       d. comment_filter.record_emitted() — commit embedding anchor           ║
║       e. broadcast()      — push event to all WebSocket clients              ║
║  Catch-up: if pending > 0 and camera still running → chain next _cycle()     ║
╚══════════════════════════════════════════════════════════════════════════════╝
       │                          │                        │
       │ run_in_executor          │ analyze()              │ store.save/get
       ▼                          ▼                        ▼
╔══════════════════╗   ╔══════════════════════╗   ╔═══════════════════════════╗
║ app/cv_enricher  ║   ║  app/analyzer.py     ║   ║  app/events.py            ║
║ CVEnricher       ║   ║  Analyzer            ║   ║  EventStore               ║
║                  ║   ║                      ║   ║                           ║
║ process_batch(): ║   ║  analyze():          ║   ║  SQLite (data/events.db)  ║
║  • YOLO on each  ║   ║  • Builds system     ║   ║                           ║
║    frame         ║   ║    prompt from role, ║   ║  save(type,msg,tags,      ║
║  • filters by    ║   ║    persona instrs,   ║   ║       offset_start/end)   ║
║    relevant_cls  ║   ║    and history       ║   ║  get_recent(limit,types)  ║
║  • annotates     ║   ║  • Appends CV        ║   ║                           ║  
║    frames if     ║   ║    metadata block    ║   ║  clear_all()              ║
║   annotate_frames║   ║  • Returns           ║   ║                           ║  
║     is True      ║   ║    list[dict]        ║   ║                           ║
║                  ║   ║                      ║   ╚═══════════════════════════╝
║                  ║   ║                      ║
╚══════════════════╝   ╚══════════════════════╝
                                  │
                                  │ is_duplicate() / record_emitted()
                                  ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                    app/similarity.py  —  CommentFilter                       ║
║                                                                              ║
║  Embedding-based duplicate suppression (cosine similarity).                  ║
║  Backend: "local" (fastembed) or "api" (compatible embeddings API).          ║
║                                                                              ║
║  is_duplicate(text, type):                                                   ║
║    • Embeds text; compares against stored reference embeddings               ║
║    • comments → compared vs last comment AND last alert                      ║
║    • alerts   → compared vs last alert only                                  ║
║    • Embeddings older than window_seconds are ignored                        ║
║    • Suppressed events do NOT shift the anchor (pending pattern)             ║
║  record_emitted(type): commits embedding only when event is actually saved   ║
╚══════════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════════╗
║               app/personas.py  —  PERSONAS                                   ║
║                                                                              ║
║  Defines role + default instructions per persona.                            ║
║  • "traffic"        — vehicle detection, traffic flow, red-light violations  ║
║  • "indoor_security"— people monitoring, access control, physical threats    ║
║  Per-persona instruction edits are tracked independently at runtime.         ║
╚══════════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════════╗
║                    app/server.py  —  FastAPI App                             ║
║                                                                              ║
║  ConnectionManager — holds active WebSocket connections; broadcasts JSON     ║
║  Lifespan tasks:                                                             ║
║    • scheduler.run()              — proactive analysis loop                  ║
║    • _stream_annotation_loop()    — if cv.annotate_stream=True               ║
╚══════════════════════════════════════════════════════════════════════════════╝
          │  WebSocket  /ws
          │  push: history, event, status, error, cleared
          ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                  Browser  —  React SPA (frontend/ → static/)                 ║
║                                                                              ║
║  HashRouter routes:                                                          ║
║    /#/            LandingPage   — hero + glass feature cards + health badge  ║
║    /#/dashboard   DashboardPage — live monitoring (default app surface)      ║
║    /#/analytics   AnalyticsPage — stats + Recharts panels                    ║
║                                                                              ║
║  useWebSocket hook (mounted at App.tsx)                                      ║
║    ws://.../ws  →  dispatches typed WsMessage union into eventStore          ║
║      • history  → setHistory(events[])                                       ║
║      • event    → addEvent(event)                                            ║
║      • status   → setStatus('idle'|'analyzing'|'paused'|'error')             ║
║      • cleared  → clear()                                                    ║
║                                                                              ║
║  ┌─────────────────── DashboardPage ───────────────────────────────────┐     ║
║  │  ┌──────────────────────────┐  ┌─────────────────────────────────┐  │     ║
║  │  │ CameraFeed.tsx           │  │ RightPanel.tsx                  │  │     ║
║  │  │ <img src="/stream">      │  │  ┌─ EventFilters ───────────┐   │  │     ║
║  │  │ CameraControls.tsx       │  │  │ type chips · search ·    │   │  │     ║
║  │  │ VideoUpload (file input) │  │  │ time-range select        │   │  │     ║
║  │  │ time badge (file source) │  │  └──────────────────────────┘   │  │     ║
║  │  │ alert overlay (8s)       │  │  ┌─ EventLog / EventCard ────┐  │  │     ║
║  │  │                          │  │  │ glass cards (comment      │  │  │     ║
║  │  └──────────────────────────┘  │  │   /alert with glow)       │  │  │     ║
║  │                                │  └───────────────────────────┘  │  │     ║
║  │                                │  ┌─ ChatPanel ───────────────┐  │  │     ║
║  │                                │  │ glass bubbles + suggestion│  │  │     ║
║  │                                │  │ chips + typing indicator  │  │  │     ║
║  │                                │  │ session_id in chatStore   │  │  │     ║
║  │                                │  └───────────────────────────┘  │  │     ║
║  │                                └─────────────────────────────────┘  │     ║
║  │  ┌─ SystemPromptPanel.tsx (collapsible) ────────────────────────┐   │     ║
║  │  │ persona select · language select · save                      │   │     ║
║  │  └──────────────────────────────────────────────────────────────┘    │    ║
║  └──────────────────────────────────────────────────────────────────────┘    ║
║                                                                              ║
║  Stack: React 19 · Vite 6 · TypeScript · Zustand (4 stores) · React Router 7 ║
║         · TanStack Query · Recharts · CSS Modules + design tokens            ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## Data Flows

### Proactive Monitoring — annotated buffer path (stream annotation enabled)

```
_capture_loop (thread, ~30fps)
    └─► _latest_frame

_stream_annotation_loop (asyncio, stream_annotation_fps)
    ├─► grab_frame() → annotate_single() [dedicated ThreadPoolExecutor]
    │       → (annotated JPEG, detections[])
    ├─► set_annotated_frame()  → served via MJPEG stream with YOLO boxes
    └─► push_annotated_frame() → _annotated_buffer (sampled at buffer_interval)

Scheduler tick (every interval_seconds)
    ├─► grab_annotated_batch() → frames[], detections[], offsets[]
    │       (YOLO already done — skips process_batch entirely)
    ├─► CVBatchContext built from cached detections
    ├─► store.get_recent()     → history context
    ├─► analyzer.analyze()     → list[{type, message, tags}]
    └─► for each event:
            cooldown check → dedup check → store.save() → broadcast()
```

### Proactive Monitoring — fallback path (stream annotation disabled)

```
_capture_loop (thread, ~30fps)
    ├─► _latest_frame
    └─► _frame_buffer (downsampled at buffer_interval, with overlap)

Scheduler tick (every interval_seconds)
    ├─► grab_frame_batch()     → raw frames[], offsets[]
    ├─► process_batch()        → YOLO inference [default ThreadPoolExecutor]
    │       → CVBatchContext (annotated frames[], FrameMeta[])
    ├─► store.get_recent()     → history context
    ├─► analyzer.analyze()     → list[{type, message, tags}]
    └─► for each event:
            cooldown check → dedup check → store.save() → broadcast()
```

### Chat (conversational, on demand)

```
Browser → POST /api/chat {message, session_id?}
    ├─► ChatSessionStore.get_or_create()
    ├─► camera.grab_frame()      → current live frame
    ├─► store.get_recent()       → recent event log for context
    ├─► analyzer.chat(frame, message, history, prior_turns)
    │       → DashScope → plain text reply
    └─► session updated → {session_id, reply} → browser
```

### Live Video Feed

```
_capture_loop → _latest_frame / _annotated_frame
    └─► camera.grab_display_frame()   (annotated if available, else raw)
            └─► generate_mjpeg()
                    └─► GET /stream → StreamingResponse → <img src="/stream">
```
---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/stream` | MJPEG live video feed (annotated if stream annotation enabled) |
| `POST` | `/api/camera/start` | Start camera |
| `POST` | `/api/camera/stop` | Stop camera |
| `GET` | `/api/camera/status` | `{running, paused, source, is_file}` |
| `POST` | `/api/camera/pause` | Pause video file playback |
| `POST` | `/api/camera/resume` | Resume video file playback |
| `GET` | `/api/camera/time` | `{offset, is_file}` — current stream position in seconds |
| `POST` | `/api/camera/use-camera` | Stop current source, switch to default webcam, start |
| `GET` | `/api/cameras` | List available webcam indices |
| `POST` | `/api/camera/switch/{index}` | Switch to a different webcam index |
| `POST` | `/api/upload-video` | Upload video file and begin streaming it |
| `POST` | `/api/chat` | Chat turn `{message, session_id?}` → `{session_id, reply}` |
| `DELETE` | `/api/chat/session/{session_id}` | Delete a chat session |
| `GET` | `/api/prompt` | Get current monitoring instructions |
| `POST` | `/api/prompt` | Update monitoring instructions (runtime only) |
| `GET` | `/api/personas` | List personas and active persona id |
| `POST` | `/api/persona/{persona_id}` | Switch active persona |
| `GET` | `/api/language` | Get current output language |
| `POST` | `/api/language` | Set output language (`english` \| `arabic`) |
| `GET` | `/api/events` | Recent events from SQLite (default limit 100) |
| `DELETE` | `/api/events` | Clear all events, CV log, prompt log, response log |
| `GET` | `/api/config` | Non-sensitive config snapshot |
| `GET` | `/api/debug/frame` | Latest YOLO-annotated middle frame as JPEG |
| `WS` | `/ws` | WebSocket — `history`, `event`, `status`, `error`, `cleared` messages |

---