# Frontend Developer Guide — IRIS Multi-Camera Surveillance

## Overview

IRIS streams 5 real factory camera feeds through go2rtc (RTSP → WebRTC) and
publishes multi-camera vehicle tracking data (global IDs, transitions, journey
timelines) via REST + WebSocket for the frontend to consume.

---

## 1. Camera Feed Streaming

### How it works

```
products/el-mohandes/videos/*.mp4
  → test_streamer/stream_recordings.py (ffmpeg -re -stream_loop -1 -c copy → RTSP)
    → go2rtc (RTSP ingest as raw-cam-N → derived cam-N with lazy H.264 transcode)
      → Browsers via WebRTC (/api/webrtc?src=cam-N)
      → DeepStream sidecar via RTSP (rtsp://go2rtc:8554/raw-cam-N)
```

### Start the streams

```bash
# 1. Start go2rtc (media server)
docker compose up -d go2rtc

# 2. Start the stream publisher (loops products/el-mohandes/videos as live cameras)
python3 test_streamer/stream_recordings.py --host localhost --port 8554

# Optionally filter to specific cameras:
python3 test_streamer/stream_recordings.py --ids cam-1,cam-2,cam-3
```

### Camera mapping

| Stream ID   | Video File               | Site Camera |
|-------------|--------------------------|-------------|
| `cam-1`     | `1-D41-enter.mp4`        | D41         |
| `cam-2`     | `2-D16_clip.mp4`         | D16         |
| `cam-3`     | `3-D52-enter.mp4`        | D52         |
| `cam-4`     | `4-ch05_clip.mp4`        | CH05        |
| `cam-5`     | `5-CH23-enter.mp4`       | CH23        |

### Access the streams

- **WebRTC (browser)**: `http://localhost:1984/api/webrtc?src=cam-1`
- **go2rtc UI**: `http://localhost:1984`
- **RTSP (internal)**: `rtsp://localhost:8554/raw-cam-1` (original codec)
- **RTSP (derived)**: `rtsp://localhost:8554/cam-1` (browser-safe H.264)

### Camera name mapping (internal → display)

The backend uses internal IDs (`cam-01` through `cam-05`) and maps them to
display names via `config.yaml → mtmc.camera_names`:

```yaml
mtmc:
  camera_names:
    cam-01: Cam-41
    cam-02: Cam-16
    cam-03: Cam-52
    cam-04: Cam-05
    cam-05: Cam-23
```

---

## 2. Multi-Camera Tracking Data (MTMC)

### Data flow

```
DeepStream sidecar (per-frame detections + Re-ID embeddings)
  → publishes to Redis streams: iris:meta:<cam-id>
    → app/tracking.py (TrackingService) reads Redis, aggregates global tracks
      → REST endpoints + WebSocket broadcast
        → Frontend (TrackingPage, MapPage)
```

### REST API Endpoints

All endpoints are served by the IRIS backend at `http://localhost:8000`.

#### `GET /api/tracking/tracks?limit=100`

Returns global vehicle tracks, newest first.

```json
[
  {
    "global_id": 1,
    "cameras_seen": ["cam-01", "cam-02", "cam-03"],
    "total_cameras": 3,
    "local_track_ids": {},
    "entry_times": { "cam-01": 12.5, "cam-02": 35.2, "cam-03": 58.1 },
    "exit_times": { "cam-01": 34.0, "cam-02": 57.0 },
    "label": "car",
    "transitions": [
      {
        "from_cam": "cam-01",
        "to_cam": "cam-02",
        "similarity": 0,
        "travel_time_s": 21.5,
        "timestamp_s": 35.2
      }
    ],
    "is_active": true,
    "first_seen_s": 12.5,
    "last_seen_s": 58.1,
    "total_duration_s": 45.6
  }
]
```

**Fields:**
- `global_id` — stable cross-camera vehicle identity (assigned by sidecar MTMC)
- `cameras_seen` — ordered list of cameras this vehicle appeared on
- `entry_times` / `exit_times` — PTS (presentation timestamp) in seconds per camera
- `transitions` — camera-to-camera handoff records with travel time
- `is_active` — true if vehicle was seen within the last 5 seconds
- `first_seen_s` / `last_seen_s` — earliest/latest PTS across all cameras
- `total_duration_s` — span from first entry to last exit

#### `GET /api/tracking/transitions?limit=100`

Returns flat cross-camera transition events, newest first.

```json
[
  {
    "global_id": 1,
    "from_cam": "cam-01",
    "to_cam": "cam-02",
    "similarity": 0,
    "travel_time_s": 21.5,
    "timestamp_s": 35.2
  }
]
```

#### `GET /api/tracking/stats`

Returns aggregate tracking statistics.

```json
{
  "total_tracks": 15,
  "active_tracks": 3,
  "total_transitions": 8,
  "avg_travel_time_s": 25.3,
  "avg_similarity": 0,
  "multi_camera_tracks": 5,
  "cameras_online": 5,
  "uptime_s": 3600.0
}
```

#### `GET /api/tracking/cameras`

Returns per-camera summary.

```json
[
  { "id": "cam-01", "name": "Cam-41", "active_vehicles": 2, "finished_vehicles": 0 },
  { "id": "cam-02", "name": "Cam-16", "active_vehicles": 1, "finished_vehicles": 0 }
]
```

#### `GET /api/tracking/config`

Returns tracking configuration.

```json
{
  "cameras": ["cam-01", "cam-02", "cam-03", "cam-04", "cam-05"],
  "camera_names": {
    "cam-01": "Cam-41",
    "cam-02": "Cam-16",
    "cam-03": "Cam-52",
    "cam-04": "Cam-05",
    "cam-05": "Cam-23"
  },
  "min_travel_s": 0,
  "sim_threshold": 0,
  "running": true
}
```

### WebSocket Messages

Connect to `ws://localhost:8000/ws`. The backend broadcasts tracking updates
every 2 seconds.

#### `tracking_update` (broadcast every 2s)

```json
{
  "type": "tracking_update",
  "tracks": [ /* same format as /api/tracking/tracks */ ],
  "stats": { /* same format as /api/tracking/stats */ },
  "cameras": [ /* same format as /api/tracking/cameras */ ]
}
```

#### `tracking_match` (real-time match event)

```json
{
  "type": "tracking_match",
  "global_id": 1,
  "from_cam": "cam-01",
  "to_cam": "cam-02",
  "similarity": 0.72,
  "travel_time_s": 21.5
}
```

### Camera topology

The 5 cameras form a unidirectional chain (vehicles flow one way):

```
Cam-41 (D41) → Cam-16 (D16) → Cam-52 (D52) → Cam-05 (CH05) → Cam-23 (CH23)
```

This topology is enforced by the sidecar's MTMC module — global IDs are only
merged along valid next-camera transitions.

---

## 3. Frontend Integration Reference

### TypeScript types

See `frontend/src/types/tracking.ts` for `GlobalTrack`, `Transition`,
`TrackingStats`, `CameraSummary`, and `TrackingWsMessage`.

See `frontend/src/types/ws.ts` for the full `WsMessage` union type.

### API client

See `frontend/src/api/tracking.ts` — functions: `getTracks`, `getTransitions`,
`getStats`, `getCameras`, `getConfig`.

### State management

See `frontend/src/stores/trackingStore.ts` — Zustand store with
`applyTrackingUpdate` and `applyTrackingMatch` for WebSocket integration.

### WebSocket hook

See `frontend/src/hooks/useWebSocket.ts` — handles connection, reconnection,
ping/pong, and dispatches `tracking_update` / `tracking_match` to the store.

### Camera feed component

`frontend/src/components/camera/CameraFeedMini.tsx` — polls
`/stream/<cameraId>/frame` for annotated JPEG snapshots from the DeepStream
sidecar. Used in the tracking detail popup.

### Key pages

- `frontend/src/pages/TrackingPage.tsx` — vehicle tracking table, journey
  timeline, camera-pair travel times, detail overlay with map + live feed
- `frontend/src/pages/MapPage.tsx` — interactive site map with camera positions
  and active vehicle indicators

---

## 4. Starting the Full Stack

```bash
# Start all services (Redis, go2rtc, IRIS backend, DeepStream sidecar, MinIO)
docker compose up -d

# Start the camera feed publisher (in a separate terminal)
python3 test_streamer/stream_recordings.py

# Access:
#   Frontend:  http://localhost:8000
#   go2rtc UI: http://localhost:1984
#   API:       http://localhost:8000/api/tracking/stats
```

### Without DeepStream (frontend-only dev)

If you only need the frontend with mock data (no GPU / no sidecar):

1. Start go2rtc + stream publisher for camera feeds
2. The IRIS backend will start but `TrackingService` will have no data (no
   sidecar publishing to Redis)
3. You can mock the REST responses or use the existing frontend store defaults

---

## 5. Environment Notes

- **Videos**: Real factory footage from `products/el-mohandes/videos/` (5 enter/intersection
  clips + 8 car-tracking crop clips, ~3.5 GB total)
- **Codec**: Videos are H.264 High-profile with B-frames; go2rtc lazily
  transcodes to browser-safe H.264 Main for WebRTC viewers
- **Sidecar**: Currently uses `file://` URIs directly (bypasses go2rtc RTSP
  to avoid HEVC corruption); the go2rtc stream is for browser preview only
- **Redis**: All tracking data flows through Redis streams
  (`iris:meta:<cam-id>`, `iris:jpeg:<cam-id>`, `iris:status:<cam-id>`)
