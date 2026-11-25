# test_streamer

Simulates live RTSP cameras from the recorded videos in `data/uploads/`, so the
DeepStream sidecar consumes them over real RTSP — the same path a physical
camera uses, including **Smart Video Record** (`recorder: auto -> rtsp:// => svr`).

This is a **development-only** tool; it is not part of the production stack.

## Why

Playing the recordings as local `file://` sources forced the sidecar down the
NVENC encode branch (record every frame to disk) plus tee contention, starving
the analysis branch to ~3–8 fps. Publishing them as RTSP engages the SVR path,
which has no encode branch, so all streams run at the full analysis fps cap
(~10 fps) with the GPU nearly idle.

## How it fits together

```
data/uploads/*.mp4
   └─ stream_recordings.py  (host ffmpeg, -re -stream_loop -1 -c copy)
        └─ rtsp://go2rtc:8554/raw-cam-N     ← RTSP ingest, ORIGINAL codecs
             ├─ rtsp://go2rtc:8554/raw-cam-N → DeepStream sidecar (originals —
             │                                 best quality for detection + SVR)
             └─ go2rtc (go2rtc.yaml derives the BROWSER-facing cam-N from raw-cam-N)
                  └─ /api/webrtc?src=cam-N (WHEP)      → browser (go2rtc transcodes to H.264
                                                         for viewers that need it)
```

`raw-cam-N` carries the upload's original codecs (`-c copy`): High-profile
H.264 and even HEVC. The **sidecar pulls `raw-cam-N` directly**, so detection
and SVR smart-record always see the originals. The derived `cam-N` streams are
purely for browsers: go2rtc pairs the original with an `ffmpeg:...#video=h264`
variant (High profile, no B-frames) and serves whichever the viewer can play.

Transcodes are **lazy**: go2rtc starts the `ffmpeg:` variant only while a
viewer actually needs it and stops it when they leave — nothing runs when
idle. The price is a ~1-2s ffmpeg spin-up the first time a stream that needs
transcoding (HEVC: cam-6/7/8; High-with-B-frames: cam-1/2/3) is watched;
cam-4/5 (Main, no B-frames) play directly with no transcode at all. The GPU is
never used for transcodes — it is reserved for DeepStream inference.

## Components

| File / service | Purpose |
| --- | --- |
| `go2rtc` compose service | The media server (`alexxit/go2rtc:latest`, **docker image — nothing to install**) with `go2rtc.yaml` (streams: raw placeholders + derived `cam-N`). Listens on 8554 (RTSP), 1984 (API/WHEP), 8555 (WebRTC media). |
| `stream_recordings.py` | Loops each upload through host ffmpeg (`-re -stream_loop -1 -c copy`) and publishes it to go2rtc's RTSP ingest as `rtsp://<host>:8554/raw-<cam-id>`. Restarts dead ffmpeg, exits cleanly on SIGINT/SIGTERM. |
| `perception/pipeline/sidecar.yaml` | Sources point at `rtsp://go2rtc:8554/raw-cam-N` (the compose service name, resolved inside the compose network) — the originals, never the browser transcode. |

## Prerequisites

- **Docker** with the compose stack (nothing else — go2rtc runs as a container).
- **ffmpeg** on the host (used to push the streams).
- The upload files named in `CAM_FILES` present in `data/uploads/`.

## Usage

```bash
# 1. start the go2rtc container (no host install needed)
docker compose up -d go2rtc

# 2. stream every cam in CAM_FILES
python3 test_streamer/stream_recordings.py

# only specific cams
python3 test_streamer/stream_recordings.py --ids cam-1,cam-4

# go2rtc not on localhost (e.g. a remote host)
python3 test_streamer/stream_recordings.py --host 192.168.1.50 --port 8554
```

Verify go2rtc is up: `ffprobe -rtsp_transport tcp -i rtsp://localhost:8554/raw-cam-1 ...`

Run it in a terminal (or `setsid nohup ... &` to detach). **Start the
streamers first, then** restart the sidecar — restarting the pushers while the
sidecar is already running can wedge some sources (stale previews / no fps):

```bash
docker compose restart deepstream-sidecar
```

Check the sidecar logs for `recorder=svr` on every source and steady
`~10 fps` analysis lines.

## Adding / removing cameras

1. Edit `CAM_FILES` in `stream_recordings.py` (cam-id -> file in `data/uploads/`,
   or an absolute path to a file anywhere on the host).
2. Add the raw placeholder + derived stream to `go2rtc.yaml`.
3. Add/remove the matching `rtsp://go2rtc:8554/raw-<cam-id>` entry in
   `perception/pipeline/sidecar.yaml`.
4. Restart go2rtc, the script, and the sidecar.

## Switching back to file playback

Restore `file://` URIs in `perception/pipeline/sidecar.yaml` (the upload names are listed in
`CAM_FILES`), then `docker compose restart deepstream-sidecar`.
