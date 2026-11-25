# IRIS DeepStream — Benchmarking & Live-Stream Tooling

Reproducible benchmarking for the DeepStream sidecar enhancements, plus a tool
to run any number of local videos as a live multi-stream wall on the dashboard.

Three suites, one repo:

| Suite | Tool | What it proves |
|---|---|---|
| **A. Before/after enhancements** | `benchmarks/run_bench.py` | The DeepStream changes' delta (boot time, GPU util/mem, probe busy %, rotation, face recognition) measured on the *same* machine, *same* instrumentation, pre-change (`32aaf69`) vs post-change (HEAD, INT8). |
| **B. Stream capacity** | `benchmarks/stream_capacity.py` | How many streams this GPU can serve before saturating, across N sources × source fps (25/10/5). |
| **C. Live wall** | `scripts/live_streams.py` | Run N local videos (plus the 2 LAN RTSP cams) through the pipeline and watch every tile live on `http://localhost:8000`. |

`benchmarks/RESULTS.md` documents the numbers measured on the dev laptop
(RTX-class GPU) and the two real-world bugs the live wall exposed (browser
connection cap → black tiles; mpeg4 AVIs not looping on NVDEC). Your numbers
will differ — that's the point: run the same scripts on another GPU and
compare.

---

## 1. Prerequisites

- **NVIDIA GPU + driver**, Docker with the **NVIDIA Container Toolkit**
  (the sidecar image runs DeepStream 9.0 + TensorRT; `nvidia-smi` must work on
  the host — the benchmarks sample it directly).
- **ffmpeg** on the host (used to build the fps fixtures and to transcode
  uploads to h264).
- **Full git history** — suite A materializes the pre-enhancement code from
  commit `32aaf69`, so a shallow clone will not work:

  ```bash
  git clone --no-single-branch <repo-url>
  ```

- **Test videos** — `data/uploads/` is gitignored. Drop your camera/street
  videos there (any codec; see §5 to transcode non-h264). The capacity bench
  defaults to `data/uploads/attendance/camera_5_2023-06-08-12-29-08.avi`; use `--source`
  to point at any other video.
- **Start the stack** (first boot builds the TensorRT engines — can take
  several minutes):

  ```bash
  docker compose up -d --build
  ```

  The sidecar container is *auto-detected* by both tools (the compose project
  name follows the checkout directory), so the exact container name doesn't
  matter.

---

## 2. Suite A — before/after enhancements benchmark

Measures the same instrumented hot paths on the pre-enhancement code
(`32aaf69` + FP32) vs the enhanced code (HEAD + INT8):

```bash
# 1. baseline: pre-enhancement code + FP32 pgie config
python3 benchmarks/run_bench.py before

# 2. enhanced: HEAD code + INT8 pgie config
python3 benchmarks/run_bench.py after

# 3. (optional) isolates INT8's delta: HEAD code + FP32
python3 benchmarks/run_bench.py after_fp32

# 4. print the comparison table from benchmarks/results/*.json
python3 benchmarks/run_bench.py compare
```

Any run mode accepts `--telemetry` to also enable the per-stage `[telemetry]`
timing (see Suite B below) for that run; the stage busy%/avg table is folded
into `results/<mode>.json` and printed by `compare`. `sidecar.yaml` is restored
after the run. Note it only produces rows on code versions that carry the
instrumentation (HEAD) — `before` (32aaf69) will show no `[telemetry]` lines.

Each run (≈2–4 min) does this automatically: restores/checks out the target
code into the working tree → injects `[bench]` timing wrappers
(`benchmarks/instrument.py`) → restarts the sidecar → waits for the first
`fps` line anchored on a fresh `sidecar starting` boot → samples GPU/VRAM for
90 s → writes `benchmarks/results/<mode>.json`.

Notes:

- **Commit or stash your working changes first.** The tool mutates the working
  tree on purpose (the sidecar container bind-mounts the repo), and every mode
  starts with `git restore` of the swapped files — uncommitted edits to
  the engine under `perception/engine/`, `perception/modules/face_recognition/recognizer.py` or the pgie config would
  be **silently discarded**. Running `before` then `after` is safe on a clean
  tree. After the *last* run, leave the tree clean:

  ```bash
  git restore perception/engine perception/modules/face_recognition/recognizer.py \
              perception/models/DeepStream-Yolo/config_infer_primary_yolo11.txt
  ```

- For meaningful face-recognition numbers, register a couple of gallery faces
  from the live frames first (run inside the sidecar container):

  ```bash
  docker exec <sidecar-container> python3 \
      /workspace/iris/benchmarks/bench_gallery.py
  ```

  `benchmarks/bench_face_crop.py` is the standalone micro-benchmark of the
  recognition-only `_embed_crop` path (Change 4) and also verifies the two
  paths produce agreeing embeddings.

---

## 3. Suite B — stream-capacity sweep

Sweeps N sources × source frame rate and reports GPU util/VRAM, per-stream
fps, and saturation warnings per cell, so you can read off "comfortable max
streams @ 25/10/5 fps" for *this* GPU.

```bash
# 0. build the fps fixtures once (fps5/10/25 h264 versions of a source video)
#    --source defaults to data/uploads/attendance/camera_5_2023-06-08-12-29-08.avi
python3 benchmarks/stream_capacity.py fixtures --source /path/to/any-video.mp4

# 1. single cell (quick sanity check)
python3 benchmarks/stream_capacity.py run --n 6 --fps 10

# 2. full grid — n ∈ {1,2,4,6,8,10,12} × fps ∈ {25,10,5}  (~1 h on a laptop)
python3 benchmarks/stream_capacity.py sweep

# 3. print the accumulated grid
python3 benchmarks/stream_capacity.py summary

# 4. restore the original sidecar.yaml + restart the sidecar
python3 benchmarks/stream_capacity.py restore
```

Each cell: rewrites `perception/pipeline/sidecar.yaml` (N copies of a `<fps>-fps` h264
fixture, `stream_fps` matched) → restarts the sidecar → waits for the first
fps line → samples GPU/CPU for 30 s → writes
`benchmarks/results/capacity_<n>s_<fps>fps.json` and appends a row to
`benchmarks/results/capacity_summary.jsonl`.

#### Per-stage telemetry (`--telemetry`) — find where the pipeline spends its time

Capacity sweeps answer *how many streams fit*. The `--telemetry` flag answers
*where the per-stream budget goes*. It turns on the sidecar's per-stage timing
(`telemetry_enabled: true`, `perception/engine/telemetry.py`) for each cell, collects the
`[telemetry]` log lines into the cell JSON, and prints a per-stage breakdown
right after the cell:

```bash
python3 benchmarks/stream_capacity.py run --n 6 --fps 10 --telemetry
# later, on any cell JSON:
python3 benchmarks/stream_capacity.py telemetry benchmarks/results/capacity_6s_10fps.json
```

Each flush window emits one line per stage with call count, total ms, **busy %**
(share of that stage's thread wall-clock — the number to compare across
stages), avg/max per call, and the thread(s) it ran on, plus gauge samples for
backpressure (publisher/recognition/segment-uploader queue depths, per-source
mux-queue fill):

```
[telemetry] probe  n=1250 total=9123.4ms busy=24.1% avg=7.30ms max=48.2ms thread=streaming window=38.0s
[telemetry] publish n=310  total=1420.8ms busy=63.0% avg=4.58ms max=31.9ms thread=frame-publisher-cam-1 window=38.0s
[telemetry] gpu.infer.pgie n=120 total=91200.0ms busy=95.2% avg=760.0ms max=890.0ms thread=gpu window=38.0s
[telemetry] gauge publisher_pending=2
```

Stages instrumented:

| Stage | Thread | What it measures |
|---|---|---|
| `probe` (+ `probe.surface_copy` / `.frame_objects` / `.dets`) | streaming | the GStreamer pad callback: batch meta parse, zones/analytics, detection/box/face collection, the GPU→CPU surface copy, publisher submit |
| `publish` (+ `publish.draw` / `.redis`) | per-stream publisher workers | box drawing + JPEG encode (GPU nvjpegenc or cv2) and the Redis meta/jpeg writes |
| `recognizer` (+ `.embed` / `.stabilize` / `.publish`) | recognition | InsightFace crop embedding, name stabilization, result publish |
| `upload` | segment uploader | MinIO/local segment upload + index per rotation |
| `rotation` | recording streaming | the splitmuxsink `format-location-full` callback |
| `gpu.infer.pgie` / `gpu.infer.sgie_face` | (pad probes) | **nvinfer** inference time per batch — the element-level counterpart of the Python stages. busy% is the element's processing time as a share of the wall-clock window (its utilization) |
| `gpu.decode.<cam>` | (pad probes) | the **decoder** inside each source bin (nvv4l2decoder / avdec_h264-265), per frame — decode latency + utilization per source |
| `gpu.jpeg` | (pad probes) | **nvjpegenc** in the publisher's GPU-JPEG mini-pipeline, per encode — the GPU half of `publish.draw` |

GPU-element rows (`gpu.*`) come from sink/src BUFFER pad probes attached around
the element when telemetry is enabled (they are never installed when it is
off, so the live pipeline keeps its exact zero-overhead path); the probe
callbacks just feed `telemetry.gpu_enter/gpu_exit`, and the row prints with
`thread=gpu`. They measure the element's **processing latency**, not its GPU
SM utilization — total GPU load still comes from the host-sampled `nvidia-smi`
util, and standalone element profiling with `trtexec`/`nvidia-smi` remains
available (RESULTS.md R7). For nvinfer the src pad emits one buffer per
inference batch, so the delta is batch latency; decoder rows are per frame.
The analysis-branch queue fill is tracked as a gauge, which is where
backpressure against the mux shows up.

Notes:

- The yaml is **restored automatically** after `run`/`sweep` (even on
  failure), and the container is restarted on the restored config at the end
  of a sweep.
- **Fresh machine, fresh sweep:** the committed reference rows live in
  `benchmarks/results/capacity_*.json` + `capacity_summary.jsonl`. Delete
  those files before a new machine's sweep, or `summary` will print a grid
  mixing both machines' rows.
- On consumer GPUs, NVENC caps concurrent encode sessions and can prevent the
  pipeline from starting past ~12–15 streams while `record_enabled: true`.
  For capacity sweeps on such GPUs, set `record_enabled: false` in
  `perception/pipeline/sidecar.yaml` first (or run the wall via
  `scripts/live_streams.py run --no-record`, which sets it for you).
- `benchmarks/fixtures/` and the JSONs are written next to the scripts; the
  fixtures dir is gitignored (generated), the results are committed for
  reference.

---

## 4. Suite C — live multi-stream wall

Run any number of local videos (plus the two LAN RTSP cams by default)
through the pipeline and watch them on the dashboard.

```bash
# what can run right now (codec/resolution/fps/duration + skipped files)
python3 scripts/live_streams.py list

# non-h264 uploads (mpeg4 AVIs) play once and freeze: they software-decode and
# nvurisrcbin's file-loop never fires on that path. Transcode them to h264
# twins (<stem>_h264.mp4) so NVDEC decodes them and they loop like real CCTV.
python3 scripts/live_streams.py transcode

# run the full wall: every usable upload, no NVENC recording branches,
# analyze only ~5 fps per stream while the streams still play realtime
python3 scripts/live_streams.py run --all --no-record --process-fps 5

# or a subset / explicit selection
python3 scripts/live_streams.py run --n 6 --no-record --process-fps 5
python3 scripts/live_streams.py run --ids cam-2,cam-4 --no-record

# what is live right now / back to the pre-tool config
python3 scripts/live_streams.py status
python3 scripts/live_streams.py restore
```

Then open **http://localhost:8000** — every stream is a live tile.

Notes:

- Grid tiles poll the new `GET /stream/{id}/frame` snapshot endpoint instead
  of holding one MJPEG connection each. This matters: browsers cap ~6
  concurrent HTTP/1.1 connections per origin, so a >6-stream wall of
  permanent MJPEG streams renders most tiles black (a real bug the wall
  exposed — fixed in `app/server.py` + `StreamTile.tsx`). The frontend fix
  ships in the `iris` image, so `docker compose build iris && docker compose up -d iris`
  must be run at least once after pulling these changes.
- `--process-fps 5` throttles inference/analysis to ~5 fps per stream
  (DeepStream `nvinfer interval` + probe gating) while sources still decode
  and play at full realtime rate — the same load profile as the 5-fps
  capacity fixtures.
- `--no-record` sets `record_enabled: false` to dodge the consumer-GPU NVENC
  session cap at large stream counts.
- The sidecar config it writes is machine-local; `restore` always brings back
  the committed single-source default.

---

## 5. Results & reporting

- Raw per-run metrics: `benchmarks/results/before.json`, `after.json`,
  `after_fp32.json`, `capacity_*.json`, `capacity_summary.jsonl`.
- Written-up results + capacity matrix + bug notes: `benchmarks/RESULTS.md`.
- PR write-up of every enhancement with measured deltas: `report.md` (repo
  root).

To publish a fresh machine's numbers, copy the new JSONs into
`benchmarks/results/` and update `benchmarks/RESULTS.md` / `report.md` with
the machine's GPU model and the new matrix — the `compare` table and the
`summary` grid regenerate from the JSONs automatically.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Most tiles black on the dashboard | Stale `iris` image — rebuild: `docker compose build iris && docker compose up -d iris` |
| Streams freeze at end of video, don't loop | Source is mpeg4/AVI → software decode → `file-loop` never fires. `python3 scripts/live_streams.py transcode` |
| Long first boot per mode | INT8 engine build + calibration on a fresh clone; cached afterwards under `perception/models/DeepStream-Yolo/` |
| Pipeline won't start past ~12 streams | Consumer-GPU NVENC session cap while recording. `--no-record` / `record_enabled: false` |
| `run_bench.py` says container not found | Stack not up: `docker compose up -d`; the tool auto-detects the container, no rename needed |
| Engine rebuild on every boot | Calibration cache not written — check `int8_calib_img_path` in `perception/pipeline/sidecar.yaml` points at a real image list |
