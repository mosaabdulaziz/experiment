# Bottleneck report & tackle plan (2026-08-12)

Measured with the new per-stage telemetry (`stream_capacity.py --telemetry`) on
the current branch — see `benchmarks/RESULTS.md` → "Current-state reference —
clever sweep with per-stage telemetry" for the full tables and cell JSONs.

## Measured state (RTX 5050 laptop, 25 fps fixtures, no record)

| analysis rate | solid | wall (GPU 97–99 %) |
|---|---|---|
| af=5 | 24 streams (66 % GPU) | **32** |
| af=10 | 16 streams (77 %) | **20** |
| af=25 (full rate) | 8 streams (80 %) | **12** |

Per-stage (the numbers that say *where* the budget goes):

- **Streaming thread (`probe`) is 25–37 % busy and ~95 % of that is
  `probe.surface_copy`** (2.1–2.4 ms/frame GPU→CPU memcpy of every analyzed
  frame of every stream, watched or not). `probe.dets`/`frame_objects` are
  noise (~0.1–0.5 %). The streaming thread is NOT the wall — but the copy is
  the single largest removable Python-side cost.
- **Frame publisher is the only Python-side backpressure point**: drops start
  at ~120–160 aggregate published fps (`n=16 af=10`: 363 drops, `n=8 af=25`:
  995), and `publish.draw` (shared nvjpegenc) balloons from ~5 ms/call at low N
  to 33→117 ms/call at the walls — encode contention under GPU saturation.
  `publish.redis` is fine (~1 ms/call).
- **mux queues never fill** (all gauges 0) — the `analysis_queue_buffers`
  mechanism adds no backpressure anywhere measured.
- **Recognition costs ~nothing with an empty gallery** — but `sgie_face` still
  runs a full FP32 detection pass per analyzed frame whose result is only ever
  "unknown" (overhead audit #1 in RESULTS.md: it costs MORE than the pgie).
- **GPU is the binding wall at every rate** — decode + inference + NVJPEG share
  one GPU. This matches every prior laptop verdict; the new telemetry proves
  the Python side still has headroom when the GPU is quiet.

> **GPU-element attribution (2026-08-13, pad-probe rows):** the wall cells were
> re-run with the `gpu.*` rows live, and the GPU wall is now attributed —
> **decode is the dominant GPU consumer, not inference**: every decoder runs
> ~40–50 % busy and the sum scales linearly with N (Σ 330 % at n=8 → 1488 % at
> n=32), while `gpu.infer.pgie` is only 18–39 % busy even at full-rate af=25.
> `gpu.jpeg` (NVJPEG) tracks `publish.draw` (21 % → 64 %). So the binding GPU
> work is the *decode* side: the levers in item 4 that actually move the wall
> are decode-resolution / per-source decode rate, not inference. Full table in
> RESULTS.md → "GPU-element attribution"

## Implemented (2026-08-13) — the structural preview fix, verified

The inference-only preview mode is built and A/B-measured (RESULTS.md →
"Inference-only preview mode"): with `pipeline_preview: false` the sidecar
stops surface-copying, box-drawing and NVJPEG-encoding preview frames entirely
— the website plays each source's **raw stream** over WebRTC (go2rtc WHEP,
on-demand H.264 transcode for browsers) and overlays detections from
`iris:meta`. The af=5 sweep wall moved **n=32 → n=64–72** on the workstation
GPU (96 % → 61 % at n=48) with `probe.surface_copy`/`publish.draw`/`gpu.jpeg`
all gone and 0 publisher drops even at the new wall. The sgie auto-toggle was
also fixed along the way (see item 2 — `interval` does not gate secondary
GIEs; `infer-on-class-ids` does). This collapses items 1 and 3 and most of
item 4's *preview-side* levers; item 4's decode-side levers (R5 960×540) remain
open for the inference-only wall.

## Tackle plan (priority order)

Each item = measured problem → approach → expected impact → how the new
telemetry verifies it. Items 1–3 are config/contained changes; 4 is the
structural lever.

### 1. Gate the probe surface copy on watched streams  *(quick win)*
- **Problem:** `probe.surface_copy` is ~95 % of the streaming thread's 25–37 %
  busy — every analyzed frame of every source gets a full GPU→CPU copy even
  when nothing consumes the pixels (the meta/dets feed does not need them; only
  the preview encode and face recognition do).
- **Approach:** extend the existing `ViewerGate` (already used for
  `viewer_gated_preview`) to the copy: skip `_map_frame_bgr` (and the whole
  publish job) when the frame is not needed for a *watched* stream's preview
  and carries no faces for recognition. Cache the last annotated frame so an
  unwatched→watched transition has something to show.
- **Impact:** at the observed ~10 % watched ratio (16/192 streams), most of the
  streaming-thread busy and the PCIe/CPU memcpy traffic disappear; on the
  probe-bound workstation (the M1 A/B machine) this raises the aggregate
  probe wall (620 → ~1100 → ~1600 fps project targets).
- **Verify:** telemetry `probe.surface_copy` busy% collapses toward ~0 for
  unwatched streams while `probe` busy% drops; same cells before/after.

### 2. Auto-disable the face SGIE when no gallery is registered  *(quick win — mechanism confirmed in DeepStream source)*
- **Problem:** the 2026-08-13 GPU rows show `gpu.infer.sgie_face` at 20–51 %
  busy — *consistently more than the pgie* (18–39 %), because it's FP32 vs the
  pgie's INT8 — while its only output is `"unknown"`: the live gallery is
  empty (`HLEN iris:face:gallery` = 0). Every analyzed frame of every source
  pays for a face-detection pass whose result is discarded.
- **Feasibility (verified in the DeepStream source, helper-repos/DeepStream):**
  nvinfer's `interval` property is the runtime toggle — `GST_PARAM_MUTABLE_READY`
  (`gstnvinfer.cpp` property spec), and the skip check is
  `skip_batch = (interval_counter++ % (interval + 1) > 0)` with the skipped
  path returning `GST_FLOW_OK` immediately (`gstnvinfer.cpp`), so the buffer
  flows through untouched with the pgie/tracker metadata (no face sub-objects)
  and **zero inference / zero GPU**. Setting `interval` to a large value (e.g.
  1000000) = effectively off; `0` = every batch (today's config). There is also
  the official event route used by deepstream-app itself:
  `gst_nvevent_infer_interval_update(stream_id, interval)` ("nv-infer-interval-update"
  custom upstream event, `gst-nvdscustomevent.c`, handled at `gstnvinfer.cpp`
  line ~852). From Python the property route (`set_property("interval", n)`)
  is simplest and equally valid — same unlocked guint write the event path does.
- **Design — auto-toggle on gallery state:**
  1. `build_pipeline` returns the `sgie_face` element handle (it's a local today).
  2. Rule: **sgie runs when the gallery is non-empty OR `face_capture_enabled`**
     (capture *consumes* sgie face detections to build the gallery — the
     gallery can never fill if the sgie is off while capture is on). Disabled
     only when both are off.
  3. A small monitor (reuse the FaceRecognizer's 30 s gallery-reload cadence)
     checks `HLEN iris:face:gallery` (Redis **hash** — the app registers via
     `r.hset`, `perception/pipeline/register_face.py` seeds the same key) and flips the
     SGIE's enabled state on transitions, with a log line each way.
  4. Boot: initialize the state from the gallery at startup, so an empty
     gallery boots with the sgie off (the current production state).
  5. Config: keep `sgie_face_enabled` as the master "build the element" switch
     and add `sgie_auto_toggle: true` (default) to opt into the toggle;
     `false` keeps today's always-on behavior.
- **The toggle mechanism (corrected 2026-08-13):** the first implementation
  flipped nvinfer's `interval` property — which is a **no-op for secondary
  GIEs**: `interval` only gates the full-frame path
  (`gst_nvinfer_process_full_frame`, gstnvinfer.cpp:1715); process-mode=2
  (sgie) runs `gst_nvinfer_process_objects` (gstnvinfer.cpp:1922), which has
  no interval check. The object path's gate is `infer-on-class-ids`
  (`GST_PARAM_MUTABLE_READY`): `should_infer_object()` skips every object
  whose class isn't in the list, so pointing it at a never-appearing class
  (`"999"`) makes the SGIE queue zero inferences — zero GPU — while buffers
  flow through. Enable restores the element's configured class list.
- **Edge cases:** re-enabling takes effect on the very next batch (class check
  is per-object); `face_enhancement_enabled` only touches crops that exist
  when the sgie runs, so it goes dormant automatically; the property write is
  a string set on a MUTABLE_READY property — benign while PLAYING.
- **Impact:** `gpu.infer.sgie_face` busy (20–51 %) → ~0 when off — a real
  double-digit % of the GPU at the wall, with zero behavior change until the
  first face is registered.
- **Status: IMPLEMENTED (2026-08-13).** (Now `perception/modules/face_detect/idle.py`, setting `idle_when_unused`.) `sgie_gallery.py` (pure
  stdlib, unit-testable off-GPU) holds the rule (`sgie_should_run`) and a
  `SgieGalleryMonitor` that polls `HLEN iris:face:gallery` and flips
  enabled/disabled on the GLib main thread. `build_pipeline` returns the
  `sgie_face` element; config keys `sgie_auto_toggle: true` (default) +
  `sgie_auto_poll_seconds: 10` opt in; `false` keeps the pre-toggle always-on
  behavior. A `sgie_mode` telemetry gauge exposes the current state.
- **Verified live (toggle wiring + rule):** boot `sgie_face auto-disabled at
  boot (gallery=0, capture=False)`; `HSET iris:face:gallery` → `auto-enabled
  (gallery=1)` within one poll; `HDEL` → `auto-disabled (gallery=0)`. RTSP
  cams stayed at ~10 fps, 0 errors. (The earlier interval-based verification
  proved the *property flip* — the 2026-08-13 sweep showed interval does NOT
  stop the sgie's object-path GPU work, which is why the mechanism was
  switched to `infer-on-class-ids`.)
- **Verified in the 2026-08-13 inference-only sweep** (af=5, same laptop):
  with the corrected mechanism, GPU util at n=48 dropped 96 % → 61 % and
  `probe.surface_copy` went to zero (no face detections → no recognition
  crops). NOTE: the `gpu.infer.sgie_face` telemetry row does NOT collapse — it
  measures the element's sink→src buffer transit, dominated by fixed per-batch
  framework overhead rather than the face inference (the inference itself is
  what the GPU util drop shows). The real savings show in GPU util and the
  overall wall: af=5 n=32 → n=64–72 in inference-only mode.

### 3. Frame-publisher throughput  *(config first, then code)*
- **Problem:** `publisher_drops`/`publisher_pending` gauges climb past ~120–160
  aggregate published fps, and `publish.draw` avg jumps 33→117 ms at the walls.
- **Approach (in order of simplicity):**
  1. `viewer_gated_preview: true` in production — the M1 A/B already showed
     this eliminated publisher saturation at n=192 (0 vs 14/window).
  2. Per-stream GPU encoder instances to remove shared-encoder lock contention
     (watch the NVENC/nvjpegenc session budget).
  3. Cap the preview cadence (`stream_fps`) or `jpeg_quality` at high N.
- **Impact:** removes the only Python-side ceiling; on the workstation this is
  the first thing that binds at ≥96 sources.
- **Verify:** `publisher_drops`/`publisher_pending` gauges and `publish.draw`
  avg_ms in the telemetry table.

### 4. Cut GPU work per frame  *(the real wall; M2 levers)*
- **Problem:** GPU 97–99 % at the walls. The 2026-08-13 GPU rows now attribute
  it: **decode dominates** (per-cam decode ~40–50 % busy, Σ scaling linearly
  with N), with NVJPEG second and nvinfer a distant third (18–39 % even at
  full rate). R7 (engine builder) and R8 (decoder drop for files) are closed
  negative results; the levers left are per-frame GPU work — and the data says
  they must be *decode-side* levers to move the wall.
- **Approach:** the documented M2 sub-levers, weighted by the new attribution:
  **R5** 960×540 mux (−44 % pixels through decode/convert/encode — directly
  cuts the dominant decode+NVJPEG work) and **R6** yolo11n (≈⅓ FLOPs — but
  inference is already cheap, so R6's win is secondary), with engine A/B
  measurement. Also consider decode-side caps: lower decode resolution/fps for
  non-analyzed streams (decode runs at source rate regardless of
  `analysis_fps`, so the analysis-rate cap buys no decode headroom).
  `analysis_fps` cap-compensation (target × 1.20) already handles the ~0.84×
  videorate under-delivery.
- **Impact:** raises the wall at every analysis rate on the laptop; on the
  workstation it is the step past the current 870–1075 aggregate-fps proofs
  toward the ~1100/~1600 project targets.
- **Verify:** same cells → GPU util at equal n drops; `gpu.decode.*` per-cam
  busy% and Σ fall; capacity table shifts.

### 5. Re-measure recognition with a real gallery
- **Problem:** this reference had an empty gallery, so `recognizer` cost is
  unmeasured (~0 here). With a loaded gallery, `recognizer.embed` (ArcFace
  `get_feat`, ~5.7 ms/crop) runs per face per analyzed frame.
- **Action:** `benchmarks/bench_gallery.py` + a face-bearing fixture, re-run
  the af=10 cells with `--telemetry`; read `recognizer.embed` busy% and decide
  whether recognition needs its own throttle.

## How each tackle gets verified

All items A/B on the same machine with the same harness:
`stream_capacity.py sweep --fpss 25 --ns <near-wall grid> --analysis-fps <rate>
--telemetry --no-record`, then compare `probe.surface_copy` /
`publish.draw` / `publisher_drops` / GPU util% rows from the cell JSONs.
The `gpu.*` rows (`gpu.decode.<cam>` / `gpu.infer.pgie` / `gpu.infer.sgie_face`
/ `gpu.jpeg`) split the GPU wall into decode vs inference vs NVJPEG; the
2026-08-13 baseline cells in RESULTS.md now carry those rows, so a tackle's A/B
compares against them directly.
