# shelf_void

Empty shelf space ("voids") on supermarket shelves: a void detector on the full frame, a heat map over time per camera,
and `shelf_empty` / `shelf_restocked` alerts. Ported from the standalone pipeline `~/IRIS/shelf-monitoring` (commit
02538da): `core/` is its heat map and alerter, unchanged except that the boxes now come from nvinfer.

The model has **no licence** (trained with Ultralytics, AGPL-3.0; the source `.pt` is a pickle): see
`perception/models/DeepStream-Yolo-Shelf/README.md`. It is versioned with DVC (`just data-pull models-shelf`).

## Run the demo

The model and the demo clip are versioned with DVC (DATA.md), about 34 MB. On a machine with an NVIDIA GPU, Docker
(with the NVIDIA container toolkit) and `just`:

```bash
just data-pull models-shelf samples-shelf   # the model and the demo clip
just shelf-check                     # the model's SHA-256, the video, docker, GPU
just shelf-build                     # once: the images (or name existing ones, below)
just shelf-live-sim                  # demo.mp4 -> camera cam-shelf-demo
just shelf-live-status               # containers, streamer, last fps and shelf alert lines
just shelf-live-down                 # stop and remove it all (keeps scratch/shelf-live/)
```

The sidecar runs the checkout's own code (the repo is mounted at `/workspace/iris`), so any sidecar image built from
this repo's `perception/pipeline/Dockerfile` (DeepStream 9.0 with the YOLO parser) can be used instead of building one:
`SHELF_SIDECAR_IMAGE=<image> SHELF_IRIS_IMAGE=<app image> just shelf-live-sim` (or the same lines in `.env`).
It mirrors `just ppe-live-sim` (perception/modules/ppe/README.md, "Run the demo"): the video loops at its real speed
into the stack's own go2rtc as a simulated IP camera, and the stack runs as compose project `shelflive` on its own
ports (`SHELF_IRIS_PORT` 47000, `SHELF_REDIS_PORT` 47379, go2rtc 47554/47984, MinIO 47900/47901, telemetry 47101),
beside any other stack. Open `http://<host>:47000/stream/cam-shelf-demo` (MJPEG) and
`http://<host>:47000/#/iris/dashboard` (alerts).

## Switching it on

Off unless a pipeline file lists it. Either `SIDECAR_CONFIG=configs/presets/shelf/sidecar.yaml just up`
(the shared pipeline plus this module), or the one line `shelf_void: {}` under `modules:` in a product's sidecar yaml.
A module runs on every camera of its pipeline; a camera-specific region is a `roi` entry (below).

## The pipeline

```
nvstreammux (1080x720) -> pgie YOLO11x (persons) -> nvtracker -> pgie_shelf: void detector on the full frame
                                                              -> shelf_void: heat map per camera, alerts
```

- **After the tracker, on the full frame** (`role="secondary"` in `gies()`, `process-mode=1` in
  `config_infer_shelf.txt`, gie id 5, the mux's batch): nvtracker deletes every object under its own floor (0.19);
  the void detector is useful down to 0.02, so its boxes must never be tracked.
- **1280 input**: the ONNX is exported at 1280x1280; far voids were only found at 1280 in the standalone runs.
- **Low floor on purpose**: `pre-cluster-threshold=0.02`, NMS IoU 0.3, as in the standalone pipeline. The model is
  low-confidence on angled shelf views (top scores about 0.08 on the demo). One frame of these boxes is noise; the heat
  map is the filter.
- **Model-agnostic**: the module reads the label `empty` (setting `label`) from the GIE `pgie_shelf`, never a class id.
  Another model is a new ONNX, labels file and config (the model README, "Swapping the model").

## Heat map and alerts (core/)

Per camera, at most `eval_fps` (30) times a second, in `on_frame` (every frame, ahead of the publish gate):

1. every `empty` box of at least `min_area` px^2 is painted into a mask (inside the camera's `roi`, if any);
2. `heat = decay * heat + (1 - decay) * mask`; pixels above `show_thresh` are stable;
3. a morphological close with a `close_kernel` (width, height) rectangle joins fragments of a void along the shelf;
4. each connected stable region is a void; the alerter matches voids across updates by IoU (`match_iou`).

A void seen for `confirm_s` (2 s) raises **`shelf_empty`**; once it has been gone for `clear_s` (3 s), e.g. after
restocking, **`shelf_restocked`** closes it. A place that becomes empty again gets a new alert id. Times are the
engine's monotonic clock (pts restarts when a file loops).

| field | `shelf_empty` | `shelf_restocked` |
|---|---|---|
| `kind` (the event's tag) | `shelf_empty` | `shelf_restocked` |
| `severity` / `category` | `warning` / `operations` | `info` / `operations` |
| `message` | `Empty shelf space on cam-1 at (x1,y1)-(x2,y2)` | `Shelf restocked on cam-1 at (...) (empty for 12 s)` |
| `pts` | the frame the alert was raised on; the clip is centred on it (`request_clip`) | the frame it cleared on |
| `alert_id`, `bbox` (mux px), `score` (mean heat) | yes | yes, the same `alert_id` |
| `empty_for_s` | | yes |

Severity and category use the platform's closed sets (`app/severity.py`): PPE and fall use `warning` / `safety`;
an empty shelf is a store-operations matter, so `operations`, and a restock is `info`. The app reads `kind`,
`severity`, `category`, `message` and `pts`; the other fields stay on the `iris:alert:<camera>` entry.

## stability: tracker (the accepted shelf-demo logic)

`stability: tracker` (the shelf preset's choice) replaces the heat map and alerter with the accepted shelf-demo's
per-region logic. `core/tracker.py`, `core/person_gate.py` and `core/situations.py` are **vendored unchanged** from
shelf-demo `service/` (each file's header names the commit; `tests/test_shelf_void_tracker_mode.py` checks the copy
against that commit when a shelf-demo checkout is present). `stability.py` wires them like the demo's CameraWorker,
once per named region, every frame:

1. **Regions**: `roi.<camera>` as the demo's `roi.json` (`{size: [w, h], regions: [polygon, ...]}`, named R1, R2, ...;
   or `regions: {name: polygon}`), scaled to the mux. A void box belongs to the first region containing its centre.
2. **Confidence filter**: boxes below `tracker.conf` (0.08, the demo's `--conf`) never count.
3. **Person pause**: the shared pgie's `person` boxes (conf >= 0.35, grown 15% per side) pause every region they touch
   until it has been person-free for `person.resume_s` (0.4 s = the demo's 10 frames at 25 fps). A paused region keeps
   its confirmed spots and drops its candidates.
4. **StableTracker** per region: a spot is confirmed once it stayed in place (IoU >= 0.3) for `tracker.confirm_s` (2 s)
   and was seen in >= 50% of the frames; it is forgotten after `tracker.max_miss_s` (1 s) unseen. Each frame counts as
   1/fps s (fps measured from the pts, or `tracker.fps`).
5. **Situation** per region, one alert id from raise to clear:

| intent | alert | when |
|---|---|---|
| raised, reason `new` | `shelf_empty` (warning, with a clip) | the region's first confirmed spot |
| raised, reason `more` | `shelf_empty`, same `alert_id` | more spots than last reported |
| raised, reason `periodic` | `shelf_empty`, same `alert_id` | every `situation.reraise_s` (300 s) while open |
| cleared, reason `gone` | `shelf_restocked` (info) | `situation.clear_s` (30 s) of person-free time without spots |

Fields on the `iris:alert:<camera>` entry, beyond `kind`/`severity`/`category`/`message`/`pts`: `alert_id`, `region`,
`reason`, `count` (spots), `occurred_at` (UTC, ISO with `Z`), and on `shelf_empty` `bbox` (best spot, mux px),
`score` (best detector score, latest detection), `stability` (best hit ratio) and `spots` (JSON: every spot's
normalised `bbox`, `detectorScore`, `stability`); on `shelf_restocked` `empty_for_s` and `clear_s`. The iris-platform bridge
(`products/shelf_iris_bridge`) turns these into IRIS events.

### Parity with the accepted shelf-demo (test.mp4, 2026-11-12)

The shelf preset (`stability: tracker`, 640, R1/R2 from the demo's `roi.json`; `configs/presets/shelf/live-test.yaml`)
on the demo's `test.mp4` (1920x1080, 25 fps, 3,000 frames), played once in real time over RTSP from frame 0, against
the demo's baseline (`void_roi.py`, default settings):

| | shelf-demo baseline | DeepStream shelf preset |
|---|---|---|
| R1 | confirmed at frame 1177 (47.08 s), open to the end | `shelf_empty` at source frame 1041 / 1043 (two runs) ~= video frame 1080 (~43.2 s), open to the end, no clear |
| R1 box (1920x1080 px) | (1331,712)-(1385,823) | (1332,710)-(1387,825) (mux box x 1920/1080, x 1.5) |
| R1 detector score | 0.109-0.15 | 0.30-0.37 |
| R2 | nothing | nothing |
| frames with a region paused | 1,352 of 3,000 | 1,036 of 2,961 |

Why R1 comes ~3.9 s earlier (a per-frame debug run, frames 900-1200):

- **Timeline:** the sidecar got 2,961 frames; ~39 were lost while the RTSP session started (drop-on-latency), so source
  frame N is video frame ~N+39. Its pause/resume edges sit 33-45 frames before the baseline's all along.
- **Persons (YOLO11x via nvtracker on the squeezed mux frame vs yolo11s on the full frame):** both see the same
  shoppers, but a person half out of frame at the bottom of R1 (conf 0.6 at source frame 981) stops being reported
  ~0.5 s sooner. R1's first person-free window is then ~2.0 s (just long enough to confirm), the baseline's 1.6 s
  (40 frames, too short; the next pause drops its candidate, so it confirms only in the second window, 1128 -> 1177).
- **Mux scaling and input:** 1920x1080 is squeezed to 1080x720 and letterboxed to 640x640 (the demo letterboxes the
  1920x1080 frame), which moves the void scores up (same box); persons at the frame edge differ most.
- **Sampling:** both run every frame (the demo service samples at 5 fps; with its time-based tracker the confirm
  time is the same).

## Overlay and meta

- The detector's own boxes become kind `shelf_void_raw`: never drawn, never published (whatever `relevant_classes`
  says).
- Every void with an open alert is drawn by the module (`out.draw`) in violet `(211, 0, 148)` with the caption
  `empty`, and listed on meta frames as the `shelf_void` field of `iris:meta`: `[{alert_id, bbox}]`.

## Settings

```yaml
modules:
  shelf_void:
    config: /workspace/iris/perception/models/DeepStream-Yolo-Shelf/config_infer_shelf.txt
    interval: 0              # MUST stay 0: a skipped batch would cool the heat map as if the shelf were full
    label: empty
    eval_fps: 30
    min_area: 400            # mux px^2
    decay: 0.85
    show_thresh: 0.5
    close_kernel: [40, 9]    # mux px
    roi: {}                  # {camera id: [[x, y], ...]} in mux px
    confirm_s: 2.0
    clear_s: 3.0
    match_iou: 0.3
    stability: heatmap       # or tracker (above); then:
    tracker: {conf: 0.08, confirm_s: 2.0, max_miss_s: 1.0, min_iou: 0.3, min_hit_ratio: 0.5, fps: 0}
    person: {enabled: true, conf: 0.35, pad: 0.15, resume_s: 0.4}
    situation: {clear_s: 30.0, reraise_s: 300.0}
```

The defaults are the standalone pipeline's, which was tuned on 1280x720 frames at 30 fps. In IRIS the boxes are in
mux pixels (1080x720 by default, so a 1280x720 camera is squeezed to 0.84 of its width) and `decay` is per update,
so a camera below 30 fps warms and cools more slowly. See "Tuning". The shelf preset
(`configs/presets/shelf/sidecar.yaml`) sets `min_area: 800` and `match_iou: 0.2`: see "Results on video".

## Tuning

| Problem | Change |
|---|---|
| false boxes | raise `show_thresh` (0.6) or `min_area` (800) |
| a real void missed, or its box too small | lower `show_thresh` (0.35) |
| slow to appear | lower `decay` (0.7) |
| two voids merged | a narrower `close_kernel` |
| voids outside the shelf | a `roi` polygon for that camera |

## Results on video (2026-11-02)

`demo.mp4` (supermarket aisle, 1280x720, 30 fps, 21 s, looped) through `just shelf-live-sim`, 1080x720 mux,
shelf engine FP16 at 1280, YOLO11x pgie FP32 beside it, on a GPU other stacks kept at 100%: 24-34 fps
(mean 27.8), `pgie_shelf` 11.5 ms per call on average.

| | standalone (1280x720 frames) | IRIS, module defaults (`min_area` 400) | IRIS, shelf preset (`min_area` 800, `match_iou` 0.2) |
|---|---|---|---|
| void at the Coke shelf, left | RAISED (501,365)-(552,420) | RAISED (425,362)-(471,422) | RAISED (425,361)-(473,422) |
| void top right | RAISED (991,225)-(1124,285) | RAISED (838,227)-(948,297) | RAISED (838,227)-(948,297), open the whole run |
| narrow gap (926,363)-(940,394) | RAISED, CLEARED | not seen | dropped (372 mux px^2) |
| two dark gaps between bottles, top left | -- | RAISED (105,103)-(126,132), (38,99)-(55,128) | dropped (609, 493 mux px^2) |

Mux boxes are the standalone ones x 0.84 in x: the two real voids land within a few pixels.

**`match_iou` 0.2 in the shelf preset.** With the default 0.3 the left void re-alerted every loop or two: its stable
region switches between the void alone (about 48x61) and the void joined with the heat next to it (about 106x92,
(416,330)-(522,422)), and the two boxes overlap at IoU 0.30, right at the threshold, so the alerter sometimes took the
larger box for a new place (a `shelf_empty`, then a false `shelf_restocked` once it shrank back: 4 such pairs in 7
loops). With 0.2 the same box keeps its alert: over 9 loops exactly the two real voids raised (both with clips), and
nothing cleared.

## Testing

```bash
python3 -m pytest tests/test_shelf_void_*.py -q
```

`test_shelf_void_core.py` holds the 5 standalone tests, ported with only the import changed.
