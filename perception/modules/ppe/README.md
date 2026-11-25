# ppe

Helmet and high-visibility vest detection on people, for candidate PPE violations a person reviews. Detector,
rules, alerts and drawing are in place and were run on two recorded videos ("Results on video"); person-level accuracy
on the dataset is below ("Accuracy in IRIS"). The model is **approved for non-commercial use only** and is versioned
with **DVC** (`just data-pull models-ppe`): see `perception/models/DeepStream-Yolo-PPE/README.md`. The plan is in `docs/PPE_MODULE_PLAN.md` (local).

## Run the demo

The PPE model and the two test videos (`data/uploads/ppe/test.mp4`, `test2.mp4`) are versioned with DVC (DATA.md),
about 550 MB.
On a machine with an NVIDIA GPU, Docker (with the NVIDIA container toolkit) and `just`:

```bash
git clone ssh://git@gitlab.urca.ai:2222/Mohamed.Abdulhafez/surveillance-camera-sys.git
cd surveillance-camera-sys
git switch ppe-compliance            # the PPE branch (ppe-demo-assets until its merge request is merged)
just bootstrap                       # once: tools, .env, the default data groups
just data-pull models-ppe samples-ppe   # the model and the two videos (also offered if you forget)
just ppe-check                       # the model's SHA-256, the videos, docker, GPU
just ppe-build                       # once: the docker images (the sidecar's first build is long, see below)
just ppe-live-sim VIDEO=test.mp4     # or VIDEO=test2.mp4
```

`just ppe-build` builds the sidecar image from `perception/pipeline/Dockerfile` (about 31 GB) and the app image. Build
times are estimates, not measured: roughly 30-60 min for the sidecar the first time, most of it downloading the
DeepStream 9.0 base image (~26 GB), 10-20 min when that is already on the machine, and a few minutes for the app.
`ppe-live-sim` never builds: when an image is missing it says so and points here. To use a sidecar image this machine
already has, built from this repo's Dockerfile, set `PPE_SIDECAR_IMAGE` in `.env`.

The containers run as you: the recipes set `PPE_UID`/`PPE_GID` from `id -u`/`id -g`, so engines and data they write
stay yours, removable without sudo. Every other setting has a default and can be changed in `.env` (the PPE block of
`.env.example`): the compose project `PPE_COMPOSE_PROJECT` (`ppelive`), the image `PPE_SIDECAR_IMAGE`
(`ppelive-deepstream-sidecar:latest`), the data dir `PPE_LIVE_DIR` (`scratch/ppe-live`), the videos `PPE_LIVE_VIDEOS`,
and the host ports `PPE_IRIS_PORT` (48000), `PPE_GO2RTC_RTSP_PORT` (48554), `PPE_GO2RTC_API_PORT` (41984),
`PPE_REDIS_PORT` (46379), `PPE_MINIO_API_PORT`/`PPE_MINIO_CONSOLE_PORT` (49000/49001) and
`PPE_SIDECAR_TELEMETRY_PORT` (49101). A second user running the demo on the same machine changes the project, the
ports and the data dir. No WebRTC setting is needed: the video is the app's JPEG preview.

`ppe-live-sim` loops the video at its real speed into the stack's own go2rtc as a simulated IP camera: re-encoded to
H.264 baseline (no B-frames), a keyframe every second, the video's own bitrate; it runs the IRIS pipeline with PPE on
(`configs/presets/ppe/live-<video>.yaml`, preview JPEG quality 75) and the IRIS app, as compose project `ppelive` on
its own ports, beside any other stack. The first start builds the TensorRT engines (a few minutes). Then open, with
`<host>` this machine's address and 48000 the default `PPE_IRIS_PORT` (the recipe prints both):

| | |
|---|---|
| MJPEG stream | `http://<host>:48000/stream/cam-ppe-live` (test.mp4) or `http://<host>:48000/stream/cam-ppe-test2` (test2.mp4) |
| dashboard (alerts) | `http://<host>:48000/#/iris/dashboard` |

`just ppe-live-status` (video, containers, last fps and alert lines), `just ppe-live-logs` (`svc=iris` for the app),
`just ppe-live-sim VIDEO=...` again to switch videos, and `just ppe-live-down` to stop and remove it all. Its data
stays in `scratch/ppe-live/` (`PPE_LIVE_DIR`). **test2.mp4 shows real workers: internal use only, never shared outside the team.**

## Switching it on

Off unless a pipeline file lists it (the shared `perception/pipeline/sidecar.yaml` starts no module). Either:

- run the shared IRIS pipeline with PPE on, through the opt-in preset overlay:
  `SIDECAR_CONFIG=configs/presets/ppe/sidecar.yaml just up` (pipeline only; not a `just switch` preset);
- or add the one line `ppe: {}` under `modules:` in a product's sidecar yaml.

The model file is versioned with DVC: `just data-pull models-ppe` fetches `voxdroid_yolov8s_css.onnx` (or rebuild it:
"Getting the model" in `perception/models/DeepStream-Yolo-PPE/README.md`); without it the module refuses to start and
says so. Its licence covers non-commercial use only (same README).

All settings, with their defaults:

```yaml
modules:
  ppe:
    config: /workspace/iris/perception/models/DeepStream-Yolo-PPE/config_infer_ppe.txt
    interval: 0
    required_items: [helmet, vest]
    items: {}                    # per-item overrides by name, e.g. {vest: {absent_conf: 0.5}}
    min_person_height_px: 32     # mux pixels, every item unless `items` overrides it
    edge_margin_px: 2            # mux pixels
    eval_fps: 20                 # rules run at this rate per stream
    temporal: {}                 # window, min_absent_frames, min_present_frames, cooldown_s, forget_after_s,
                                 # close_after_present_s (module default 2)
    reid: {enabled: false}       # re-ID grace, off by default; max_gap_s 2, max_speed 0.7, recent_s 1, ...
    links: {}                    # the duplicate hint's track links: max_gap_s 4, max_dist 1.2
    alert_hold_s: 1              # a new violation waits this long for a second item of the same track
    zones_path: data/zones_config.txt   # zones whose `ppe` block is on (see "PPE zones" below)
```

Every value not set keeps the evaluated default from `compliance/config.py`, except four IRIS changes: persons are
the pgie's `person` label, the two pixel thresholds are in mux pixels (evaluated: 48 px and 3 px on 1080-high
frames, scaled to the 720-high mux), and an open episode closes only after 2 s of unbroken PRESENT
(`temporal.close_after_present_s`; evaluated: on the first PRESENT frame, which is 0). `eval_fps` keeps the temporal window ("absent in 6 of the last 8 evaluated
frames", tuned at 20 fps) about 0.4 s long on every camera. A bad item name, field or key refuses startup.

## The pipeline: persons, tracker, PPE detector on each person's crop

```
nvstreammux (1080x720) -> pgie YOLO11x (persons) -> nvtracker -> pgie_ppe: VoxDroid on each tracked person's crop
                                                              -> ppe module: rules per person, smoothing per track, alerts
```

VoxDroid YOLOv8s (Construction Site Safety dataset), `perception/models/DeepStream-Yolo-PPE/config_infer_ppe.txt`,
gie id 4:

- **On person crops** (`process-mode=2`, `operate-on-gie-id=1`, `operate-on-class-ids=0`): the detector runs on each
  tracked YOLO11x person, cut from the mux frame, and nvinfer maps its boxes back to frame coordinates. On the full
  frame people in CCTV views are too small for it: on test2.mp4 (3200x1800 camera, 1080x720 mux, model input 640),
  96.5% of person-frames stayed helmet `unknown`, mostly `no_evidence`. Crops are batched 16 per inference (the
  config's `batch-size`, engine `voxdroid_yolov8s_css.onnx_b16_gpu0_fp16.engine`); a frame with more persons runs more
  batches.
- **Bilinear crop scaling** (`scaling-filter=1`). nvinfer's default (nearest neighbour) gave crops the model could not
  read: on a 30 s clip of test2.mp4, PPE on 0 of 1,722 persons, against 765 with `scaling-filter=1`. Offline, the
  same model on our own crops of 60 persons found PPE on 31; DeepStream with the fix on 32.
- **After the tracker**: the module declares it `role="secondary"`. nvtracker deletes every object it does not track
  (below its 0.19 floor, new targets in probation, refused duplicates; ~19% of the confident PPE boxes on test.mp4),
  and the rules need every PPE box down to the detector's 0.10 floor. PPE boxes therefore carry no track id: never key
  anything on `o.id` for them.
- **Same thresholds as evaluated**: `pre-cluster-threshold=0.10`, `nms-iou-threshold=0.6`, the same four classes.
- nvinfer makes each box's person its parent, but the link does not reach the module on this pipeline (0 of 25,028
  boxes on test2.mp4); the rules match boxes to persons by position, as they did when evaluated.

A detector config with `process-mode=1` still works (full frame, the mux's batch size), for comparison.

## The rules: vendored from ppe-compliance

`compliance/` is a copy of the rule code of the ppe-compliance project (`/home/iris2/IRIS/ppe-compliance`) at commit
`b07c335`: `association.py`, `compliance.py`, `geometry.py`, `types.py` unchanged, and `config.py` trimmed to the
three rule configs. Each file's header says so. **Exception (IRIS change, not yet in ppe-compliance):** `temporal.py` and `TemporalConfig` gained `close_after_present_s`; its vendored default 0 keeps the
evaluated behaviour, and the ported tests pass unchanged.

**Rule changes must be made in both places**, here and in ppe-compliance. The precision, recall and limitations in
ppe-compliance's `docs/EVALUATION.md` were measured with that code; a change made only here would quietly invalidate
them, and one made only there would never reach the cameras. When you change a rule, change both, re-run the
evaluation there, and update the commit noted in the headers.

`adapter.py` is the only glue between the engine and `compliance/`: persons are the pgie's tracked, detected `person`
boxes, PPE is every box of the four PPE labels from the PPE detector, boxes go from `(left, top, width, height)` to
`(x1, y1, x2, y2)` in mux pixels, the frame size is the mux size, and track ids go back onto the results
(`tests/test_ppe_adapter.py`).

`tests/test_ppe_compliance.py` holds the 34 ported rule tests (association, per-person decisions, temporal smoothing)
with their assertions unchanged.

## Per-stream rules

`state.py` holds one `StreamRules` per camera, made in `on_built`. In `on_frame` (every frame, ahead of the publish
gate) it skips frames arriving faster than `eval_fps`, and on an evaluated frame runs the adapter, the per-frame rules
and the temporal filter, using the engine's monotonic `now` (pts restarts when a file loops). The smoothed persons go
to `frame.scratch["ppe"]` for the later hooks. Smoothing is per tracker id (or the id re-ID lets it inherit, below); an
id unseen for `temporal.forget_after_s` is dropped.

## Repeat alerts after a tracker id switch

nvtracker gives a worker a new id when it loses and re-finds them; the new id starts a new window, so the same
violation can alert again. Two mechanisms, both by box geometry and time only (no appearance features):

**Duplicate hint (always on): grouping, not suppression.** Every alert is raised. One that may repeat an earlier alert
of the same camera names it: the message ends `(possibly #22 again, alerted 9 s earlier)` and the entry carries
`possible_duplicate_of` (the earlier tracker id), `possible_duplicate_of_pts`, `possible_duplicate_gap_s` and
`possible_duplicate_distance`. The rule (IRIS; it replaces ppe-compliance's issue #15 rule, which on test2.mp4
named the wrong worker in 8 of its 14 hints): track links (`reid.py` `TrackLinks`). A track that appears, a new id or an
id back after missing a frame, continues the nearest lost track: one not in this frame, last seen at most 4 s ago, its
last box's centre at most 1.2 of that box's heights away (`links: {max_gap_s, max_dist}`), one-to-one, closest first,
no height test (a close-up or partial box changes the height a lot). Linked tracks form a chain; an alert names the
most recent earlier alert, for a shared item, of a different track of its chain. `possible_duplicate_gap_s` and
`_distance` are the link this track joined the chain by (absent when it did not join one itself). On the replayed
test2.mp4 run: 10 hints name the same worker, 1 another worker, 10 repeats are not hinted (all re-finds more than 4 s
or more than 1.2 heights away).

**Re-ID grace (`reid.py`): OFF by default** (`reid: {enabled: true}` turns it on). When on, a new id takes over a
recently lost one (lost at most 2 s ago, at most max(0.5, 0.7 x gap) heights away, heights within x1.5, the lost
worker's tallest box of their last second, one-to-one closest first, never boxes taller than half the frame:
ppe-compliance issue #13), and smoothing continues with the lost id's window, open episode and cooldown, so no second
alert. It is off because a wrong match is worse than a duplicate: on test.mp4 it joined the coverall worker to a
compliant worker walking next to them, which names the wrong person, and a wrong match can also swallow a real alert.
**Judge it on fixed-camera footage before enabling it anywhere**: test.mp4 is hand-held with close-up workers covering
the scene, and of its 7 matches (with the current rules) 3 were clearly right, 2 plausible and 2 could not be
confirmed, all with the lost box mostly hidden behind a foreground worker. Either way, alerts and the overlay always
name the tracker's CURRENT id (the box actually in the frame); the id re-ID inherits stays inside the temporal filter.

## Alerts

When the smoothed rules start a violation episode for a person (an item ABSENT in 6 of the last 8 evaluated frames,
not within the item's 30 s cooldown), the module raises one alert for that person through `services.alert`:

| field | value |
|---|---|
| `kind` | `ppe_violation` (the event's tag) |
| `severity` / `category` | `warning` / `safety` |
| `message` | `Person #12 without helmet on cam-1`, or `... without helmet and vest ...`: every item missing now; `#12` is the tracker's current id; a possible repeat ends `(possibly #9 again, ...)` |
| `pts` | the time of the frame the violation started on; the clip is centred on it |
| `count` | how many items are missing |
| `track_id`, `new`, `missing`, `possible_duplicate_*` | kept on the `iris:alert:<camera>` entry; the app does not read them, hence the message |

It also asks the engine for a clip (`request_clip`, as `fire_alerts` does), once per episode, when the violation
starts. The alert itself is held for `alert_hold_s` (1 s): the rules decide each item on its own, and an item that
starts on the same track while the alert is held joins it (one alert "without helmet and vest", `new` naming both); an
item that starts later gives its own alert, whose message names everything missing by then (`new` says which item
raised it). Held alerts are raised on the first frame after the hold, whether the person is still there or not, and
at shutdown. An open episode raises nothing more until the item has been seen PRESENT for 2 s without a break and the
cooldown has passed. Truncated or too-small persons, and
anything uncertain, are `unknown` and never alert.

### PPE zones

A camera with zones whose `ppe` block is on (`{"ppe": {"enabled": true}}` on a zone of the shared zones file,
`zones_path`) alerts only for a worker standing inside one -- the bottom centre of their box -- and the alert's
`zone` names it; a worker outside them starts no alert and no clip. A camera with no such zone covers the whole
frame, as before. The zones reload on `iris:ctrl:zones_version` (acknowledged on `iris:analytics:ppe_zones_applied`).
The zone's points are in mux pixels (perception/lib/feature_zones.py).

A zone can check only some of the equipment: `{"ppe": {"enabled": true, "items": ["helmet"]}}` (names from
`required_items`; the zone editor's PPE section, "Check for"). A worker standing in it is held only to those items --
the alert, the clip's label and the yellow highlight name only them. An item missing where it is not checked (outside
every PPE zone, or in a zone that does not check it) is kept for the track like any deferred violation, and alerts if
the worker walks into a zone that checks it while it is still missing. Without `items`, and on a camera with no PPE
zone, every required item is checked. The model knows two items: the hard hat (Hardhat / NO-Hardhat) and the vest
(Safety Vest / NO-Safety Vest); its mask classes are filtered out in its nvinfer config.

## Overlay

- **PPE boxes** (`kind = "ppe_item"`, layer 2, above the person boxes) are drawn in sky blue `(255, 191, 0)` with the
  caption `Hardhat 0.80` (the overlay would otherwise print the untracked id), and are **never published**: they are
  evidence, not detections of their own, so `iris:meta` and the VLM prompt do not see them. A product does not need to
  list them in `relevant_classes`.
- **A person in violation now** (smoothed ABSENT for any required item) is drawn yellow `(0, 255, 255)` at priority 8
  with `#12 no helmet`. The highlight follows the last evaluated frame, so it holds on skipped frames and clears once
  the item is no longer ABSENT. Priority 8 beats the zone colour (1) and loses to fire orange (10), in any module order
  (`perception/NEXT_MODULES.md` colour table).

## Results on video

Two recorded videos, the module's default settings, no tuning per video. Decisions are smoothed statuses per evaluated
person-frame (eval_fps 20); the same YOLO11x persons and tracker in both columns, only the PPE detector's input differs.

**test2.mp4** (CCTV view, 3200x1800, 20 fps, 372 s, small people; 22,210 person-frames):

| | full frame | crops |
|---|---|---|
| helmet present / absent / unknown | 390 / 383 / 21,437 (96.5%) | 508 / 3,310 / 18,392 (82.8%) |
| vest present / absent / unknown | 1,135 / 1,746 / 19,329 (87.0%) | 1,328 / 2,272 / 18,610 (83.8%) |
| alerts, before the repeat-alert settings (helmet / vest; some both) | 27 (7 / 23) | 46 (34 / 20) |

All 46 crop alerts were reviewed by eye: **54 of 54 item answers correct (precision 100%; helmet 34/34, vest 20/20)**,
none wrong or unsure. Recall was not measured. The 46 alerts came from 12 workers; replaying the same run with the
three repeat-alert settings below gives **36 alerts**, every worker keeps their first alert, and hints naming the
wrong worker go from 8 to 1.

**test.mp4** (hand-held, construction site, 1920x1080, 30 fps, 22 s, most workers wear PPE; 2,110 person-frames),
both runs with the repeat-alert settings:

| | full frame | crops |
|---|---|---|
| helmet present / absent / unknown | 1,037 / 0 / 1,073 (51%) | 1,578 / 0 / 533 (25%) |
| vest present / absent / unknown | 1,040 / 27 / 1,043 (49%) | 1,389 / 147 / 575 (27%) |
| alerts (helmet / vest) | 2 (0 / 2) | 2 (0 / 2) |

Both runs alert twice, for vest, on the worker in a blue coverall with reflective strips (two tracker ids, not linked:
the first id's last box was half hidden). A worker whose head is cut off by the top of the frame stays helmet
`unknown` (`person_truncated`) or present, never absent; an open vest over a dark shirt is never a vest alert.

### Why the three repeat-alert settings

On test2.mp4 (crops) the 46 alerts came from 12 workers. The extra alerts had three causes, each with one setting:

| setting (default) | cause on test2.mp4 | effect |
|---|---|---|
| `temporal.close_after_present_s` (2 s) | 6 alerts: the same track re-alerted after a short false PRESENT (a bare head or cap read as a hardhat for 0.4-1.45 s) closed the open episode | an episode closes only after 2 s of unbroken PRESENT |
| `links` (`max_gap_s` 4, `max_dist` 1.2) | 21 alerts: a new tracker id for the same worker; the old hint named the wrong worker in 8 of 14 cases | `possible_duplicate_of` follows track links (grouping only, nothing is dropped) |
| `alert_hold_s` (1 s) | 7 alerts: a second item of the same track a moment later | the second item joins the first alert |

## Accuracy in IRIS (2026-10-26)

Measured with the PPE detector on the full frame, before it moved to person crops; not yet re-measured with crops.
Person-level, on the Kaggle css-data test split ppe-compliance was evaluated on (82 images, 174 labelled persons),
run through the real sidecar: YOLO11x persons (tracked), the PPE detector after the tracker, the module's default
settings. Scored exactly as ppe-compliance's `person_eval` does (ground truth per person from the labels by the same
association, one-to-one IoU >= 0.5 matching, the same metrics), next to its own numbers (VoxDroid's own persons, no
tracker, images fed unstretched):

| metric | helmet, ppe-compliance | helmet, IRIS | vest, ppe-compliance | vest, IRIS |
|---|---|---|---|---|
| violation precision, strict | 0.79 (15/19) | 0.76 (13/17), CI 0.53–0.90 | 0.98 (49/50) | 0.96 (45/47), CI 0.86–0.99 |
| violation precision, definite label | 1.00 (15/15) | 1.00 (13/13) | 1.00 (49/49) | 0.98 (45/46) |
| violation recall | 0.38 (15/40) | 0.33 (13/40) | 0.66 (49/74) | 0.61 (45/74) |
| present precision, strict | 0.94 (81/86) | 0.99 (71/72) | 0.94 (50/53) | 0.95 (39/41) |
| unknown rate, matched persons | 0.28 (40/142) | 0.22 (25/113) | 0.29 (41/142) | 0.24 (27/113) |
| unknown rate, definite label | 0.18 (21/117) | 0.15 (15/99) | 0.11 (12/111) | 0.12 (12/97) |
| persons matched to a label | 142 of 174 | 113 of 174 | | |

- **Recall is lower mainly because fewer people are found**: the pgie finds 131 persons before the tracker, and the
  tracker deletes 14 more (nested duplicates, held-back new tracks), so 117 reach the module and 113 match a label.
  Missed people are 14 of the 27 missed helmet violations and 18 of the 29 missed vest violations.
- **Helmet strict errors**: all 4 on one image, confident NO-Hardhat on persons whose helmet label is ambiguous; the
  same 4 ppe-compliance had and judged the system right on.
- **Vest**: one new definite error, a flight-deck crew member in a yellow jersey labelled "Safety Vest" with
  NO-Safety Vest 0.48 (just over 0.40); whether a jersey counts is debatable. The other strict error is a person with a
  NO-Safety Vest label but no Person label, where the system's "absent" agrees with the label.

Caveats: the 640x640 images are **stretched to the 1080x720 mux** like any camera (1.69x wider, 1.13x taller) and
letterboxed back to 640 for the detector, so people reach it at about 2/3 of their original height relative to width;
ppe-compliance fed them unstretched. Each image was held for 10 frames so the tracker confirms each person, and scored
on the last one. The test/train leakage check of ppe-compliance is still open, so both systems' numbers may be
optimistic. Samples are small (13–45 true violations per item) and the confidence intervals overlap throughout. On
video (test.mp4) the module's alerts matched the demo's, both for the coverall worker (an open question: does a
coverall with reflective strips count as a vest?).

How it was run: `/home/iris2/IRIS/ppe-spike/scripts/` (`make_test_video.py`, `score_css.py`, local, not in the repo).

## Testing

```bash
python3 -m pytest tests/test_ppe_*.py -q
```
