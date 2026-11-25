# Fall detection on the IRIS cameras — design

**Status:** draft for review, 2026-10-27 · **Audience:** supervisor + the implementer
**Scope:** the pose model, the pose module, the vendored fall monitor, the fall product service, the
scoring policy, the deployment, and the live annotation overlay (the next release).
**Plan:** `~/.qoder/plans/polished-glen-newt.md` · **Requirements:** [`FALL_DETECTION_REQUIREMENTS.md`](FALL_DETECTION_REQUIREMENTS.md)

---

## 1. What this is

One **pose** model runs on each tracked person in the sidecar. The keypoints ride on the sidecar's
existing frame stream, `iris:meta:<cam>`, as the namespaced field `pose`. A **fall product service**
reads that field, runs the vendor's `FallMonitor` per camera, scores each candidate against a policy
the product owns, and raises the platform's one alert contract.

Nothing in `perception/engine/` changes. Nothing in `app/` changes. No other product's
`sidecar.yaml` changes. The fall product owns its own sidecar overlay, its own compose service, its
own role and its own cameras.

### 1.1 The shape

```
  the fall sidecar (the engine process)                 the fall service (a host process)
  ────────────────────────────────────────               ────────────────────────────────
  pgie (persons) → tracker → sgie_pose (17 kps)
                              │
                              ▼
  iris:meta:<cam>  field `pose` = {shape:[w,h], items:[{id, box, kps[17][3]}]}
                              │
                              └──────────── XREAD ──────────▶ the consumer thread
                                                             → a bounded per-camera queue
                                                             → a per-camera worker: FallMonitor.update()
                                                             → the scoring policy
                                                             → the routes
                                                                  ├─▶ iris:svr_request:<cam> {pts}
                                                                  └─▶ iris:alert:<cam>  kind=fall
```

### 1.2 The components

| # | Component | Where | New or copied |
|---|---|---|---|
| C1 | the pose model + its parser + its nvinfer config | `perception/models/DeepStream-Yolo-Pose/` | new (exported from `yolo11n-pose`, then re-exported from `yolo26x-pose`) |
| C2 | the pose module | `perception/modules/pose/` | new, follows `face_detect` |
| C3 | the vendor's monitor | `products/fall_detection/fall_detection/` | a copy of `F:\fall_detection_module\src` |
| C4 | the fall product service | `products/fall_detection/` | new, forks the barber's product |
| C5 | the scoring policy | `products/fall_detection/scoring.py` | new, inside the IRIS repo |
| C6 | the boot contract | `products/fall_detection/` | new, inside the vendor's copy's config |
| C7 | the deployment | compose + `justfile` + the fall sidecar yaml | new |
| C8 | the live annotation overlay (**next release**) | `perception/modules/fall_overlay/` | new |

### 1.3 The decisions this design rests on

These are the plan's locked decisions; the design adds nothing to them. Where the design **refines**
one, it says so.

1. The combined score is computed in the IRIS service.
2. The threshold decides whether a human is needed, not how urgent the event is.
3. The action-recognition path (ST-GCN) is in the first release, on the GPU, behind a start-up gate.
4. The vendor's code is copied into the product, not depended on.
5. The pilot starts at 12.5 Hz (`interval: 2`); 25 Hz is a required experiment before shipping.
6. The fall product owns its sidecar; it replaces the attendance deployment on its cameras.
7. The score is the confidence in the evaluation; the paths combine as `1 - Π(1 - trust·score)`.
8. The live annotation overlay is the next release, and the simple way (Option A) is first.
9. **Refinement.** The keypoints ride on `iris:meta:<cam>` as the field `pose`, not on a dedicated
   channel. The engine's `emit` is a plain `xadd` (`perception/engine/gst_build.py:418-424`), so a
   nested `items:[{...}]` value would be rejected by redis-py; the meta field is JSON-encoded by the
   publisher instead (`perception/engine/publisher.py:348-349`). One stream to read, and the data
   arrives with the frame's own clock.
10. The alert is the platform's contract, used as it is.
11. The repo's patterns are reused, not re-invented.

---

## 2. C1 — the pose model

**Contract.** A secondary GIE that runs on each tracked person's crop and produces **17 keypoints ×
(x, y, conf)** for that person, in the network's input coordinates, plus the network's input size.

**Why a secondary GIE.** The people are already detected and tracked by the product's primary
detector; a secondary GIE runs on the tracked objects' crops, so no new full-frame inference is
added (`perception/NEXT_MODULES.md:41-53`). The face detector is the pattern to copy
(`perception/modules/face_detect/__init__.py:126-130`).

**The export.** `yolo11n-pose` exported to ONNX (the tool's default checkpoint; the shipped
`yolo26x-pose.onnx` goes through the same export script). The stock Ultralytics pose export does **not**
produce the tensor the parser needs, so the export script is model-agnostic and hand-written
(`export_pose.py`):

| Tensor | Shape | Meaning |
|---|---|---|
| boxes | `[batch, N, 4]` | x1, y1, x2, y2 in network-input pixels |
| scores | `[batch, N]` | the detection confidence |
| keypoints | `[batch, N, 17, 3]` | x, y, conf, in network-input pixels |

The batch axis is there because nvinfer batches the people's crops into one inference. The parser
reads one crop's output — `[N,4]`, `[N]`, `[N,17,3]` — and accepts the batched view as well, so it
does not depend on which of the two nvinfer hands it.

The proof that the export is right is numeric equality against `model.predict()` on a real
image, to within a couple of pixels — every detection, not only the best one, and every
confidence. Any image the model finds people in serves (`--check <image>`); the shipped default is
ultralytics' own `bus.jpg`, so the proof runs anywhere, and the deployment proves it once more on
its own footage. The joints the network never saw are excused: their positions are noise in both
runtimes, so the proof counts them and reports them instead of judging them. A graph whose three
tensors have another **shape** is refused with a verdict, not read: a wrong export must be a failed
proof, not a crash.

**The model is a switch, not a constant.** The checkpoint's own head says how many joints it has,
and the pose model has one class — the person — which is the class the parser reports, so
`yolo26x-pose` goes through the same script and produces the same three tensors. Which model runs is
stated in two places that must agree — the config file (what nvinfer loads) and the module's `model`
setting — and the module refuses to start when they disagree, so a half-finished switch fails the
boot instead of quietly running the wrong model. The procedure is
`perception/models/DeepStream-Yolo-Pose/MODEL_SWITCH.md`. The pilot started on the **nano**
model and ships `yolo26x-pose` now: this GIE runs on every tracked person's crop, so its size
is a GPU lever (§3.5).

The ONNX is **one file** and it is **tracked**: the engine image carries no torch, so it cannot
export the model itself, and every model in `perception/models/` is a single self-contained `.onnx`
(the export writes no `*.onnx.data` sidecar).

**The parser.** `nvdsparsepose_Yolo.cpp` in `nvdsinfer_custom_impl_Yolo_pose/`, modelled on the face
parser (`nvdsparseface_Yolo.cpp`): the threshold and the NMS live in the parser, and the payload
lands in the object's instance-mask field, exactly as the face parser packs its five landmarks
there. The pose parser packs **51 floats** (17 × 3) plus the network input size — the face parser's
shape, scaled (`perception/modules/face_detect/__init__.py:48-57`).

The image builds every `nvdsinfer_custom_impl_*` automatically
(`perception/pipeline/Dockerfile:72-81`), so a new parser directory needs no Dockerfile change of
its own.

**Where the implementer must focus.** The crop-to-network-input mapping. The face network runs on
the person's box scaled to fit `net_w × net_h` keeping its aspect ratio, centred with equal padding
on both sides (`config_sgie_face.txt`: `maintain-aspect-ratio=1`, `symmetric-padding=1`); the pose
network must do the same, or every keypoint is offset. `landmarks_to_capture`
(`perception/modules/face_detect/__init__.py:60-76`) already does the inverse of that mapping for
five points and is the code to generalize.

**How it is proven correct and not slow.** Correct: the exported tensor's shape and values against
`model.predict()`, and the parser fed a tensor whose contents are known
(`nm -D` shows `NvDsInferParseYoloPose`; `tests/test_pose_parser.py` compiles the parser against a
stub of DeepStream's header and runs it on a plain host, so a field DeepStream does not have is a
compile error here rather than a container build failure).
Not slow: `tools/engine_gpu_budget.sh` before and after; the delta is the cost of the feature.

---

## 3. C2 — the pose module

**Contract.** One module, `perception/modules/pose/`, that adds the secondary GIE, turns each
person's raw keypoints into **capture pixels**, and publishes the `pose` field on the frame.

```
settings:
  config: /workspace/iris/perception/models/DeepStream-Yolo-Pose/config_sgie_pose.txt
  engine: ""                       # the engine path; empty means the repo's own derivation from the config file
  model: yolo26x-pose              # the model this module runs; the config file names the ONNX it loads
  detection_threshold: 0.25        # the detection floor; the config file declares the same number
  keypoint_confidence_threshold: 0.30   # the JOINT floor: a joint below it is not seen
  min_keypoints: 6                 # how many joints a person must be seen through to be published
  interval: 2                      # the SGIE's own re-inference interval: 2 = 12.5 Hz at a 25 Hz pipeline
```

The engine's name is **derived**, not declared: nvinfer serializes it beside the ONNX as
`<onnx>_b<batch>_gpu<id>_<mode>.engine`, and the module derives that name from the config file's
own `onnx-file`, `batch-size`, `gpu-id` and `network-mode` (`perception/engine/engine_path.py`,
the same helper the fire module uses). The `engine` setting is only an override for the rare case
where the engine lives somewhere else.

The `interval` number lives in **two** places, and they must agree: the module's setting above and
the config file's `secondary-reinfer-interval`. nvinfer reads a secondary GIE's re-inference
interval from its **config file** when it creates the element — the element property `interval`
gates the *primary* GIE only, and no element property exists for the secondary one. The module's
setting is therefore the *declaration*, and `on_built` compares it with the file and raises
`ModuleStartupError` when they disagree: a silent mismatch would run the feature's first GPU lever
at an interval nobody asked for. Change both, or neither.

### 3.1 The hooks

| Hook | What it does |
|---|---|
| `gies()` | one `GieSpec(name="sgie_pose", role="secondary", batch_from_mux=False)` |
| `on_built(pipeline)` | compares the setting's `interval` with the config file's `secondary-reinfer-interval`, the setting's `model` with the config file's `onnx-file`, and the setting's `detection_threshold` with the config file's `pre-cluster-threshold`; it also refuses to start when the config file names an ONNX that is not beside it, or when it declares a scaling of the crop into the network's input other than the letterbox the module inverts (`maintain-aspect-ratio=1`, `symmetric-padding=1`), and raises `ModuleStartupError` when any of these disagree |
| `extract(frame)` | reads the object's instance mask through `perception.engine.nvds.instance_mask` and stores the raw keypoints + the network size in `obj.attrs["keypoints_raw"]` |
| `transform(frame)` | maps the raw keypoints to capture pixels, attaches them to the person, and writes `frame.scratch["pose"]` |
| `analyze(frame, out)` | `out.fields["pose"] = {"shape": [w, h], "items": [...]}` |

`transform` is where the crop math lives, and it is the highest-risk correctness item in the whole
feature: the box and the keypoints must end up in **one** coordinate space.

**Three rules the module holds to, each of them a way the feature could lie:**

1. **One entry per person, not per detection.** The network can report the same person twice; the
   monitor would then see one track twice in one frame, and its kinematics (two observations of one
   `t`) would be corrupt. The detection whose joints land inside its own box is the one that
   explains it, so it wins.
2. **A person seen through fewer than `min_keypoints` joints is omitted**, not published with
   zeroes: one confident joint is not an observation. The vendor's kinematics read the two legs
   (three joints each) and the torso, so below the floor the monitor would be measuring a body the
   network mostly did not see.
3. **A frame the network did not run on carries no `pose` field.** The SGIE's re-inference interval
   means it does not run on every frame, and "the network did not look" is not "there is nobody
   there": the vendor's monitor moves every track it cannot see into the lost ones, so an empty
   field on a skipped frame would drop every track at the pilot's own 12.5 Hz. An **empty scene**
   (no people, no items) does publish an empty field — the monitor has to hear that the people are
   gone. The consumer reads the three cases apart (`products/fall_detection/consumer.py`).

### 3.2 The coordinate contract

Three spaces exist in this pipeline, and mixing them is the failure mode to design against:

| Space | What it is | Where it comes from |
|---|---|---|
| network input | the cropped, letterboxed person image the pose net saw | the parser's `net_w × net_h` |
| mux | the sidecar's tiled frame, where the detections live | `services.mux_size` |
| capture | the camera's own frame, which the published JPEG is | `frame.capture_scale()` → `sx, sy` |

The rule, fixed for the whole feature:

> **The `pose` field is written in capture pixels**: `box` = `[x1, y1, x2, y2]` in capture px,
> `kps` = 17 × `[x, y, conf]` in capture px, and `shape` = `[capture_w, capture_h]`.

Why capture px and not mux px: the vendor's `FallMonitor` normalizes the keypoints against
`img_shape` (`F:\fall_detection_module\src\keypoint_adapter.py:111-128`) and sanitizes them against
the person's box (`sanitize_keypoints_17`), so the box and the keypoints must be in the same space,
and that space must be the one the service passes as `img_shape`. The publisher scales mux → capture
when it draws (`perception/engine/publisher.py:43-44`), so capture px is the space the operator sees.

`shape` travels with the data precisely so the service never has to guess: it passes
`img_shape=(shape[1], shape[0])` — `(h, w)`, the order the vendor expects.

### 3.3 The owner (which person a keypoint set belongs to)

The secondary GIE's items arrive with `parent_id` **unset** on this pipeline
(`perception/modules/face_detect/__init__.py:61-65` finds it unset; `perception/NEXT_MODULES.md:58`
says it should be set). The module must not rely on it.

Two questions have to be answered separately, and they have different answers:

1. **Which person does this keypoint set belong to?** The **containment match** `face_detect`
   already ships: `perception/modules/face_detect/owner.py:117-158` (`_overlap`, `_one_per_face`,
   `crop_box`). The pose module reuses that match to find the person each keypoint set belongs to,
   and records which of the two is true in S1 (before anything is built on it) so the choice is
   evidence-based, not assumed.
2. **Which rectangle did the network actually see?** The **owner's** rectangle. A secondary GIE
   runs on the tracked object's *crop* (§1.3, "a secondary GIE runs on the tracked objects' crops"),
   so the crop rect is the person's rect — *not* the pose item's own box, which is the network's
   **detection** box inside the crop. The module recovers the crop rect from the geometry: the
   person's rect whose letterbox inverse puts the keypoints inside the item's own box is the one
   the network saw; when no rect explains the item, the owner's rect is the fallback (the face
   module falls back to the owner's rect for the same reason). Getting this wrong displaces every
   keypoint, so it is the first thing the unit test pins down.

### 3.4 The published field

```json
"pose": {
  "shape": [1280, 720],
  "items": [
    {"id": 42, "box": [310.5, 180.0, 455.25, 612.75], "kps": [[x, y, conf], ... 17 ...]}
  ]
}
```

* `id` is the **DeepStream tracker id** — the same id the engine draws as `#<oid>` on the person's
  box (`perception/engine/publisher.py:56-70`). It is the join key for the whole chain: the module →
  the service → the monitor → the state map → the overlay.
* `box` is **xyxy** (the vendor's convention: `update_ref_scale` reads `b[2] - b[0]`,
  `F:\fall_detection_module\src\fall_monitor.py:120-150`), not the engine's `left, top, width, height`.
* `kps` are the 17 keypoints in the COCO order the vendor's `sanitize_keypoints_17` expects.
* People seen through fewer than `min_keypoints` joints are **omitted** from `items`, not published
  with zeroes: the monitor's own sanitization would reject them anyway, and the service treats a
  missing person as "not observed" rather than "observed as a heap of zeroes".

### 3.5 How it is proven correct and not slow

* **Correct:** a unit test of the crop → capture mapping against a known crop; a golden replay
  (`tests/configs/golden-fall_detection.yaml`) asserting the `pose` field's shape and a person's
  keypoints; the on-video check — the pose field's keypoints land on the person's joints.
* **Not slow:** `tools/engine_gpu_budget.sh` before and after. An ungated secondary GIE was ~40 % of
  the sidecar's GPU time; the pilot runs at `interval: 2` (12.5 Hz) and the 3-camera preset is
  measured separately.
* **The GPU levers, in order:** the SGIE's own re-inference interval (the config file's
  `secondary-reinfer-interval`, gated by the module's `interval` setting) → the camera count → the
  model size. The sampling gate (`perception/engine/gie_gate.py`, `GieGate.install`) is **not**
  available here: it returns `None` when `pipeline_preview` is true, and the preview is what the
  overlay needs.

---

## 4. C3 — the vendored monitor, and the one edit it needs

**Contract.** `products/fall_detection/fall_detection/` is a byte-for-byte copy of
`F:\fall_detection_module\src`, with **one additive edit** to the candidate packaging.

### 4.1 The copy, and the two files that must sit next to it

The vendor's package computes its own config path at import time:

```python
ROOT = Path(__file__).resolve().parent.parent
CFG = load_cfg()          # ROOT/config.yaml, or ./config.yaml
```

(`F:\fall_detection_module\src\fall_monitor.py:58-72`). Two consequences, both load-bearing:

1. **The package must live one directory below the product root** — `products/fall_detection/fall_detection/` —
   so that `ROOT` is `products/fall_detection/`. Any other depth points the config loader at the
   wrong directory and the import fails.
2. **The vendor's master config must be copied to `products/fall_detection/config.yaml`** — beside the
   package, where the loader looks for it. Without it the import raises `FileNotFoundError`.

This is also why the product's **own** config is named `fall_detection.yaml` and not `config.yaml`:
two files, two names, one directory. The product's config is the *policy*; the vendor's is the
*mechanism's* thresholds.

| File | Whose it is | What it holds |
|---|---|---|
| `products/fall_detection/config.yaml` | the vendor's (copied, **plus the score spans**) | the gates' thresholds: `path_a`, `path_b`, `path_c`, `fall`, `immobility`, `normalization`, … |
| `products/fall_detection/fall_detection.yaml` | the product's | the policy: `scoring`, the class sets, the trust weights, the threshold, the routes |

The score spans live in the **vendor's** config, next to the gates they measure the margin against —
one source of truth per quantity, and no second copy of a threshold to drift.

### 4.2 The one edit: the seam

The monitor's **boolean gate stays exactly as it is**. A candidate is still only created when a
path's hard condition is met. The edit is **additive**: it puts the soft evidence the product's
policy needs on the candidate that was already created.

Two keys are added to every emitted candidate (`rec`, the packaging at
`fall_monitor.py:1007-1024`):

```python
rec['reading'] = ({'probs': probs_smooth.tolist(), 'top': int(np.argmax(probs_smooth))}
                  if has_win else None)
rec['fall_paths'] = ev.get('fall_paths')          # {"a": bool, "b": bool, "c": bool}
```

* `reading` — the classifier's **smoothed** 8-class vector over the window
  (`fall_monitor.py:690`) and its top class, or `None` when no window existed. The smoothed
  vector, not the raw one: the raw `probs_7` is already on the record as
  `evidence.tsstg_probs`, but the paths' decisions use the smoothed vector, and the score must
  measure the same thing the gate measured.
* `fall_paths` — which of the three fall paths latched for the event. It is the copy's record,
  carried as **data**; it is not a score.

**Which path spoke.** The fall paths are latched in one place (`fall_monitor.py:772-777`:
`if path_a or path_b or path_c: d['pending_fall_t0'] = t`). The edit records the latching set on
the track (`d['fall_paths']`), carries it onto the event (`fall_monitor.py:845`), and clears it
wherever the vendor clears `pending_fall_t0`. A fall that satisfies both A and B contributes
both scores; the combination is over the paths that spoke.

**The scores are the product's, computed outside the copy.** The service's monitor
(`monitor.attach_path_scores` → `scoring.fall_path_scores`) turns the latch and the candidate's
own `evidence.kinematics` into:

```python
rec['path_scores'] = {'action': None, 'kinematic': <float|None>, 'cold': <float|None>}
```

* the **action** slot stays `None` there: the action score is the reading's *margin*, and the
  margin needs the product's class sets (§6.2), so the consumer fills it at scoring time;
* the **kinematic** and **cold** slots are the collapse margin of each path that latched.

The copy therefore imports nothing from the product and the formula has one implementation
(`scoring.py`). The seam is recorded as nine exact edits in
`products/fall_detection/tests/pristine_manifest.json`; `tests/test_seam_vendor.py` reverses
them to prove every other byte of every copied module is the pristine text.

Six of the nine are the seam itself. Two of the other three -- one in `fall_monitor.py`, one in
`tsstg_model.py` -- rewrite the vendor's `ROOT = Path(__file__).resolve().parent.parent` as
`ROOT = Path(__file__).resolve().parents[1]`. **That is the same path** (`parents[1]` *is*
`parent.parent`, in every layout), so no behaviour changes; the copy lives two levels deep
(`products/fall_detection/fall_detection/`), and the repo's own rule
(`tests/test_product_paths.py`) bans counting parents in a product's sources, because it broke
silently when a product moved one level deeper. The copy satisfies the rule instead of
weakening it.

The ninth (added 2026-10-29) sits at the top of the dedup section: once a ledger of the dedup
state passes `events.state_cap` (default 4096) the ledgers halve by insertion order -- the
oldest keys first, in plain dict order and with no clock arithmetic, so a PTS restart cannot
corrupt the pruning. It bounds the memory of a days-long run. The accepted cost: a key old in
position while fresh in value (a track that keeps re-emitting) loses its `realert_s` state
once, and the next candidate re-arms it.

**The physical score.** For a fall candidate it is the collapse margin over the vendor's own
thresholds — the same quantities the boolean gates compute, turned into soft numbers:

```
s_collapse = (s_drop · s_vdown · s_hip · s_posture) ^ 0.25
s_drop    = clip((drop    - drop_min)  / drop_span,  0, 1)
s_vdown   = clip((vdown   - vdown_min) / vdown_span, 0, 1)
s_hip     = clip((hip_max - hip_rel)  / hip_span,   0, 1)
s_posture = max(angle_score, aspect_score)
```

The thresholds (`drop_min`, `vdown_min`, `hip_max`, the angle and the aspect thresholds) and the
spans come from the path's own config block — `path_b` for the kinematic path, `path_c` for the cold
path — so the soft score and the gate that admitted the candidate read the same number.

The geometric mean (the `^ 0.25`) is deliberate: it is the one aggregate where a single collapsed
condition cannot be averaged away, which is what a conjunction means.

For an **immobility** candidate the physical score is the stillness margin:
`clip((disp_thresh - disp_win) / disp_span, 0, 1)` — the same shape against the immobility
detector's own threshold.

### 4.3 The silent degradation, closed

`FallMonitor.__init__` builds the classifier inside a bare
`except Exception: self.classifier = None` (`fall_monitor.py:459-469`), so a missing or corrupt
checkpoint turns the system into the kinematic-only one while the operator believes all three paths
run. The product does not inherit that behaviour:

* the **product** builds the classifier itself, verifies both checkpoints against the vendor's
  SHA-256 registry (`WEIGHT_SHA256_REGISTRY`, `tsstg_model.py:122-131`), and hands the built
  classifier to `FallMonitor(classifier=...)` — so the monitor's own `except` never fires, because
  the classifier is never `None`;
* the **product** refuses to start without it (§8).

The vendor's `except` stays in the copied code — it is the vendor's code, and the copy is not
edited to remove it.

### 4.4 How it is proven correct and not slow

* **Correct:** the vendor's own nine test files come with the copy and run under `just fall-test`;
  the seam edit gets its own test (a candidate carries the reading and the latching paths, the
  product's monitor turns the latch into the path scores, and the boolean gate's behaviour is
  unchanged — the same fixture produces the same candidates before and after; every other byte of
  the copy is the pristine text, checked from the committed manifest).
* **Not slow:** the classifier runs in the **product service** (a host process), not in the sidecar —
  the sidecar's GPU budget is untouched by it. The cost that matters is the service's: milliseconds
  per forward on `cuda:0`, the number of concurrent tracks, and the whole-card view.

---

## 5. C4 — the fall product service

**Contract.** A host process that reads the `pose` field, runs one `FallMonitor` per camera, applies
the policy, and raises the platform's alert.

**It is the barber's product, forked.** `products/barber/` already is the pattern for a product that
runs beside its own sidecar: `acl.py` (the anti-corruption layer over Redis), `config.py` (the yaml
loader), `storage.py` (SQLite), `server.py` (the HTTP API), a `sidecar.yaml` overlay, a compose
service and `just` recipes. The fall product copies that shape and changes nothing in the barber's
folder.

| File | What it is | Where it comes from |
|---|---|---|
| `products/fall_detection/acl.py` | the Redis reads: the meta cursor, the alert writer, the clip requester | **the pattern**, not the file: the fall product re-declares its shapes. The reusable piece is the cursor reader (`_Cursor.read`, `products/barber/acl.py:155-163`)  |
| `products/fall_detection/config.py` | the product's **own** yaml loader (see the section below) | **new** — only `_merge` and `_read` are borrowed |
| `products/fall_detection/storage.py` | the review queue in **SQLite** | **the pattern**: the barber's is itself forked from attendance's — *kept: `connect()`* (busy timeout, WAL, schema + migrations on every connect); everything else is the fall tables |
| `products/fall_detection/server.py` | the HTTP API (`/status`, `/cameras`, `/candidates`, `/review/<id>`) | **the pattern**: FastAPI + uvicorn with an app factory (`create_app`) so the tests can run the API without the Redis consumer |
| `products/fall_detection/monitor.py` | the per-camera `FallMonitor` wrapper + the latch→path-scores step | new |
| `products/fall_detection/scoring.py` | the policy: the class sets, the margin, the combination, the routes | new |
| `products/fall_detection/consumer.py` | the Redis consumer thread + the per-camera queues + the workers | new |

**"The pattern, not the file" is the repo's own idiom, and the products say so in their own
docstrings.** The barber's ACL: *"Forked from products/attendance/acl.py @ 1f5b27b — kept: the camera
clock and LiveCameraWatch; the rest is new."* The barber's storage: *"connect() is forked from
products/attendance/storage.py @ 1f5b27b (busy timeout, WAL, schema + migrations on every connect) —
changes: everything else is the barber tables."* The barber's server: *"The websocket is forked from
products/attendance/server.py @ 1f5b27b — changes: staff instead of employees, an app factory
(create_app) so tests can run the API without the Redis consumer."*

So "forked from" in this repo means **copy the file, keep the generic helper, write the rest for
your product** — never "import the other product's module". The fall product does the same, and
nothing in another product's folder changes.

The ACL's rule is the one to keep: **nothing in the product imports another product's code — the
shapes are re-declared** (`products/barber/acl.py:1-16`). The fall product declares its own stream
prefixes, its own event shape and its own tables.

### The fall product's configuration

Three files, three loaders, and **none of them is the barber's**:

| File | Loaded by | Holds |
|---|---|---|
| `products/fall_detection/sidecar.yaml` | the **engine's** loader (`perception/engine/sources.py:186-202`) | the fall product's overlay: its cameras, its role, the `pose` module |
| `products/fall_detection/fall_detection.yaml` | the **product's own** `config.py` | the policy: the scoring block, the classifier's device, the weights dir, the service's own settings |
| `products/fall_detection/config.yaml` | the **vendor's** code, at import (`fall_monitor.py:58-72`) | the monitor's thresholds — the vendor's master config, copied |

The engine's loader and the vendor's loader are shared and need nothing from the product. Only the
middle one is the product's own code, and it is **written for this product**.

The barber's `config.py` is ~85 % barber domain: it imports `saved_reports`, `policy` and `zones`,
and its dataclass carries `engine: shop|chair`, `wait_camera`, `branch`, `branch_slug`, `place`,
`logo`, `kids_car_model`, `saved_reports` and `footage_start`. Copying that file and deleting the
barber parts would import another product's modules — the one thing the ACL rule forbids — and
would be more work than writing the fall product's own.

What the fall product **borrows** from it is eleven lines of generic, pure code
(`products/barber/config.py:64-79`):

* `_merge(base, overlay)` — the deep dict merge;
* `_read(path)` — the `extends:` overlay with its cycle guard.

And the *shape*, which is the pattern worth keeping:

* `SERVICE_DIR` / `REPO_DIR` computed from `__file__`, so the product finds its own files wherever
  the checkout is;
* the path and the data dir overridable by the environment (`FALL_DETECTION_CONFIG`,
  `FALL_DETECTION_DATA_DIR`), so one build serves every site;
* a frozen dataclass for the loaded config, so nothing mutates it at runtime;
* **fail fast** — a missing camera or an invalid setting raises at start-up, not at the first
  request.

**This is the repo's own pattern, not a deviation from it.** `products/attendance/` has no config
loader at all (its cameras are hard-coded in `cameras.py`, which is what the barber's loader
replaced); `products/multicam/` has no service; `perception/lib/` holds three pure helpers that only
the **modules** use. The products are autonomous and the duplication between them is the accepted
cost (`MULTI_PRODUCT_PLAN.md`, decision 8).

The `extends:` key is the one feature worth keeping for the fall product's own yaml: it is how the
barber makes a site variant (`barber.palm-hills.yaml` overlays `barber.yaml`,
`products/barber/config.py:71-79`), and it costs the two borrowed functions.

### 5.1 The consumer

```
Redis consumer thread  --XREAD iris:meta:<cam>-->  one entry per frame
        │
        ├─ parse the `pose` field (JSON) -> the items
        ├─ hand (t, ids, boxes, kps) to the camera's bounded queue (drop the oldest when full)
        │
        └─ the per-camera worker: FallMonitor.update(t, ids, boxes, kps)
                                  -> the candidates -> the scoring -> the routes
```

* **One `FallMonitor` per camera.** `FallMonitor.tracks` is keyed by the track id alone, so two
  cameras sharing a monitor would cross-talk. This is a unit test, not a comment.
* **The clock is the frame's own.** The entry's `pts` (the sidecar's frame clock) is the timestamp
  the monitor is given, because the clip request must use the same clock
  (`app/usecases/recordings.py:63-72`).
* **The queue is bounded and drops the oldest.** The monitor's own `eval_every_s: 0.5` throttles its
  work, so a stale frame is worth nothing; a laggy fallback is worse than a skipping one.
* **The consumer reads the tail.** A restarted consumer reads the tail of `iris:meta:<cam>` and
  treats a gap as a gap. The monitor's window is 30 s, so a consumer that is back inside 30 s loses
  nothing that matters.

### 5.2 The per-camera worker

```python
for entry in the_camera_queue:
    pose = json.loads(entry.fields["pose"])
    ids    = [it["id"] for it in pose["items"]]
    boxes  = [it["box"] for it in pose["items"]]        # xyxy, capture px
    kps    = [it["kps"] for it in pose["items"]]         # 17x3, capture px
    candidates = monitor.update(entry.pts, ids, boxes, kps)
    for c in candidates:
        route(scoring.score(c, policy))
```

`FallMonitor` is constructed per camera with `img_shape=(shape[1], shape[0])` from the frame's own
`pose.shape`, and with the classifier the boot contract built.

### 5.3 The routes

| The combined score | The evaluation | The alert | The severity | The review |
|---|---|---|---|---|
| at or above `confirm_threshold` | `falling`, `immobile` | yes, immediately | `critical` | none — auto-confirmed |
| at or above `confirm_threshold` | `lying` | yes, immediately | `warning` | none — auto-confirmed |
| at or above `confirm_threshold` | `upright` | **no** | — | none — a confirmed false alarm, closed |
| below `confirm_threshold` | any | yes, immediately | `warning` | queued for a human |

The alert is the platform's contract, used as it is:

```python
services-equivalent:  XADD iris:alert:<cam>
    v=1  kind=fall  severity=critical|warning  category=safety
    message="falling 0.93"   pts=<the sidecar's frame clock>
```

The field set is exactly `services.alert()`'s (`perception/engine/modules.py:99-111`): `v`, `kind`,
`severity`, `category`, `message`, `zone`, `pts`, `count` + extras. The app's reader
(`app/usecases/module_alerts.py:41-54`) turns it into an event with no code of the module's own, and
the severity is one of the app's closed sets (`app/severity.py:8-9`).

The clip is requested on the same Redis contract: `XADD iris:svr_request:<cam> {pts}` — the sidecar
answers it with the clip centred on that frame (`perception/engine/main.py:258-289`).

### 5.4 The review queue

A candidate below the threshold goes to a human, with the three statuses the vendor's own operator
API already defines — `acknowledged`, `confirmed`, `dismissed`
(`F:\fall_detection_module\src\fall_monitor.py:511-522`), and the operator's note.

The vendor keeps its events **in memory** (`FallMonitor.events`), which does not survive a restart,
so the product persists the queue in **SQLite** — the backend both other products use (no
`psycopg` anywhere in the repo today). A new table in the fall product's `schema.sql`, and a
`storage.py` written the way the barber's was: `connect()` kept from the barber's (which is itself
the attendance's — busy timeout, WAL, `executescript(schema.sql)`, `_migrate`,
`products/barber/storage.py:27-38`), everything else the fall tables. The barber has no review table
of its own, so the fall product's is new; it is not a copy of one.

**The candidate's own fields are stored with the review**: the evaluation, the combined score, the
per-path scores, the reading's top class and the clip's time range. An operator's decision is only
useful next to the evidence that produced it.

---

## 6. C5 — the scoring policy

**Contract.** `products/fall_detection/scoring.py` + the `scoring:` block of
`products/fall_detection/fall_detection.yaml`. Pure functions, no Redis, no GPU, no I/O — so the
repo's own test suite covers the maths.

### 6.1 The configuration

```yaml
# products/fall_detection/fall_detection.yaml
scoring:
  confirm_threshold: 0.70      # the confidence in the evaluation needed to skip the human review
  margin_span: 0.80            # the classifier's decision margin that counts as fully decisive
  trust:
    action:    1.00            # the classifier's reading of the state - the evaluation itself
    kinematic: 0.50            # the collapse kinematics corroborating the reading
    cold:      0.40            # the same collapse kinematics with no reading to corroborate them
  evaluations:                 # the state the system concludes, and the classes that support it
    falling:  [6]              # Fall
    lying:    [3]              # Lying
    immobile: [3, 2]           # Lying or Sitting - the immobility detector's own reading
    unstable: [7]              # Instability
    upright:  [0, 1, 2, 4, 5]  # Standing, Walking, Sitting, StandUp, SitDown
```

The class order is the vendor's (`action_classes.names`: `Standing, Walking, Sitting, Lying,
StandUp, SitDown, Fall, Instability`). The class **sets** are the product's policy and live here,
not in the vendor's copy — one place, and it is the place a reviewer looks.

### 6.2 The evaluation, the reading and the margin

**The evaluation** of a candidate is decided by **its detector first, the classifier second**:

| The candidate | The evaluation |
|---|---|
| `fall` | the group its reading's top class falls into |
| `immobility` | `immobile` — unless the reading's top class is a decisive `Fall` (6), because a person who is falling is not immobile |

*The refinement:* the plan had the monitor compute the evaluation. The **service** computes it,
because the evaluation is a lookup in the product's class sets, and a second copy of those sets
inside the vendor's file would drift. The monitor hands over the raw reading; the service turns it
into an evaluation.

**The reading's margin** — how decisively the classifier reached that evaluation:

```python
support = max(p[c] for c in evaluations[verdict])
oppose  = max(p[c] for c in range(len(p)) if c not in evaluations[verdict])
margin  = support - oppose
reading = clip(margin / margin_span, 0, 1)
```

**The margin is the point of the change.** A classifier that cannot tell "Fall" from "Lying"
(`p[6] = 0.45` against `p[3] = 0.42`) has a margin of `0.03`: the system knows that it does not
know, and the candidate goes to a human. A decisive reading (`p[6] = 0.95` against `p[3] = 0.02`) is
worth the full weight, and the event is confirmed without anyone looking at it.

### 6.3 The combination

```python
combined = 1.0 - prod([1.0 - trust[k] * s for k, s in path_scores.items() if s is not None])
```

**Why this and not the trust-weighted mean.** A weighted mean renormalizes over the paths that spoke,
so a single path's score always comes out at full strength: the cold-start path would have been able
to confirm a candidate by itself, and the action path's own weight would have been arithmetic noise.
The product form keeps every weight meaningful and states the ceilings in the config itself:

* the **action path's ceiling is 1.00** — a decisive reading confirms on its own;
* the **kinematic path's ceiling is 0.50** — the physics alone, corroborated by a reading, is never
  enough on its own;
* the **cold path's ceiling is 0.40** — a candidate with **no reading** (no window yet) can never
  reach the 0.70 threshold, so it is always delegated. That rule is the config, not a special case
  in the code.

Nothing is renormalized over the paths that spoke: a path that did not speak leaves its trust
**unclaimed**.

### 6.4 The three paths, and how a candidate's scores are filled

| Path | The score | The vendor's route | When it speaks |
|---|---|---|---|
| `action` | the **reading's margin** | Path A (`fall_monitor.py:740-748`) | the classifier's top class is `Fall`, with a full window |
| `kinematic` | the **collapse margin** | Path B (`fall_monitor.py:749-763`) | the kinematics collapsed, with the classifier's support |
| `cold` | the **collapse margin** | Path C (`fall_monitor.py:765-769`) | the kinematics collapsed, with **no** window |

The action path is the only one that measures the **evaluation**; the physical paths measure the
**event**. Both belong in the score, because a confident reading of a weak event and a strong event
with an undecided reading are different failures.

**The immobility candidate** uses the same three weights, with the stillness margin as its physical
score: the reading (trust 1.00) and the stillness (trust 0.50) when a window exists, and the
stillness alone (trust 0.40) when it does not — so an immobility candidate with no reading is also
always delegated.

### 6.5 The vendor's `lying` candidates

`detect_lying` (`fall_monitor.py:303-319`) fires after `T_lying = 2.0` s of horizontal posture with
low motion, so it fires on anyone lying on a sofa, and it double-reports a fall's aftermath (the
same person produces a `fall` candidate **and** a `lying` candidate). The first release **alerts**
them, as it alerts the vendor's `fall` and `immobility` candidates (decision 3, §13.3): all three
route through the same policy. The double report is what the aftermath suppression is for — a
`lying` candidate on the same track within `suppress_after_fall_s` of an alerted fall is scored,
but neither alerted nor queued (§13.3).

---

## 7. The data contracts

Three Redis contracts, all existing:

| Key | Written by | Read by | Shape |
|---|---|---|---|
| `iris:meta:<cam>` | the sidecar (the publisher) | the fall service | the frame's fields + `pose` (JSON) |
| `iris:alert:<cam>` | the fall service | the app | `v, kind, severity, category, message, zone, pts, count` |
| `iris:svr_request:<cam>` | the fall service | the sidecar | `{pts}` |

And one key the next release adds:

| Key | Written by | Read by | Shape |
|---|---|---|---|
| `iris:fall_state:<cam>` | the fall service | the fall sidecar's overlay module | `{"<track id>": {"state", "confidence", "confirmed", "ts"}}` |

The meta stream's `maxlen` is the engine's own (64, approximate). A consumer that restarts reads the
tail and treats a gap as a gap.

---

## 8. The boot contract — the classifier is mandatory

The product's start-up is a **gate, not a warning**:

```
1. read products/fall_detection/fall_detection.yaml   -> classifier.device (cuda:0), weights_dir
2. verify both checkpoints against WEIGHT_SHA256_REGISTRY      -> mismatch = refuse to start
3. torch.cuda.is_available() AND a real CUDA op on the device  -> no GPU    = refuse to start
4. build TSSTGClassifier(device="cuda:0") and hand it to FallMonitor(classifier=...)
5. only then: start the Redis consumer, the per-camera workers, the HTTP API
```

Three failures the vendor's own code would have swallowed, turned here into a **non-zero exit** with
a message naming the cause:

* a missing checkpoint (the vendor's `except Exception` → `classifier = None`);
* a corrupt checkpoint (same);
* a CUDA wheel without this GPU's kernels — it loads, reports a healthy CUDA, and fails at the
  first op, so the check runs **a real CUDA op**, not `cuda.is_available()`.

The service's `/status` reports the same facts (`device`, `weights_verified`, `classifier_ready`,
which paths are live), so an operator sees the classifier's state instead of assuming it.

---

## 9. The deployment

**The fall product owns its sidecar.** `products/fall_detection/sidecar.yaml` extends the shared
file (which keeps `modules: {}`) and adds:

```yaml
modules:
  pose:
    config: /workspace/iris/perception/models/DeepStream-Yolo-Pose/config_sgie_pose.txt
    interval: 2
sources:
  - id: cam-1
    uri: ...
    roles: [fall_detection]
```

* **No other product's `sidecar.yaml` changes.** A guard test asserted that
  `products/attendance/sidecar.yaml`, `products/barber/sidecar.yaml` and
  `perception/pipeline/sidecar.yaml` are byte-identical after the change; the test was removed
  with the branch's stale-test sweep, so the claim is a hand-run `git diff main` now.
* **The config file declares the same interval.** `config_sgie_pose.txt` carries
  `secondary-reinfer-interval=2`, the number nvinfer actually reads; the module's `interval: 2` is
  the declaration, and the module refuses to start when the two disagree (§3).
* **The compose service** `deepstream-sidecar-fall` (profile `fall`, telemetry 9105 — free) is
  modelled on the barber's (`docker-compose.yml:356-383`): `IRIS_SIDECAR_ROLE=fall_detection` and
  `${FALL_SIDECAR_CONFIG:-products/fall_detection/sidecar.yaml}`.
* **The `just` recipes** `fall-up` / `fall-down` / `fall-log` are modelled on the barber's
  (`justfile:677-737`). `fall-up` runs the preflight (`products/fall_detection/preflight.py`),
  then the service — its boot gate refuses in seconds — then the sidecar, with a bounded wait
  for `pipeline PLAYING` that exits non-zero naming the compose log. The vendor's checkpoints
  have no public download this repo has verified; the shipped ones are **committed** under
  `products/fall_detection/weights/` (a team-lead decision), and the boot gate verifies
  them (§8).
* **The exclusivity guard.** The fall deployment **replaces** the attendance deployment on its
  cameras. The `roles` tag names the recording's owner but does **not** exclude a second process
  (`perception/engine/sources.py:52-74`), so the enforcement is the **preflight in `just fall-up`**:
  it refuses to start beside a live attendance sidecar, and names the sidecar it found.
* **The golden config** `tests/configs/golden-fall_detection.yaml` + its baseline — the repo's rule
  for a new product (`perception/ARCHITECTURE.md:422`, `perception/modules/README.md:120`).

---

## 10. C8 — the next release: the live annotation overlay

**What was asked for** (2026-10-26): the live camera in the UI, annotated with each person's
bounding box, and the person's **condition** (falling / lying / immobile) popping up on his box when
the fall service detects a critical state in real time — with the person tracked as he moves.

**Option A — the engine draws the condition (the simple one, and the recommendation).** The fall
service publishes the per-track state to `iris:fall_state:<cam>`; a small module in the fall sidecar
reads it and annotates the person's box. The condition is then **baked into the JPEG the platform
already serves**, so it shows on every view of that camera — the dashboard wall included — with **no
frontend work at all**.

```
 the fall service (host process)
      |  SET iris:fall_state:<cam>  {"12": {"state": "falling", "confidence": 0.93,
      |                                     "confirmed": true, "ts": 1761234567.9}}
      v
 the fall sidecar's module (the engine process)
      |  the engine's control poller: one GET per second, on its own thread
      v
 out.annotate(person, color=magenta, priority=12, text="FALLING 0.93 #12")
      |
      v
 the published JPEG -> GET /stream/<cam>/frame -> the dashboard wall
```

**Everything it needs already exists:**

| The piece | Where |
|---|---|
| the person boxes are already drawn on the published video | `perception/engine/publisher.py:48-70` |
| a module can relabel and recolour a box | `FrameOutput.annotate(obj, color=, text=, priority=)` (`perception/engine/frame.py:126-136`) |
| the label is drawn just above the box's top edge | `publisher.py:63-70` — "the condition pops up on his bounding box" is the label slot |
| the tracker id is on the box (`#<oid>`) | `publisher.py:56-70`, `perception/engine/objects.py:25` |
| a module can be told about a Redis key | `Module.control()` → `{key: handler}` (`perception/engine/modules.py:176-178`), polled by `ControlPoller` on its own thread (`perception/engine/control.py`) |
| the live frame is already served | `GET /stream/{source_id}/frame` (`app/server.py:847-875`) |

**Why `control()` and not a `redis.get` in the hook.** A hook must never block
(`perception/modules/README.md:45`), and the shared Redis client has no socket timeouts (the
precedent that gives the face module a private client,
`perception/modules/face_detect/__init__.py:148-159`). The engine's control poller fetches the key
on its own thread and calls the handler only when the value changed
(`perception/engine/control.py:38-42`) — the sanctioned "external state → module" mechanism, with
no new plumbing.

**The state, per track:**

* **set** when the service scores a candidate: `state` = the evaluation, `confidence` = the combined
  score, `confirmed` = whether it reached the threshold;
* **held** for `state_hold_s` (default 10 s) from the last candidate for that track; a new candidate
  refreshes it;
* **cleared** when the track is gone — the monitor knows the live tracks from the meta stream's
  `pose` field — and **by age**, so a condition never hangs in space after the person leaves;
* an `upright` evaluation (a confirmed false alarm) draws **no** condition: the person is fine, and
  that is the point of the score.

**The tracking.** The condition follows the person because the whole chain is keyed on the
**DeepStream tracker id**: the pose field's `id` → the service's per-camera `FallMonitor` (whose
state is keyed by the track id) → the state map → the annotation. The engine's baked-in box prints
the same id (`#<oid>`), so the condition and the box can be matched by eye.

**The colour.** **Magenta at priority 12** — the colour `perception/NEXT_MODULES.md:104-111`
already reserves for a fall. It beats the zone colour at 1 and the fire colour at 10. One colour for
all three states: the annotation's *text* carries the state, so no new colour is invented.

**What it costs.** One Redis key, one small module, **zero frontend**.

**Option B — a DOM overlay on the fall product's own page (the richer alternative).** The service
publishes a scene over **its own** WebSocket and the fall product's page draws a chip over the
camera. It buys a chip the DOM styles (a confidence bar, a hover card with the track's history) and
a live page that belongs to the fall product. It costs a new WebSocket, a new page, an overlay
canvas, and the coordinate contract below — a permanent alignment problem between the chip and the
box that is already baked into the JPEG. The platform's own `/ws` is **not** an option: its message
union is switched exhaustively (`frontend/src/products/iris/types/ws.ts:20-27`,
`useWebSocket.ts:88-91`), so a scene variant means changing the platform's server, and `app/**` is
on the not-touched list. The barber solved this by serving its own `/barber/ws`; the fall product
would serve `/fall_detection/ws`.

**The latency, stated plainly.** The chip is not instantaneous, and it cannot be. The action path
reads a **2.5 s window** and the monitor evaluates every `0.5 s`, so a `falling` verdict lands
roughly **3 s after the fall begins**. The kinematic and cold paths can fire sooner — they need no
window. `lying` and `immobile` are slower by nature (`T_lying: 2.0 s`). The alert path carries the
same latency, and the clip window (`anchor ± 5 s`) is sized for it.

**Option B's coordinate contract** (for the record; Option A does not depend on it): the boxes the
modules receive are in **mux** coordinates and the publisher scales them into the **capture** frame
(`publisher.py:44-48`). A chip positioned from the pose field (capture px) against a capture-sized
JPEG needs no scaling at all — the pose field's coordinate contract (§3.2) is what makes Option B
possible later, at no extra cost.

---

## 11. How the whole thing is proven

| Component | The focus | The proof |
|---|---|---|
| the export | three tensors — boxes `[batch,N,4]`, scores `[batch,N]`, keypoints `[batch,N,17,3]` — in network-input pixels (§2's table) | numeric equality with `model.predict()` on a real image: every detection, every confidence, 2 px; and the tracked graph's own three names and shapes (`products/fall_detection/tests/test_export_pose.py`) |
| the parser | the face parser's style (threshold + NMS in the parser, payload in the instance-mask field) | `nm -D` shows `NvDsInferParseYoloPose`; the parser is fed a tensor whose contents are known (`tests/test_pose_parser.py` compiles it against a stub of DeepStream's header and runs it on a host, in both the per-crop and the batched shape) |
| the module | the **crop → capture mapping** | a unit test against a known crop, plus the keypoints landing on the joints in the live preview |
| the module | the GPU cost | `tools/engine_gpu_budget.sh` before/after |
| the product | the fall sidecar is the fall product's **own** | the other products' sidecar yamls are byte-identical after the change (the asserting guard test was removed with the branch's stale-test sweep; the proof is a hand-run `git diff main` at merge) |
| the product | the fall deployment must not run beside a live attendance sidecar | `just fall-up` refuses, naming the running sidecar |
| the ACL | the timeline is the sidecar's **`pts`** — the same clock the clip request is centred on | the consumer passes each frame's `pts` through unchanged and the worker drops an entry whose `pts` is not a number (`tests/test_worker.py`); the entry id is **not** a clock: it orders the stream, and nothing in the product reads it |
| the monitor | **one instance per camera** | a unit test: two cameras, the same track id, no cross-talk |
| the scoring | the **margin** (a classifier that ties Fall against Lying must score near zero) and each path's **ceiling** | a test that a Fall-vs-Lying tie lands below the threshold; a test that a cold-start candidate can never reach it |
| the classifier | it must **fail loudly** | a corrupted `.pth`, a missing `.pth`, and `device: cuda:0` on a GPU-less box — all three refuse to start, naming the cause |
| the action path | **our own accuracy**, not the vendor's | the fall corpus replayed through our own pose stream: precision/recall and the false alarms per hour |
| the consumer | the two XADDs in order, `iris:svr_request` with the **sidecar's** frame clock | an integration test against a live sidecar: the SVR session opens over the fall |
| the whole | the decision-to-alert latency under ~5 s (`pad_seconds: 5.0`) | measured end to end |

The corpus is at `F:\fall_detection\data` (290 videos, the manifests, the OmniFall annotations). The
acceptance harness takes its root as a parameter and **skips when it is absent**, so the IRIS test
suite stays runnable without it — but the accuracy numbers are only produced when it is there.

---

## 12. Out of scope

* **Running two products at once on the same cameras.** The fall deployment replaces the attendance
  deployment on its cameras; the preflight guard enforces it.
* **Fine-tuning the ST-GCN.** The vendor ships a transfer gate for adapting the checkpoint to a new
  domain. The first release uses the published weights as they are.
* **An automatic CPU fallback for the classifier.** The configured device must work, or the product
  refuses to start. (`classifier.device` is a config value, so a GPU-less dev box can still run the
  product on `cpu`; the **shipped** configuration is `cuda:0`.)
* `pipeline_preview: false` (it would unlock the sampling gate but changes the dashboard).
* A live skeleton overlay on the attendance tiles.
* **The keypoints or the skeleton in the UI** — the overlay draws the condition on the box; the 17
  keypoints stay in the service.
* **A DOM overlay on the platform's dashboard wall** — the condition is baked into the published JPEG
  by the engine, so the wall shows it for free; a separate DOM overlay there would mean the platform
  consuming the fall service's state, which crosses `app/**`.
* Any change to `app/`.
* Any change to `perception/engine/`.

---

## 13. The open questions for the user, and the answers (2026-10-27)

The four questions are answered. This section records each answer and what it changes. **Where this
section contradicts an earlier one, this section wins** — §§1–12 are untouched, so §6.5's closing
sentence ("the first release alerts the vendor's `fall` and `immobility` candidates") is superseded
by question 3 below.

| # | The question | The answer |
|---|---|---|
| 1 | the pilot's camera set | **`cam-1` only** |
| 2 | the confirm threshold's first value | **`0.70`**, kept configurable and measured |
| 3 | does `lying` alert from day one | **yes** — three alerted kinds from day one, with the fall's aftermath suppressed |
| 4 | the review queue's operator surface | **still open** — a frontend decision that does not block the service |

### 13.1 The pilot's camera set — **decided: `cam-1` only**

One camera. The GPU is the top risk, and one camera is the only way to attribute a regression to the
pose GIE. The trade-off stands: the false-alarm rate per hour is only meaningful over hours of
footage, so a longer pilot on one camera beats a shorter one on three.

### 13.2 The confirm threshold's first value — **decided: `0.70`**

Accepted. The value stays **configurable and measured** — it is a guess until the corpus is
replayed, and the replay (N4) is what turns it into a number. `confirm_threshold: 0.70` in
`products/fall_detection/fall_detection.yaml` is the shipped default; changing it is one line in
the product's own yaml and no other file moves.

### 13.3 The `lying` evaluation's alert — **decided: `lying` alerts from day one**

Three candidate kinds alert from day one: the vendor's **`fall`**, **`lying`** and **`immobility`**
candidates. The vendor's `lying` candidate (`detect_lying`, `fall_monitor.py:303-319`) is no longer
an evaluation only — it becomes an event of its own, and it rides the **same route table** (§5.3)
every other candidate rides.

**How a `lying` candidate is scored.** The same three paths and the same threshold as every other
candidate, so no new trust value and no new route:

* the **evaluation** is `lying` — the class set `[3]` (§6.1's `class_sets`);
* the **reading** (the `action` path, trust 1.00) is the classifier's margin when the window
  exists; a window that reads something else leaves the candidate's reading at zero;
* the **physical** score is the posture and the stillness against the vendor's own `lying`
  thresholds (`config.yaml`: `lying {angle_in 35.0, aspect_in 0.5, max_motion 1.0}`) — the same
  shape as the collapse margin:

```
s_posture = clip(min((angle - angle_in) / angle_span, (aspect - aspect_in) / aspect_span), 0, 1)
s_still   = clip((max_motion - motion) / motion_span, 0, 1)
s_lying   = (s_posture · s_still) ^ 0.5
```

  It is trusted at the **kinematic** ceiling (0.50) when a window exists and at the **cold**
  ceiling (0.40) when it does not — the same two weights the other candidates use, so the trust
  table is unchanged. The three spans (`angle_span`, `aspect_span`, `motion_span`) sit beside the
  thresholds they measure, in the copied `products/fall_detection/config.yaml` — one source of
  truth per quantity, as everywhere else.

**The double report — decided: the aftermath is suppressed.** A fall's aftermath produces a
`lying` candidate for the same track a couple of seconds later (the person is horizontal and still
for `T_lying = 2.0 s`). That candidate is **not alerted and not queued**: it is suppressed while
the track has an alerted `fall` within `suppress_after_fall_s` (default 10 s, beside
`confirm_threshold` in the product's own yaml). It still **scores**, and it still **annotates the
box** — the operator still sees that the person is down — but it does not become a second event.
The suppression is per track and per fall event, not a mute on the kind: a lying event that comes
later than the window (a person who stays down, or a genuine sofa) alerts normally.

### 13.4 The frontend tab — **decided: a dedicated tab, drawing the engine-baked JPEG**

The fall product gets **its own tab** in the platform's frontend — a page registered in the
products registry, showing the fall product's own view of the camera. The tab draws the
**engine-baked JPEG** (Option A of §10): `GET /stream/<cam>/frame`, the same frame the dashboard
wall already shows, with the condition already drawn on the person's box by the fall sidecar's
module. No new Redis contract, no WebSocket, no box math, and no change to `app/**` or
`perception/engine/**`.

The cost, stated plainly: the chip is baked into the JPEG, so the tab cannot show a confidence
bar, a hover card, or a per-track history without the page knowing the box's geometry. §13.5 is how
that is bought later.

### 13.5 The WebSocket option is extendible from this design — **decided: it lands later, on top**

Option B of §10 (the fall service's own scene WebSocket, `/fall_detection/ws`, and a DOM chip drawn
over the JPEG) is **not** built now, and it needs **no change to §§1–12**. It lands later on the
same state, in two later slices:

```
now        the tab + the baked JPEG          (Option A's set, §10)
B1         the service publishes the scene    (later)
B2         the tab's chips over the scene     (later, optional)
```

**Why it fits on top of what is already decided.** The current design already fixes the four things
Option B needs, and none of them has to change:

1. **the coordinates** — the `pose` field's box and keypoints are in **capture pixels** (F4, §3.2),
   the same space as the JPEG the tab draws. A chip positioned from the pose field against a
   capture-sized JPEG needs no scaling at all.
2. **the state** — the next release's service publishes the per-track evaluation, the confidence
   and the timestamp to `iris:fall_state:<cam>` (§10). Option B publishes the **same map** as its
   scene, so the overlay and the engine's baked label read one source of truth.
3. **the key** — everything is keyed by the **DeepStream tracker id**, so a chip and a box are the
   same track by id and not by proximity.
4. **the endpoint** — the platform's `/ws` message union is closed
   (`frontend/src/products/iris/types/ws.ts:20-27`) and `app/**` is not touched, so the fall
   product serves **its own** WebSocket, the way the barber serves `/barber/ws` (§10).

**What Option B adds when it lands:** one WebSocket endpoint on the fall product, the scene
contract (per track: the id, the box in capture px, the evaluation, the confidence, the ts) and
the chip layer in the tab's DOM. **What it does not touch:** the engine, the app, the pose module,
the monitor, the scoring policy, `iris:fall_state`, and the tab's page — the JPEG layer and the
chip layer are two separate layers in the page, so the chip lands on a page that already works.

**Its price, deferred and not avoided:** a permanent alignment problem between the chip and the
box that is already baked into the JPEG (~80–100 ms of skew), and a second transport to keep alive.

### 13.6 The one question still open — the review queue's operator surface

The queue exists (F22) and the API exists; the open question is **where the operator works the
queue**. The recommendation: in the **same tab** (a list, with acknowledge / confirm / dismiss) —
the operator is already there for the live view, and the two surfaces belong to one product. The
trade-off: a separate page keeps the live view and the queue from crowding each other, and a
read-only list inside the existing dashboard avoids a new page entirely, but it puts the fall
product's operator surface inside another product's tab. It is a frontend decision and it **does
not block the service**: the queue, the three statuses and the API are the same either way.
