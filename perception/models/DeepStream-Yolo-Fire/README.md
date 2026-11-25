# perception/models/DeepStream-Yolo-Fire

The fire/smoke primary detector (`pgie_fire`) run alongside the COCO `pgie` (see
`perception/pipeline/sidecar.yaml`'s `pgie_fire_*` keys and `CLAUDE.md`'s *Architecture* section).

`sidecar.yaml`'s `pgie_fire_config` currently points at `config_infer_fire_box.txt`
(model: `fire_smoke_box.onnx`). The original `config_infer_fire.txt` / `fire_s.onnx`
pair is kept in this directory, untouched, as the prior model and a rollback path —
see "Model history" below.

## Model provenance

### Active: `fire_smoke_box.onnx` (from `fire_smoke_detection_box`)

Exported from `weights/best.pt` in the HF repo
[`weslleyskah/fire_smoke_detection_box`](https://huggingface.co/weslleyskah/fire_smoke_detection_box),
a **YOLO26-nano** fine-tune trained on the
[FASDD_CV](https://huggingface.co/datasets/weslleyskah/fire_smoke_dataset_fasdd_cv)
dataset (100 epochs, imgsz 640). Reported mAP@50 (box) is 0.772. The HF repo tags
the weights `cc-by-4.0`, but see "Licence" below — that tag does not settle the
question the way it looks like it does.

**The parser's actual contract is *not* end-to-end-NMS.** `fire_s.onnx` (below)
is `['batch', 8400, 6]` — every raw anchor, already box-decoded to xyxy and
already class-argmaxed to one (conf, classid) pair, but **not NMS'd**.
`cluster-mode=2` + `nms-iou-threshold` + `pre-cluster-threshold` + `topk` in
`config_infer_fire_box.txt`/`config_infer_fire.txt` are what actually do NMS,
entirely inside DeepStream. `fire_smoke_box.onnx` replicates that exact
contract, built in two steps because Ultralytics has no built-in export mode
that produces it directly:

```
# 1. Raw export: dynamic batch, fixed 640x640 spatial, no NMS/argmax baked in.
#    (dynamic=True also floats height/width to symbolic dims -- pin them back
#    to 640 by hand afterwards, e.g. via onnx's dim_value/dim_param editing,
#    or nvinfer has no fixed resolution to build a profile against.)
YOLO("weights/best.pt").export(format="onnx", imgsz=640, simplify=True, opset=17, dynamic=True, batch=1)
# -> input ['batch',3,640,640], output ['batch',6,8400]: [cx,cy,w,h,score_fire,score_smoke]
# per anchor, scores already sigmoid-activated (verified: raw score at a known
# frame matched confidence from an independent nms=True export almost exactly,
# once preprocessing used a real letterbox resize -- a naive stretch-resize
# alone made confidence look ~3x lower and nearly cost a debugging session
# chasing a phantom "missing sigmoid" bug instead of a preprocessing bug).

# 2. ONNX graph surgery: append decode(cxcywh->xyxy) + per-anchor argmax/max
#    over the class-score channels + transpose to [batch, anchors, 6],
#    matching fire_s.onnx's contract exactly. Both steps are wrapped in
#    export_raw_anchor_onnx.py (this directory) -- reuse it for the next swap:
#    python export_raw_anchor_onnx.py <checkpoint.pt> <output.onnx>
```

Verified against `data/uploads/fire`: `fire` detected in
`bucketfire.mp4` at ~0.6–0.7 confidence, `smoke`/`fire` both detected in
`roomfire.mp4` — and see "The batch>1 trap" below for how this was verified at
the actual mux batch size, not just solo-frame.

### The batch>1 trap — do not export with `nms=True` for this parser

The first version of this model was exported with Ultralytics' `nms=True`
(end-to-end, fixed `[1, 300, 6]` output) rather than the raw-anchor contract
above, on the mistaken belief that this repo's "`[N,6]` end-to-end-NMS"
phrasing meant NMS baked into the ONNX. It does not (see above) — but the
`nms=True` export ran and looked correct for a while regardless, which is
what makes this trap dangerous:

- Exported with a **static** batch dimension (no `dynamic=True`), it worked
  correctly the first time it was engine-built and verified live at
  `nvstreammux` batch=2 (cam-7 + cam-8 together) — right up until the cached
  `.engine` file was deleted and rebuilt from the *same* onnx. The rebuild
  silently returned zero detections for cam-8 (whichever source lands at
  batch index ≥1) while cam-7 (index 0) kept working. Running a model at a
  batch size its ONNX graph declares as a fixed literal (`1`, not a symbolic
  `'batch'` dim) is undefined behavior for TensorRT — `onnxruntime` refuses it
  outright (`Got: 2 Expected: 1`), but `nvinfer` silently built an engine
  for it anyway, and *which* build gives right vs. silently-empty results per
  batch index is apparently not stable across rebuilds. Found live 2026-09-08.
- Re-exporting the same `nms=True` graph with `dynamic=True` doesn't fix it
  either — it produces all-zero confidences even at batch=1, solo. This
  matches a known, unresolved upstream limitation combining Ultralytics'
  end-to-end NMS export with a dynamic batch axis
  ([ultralytics#23339](https://github.com/ultralytics/ultralytics/issues/23339),
  closed not-planned) — a different symptom than what we hit (that issue is
  about YOLOv8 forcing `nms=False`; here YOLO26 accepted the flags and just
  produced silently-wrong output), but the same underlying combination.

**Lesson for the next swap: never export with `nms=True` for this repo's
parser.** Use the raw-anchor + graph-surgery recipe above — it's what
`fire_s.onnx` already does, it composes cleanly with a true dynamic batch
axis, and DeepStream's own clustering does the NMS anyway. Whatever export
path is used, verify correctness **at the real mux batch size** (≥2 sources
sharing this pgie), not just solo-frame — a solo-batch check would not have
caught either failure mode above.

### Model history: `fire_s.onnx` (superseded, kept for rollback)

`fire_s.onnx` is exported from `fire_s.pt`, a YOLOv8 checkpoint published in the
GitHub repo [`AlimTleuliyev/wildfire-detection`](https://github.com/AlimTleuliyev/wildfire-detection)
(`fire-models/fire_s.pt`). It is trained on **D-Fire** — roughly 21.5k images and
26.5k boxes, including ~9.8k hard negatives (sunsets, lamps, clouds, headlights,
and similar fire-colored-but-not-fire scenes) specifically to hold down false
positives on real footage. Reported mAP@50 on the D-Fire test split is ≈0.80 for
this size tier (the repo's own published range across sizes is 0.787 nano →
0.814 XL). This was the only ready-made checkpoint found trained against hard
negatives rather than a small scraped positives-only set — see
`docs/superpowers/specs/2026-09-02-fire-smoke-detection-design.md` for the full
model survey and rejected alternatives that led to it.

To roll back: repoint `sidecar.yaml`'s `pgie_fire_config` at `config_infer_fire.txt`
and restart the sidecar. No file in this directory needs to change to do that —
`config_infer_fire.txt` and `labels.txt` were never touched by the swap.

## Licence — deliberate exception

Both models here are YOLO-derived, so **AGPL-3.0** applies to their fine-tuned
weights regardless of what license tag the uploading repo carries (Ultralytics
licenses the YOLO architecture/training code itself under AGPL-3.0 — including
YOLO26 — and a fine-tune, or a permissive tag applied by whoever uploaded the
weights, does not change that upstream obligation).

This repo otherwise deliberately avoids AGPL — see `CLAUDE.md`'s note on the
primary COCO detector having been moved off YOLO11 (AGPL-3.0) onto DAMO-YOLO-S
(Apache-2.0) for exactly that reason. The fire/smoke detector is a **knowing,
scoped exception** to that posture, accepted for demo purposes only. It is not
a precedent for adding other AGPL models casually.

### Escape hatch

Nothing downstream of `pgie_fire` knows or cares which model produced a given
box — detections flow through the same generic `[N,6]` per-anchor,
already-decoded-but-not-NMS'd contract every primary detector in this repo
uses (see `CLAUDE.md`'s note on the pgie being architecture-swappable).
Moving off the active checkpoint again (e.g. to a permissively-licensed
alternative) is therefore just:

1. A new ONNX export + `config_infer_fire.txt`-shaped config in this
   directory. For a YOLO (v8/11/26/...) checkpoint, use the raw-anchor +
   graph-surgery recipe above (**not** `nms=True` — see "The batch>1 trap")
   and verify the output is `['batch', N, 6]` with a genuinely symbolic batch
   dim, tested at batch ≥2, not just solo-frame.
2. Repoint `sidecar.yaml`'s `pgie_fire_config` at it.
3. No manual engine deletion needed: `perception/engine/engine_path.py`'s
   `gie_engine_path`/`invalidate_stale_engine` derives the engine path from the
   ONNX filename and rebuilds automatically when it's new or the onnx is newer
   than a cached engine. A fresh onnx filename (as used here) always gets a
   fresh engine path.

No perception/pipeline/probe/app code changes required.

## Class order — read before touching a labels file

**Two labels files, two orders — both intentional, do not "reconcile" them
without re-checking the actual checkpoint:**

- `labels_fire_smoke_box.txt` (active): `fire` then `smoke` — checkpoint's
  `names` is `{0: 'fire', 1: 'smoke'}`.
- `labels.txt` (`fire_s.onnx`, history): `smoke` then `fire` — checkpoint's
  `names` is `{0: 'smoke', 1: 'fire'}`.

`nvinfer` populates each detection's class index purely positionally from
whichever labels file its config's `labelfile-path=` names. A labels file's
line order must match its own checkpoint's index order exactly, or every
detection's label silently swaps (a fire renders as "smoke" and vice versa,
with no error anywhere) — this is *why* there are two separate labels files
here instead of one shared `labels.txt`: sharing one across two checkpoints
with different index orders would make one of them wrong. If in doubt,
re-verify against the checkpoint rather than trusting either file's history:

```
python -c "from ultralytics import YOLO; print(YOLO('<checkpoint>.pt').names)"
```

## Config notes

Both `config_infer_fire_box.txt` and `config_infer_fire.txt` deliberately omit
two properties (commented in each file with the same reasoning, kept in sync
here):

- **`model-engine-file`** — set from code (`perception/engine/gst_build.py`, via
  `perception/engine/engine_path.py`'s `gie_engine_path()`), not hardcoded here, so the
  path always matches the name nvinfer's ONNX branch actually derives
  (`<onnx>_b<batch>_gpu<id>_<mode>.engine`, beside the ONNX file). Writing a
  fixed name would go stale against the mux batch size and force a rebuild
  every boot.
- **`engine-create-func-name`** — deliberately absent so nvinfer takes the
  ONNX branch above instead of the custom-engine branch. Adding it would
  derive a `<cwd>`-based engine name that collides with the pgie's own
  custom-engine path, silently deserializing the wrong model into `pgie_fire`
  (see `perception/engine/engine_path.py`'s `assert_distinct_engine_paths`, which
  fails the boot loudly if this is ever reintroduced).
