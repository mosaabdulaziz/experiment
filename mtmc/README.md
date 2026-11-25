# MTMC — Multi-Camera Multi-Object Vehicle Tracking

A research pipeline for cross-camera vehicle tracking, validated end-to-end
against the **AICity22 Track1 (MTMC Tracking)** ground truth.

```
provided dets ──► single-cam tracking ──► Re-ID embeddings ──► cross-camera association ──► eval (IDF1)
 (no road mask)   (IoU tracklets)         (fast-reid VeRi)      (time gate + cosine sim)     (official eval.py)
```

## Design decisions

- **Provided detections, no road masks.** The pipeline consumes the
  challenge's `det_mask_rcnn.txt` detections directly, so vehicles parked off
  the annotated ROI (and everything else in frame) are tracked like any other
  vehicle — exactly the behavior you asked for. The official eval's ROI filter
  is neutralized with all-white ROI images.
- **SCT** is a greedy IoU tracker (`tracker.py`): no motion model, tuned for
  10 fps surveillance where even parked vehicles have high inter-frame IoU.
- **Re-ID** (`reid.py`) uses fast-reid's **VeRi-trained SBS(R50-ibn)**
  (Rank@1 97% on VeRi) to embed each tracklet's crops (mean-pooled, L2-normed).
- **Association** (`associate.py`) merges tracklets into global identities by
  cosine similarity, gated by wall-clock time overlap. The AICity22 cameras
  are near-synchronous (same square viewed from opposite sides), so the same
  vehicle appears in different cameras at almost the same wall time — **time
  overlap + appearance is the association signal**, with at most one tracklet
  per camera per identity.

## Results on S01 (5 cams, 95 GT IDs, ~3.3 min per camera)

Default pipeline = time gate + Re-ID appearance (homography camera-link model
is opt-in, see below). `sim` is tuned to 0.55; true-pair similarities average
only ~0.54, so a higher bar starves recall.

### Isolating tracking quality from detector noise

Feed the **GT boxes as detections** (`--det-source gt`) to measure the MTMC
core (SCT + Re-ID + association) in isolation — with perfect detections:

| Method | IDF1 | IDP | IDR |
|---|---|---|---|
| **MTMC core, GT-fed (`--det-source gt`)** | **0.61** | 0.71 | 0.54 |
| det-fed (real detections, default) | 0.32 | 0.24 | 0.46 |
| provided tc_mask_rcnn | 0.261 | 0.190 | 0.420 |
| provided deepsort_mask_rcnn | 0.254 | 0.183 | 0.413 |
| provided moana_mask_rcnn | 0.254 | 0.186 | 0.403 |

(There is no "0.85–0.87 GT-fed" anymore — see the camera-link section below;
those numbers were inflated by wrong merges from the homography tier.)

The GT-fed → det-fed drop (0.61 → 0.32) is dominated by **detector noise**:
`det_mask_rcnn` emits 2.3× the GT box count (parked cars, duplicate boxes,
reflections), and every extra box is a false positive in the eval. Raising the
detection confidence threshold (`--conf`) prunes some of it.

## The camera-link model is OPT-IN (`--camera-model`)

`cameramodel.py` learns per-camera-pair homographies from GT bottom-centers and
uses them as a "spatial proof" (two time-overlapping tracklets projecting to
the same ground point are merged unconditionally) plus a spatial gate. It is
**off by default**, because it was measured to do more harm than good on this
camera pair:

- The c001–c002 cameras sit on **opposite sides of the square**, so the same
  car is seen front-on in one camera and back-on in the other. The bounding-box
  **bottom-center is a different physical ground point per camera** (front vs
  rear bumper), so a bottom-center homography is systematically biased by
  vehicle orientation — median 507 px projection error on TRUE same-vehicle
  pairs (only ~30% fit within the RANSAC threshold).
- Consequence on GT-fed c001–c002: the "spatial proof" made **28 merges, 27 of
  which joined DIFFERENT GT vehicles (4% correct)**, while the spatial gate
  rejected 30/58 true pairs before appearance could vote. The earlier
  0.85–0.87 GT-fed / 0.47 det-fed claims were inflated by these wrong merges.
- With the homography off, appearance-only merges are 100% precise where they
  fire, and the honest eval numbers are the 0.61 / 0.32 above.

Keep `--camera-model` available for camera pairs where the homography is
verified tight (e.g. side-by-side views of the same lane), or as a candidate
pruner in front of appearance.

## Visual inspection: side-by-side cross-camera video

To SEE the association with your own eyes — watch a car leave cam 1 and
reappear in cam 2 under the same global id — render an aligned side-by-side
video with our Re-ID (no GT):

```bash
tools/mtmc-research/.venv/bin/python -m mtmc.visualize --scene S01 --cams c001 c002
# -> mtmc/output/side_by_side_S01_c001_c002.mp4  (10 fps, 1920x600, ~3 min)
```

Both panels are aligned by wall-clock time (each camera's `start_offset_s`), so
the same moment in both cameras is side by side. Every tracked vehicle gets a
stable per-global-id color; vehicles whose identity spans BOTH cameras are
drawn thick with a filled "id N" label, single-camera vehicles thin/grey.
The header lists how many vehicles are tracked across both cameras and their ids.

By default the boxes are the challenge's **detector output** (`det_mask_rcnn`,
noisy — parked cars, reflections). To inspect our MTMC core with clean boxes
instead, feed the **GT boxes as detections** while keeping OUR ids, and add
`--gt-labels` to annotate every box with the ground-truth id it matches — an id
miss across cameras then shows up directly (e.g. `id 12 gt 7` in cam 1 next to
`id 33 gt 7` in cam 2 means we failed to merge that vehicle):

```bash
tools/mtmc-research/.venv/bin/python -m mtmc.visualize --mode ours --det-source gt --gt-labels
```

GT ids are ignored for the association; the colored ids are ours. The default
tracklet filter is lenient (min 3 frames / 0.2 s) so few vehicles vanish from
the picture.

To verify the ground truth itself is sane, render a **pure GT video** — GT
boxes colored by GT id, no pipeline at all; vehicles whose id appears in both
cameras are exactly the ones our MTMC should merge:

```bash
tools/mtmc-research/.venv/bin/python -m mtmc.visualize --mode gt
```

## Diagnosing merge failures

`mtmc/diagnose.py` explains, per GT-shared vehicle, exactly why the association
merged or failed to merge its two tracklets (time overlap, projected distance,
Re-ID similarity, and the rejecting gate):

```bash
tools/mtmc-research/.venv/bin/python -m mtmc.diagnose --scene S01 --cams c001 c002
```

## Setup

```bash
# 1. deps, from the lock in tools/mtmc-research (DEPENDENCIES.md): CUDA torch (the sidecar's
#    2.14.1+cu130), ultralytics, transformers, sahi, fast-ReID's helpers, the evaluation tools.
#    pytrec-eval builds from source, so a C++ compiler is needed. The venv is
#    tools/mtmc-research/.venv; the commands below run from the repo root.
uv sync --locked --project tools/mtmc-research

# 2. Re-ID checkpoint (VeRi SBS R50-ibn, ~198 MB) — already at
#    helper-repos/multi-camera-tracking/fast-reid/checkpoints/veri_sbs_R50-ibn.pth
#    or set REID_CHECKPOINT to wherever you keep it.
```

Paths are configured in `mtmc/config.py` (override with `AICITY22_ROOT`,
`FASTREID_ROOT`, `REID_CHECKPOINT`, `MTMC_OUTPUT_DIR`).

## Running

```bash
# Default: global sim=0.55 for all pairs, time + appearance only
tools/mtmc-research/.venv/bin/python -m mtmc.run_pipeline --scene S01

# GT-fed: isolate MTMC quality from detector noise
tools/mtmc-research/.venv/bin/python -m mtmc.run_pipeline --scene S01 --det-source gt

# Camera graph: per-edge association gates (recommended for production)
tools/mtmc-research/.venv/bin/python -m mtmc.run_pipeline --scene S01 --cam-graph mtmc/examples/cam_graph.yaml

# CLI sim override applies to all edges (overrides YAML defaults)
tools/mtmc-research/.venv/bin/python -m mtmc.run_pipeline --scene S01 --sim 0.80

# Opt-in homography for all pairs (use with caution — measured unreliable
# on opposite-view pairs like c001↔c002)
tools/mtmc-research/.venv/bin/python -m mtmc.run_pipeline --scene S01 --camera-model
```

### Camera graph (`--cam-graph`)

The camera graph lets you configure each camera pair independently — timing
mode, similarity threshold, camera-model opt-in, spatial gates — instead of
a single global `--sim 0.55` for all pairs.  This is the recommended way to
configure the pipeline for production.

```bash
# Use the S01 example graph
tools/mtmc-research/.venv/bin/python -m mtmc.run_pipeline --cam-graph mtmc/examples/cam_graph.yaml

# Same for the visualizer and diagnostic tools
tools/mtmc-research/.venv/bin/python -m mtmc.visualize --cam-graph mtmc/examples/cam_graph.yaml
tools/mtmc-research/.venv/bin/python -m mtmc.diagnose  --cam-graph mtmc/examples/cam_graph.yaml
```

The YAML schema (see `mtmc/examples/cam_graph.yaml` for a full annotated
example):

```yaml
defaults:                           # fallback for every edge not listed
  time_tolerance_s: 1.5
  timing: near_synchronous          # near_synchronous | sequential | either
  reid_sim: 0.55
  use_camera_model: false
  spatial_gate_px: 400.0
  spatial_merge_px: 120.0

cameras:                            # optional per-camera metadata
  c001:
    fps: 10.0

edges:                              # per-pair overrides (order-independent)
  - pair: [c001, c002]              # use defaults as-is
  - pair: [c001, c003]              # same
  - pair: [c001, c004]              # override specific fields
    timing: near_synchronous
    time_tolerance_s: 2.0
    reid_sim: 0.50
    use_camera_model: false
```

Key per-edge fields:

| Field | What it controls |
|---|---|
| `timing` | `near_synchronous` (overlap), `sequential` (transit), `either` (appearance only) |
| `time_tolerance_s` | Max gap between tracklets to allow merging |
| `reid_sim` | Cosine similarity threshold (true-pair sims average ~0.54 on VeRi) |
| `use_camera_model` | Enable homography spatial proof for this pair only |
| `spatial_merge_px` | Auto-merge below this projected distance (spatial proof) |
| `spatial_gate_px` | Hard reject above this projected distance |
| `max_transit_s` | Sequential-only: max mid→mid transit time (seconds) |

CLI flags (`--sim`, `--time-tol`, etc.) override the YAML defaults when
explicitly passed.

Outputs land in `mtmc/output/`:
- `prediction.txt` — global-ID boxes in eval format
- `ground_truth_S01.txt` — GT subset for the scene
- `embeddings_*.npz` — cached tracklet embeddings (re-runs that keep the same
  tracking params skip the ~5 min embedding stage)
- `report.json` — params + metrics

## Structure

| File | Role |
|---|---|
| `camgraph.py` | per-edge camera graph loader + EdgeConfig |
| `config.py` | paths + pipeline parameters |
| `data.py` | load dets / GT / cam timing |
| `tracker.py` | single-camera IoU tracking → tracklets |
| `reid.py` | fast-reid feature extraction + tracklet pooling |
| `associate.py` | cross-camera association (time + appearance) → global IDs |
| `cameramodel.py` | opt-in homography/transition-window gates |
| `evaluate.py` | prediction serialization + official eval (white ROIs) |
| `run_pipeline.py` | end-to-end CLI |
| `visualize.py` | side-by-side inspection videos (ours / pure GT modes) |
| `diagnose.py` | per-vehicle merge-failure explanations |

## Next levers (in rough order of expected gain)

1. **Better Re-ID model** — the GT-fed ceiling is 0.61, and the reason is
   weak appearance: true cross-cam pair similarities average ~0.54, barely
   above false pairs (~0.36). The VeRi-trained model transfers poorly to this
   intersection. A vehicle-specific Re-ID (train fast-reid on VeRi-776 /
   VehicleID, or TransReID) is the single highest-value upgrade for MTMC.
2. **Cleaner detections (det-fed path)** — the det-fed 0.32 vs GT-fed 0.61 gap
   is 2.3× box over-production from `det_mask_rcnn`; NMS over tracked boxes,
   higher `--conf`, or a better detector closes most of it.
3. **Same-camera stitching, done right** — measured on S01 it over-merges and
   hurts; needs stricter spatial overlap gating or a motion model to be viable.
4. **Use `calibration.txt` directly** — the provided per-camera homographies
   map to local ground planes (not a shared frame); aligning them once could
   give the spatial proof a better starting point than GT-fitted bottom-centers.
5. **Different det source / scenes** — swap in the pipeline's own YOLO
   detections (`--det`), or run on scenes S03/S02/S05.

## Re-ID benchmark on your own footage

For your own multi-camera clips (`products/el-mohandes/videos/`), validate whether the
appearance-embedding matcher is good enough before trusting cross-camera
merges.  `benchmark_reid.py` scores any model on the vehicle crops
(`extract_vehicle_crops.py` output) against hand-curated same-vehicle labels
(`output/vehicle_crops/chosen crops/vehicle_*` — one folder per physical
vehicle, crops copied from each camera it appeared on):

- **retrieval**: Rank@1 / Rank@2 ("within 2 best") / mAP — can the right
  vehicle be found among the other cameras' candidates?
- **verification / false positives**: similarity distribution of true vs false
  pairs, precision/recall/FPR over a threshold sweep, FP count at the
  production 0.55 threshold, plus FP picture sheets.
- **efficiency**: crops/s, ms/crop, VRAM, params/size per model.

```bash
# models: fastreid-veri-sbs-r50-ibn, resnet50-imagenet, clip-vit-b32,
#         dinov2-base, hsv-hist (control).  transformers (CLIP/DINOv2) is in the lock.
tools/mtmc-research/.venv/bin/python -m mtmc.benchmark_reid --models resnet50-imagenet fastreid-veri-sbs-r50-ibn
tools/mtmc-research/.venv/bin/python -m mtmc.benchmark_reid --report-only   # re-render tables from saved JSONs
```

Results land in `output/reid_bench/` (`REPORT.md` tables, `result_*.json`
per model, `ANALYSIS.md` findings, `fp_montages/<model>/` false-positive
sheets).  Ground truth assembly + sampling: `reid_bench_data.py`; model
registry: `reid_models.py`.  Caveat: unknown vehicles are treated as
negatives although some may be unlabeled appearances of the curated ones, so
metrics are pessimistic — grow `chosen crops/` (or auto-pair by plate) before
acting on small deltas.

## Models and datasets

Weights (`mtmc/models/**`), training sets (`datasets/`) and the multicam sample video (`products/el-mohandes/videos/`) are versioned
with DVC, not git: `just data-pull` for the models, `just data-pull datasets samples-multicam` for the rest (see
[DATA.md](../DATA.md)). The `finetune_*` scripts read `datasets/<name>/data.yaml`; the plate OCR sets are under
`products/el-mohandes/datasets/plate_ocr/`. The Arabic plate OCR model is `models/v5_arabic_mobile-fine-tuned` (training record and the
resume checkpoint: `just data-pull training-artifacts`); the Kaggle notebook that trained it is
`training/notebooks/PP-OCRv5_arabic_mobile_direct_Kaggle.ipynb`. The pipeline report and the Neurocle evaluations are
documents, not data: they are in the team folder `PhyAI/IRIS/repo-data/docs/`.
