# VEMA Tracker — Prototyping & SOTA Benchmarking Suite

Standalone research workspace for developing, ablating, and benchmarking the **VEMA (Velocity-EMA Three-Regime)** multi-object tracking algorithm against SOTA pure-motion trackers (**ByteTrack**, **OC-SORT**, and **SORT**) on public MOT benchmarks (**MOT17**, **DanceTrack**, **MOT20**) and deterministic synthetic stress tests.

## Quick Start

### 1. Run Unit & Synthetic Stress Tests
```bash
pytest vema_tracker/tests/ -v
```

### 2. Compare VEMA vs. SOTA Baselines on Synthetic Occlusion Scenarios
```bash
python -m vema_tracker.eval.runner --dataset synthetic --trackers vema bytetrack ocsort sort
```

### 3. Run Component Ablation Study
```bash
python -m vema_tracker.eval.ablation --dataset synthetic
```

### 4. Benchmark on Public MOT Datasets (MOT17-val / DanceTrack-val / MOT20-val)
Place the dataset under `vema_tracker/data/` (gitignored). Only `seqinfo.ini`, `gt/gt.txt`, and `det/*.txt` are required (no video frames needed):
```text
vema_tracker/data/
  MOT17/
    train/
      MOT17-02-FRCNN/
        seqinfo.ini
        gt/gt.txt
        det/det.txt          # or pass --det-dir path/to/yolox_dets/
  DanceTrack/
    val/
      dancetrack0004/
        seqinfo.ini
        gt/gt.txt
        det/det.txt
```
Inspect or verify your local dataset directory:
```bash
python -m vema_tracker.datasets.fetch_datasets --check
```
Then run the SOTA comparison:
```bash
python -m vema_tracker.eval.runner \
  --dataset mot17-val \
  --trackers vema bytetrack ocsort sort \
  --config vema_tracker/configs/mot17.yaml \
  --out vema_tracker/results/mot17_val.json
```

## Directory Layout
- `core/` — `VEMAConfig`, `VEMATrack` (3-regime state machine), vectorized `costs.py` (IoU, BIoU, Anisotropic $C_{\text{pos}}$, Observation-Centric Direction Cosine), and 3-stage `VEMATracker`.
- `baselines/` — Unified `BaseTracker` interface with self-contained implementations of **ByteTrack**, **OC-SORT**, and **SORT**.
- `datasets/` — MOTChallenge (`MOT17`, `DanceTrack`, `MOT20`) loaders, deterministic synthetic stress suite, and dataset verification CLI.
- `eval/` — Official TrackEval bridge + built-in exact HOTA/AssA/DetA/IDF1/MOTA/IDSW evaluator, benchmark runner, ablation runner, and trajectory plotter.
- `configs/` — Tuned YAML presets (`default.yaml`, `mot17.yaml`, `dancetrack.yaml`, `mot20.yaml`).
- `tests/` — Unit tests for every regime, edge case, and end-to-end synthetic benchmarks.
