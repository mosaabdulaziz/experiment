#!/usr/bin/env python3
"""Plate-anchored recall audit for a vehicle detector.

By default it audits the vehicle detector multicam runs: the 1-class
``products/el-mohandes/models/vehicle_1280_1cls/best.onnx`` at 1280, the model
``config_infer_primary_vehicle_1cls.txt`` loads (the same principle as the
finetune defaults: no arguments = what is deployed). The COCO yolo11s it was
first written for stays reachable explicitly:
``--vehicle-model perception/models/DeepStream-Yolo/yolo11s.onnx --vehicle-imgsz 640``.

There is no vehicle ground truth in this repo,
so this script measures the vehicle detector's recall using the plate detector
as an anchor — every detected plate implies a vehicle exists at that spot.

What it reports, per camera and overall:

  * plate-anchored recall — a plate counts as explained when a vehicle box both
    covers the plate centre and is scale-consistent with it (the plate is
    8-60% of the box width; see PLATE_W_RATIO_RANGE).  Containment alone is too
    weak in congested scenes, where 30+ boxes per frame means a background box
    often happens to cover the plate
  * raw containment — the looser upper bound, reported alongside
  * IoU against the plate-expanded box — DIAGNOSTIC ONLY.  That 2.5x/3.5x
    expansion under-estimates a real vehicle box, so measured IoU stays ~0.07
    even for correct matches and must not be read as recall
  * recall split by plate size bucket, which is where small/distant vehicles show up
  * recall at the as-wired conf (the audited model's nvinfer pre-cluster-threshold,
    e.g. 0.5 for the deployed vehicle_1280_1cls; --conf when no config loads it) vs a
    looser 0.10, to separate "model can't
    see it" from "it is detected but just under the threshold"
  * vehicle detections with no plate (informational — side/front views and
    occluded plates legitimately have none), so the report is not misread as
    "every vehicle must have a plate"

Class filtering comes from the model's own label names (`resolve_vehicle_class_ids`),
not hardcoded COCO ids, and a model whose labels cannot be resolved is refused:
the ONNX the as-wired PGIE config points at is a 2-class export whose output
(1, 8400, 6) does not match that config's `num-detected-classes=80` + COCO
labelfile, so auditing it would report numbers for the wrong classes.

Faithfulness to the running pipeline: frames are first downscaled to the
streammux resolution (`mux_width`/`mux_height`, 1280x720 in the multicam preset)
and only then fed to the models, and the vehicle model is the exact ONNX file
the PGIE config points at.  Both models run at their own ONNX input size (1280
each by default; the old yolo11s ONNX is 640), which mirrors the real pipeline.

Contact sheets with vehicle boxes (green), plate boxes (red) and anchor hits
(white dot) are written for eyeball review.

MLflow (mtmc/tracking/eval.py): the audit is logged into the run of the vehicle
model it audited (its ``<model>.mlflow.json`` sidecar, or ``--run-id``) as
``eval/<set>/*`` (recall at both thresholds, containment, by plate size), with
report.json under ``eval/<set>/``. The set (``--eval-set``, default
``el-mohandes-13clips``) is versioned by the audited clips' DVC pointers. The
anchor plate model's sha256 is tagged too, since it decides what counts as a
vehicle. Contact sheets stay local (client footage). ``--no-track`` records nothing.

Usage:
    .venv/bin/python -m mtmc.audit_vehicle_detector
    .venv/bin/python -m mtmc.audit_vehicle_detector --clips 1-D41-enter.mp4 3-D52-enter.mp4
    .venv/bin/python -m mtmc.audit_vehicle_detector --every-s 5 --limit-clips 3
    .venv/bin/python -m mtmc.audit_vehicle_detector --no-mux --no-contact-sheet
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from .tracking.eval import add_eval_args, model_tags, prepare_evaluation
from .yolo_detector import (
    DEFAULT_PLATE_DETECTOR,
    PLATE_EXPAND_H,
    PLATE_EXPAND_W,
    VEHICLE_CLASSES,
    _plate_to_vehicle_box,
)

# ---------------------------------------------------------------------------
# Paths / defaults (mirror products/multicam/sidecar.yaml)
# ---------------------------------------------------------------------------
VIDEO_DIR = Path("products/el-mohandes/videos")
OUT_DIR = Path("mtmc/output/vehicle_det_audit")

# The ONNX the deployed vehicle PGIE config (perception/models/DeepStream-Yolo/config_infer_primary_vehicle_1cls.txt)
# points at: the 1-class vehicle detector, fixed 1280x1280 input.  The COCO yolo11s audited before
# (perception/models/DeepStream-Yolo/yolo11s.onnx, 640, config_infer_primary_yolo11.txt) is LEGACY_VEHICLE_MODEL.
VEHICLE_MODEL = Path("products/el-mohandes/models/vehicle_1280_1cls/best.onnx")
VEHICLE_IMGSZ = 1280  # module-global; overridden by main() --vehicle-imgsz
LEGACY_VEHICLE_MODEL = Path("perception/models/DeepStream-Yolo/yolo11s.onnx")   # with --vehicle-imgsz 640

# Fine-tuned plate detector, as used by the running plate PGIE.  Its ONNX has a
# 1280x1280 input, so it effectively sees ~2x the linear resolution of the
# vehicle model on the same muxed frame.
PLATE_IMGSZ = 1280

# Multicam preset sources, in preset order.  The audit defaults to exactly the
# clips the running pipeline processes.
MULTICAM_CLIPS = (
    "1-D41-enter.mp4",
    "2-D16_clip.mp4",
    "3-D52-enter.mp4",
    "4-ch05_clip.mp4",
    "5-CH23-enter.mp4",
    "D01-crop.mp4",
    "D05-crop.mp4",
    "D09-crop.mp4",
    "D11-crop.mp4",
    "D15-crop.mp4",
    "D18-crop.mp4",
    "D20-crop.mp4",
    "D21-crop.mp4",
)

# Detection floor; confs are recorded unfiltered so thresholds can be swept
# afterwards without a second inference pass.
DETECT_CONF = 0.05
# The vehicle model's as-wired threshold is NOT a constant: it is the [class-attrs-all] pre-cluster-threshold of the
# nvinfer config that loads the audited model (as_wired_conf), or --conf. main() sets it before auditing.
CONF_AS_WIRED = None
CONF_LOOSE = 0.10         # the low-threshold reference: "detected, just under the threshold"
# Plates at or above this are anchors. Kept at the value the audit always used, independent of the vehicle model's
# threshold: changing the vehicle model must not change which plates count as vehicles that exist.
ANCHOR_CONF = 0.25

# Report keys for the per-threshold sections, derived from the thresholds (set_thresholds) so they never drift.
KEY_AS_WIRED = KEY_LOOSE = None

# Where nvinfer configs live (repo-relative), and the container path their absolute onnx-file entries start with.
NVINFER_CONFIG_GLOBS = ("perception/models/**/config_infer*.txt", "products/**/config_infer*.txt")
CONTAINER_REPO = "/workspace/iris/"
_REPO = Path(__file__).resolve().parents[1]


def conf_label(conf: float) -> str:
    """0.5 -> "0.5", 0.10 -> "0.1": the threshold as it appears in report keys and metric names."""
    return f"{conf:g}"


def set_thresholds(as_wired: float) -> None:
    global CONF_AS_WIRED, KEY_AS_WIRED, KEY_LOOSE
    if abs(as_wired - CONF_LOOSE) < 1e-9:
        raise SystemExit(f"the as-wired threshold {as_wired} equals the low-threshold reference {CONF_LOOSE}: "
                         "the two sections would be the same; pass another --conf")
    CONF_AS_WIRED = as_wired
    KEY_AS_WIRED, KEY_LOOSE = f"at_conf_{conf_label(as_wired)}", f"at_conf_{conf_label(CONF_LOOSE)}"


def _onnx_of(config: Path) -> Path | None:
    """The model file an nvinfer config loads: onnx-file, resolved like the container does."""
    for line in config.read_text(encoding="utf-8", errors="replace").splitlines():
        key, _, val = line.partition("=")
        if key.strip() == "onnx-file" and val.strip():
            v = val.strip()
            if v.startswith(CONTAINER_REPO):
                return _REPO / v[len(CONTAINER_REPO):]
            return Path(v) if Path(v).is_absolute() else config.parent / v
    return None


def pre_cluster_threshold(config: Path) -> float | None:
    """[class-attrs-all] pre-cluster-threshold of an nvinfer config, or None."""
    section = None
    for line in config.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.split("#", 1)[0].strip()
        if line.startswith("["):
            section = line
        elif section == "[class-attrs-all]" and line.partition("=")[0].strip() == "pre-cluster-threshold":
            return float(line.partition("=")[2])
    return None


def as_wired_conf(model: str | Path, conf: float | None = None,
                  config: str | Path | None = None) -> tuple[float, str]:
    """(threshold, where it came from) for the audited vehicle model: --conf, else the pre-cluster-threshold of
    --nvinfer-config, else of the one nvinfer config in the repo that loads this model file. None found, or several
    disagreeing: refused, --conf required."""
    if conf is not None:
        return float(conf), "--conf"
    if config is not None:
        t = pre_cluster_threshold(Path(config))
        if t is None:
            raise SystemExit(f"{config} has no [class-attrs-all] pre-cluster-threshold; pass --conf")
        return t, str(config)
    target = Path(model).resolve()
    found = []
    for pattern in NVINFER_CONFIG_GLOBS:
        for cfg in sorted(_REPO.glob(pattern)):
            onnx = _onnx_of(cfg)
            if onnx is not None and onnx.resolve() == target and (t := pre_cluster_threshold(cfg)) is not None:
                found.append((t, str(cfg.relative_to(_REPO))))
    if not found:
        raise SystemExit(f"no nvinfer config in the repo loads {model}, so its as-wired threshold is unknown: pass "
                         "--conf <the pre-cluster-threshold the deployment uses> (or --nvinfer-config <file>)")
    if len({t for t, _ in found}) > 1:
        raise SystemExit(f"nvinfer configs disagree on {model}'s threshold ("
                         + ", ".join(f"{c}: {t}" for t, c in found) + "): pass --conf or --nvinfer-config")
    return found[0][0], found[0][1]

ANCHOR_IOU_HIT = 0.30     # diagnostic only — see the note on the expansion below
# Containment alone is too weak in congested scenes: with 30+ boxes per frame, a
# background or neighbouring box often happens to cover the plate.  A plate is
# roughly 8-60% of its vehicle's VISIBLE width, so we additionally require the
# containing box to be scale-consistent with the plate it covers.  This is
# scale-free, unlike an absolute area check.
PLATE_W_RATIO_RANGE = (0.08, 0.60)
PLATE_W_RATIO_TYPICAL = 0.35
SIZE_BUCKETS = ((0, 32, "w<32px"), (32, 64, "w32-64"), (64, 128, "w64-128"), (128, 10**6, "w>=128"))

# Label names treated as a vehicle.  Matches mtmc.yolo_detector.VEHICLE_CLASSES
# (COCO car/truck/bus/motorcycle) — bicycle is deliberately excluded — plus the
# single `vehicle` class of the deployed 1-class detector (vehicle_1280_1cls).
VEHICLE_NAME_ALIASES = {"car", "truck", "bus", "motorcycle", "motorbike", "van", "vehicle"}


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested in tests/test_audit_vehicle_detector.py)
# ---------------------------------------------------------------------------
def resolve_vehicle_class_ids(model: YOLO) -> dict[int, str]:
    """Map the vehicle class ids of a detector, using its own label names.

    Filtering by hardcoded COCO ids is only valid for a genuine 80-class COCO
    detector.  A model whose labels are generic (`class0`, `class1`, ...) and
    whose class count is not COCO's cannot be class-filtered at all — that means
    the weights do not match the PGIE config that loads them, so we refuse to
    report numbers rather than quietly describing the wrong classes.
    """
    names = {int(i): str(n) for i, n in (getattr(model, "names", None) or {}).items()}
    matched = {i: n for i, n in names.items() if n.lower() in VEHICLE_NAME_ALIASES}
    if matched:
        return matched
    generic = (not names) or all(re.fullmatch(r"class\d+", n) for n in names.values())
    if not generic and len(names) == 80:
        # Real COCO detector whose labels use different spellings.
        return {i: n for i, n in names.items() if i in VEHICLE_CLASSES}
    raise SystemExit(
        f"cannot resolve vehicle classes for {getattr(model, 'ckpt_path', 'model')}: "
        f"{len(names)} labels, none of them a vehicle name "
        f"{'and all generic (classN)' if generic else ''}.\n"
        "The weights likely do not match the PGIE config that loads them "
        "(the deployed yolo11s.onnx is a 2-class model while "\
        "config_infer_primary_yolo11.txt declares num-detected-classes=80). "
        "Point --vehicle-model at a real COCO detector or a re-exported ONNX."
    )


def size_bucket(width: float) -> str:
    """Bucket label for a box width in pixels."""
    for lo, hi, name in SIZE_BUCKETS:
        if lo <= width < hi:
            return name
    return SIZE_BUCKETS[-1][2]


def _iou_one(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    """IoU of a single box against every box in `others` (both xyxy)."""
    if len(others) == 0:
        return np.zeros(0)
    ix1 = np.maximum(box[0], others[:, 0])
    iy1 = np.maximum(box[1], others[:, 1])
    ix2 = np.minimum(box[2], others[:, 2])
    iy2 = np.minimum(box[3], others[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    union = a + b - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def anchor_plate(
    plate_box: np.ndarray,
    vehicle_boxes: np.ndarray,
    vehicle_confs: np.ndarray,
    frame_w: int,
    frame_h: int,
    ratio_range: tuple[float, float] = PLATE_W_RATIO_RANGE,
) -> dict:
    """Score one plate detection against the vehicle detections in its frame.

    Primary signal: a vehicle box must cover the plate centre (containment) AND
    be scale-consistent with the plate it covers (plate width as a fraction of
    the box width inside `ratio_range`).  Of the containing candidates we take
    the one closest to the usual plate/vehicle width ratio.

    IoU against the plate-expanded box is recorded as a diagnostic only: that
    expansion under-estimates the true vehicle extent (measured mean IoU ~0.07
    even when the containing box is plainly the right vehicle), so it cannot be
    the recall criterion.
    """
    expected = _plate_to_vehicle_box(plate_box, frame_w, frame_h)
    ious = _iou_one(expected, vehicle_boxes)
    best_iou = float(ious.max()) if len(ious) else 0.0
    best_idx = int(ious.argmax()) if len(ious) else -1
    best_conf = float(vehicle_confs[best_idx]) if best_idx >= 0 else 0.0

    plate_w = float(plate_box[2] - plate_box[0])
    pcx = (plate_box[0] + plate_box[2]) / 2.0
    pcy = (plate_box[1] + plate_box[3]) / 2.0
    containing = (
        (vehicle_boxes[:, 0] <= pcx)
        & (vehicle_boxes[:, 2] >= pcx)
        & (vehicle_boxes[:, 1] <= pcy)
        & (vehicle_boxes[:, 3] >= pcy)
        if len(vehicle_boxes)
        else np.zeros(0, dtype=bool)
    )
    contained = bool(containing.any())

    ratio = None
    if contained:
        cand = vehicle_boxes[containing]
        widths = cand[:, 2] - cand[:, 0]
        ratios = plate_w / np.maximum(widths, 1e-6)
        # candidate closest to the typical plate/vehicle width ratio
        ratio = float(ratios[np.abs(ratios - PLATE_W_RATIO_TYPICAL).argmin()])

    lo, hi = ratio_range
    return {
        "plate_w": plate_w,
        "plate_h": float(plate_box[3] - plate_box[1]),
        "bucket": size_bucket(plate_w),
        "best_iou": round(best_iou, 4),
        "iou_hit": bool(best_iou >= ANCHOR_IOU_HIT),
        "contained": contained,
        "plate_w_ratio": round(ratio, 4) if ratio is not None else None,
        "hit": bool(contained and ratio is not None and lo <= ratio <= hi),
        "best_vehicle_conf": round(best_conf, 4),
    }


def _score(anchors: np.ndarray, v_boxes: np.ndarray, v_confs: np.ndarray,
           frame_w: int, frame_h: int, conf: float) -> list[dict]:
    """Score every anchor plate against the vehicle boxes that clear `conf`.

    A vehicle box below the conf threshold does not exist for the as-wired run,
    so hits are evaluated per threshold rather than filtered afterwards.
    """
    keep = v_confs >= conf
    return [anchor_plate(pb, v_boxes[keep], v_confs[keep], frame_w, frame_h)
            for pb in anchors]


def summarize(records: list[dict]) -> dict:
    """Aggregate plate-anchor records that were already scored at one conf threshold."""
    n = len(records)
    hits = sum(1 for r in records if r["hit"])
    contained = sum(1 for r in records if r["contained"])
    iou_hits = sum(1 for r in records if r["iou_hit"])
    ious = [r["best_iou"] for r in records]

    by_bucket: dict[str, dict] = {}
    for name in [b[2] for b in SIZE_BUCKETS]:
        rs = [r for r in records if r["bucket"] == name]
        if not rs:
            continue
        bucket_hits = sum(1 for r in rs if r["hit"])
        by_bucket[name] = {
            "plates": len(rs),
            "hits": bucket_hits,
            "recall": round(bucket_hits / len(rs), 4),
            "containment": round(sum(1 for r in rs if r["contained"]) / len(rs), 4),
        }

    return {
        "plates": n,
        "hits": hits,
        "recall": round(hits / n, 4) if n else 0.0,
        "containment": round(contained / n, 4) if n else 0.0,
        "iou_recall": round(iou_hits / n, 4) if n else 0.0,
        "mean_best_iou": round(float(np.mean(ious)), 4) if ious else 0.0,
        "by_bucket": by_bucket,
    }


# ---------------------------------------------------------------------------
# Detection / sampling
# ---------------------------------------------------------------------------
def _detect(model: YOLO, frame: np.ndarray, imgsz: int, conf: float,
            class_ids: set[int] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run one frame. Returns (boxes xyxy, confs, class ids), filtered by class.

    `rect=False` forces square letterboxing: the vehicle ONNX has a static
    640x640 input, and DeepStream letterboxes square too (maintain-aspect-ratio
    + symmetric-padding), so rectangular inference would be both a runtime error
    and a fidelity break.
    """
    res = model(frame, imgsz=imgsz, conf=conf, iou=0.45, rect=False, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
    boxes = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    clss = res.boxes.cls.cpu().numpy().astype(int)
    if class_ids is not None:
        mask = np.isin(clss, list(class_ids))
        boxes, confs, clss = boxes[mask], confs[mask], clss[mask]
    return boxes, confs, clss


def sample_frame_indices(total_frames: int, fps: float, every_s: float,
                         max_samples: int) -> list[int]:
    """Frame indices sampled every `every_s` seconds across a clip."""
    if fps <= 0 or every_s <= 0:
        return []
    step = max(1, int(round(every_s * fps)))
    idx = list(range(0, total_frames, step))
    return idx[:max_samples] if max_samples > 0 else idx


def _draw_tile(frame: np.ndarray, vehicle_boxes: np.ndarray, vehicle_confs: np.ndarray,
               plate_boxes: np.ndarray, hits: list[bool]) -> np.ndarray:
    """Annotate one sampled frame for the contact sheet."""
    img = frame.copy()
    for (x1, y1, x2, y2), c in zip(vehicle_boxes, vehicle_confs):
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(img, f"veh {c:.2f}", (int(x1), max(14, int(y1) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    for (x1, y1, x2, y2), hit in zip(plate_boxes, hits):
        colour = (0, 0, 255) if hit else (0, 165, 255)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), colour, 2)
        label = "plate OK" if hit else "plate MISS"
        cv2.putText(img, label, (int(x1), min(img.shape[0] - 4, int(y2) + 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
    return img


def audit_clip(
    clip: Path,
    vehicle_model: YOLO,
    plate_model: YOLO,
    vehicle_class_ids: set[int],
    every_s: float,
    mux: tuple[int, int] | None,
    max_samples: int,
    contact_sheet: bool,
    out_dir: Path,
) -> dict:
    """Audit one clip and return its report section."""
    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        return {"clip": clip.name, "error": "cannot open"}

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    indices = sample_frame_indices(total, fps, every_s, max_samples)

    n_vehicle_boxes = 0
    class_counts: dict[str, int] = {}
    vehicles_without_plate = 0
    tiles: list[np.ndarray] = []
    # Anchor records accumulated ACROSS sampled frames, one list per threshold.
    per_records: dict[str, list[dict]] = {KEY_AS_WIRED: [], KEY_LOOSE: []}
    t0 = time.time()

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        if mux is not None and (src_w, src_h) != mux:
            # Reproduce the streammux downscale the PGIE actually sees.
            frame = cv2.resize(frame, mux, interpolation=cv2.INTER_AREA)
        h, w = frame.shape[:2]

        v_boxes, v_confs, v_clss = _detect(
            vehicle_model, frame, VEHICLE_IMGSZ, DETECT_CONF, vehicle_class_ids
        )
        p_boxes, p_confs, _ = _detect(plate_model, frame, PLATE_IMGSZ, DETECT_CONF)

        n_vehicle_boxes += len(v_boxes)
        for c in v_clss:
            name = getattr(vehicle_model, "names", {}).get(int(c), str(c))
            class_counts[name] = class_counts.get(name, 0) + 1

        # Plate anchor: only plates at the anchor conf are anchors (not the vehicle model's threshold).
        anchors = p_boxes[p_confs >= ANCHOR_CONF]

        # Score anchors at each vehicle conf threshold and accumulate across the clip.
        as_wired = _score(anchors, v_boxes, v_confs, w, h, CONF_AS_WIRED)
        per_records[KEY_AS_WIRED].extend(as_wired)
        per_records[KEY_LOOSE].extend(_score(anchors, v_boxes, v_confs, w, h, CONF_LOOSE))
        hits = [r["hit"] for r in as_wired]

        # Vehicles with no plate overlapping them (front/side views, occlusion).
        paid = np.zeros(len(v_boxes), dtype=bool)
        for pb in anchors:
            ious = _iou_one(pb, v_boxes)
            if len(ious) and ious.max() > 0.1:
                paid[int(ious.argmax())] = True
        vehicles_without_plate += int((~paid).sum()) if len(v_boxes) else 0

        if contact_sheet:
            tiles.append(_draw_tile(frame, v_boxes, v_confs, anchors, hits))

    cap.release()

    if contact_sheet and tiles:
        out_dir.mkdir(parents=True, exist_ok=True)
        cols = 3
        rows = (len(tiles) + cols - 1) // cols
        tile_w = 480
        scaled = [cv2.resize(t, (tile_w, int(tile_w * t.shape[0] / t.shape[1]))) for t in tiles]
        th = max(t.shape[0] for t in scaled)
        grid = np.zeros((rows * th, cols * tile_w, 3), dtype=np.uint8)
        for i, t in enumerate(scaled):
            r, c = divmod(i, cols)
            grid[r * th:r * th + t.shape[0], c * tile_w:c * tile_w + t.shape[1]] = t
        sheet_path = out_dir / f"contact_{clip.stem}.jpg"
        cv2.imwrite(str(sheet_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])
    else:
        sheet_path = None

    return {
        "clip": clip.name,
        "vehicle_model": getattr(vehicle_model, "ckpt_path", "model"),
        "source_res": f"{src_w}x{src_h}",
        "mux_res": f"{mux[0]}x{mux[1]}" if mux else "off (native)",
        "fps": round(fps, 2),
        "frames_total": total,
        "frames_sampled": len(indices),
        "vehicle_boxes": n_vehicle_boxes,
        "vehicle_class_counts": class_counts,
        "plates_as_wired": len(anchors),
        "vehicle_boxes_without_plate": vehicles_without_plate,
        "plate_anchor": {k: summarize(v) for k, v in per_records.items()},
        "contact_sheet": str(sheet_path) if sheet_path else None,
        "seconds": round(time.time() - t0, 1),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_report(sections: list[dict]) -> None:
    ok = [s for s in sections if "error" not in s]
    print("\n=== VEHICLE DETECTOR AUDIT (plate-anchored recall) ===")
    print(f"vehicle: {sections[0].get('vehicle_model', '?')} @{VEHICLE_IMGSZ}   "
          f"plate: finetuned @{PLATE_IMGSZ}")
    print("anchor: every plate detection above the as-wired conf is one vehicle that exists\n")

    print("recall = plate centre inside a vehicle box whose width is scale-consistent with")
    print(f"         the plate (plate/box width ratio {PLATE_W_RATIO_RANGE[0]}-{PLATE_W_RATIO_RANGE[1]}) — primary signal")
    print("raw    = containment only (upper bound)   iou.3 = IoU vs the expanded box")
    print(f"(diagnostic only — the {PLATE_EXPAND_W}x/{PLATE_EXPAND_H}x expansion under-estimates")
    print("a real vehicle box, so that IoU is near zero for correct matches)\n")

    hdr = (f"{'clip':22s} {'sampled':>7s} {'veh':>5s} {'plates':>6s} "
           f"{'recall':>7s} {'at' + conf_label(CONF_LOOSE)[1:]:>6s} {'raw':>6s} {'iou.3':>6s}")
    print(hdr)
    print("-" * len(hdr))
    for s in ok:
        r1 = s["plate_anchor"][KEY_AS_WIRED]
        r2 = s["plate_anchor"][KEY_LOOSE]
        print(f"{s['clip'][:22]:22s} {s['frames_sampled']:7d} {s['vehicle_boxes']:5d} "
              f"{r1['plates']:6d} {r1['recall']:7.3f} {r2['recall']:6.3f} "
              f"{r1['containment']:6.3f} {r1['iou_recall']:6.3f}")

    agg25 = _pool(ok, KEY_AS_WIRED)
    agg10 = _pool(ok, KEY_LOOSE)
    print("-" * len(hdr))
    print(f"{'ALL':22s} {'':7s} {'':5s} {agg25['plates']:6d} {agg25['recall']:7.3f} "
          f"{agg10['recall']:6.3f} {agg25['containment']:6.3f} {agg25['iou_recall']:6.3f}")

    print(f"\nmean best IoU (against expanded box): {agg25['mean_best_iou']:.3f}")
    print("\nrecall by plate width:")
    for name in [b[2] for b in SIZE_BUCKETS]:
        b25 = agg25["by_bucket"].get(name)
        if not b25:
            continue
        b10 = agg10["by_bucket"].get(name, {})
        print(f"  {name:9s} plates={b25['plates']:5d}  recall@{conf_label(CONF_AS_WIRED)}={b25['recall']:.3f}  "
              f"recall@{conf_label(CONF_LOOSE)}={b10.get('recall', 0.0):.3f}  contained={b25['containment']:.3f}")

    print("\nvehicle detections by class: " + json.dumps(_merge_counts(ok)))
    print(f"vehicle boxes with no plate near them: {sum(s['vehicle_boxes_without_plate'] for s in ok)} "
          f"(expected for front/side views)")

    for s in sections:
        if "error" in s:
            print(f"  ! {s['clip']}: {s['error']}")


def _pool(sections: list[dict], key: str) -> dict:
    """Pool per-clip bucket counts so buckets aggregate across clips."""
    tot: dict[str, dict] = {}
    plates = hits = contained = iou_hits = 0
    ious: list[float] = []
    for s in sections:
        r = s["plate_anchor"][key]
        plates += r["plates"]
        hits += r["hits"]
        contained += int(round(r["containment"] * r["plates"]))
        iou_hits += int(round(r["iou_recall"] * r["plates"]))
        ious.append(r["mean_best_iou"])
        for name, b in r["by_bucket"].items():
            agg = tot.setdefault(
                name, {"plates": 0, "hits": 0, "recall": 0.0, "containment": 0.0, "_c": 0}
            )
            agg["plates"] += b["plates"]
            agg["hits"] += b["hits"]
            agg["_c"] += int(round(b.get("containment", 0.0) * b["plates"]))
    for agg in tot.values():
        agg["recall"] = round(agg["hits"] / agg["plates"], 4) if agg["plates"] else 0.0
        agg["containment"] = round(agg.pop("_c") / agg["plates"], 4) if agg["plates"] else 0.0
    return {
        "plates": plates,
        "hits": hits,
        "recall": round(hits / plates, 4) if plates else 0.0,
        "containment": round(contained / plates, 4) if plates else 0.0,
        "iou_recall": round(iou_hits / plates, 4) if plates else 0.0,
        "mean_best_iou": round(float(np.mean(ious)), 4) if ious else 0.0,
        "by_bucket": tot,
    }


def _merge_counts(sections: list[dict]) -> dict:
    out: dict[str, int] = {}
    for s in sections:
        for k, v in s["vehicle_class_counts"].items():
            out[k] = out.get(k, 0) + v
    return dict(sorted(out.items()))


def _metric_bucket(name: str) -> str:
    """A size bucket's name as a metric-name segment (MLflow allows no < or >)."""
    return name.replace(">=", "ge").replace("<", "lt").replace(">", "gt")


def audit_metrics(sections: list[dict]) -> dict[str, float]:
    """The pooled numbers print_report shows, as metrics for the audited model's run."""
    ok = [s for s in sections if "error" not in s]
    m: dict[str, float] = {"clips": len(ok), "clips_failed": len(sections) - len(ok)}
    if not ok:
        return m
    agg25, agg10 = _pool(ok, KEY_AS_WIRED), _pool(ok, KEY_LOOSE)
    as_wired, loose = f"conf{conf_label(CONF_AS_WIRED)}", f"conf{conf_label(CONF_LOOSE)}"
    m.update({
        "plates": agg25["plates"],
        f"recall_{as_wired}": agg25["recall"], f"recall_{loose}": agg10["recall"],
        f"containment_{as_wired}": agg25["containment"], f"iou_recall_{as_wired}": agg25["iou_recall"],
        "mean_best_iou": agg25["mean_best_iou"],
        "vehicle_boxes_without_plate": sum(s["vehicle_boxes_without_plate"] for s in ok),
    })
    for key, agg in ((as_wired, agg25), (loose, agg10)):
        for name, b in agg["by_bucket"].items():
            m[f"recall_{key}/{_metric_bucket(name)}"] = b["recall"]
    return {k: float(v) for k, v in m.items()}


def main(argv: list[str] | None = None) -> None:
    global VEHICLE_IMGSZ
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", nargs="*", default=None,
                    help="clip filenames to audit (default: the 13 multicam preset clips)")
    ap.add_argument("--limit-clips", type=int, default=0,
                    help="only audit the first N clips (0 = all)")
    ap.add_argument("--every-s", type=float, default=2.0,
                    help="sample one frame every N seconds (default: 2.0)")
    ap.add_argument("--max-samples", type=int, default=120,
                    help="cap sampled frames per clip (default: 120)")
    ap.add_argument("--mux-width", type=int, default=1280)
    ap.add_argument("--mux-height", type=int, default=720)
    ap.add_argument("--no-mux", action="store_true",
                    help="feed native-resolution frames instead of the streammux downscale")
    ap.add_argument("--vehicle-model", default=str(VEHICLE_MODEL),
                    help=f"vehicle weights (default: the deployed {VEHICLE_MODEL}; the COCO yolo11s audited "
                         f"before is {LEGACY_VEHICLE_MODEL} with --vehicle-imgsz 640)")
    ap.add_argument("--vehicle-imgsz", type=int, default=VEHICLE_IMGSZ,
                    help="vehicle inference size (default 1280 = the deployed ONNX input; 640 for the legacy "
                         "yolo11s ONNX)")
    ap.add_argument("--conf", type=float, default=None,
                    help="the vehicle model's as-wired threshold (default: the [class-attrs-all] pre-cluster-threshold "
                         "of the nvinfer config that loads --vehicle-model; required when no config does)")
    ap.add_argument("--nvinfer-config", default=None,
                    help="read the as-wired threshold from this nvinfer config instead of searching the repo")
    ap.add_argument("--plate-model", default=str(DEFAULT_PLATE_DETECTOR))
    ap.add_argument("--out", default=str(OUT_DIR / "report.json"))
    ap.add_argument("--no-contact-sheet", action="store_true")
    add_eval_args(ap, "el-mohandes-13clips")
    args = ap.parse_args(argv)
    VEHICLE_IMGSZ = args.vehicle_imgsz
    conf, conf_source = as_wired_conf(args.vehicle_model, args.conf, args.nvinfer_config)
    set_thresholds(conf)
    print(f"[audit] vehicle as-wired conf {conf_label(conf)} (from {conf_source}), low reference "
          f"{conf_label(CONF_LOOSE)}, plate anchors at >= {ANCHOR_CONF}")

    names = args.clips or list(MULTICAM_CLIPS)
    if args.limit_clips:
        names = names[: args.limit_clips]
    clips = [VIDEO_DIR / n for n in names]
    for c in clips:
        if not c.exists():
            print(f"  ! missing: {c}")
    present = [c for c in clips if c.exists()]
    if not present:
        raise SystemExit(f"no clip to audit in {VIDEO_DIR} (dvc pull the el-mohandes-samples group)")
    # the set is the clips actually audited: a partial `dvc pull` is another version, never the full set's
    target = prepare_evaluation(
        args, args.vehicle_model, present, "mtmc.audit_vehicle_detector",
        tags={"vehicle_imgsz": args.vehicle_imgsz, "mux": "off" if args.no_mux else f"{args.mux_width}x{args.mux_height}",
              "every_s": args.every_s, "max_samples": args.max_samples, "clips": len(present),
              "conf_as_wired": conf_label(CONF_AS_WIRED), "conf_as_wired_source": conf_source,
              "conf_low": conf_label(CONF_LOOSE), "anchor_conf": ANCHOR_CONF,
              **model_tags(args.plate_model, "anchor")})

    mux = None if args.no_mux else (args.mux_width, args.mux_height)
    print(f"[audit] vehicle={args.vehicle_model} imgsz={VEHICLE_IMGSZ}  "
          f"plate={args.plate_model} imgsz={PLATE_IMGSZ}")
    print(f"[audit] mux={'off' if mux is None else f'{mux[0]}x{mux[1]}'}  "
          f"every={args.every_s}s  max_samples={args.max_samples}")

    vehicle_model = YOLO(args.vehicle_model, task="detect")
    plate_model = YOLO(args.plate_model, task="detect")
    vehicle_classes = resolve_vehicle_class_ids(vehicle_model)
    print(f"[audit] vehicle classes: {vehicle_classes}")

    out_dir = Path(args.out).parent
    sections: list[dict] = []
    for i, clip in enumerate([c for c in clips if c.exists()], 1):
        print(f"[{i}/{len(clips)}] {clip.name} ...", flush=True)
        sections.append(audit_clip(
            clip, vehicle_model, plate_model, set(vehicle_classes), args.every_s, mux,
            args.max_samples, not args.no_contact_sheet, out_dir,
        ))

    print_report(sections)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "config": {
            "vehicle_model": args.vehicle_model,
            "vehicle_imgsz": VEHICLE_IMGSZ,
            "plate_model": args.plate_model,
            "plate_imgsz": PLATE_IMGSZ,
            "mux": mux,
            "every_s": args.every_s,
            "anchor_iou_hit": ANCHOR_IOU_HIT,
            "conf_as_wired": CONF_AS_WIRED,
            "conf_as_wired_source": conf_source,
            "conf_low": CONF_LOOSE,
            "anchor_conf": ANCHOR_CONF,
            "plate_expand": [PLATE_EXPAND_W, PLATE_EXPAND_H],
        },
        "clips": sections,
    }, indent=2))
    print(f"\nreport -> {args.out}")
    if not args.no_contact_sheet:
        print(f"contact sheets -> {out_dir}/contact_*.jpg")
    target.log(audit_metrics(sections), artifacts=[args.out])      # report.json only: no client footage


if __name__ == "__main__":
    main()
