#!/usr/bin/env python3
"""Benchmark plate detection models on CCTV footage.

Compares all available models (YOLOv8n Egyptian detector + YOLOv11 variants)
against ground-truth labels from CCTV cameras only (excludes the generic
EALPR dataset).  Reports mAP@50, mAP@50-95, precision, recall per model,
with per-camera breakdowns and optional SAHI slicing for small plates.

Usage:
    # Standard inference (640×640)
    python -m mtmc.benchmark_models

    # With SAHI slicing (recommended for CCTV — plates are tiny)
    python -m mtmc.benchmark_models --sahi

    # Custom confidence threshold
    python -m mtmc.benchmark_models --sahi --conf 0.25

    # Filter to specific cameras
    python -m mtmc.benchmark_models --cameras ch23 D15

    # Save results to JSON
    python -m mtmc.benchmark_models --sahi --output results.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REVIEWED_DIR = Path("mtmc/output/plate_crops/a_REVIEWED")
MODELS_DIR = Path("mtmc/models")

# Models to benchmark: (display_name, path)
MODEL_REGISTRY: list[tuple[str, Path]] = []

# Egyptian detector (.pt weights — can be fine-tuned)
eg_dir = MODELS_DIR / "egyptian-license-plate-detector"
for pt_file in sorted(eg_dir.glob("*.pt")):
    MODEL_REGISTRY.append((f"Egyptian-{pt_file.stem}", pt_file))

# YOLOv11 ONNX variants (inference only — for benchmarking, not fine-tuning)
yolo11_dir = MODELS_DIR / "yolov11-license-plate-detection"
for onnx_file in sorted(yolo11_dir.glob("*.onnx")):
    size = onnx_file.stem.split("-")[-1].replace("v1", "")  # n, s, m, l, x
    MODEL_REGISTRY.append((f"YOLOv11-{size}", onnx_file))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Sample:
    image_path: Path
    label_path: Path
    camera: str  # e.g. "ch23", "D15", "D16"
    folder: str  # full folder name


def load_cctv_data(cameras: list[str] | None = None) -> list[Sample]:
    """Load CCTV frame/label pairs, excluding EALPR."""
    samples: list[Sample] = []
    for folder_dir in sorted(REVIEWED_DIR.iterdir()):
        if not folder_dir.is_dir():
            continue
        # Skip EALPR and any non-CCTV folders
        if "EALPR" in folder_dir.name or "ealpr" in folder_dir.name.lower():
            continue

        # Extract camera prefix (e.g. "ch23" from "ch23_20260726055016")
        cam = folder_dir.name.split("_")[0]
        if cameras and cam not in cameras:
            continue

        frames_dir = folder_dir / "frames"
        labels_dir = folder_dir / "labels"
        if not frames_dir.exists():
            continue

        for img_path in sorted(frames_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            lbl_path = labels_dir / (img_path.stem + ".txt")
            samples.append(Sample(
                image_path=img_path,
                label_path=lbl_path,
                camera=cam,
                folder=folder_dir.name,
            ))

    return samples


def read_yolo_labels(label_path: Path, img_w: int, img_h: int) -> list[tuple[int, int, int, int]]:
    """Read YOLO labels → list of (x1, y1, x2, y2) in pixel coords."""
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            try:
                _cls, xc, yc, w, h = parts[:5]
                xc, yc, w, h = float(xc), float(yc), float(w), float(h)
                x1 = int((xc - w / 2) * img_w)
                y1 = int((yc - h / 2) * img_h)
                x2 = int((xc + w / 2) * img_w)
                y2 = int((yc + h / 2) * img_h)
                boxes.append((x1, y1, x2, y2))
            except (ValueError, IndexError):
                pass
    return boxes


# ---------------------------------------------------------------------------
# IoU and mAP computation
# ---------------------------------------------------------------------------
def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """IoU between two boxes (x1,y1,x2,y2)."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_predictions(
    pred_boxes: list[np.ndarray],
    pred_scores: list[float],
    gt_boxes: list[np.ndarray],
    iou_threshold: float = 0.5,
) -> tuple[int, int, int]:
    """Match predictions to ground truth. Returns (tp, fp, fn)."""
    if not gt_boxes and not pred_boxes:
        return 0, 0, 0
    if not gt_boxes:
        return 0, len(pred_boxes), 0
    if not pred_boxes:
        return 0, 0, len(gt_boxes)

    # Sort predictions by confidence (descending)
    order = np.argsort(-np.array(pred_scores))
    pred_boxes = [pred_boxes[i] for i in order]
    pred_scores = [pred_scores[i] for i in order]

    gt_matched = [False] * len(gt_boxes)
    tp = 0
    fp = 0

    for pb in pred_boxes:
        best_iou = 0.0
        best_gt = -1
        for j, gb in enumerate(gt_boxes):
            if gt_matched[j]:
                continue
            iou = compute_iou(pb, gb)
            if iou > best_iou:
                best_iou = iou
                best_gt = j
        if best_iou >= iou_threshold and best_gt >= 0:
            gt_matched[best_gt] = True
            tp += 1
        else:
            fp += 1

    fn = sum(1 for m in gt_matched if not m)
    return tp, fp, fn


def compute_map(
    all_tp: dict[float, int],
    all_fp: dict[float, int],
    all_fn: dict[float, int],
    n_gt_total: int,
) -> float:
    """Compute Average Precision at a single IoU threshold (AP@k)."""
    if n_gt_total == 0:
        return 0.0

    # Build precision-recall curve
    thresholds = sorted(all_tp.keys(), reverse=True)
    tp_cumsum = 0
    fp_cumsum = 0
    precisions = []
    recalls = []

    for conf in thresholds:
        tp_cumsum += all_tp[conf]
        fp_cumsum += all_fp[conf]
        fn = all_fn[conf]  # remaining FN at this threshold
        precision = tp_cumsum / (tp_cumsum + fp_cumsum) if (tp_cumsum + fp_cumsum) > 0 else 0.0
        recall = tp_cumsum / n_gt_total if n_gt_total > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)

    # AP = area under precision-recall curve (11-point interpolation)
    ap = 0.0
    for r_thresh in np.arange(0, 1.1, 0.1):
        prec_at_r = [p for p, r in zip(precisions, recalls) if r >= r_thresh]
        if prec_at_r:
            ap += max(prec_at_r) / 11.0

    return ap


def compute_metrics(
    samples: list[Sample],
    model,
    imgsz: int = 640,
    conf: float = 0.25,
    use_sahi: bool = False,
    sahi_slice: int = 640,
    sahi_overlap: float = 0.2,
    device: str | None = None,
) -> dict:
    """Run inference on all samples and compute metrics."""
    from ultralytics import YOLO

    n_images = 0
    n_images_with_gt = 0
    t0 = time.time()

    # SAHI setup
    sahi_model = None
    if use_sahi:
        try:
            from sahi.model import UltralyticsYolov8DetectionModel
            sahi_model = UltralyticsYolov8DetectionModel(
                model_path=str(model),
                confidence_threshold=conf,
                device=str(device) if device else "cpu",
                image_size=imgsz,
            )
        except Exception as e:
            print(f"  [WARN] SAHI setup failed: {e}")
            print("  Falling back to standard inference")
            sahi_model = None

    # Single pass: collect all predictions and ground truth
    all_predictions: list[tuple[float, int, np.ndarray]] = []  # (score, image_idx, box)
    all_gt_by_image: dict[int, list[np.ndarray]] = {}  # image_idx -> [boxes]

    for i, sample in enumerate(samples):
        img = cv2.imread(str(sample.image_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]
        n_images += 1

        # Ground truth
        gt_boxes = [np.array(b, dtype=float) for b in read_yolo_labels(sample.label_path, img_w, img_h)]
        all_gt_by_image[i] = gt_boxes
        if gt_boxes:
            n_images_with_gt += 1

        # Inference
        if sahi_model is not None:
            pred_boxes_px, pred_scores = _sahi_inference(
                sahi_model, img, img_w, img_h, sahi_slice, sahi_overlap
            )
        else:
            pred_boxes_px, pred_scores = _standard_inference(
                model, img, imgsz, conf, device
            )

        for box, score in zip(pred_boxes_px, pred_scores):
            all_predictions.append((float(score), i, np.array(box, dtype=float)))

    elapsed = time.time() - t0

    # Sort predictions by confidence (descending)
    all_predictions.sort(key=lambda x: -x[0])

    n_gt_total = sum(len(v) for v in all_gt_by_image.values())

    # Compute AP@50 and AP@50-95
    ap50 = _compute_ap_for_iou(all_predictions, all_gt_by_image, 0.5)
    ap_all = np.mean([
        _compute_ap_for_iou(all_predictions, all_gt_by_image, iou_t)
        for iou_t in np.arange(0.5, 1.0, 0.05)
    ])

    # Final precision/recall at the given conf threshold
    tp_final, fp_final, fn_final = 0, 0, 0
    for i in all_gt_by_image:
        gt = all_gt_by_image[i]
        img_preds = [(s, b) for s, img_i, b in all_predictions if img_i == i and s >= conf]
        pred_boxes_list = [b for _, b in img_preds]
        pred_scores_list = [s for s, _ in img_preds]
        tp, fp, fn = match_predictions(pred_boxes_list, pred_scores_list, gt, iou_threshold=0.5)
        tp_final += tp
        fp_final += fp
        fn_final += fn

    precision = tp_final / (tp_final + fp_final) if (tp_final + fp_final) > 0 else 0.0
    recall = tp_final / (tp_final + fn_final) if (tp_final + fn_final) > 0 else 0.0

    return {
        "mAP@50": round(ap50, 4),
        "mAP@50-95": round(ap_all, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "tp": tp_final,
        "fp": fp_final,
        "fn": fn_final,
        "gt_boxes": n_gt_total,
        "pred_boxes": len(all_predictions),
        "images": n_images,
        "images_with_gt": n_images_with_gt,
        "time_s": round(elapsed, 2),
        "fps": round(n_images / elapsed, 1) if elapsed > 0 else 0,
    }


def _compute_ap_for_iou(
    all_predictions: list[tuple[float, int, np.ndarray]],
    all_gt_by_image: dict[int, list[np.ndarray]],
    iou_threshold: float,
) -> float:
    """Compute Average Precision for a given IoU threshold."""
    n_gt = sum(len(v) for v in all_gt_by_image.values())
    if n_gt == 0:
        return 0.0

    gt_matched: dict[int, list[bool]] = {
        img_idx: [False] * len(boxes)
        for img_idx, boxes in all_gt_by_image.items()
    }

    tp_cumsum = 0
    fp_cumsum = 0
    precisions = []
    recalls = []

    for score, img_idx, pred_box in all_predictions:
        if img_idx not in all_gt_by_image:
            fp_cumsum += 1
            continue

        gt_boxes = all_gt_by_image[img_idx]
        matched = gt_matched[img_idx]

        best_iou = 0.0
        best_j = -1
        for j, gt_box in enumerate(gt_boxes):
            if matched[j]:
                continue
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_iou >= iou_threshold and best_j >= 0:
            matched[best_j] = True
            tp_cumsum += 1
        else:
            fp_cumsum += 1

        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        recall = tp_cumsum / n_gt
        precisions.append(precision)
        recalls.append(recall)

    if not precisions:
        return 0.0

    # AP = area under PR curve using all-point interpolation (COCO style)
    # Pad with endpoint
    precisions.append(0.0)
    recalls.append(0.0)

    # Compute using the 101-point interpolation (COCO)
    ap = 0.0
    for r in np.arange(0, 1.01, 0.01):
        prec_at_r = [p for p, rec in zip(precisions, recalls) if rec >= r]
        if prec_at_r:
            ap += max(prec_at_r) / 101.0

    return ap


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
def _standard_inference(
    model,
    img: np.ndarray,
    imgsz: int,
    conf: float,
    device: str | None,
) -> tuple[list[np.ndarray], list[float]]:
    """Standard YOLO inference. Returns (boxes_px, scores)."""
    results = model(img, imgsz=imgsz, conf=conf, verbose=False, device=device)
    boxes = []
    scores = []
    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                s = float(box.conf[0])
                boxes.append(np.array([x1, y1, x2, y2]))
                scores.append(s)
    return boxes, scores


def _sahi_inference(
    sahi_model,
    img: np.ndarray,
    img_w: int,
    img_h: int,
    slice_size: int,
    overlap: float,
) -> tuple[list[np.ndarray], list[float]]:
    """SAHI sliced inference. Returns (boxes_px, scores)."""
    from sahi.predict import get_sliced_prediction

    result = get_sliced_prediction(
        image=img,
        detection_model=sahi_model,
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap,
        overlap_width_ratio=overlap,
        perform_standard_pred=True,
        postprocess_type="GREEDYNMM",
        postprocess_match_metric="IOS",
        postprocess_match_threshold=0.5,
        verbose=0,
    )

    boxes = []
    scores = []
    for pred in result.object_prediction_list:
        bbox = pred.bbox
        x1, y1 = max(0, int(bbox.minx)), max(0, int(bbox.miny))
        x2, y2 = min(img_w, int(bbox.maxx)), min(img_h, int(bbox.maxy))
        if x2 > x1 and y2 > y1:
            boxes.append(np.array([x1, y1, x2, y2]))
            scores.append(pred.score.value)

    return boxes, scores


# ---------------------------------------------------------------------------
# Per-camera breakdown
# ---------------------------------------------------------------------------
def compute_per_camera(
    samples: list[Sample],
    model,
    imgsz: int,
    conf: float,
    use_sahi: bool,
    sahi_slice: int,
    sahi_overlap: float,
    device: str | None,
) -> dict[str, dict]:
    """Compute metrics broken down by camera."""
    cameras = sorted(set(s.camera for s in samples))
    results = {}
    for cam in cameras:
        cam_samples = [s for s in samples if s.camera == cam]
        if not cam_samples:
            continue
        print(f"    Camera {cam}: {len(cam_samples)} images ...", end=" ", flush=True)
        metrics = compute_metrics(cam_samples, model, imgsz, conf, use_sahi, sahi_slice, sahi_overlap, device)
        print(f"mAP@50={metrics['mAP@50']:.3f}  R={metrics['recall']:.3f}  P={metrics['precision']:.3f}")
        results[cam] = metrics
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cameras", nargs="*", default=None,
                        help="Filter to specific camera prefixes (e.g. ch23 D15 D16)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Inference image size (default: 640)")
    parser.add_argument("--sahi", action="store_true",
                        help="Enable SAHI sliced inference (recommended for CCTV)")
    parser.add_argument("--sahi-slice", type=int, default=640,
                        help="SAHI slice size (default: 640)")
    parser.add_argument("--sahi-overlap", type=float, default=0.2,
                        help="SAHI overlap ratio (default: 0.2)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cuda:0, cpu, etc. (default: auto)")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Filter to specific model names (e.g. YOLOv11-n Egyptian-best)")
    parser.add_argument("--per-camera", action="store_true",
                        help="Show per-camera breakdown")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")
    parser.add_argument("--list-models", action="store_true",
                        help="List available models and exit")
    args = parser.parse_args()

    # List models
    if args.list_models:
        print("Available models:")
        for name, path in MODEL_REGISTRY:
            size_mb = path.stat().st_size / 1024 / 1024
            fmt = "pt" if path.suffix == ".pt" else "onnx"
            print(f"  {name:30s}  {size_mb:6.1f} MB  ({fmt})")
        return

    # Filter models
    models_to_test = MODEL_REGISTRY
    if args.models:
        models_to_test = [(n, p) for n, p in MODEL_REGISTRY if n in args.models]
        if not models_to_test:
            print(f"No matching models found for: {args.models}")
            print("Use --list-models to see available models.")
            sys.exit(1)

    # Load data
    print("Loading CCTV data ...")
    samples = load_cctv_data(args.cameras)
    if not samples:
        print("No CCTV samples found!")
        sys.exit(1)

    cameras = sorted(set(s.camera for s in samples))
    n_folders = len(set(s.folder for s in samples))
    print(f"  {len(samples)} images across {len(cameras)} cameras ({', '.join(cameras)}), {n_folders} folders")
    print()

    # Run benchmarks
    all_results = {}
    for model_name, model_path in models_to_test:
        print(f"{'='*60}")
        print(f"Model: {model_name}  ({model_path.name})")
        print(f"{'='*60}")

        try:
            from ultralytics import YOLO
            model = YOLO(str(model_path))
        except Exception as e:
            print(f"  [ERROR] Failed to load model: {e}")
            print()
            continue

        # Overall metrics
        print(f"  Running inference on {len(samples)} images (conf={args.conf}, imgsz={args.imgsz}"
              f"{', SAHI' if args.sahi else ''}) ...", flush=True)

        metrics = compute_metrics(
            samples, model,
            imgsz=args.imgsz,
            conf=args.conf,
            use_sahi=args.sahi,
            sahi_slice=args.sahi_slice,
            sahi_overlap=args.sahi_overlap,
            device=args.device,
        )

        print(f"  Results:")
        print(f"    mAP@50:       {metrics['mAP@50']:.4f}")
        print(f"    mAP@50-95:    {metrics['mAP@50-95']:.4f}")
        print(f"    Precision:    {metrics['precision']:.4f}")
        print(f"    Recall:       {metrics['recall']:.4f}")
        print(f"    TP/FP/FN:     {metrics['tp']}/{metrics['fp']}/{metrics['fn']}")
        print(f"    GT boxes:     {metrics['gt_boxes']}")
        print(f"    Speed:        {metrics['time_s']:.1f}s ({metrics['fps']:.1f} img/s)")
        print()

        # Per-camera breakdown
        if args.per_camera:
            print(f"  Per-camera breakdown:")
            cam_results = compute_per_camera(
                samples, model,
                imgsz=args.imgsz,
                conf=args.conf,
                use_sahi=args.sahi,
                sahi_slice=args.sahi_slice,
                sahi_overlap=args.sahi_overlap,
                device=args.device,
            )
            print()
            metrics["per_camera"] = cam_results

        all_results[model_name] = metrics

    # Summary table
    if all_results:
        print()
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        header = f"{'Model':<25} {'mAP@50':>8} {'mAP@50-95':>10} {'Prec':>8} {'Recall':>8} {'FPS':>8}"
        print(header)
        print("-" * 80)
        for name, res in sorted(all_results.items(), key=lambda x: -x[1]["mAP@50"]):
            print(f"{name:<25} {res['mAP@50']:>8.4f} {res['mAP@50-95']:>10.4f} "
                  f"{res['precision']:>8.4f} {res['recall']:>8.4f} {res['fps']:>8.1f}")
        print("-" * 80)

        # Recommendation
        best = max(all_results.items(), key=lambda x: x[1]["mAP@50"])
        print(f"\n🏆 Best mAP@50: {best[0]} ({best[1]['mAP@50']:.4f})")
        best_recall = max(all_results.items(), key=lambda x: x[1]["recall"])
        print(f"🎯 Best Recall: {best_recall[0]} ({best_recall[1]['recall']:.4f})")

        # Note about fine-tuning
        pt_models = [n for n in all_results if n.startswith("Egyptian-")]
        onnx_models = [n for n in all_results if n.startswith("YOLOv11-")]
        if pt_models and onnx_models:
            print(f"\n⚠️  Note: Only .pt models ({', '.join(pt_models)}) can be fine-tuned.")
            print(f"   ONNX models ({', '.join(onnx_models)}) are inference-only.")
            print(f"   To fine-tune YOLOv11, download .pt weights from HuggingFace:")
            print(f"   https://huggingface.co/morsetechlab/yolov11-license-plate-detection")

    # Save results
    if args.output and all_results:
        out_path = Path(args.output)
        # Convert numpy types for JSON serialization
        def sanitize(obj):
            if isinstance(obj, dict):
                return {k: sanitize(v) for k, v in obj.items()}
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            return obj

        out_path.write_text(json.dumps(sanitize(all_results), indent=2))
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
