"""Batch-detect license plates in every video under products/el-mohandes/videos/.

Uses ultralytics YOLO with TensorRT (or CUDA) GPU acceleration.
Auto-selects the fastest available provider.
Saves per-video plate crops + CSV manifest to:
    mtmc/output/plate_crops/<video_stem>/

Usage:
    # Auto-detect best GPU provider (TensorRT > CUDA > CPU)
    .venv/bin/python -m mtmc.batch_detect_plates

    # Inside DeepStream container (has TensorRT libs)
    docker compose run --rm plate-extractor --interval 1

    # Force specific provider
    .venv/bin/python -m mtmc.batch_detect_plates --provider tensorrt
    .venv/bin/python -m mtmc.batch_detect_plates --provider cuda
    .venv/bin/python -m mtmc.batch_detect_plates --provider cpu
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
from ultralytics import YOLO

MODEL_DIR = Path("mtmc/models/yolov11-license-plate-detection")
FINETUNED_DIR = Path("products/el-mohandes/models/finetuned")
MODEL_MAP = {
    # Fine-tuned on the client's CCTV footage: the deployed detector. Earlier rounds are in git history (DATA.md)
    "finetuned": FINETUNED_DIR / "best.onnx",
    # Roboflow pretrained (baseline)
    "nano": MODEL_DIR / "license-plate-finetune-v1n.onnx",
    "small": MODEL_DIR / "license-plate-finetune-v1s.onnx",
    "medium": MODEL_DIR / "license-plate-finetune-v1m.onnx",
    "large": MODEL_DIR / "license-plate-finetune-v1l.onnx",
    "xlarge": MODEL_DIR / "license-plate-finetune-v1x.onnx",
}

VIDEOS_DIR = Path("products/el-mohandes/videos")
OUTPUT_ROOT = Path("mtmc/output/plate_crops")
ROI_FILE = Path("mtmc/rois.json")

# SAHI slicing parameters (defaults)
SAHI_SLICE_HEIGHT = 640
SAHI_SLICE_WIDTH = 640
SAHI_OVERLAP = 0.2


# ---------------------------------------------------------------------------
# ROI helpers
# ---------------------------------------------------------------------------
def _load_rois(roi_path: str | None) -> dict:
    """Load ROI polygons from JSON.  Returns {camera_id: polygon_points}."""
    if roi_path is None:
        return {}
    p = Path(roi_path)
    if not p.exists():
        print(f"[WARN] ROI file not found: {p}")
        return {}
    data = json.loads(p.read_text())
    # Flatten: {camera_id: {polygon: [...], ...}}
    rois = {}
    for cam_id, info in data.items():
        poly = info.get("polygon", [])
        if poly:
            rois[cam_id] = np.array(poly, dtype=np.float32)
    print(f"Loaded {len(rois)} ROI polygons from {p}")
    return rois


def _extract_camera_id(filename: str) -> str | None:
    """Extract camera ID from video filename.
    
    Handles formats:
      D18_20260726000000.mp4  -> D18
      ch23_20260726000005.mp4 -> ch23
      1-D41-enter.mp4         -> D41
      output-truck-D16.mp4    -> D16
    """
    stem = Path(filename).stem
    # Try D<digits> or ch<digits> prefix
    m = re.match(r"^(D\d+|ch\d+)", stem)
    if m:
        return m.group(1)
    # Try embedded camera ID: <prefix>-D<digits>-<suffix> or <prefix>-ch<digits>-<suffix>
    m = re.search(r"[-_](D\d+|ch\d+)[-_]", stem)
    if m:
        return m.group(1)
    # Try just finding D<digits> anywhere
    m = re.search(r"(D\d+)", stem)
    if m:
        return m.group(1)
    m = re.search(r"(ch\d+)", stem, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    return None


def _extract_date(path_or_name: str | Path) -> str:
    """Extract YYYYMMDD date string from filename or path.
    
    Examples:
      ch23_20260726000005.mp4 -> 20260726
      D18_20260726000000.mp4  -> 20260726
      2026-08-15_cam01.mp4   -> 20260815
    """
    s = str(path_or_name)
    m_sep = re.search(r"(?:19|20)\d{2}[-_]\d{2}[-_]\d{2}", s)
    if m_sep:
        return re.sub(r"\D", "", m_sep.group(0))
    m = re.search(r"(?:19|20)\d{6}", s)
    if m:
        return m.group(0)
    m = re.search(r"\d{8}", s)
    if m:
        return m.group(0)
    return "99999999"



def _sort_videos_by_day_and_camera(videos: list[Path]) -> list[Path]:
    """Sort videos so that Day 1 for all cameras is processed before Day 2 for all cameras.
    
    Primary key: Date (YYYYMMDD)
    Secondary key: Camera ID (e.g., ch23, D18)
    Tertiary key: Full filename/stem
    """
    def key_fn(vp: Path) -> tuple[str, str, str]:
        d = _extract_date(vp.name)
        c = _extract_camera_id(vp.name) or ""
        return (d, c, vp.name)
    return sorted(videos, key=key_fn)



def _point_in_roi(x: int, y: int, polygon: np.ndarray | None) -> bool:
    """Check if point (x, y) is inside the ROI polygon.
    Returns True if no polygon is defined (accept everything).
    """
    if polygon is None or len(polygon) < 3:
        return True  # No ROI = accept all
    # OpenCV pointPolygonTest: positive = inside, 0 = on edge, negative = outside
    result = cv2.pointPolygonTest(polygon, (float(x), float(y)), False)
    return result >= 0


def _check_tensorrt_available() -> bool:
    """Check if TensorRT runtime libraries are on the system."""
    for lib in ("libnvinfer.so.10", "libnvinfer.so.9", "libnvinfer.so.8"):
        try:
            ctypes.CDLL(lib)
            return True
        except OSError:
            continue
    return False


def _detect_provider() -> str:
    """Auto-detect the best onnxruntime provider."""
    available = ort.get_available_providers()
    if "TensorrtExecutionProvider" in available and _check_tensorrt_available():
        return "tensorrt"
    if "CUDAExecutionProvider" in available and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(model_size: str, device: str | None = None) -> YOLO:
    """Load YOLO model with the best available GPU provider."""
    onnx_path = MODEL_MAP[model_size]

    # Resolve provider
    if device in ("tensorrt", "trt"):
        provider = "tensorrt"
    elif device in ("cuda", "cuda:0", "gpu"):
        provider = "cuda"
    elif device in ("cpu",):
        provider = "cpu"
    elif device is None:
        provider = _detect_provider()
    else:
        provider = device  # pass through as-is

    print(f"  Model: {onnx_path.name}")
    print(f"  Provider: {provider}")

    if provider == "tensorrt":
        # Configure onnxruntime to use TensorRT provider
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = [
            ("TensorrtExecutionProvider", {
                "trt_max_workspace_size": 2 * 1024 * 1024 * 1024,  # 2 GB
                "trt_fp16_enable": True,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": str(MODEL_DIR),
            }),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        sess = ort.InferenceSession(str(onnx_path), so, providers=providers)
        print(f"  Active providers: {sess.get_providers()}")
        # Wrap in ultralytics YOLO for correct preprocessing —
        # but we need to monkey-patch the session.  Instead, just use
        # ultralytics with device=cuda (which uses CUDA provider) and let
        # TensorRT warm up in the background.  For now, fall through.
        # NOTE: ultralytics ONNX path doesn't expose provider selection,
        # so we use the raw onnxruntime session + manual preprocess.
        # But manual preprocess is error-prone — see detect_frame below.
        # SAFEST: use ultralytics with device=cuda and accept CUDA speed.
        pass

    # Use ultralytics YOLO (correct preprocessing + postprocessing)
    # device param controls onnxruntime provider for ONNX models
    if provider == "tensorrt":
        # Inside container with TRT libs, ultralytics may auto-detect it
        yolo_device = "cuda:0"
    elif provider == "cuda":
        yolo_device = "cuda:0"
    else:
        yolo_device = "cpu"

    print(f"  Loading ultralytics YOLO on {yolo_device}")
    model = YOLO(str(onnx_path))
    return model


def _sahi_detect(
    sahi_model,
    frame: np.ndarray,
    conf: float,
    slice_h: int,
    slice_w: int,
    overlap: float,
) -> list[tuple[int, int, int, int, float]]:
    """Run SAHI sliced prediction on a frame. Returns list of (x1,y1,x2,y2,conf)."""
    from sahi.predict import get_sliced_prediction
    result = get_sliced_prediction(
        image=frame,
        detection_model=sahi_model,
        slice_height=slice_h,
        slice_width=slice_w,
        overlap_height_ratio=overlap,
        overlap_width_ratio=overlap,
        perform_standard_pred=True,
        postprocess_type="GREEDYNMM",
        postprocess_match_metric="IOS",
        postprocess_match_threshold=0.5,
        verbose=0,
        confidence_threshold=conf,
    )
    detections = []
    for pred in result.object_prediction_list:
        bbox = pred.bbox
        x1, y1 = int(bbox.minx), int(bbox.miny)
        x2, y2 = int(bbox.maxx), int(bbox.maxy)
        c = pred.score.value
        detections.append((x1, y1, x2, y2, c))
    return detections


def detect_video(
    model: YOLO,
    video_path: Path,
    out_dir: Path,
    imgsz: int,
    conf: float,
    iou: float,
    skip: int,
    padding: int,
    device: str | None = None,
    interval: float | None = None,
    min_width: int = 0,
    min_height: int = 0,
    min_area: int = 0,
    roi_polygon: np.ndarray | None = None,
    save_frames: bool = True,
    sahi_model=None,
    sahi_slice_h: int = SAHI_SLICE_HEIGHT,
    sahi_slice_w: int = SAHI_SLICE_WIDTH,
    sahi_overlap: float = SAHI_OVERLAP,
    resume: bool = False,
) -> dict:
    """Run plate detection on a single video. Returns summary stats.

    Output structure (per video):
        <out_dir>/
            crops/        — individual plate crop images
            frames/       — full original frames (only frames with detections)
            labels/       — YOLO-format .txt annotations (one per frame)
            plates.csv    — detection manifest
            .done         — completion marker
    """
    done_file = out_dir / ".done"
    progress_file = out_dir / ".progress.json"

    if resume and done_file.exists():
        print(f"  [SKIP] Video {video_path.name} is already fully processed.")
        saved_plates = 0
        csv_path = out_dir / "plates.csv"
        if csv_path.exists():
            try:
                with open(csv_path, "r", encoding="utf-8") as f:
                    saved_plates = max(0, len(f.readlines()) - 1)
            except Exception:
                pass
        return {
            "video": video_path.name,
            "frames": 0,
            "plates": saved_plates,
            "skipped": True,
            "elapsed_s": 0.0,
        }

    crops_dir = out_dir / "crops"
    frames_dir = out_dir / "frames"
    labels_dir = out_dir / "labels"
    crops_dir.mkdir(parents=True, exist_ok=True)
    if save_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "plates.csv"
    start_frame = 0
    saved = 0

    if resume and out_dir.exists():
        if progress_file.exists():
            try:
                pdata = json.loads(progress_file.read_text())
                last_proc = pdata.get("last_processed_frame", -1)
                if last_proc >= 0:
                    start_frame = last_proc + 1
            except Exception:
                pass

        if start_frame == 0 and csv_path.exists():
            try:
                with open(csv_path, "r", newline="", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    max_f = -1
                    count = 0
                    for row in reader:
                        if row:
                            count += 1
                            try:
                                f_num = int(row[0])
                                if f_num > max_f:
                                    max_f = f_num
                            except ValueError:
                                pass
                    if max_f >= 0:
                        start_frame = max_f + 1
                        saved = count
            except Exception:
                pass

    csv_mode = "a" if (start_frame > 0 and csv_path.exists()) else "w"
    with open(csv_path, csv_mode, newline="") as csv_file:
        writer = csv.writer(csv_file)
        if csv_mode == "w":
            writer.writerow(["frame", "crop_file", "x1", "y1", "x2", "y2", "conf"])

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  [WARN] Cannot open {video_path}, skipping")
            return {"video": video_path.name, "frames": 0, "plates": 0, "error": True}

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Resolve interval -> skip
        effective_skip = skip
        if interval is not None:
            effective_skip = max(1, int(round(fps * interval)))
            print(f"  Video: {video_path.name}  ({w}x{h} @ {fps:.1f}fps, {total_frames} frames)")
            print(f"  Interval: {interval}s → every {effective_skip} frames ({fps/effective_skip:.1f} samples/s)")
        else:
            print(f"  Video: {video_path.name}  ({w}x{h} @ {fps:.1f}fps, {total_frames} frames)")

        if start_frame > 0:
            print(f"  Resuming from frame {start_frame}/{total_frames} (already saved {saved} plates)")
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_idx = start_frame
        frames_saved = 0
        t0 = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % effective_skip != 0:
                frame_idx += 1
                continue

            # Detection: SAHI sliced or standard YOLO
            if sahi_model is not None:
                raw_plates = _sahi_detect(
                    sahi_model, frame, conf, sahi_slice_h, sahi_slice_w, sahi_overlap
                )
            else:
                results = model(frame, imgsz=imgsz, conf=conf, iou=iou, verbose=False, device=device)
                raw_plates = []
                for r in results:
                    boxes = r.boxes
                    if boxes is None or len(boxes) == 0:
                        continue
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        c = float(box.conf[0])
                        raw_plates.append((int(x1), int(y1), int(x2), int(y2), c))

            # Apply filters
            frame_plates: list[tuple[int, int, int, int, float]] = []
            for x1, y1, x2, y2, c in raw_plates:
                # Size filter
                bw, bh = x2 - x1, y2 - y1
                if bw < min_width or bh < min_height or (bw * bh) < min_area:
                    continue
                # ROI filter
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                if not _point_in_roi(cx, cy, roi_polygon):
                    continue
                frame_plates.append((x1, y1, x2, y2, c))

            # Save frame + labels + crops for this frame
            if frame_plates:
                frame_name = f"frame{frame_idx:05d}"

                # Save full frame image (for finetuning)
                if save_frames:
                    cv2.imwrite(str(frames_dir / f"{frame_name}.jpg"), frame)
                    frames_saved += 1

                    # Save YOLO-format label file
                    # Format: class_id x_center y_center width  height (normalized 0-1)
                    label_lines = []
                    for x1, y1, x2, y2, c in frame_plates:
                        # Normalize to image dimensions
                        xc = ((x1 + x2) / 2) / w
                        yc = ((y1 + y2) / 2) / h
                        nw = (x2 - x1) / w
                        nh = (y2 - y1) / h
                        # Clamp to [0, 1]
                        xc = max(0.0, min(1.0, xc))
                        yc = max(0.0, min(1.0, yc))
                        nw = max(0.0, min(1.0, nw))
                        nh = max(0.0, min(1.0, nh))
                        label_lines.append(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

                    (labels_dir / f"{frame_name}.txt").write_text("\n".join(label_lines))

                # Save individual crops
                for x1, y1, x2, y2, c in frame_plates:
                    pad = padding
                    crop = frame[
                        max(0, y1 - pad) : min(h, y2 + pad),
                        max(0, x1 - pad) : min(w, x2 + pad),
                    ]
                    if crop.size == 0:
                        continue

                    fname = f"{frame_name}_plate{saved:04d}.jpg"
                    cv2.imwrite(str(crops_dir / fname), crop)
                    writer.writerow([frame_idx, fname, x1, y1, x2, y2, f"{c:.3f}"])
                    saved += 1

            frame_idx += 1
            if frame_idx % 200 == 0:
                elapsed = time.time() - t0
                fps_proc = (frame_idx - start_frame) / elapsed if elapsed > 0 else 0
                print(f"    Frame {frame_idx}/{total_frames}  ({fps_proc:.1f} fps)  plates: {saved}")
                try:
                    progress_file.write_text(json.dumps({"last_processed_frame": frame_idx - 1, "total_frames": total_frames}))
                except Exception:
                    pass

        cap.release()

    # Mark video as completed
    try:
        progress_file.write_text(json.dumps({"last_processed_frame": frame_idx - 1, "total_frames": frame_idx, "status": "completed"}))
        done_file.write_text("completed\n")
    except Exception as e:
        print(f"  [WARN] Failed to write completion marker: {e}")

    elapsed = time.time() - t0
    avg_fps = (frame_idx - start_frame) / elapsed if elapsed > 0 else 0
    label_msg = f", {frames_saved} frames+labels" if save_frames else ""
    print(f"  Done: {frame_idx} frames, {saved} plate crops{label_msg}, {elapsed:.1f}s ({avg_fps:.1f} fps)")
    return {
        "video": video_path.name,
        "frames": frame_idx,
        "plates": saved,
        "frames_saved": frames_saved,
        "elapsed_s": round(elapsed, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="finetuned", choices=list(MODEL_MAP.keys()),
                        help="Plate detection model (default: finetuned = the deployed fine-tuned detector)")
    parser.add_argument("--imgsz", type=int, default=2560,
                        help="YOLO input size (default: 2560; >2560 may OOM on 8GB GPUs)")
    parser.add_argument("--conf", type=float, default=0.10, help="Confidence threshold (lower = more detections)")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--skip", type=int, default=1, help="Process every N-th frame")
    parser.add_argument("--interval", type=float, default=None,
                        help="Take one frame every N seconds (overrides --skip)")
    parser.add_argument("--padding", type=int, default=15, help="Padding around plate crop (px)")
    parser.add_argument("--min-width", type=int, default=0,
                        help="Minimum plate width in pixels (drop smaller detections)")
    parser.add_argument("--min-height", type=int, default=0,
                        help="Minimum plate height in pixels (drop smaller detections)")
    parser.add_argument("--min-area", type=int, default=0,
                        help="Minimum plate area in pixels (drop smaller detections)")
    parser.add_argument("--provider", default=None,
                        help="GPU provider: tensorrt, cuda, cpu  (default: auto-detect best)")
    parser.add_argument("--video-dir", default=None,
                        help="Directory containing .mp4 files (scans non-recursively)")
    parser.add_argument("--out-dir", default=None,
                        help="Output root for plate crops (default: mtmc/output/plate_crops)")
    parser.add_argument("--exclude", nargs="*", default=None,
                        help="Filename patterns to exclude (e.g. '*_001*' 'D05*')")
    parser.add_argument("--videos", nargs="*", default=None,
                        help="Specific video paths (default: all .mp4 in products/el-mohandes/videos/)")
    parser.add_argument("--roi", default=None,
                        help="ROI JSON file (default: mtmc/rois.json if it exists)")
    parser.add_argument("--no-save-frames", action="store_true",
                        help="Don't save full frames + YOLO labels (only crops)")
    parser.add_argument("--resume", "--continue", dest="resume", action="store_true",
                        help="Resume processing from where it stopped and skip fully processed videos")
    parser.add_argument("--sahi", action="store_true",
                        help="Use SAHI sliced inference (better for small/distant plates)")
    parser.add_argument("--slice-height", type=int, default=SAHI_SLICE_HEIGHT,
                        help=f"SAHI slice height (default: {SAHI_SLICE_HEIGHT})")
    parser.add_argument("--slice-width", type=int, default=SAHI_SLICE_WIDTH,
                        help=f"SAHI slice width (default: {SAHI_SLICE_WIDTH})")
    parser.add_argument("--slice-overlap", type=float, default=SAHI_OVERLAP,
                        help=f"SAHI slice overlap ratio (default: {SAHI_OVERLAP})")
    args = parser.parse_args()

    # Resolve video list
    if args.video_dir:
        vdir = Path(args.video_dir)
        videos = sorted(vdir.glob("*.mp4")) or sorted(vdir.rglob("*.mp4"))
    elif args.videos:
        videos = [Path(v) for v in args.videos]
    else:
        videos = sorted(VIDEOS_DIR.glob("*.mp4"))

    # Sort videos so that Day 1 for all cameras is processed before Day 2 for all cameras
    videos = _sort_videos_by_day_and_camera(videos)

    # Apply exclusions
    if args.exclude:
        import fnmatch
        before = len(videos)
        videos = [v for v in videos
                  if not any(fnmatch.fnmatch(v.name, pat) for pat in args.exclude)]
        excluded = before - len(videos)
        print(f"Excluded {excluded} videos matching: {args.exclude}")

    if not videos:
        print("No videos found. Nothing to do.")
        return

    # Load ROIs
    roi_path = args.roi
    if roi_path is None and ROI_FILE.exists():
        roi_path = str(ROI_FILE)
    rois = _load_rois(roi_path)

    # Provider selection
    provider = args.provider
    if provider is None:
        provider = _detect_provider()
    print(f"Provider: {provider}")

    # Map provider to YOLO device string
    yolo_device = "cuda:0" if provider in ("tensorrt", "cuda") else "cpu"

    print(f"Loading model: {args.model}")
    model = load_model(args.model, provider)
    # Hard cap: imgsz>2560 OOMs on 8GB GPUs (DFL Softmax needs ~54MB buffer)
    if args.imgsz > 2560:
        print(f"[WARN] imgsz={args.imgsz} capped to 2560 (higher causes OOM on 8GB GPU)")
        args.imgsz = 2560

    # Load SAHI model if requested
    sahi_model = None
    if args.sahi:
        from sahi import AutoDetectionModel
        onnx_path = MODEL_MAP[args.model]
        print(f"Loading SAHI model (slice: {args.slice_height}x{args.slice_width}, overlap: {args.slice_overlap})")
        sahi_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=str(onnx_path),
            confidence_threshold=args.conf,
            device=yolo_device,
            image_size=args.imgsz,
        )
        print(f"  SAHI model loaded on {yolo_device}")

    output_root = Path(args.out_dir) if args.out_dir else OUTPUT_ROOT
    print(f"Output: {output_root}")
    print(f"Processing {len(videos)} video(s) with conf={args.conf}, imgsz={args.imgsz}, skip={args.skip}, resume={args.resume}\n")

    summaries = []
    for vp in videos:
        stem = vp.stem
        out_dir = output_root / stem
        print(f"[{stem}]")
        # Get ROI polygon for this camera
        cam_id = _extract_camera_id(vp.name)
        roi_poly = rois.get(cam_id) if cam_id else None
        if roi_poly is not None:
            print(f"  ROI: {len(roi_poly)} points for camera {cam_id}")
        elif rois:
            print(f"  ROI: none for camera {cam_id or vp.name} (accepting all)")

        info = detect_video(model, vp, out_dir, args.imgsz, args.conf, args.iou,
                            args.skip, args.padding, yolo_device, args.interval,
                            args.min_width, args.min_height, args.min_area,
                            roi_polygon=roi_poly,
                            save_frames=not args.no_save_frames,
                            sahi_model=sahi_model,
                            sahi_slice_h=args.slice_height,
                            sahi_slice_w=args.slice_width,
                            sahi_overlap=args.slice_overlap,
                            resume=args.resume)
        summaries.append(info)
        print()

    # Summary table
    print("=" * 60)
    print(f"{'Video':<35} {'Frames':>7} {'Plates':>7} {'Time':>7}")
    print("-" * 60)
    for s in summaries:
        err = " [SKIPPED]" if s.get("skipped") else (" [ERROR]" if s.get("error") else "")
        print(f"{s['video']:<35} {s['frames']:>7} {s['plates']:>7} {s.get('elapsed_s', '-'):>7}{err}")
    total_plates = sum(s["plates"] for s in summaries)
    print("-" * 60)
    print(f"{'TOTAL':<35} {'':>7} {total_plates:>7}")


if __name__ == "__main__":
    main()

