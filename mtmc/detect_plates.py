"""Detect license plates in a video and save cropped plate images.

Uses the fine-tuned YOLOv11 license plate detection model with CUDA acceleration.

Usage:
    .venv/bin/python -m mtmc.detect_plates --video products/el-mohandes/videos/output.mp4
    .venv/bin/python -m mtmc.detect_plates --video products/el-mohandes/videos/output.mp4 --interval 1
    .venv/bin/python -m mtmc.detect_plates --video products/el-mohandes/videos/output.mp4 --device cpu
    .venv/bin/python -m mtmc.detect_plates --video products/el-mohandes/videos/output.mp4 --roi mtmc/rois.json
"""

import argparse
import csv
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
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

ROI_FILE = Path("mtmc/rois.json")


def _load_rois(roi_path: str | None) -> dict:
    if roi_path is None:
        return {}
    p = Path(roi_path)
    if not p.exists():
        print(f"[WARN] ROI file not found: {p}")
        return {}
    data = json.loads(p.read_text())
    rois = {}
    for cam_id, info in data.items():
        poly = info.get("polygon", [])
        if poly:
            rois[cam_id] = np.array(poly, dtype=np.float32)
    print(f"Loaded {len(rois)} ROI polygons from {p}")
    return rois


def _extract_camera_id(filename: str) -> str | None:
    stem = Path(filename).stem
    m = re.match(r"^(D\d+|ch\d+)", stem)
    if m:
        return m.group(1)
    m = re.search(r"[-_](D\d+|ch\d+)[-_]", stem)
    if m:
        return m.group(1)
    m = re.search(r"(D\d+)", stem)
    if m:
        return m.group(1)
    m = re.search(r"(ch\d+)", stem, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    return None


def _point_in_roi(x: int, y: int, polygon: np.ndarray | None) -> bool:
    if polygon is None or len(polygon) < 3:
        return True
    return cv2.pointPolygonTest(polygon, (float(x), float(y)), False) >= 0


def main():
    parser = argparse.ArgumentParser(description="Detect and crop license plates from video")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--model", default="finetuned", choices=list(MODEL_MAP.keys()),
                        help="Plate detection model (default: finetuned = the deployed fine-tuned detector)")
    parser.add_argument("--imgsz", type=int, default=2560, help="YOLO input size")
    parser.add_argument("--conf", type=float, default=0.10, help="Confidence threshold")
    parser.add_argument("--skip", type=int, default=1, help="Process every N-th frame")
    parser.add_argument("--interval", type=float, default=None,
                        help="Take one frame every N seconds (overrides --skip)")
    parser.add_argument("--padding", type=int, default=15, help="Padding around plate crop (px)")
    parser.add_argument("--device", default=None,
                        help="Device: cuda:0, cpu  (default: auto-detect GPU)")
    parser.add_argument("--out", default=None, help="Output directory")
    parser.add_argument("--roi", default=None, help="ROI JSON file for road polygon filter")
    args = parser.parse_args()

    video_path = args.video
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load ROI
    roi_path = args.roi
    if roi_path is None and ROI_FILE.exists():
        roi_path = str(ROI_FILE)
    rois = _load_rois(roi_path)

    video_name = Path(video_path).stem
    out_dir = Path(args.out) if args.out else Path("mtmc/output/plate_crops") / video_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading plate detection model: {args.model} on {device}")
    model = YOLO(str(MODEL_MAP[args.model]))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {video_path}  ({w}x{h} @ {fps:.1f}fps, {total_frames} frames)")

    # ROI
    cam_id = _extract_camera_id(video_name)
    roi_poly = rois.get(cam_id) if cam_id else None
    if roi_poly is not None:
        print(f"ROI: {len(roi_poly)} points for camera {cam_id}")
    elif rois:
        print(f"ROI: none for camera {cam_id or video_name} (accepting all)")

    # Resolve interval -> skip
    effective_skip = args.skip
    if args.interval is not None:
        effective_skip = max(1, int(round(fps * args.interval)))
        print(f"Interval: {args.interval}s → every {effective_skip} frames ({fps/effective_skip:.1f} samples/s)")

    csv_path = out_dir / "plates.csv"
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["frame", "crop_file", "x1", "y1", "x2", "y2", "conf"])

    frame_idx = 0
    saved = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % effective_skip != 0:
            frame_idx += 1
            continue

        results = model(frame, imgsz=args.imgsz, conf=args.conf, verbose=False, device=device)

        for r in results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # ROI filter
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                if not _point_in_roi(cx, cy, roi_poly):
                    continue

                # Add padding
                pad = args.padding
                crop_y1 = max(0, y1 - pad)
                crop_y2 = min(h, y2 + pad)
                crop_x1 = max(0, x1 - pad)
                crop_x2 = min(w, x2 + pad)

                crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                if crop.size == 0:
                    continue

                fname = f"frame{frame_idx:05d}_plate{saved:04d}.jpg"
                cv2.imwrite(str(out_dir / fname), crop)
                writer.writerow([frame_idx, fname, x1, y1, x2, y2, f"{conf:.3f}"])
                saved += 1

        frame_idx += 1
        if frame_idx % 200 == 0:
            elapsed = time.time() - t0
            fps_proc = frame_idx / elapsed if elapsed > 0 else 0
            print(f"  Frame {frame_idx}/{total_frames}  ({fps_proc:.1f} fps)  plates: {saved}")

    cap.release()
    csv_file.close()

    elapsed = time.time() - t0
    print(f"\nDone! Processed {frame_idx} frames in {elapsed:.1f}s ({frame_idx/elapsed:.1f} fps)")
    print(f"Saved {saved} plate crops to: {out_dir}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
