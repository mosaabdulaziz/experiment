#!/usr/bin/env python3
"""License plate detection + OCR pipeline.

Runs YOLOv11 for plate detection (via ultralytics on GPU), then PP-OCRv3
ONNX for text recognition on each detected plate crop.

Usage:
    .venv/bin/python -m mtmc.plate_ocr --video products/el-mohandes/videos/output.mp4
    .venv/bin/python -m mtmc.plate_ocr --video products/el-mohandes/videos/output.mp4 --conf 0.25 --skip 2
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
import onnxruntime as ort

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_MODELS_DIR = Path(__file__).resolve().parent / "models"
_YOLO_PATH = _MODELS_DIR / "yolov11-license-plate-detection" / "license-plate-finetune-v1s.onnx"
_OCR_DIR = _MODELS_DIR / "paddleocr-onnx"
_OCR_PATH = _OCR_DIR / "rec.onnx"
_DICT_PATH = _OCR_DIR / "dict.txt"

# ---------------------------------------------------------------------------
# Character dictionary for CTC decoding (from dict.txt)
# ---------------------------------------------------------------------------
_char_dict: list[str] | None = None


def _load_char_dict() -> list[str]:
    global _char_dict
    if _char_dict is None:
        _char_dict = ["blank"] + _DICT_PATH.read_text(encoding="utf-8").splitlines()
    return _char_dict


# ---------------------------------------------------------------------------
# YOLO detection via ultralytics (uses torch CUDA)
# ---------------------------------------------------------------------------
DEFAULT_YOLO_IMGsz = 1280


def yolo_detect(
    model: YOLO,
    img_bgr: np.ndarray,
    conf_thresh: float = 0.4,
    iou_thresh: float = 0.45,
    imgsz: int = DEFAULT_YOLO_IMGsz,
) -> np.ndarray:
    """Run YOLO detection via ultralytics.  Returns (N,5) [x1,y1,x2,y2,conf]."""
    results = model.predict(
        img_bgr,
        imgsz=imgsz,
        conf=conf_thresh,
        iou=iou_thresh,
        verbose=False,
    )
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return np.empty((0, 5), dtype=np.float32)
    boxes = results[0].boxes.xyxy.cpu().numpy()  # (N, 4)
    confs = results[0].boxes.conf.cpu().numpy()   # (N,)
    return np.hstack([boxes, confs[:, None]]).astype(np.float32)


# ---------------------------------------------------------------------------
# OCR helpers
# ---------------------------------------------------------------------------
OCR_INPUT_H = 48
OCR_INPUT_W = 320
OCR_PAD_VALUE = 0  # black padding


def _ctc_decode(logits: np.ndarray, char_dict: list[str]) -> str:
    """CTC greedy decode.  logits: (seq_len, vocab_size)."""
    indices = logits.argmax(axis=1)
    chars = []
    prev = -1
    blank_id = 0  # CTC blank is index 0
    for idx in indices:
        if idx != prev:
            if idx != blank_id and idx < len(char_dict):
                chars.append(char_dict[idx])
            prev = idx
    return "".join(chars)


def ocr_recognize(
    sess: ort.InferenceSession,
    img_bgr: np.ndarray,
    char_dict: list[str],
) -> tuple[str, float]:
    """Run PP-OCR text recognition on a single image crop.

    Returns (text, confidence).
    """
    h, w = img_bgr.shape[:2]
    target_h = OCR_INPUT_H
    target_w = OCR_INPUT_W
    scale = target_h / h
    new_w = min(int(w * scale), target_w)
    resized = cv2.resize(img_bgr, (new_w, target_h), interpolation=cv2.INTER_LINEAR)

    # Pad to target width
    if new_w < target_w:
        pad = np.full((target_h, target_w - new_w, 3), 0, dtype=np.uint8)
        resized = np.hstack([resized, pad])
    elif new_w > target_w:
        resized = resized[:, :target_w]

    # Normalize: PP-OCR standard — /255 then (x-0.5)/0.5
    blob = resized[:, :, ::-1].astype(np.float32) / 255.0  # BGR -> RGB
    blob = (blob - 0.5) / 0.5
    blob = blob.transpose(2, 0, 1)  # CHW
    blob = np.expand_dims(blob, 0)  # NCHW

    logits = sess.run(None, {"x": blob})[0]  # (1, seq_len, vocab)
    logits = logits[0]  # (seq_len, vocab)

    text = _ctc_decode(logits, char_dict)
    # Confidence: sum of max probs for non-blank positions
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    max_probs = probs.max(axis=1)
    blank_id = 0
    non_blank = max_probs[1:]  # skip blank index
    if len(non_blank) > 0:
        conf = float(non_blank[non_blank > 0.3].mean()) if (non_blank > 0.3).any() else 0.0
    else:
        conf = 0.0
    return text, conf


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_pipeline(
    video_path: str,
    yolo_conf: float = 0.4,
    yolo_iou: float = 0.45,
    imgsz: int = DEFAULT_YOLO_IMGsz,
    frame_skip: int = 1,
    out_dir: str | None = None,
    max_frames: int | None = None,
) -> list[dict]:
    """Process video: detect plates + OCR.  Returns list of per-frame results."""
    # Load YOLO model via ultralytics (torch CUDA)
    print(f"[plate_ocr] Loading YOLO: {_YOLO_PATH.name}")
    yolo_model = YOLO(str(_YOLO_PATH))

    # Load OCR model via onnxruntime (CPU — small model, fast enough)
    print(f"[plate_ocr] Loading OCR: {_OCR_PATH.name}")
    ocr_sess = ort.InferenceSession(str(_OCR_PATH), providers=["CPUExecutionProvider"])
    char_dict = _load_char_dict()
    print(f"[plate_ocr] OCR vocab size: {len(char_dict)} (dict lines: {len(char_dict)-1})")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[plate_ocr] Video: {vid_w}x{vid_h} @ {fps:.1f}fps, {total_frames} frames")

    # Output setup
    out_path = None
    writer = None
    if out_dir:
        out_dir_path = Path(out_dir)
        out_dir_path.mkdir(parents=True, exist_ok=True)
        out_path = out_dir_path / "plate_ocr_results.json"
        vid_out = out_dir_path / "annotated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(vid_out), fourcc, fps, (vid_w, vid_h))
        print(f"[plate_ocr] Writing annotated video to {vid_out}")

    results: list[dict] = []
    frame_idx = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames and frame_idx >= max_frames:
            break
        if frame_skip > 1 and frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        # --- Detection (GPU via ultralytics) ---
        dets = yolo_detect(yolo_model, frame, conf_thresh=yolo_conf, iou_thresh=yolo_iou, imgsz=imgsz)

        frame_results: list[dict] = []
        annotated = frame.copy()

        for det in dets:
            x1, y1, x2, y2, conf = det
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # Generous padding: 15% of box size, min 8px
            bw, bh = x2 - x1, y2 - y1
            pad_x = max(int(bw * 0.15), 8)
            pad_y = max(int(bh * 0.15), 8)
            crop = frame[max(0, y1 - pad_y):min(vid_h, y2 + pad_y),
                         max(0, x1 - pad_x):min(vid_w, x2 + pad_x)]
            if crop.size == 0:
                continue

            # --- OCR (CPU onnxruntime) ---
            text, ocr_conf = ocr_recognize(ocr_sess, crop, char_dict)

            entry = {
                "box": [x1, y1, x2, y2],
                "det_conf": round(float(conf), 4),
                "plate_text": text,
                "ocr_conf": round(ocr_conf, 4),
            }
            frame_results.append(entry)

            # Draw on annotated frame
            color = (0, 255, 0) if text else (0, 0, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            label = f"{text} ({ocr_conf:.2f})" if text else f"? ({conf:.2f})"
            # Background rectangle for text readability
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(annotated, (x1, y1 - th - 12), (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        results.append({"frame": frame_idx, "plates": frame_results})

        if writer:
            writer.write(annotated)

        if frame_idx % 60 == 0:
            elapsed = time.time() - t0
            fps_proc = (frame_idx + 1) / elapsed if elapsed > 0 else 0
            print(f"[plate_ocr] frame {frame_idx}/{total_frames}  "
                  f"plates={len(dets)}  {fps_proc:.1f} fps  elapsed={elapsed:.1f}s")

        frame_idx += 1

    cap.release()
    if writer:
        writer.release()

    elapsed = time.time() - t0
    n_plates = sum(len(r["plates"]) for r in results)
    avg_fps = frame_idx / elapsed if elapsed > 0 else 0
    print(f"[plate_ocr] Done: {frame_idx} frames, {n_plates} plate detections, "
          f"{elapsed:.1f}s ({avg_fps:.1f} fps)")

    if out_path:
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"[plate_ocr] Results saved to {out_path}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True, help="path to input video")
    p.add_argument("--conf", type=float, default=0.25, help="YOLO detection confidence threshold")
    p.add_argument("--iou", type=float, default=0.45, help="YOLO NMS IoU threshold")
    p.add_argument("--imgsz", type=int, default=DEFAULT_YOLO_IMGsz,
                   help=f"YOLO input size (default: {DEFAULT_YOLO_IMGsz})")
    p.add_argument("--skip", type=int, default=1, help="process every Nth frame (1=all)")
    p.add_argument("--out", default=None, help="output directory (default: mtmc/output/plate_ocr)")
    p.add_argument("--max-frames", type=int, default=None, help="max frames to process")
    args = p.parse_args()

    out = args.out or str(Path(__file__).resolve().parent / "output" / "plate_ocr")
    run_pipeline(args.video, args.conf, args.iou, args.imgsz, args.skip, out, args.max_frames)


if __name__ == "__main__":
    main()
