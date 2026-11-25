#!/usr/bin/env python3
"""Batch OCR on existing plate crops using Arabic PP-OCRv3 ONNX model.

Reads all plate crop images from plate_crops/, runs text recognition on each,
and produces annotated copies with Arabic plate text overlaid.

Usage:
    .venv/bin/python -m mtmc.batch_ocr_plates
    .venv/bin/python -m mtmc.batch_ocr_plates --crops-dir mtmc/output/plate_crops
    .venv/bin/python -m mtmc.batch_ocr_plates --crops-dir mtmc/output/plate_crops --confidence 0.5
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_MODELS_DIR = Path(__file__).resolve().parent / "models"
_OCR_DIR = _MODELS_DIR / "paddleocr-onnx"
_OCR_PATH = _OCR_DIR / "rec.onnx"
_DICT_PATH = _OCR_DIR / "dict.txt"

# Arabic font for annotation (Noto Naskh Arabic is available on this system)
_ARABIC_FONT_PATHS = [
    Path("/usr/share/fonts/truetype/noto/NotoNaskhArabic-Bold.ttf"),
    Path("/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
]


def _find_font() -> str | None:
    for p in _ARABIC_FONT_PATHS:
        if p.exists():
            return str(p)
    return None


# ---------------------------------------------------------------------------
# CTC decode
# ---------------------------------------------------------------------------
def _load_char_dict() -> list[str]:
    return ["blank"] + _DICT_PATH.read_text(encoding="utf-8").splitlines()


def _ctc_decode(logits: np.ndarray, char_dict: list[str]) -> str:
    indices = logits.argmax(axis=1)
    chars, prev = [], -1
    for idx in indices:
        if idx != prev:
            if idx != 0 and idx < len(char_dict):
                chars.append(char_dict[idx])
            prev = idx
    return "".join(chars)


def _ctc_confidence(logits: np.ndarray) -> float:
    """Confidence: mean of max-prob for non-blank, non-pad positions."""
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    max_probs = probs.max(axis=1)
    indices = logits.argmax(axis=1)
    # Only count positions where the model predicted a real character (not blank)
    non_blank = [(max_probs[i], indices[i]) for i in range(len(indices)) if indices[i] != 0]
    if not non_blank:
        return 0.0
    chars = [p for p, _ in non_blank]
    # Mean confidence of decoded characters
    return float(np.mean(chars))


# ---------------------------------------------------------------------------
# OCR recognize (single crop)
# ---------------------------------------------------------------------------
OCR_INPUT_H = 48
OCR_INPUT_W = 320


def ocr_recognize(
    sess: ort.InferenceSession,
    img_bgr: np.ndarray,
    char_dict: list[str],
    upscale: int = 2,
) -> tuple[str, float]:
    """Run PP-OCR recognition on a single image crop.

    Args:
        upscale: Upscale factor for small crops (<100px height). 2 = double.
                 Helps the recognition model see details on tiny license plates.

    Returns:
        (text, confidence)
    """
    h, w = img_bgr.shape[:2]

    # Upscale small crops for better OCR accuracy
    if h < 100 and upscale > 1:
        img_bgr = cv2.resize(img_bgr, (w * upscale, h * upscale), interpolation=cv2.INTER_CUBIC)
        h, w = img_bgr.shape[:2]

    scale = OCR_INPUT_H / h
    new_w = min(int(w * scale), OCR_INPUT_W)
    resized = cv2.resize(img_bgr, (new_w, OCR_INPUT_H), interpolation=cv2.INTER_LINEAR)

    # Pad to target width
    if new_w < OCR_INPUT_W:
        pad = np.full((OCR_INPUT_H, OCR_INPUT_W - new_w, 3), 0, dtype=np.uint8)
        resized = np.hstack([resized, pad])
    elif new_w > OCR_INPUT_W:
        resized = resized[:, :OCR_INPUT_W]

    # PP-OCR preprocessing: BGR→RGB, /255, (x-0.5)/0.5
    blob = resized[:, :, ::-1].astype(np.float32) / 255.0
    blob = (blob - 0.5) / 0.5
    blob = blob.transpose(2, 0, 1)[np.newaxis]  # NCHW

    logits = sess.run(None, {"x": blob})[0][0]  # (seq_len, vocab)
    text = _ctc_decode(logits, char_dict)
    conf = _ctc_confidence(logits)
    return text, conf


# ---------------------------------------------------------------------------
# Arabic text annotation on image (PIL for proper RTL + reshaping)
# ---------------------------------------------------------------------------
def annotate_image(
    img_bgr: np.ndarray,
    text: str,
    conf: float,
    font_path: str | None,
    font_size: int = 18,
) -> np.ndarray:
    """Draw plate text on image using PIL for proper Arabic rendering.

    Returns annotated BGR image.
    """
    if not text:
        return img_bgr

    h, w = img_bgr.shape[:2]

    # Convert BGR → RGB PIL image
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    # Load font
    try:
        font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    # Reshape and reorder Arabic text for display
    display_text = text
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        reshaped = arabic_reshaper.reshape(text)
        display_text = get_display(reshaped)
    except ImportError:
        pass  # Fall back to raw text (may look broken for Arabic)

    # Compute text size
    bbox = draw.textbbox((0, 0), display_text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    # Draw background + text at top of crop
    pad = 3
    # Extend canvas downward if text is taller than image
    if th + 2 * pad > h:
        new_h = th + 2 * pad + 2
        extended = Image.new("RGB", (w, new_h), (0, 0, 0))
        extended.paste(pil_img, (0, 0))
        pil_img = extended
        draw = ImageDraw.Draw(pil_img)

    # Background rectangle
    draw.rectangle([0, 0, tw + 2 * pad, th + 2 * pad], fill=(0, 0, 0))
    # Text (white for contrast)
    draw.text((pad, pad), display_text, fill=(255, 255, 0), font=font)

    # Confidence badge
    conf_text = f"{conf:.0%}"
    conf_bbox = draw.textbbox((0, 0), conf_text, font=font)
    conf_w = conf_bbox[2] - conf_bbox[0]
    draw.rectangle([tw + 2 * pad, 0, tw + 2 * pad + conf_w + 2 * pad, th + 2 * pad], fill=(0, 100, 0))
    draw.text((tw + 3 * pad, pad), conf_text, fill=(255, 255, 255), font=font)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_batch_ocr(
    crops_dir: str,
    out_dir: str | None = None,
    min_confidence: float = 0.0,
    font_size: int = 18,
) -> dict:
    """Run OCR on all plate crops across all video subdirectories.

    Args:
        crops_dir: Root directory with per-video subfolders of plate crops.
        out_dir: Output directory for annotated images + results.
                  Defaults to <crops_dir>_annotated.
        min_confidence: Minimum OCR confidence to include in results.
        font_size: Font size for Arabic text annotation.

    Returns:
        Summary dict with per-video stats.
    """
    crops_root = Path(crops_dir)
    if not crops_root.exists():
        raise FileNotFoundError(f"Crops directory not found: {crops_dir}")

    out_root = Path(out_dir) if out_dir else crops_root.parent / (crops_root.name + "_annotated")
    out_root.mkdir(parents=True, exist_ok=True)

    # Load OCR model
    print(f"[batch_ocr] Loading OCR model: {_OCR_PATH.name}")
    char_dict = _load_char_dict()
    print(f"[batch_ocr] Vocab size: {len(char_dict)} (dict: {len(char_dict) - 1} chars)")

    sess = ort.InferenceSession(
        str(_OCR_PATH),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    print(f"[batch_ocr] Provider: {sess.get_providers()[0]}")

    font_path = _find_font()
    if font_path:
        print(f"[batch_ocr] Arabic font: {font_path}")

    # Find all video subdirectories
    video_dirs = sorted([d for d in crops_root.iterdir() if d.is_dir()])
    print(f"[batch_ocr] Found {len(video_dirs)} video directories")

    summary = {}
    total_plates = 0
    t0 = time.time()

    for vdir in video_dirs:
        vname = vdir.name
        crop_files = sorted(vdir.glob("*.jpg"))
        csv_file = vdir / "plates.csv"

        # Read existing CSV for bbox info
        bboxes: dict[str, dict] = {}
        if csv_file.exists():
            with open(csv_file) as f:
                for row in csv.DictReader(f):
                    bboxes[row["crop_file"]] = row

        # Output subdirectory
        v_out = out_root / vname
        v_out.mkdir(parents=True, exist_ok=True)
        # Copy CSV to output
        if csv_file.exists():
            import shutil
            shutil.copy2(csv_file, v_out / "plates.csv")

        # OCR results CSV
        ocr_csv = v_out / "ocr_results.csv"
        ocr_rows = []

        vt0 = time.time()
        plates_done = 0

        for cfile in crop_files:
            img = cv2.imread(str(cfile))
            if img is None:
                continue

            text, conf = ocr_recognize(sess, img, char_dict)

            if text and conf >= min_confidence:
                annotated = annotate_image(img, text, conf, font_path, font_size)
                cv2.imwrite(str(v_out / cfile.name), annotated)
            else:
                # Save original even if no text
                cv2.imwrite(str(v_out / cfile.name), img)

            # Gather bbox from detection CSV
            bbox_info = bboxes.get(cfile.name, {})

            ocr_rows.append({
                "crop_file": cfile.name,
                "frame": bbox_info.get("frame", ""),
                "plate_text": text,
                "ocr_confidence": round(conf, 4),
                "det_confidence": bbox_info.get("conf", ""),
                "x1": bbox_info.get("x1", ""),
                "y1": bbox_info.get("y1", ""),
                "x2": bbox_info.get("x2", ""),
                "y2": bbox_info.get("y2", ""),
            })
            plates_done += 1

        # Write OCR results CSV
        if ocr_rows:
            with open(ocr_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(ocr_rows[0].keys()))
                writer.writeheader()
                writer.writerows(ocr_rows)

        v_elapsed = time.time() - vt0
        has_text = sum(1 for r in ocr_rows if r["plate_text"])
        print(f"  {vname}: {plates_done} crops, {has_text} with text, {v_elapsed:.1f}s")
        summary[vname] = {
            "total": plates_done,
            "with_text": has_text,
            "elapsed": round(v_elapsed, 1),
        }
        total_plates += plates_done

    elapsed = time.time() - t0
    print(f"\n[batch_ocr] Done: {total_plates} crops OCR'd across {len(video_dirs)} videos in {elapsed:.1f}s")
    print(f"[batch_ocr] Annotated output: {out_root}")

    # Print summary table
    print(f"\n{'Video':<30} {'Crops':>7} {'With Text':>10} {'Time':>7}")
    print("-" * 56)
    for vname, s in summary.items():
        print(f"{vname:<30} {s['total']:>7} {s['with_text']:>10} {s['elapsed']:>6.1f}s")
    print("-" * 56)
    print(f"{'TOTAL':<30} {total_plates:>7}")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--crops-dir",
        default=str(Path(__file__).resolve().parent / "output" / "plate_crops"),
        help="Root directory with per-video plate crop subfolders",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for annotated images (default: <crops-dir>_annotated)",
    )
    p.add_argument(
        "--confidence",
        type=float,
        default=0.0,
        help="Minimum OCR confidence to report (0.0 = show all)",
    )
    p.add_argument(
        "--font-size",
        type=int,
        default=18,
        help="Font size for Arabic text annotation on crops",
    )
    args = p.parse_args()
    run_batch_ocr(args.crops_dir, args.out_dir, args.confidence, args.font_size)


if __name__ == "__main__":
    main()
