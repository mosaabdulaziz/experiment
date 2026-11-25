#!/usr/bin/env python3
"""Evaluate a plate OCR model on plate crops (quality indicators: the crops have no labels).

Default: the model multicam runs, read exactly as the sidecar reads it
(``perception/modules/mtmc/plate_reader.py`` PlateReader, loaded by file path:
same 48x320 input, normalisation, mid-grey padding, 3x enhancement, CTC decode),
``--model`` + ``--dict`` for another export. Every crop is read once; the script
writes annotated crops, ``ocr_results.csv`` and ``ocr_summary.json`` to
``--out-dir``, and reports, overall and per camera, the share of crops with any
text and the mean confidence. Without labels these are NOT accuracy: a confident
wrong reading counts as text.

The crops: ``--crops-dir`` with one subfolder per camera of ``*.jpg`` crops,
default ``mtmc/output/Good-plates-OCR/<camera>/*.jpg``. That set does not exist in
the repo yet (DATA.md lists it as to be versioned): until it is ``dvc add``-ed as
``mtmc/output/Good-plates-OCR.dvc`` its evaluations are tagged ``unversioned``.

MLflow (mtmc/tracking/eval.py): the evaluation is logged into the run of the
model it evaluated (its ``plate_rec.onnx.mlflow.json`` sidecar, written when the
model is imported or exported with tracking, or ``--run-id``) as
``eval/<set>/*`` (``--eval-set``, default ``plate-ocr-golden``), with
ocr_summary.json (aggregate numbers only) under ``eval/<set>/``. Nothing that
holds a plate's text leaves this machine: ocr_results.csv (one row per crop,
with the text read) and the annotated crops stay local. ``--no-track`` records
nothing.

``--legacy-models`` runs the older side-by-side comparison of the two PaddleOCR
models in mtmc/models/ (paddleocr-onnx, arabic_PP-OCRv3_mobile_rec), untracked:
neither is deployed nor registered.

Usage:
    .venv/bin/python -m mtmc.eval_ocr_quality
    .venv/bin/python -m mtmc.eval_ocr_quality --crops-dir <dir> --eval-set plate-ocr-site2
    .venv/bin/python -m mtmc.eval_ocr_quality --model <export>.onnx --dict <dict>.txt --run-id <run>
    .venv/bin/python -m mtmc.eval_ocr_quality --legacy-models --upscale 4
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont

from .tracking import sha256_file
from .tracking.eval import add_eval_args, prepare_evaluation

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parents[1]
# What products/multicam/sidecar.yaml `modules.mtmc.plates` loads (ocr_model, ocr_dict, ocr_enhance: true).
DEPLOYED_DIR = _REPO / "products/el-mohandes/models/v5_arabic_mobile-fine-tuned"
DEPLOYED_MODEL = DEPLOYED_DIR / "inference" / "plate_rec.onnx"     # DVC: inference/plate_rec.onnx.dvc
DEPLOYED_DICT = DEPLOYED_DIR / "plate_dict.txt"
PLATE_READER = _REPO / "perception/modules/mtmc/plate_reader.py"   # the sidecar's reader (stdlib + cv2 + numpy)
GOLDEN_CROPS = Path(__file__).resolve().parent / "output" / "Good-plates-OCR"

_MODELS_DIR = Path(__file__).resolve().parent / "models"

# Model 1: PaddleOCR PP-OCRv3 (smaller, 128-char dict)
_MODEL1_DIR = _MODELS_DIR / "paddleocr-onnx"
_MODEL1_PATH = _MODEL1_DIR / "rec.onnx"
_MODEL1_DICT_PATH = _MODEL1_DIR / "dict.txt"

# Model 2: PaddleOCR PP-OCRv5 (larger, 700+ char dict)
_MODEL2_DIR = _MODELS_DIR / "arabic_PP-OCRv3_mobile_rec "
_MODEL2_PATH = _MODEL2_DIR / "model.onnx"

# Arabic font for annotation
_ARABIC_FONT_PATHS = [
    Path("/usr/share/fonts/truetype/noto/NotoNaskhArabic-Bold.ttf"),
    Path("/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
]


# ---------------------------------------------------------------------------
# Font discovery
# ---------------------------------------------------------------------------
def _find_font() -> str | None:
    for p in _ARABIC_FONT_PATHS:
        if p.exists():
            return str(p)
    return None


# ---------------------------------------------------------------------------
# Character dictionaries
# ---------------------------------------------------------------------------
def _load_dict_v3() -> list[str]:
    """PP-OCRv3 dict (128 chars, one per line in dict.txt)."""
    return ["blank"] + _MODEL1_DICT_PATH.read_text(encoding="utf-8").splitlines()


def _build_fallback_dict() -> list[str]:
    """Build a comprehensive character dict from Unicode ranges (avoids literal issues)."""
    basic = list("!#%&'()*+,-./0123456789:;<=>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]_`abcdefghijklmnopqrstuvwxyz")
    latin_ext = [chr(c) for c in range(0x00C0, 0x00FF + 1)]  # A-grave through y-diaeresis
    greek = [chr(c) for c in range(0x0391, 0x03C9 + 1)]  # Alpha through omega
    arabic_main = [chr(c) for c in range(0x060C, 0x06FF + 1)]  # Arabic block
    arabic_ext1 = [chr(c) for c in range(0x0750, 0x077F + 1)]  # Arabic Supplement
    arabic_ext2 = [chr(c) for c in range(0x08A0, 0x08FF + 1)]  # Arabic Extended-A
    arabic_digits = [chr(c) for c in range(0x0660, 0x0669 + 1)]  # Arabic-Indic digits
    persian_digits = [chr(c) for c in range(0x06F0, 0x06F9 + 1)]  # Extended Arabic-Indic digits
    return basic + latin_ext + greek + arabic_main + arabic_ext1 + arabic_ext2 + arabic_digits + persian_digits


def _load_dict_v5() -> list[str]:
    """PP-OCRv5 dict extracted from inference.yml postprocess section.

    The dict is embedded in the inference.yml/JSON. We parse it at runtime
    from the model directory. If unavailable, fall back to a comprehensive set.
    """
    yml_path = _MODEL2_DIR / "inference.yml"
    json_path = _MODEL2_DIR / "inference.json"

    chars: list[str] = []

    if json_path.exists():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            chars = _extract_dict_from_json(data)
        except Exception:
            pass

    if not chars and yml_path.exists():
        try:
            import yaml
            data = yaml.safe_load(yml_path.read_text(encoding="utf-8"))
            chars = data.get("PostProcess", {}).get("character_dict", [])
        except Exception:
            pass

    if not chars:
        chars = _build_fallback_dict()

    return ["blank"] + chars


def _extract_dict_from_json(data) -> list[str]:
    """Recursively find character_dict in the PaddlePaddle inference JSON."""
    if isinstance(data, dict):
        for key, val in data.items():
            if key == "character_dict" and isinstance(val, list):
                return val
            result = _extract_dict_from_json(val)
            if result:
                return result
    elif isinstance(data, list):
        for item in data:
            result = _extract_dict_from_json(item)
            if result:
                return result
    return []


# ---------------------------------------------------------------------------
# CTC decoding
# ---------------------------------------------------------------------------
def _ctc_decode(logits: np.ndarray, char_dict: list[str]) -> str:
    """CTC greedy decode. logits: (seq_len, vocab_size)."""
    indices = logits.argmax(axis=1)
    chars: list[str] = []
    prev = -1
    for idx in indices:
        if idx != prev:
            if idx != 0 and idx < len(char_dict):
                chars.append(char_dict[idx])
            prev = idx
    return "".join(chars)


def _ctc_confidence(logits: np.ndarray) -> float:
    """Mean confidence of decoded (non-blank) characters."""
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    max_probs = probs.max(axis=1)
    indices = logits.argmax(axis=1)
    non_blank = [(max_probs[i], indices[i]) for i in range(len(indices)) if indices[i] != 0]
    if not non_blank:
        return 0.0
    return float(np.mean([p for p, _ in non_blank]))


# ---------------------------------------------------------------------------
# OCR recognition (works for both models - same ONNX input interface)
# ---------------------------------------------------------------------------
OCR_INPUT_H = 48
OCR_INPUT_W = 320


def ocr_recognize(
    sess: ort.InferenceSession,
    img_bgr: np.ndarray,
    char_dict: list[str],
    upscale: int = 2,
    model_version: str = "v3",
) -> tuple[str, float]:
    """Run PP-OCR recognition on a single image crop.

    Args:
        upscale: Upscale factor for small crops (<100px height).
        model_version: 'v3' uses (x-0.5)/0.5 normalization;
                       'v5' uses plain /255 normalization.
    Returns:
        (text, confidence)
    """
    h, w = img_bgr.shape[:2]

    # Upscale small crops
    if h < 100 and upscale > 1:
        img_bgr = cv2.resize(img_bgr, (w * upscale, h * upscale), interpolation=cv2.INTER_CUBIC)
        h, w = img_bgr.shape[:2]

    input_h = OCR_INPUT_H

    scale = input_h / h
    new_w = min(int(w * scale), OCR_INPUT_W)
    resized = cv2.resize(img_bgr, (new_w, input_h), interpolation=cv2.INTER_LINEAR)

    # Pad to target width
    if new_w < OCR_INPUT_W:
        pad = np.full((input_h, OCR_INPUT_W - new_w, 3), 0, dtype=np.uint8)
        resized = np.hstack([resized, pad])
    elif new_w > OCR_INPUT_W:
        resized = resized[:, :OCR_INPUT_W]

    # PP-OCR preprocessing: BGR->RGB, /255
    blob = resized[:, :, ::-1].astype(np.float32) / 255.0
    if model_version == "v3":
        # PP-OCRv3 uses (x-0.5)/0.5 normalization
        blob = (blob - 0.5) / 0.5
    # PP-OCRv5 uses plain /255 (already done above)
    blob = blob.transpose(2, 0, 1)[np.newaxis]  # NCHW

    # Get the input name from the session
    input_name = sess.get_inputs()[0].name
    logits = sess.run(None, {input_name: blob})[0][0]  # (seq_len, vocab)
    text = _ctc_decode(logits, char_dict)
    conf = _ctc_confidence(logits)
    return text, conf


# ---------------------------------------------------------------------------
# Annotation (PIL for proper Arabic rendering)
# ---------------------------------------------------------------------------
def annotate_image(
    img_bgr: np.ndarray,
    model1_text: str,
    model1_conf: float,
    model2_text: str,
    model2_conf: float,
    font_path: str | None,
    font_size: int = 16,
) -> np.ndarray:
    """Draw OCR results from both models on the image.

    Top bar:  Model 1 (PP-OCRv3) text + confidence
    Bottom:   Model 2 (PP-OCRv5) text + confidence

    Returns annotated BGR image.
    """
    h, w = img_bgr.shape[:2]

    # Convert BGR -> RGB PIL image
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    try:
        font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    # Try Arabic reshaping
    def _reshape(text: str) -> str:
        try:
            import arabic_reshaper
            from bidi.algorithm import get_display
            return get_display(arabic_reshaper.reshape(text))
        except ImportError:
            return text

    # Compute label heights
    sample_bbox = draw.textbbox((0, 0), "Ay", font=font)
    line_h = sample_bbox[3] - sample_bbox[1]
    bar_h = line_h + 8

    # Extend canvas to fit two label bars
    new_h = h + bar_h * 2
    extended = Image.new("RGB", (w, new_h), (30, 30, 30))
    extended.paste(pil_img, (0, bar_h))
    draw = ImageDraw.Draw(extended)

    # Top bar - Model 1 (PP-OCRv3) - green theme
    m1_text = _reshape(model1_text) if model1_text else "(no text)"
    m1_label = f"v3: {m1_text}"
    m1_conf_label = f"{model1_conf:.0%}"
    draw.rectangle([0, 0, w, bar_h], fill=(0, 60, 0))
    draw.text((4, 4), m1_label, fill=(0, 255, 0), font=font)
    # Confidence badge
    cbbox = draw.textbbox((0, 0), m1_conf_label, font=font)
    cw = cbbox[2] - cbbox[0]
    draw.rectangle([w - cw - 10, 0, w, bar_h], fill=(0, 120, 0))
    draw.text((w - cw - 6, 4), m1_conf_label, fill=(255, 255, 255), font=font)

    # Bottom bar - Model 2 (PP-OCRv5) - blue theme
    m2_text = _reshape(model2_text) if model2_text else "(no text)"
    m2_label = f"v5: {m2_text}"
    m2_conf_label = f"{model2_conf:.0%}"
    draw.rectangle([0, h + bar_h, w, h + bar_h * 2], fill=(0, 40, 80))
    draw.text((4, h + bar_h + 4), m2_label, fill=(100, 180, 255), font=font)
    cbbox2 = draw.textbbox((0, 0), m2_conf_label, font=font)
    cw2 = cbbox2[2] - cbbox2[0]
    draw.rectangle([w - cw2 - 10, h + bar_h, w, h + bar_h * 2], fill=(0, 80, 160))
    draw.text((w - cw2 - 6, h + bar_h + 4), m2_conf_label, fill=(255, 255, 255), font=font)

    return cv2.cvtColor(np.array(extended), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# One model, read as the sidecar reads it
# ---------------------------------------------------------------------------
def load_plate_reader(model: Path, dict_path: Path, enhance: bool = True):
    """The sidecar's PlateReader, loaded from its file: importing the perception.modules.mtmc package would pull in
    the engine and the MTMC core, while plate_reader.py itself needs only cv2, numpy and onnxruntime."""
    spec = importlib.util.spec_from_file_location("iris_plate_reader", PLATE_READER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PlateReader(str(model), str(dict_path), enhance)


def discover_crops(crops_root: Path) -> list[tuple[str, Path]]:
    """(camera, crop) for every <camera>/*.jpg under crops_root, sorted."""
    return [(d.name, c) for d in sorted(x for x in crops_root.iterdir() if x.is_dir()) for c in sorted(d.glob("*.jpg"))]


def annotate_one(img_bgr: np.ndarray, text: str, conf: float, font_path: str | None, font_size: int = 16) -> np.ndarray:
    """The crop with one bar under it: the text read and its confidence."""
    h, w = img_bgr.shape[:2]
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    try:
        font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        shown = get_display(arabic_reshaper.reshape(text)) if text else "(no text)"
    except ImportError:
        shown = text or "(no text)"
    probe = ImageDraw.Draw(pil).textbbox((0, 0), "Ay", font=font)
    bar = probe[3] - probe[1] + 8
    out = Image.new("RGB", (max(w, 1), h + bar), (30, 30, 30))
    out.paste(pil, (0, 0))
    draw = ImageDraw.Draw(out)
    draw.text((4, h + 4), f"{shown}  {conf:.0%}", fill=(100, 255, 100), font=font)
    return cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)


def _segment(name: str) -> str:
    return re.sub(r"[^\w\-.]", "_", name)


def quality_metrics(rows: list[dict]) -> dict[str, float]:
    """Overall and per-camera crop counts, share of crops with text, mean confidence of those with text."""
    def block(rs: list[dict]) -> dict[str, float]:
        texts = [r for r in rs if r["text"]]
        return {"crops": len(rs), "with_text": len(texts),
                "with_text_rate": len(texts) / len(rs) if rs else 0.0,
                "mean_conf": float(np.mean([r["conf"] for r in texts])) if texts else 0.0}
    m = block(rows)
    for cam in sorted({r["camera"] for r in rows}):
        for k, v in block([r for r in rows if r["camera"] == cam]).items():
            m[f"camera/{_segment(cam)}/{k}"] = v
    return {k: float(v) for k, v in m.items()}


def evaluate_model(reader, crops: list[tuple[str, Path]], out_root: Path, font_size: int = 16,
                   annotate: bool = True) -> tuple[list[dict], dict[str, float], Path, Path]:
    """Read every crop with `reader` (.read(bgr) -> (text, conf)); write annotated crops, ocr_results.csv and
    ocr_summary.json into out_root. Returns (rows, metrics, csv path, summary path)."""
    out_root.mkdir(parents=True, exist_ok=True)
    font_path = _find_font()
    rows: list[dict] = []
    t0 = time.time()
    for cam, crop in crops:
        img = cv2.imread(str(crop))
        if img is None:
            print(f"  ! unreadable: {crop}")
            continue
        text, conf = reader.read(img)
        if annotate:
            (out_root / cam).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_root / cam / crop.name), annotate_one(img, text, conf, font_path, font_size),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
        rows.append({"camera": cam, "crop_file": crop.name, "text": text, "conf": round(float(conf), 4),
                     "img_w": int(img.shape[1]), "img_h": int(img.shape[0])})
    metrics = quality_metrics(rows)
    csv_path, summary_path = out_root / "ocr_results.csv", out_root / "ocr_summary.json"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["camera", "crop_file", "text", "conf", "img_w", "img_h"])
        writer.writeheader()
        writer.writerows(rows)
    # aggregates only, never a plate string: this file is the one uploaded to MLflow
    summary_path.write_text(json.dumps({"metrics": metrics, "seconds": round(time.time() - t0, 1),
                                        "note": "no labels: text rate and confidence, not accuracy"}, indent=2))
    return rows, metrics, csv_path, summary_path


# ---------------------------------------------------------------------------
# Legacy: the two mtmc/models/ PaddleOCR models side by side (untracked)
# ---------------------------------------------------------------------------
def run_evaluation(
    crops_dir: str,
    out_dir: str | None = None,
    upscale: int = 2,
    font_size: int = 16,
) -> None:
    """Run both OCR models on all plate crops and produce annotated outputs."""
    crops_root = Path(crops_dir)
    if not crops_root.exists():
        raise FileNotFoundError(f"Crops directory not found: {crops_dir}")

    out_root = Path(out_dir) if out_dir else crops_root.parent / "ocr_evaluation"
    out_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load Model 1: PP-OCRv3
    # ------------------------------------------------------------------
    print(f"[eval] Loading Model 1 (PP-OCRv3): {_MODEL1_PATH.name}")
    char_dict_v3 = _load_dict_v3()
    print(f"[eval]   Vocab size: {len(char_dict_v3)} chars")
    sess_v3 = ort.InferenceSession(
        str(_MODEL1_PATH),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # ------------------------------------------------------------------
    # Load Model 2: PP-OCRv5
    # ------------------------------------------------------------------
    print(f"[eval] Loading Model 2 (PP-OCRv5): {_MODEL2_PATH.name}")
    char_dict_v5 = _load_dict_v5()
    print(f"[eval]   Vocab size: {len(char_dict_v5)} chars")
    sess_v5 = ort.InferenceSession(
        str(_MODEL2_PATH),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    font_path = _find_font()
    if font_path:
        print(f"[eval] Font: {font_path}")

    # ------------------------------------------------------------------
    # Discover crops
    # ------------------------------------------------------------------
    video_dirs = sorted([d for d in crops_root.iterdir() if d.is_dir()])
    print(f"[eval] Found {len(video_dirs)} camera directories")

    all_crops: list[Path] = []
    for vdir in video_dirs:
        crops = sorted(vdir.glob("*.jpg"))
        all_crops.extend(crops)
    print(f"[eval] Total plate crops: {len(all_crops)}")

    # ------------------------------------------------------------------
    # Process each crop
    # ------------------------------------------------------------------
    csv_rows: list[dict] = []
    t0 = time.time()

    for vdir in video_dirs:
        cam_name = vdir.name
        crops = sorted(vdir.glob("*.jpg"))
        cam_out = out_root / cam_name
        cam_out.mkdir(parents=True, exist_ok=True)

        for crop_path in crops:
            img = cv2.imread(str(crop_path))
            if img is None:
                continue

            # Run Model 1 (PP-OCRv3)
            text_v3, conf_v3 = ocr_recognize(sess_v3, img, char_dict_v3, upscale=upscale, model_version="v3")

            # Run Model 2 (PP-OCRv5)
            text_v5, conf_v5 = ocr_recognize(sess_v5, img, char_dict_v5, upscale=upscale, model_version="v5")

            # Annotate
            annotated = annotate_image(
                img, text_v3, conf_v3, text_v5, conf_v5,
                font_path, font_size,
            )
            out_path = cam_out / crop_path.name
            cv2.imwrite(str(out_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])

            csv_rows.append({
                "camera": cam_name,
                "crop_file": crop_path.name,
                "model_v3_text": text_v3,
                "model_v3_conf": round(conf_v3, 4),
                "model_v5_text": text_v5,
                "model_v5_conf": round(conf_v5, 4),
                "img_w": img.shape[1],
                "img_h": img.shape[0],
            })

    elapsed = time.time() - t0

    # ------------------------------------------------------------------
    # Write CSV summary
    # ------------------------------------------------------------------
    csv_path = out_root / "ocr_results.csv"
    if csv_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print(f"\n{'='*80}")
    print(f"OCR EVALUATION RESULTS")
    print(f"{'='*80}")

    # Per-camera stats
    cameras = sorted(set(r["camera"] for r in csv_rows))
    for cam in cameras:
        cam_rows = [r for r in csv_rows if r["camera"] == cam]
        v3_with_text = sum(1 for r in cam_rows if r["model_v3_text"])
        v5_with_text = sum(1 for r in cam_rows if r["model_v5_text"])
        v3_confs = [r["model_v3_conf"] for r in cam_rows if r["model_v3_text"]]
        v5_confs = [r["model_v5_conf"] for r in cam_rows if r["model_v5_text"]]
        v3_avg = float(np.mean(v3_confs)) if v3_confs else 0.0
        v5_avg = float(np.mean(v5_confs)) if v5_confs else 0.0
        print(f"\n  Camera {cam} ({len(cam_rows)} crops):")
        print(f"    PP-OCRv3: {v3_with_text}/{len(cam_rows)} with text, avg conf: {v3_avg:.1%}")
        print(f"    PP-OCRv5: {v5_with_text}/{len(cam_rows)} with text, avg conf: {v5_avg:.1%}")

    # Overall
    total = len(csv_rows)
    v3_total = sum(1 for r in csv_rows if r["model_v3_text"])
    v5_total = sum(1 for r in csv_rows if r["model_v5_text"])
    print(f"\n  Overall: {total} crops")
    print(f"    PP-OCRv3: {v3_total}/{total} with text ({v3_total/total*100:.0f}%)")
    print(f"    PP-OCRv5: {v5_total}/{total} with text ({v5_total/total*100:.0f}%)")

    # Show individual results
    print(f"\n  {'File':<42} {'v3 text':<20} {'v3 conf':>8} {'v5 text':<20} {'v5 conf':>8}")
    print(f"  {'-'*42} {'-'*20} {'-'*8} {'-'*20} {'-'*8}")
    for r in csv_rows:
        t3 = r["model_v3_text"][:18] if r["model_v3_text"] else "(empty)"
        t5 = r["model_v5_text"][:18] if r["model_v5_text"] else "(empty)"
        print(f"  {r['camera']+'/'+r['crop_file']:<42} {t3:<20} {r['model_v3_conf']:>7.0%} {t5:<20} {r['model_v5_conf']:>7.0%}")

    print(f"\n{'='*80}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Annotated output: {out_root}")
    print(f"  Results CSV: {csv_path}")
    print(f"{'='*80}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", default=str(DEPLOYED_MODEL),
                   help=f"OCR ONNX to evaluate (default: the deployed {DEPLOYED_MODEL.relative_to(_REPO)})")
    p.add_argument("--dict", default=str(DEPLOYED_DICT),
                   help=f"its character list, one per line (default: {DEPLOYED_DICT.relative_to(_REPO)})")
    p.add_argument("--no-enhance", action="store_true",
                   help="skip the reader's 3x enlarge + sharpen (the sidecar runs with ocr_enhance: true)")
    p.add_argument("--no-annotate", action="store_true", help="write no annotated crops")
    p.add_argument("--legacy-models", action="store_true",
                   help="the old side-by-side run of the two mtmc/models/ PaddleOCR models (untracked)")
    p.add_argument(
        "--crops-dir",
        default=str(GOLDEN_CROPS),
        help="Directory with per-camera plate crop subfolders (default: mtmc/output/Good-plates-OCR)",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for annotated images + results (default: <crops-dir>/../ocr_evaluation)",
    )
    p.add_argument(
        "--upscale",
        type=int,
        default=2,
        help="--legacy-models only: upscale factor for small crops < 100px (default: 2)",
    )
    p.add_argument(
        "--font-size",
        type=int,
        default=16,
        help="Font size for Arabic text annotation (default: 16)",
    )
    add_eval_args(p, "plate-ocr-golden")
    args = p.parse_args(argv)
    crops_root = Path(args.crops_dir)
    if not crops_root.is_dir():
        raise SystemExit(f"crops directory not found: {crops_root}\nExpected <camera>/*.jpg plate crops, by default in "
                         f"{GOLDEN_CROPS.relative_to(_REPO)} (not in the repo yet: it needs to be supplied and dvc add-ed "
                         "as mtmc/output/Good-plates-OCR.dvc), or pass --crops-dir.")
    if args.legacy_models:
        print("[eval] --legacy-models: not recorded in MLflow (these models are neither deployed nor registered)")
        run_evaluation(args.crops_dir, args.out_dir, args.upscale, args.font_size)
        return
    model, dict_path = Path(args.model), Path(args.dict)
    for f, what in ((model, "model"), (dict_path, "dict")):
        if not f.is_file():
            raise SystemExit(f"OCR {what} not found: {f}" + ("\n`dvc pull` it (its .dvc pointer is beside it)"
                                                             if Path(str(f) + ".dvc").is_file() else ""))
    crops = discover_crops(crops_root)
    if not crops:
        raise SystemExit(f"no <camera>/*.jpg crops under {crops_root}")
    target = prepare_evaluation(args, model, crops_root, "mtmc.eval_ocr_quality",
                                tags={"dict_sha256": sha256_file(dict_path), "enhance": not args.no_enhance,
                                      "metric_kind": "unlabelled-quality", "crops": len(crops),
                                      "reader": f"{PLATE_READER.relative_to(_REPO)}@{sha256_file(PLATE_READER)[:12]}"})
    out_root = Path(args.out_dir) if args.out_dir else crops_root.parent / "ocr_evaluation"
    reader = load_plate_reader(model, dict_path, enhance=not args.no_enhance)
    print(f"[eval] {model} on {len(crops)} crops from {crops_root}")
    rows, metrics, csv_path, summary_path = evaluate_model(reader, crops, out_root, args.font_size,
                                                           annotate=not args.no_annotate)
    print(f"\n  {int(metrics['crops'])} crops: {metrics['with_text_rate']:.0%} with text, "
          f"mean confidence {metrics['mean_conf']:.1%} (no labels: not accuracy)")
    for cam in sorted({r["camera"] for r in rows}):
        k = f"camera/{_segment(cam)}"
        print(f"    {cam}: {int(metrics[k + '/crops'])} crops, {metrics[k + '/with_text_rate']:.0%} with text, "
              f"mean conf {metrics[k + '/mean_conf']:.1%}")
    print(f"  results: {csv_path}\n  summary: {summary_path}")
    # summary only: ocr_results.csv carries the plate text read from each crop and stays local
    target.log(metrics, artifacts=[summary_path])


if __name__ == "__main__":
    main()
