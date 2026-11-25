#!/usr/bin/env python3
"""Offline lab for plate-crop enhancements: the same OCR model as the pipeline, many ways to prepare the crop.

Every plate crop (a folder of images, e.g. the data/plates/<gid>_<cam>.jpg the sidecar saves) goes through each
enhancement method, then through the production reader (perception/modules/mtmc/plate_reader.py, its own enhancement switched off
so a method is the only thing that changes). Out come:

  <out>/sheets/<crop>.png   one picture per crop: a row per method with the processed crop and what was read
  <out>/results.csv         crop, method, text, logical text, confidence, [truth, correct]
  <out>/summary.md          per method: mean confidence, how often it agrees with the production method,
                            and (with --truth) how many readings were exactly right

  python3 scripts/plate_enhance_lab.py data/plates -o /tmp/plate_lab
  python3 scripts/plate_enhance_lab.py crops/ -o out --methods raw,production,clahe --truth truth.csv
  python3 scripts/plate_enhance_lab.py crops/ -o out --plugin my_methods.py   # adds/overrides METHODS

truth.csv has two columns, no header needed: file name (with or without extension) and the plate as the reader
writes it (digits first, then the letters left to right as they stand on the plate, e.g. 5356روس).
A plugin is a python file with a dict METHODS = {"name": fn(bgr_uint8) -> bgr_uint8}.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import onnxruntime  # noqa: E402
onnxruntime.set_default_logger_severity(3)      # the model export leaves a warning per unused weight
from perception.modules.mtmc.plate_reader import PlateReader, enhance as production_enhance  # noqa: E402

MODEL_DIR = REPO / "products/el-mohandes/models/v5_arabic_mobile-fine-tuned"
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FONTS = ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
         "/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf")


# ---- the enhancement methods: BGR uint8 in, BGR uint8 out (any size; the reader scales to 48 px high) -----------

def _luma_apply(im, fn):
    """Apply fn to the lightness channel only, so colours do not shift."""
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = fn(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _unsharp(im, sigma=1.2, amount=0.8):
    blur = cv2.GaussianBlur(im, (0, 0), sigma)
    return cv2.addWeighted(im, 1 + amount, blur, -amount, 0)


def _up(im, k, interp=cv2.INTER_CUBIC):
    return cv2.resize(im, None, fx=k, fy=k, interpolation=interp)


def _auto_gamma(im):
    """Brighten dark crops, tame bright ones, towards a mid-grey mean."""
    mean = max(1.0, float(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).mean()))
    gamma = float(np.clip(np.log(0.5) / np.log(mean / 255.0), 0.5, 2.0))
    lut = (np.linspace(0, 1, 256) ** gamma * 255).astype(np.uint8)
    return cv2.LUT(im, lut)


def _wiener_deblur(im, sigma=1.5, k=0.02):
    """Wiener deconvolution against a gaussian blur of the given sigma (guess at the camera's softness)."""
    out = np.empty_like(im)
    h, w = im.shape[:2]
    psf = np.zeros((h, w), np.float32)
    kh = min(15, h) | 1
    kw = min(15, w) | 1
    g = cv2.getGaussianKernel(kh, sigma) @ cv2.getGaussianKernel(kw, sigma).T
    psf[:kh, :kw] = g
    psf = np.roll(psf, (-(kh // 2), -(kw // 2)), (0, 1))
    H = np.fft.fft2(psf)
    for c in range(3):
        F = np.fft.fft2(im[:, :, c].astype(np.float32))
        R = np.conj(H) / (np.abs(H) ** 2 + k) * F
        out[:, :, c] = np.clip(np.real(np.fft.ifft2(R)), 0, 255).astype(np.uint8)
    return out


METHODS = {
    # what the reader gets today in the pipeline and what it would get with no help at all
    "production": production_enhance,
    "raw": lambda im: im,
    # enlargement only
    "up3_cubic": lambda im: _up(im, 3),
    "up4_lanczos": lambda im: _up(im, 4, cv2.INTER_LANCZOS4),
    # contrast
    "clahe": lambda im: _luma_apply(_up(im, 3), lambda l: cv2.createCLAHE(2.0, (4, 4)).apply(l)),
    "equalize": lambda im: _luma_apply(_up(im, 3), cv2.equalizeHist),
    "auto_gamma": lambda im: _auto_gamma(_up(im, 3)),
    # sharpness
    "unsharp": lambda im: _unsharp(_up(im, 3)),
    "unsharp_strong": lambda im: _unsharp(_up(im, 3), 1.6, 1.5),
    "wiener": lambda im: _wiener_deblur(_up(im, 3)),
    # noise
    "denoise_nlm": lambda im: cv2.fastNlMeansDenoisingColored(_up(im, 3), None, 5, 5, 7, 21),
    "bilateral": lambda im: cv2.bilateralFilter(_up(im, 3), 7, 40, 7),
    # combinations
    "denoise+clahe+unsharp": lambda im: _unsharp(
        _luma_apply(cv2.bilateralFilter(_up(im, 3), 7, 40, 7), lambda l: cv2.createCLAHE(2.0, (4, 4)).apply(l))),
    "gray_otsu": lambda im: cv2.cvtColor(cv2.threshold(
        cv2.cvtColor(_up(im, 3), cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        cv2.COLOR_GRAY2BGR),
}


# ---- helpers ----------------------------------------------------------------------------------------------------

def logical(text: str) -> str:
    """The plate as it is spoken (letters right to left, then the digits), like app/plate_reads.logical."""
    digits = "".join(c for c in text if c.isdigit())
    letters = [c for c in text if "ء" <= c <= "ي"]
    return (" ".join(reversed(letters)) + " " + digits).strip()


def load_truth(path: str | None) -> dict[str, str]:
    truth: dict[str, str] = {}
    if not path:
        return truth
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.reader(fh):
            if len(row) >= 2 and row[0].strip() and row[0].strip().lower() not in ("file", "crop"):
                truth[Path(row[0].strip()).stem] = row[1].strip().replace(" ", "")
    return truth


def parse_since(text: str) -> float:
    """Epoch seconds of "HH:MM" (today, local time) or an ISO date/time."""
    import datetime as dt
    try:
        t = dt.datetime.strptime(text, "%H:%M").time()
        return dt.datetime.combine(dt.date.today(), t).timestamp()
    except ValueError:
        return dt.datetime.fromisoformat(text).timestamp()


def load_plugin(path: str) -> dict:
    spec = importlib.util.spec_from_file_location("plate_lab_plugin", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return dict(getattr(mod, "METHODS"))


def sheet(name: str, crop: np.ndarray, rows: list[dict], truth: str | None) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont
    path = next((p for p in FONTS if os.path.exists(p)), None)
    big, small = (ImageFont.truetype(path, 28), ImageFont.truetype(path, 15)) if path else \
                 (ImageFont.load_default(), ImageFont.load_default())
    row_h, thumb_w, width = 76, 300, 900
    img = Image.new("RGB", (width, 80 + row_h * len(rows)), (24, 24, 24))
    d = ImageDraw.Draw(img)
    d.text((8, 6), name + (f"    truth: {truth}" if truth else ""), font=small, fill=(220, 220, 120))
    s = min(60 / crop.shape[0], thumb_w / crop.shape[1])
    img.paste(Image.fromarray(cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))))[:, :, ::-1]), (8, 26))
    d.text((thumb_w + 24, 40), "original crop", font=small, fill=(150, 150, 150))
    for i, r in enumerate(rows):
        y = 80 + i * row_h
        im = r["image"]
        s = min((row_h - 10) / im.shape[0], thumb_w / im.shape[1])
        img.paste(Image.fromarray(cv2.resize(im, (max(1, int(im.shape[1] * s)), max(1, int(im.shape[0] * s))))[:, :, ::-1]), (8, y + 5))
        ok = r.get("correct")
        colour = (120, 230, 120) if ok else (230, 120, 120) if ok is False else (240, 240, 240)
        d.text((thumb_w + 24, y + 6), r["text"] or "-", font=big, fill=colour)
        d.text((thumb_w + 24, y + 46), f"{r['method']}   conf {r['conf']:.2f}", font=small, fill=(150, 150, 150))
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("crops", nargs="?", help="a folder of plate crops (or one image)")
    ap.add_argument("-o", "--out", default="plate_lab_out")
    ap.add_argument("--methods", help="comma list (default: all); see --list")
    ap.add_argument("--list", action="store_true", help="print the method names and quit")
    ap.add_argument("--latest-run", action="store_true",
                    help="only the crops of the newest sidecar start (global ids restart every run, so a folder holds many\n                    vehicles under the same id); needs crops saved by this version")
    ap.add_argument("--gid", type=int, help="only the crops of this global id (the number in the truck's ااا-000N); combine with --latest-run or --since, ids repeat across runs")
    ap.add_argument("--since", help="only crops saved after this local time: HH:MM today, or YYYY-MM-DD[THH:MM]")
    ap.add_argument("--truth", help="csv: file, plate text (enables accuracy)")
    ap.add_argument("--plugin", help="python file with METHODS = {name: fn(bgr) -> bgr}")
    ap.add_argument("--model", default=str(MODEL_DIR / "inference/plate_rec.onnx"))
    ap.add_argument("--dict", default=str(MODEL_DIR / "plate_dict.txt"))
    args = ap.parse_args()

    methods = dict(METHODS)
    if args.plugin:
        methods.update(load_plugin(args.plugin))
    if args.list:
        print("\n".join(methods))
        return 0
    if args.methods:
        want = [m.strip() for m in args.methods.split(",") if m.strip()]
        missing = [m for m in want if m not in methods]
        if missing:
            ap.error(f"unknown method(s) {missing}; --list shows them")
        methods = {m: methods[m] for m in want}

    if not args.crops:
        ap.error("give a folder of plate crops (or one image)")
    src = Path(args.crops)
    if src.is_dir() and (src / "crops").is_dir():          # the sidecar's data/plates folder: its saved crops
        src = src / "crops"
    pipeline = {}                                          # what the pipeline itself read of each saved crop
    keep_stems = None
    if (src / "index.jsonl").exists() if src.is_dir() else False:
        import json
        index = [json.loads(line) for line in (src / "index.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        for row in index:
            pipeline[Path(row["file"]).stem] = row["text"]
        wanted = index
        if args.latest_run:
            newest = max((r.get("run", 0) for r in index), default=0)
            if not newest:
                ap.error("--latest-run: these crops carry no run id (saved by an older sidecar); use --since")
            wanted = [r for r in wanted if r.get("run") == newest]
        if args.since:
            wanted = [r for r in wanted if r["at"] >= parse_since(args.since)]
        if args.gid is not None:
            wanted = [r for r in wanted if r["gid"] == args.gid]
        if args.latest_run or args.since or args.gid is not None:
            keep_stems = {Path(r["file"]).stem for r in wanted}
    files = [src] if src.is_file() else sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXT)
    # the sidecar's own sheets (<gid>.png) show several crops; the single crops are <gid>_<cam>.jpg
    files = [p for p in files if not (p.suffix.lower() == ".png" and p.stem.isdigit())]
    if keep_stems is not None:
        files = [p for p in files if p.stem in keep_stems]
    if not files:
        ap.error(f"no plate images in {src}")

    reader = PlateReader(args.model, args.dict, enhance_crops=False)
    truth = load_truth(args.truth)
    out = Path(args.out)
    (out / "sheets").mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for f in files:
        crop = cv2.imread(str(f))
        if crop is None or crop.size == 0:
            print(f"skip {f.name}: unreadable")
            continue
        t = truth.get(f.stem)
        rows = []
        for name, fn in methods.items():
            try:
                im = fn(crop.copy())
                text, conf = reader.read(im)
            except Exception as exc:                        # one broken method must not stop the lab
                print(f"{f.name}: {name} failed: {exc}")
                im, text, conf = crop, "", 0.0
            row = {"crop": f.stem, "method": name, "text": text, "logical": logical(text), "conf": conf,
                   "truth": t, "correct": (text == t) if t else None, "image": im}
            rows.append(row)
            results.append(row)
        cv2.imwrite(str(out / "sheets" / f"{f.stem}.png"), sheet(f.stem, crop, rows, t))
        print(f"{f.name}: " + "  ".join(f"{r['method']}={r['text'] or '-'}" for r in rows))

    with open(out / "results.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["crop", "method", "text", "logical", "conf", "truth", "correct"])
        for r in results:
            w.writerow([r["crop"], r["method"], r["text"], r["logical"], f"{r['conf']:.3f}",
                        r["truth"] or "", "" if r["correct"] is None else int(r["correct"])])

    base = {r["crop"]: r["text"] for r in results if r["method"] == "production"}
    lines = ["| method | crops | mean conf | same as production | exact vs truth |", "|---|---|---|---|---|"]
    for name in methods:
        rs = [r for r in results if r["method"] == name]
        if not rs:
            continue
        same = sum(r["text"] == base[r["crop"]] for r in rs) if base else 0
        graded = [r for r in rs if r["correct"] is not None]
        exact = f"{sum(r['correct'] for r in graded)}/{len(graded)}" if graded else "-"
        lines.append(f"| {name} | {len(rs)} | {np.mean([r['conf'] for r in rs]):.3f} | "
                     f"{f'{same}/{len(rs)}' if base else '-'} | {exact} |")
    if pipeline:
        prod = [r for r in results if r["method"] == "production" and r["crop"] in pipeline]
        same = sum(r["text"] == pipeline[r["crop"]] for r in prod)
        lines.append(f"\nproduction method vs what the pipeline read of the same saved crop: {same}/{len(prod)} identical")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines) + f"\n\nsheets: {out / 'sheets'}   table: {out / 'results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
