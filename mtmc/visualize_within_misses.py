#!/usr/bin/env python3
"""Render matching exceptions as a labeled contact sheet.

For every comparison whose verdict is NOT the expected one, draw a row:

    [crop A]  ->  [crop B]
    <path A>           <path B>  (verdict/confidence/explanation)

Two sections:
  --section within  : within_vehicle pairs where verdict != 'same'
                      (a same-vehicle pairing judged 'different')
  --section cross   : cross_vehicle pairs where verdict != 'different'
                      (a different-vehicle pairing judged 'same' — the
                       dangerous false matches)

Output: mtmc/output/{section}_misses_sheet.png
"""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).parent / "output" / "vehicle_crops" / "chosen crops"
COMPARISONS = Path(__file__).parent / "vehicle_comparisons.json"
OUT = Path(__file__).parent / "output" / "within_misses_sheet.png"

THUMB_W, THUMB_H = 420, 360
PAD, LABEL_H, COL_GAP = 14, 84, 40
ROW_H = THUMB_H + LABEL_H + PAD * 2


def _thumb(rel: str) -> Image.Image:
    im = Image.open(BASE / rel).convert("RGB")
    im.thumbnail((THUMB_W, THUMB_H), Image.LANCZOS)
    canvas = Image.new("RGB", (THUMB_W, THUMB_H), (245, 245, 245))
    x = (THUMB_W - im.width) // 2
    y = (THUMB_H - im.height) // 2
    canvas.paste(im, (x, y))
    return canvas


def _font(size: int) -> ImageFont:
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _collect(data: dict, section: str):
    """Return rows: list of (left_rel, right_rel, verdict)."""
    rows = []
    if section == "within":
        for veh, pairs in data.get("within_vehicle", {}).items():
            images = sorted((BASE / veh).glob("*.jpg"))
            rep = None
            if images:
                rep = max(images, key=lambda p: p.stat().st_size).name
            for other_rel, verdict in pairs.items():
                if verdict.get("decision") != "same" or "error" in verdict:
                    other = other_rel.split("/")[-1]
                    left = rep or other
                    rows.append((f"{veh}/{left}", f"{veh}/{other}", verdict))
    elif section == "cross":
        for key, verdict in data.get("cross_vehicle", {}).items():
            if verdict.get("decision") != "different":
                left, right = (p.strip() for p in key.split("<->"))
                rows.append((left, right, verdict))
    return rows


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--comparisons", type=Path, default=COMPARISONS,
                    help="comparison JSON (default: %(default)s)")
    ap.add_argument("--section", choices=("within", "cross"), default="within",
                    help="which section to render (default: %(default)s)")
    ap.add_argument("--out", type=Path, default=OUT,
                    help="output PNG path")
    args = ap.parse_args()

    data = json.loads(args.comparisons.read_text())
    rows = _collect(data, args.section)
    if not rows:
        print(f"No {args.section} exceptions found — nothing to render.")
        return

    # sheet dimensions
    W = PAD * 2 + THUMB_W * 2 + COL_GAP + 260  # extra panel for explanation
    H = PAD * 2 + len(rows) * ROW_H
    sheet = Image.new("RGB", (W + 100, H), (250, 250, 250))
    draw = ImageDraw.Draw(sheet)
    f_big = _font(22)
    f_small = _font(16)
    f_smallb = _font(15)

    titles = {
        "within": "WITHIN-folder MATCHING EXCEPTIONS — same-vehicle pairs judged different",
        "cross": "CROSS-vehicle FALSE MATCHES — different vehicles judged same",
    }
    draw.text((PAD, PAD - 6), titles[args.section], font=f_big, fill=(0, 0, 0))

    y = PAD * 2
    for i, (left_rel, right_rel, verdict) in enumerate(rows):
        top = y + PAD
        header = f"[{i + 1}] {left_rel}  vs  {right_rel}"
        draw.text((PAD + 4, top - 30), header, font=f_smallb, fill=(30, 30, 30))

        left_img = _thumb(left_rel)
        right_img = _thumb(right_rel)
        sheet.paste(left_img, (PAD, top))
        sheet.paste(right_img, (PAD + THUMB_W + COL_GAP, top))

        draw.text((PAD, top + THUMB_H + 6), left_rel, font=f_smallb, fill=(0, 80, 160))
        draw.text((PAD + THUMB_W + COL_GAP, top + THUMB_H + 6), right_rel,
                  font=f_smallb, fill=(200, 20, 20))

        # explanation panel
        ex = verdict.get("explanation", "") or ""
        conf = verdict.get("confidence")
        decision = verdict.get("decision") or ("ERROR: " + str(verdict.get("error", "")))
        color = (200, 20, 20)
        fallback = verdict.get("fallback")
        exx = PAD + THUMB_W * 2 + COL_GAP + 14
        draw.text((exx, top + 6), f"Verdict: {decision}", font=f_smallb, fill=color)
        y2 = top + 30
        if fallback:
            draw.text((exx, y2), f"(via {fallback})", font=f_small, fill=(120, 120, 120))
            y2 += 20
        draw.text((exx, y2), f"confidence: {conf}", font=f_small, fill=(60, 60, 60))
        wrap = []
        for line in ex.split():
            if not wrap or len(wrap[-1]) + len(line) + 1 > 34:
                wrap.append("")
            wrap[-1] = (wrap[-1] + " " + line).strip()
        for j, line in enumerate(wrap[:10]):
            draw.text((exx, y2 + 24 + j * 17), line, font=f_small, fill=(40, 40, 40))

        y = top + THUMB_H + LABEL_H

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"Saved {len(rows)} exception pairs sheet -> {out}")
    print(f"Dimensions: {sheet.width}x{sheet.height}")


if __name__ == "__main__":
    main()