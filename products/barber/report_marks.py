"""Alert pictures marked for the reports (2026-10-18, user: like the dashboard's Preview): a red box round the
waiting customer, or the long haircut's chair zone outline, with a small tag -- the chair, or the wait's ticket.
Latin text only (chair names, #ticket), so one picture serves both languages."""
from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

RED = (229, 57, 53)
FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


def mark(jpeg: bytes, box: tuple | None, outline: list | None, tag: str | None) -> bytes:
    """box: normalised (x, y, w, h); outline: normalised polygon points, drawn instead of the box. A picture that
    does not open, or nothing to draw, comes back unchanged."""
    if not box and not (outline and len(outline) >= 3):
        return jpeg
    try:
        im = Image.open(io.BytesIO(jpeg)).convert("RGB")
    except OSError:
        return jpeg
    w, h = im.size
    d = ImageDraw.Draw(im)
    lw = max(3, round(w / 200))                 # a report shows the picture at about a third of its size
    if outline and len(outline) >= 3:
        pts = [(x * w, y * h) for x, y in outline]
        d.line(pts + [pts[0]], fill=RED, width=lw, joint="curve")
        x0, y0 = min(p[0] for p in pts), min(p[1] for p in pts)
    else:
        x, y, bw, bh = box
        x0, y0 = x * w, y * h
        d.rectangle([x0, y0, x0 + bw * w, y0 + bh * h], outline=RED, width=lw)
    if tag:
        font = ImageFont.truetype(str(FONT), max(12, round(h / 24))) if FONT.is_file() else ImageFont.load_default()
        l, t, r, b = d.textbbox((0, 0), tag, font=font)
        pad = 4
        tw, th = r - l + 2 * pad, b - t + 2 * pad
        tx = min(max(0.0, x0), w - tw)
        ty = y0 - th - 2 if y0 - th - 2 >= 0 else y0 + lw + 2        # on top of the mark, else just inside it
        d.rounded_rectangle([tx, ty, tx + tw, ty + th], radius=4, fill=(255, 255, 255))
        d.text((tx + pad - l, ty + pad - t), tag, fill=RED, font=font)
    out = io.BytesIO()
    im.save(out, "JPEG", quality=88)
    return out.getvalue()
