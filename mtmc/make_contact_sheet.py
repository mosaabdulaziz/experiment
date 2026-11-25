#!/usr/bin/env python3
"""Render a contact sheet per video: one row of frames sampled at fixed times.

Lets a human eyeball whether the 13 sources share content and whether any
burned-in clock/watermark is visible. Output: data/frames_dump/contact/<name>.jpg
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

UPLOADS = Path("data/uploads")
OUT = Path("mtmc/output/contact_sheets")
FILES = sorted(UPLOADS.glob("*.mp4"))
TIMES = [2, 20, 40, 60, 80, 100, 120, 138]
THUMB_W = 320


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for p in FILES:
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            print("skip", p)
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        thumbs = []
        for t in TIMES:
            fnum = min(int(t * fps), max(n - 1, 0))
            cap.set(cv2.CAP_PROP_POS_FRAMES, fnum)
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            scale = THUMB_W / w
            thumb = cv2.resize(frame, (THUMB_W, int(h * scale)),
                               interpolation=cv2.INTER_AREA)
            # label with timestamp
            cv2.rectangle(thumb, (0, 0), (THUMB_W, 24), (0, 0, 0), -1)
            cv2.putText(thumb, f"t={t}s  f={fnum}", (6, 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1,
                        cv2.LINE_AA)
            thumbs.append(thumb)
        cap.release()
        if not thumbs:
            continue
        hh = max(t.shape[0] for t in thumbs)
        sheet = np.full((hh, THUMB_W * len(thumbs), 3), 30, np.uint8)
        for i, t in enumerate(thumbs):
            sheet[: t.shape[0], i * THUMB_W:(i + 1) * THUMB_W] = t
        out_path = OUT / (p.stem + ".jpg")
        cv2.imwrite(str(out_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
        print(out_path, sheet.shape)


if __name__ == "__main__":
    main()
