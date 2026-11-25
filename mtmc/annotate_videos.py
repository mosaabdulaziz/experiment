#!/usr/bin/env python3
"""Render annotated videos from a trained detector for manual model review.

Runs a YOLO model over video files frame-by-frame, draws class + confidence
boxes on the ORIGINAL frame (downscaled to --max-side for a practical review
size), and writes an annotated mp4 per input.

Usage:
    # the 5 review clips from products/el-mohandes/videos, fine-tuned 7-class vehicle model:
    .venv/bin/python -m mtmc.annotate_videos

    # any files / a directory:
    .venv/bin/python -m mtmc.annotate_videos --videos products/el-mohandes/videos/5-CH23-enter.mp4 products/el-mohandes/videos/4-ch05_clip.mp4

    # stock COCO weights for comparison:
    .venv/bin/python -m mtmc.annotate_videos --model yolo11s.pt --conf 0.3

    # disable the nested-box suppression (contained boxes kept):
    .venv/bin/python -m mtmc.annotate_videos --containment 0
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from mtmc.detection_postprocess import containment_suppression

DEFAULT_MODEL = Path("products/el-mohandes/runs/vehicle_detector/vehicle_yolo11s/weights/best.pt")
DEFAULT_VIDEOS = [
    Path("products/el-mohandes/videos/1-D41-enter.mp4"),
    Path("products/el-mohandes/videos/2-D16_clip.mp4"),
    Path("products/el-mohandes/videos/3-D52-enter.mp4"),
    Path("products/el-mohandes/videos/4-ch05_clip.mp4"),
    Path("products/el-mohandes/videos/5-CH23-enter.mp4"),
]
DEFAULT_OUT = Path("mtmc/output/annotated_videos")

# One color per class id (BGR), distinct for a 7-class palette.
CLASS_COLORS = [
    (0, 0, 255), (0, 165, 255), (0, 215, 255), (0, 255, 0),
    (255, 0, 0), (230, 216, 173), (0, 128, 255),
]


def class_color(cls: int):
    return CLASS_COLORS[cls % len(CLASS_COLORS)]


def annotate_video(video: Path, out_path: Path, imgsz: int, conf: float,
                   max_side: int, every: int, track: bool, weights_path: str,
                   containment: float, hold_frames: int):
    # Fresh model per video: ultralytics' tracker state (persist=True) lives on
    # the predictor, so a new instance cleanly resets ids between videos.
    model = YOLO(str(weights_path))
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"can't open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_h = int(src_h * (max_side / src_w)) if src_w > max_side else src_h
    out_w = min(src_w, max_side)

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (out_w, out_h))
    if not writer.isOpened():
        raise SystemExit(f"can't open writer for {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    scale = out_w / src_w
    counts = Counter()
    # Track id -> canonical box (largest recently-seen size for this object),
    # last detection center (for motion transfer), class, conf, last frame.
    canon: dict[int, dict] = {}
    ema = 0.55

    def _draw(bx, c, cf, tid):
        x1, y1, x2, y2 = (int(v * scale) for v in bx)
        counts[names[c]] += 1
        col = class_color(c)
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, max(1, int(0.0016 * out_w)))
        label = f"{names[c]} {cf:.2f}" + (f" #{tid}" if tid is not None else "")
        fs = max(0.45, 0.00055 * out_w)
        lw = 2 if fs >= 0.9 else 1
        (tw, th), _base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, lw)
        if y1 - th - 6 < 0:
            cv2.rectangle(frame, (x1, y2), (x1 + tw, y2 + th + 6), col, -1)
            cv2.putText(frame, label, (x1, y2 + th + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), lw)
        else:
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), col, -1)
            cv2.putText(frame, label, (x1, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), lw)

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % every != 0:
            i += 1
            continue
        if track:
            results = model.track(frame, imgsz=imgsz, conf=conf, persist=True,
                                  tracker="bytetrack.yaml", verbose=False)[0]
        else:
            results = model.predict(frame, imgsz=imgsz, conf=conf, verbose=False)[0]
        boxes = results.boxes
        names = results.names if hasattr(results, "names") else model.names
        # Inference runs on the full-res frame; downscale for the output FIRST,
        # then draw scaled box coords on it. Drawing on full-res then resizing
        # would shrink boxes a second time and drift them up-left.
        if scale != 1.0:
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

        xyxy = boxes.xyxy.cpu().numpy()
        cfs = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        ids = (boxes.id.cpu().numpy().astype(int) if (track and boxes.id is not None and len(boxes))
               else None)

        if not track:
            # no tracker: plain per-frame containment suppression
            if containment > 0 and len(xyxy):
                keep = containment_suppression(xyxy, containment, cfs)
                xyxy, cfs, cls = xyxy[keep], cfs[keep], cls[keep]
            for j, (x1, y1, x2, y2) in enumerate(xyxy):
                _draw((x1, y1, x2, y2), int(cls[j]), float(cfs[j]), None)
        else:
            present = set(int(t) for t in ids) if ids is not None else set()
            # Virtual boxes: ids we hold but that produced no detection this
            # frame (whole-vehicle box flickered out). They stay draw-able for
            # hold_frames frames and also SUPPRESS new cab-only detections
            # that reappear for the same vehicle as a different id.
            vir_xy, vir_cf, vir_cl, vir_id = [], [], [], []
            for tid, rec in canon.items():
                if tid in present or i - rec["seen"] > hold_frames:
                    continue
                vir_xy.append(rec["box"]); vir_cf.append(rec["conf"])
                vir_cl.append(rec["cls"]); vir_id.append(tid)
            vir_xy = (np.asarray(vir_xy) if vir_xy else np.zeros((0, 4), np.float32))
            comb = np.vstack([xyxy, vir_xy]) if len(vir_xy) else xyxy
            comb_cf = np.concatenate([cfs, vir_cf]) if len(vir_xy) else cfs
            comb_cl = np.concatenate([cls, vir_cl]) if len(vir_xy) else cls
            n_cur = len(xyxy)

            if containment > 0 and len(comb) > 1:
                keep = containment_suppression(comb, containment, comb_cf)
            else:
                keep = np.ones(len(comb), dtype=bool)
            cur_k = keep[:n_cur]
            vir_k = keep[n_cur:]

            # draw surviving virtual (held) boxes as-is
            for vj, kk in enumerate(vir_k):
                if kk:
                    _draw(tuple(vir_xy[vj]), int(vir_cl[vj]), float(vir_cf[vj]),
                          int(vir_id[vj]))

            # surviving live detections: hysteresis canonical update
            idx = np.where(cur_k)[0]
            if ids is None:
                # frames where the tracker produced boxes without ids: draw raw
                for j in idx:
                    _draw(tuple(xyxy[j]), int(cls[j]), float(cfs[j]), None)
            else:
                for j in idx:
                    bx = tuple(xyxy[j]); tid = int(ids[j]); c = int(cls[j])
                    cf = float(cfs[j])
                    cx, cy = (bx[0] + bx[2]) / 2.0, (bx[1] + bx[3]) / 2.0
                    area = (bx[2] - bx[0]) * (bx[3] - bx[1])
                    rec = canon.get(tid)
                    if rec is None or i - rec["seen"] > hold_frames:
                        canon[tid] = {"box": list(bx), "cls": c, "conf": cf,
                                      "det_c": (cx, cy), "seen": i}
                        _draw(bx, c, cf, tid)
                        continue
                    oarea = (rec["box"][2] - rec["box"][0]) * (rec["box"][3] - rec["box"][1])
                    if area >= 0.85 * oarea:
                        # consistent size -> blend both position and size
                        nb = [ema * new + (1 - ema) * old for new, old in
                              zip(bx, rec["box"])]
                        rec["box"], rec["cls"], rec["conf"], rec["det_c"], rec["seen"] = \
                            nb, c, cf, (cx, cy), i
                    else:
                        # smaller (cab-only) detection: move the held whole-vehicle
                        # box by this detection's motion (center delta since its
                        # last detection), keep the held SIZE.
                        dx = cx - rec["det_c"][0]
                        dy = cy - rec["det_c"][1]
                        b = rec["box"]
                        rec["box"] = [b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy]
                        rec["det_c"], rec["seen"], rec["conf"] = (cx, cy), i, cf
                    _draw(tuple(rec["box"]), rec["cls"], rec["conf"], tid)

            # prune dead holds
            for tid in [t for t, r in canon.items() if i - r["seen"] > hold_frames]:
                del canon[tid]
        writer.write(frame)
        i += 1
        if i % 300 == 0:
            print(f"    {video.name}: {min(i, n_frames)}/{n_frames} frames, "
                  f"{len(counts)} boxes drawn", flush=True)
    cap.release()
    writer.release()
    return counts, n_frames // every


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=str, default=str(DEFAULT_MODEL),
                    help=f"model weights (default {DEFAULT_MODEL})")
    ap.add_argument("--videos", nargs="*", default=[str(v) for v in DEFAULT_VIDEOS],
                    help="input videos (default: the first 5 review clips)")
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT),
                    help=f"output dir (default {DEFAULT_OUT})")
    ap.add_argument("--imgsz", type=int, default=1280,
                    help="inference size (default 1280, matches training)")
    ap.add_argument("--conf", type=float, default=0.5,
                    help="confidence threshold (default 0.5; 0.25 to match the "
                         "DeepStream pre-cluster-threshold)")
    ap.add_argument("--max-side", type=int, default=2560,
                    help="cap annotated-video width to this (default 2560)")
    ap.add_argument("--every", type=int, default=1,
                    help="process every Nth frame (default 1)")
    ap.add_argument("--device", type=str, default="0",
                    help="cuda device or cpu (default 0)")
    ap.add_argument("--no-track", action="store_true",
                    help="disable ByteTrack + box smoothing; per-frame detections only")
    ap.add_argument("--containment", type=float, default=0.85,
                    help="drop any box whose area is >= this fraction inside another "
                         "box in the same frame (default 0.85; 0 disables)")
    ap.add_argument("--hold-frames", type=int, default=15,
                    help="keep drawing a track's canonical box for this many frames "
                         "while its detection flickers out (default 15; 0 disables)")
    args = ap.parse_args()

    if not Path(args.model).exists():
        raise SystemExit(f"model not found: {args.model}")

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    video_args = []
    for v in args.videos:
        p = Path(v)
        if p.is_dir():
            video_args += sorted(p.glob("*.mp4"))
        elif p.exists():
            video_args.append(p)
        else:
            print(f"WARNING: skipping missing {v}", flush=True)

    tracking = not args.no_track
    print(f"model: {args.model}  imgsz={args.imgsz} conf={args.conf} "
          f"track={'on' if tracking else 'off'} contained={'on' if args.containment > 0 else 'off'}")
    for video in video_args:
        out_path = out_root / f"{video.stem}_annotated.mp4"
        print(f"annotating {video.name} -> {out_path}", flush=True)
        counts, frames = annotate_video(video, out_path, args.imgsz,
                                        args.conf, args.max_side, args.every,
                                        tracking, args.model, args.containment,
                                        args.hold_frames)
        total = sum(counts.values())
        if counts:
            per = ", ".join(f"{k} {v}" for k, v in counts.most_common())
            print(f"  done: {frames} frames, {total} boxes ({per})", flush=True)
        else:
            print(f"  done: {frames} frames, no detections above conf={args.conf}",
                  flush=True)


if __name__ == "__main__":
    main()