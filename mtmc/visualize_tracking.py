#!/usr/bin/env python3
"""Generate annotated tracking videos using a YOLO detector + ByteTrack.

Uses ultralytics' built-in ByteTrack (Kalman filter motion model, two-stage
IoU matching) instead of a hand-rolled IoU tracker.

Works for both detectors:
  * plate detector  (default) — one box per plate
  * vehicle detector (COCO)   — pass --model yolo11s.pt and a --classes filter

For a vehicle run the boxes are drawn per track with a persistent colour and a
large "ID n" chip, plus a HUD with frame/time/active/unique counts.  Pass
--plates to overlay the plate detector's boxes too: a vehicle box that is
missing where a plate box is present is an obvious miss, which is what makes
these videos useful for judging recall by eye.

Note on resolution: for 16:9 sources, running the detector on the native frame
at --imgsz N sees the same scale as the sidecar's mux (1280x720) fed to an NxN
input, so these videos show what the PGIE effectively sees.

Run:
    .venv/bin/python -m mtmc.visualize_tracking
    .venv/bin/python -m mtmc.visualize_tracking \\
        --model yolo11s.pt --classes car truck bus motorcycle --plates \\
        --videos products/el-mohandes/videos/1-D41-enter.mp4
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# ── Helpers ──────────────────────────────────────────────────────────────

def id_color(tid: int) -> tuple[int, int, int]:
    """Deterministic color from track ID."""
    hue = ((tid * 0.61803398875) % 1.0) * 179.0
    hsv = np.uint8([[[hue, 255, 255]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(b), int(g), int(r)


def resolve_classes(model: YOLO, names: list[str] | None) -> set[int] | None:
    """Map --classes names to class ids; None = keep every class."""
    if not names:
        return None
    lookup = {str(n).lower(): int(i) for i, n in (model.names or {}).items()}
    ids: set[int] = set()
    for want in names:
        w = want.strip().lower()
        if w.isdigit():
            ids.add(int(w))
        elif w in lookup:
            ids.add(lookup[w])
        else:
            raise SystemExit(f"class '{want}' not in model labels: {sorted(lookup)}")
    return ids


def _put_label(frame: np.ndarray, text: str, x: int, y: int,
               color: tuple[int, int, int], font_scale: float, thickness: int,
               inside: bool = False) -> None:
    """Filled chip + high-contrast text, clamped inside the frame."""
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    pad = max(3, thickness)
    y_top = y if inside else y - th - 2 * pad
    y_top = max(0, min(y_top, frame.shape[0] - th - 2 * pad))
    x = max(0, min(x, frame.shape[1] - tw - 2 * pad))
    cv2.rectangle(frame, (x, y_top), (x + tw + 2 * pad, y_top + th + 2 * pad), color, -1)
    cv2.putText(frame, text, (x + pad, y_top + th + pad),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
                thickness, cv2.LINE_AA)


def _hud(frame: np.ndarray, lines: list[str], font_scale: float) -> None:
    """Translucent top-left panel so the HUD never hides the scene."""
    (_, th), base = cv2.getTextSize("Ag", cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    line_h = th + base + 8
    box_h = line_h * len(lines) + 10
    box_w = int(frame.shape[1] * 0.62)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (box_w, box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (12, line_h * (i + 1)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)


# ── Main tracking + visualization ────────────────────────────────────────

def track_and_visualize(
    video_path: str,
    out_path: str,
    model_path: str = "products/el-mohandes/models/finetuned/best.pt",
    conf: float = 0.25,
    imgsz: int = 1280,
    out_width: int = 1280,
    tracker_config: str = "bytetrack.yaml",
    classes: list[str] | None = None,
    plate_model_path: str | None = None,
    conf_plate: float = 0.25,
    max_frames: int = 0,
    start_frame: int = 0,
) -> dict:
    """Run detection + ByteTrack and write an annotated video."""
    model = YOLO(model_path, task="detect")
    keep = resolve_classes(model, classes)
    plate_model = YOLO(plate_model_path, task="detect") if plate_model_path else None

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    scale = min(1.0, out_width / w)
    out_w, out_h = int(w * scale), int(h * scale)
    vname = Path(video_path).stem
    # Draw sizes tied to output width so labels stay legible at any resolution.
    box_th = max(2, round(out_w / 320))
    font = round(out_w / 1500, 2)

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
    if not writer.isOpened():
        raise SystemExit(f"could not open video writer for {out_path}")

    t0 = time.time()
    frame_count = 0
    all_ids_seen: set[int] = set()
    n_plate_boxes = 0

    for result in model.track(
        source=video_path,
        imgsz=imgsz,
        conf=conf,
        iou=0.45,
        tracker=tracker_config,
        stream=True,
        save=False,
        verbose=False,
        persist=False,
    ):
        frame_count += 1
        if frame_count <= start_frame:
            continue
        if max_frames and frame_count > start_frame + max_frames:
            break

        frame = result.orig_img.copy()
        ids, boxes, clss = [], [], []
        if result.boxes is not None and len(result.boxes):
            sel = np.ones(len(result.boxes), dtype=bool)
            if keep is not None:
                sel = np.isin(result.boxes.cls.cpu().numpy().astype(int), list(keep))
            if sel.any():
                boxes = result.boxes.xyxy.cpu().numpy()[sel]
                clss = result.boxes.cls.cpu().numpy().astype(int)[sel]
                if result.boxes.id is not None:
                    ids = result.boxes.id.cpu().numpy().astype(int)[sel].tolist()
                else:
                    ids = [-1] * len(boxes)

        all_ids_seen.update(t for t in ids if t >= 0)

        # Plates first (thin, behind the vehicle boxes) so they never occlude IDs.
        if plate_model is not None:
            pres = plate_model(frame, imgsz=1280, conf=conf_plate, iou=0.45,
                               rect=False, verbose=False)[0]
            if pres.boxes is not None and len(pres.boxes):
                for pb in pres.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = [int(v) for v in pb]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                    n_plate_boxes += 1

        for tid, box, cls in zip(ids, boxes, clss):
            x1, y1, x2, y2 = [int(v) for v in box]
            color = id_color(tid) if tid >= 0 else (160, 160, 160)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_th)
            cname = str((model.names or {}).get(int(cls), cls))
            label = f"ID {tid}  {cname}" if tid >= 0 else cname
            _put_label(frame, label, x1, y1, color, font, max(1, box_th // 2))

        n_active = len(boxes)
        hud_lines = [
            f"{vname}   frame {frame_count}/{total}   t={frame_count / max(fps, 1):5.1f}s",
            f"vehicle boxes now: {n_active}    unique track IDs so far: {len(all_ids_seen)}",
            f"{Path(model_path).name} @ imgsz {imgsz}  conf {conf}"
            + (f"   cyan = plates ({n_plate_boxes} so far)" if plate_model is not None else ""),
        ]
        _hud(frame, hud_lines, font)

        writer.write(cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA))

        if frame_count % 500 == 0:
            elapsed = time.time() - t0
            done = frame_count - start_frame
            print(f"  {frame_count}/{total}  active:{n_active}  unique:{len(all_ids_seen)}"
                  f"  ({done / elapsed if elapsed else 0:.1f} fps)", flush=True)

    writer.release()
    elapsed = time.time() - t0
    print(f"  -> {out_path}  ({len(all_ids_seen)} unique tracks, {elapsed:.1f}s)", flush=True)
    return {"video": video_path, "out": out_path, "unique_ids": len(all_ids_seen),
            "frames": frame_count - start_frame, "seconds": round(elapsed, 1)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos", nargs="*",
                   default=[
                       "products/el-mohandes/videos/experiment/1-D41-enter.mp4",
                       "products/el-mohandes/videos/experiment/2-D52-enter.mp4",
                       "products/el-mohandes/videos/experiment/3-CH23-enter.mp4",
                   ])
    p.add_argument("--out-dir", default="mtmc/output/tracked_videos")
    p.add_argument("--model", default="products/el-mohandes/models/finetuned/best.pt")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--tracker", default="bytetrack.yaml",
                   help="Tracker config: bytetrack.yaml, botsort.yaml, etc.")
    p.add_argument("--out-width", type=int, default=1280)
    p.add_argument("--classes", nargs="*", default=None,
                   help="class names or ids to draw (default: every class the model outputs)")
    p.add_argument("--plates", nargs="?", const="products/el-mohandes/models/finetuned/best.pt", default=None,
                   metavar="WEIGHTS",
                   help="also overlay plate detections (default weights: finetuned/best.pt)")
    p.add_argument("--conf-plate", type=float, default=0.25)
    p.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = all)")
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--suffix", default="", help="appended to output filenames")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for vp in args.videos:
        vname = Path(vp).stem
        out_path = str(out_dir / f"{vname}{args.suffix}_tracked.mp4")
        print(f"\nProcessing {vname}...", flush=True)
        track_and_visualize(
            vp, out_path,
            model_path=args.model, conf=args.conf, imgsz=args.imgsz,
            out_width=args.out_width, tracker_config=args.tracker,
            classes=args.classes, plate_model_path=args.plates,
            conf_plate=args.conf_plate,
            max_frames=args.max_frames, start_frame=args.start_frame,
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
