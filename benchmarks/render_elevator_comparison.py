#!/usr/bin/env python3
"""Render a side-by-side OLD vs NEW annotated comparison video from eval_elevator_video.py JSON.

Reads the source video plus `report_compare.json` (generated on the GPU server with
`--photos-dir ... --compare-buffalo`), and writes an H.264 MP4 placing:
  - Left:  OLD pipeline (yolov8n-face + buffalo_l)
  - Right: NEW pipeline (face_det_lite + auraface)
with color-coded detection/identity diff highlights and live cumulative counters.

Usage:
    python benchmarks/render_elevator_comparison.py \\
        --video "C:/Users/Maha Mamdouh/Downloads/syad_6-left_2026-09-10.mkv" \\
        --report "C:/Users/Maha Mamdouh/Downloads/report_compare.json" \\
        --out-video "C:/Users/Maha Mamdouh/Downloads/elevator_old_vs_new.mp4"
"""
from __future__ import annotations

import argparse
import bisect
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

PANEL_W, PANEL_H = 960, 540
TOP_H, BOT_H = 74, 46
CANVAS_W, CANVAS_H = PANEL_W * 2, TOP_H + PANEL_H + BOT_H


def classify_frame_diff(fr: dict[str, Any]) -> tuple[str, tuple[int, int, int]]:
    """Return a concise diff banner label and BGR badge color for one sampled frame."""
    new_faces = fr.get("faces", [])
    old_faces = fr.get("yolo_faces", [])
    new_names = {f["matched_name"] for f in new_faces if f.get("matched_name")}
    old_names = {f["matched_name"] for f in old_faces if f.get("matched_name")}
    new_extra = sum(1 for f in new_faces if f.get("yolo_iou") is None)
    old_only = len(fr.get("yolo_only_faces", []))

    new_only_ids = sorted(new_names - old_names)
    old_only_ids = sorted(old_names - new_names)

    if new_only_ids and old_only_ids:
        return f"ID DIFF | OLD:{','.join(old_only_ids)} NEW:{','.join(new_only_ids)}", (0, 140, 255)
    if new_only_ids:
        return f"NEW RECOGNIZED ({','.join(new_only_ids)}) | OLD MISSED/UNKNOWN", (200, 180, 0)
    if old_only_ids:
        return f"OLD RECOGNIZED ({','.join(old_only_ids)}) | NEW MISSED/UNKNOWN", (0, 110, 255)
    if old_only > 0 and new_extra > 0:
        return f"BOX DIFF | +{new_extra} NEW_ONLY / -{old_only} YOLO_ONLY", (0, 180, 255)
    if old_only > 0:
        return f"OLD ONLY DETECTED ({old_only} face missed by NEW)", (0, 80, 255)
    if new_extra > 0:
        return f"NEW CAUGHT +{new_extra} EXTRA FACE(S)", (220, 190, 0)
    if new_names:
        return f"BOTH MATCH: {', '.join(sorted(new_names))}", (60, 180, 60)
    if new_faces or old_faces:
        return f"BOTH DETECT ({len(new_faces)} face, unknown)", (90, 120, 90)
    if fr.get("person_boxes"):
        return "PERSON VISIBLE (0 faces)", (80, 80, 80)
    return "EMPTY HALLWAY", (45, 45, 45)


def compute_keep_ranges(
    frames: Sequence[dict[str, Any]],
    mode: str,
    fps: float,
    pad_s: float = 1.0,
) -> Optional[list[tuple[int, int]]]:
    """Return merged [start_frame, end_frame] ranges to render, or None for full video."""
    if mode == "full":
        return None
    pad = int(round(fps * pad_s))
    raw: list[tuple[int, int]] = []
    for fr in frames:
        if mode == "events":
            keep = bool(fr.get("person_boxes") or fr.get("faces") or fr.get("yolo_faces"))
        else:
            label, _ = classify_frame_diff(fr)
            keep = not label.startswith(("BOTH ", "PERSON VISIBLE", "EMPTY"))
        if keep:
            fi = int(fr["frame_idx"])
            raw.append((max(0, fi - pad), fi + pad))
    if not raw:
        return []
    merged: list[tuple[int, int]] = [raw[0]]
    for s, e in raw[1:]:
        if s <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def _frame_in_ranges(idx: int, ranges: Optional[Sequence[tuple[int, int]]]) -> bool:
    if ranges is None:
        return True
    pos = bisect.bisect_right(ranges, (idx, 10**12)) - 1
    return pos >= 0 and ranges[pos][0] <= idx <= ranges[pos][1]


def _draw_panel(
    mux_bgr: np.ndarray,
    fr: dict[str, Any],
    mux_wh: tuple[int, int],
    side: str,
) -> np.ndarray:
    import cv2

    vis = mux_bgr.copy()
    sx = PANEL_W / float(mux_wh[0])
    sy = PANEL_H / float(mux_wh[1])
    panel = cv2.resize(vis, (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA)

    for px1, py1, px2, py2 in fr.get("person_boxes", []):
        cv2.rectangle(panel, (int(px1 * sx), int(py1 * sy)), (int(px2 * sx), int(py2 * sy)), (140, 120, 80), 1)

    if side == "old":
        items = fr.get("yolo_faces")
        if items is None:
            # Fallback if report was generated without yolo_faces (only yolo_only_faces + yolo_bbox).
            items = []
            for fc in fr.get("faces", []):
                if fc.get("yolo_iou") is not None:
                    items.append({
                        "bbox": fc.get("yolo_bbox") or fc["bbox"],
                        "det_conf": fc["det_conf"],
                        "matched_new": True,
                        "matched_name": None,
                    })
            for ym in fr.get("yolo_only_faces", []):
                items.append({
                    "bbox": ym["bbox"],
                    "det_conf": ym["yolo_conf"],
                    "matched_new": False,
                    "matched_name": None,
                })
        for yf in items:
            x1, y1, x2, y2 = yf["bbox"]
            p1, p2 = (int(round(x1 * sx)), int(round(y1 * sy))), (int(round(x2 * sx)), int(round(y2 * sy)))
            name = yf.get("matched_name")
            only_old = not yf.get("matched_new", True)
            if name:
                col = (0, 230, 0)
                tag = f"{name} ({yf['similarity']:.2f})"
            elif only_old:
                col = (0, 90, 255)
                tag = f"OLD-ONLY {yf['det_conf']:.2f}"
            else:
                col = (0, 220, 255)
                tag = f"det {yf['det_conf']:.2f}"
                if yf.get("similarity") is not None:
                    tag += f" [{yf['best_name']}:{yf['similarity']:.2f}]"
            cv2.rectangle(panel, p1, p2, col, 3 if only_old else 2)
            cv2.putText(panel, tag, (p1[0], max(38, p1[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1)
        title = "OLD: yolov8n-face (0.25) + buffalo_l (0.25)"
    else:
        for fc in fr.get("faces", []):
            x1, y1, x2, y2 = fc["bbox"]
            p1, p2 = (int(round(x1 * sx)), int(round(y1 * sy))), (int(round(x2 * sx)), int(round(y2 * sy)))
            name = fc.get("matched_name")
            is_extra = fc.get("yolo_iou") is None and bool(fr.get("yolo_faces") is not None or fr.get("yolo_only_faces") is not None)
            if name and is_extra:
                col = (255, 220, 0)
                tag = f"NEW-WIN {name} ({fc['similarity']:.2f})"
            elif name:
                col = (0, 230, 0)
                tag = f"{name} ({fc['similarity']:.2f})"
            elif is_extra:
                col = (240, 200, 40)
                tag = f"NEW-ONLY {fc['det_conf']:.2f}"
            else:
                col = (0, 220, 255)
                tag = f"det {fc['det_conf']:.2f}"
            if not name and fc.get("similarity") is not None:
                tag += f" [{fc['best_name']}:{fc['similarity']:.2f}]"
            cv2.rectangle(panel, p1, p2, col, 3 if is_extra else 2)
            cv2.putText(panel, tag, (p1[0], max(38, p1[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1)
        for ym in fr.get("yolo_only_faces", []):
            x1, y1, x2, y2 = ym["bbox"]
            p1, p2 = (int(round(x1 * sx)), int(round(y1 * sy))), (int(round(x2 * sx)), int(round(y2 * sy)))
            cv2.rectangle(panel, p1, p2, (0, 0, 255), 1)
            cv2.putText(panel, "MISSED", (p1[0], min(PANEL_H - 6, p2[1] + 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1)
        title = "NEW: face_det_lite (0.25) + auraface (0.28)"

    cv2.rectangle(panel, (0, 0), (PANEL_W, 28), (22, 22, 22), -1)
    cv2.putText(panel, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1)
    return panel


def _compose_canvas(
    mux_bgr: np.ndarray,
    fr: dict[str, Any],
    mux_wh: tuple[int, int],
    old_cum: dict[str, int],
    new_cum: dict[str, int],
    old_ids_seen: set[str],
    new_ids_seen: set[str],
) -> np.ndarray:
    import cv2

    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    canvas[:] = (18, 18, 18)

    left = _draw_panel(mux_bgr, fr, mux_wh, "old")
    right = _draw_panel(mux_bgr, fr, mux_wh, "new")
    canvas[TOP_H : TOP_H + PANEL_H, 0:PANEL_W] = left
    canvas[TOP_H : TOP_H + PANEL_H, PANEL_W:CANVAS_W] = right
    cv2.line(canvas, (PANEL_W, TOP_H), (PANEL_W, TOP_H + PANEL_H), (90, 90, 90), 2)

    diff_text, badge_col = classify_frame_diff(fr)
    cv2.rectangle(canvas, (10, 10), (760, 62), badge_col, -1)
    cv2.putText(canvas, diff_text, (22, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)

    meta_txt = f"{fr['timestamp_hms']}  (frame {fr['frame_idx']})"
    cv2.putText(canvas, meta_txt, (790, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (235, 235, 235), 2)
    score_txt = (
        f"CUMULATIVE ->  OLD det:{old_cum['det']} rec:{old_cum['rec']}  |  "
        f"NEW det:{new_cum['det']} rec:{new_cum['rec']}"
    )
    cv2.putText(canvas, score_txt, (790, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (180, 230, 255), 1)

    bot_y = TOP_H + PANEL_H + 30
    cv2.putText(canvas, f"OLD IDs locked so far: {', '.join(sorted(old_ids_seen)) or 'none'}", (16, bot_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.56, (210, 210, 210), 1)
    cv2.putText(canvas, f"NEW IDs locked so far: {', '.join(sorted(new_ids_seen)) or 'none'}", (PANEL_W + 16, bot_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.56, (180, 255, 180), 1)
    return canvas


def _open_writer(out_path: Path, fps: float) -> tuple[Any, str]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        cmd = [
            ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{CANVAS_W}x{CANVAS_H}", "-r", f"{fps:.3f}", "-i", "-",
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return proc, "ffmpeg"
    import cv2

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (CANVAS_W, CANVAS_H))
    return writer, "cv2"


def render_comparison(
    video_path: Path,
    report_path: Path,
    out_video: Path,
    mode: str = "events",
    pad_s: float = 1.0,
) -> int:
    import cv2

    rep = json.loads(report_path.read_text(encoding="utf-8"))
    frames = rep["frames"]
    if not frames:
        raise ValueError(f"No sampled frames in {report_path}")
    mux_wh = tuple(rep["video"].get("mux_wh") or (1280, 720))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or rep["video"].get("fps") or 13.0)
    keep_ranges = compute_keep_ranges(frames, mode=mode, fps=fps, pad_s=pad_s)

    sample_idxs = [int(fr["frame_idx"]) for fr in frames]
    writer, kind = _open_writer(out_video, fps)

    old_cum = {"det": 0, "rec": 0}
    new_cum = {"det": 0, "rec": 0}
    old_ids_seen: set[str] = set()
    new_ids_seen: set[str] = set()
    seen_sample_pos = -1

    idx = 0
    written = 0
    try:
        while True:
            pos = bisect.bisect_right(sample_idxs, idx) - 1
            if pos < 0:
                pos = 0
            while seen_sample_pos < pos:
                seen_sample_pos += 1
                sfr = frames[seen_sample_pos]
                old_list = sfr.get("yolo_faces")
                if old_list is None:
                    old_cum["det"] += sum(1 for f in sfr.get("faces", []) if f.get("yolo_iou") is not None) + len(sfr.get("yolo_only_faces", []))
                else:
                    old_cum["det"] += len(old_list)
                    for yf in old_list:
                        if yf.get("matched_name"):
                            old_cum["rec"] += 1
                            old_ids_seen.add(yf["matched_name"])
                new_cum["det"] += len(sfr.get("faces", []))
                for fc in sfr.get("faces", []):
                    if fc.get("matched_name"):
                        new_cum["rec"] += 1
                        new_ids_seen.add(fc["matched_name"])

            if not _frame_in_ranges(idx, keep_ranges):
                if not cap.grab():
                    break
                idx += 1
                continue

            ok, frame = cap.read()
            if not ok or frame is None:
                break
            mux = cv2.resize(frame, mux_wh, interpolation=cv2.INTER_LINEAR)
            canvas = _compose_canvas(mux, frames[pos], mux_wh, old_cum, new_cum, old_ids_seen, new_ids_seen)
            if kind == "ffmpeg":
                writer.stdin.write(canvas.tobytes())
            else:
                writer.write(canvas)
            written += 1
            idx += 1
    finally:
        cap.release()
        if kind == "ffmpeg":
            writer.stdin.close()
            rc = writer.wait()
            if rc != 0:
                err = writer.stderr.read().decode(errors="replace") if writer.stderr else ""
                raise RuntimeError(f"ffmpeg exited with {rc}: {err[-500:]}")
        else:
            writer.release()
    print(f"Wrote {out_video} ({written} frames at {fps:.1f} fps, encoder={kind})")
    return written


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Render side-by-side OLD vs NEW comparison video from report.json")
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--report", required=True, type=Path)
    ap.add_argument("--out-video", required=True, type=Path)
    ap.add_argument("--mode", choices=("events", "diffs", "full"), default="events",
                    help="events=all person windows +/-1s (default); diffs=only differing frames +/-1s; full=entire video")
    ap.add_argument("--pad-s", type=float, default=1.0)
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    render_comparison(args.video, args.report, args.out_video, mode=args.mode, pad_s=args.pad_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
