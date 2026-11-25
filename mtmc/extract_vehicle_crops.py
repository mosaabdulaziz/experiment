"""Detect and crop vehicles in every video under products/el-mohandes/videos/.

Runs a COCO vehicle detector (car/truck/bus/motorcycle) on the GPU, tracks
vehicles with the same greedy IoU matcher as yolo_detector.py, and saves a
time-sampled set of crops per tracklet (~3 crops/sec of tracklet by default)
instead of every frame. Everything happens in ONE decode pass over each video
— detect, track, and crop the same frame before moving on — so there is no
second seek-heavy pass over the (multi-hundred-MB) videos.

Saves per-video sessions to:
    mtmc/output/vehicle_crops/<video_stem>/
        crops/         — individual vehicle crop images
        vehicles.csv   — manifest: frame, track_id, crop_file, x1, y1, x2, y2, conf
        .done          — completion marker (resume skips videos with this)

Usage:
    .venv/bin/python -m mtmc.extract_vehicle_crops
    .venv/bin/python -m mtmc.extract_vehicle_crops --model yolo11m.pt --imgsz 1920
    .venv/bin/python -m mtmc.extract_vehicle_crops --videos D01-crop.mp4 D21-crop.mp4
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .yolo_detector import Tracklet, VEHICLE_CLASSES, _greedy_match, _iou_matrix

VIDEOS_DIR = Path("products/el-mohandes/videos")
OUTPUT_ROOT = Path("mtmc/output/vehicle_crops")


class _CsvWriter:
    """csv.writer wrapper that also counts rows written."""

    def __init__(self, path: Path):
        self._f = open(path, "w", newline="")
        self._w = csv.writer(self._f)
        self.rows = 0

    def writerow(self, row) -> None:
        self._w.writerow(row)
        self.rows += 1

    def close(self) -> None:
        self._f.close()


def _detect(model: YOLO, frame: np.ndarray, conf: float, imgsz: int,
            device: str | None) -> tuple[np.ndarray, np.ndarray]:
    """Run the model on one frame, keeping only vehicle classes."""
    results = model(frame, imgsz=imgsz, conf=conf, verbose=False, device=device)[0]
    if results.boxes is None or len(results.boxes) == 0:
        return np.zeros((0, 4)), np.zeros(0)
    boxes = results.boxes.xyxy.cpu().numpy()
    clss = results.boxes.cls.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()
    mask = np.isin(clss.astype(int), list(VEHICLE_CLASSES))
    return boxes[mask], confs[mask]


def _save_crop(frame: np.ndarray, box: np.ndarray, crops_dir: Path,
               tid: int, frame_idx: int, padding: int, w: int, h: int
               ) -> tuple[str | None, tuple[int, int, int, int]]:
    """Crop a box out of a frame and write it. Returns (filename, box_ints)."""
    x1, y1, x2, y2 = [int(v) for v in box]
    c_y1, c_y2 = max(0, y1 - padding), min(h, y2 + padding)
    c_x1, c_x2 = max(0, x1 - padding), min(w, x2 + padding)
    crop = frame[c_y1:c_y2, c_x1:c_x2]
    if crop.size == 0:
        return None, (x1, y1, x2, y2)
    fname = f"t{tid:04d}_f{frame_idx:05d}.jpg"
    cv2.imwrite(str(crops_dir / fname), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return fname, (x1, y1, x2, y2)


def extract_video(
    model: YOLO,
    video_path: Path,
    out_dir: Path,
    conf: float,
    imgsz: int,
    iou_thr: float,
    sample_rate: float,
    max_crops: int,
    padding: int,
    min_width: int,
    min_height: int,
    min_area: int,
    min_tracklet_frames: int,
    max_lost_frames: int,
    device: str | None,
    max_frames: int,
    resume: bool,
) -> dict:
    """Single decode pass: detect + track + save ~sample_rate crops/sec/tracklet."""
    done_file = out_dir / ".done"
    if resume and done_file.exists():
        print(f"  [SKIP] {video_path.name} already processed")
        return {"video": video_path.name, "frames": 0, "crops": 0, "skipped": True}

    crops_dir = out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [WARN] Cannot open {video_path.name}, skipping")
        return {"video": video_path.name, "frames": 0, "crops": 0, "error": True}
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if max_frames > 0:
        total_frames = min(total_frames, max_frames)
    # Save ~sample_rate crops per second: one every `stride` video frames.
    stride = max(1, int(round(fps / sample_rate))) if sample_rate > 0 else 1
    print(f"  {video_path.name}: {w}x{h} @ {fps:.1f}fps, {total_frames} frames "
          f"(crop every {stride}th frame = {fps/stride:.1f}/s)")

    # active tracklets; each entry also carries inline crop bookkeeping
    active: dict[int, Tracklet] = {}
    # tid -> (frame of last saved crop, crops saved so far)
    last_saved: dict[int, int] = {}
    saved_count: dict[int, int] = {}
    # staged crops for tracklets not yet confirmed (>= min_tracklet_frames)
    # tid -> list of (frame_idx, box, conf, filename, box_ints)
    staged: dict[int, list] = {}
    next_id = 0
    seen: dict[int, int] = {}  # tid -> last frame seen (for aging)

    csv_path = out_dir / "vehicles.csv"
    writer = _CsvWriter(csv_path)
    writer.writerow(["frame", "track_id", "crop_file", "x1", "y1", "x2", "y2", "conf"])
    t0 = time.time()
    frame_idx = 0

    while frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        boxes, confs = _detect(model, frame, conf, imgsz, device)

        # ---- match detections to active tracklets --------------------
        matched_det: set[int] = set()
        matched_tid: set[int] = set()
        if active:
            active_ids = list(active)
            prev_boxes = np.vstack([active[tid].boxes[-1] for tid in active_ids])
            iou = _iou_matrix(boxes, prev_boxes)
            for di, ti in _greedy_match(iou, iou_thr):
                tid = active_ids[ti]
                active[tid].add(frame_idx, boxes[di], confs[di])
                seen[tid] = frame_idx
                matched_det.add(di)
                matched_tid.add(tid)
                _maybe_save(active[tid], frame, frame_idx, crops_dir, stride,
                                max_crops, padding, w, h,
                                min_width, min_height, min_area,
                                last_saved, saved_count, staged, writer,
                                min_tracklet_frames)

        # ---- age out lost tracklets ---------------------------------
        finished = []
        for tid in list(active):
            if tid not in matched_tid and frame_idx - seen.get(tid, frame_idx) > max_lost_frames:
                finished.append(tid)
        for tid in finished:
            t = active.pop(tid)
            _flush_track(t, crops_dir, staged, writer, min_tracklet_frames,
                         last_saved, saved_count)
            last_saved.pop(tid, None)
            saved_count.pop(tid, None)

        # ---- start new tracklets -------------------------------------
        for di in range(len(boxes)):
            if di in matched_det:
                continue
            t = Tracklet(track_id=next_id)
            next_id += 1
            t.add(frame_idx, boxes[di], confs[di])
            active[t.track_id] = t
            seen[t.track_id] = frame_idx
            staged[t.track_id] = []
            _maybe_save(t, frame, frame_idx, crops_dir, stride, max_crops,
                        padding, w, h, min_width, min_height, min_area,
                        last_saved, saved_count, staged, writer,
                        min_tracklet_frames)

        frame_idx += 1
        if frame_idx % 300 == 0:
            el = time.time() - t0
            print(f"    frame {frame_idx}/{total_frames}  ({frame_idx/el:.0f} fps)  "
                  f"active={len(active)} crops={writer.rows}", end="\r")
    print()

    # flush anything still active at end of video
    for tid in list(active):
        _flush_track(active[tid], crops_dir, staged, writer, min_tracklet_frames,
                     last_saved, saved_count)
    # drop any staged crops of tracks that never reached min length
    for tlist in staged.values():
        for fname, _box_ints in tlist:
            if fname:
                (crops_dir / fname).unlink(missing_ok=True)
    writer.close()
    cap.release()

    saved = writer.rows
    done_file.write_text("completed\n")
    elapsed = time.time() - t0
    print(f"  Done: {saved} crops from {frame_idx} frames in {elapsed:.1f}s "
          f"({frame_idx/elapsed:.0f} fps, GPU)")
    return {"video": video_path.name, "frames": frame_idx, "crops": saved,
            "elapsed_s": round(elapsed, 1)}


def _maybe_save(t: Tracklet, frame: np.ndarray, frame_idx: int, crops_dir: Path,
                stride: int, max_crops: int, padding: int, w: int, h: int,
                min_width: int, min_height: int, min_area: int,
                last_saved: dict, saved_count: dict, staged: dict, writer,
                min_tracklet_frames: int) -> None:
    """Save a crop for `t` if enough video frames passed since the last one."""
    tid = t.track_id
    n = len(t.frames)
    confirmed = n >= min_tracklet_frames
    if saved_count.get(tid, 0) >= max_crops and max_crops > 0:
        return
    last = last_saved.get(tid, -10**9)
    if n > 1 and frame_idx - last < stride:
        return
    box = t.boxes[-1]
    conf = t.detections[-1]
    x1, y1, x2, y2 = [int(v) for v in box]
    if (x2 - x1) < min_width or (y2 - y1) < min_height or ((x2 - x1) * (y2 - y1)) < min_area:
        return
    fname, box_ints = _save_crop(frame, box, crops_dir, tid, frame_idx, padding, w, h)
    if fname is None:
        return
    last_saved[tid] = frame_idx
    saved_count[tid] = saved_count.get(tid, 0) + 1
    if confirmed:
        writer.writerow([frame_idx, tid, fname, *box_ints, f"{conf:.3f}"])
    else:
        staged.setdefault(tid, []).append((fname, box_ints, frame_idx, conf))


def _flush_track(t: Tracklet, crops_dir: Path, staged: dict, writer,
                 min_tracklet_frames: int, last_saved: dict, saved_count: dict) -> None:
    """When a tracklet ends, write staged crops to the CSV (if it survived)."""
    tid = t.track_id
    if len(t.frames) < min_tracklet_frames:
        for fname, _bi, _fi, _c in staged.get(tid, []):
            if fname:
                (crops_dir / fname).unlink(missing_ok=True)
        staged.pop(tid, None)
        return
    for fname, box_ints, fi, conf in staged.get(tid, []):
        writer.writerow([fi, tid, fname, *box_ints, f"{conf:.3f}"])
    staged.pop(tid, None)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default="yolo11s.pt", help="COCO YOLO weights (default: yolo11s.pt)")
    ap.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    ap.add_argument("--imgsz", type=int, default=1280, help="YOLO input size (default 1280)")
    ap.add_argument("--iou", type=float, default=0.45, help="Tracker IoU threshold")
    ap.add_argument("--device", default=None, help="CUDA device (default: auto-pick cuda:0)")
    ap.add_argument("--sample-rate", type=float, default=3.0,
                    help="Crops saved per second of tracklet (default: 3, 0 = every frame)")
    ap.add_argument("--max-crops", type=int, default=0,
                    help="Max crops saved per tracklet (default: 0 = unlimited)")
    ap.add_argument("--padding", type=int, default=10, help="Padding around crop (px)")
    ap.add_argument("--min-width", type=int, default=0, help="Drop detections narrower than this")
    ap.add_argument("--min-height", type=int, default=0, help="Drop detections shorter than this")
    ap.add_argument("--min-area", type=int, default=0, help="Drop detections with area below this")
    ap.add_argument("--min-tracklet-frames", type=int, default=5,
                    help="Tracks shorter than this are dropped entirely")
    ap.add_argument("--max-lost-frames", type=int, default=8,
                    help="Track considered lost after this many missed frames")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="Stop each video after this many frames (0 = all)")
    ap.add_argument("--video-dir", default=None, help="Directory with videos (default: products/el-mohandes/videos)")
    ap.add_argument("--out-dir", default=None, help="Output root (default: mtmc/output/vehicle_crops)")
    ap.add_argument("--videos", nargs="*", default=None,
                    help="Specific video filenames (default: all *.mp4 in video-dir)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip videos that already have a .done marker")
    args = ap.parse_args()

    vdir = Path(args.video_dir) if args.video_dir else VIDEOS_DIR
    if args.videos:
        videos = [vdir / v for v in args.videos]
    else:
        videos = sorted(vdir.glob("*.mp4"))
    if not videos:
        print("No videos found.")
        return

    device = args.device
    if device is None:
        device = "0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device != 'cpu' else 'CPU'})")

    print(f"Loading model {args.model} ...")
    t_load = time.time()
    model = YOLO(str(args.model))
    print(f"  loaded in {time.time() - t_load:.1f}s")

    out_root = Path(args.out_dir) if args.out_dir else OUTPUT_ROOT
    print(f"Model: {args.model}  conf={args.conf}  imgsz={args.imgsz}")
    print(f"Output: {out_root}")
    print(f"Processing {len(videos)} video(s)\n")

    summaries = []
    for vp in videos:
        out_dir = out_root / vp.stem
        print(f"[{vp.stem}]")
        summaries.append(extract_video(
            model, vp, out_dir,
            conf=args.conf, imgsz=args.imgsz, iou_thr=args.iou,
            sample_rate=args.sample_rate, max_crops=args.max_crops,
            padding=args.padding,
            min_width=args.min_width, min_height=args.min_height, min_area=args.min_area,
            min_tracklet_frames=args.min_tracklet_frames,
            max_lost_frames=args.max_lost_frames,
            device=device, max_frames=args.max_frames, resume=args.resume,
        ))
        print()

    print("=" * 60)
    print(f"{'Video':<32} {'Frames':>7} {'Crops':>7} {'Time':>7}")
    print("-" * 60)
    for s in summaries:
        err = " [SKIPPED]" if s.get("skipped") else (" [ERROR]" if s.get("error") else "")
        print(f"{s['video']:<32} {s['frames']:>7} {s['crops']:>7} {s.get('elapsed_s', '-'):>7}{err}")
    total = sum(s["crops"] for s in summaries)
    print("-" * 60)
    print(f"{'TOTAL':<32} {'':>7} {total:>7}")


if __name__ == "__main__":
    main()