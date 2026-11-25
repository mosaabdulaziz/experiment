"""YOLO-based vehicle detection + single-camera tracking.

Produces per-camera tracklets with bounding boxes and frame timestamps.

Two detector modes:
  1. VehicleDetector  — generic COCO YOLO (car/truck/bus/motorcycle)
  2. PlateVehicleDetector — fine-tuned plate detector; each plate = one vehicle.
     Uses ultralytics built-in ByteTrack (Kalman filter + two-stage IoU matching)
     instead of a hand-rolled greedy matcher.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# COCO vehicle classes: car=2, motorcycle=3, bus=5, truck=7
VEHICLE_CLASSES = {2, 3, 5, 7}

# Default plate detector path
DEFAULT_PLATE_DETECTOR = Path("products/el-mohandes/models/finetuned/best.pt")

# Plate-to-vehicle expansion ratios (plate is ~40% of vehicle width, ~15-25% height)
PLATE_EXPAND_W = 2.5   # expand plate width by this factor to approximate vehicle width
PLATE_EXPAND_H = 3.5   # expand plate height by this factor to approximate vehicle height
PLATE_OFFSET_Y = -0.3  # shift expanded box upward (plate is at bottom of vehicle)


@dataclass
class Tracklet:
    """A single-camera tracklet: a sequence of bounding boxes over frames."""
    track_id: int
    frames: list[int] = field(default_factory=list)
    boxes: list[np.ndarray] = field(default_factory=list)   # [x1,y1,x2,y2] per frame
    detections: list[float] = field(default_factory=list)   # conf per frame

    @property
    def start_frame(self) -> int:
        return self.frames[0]

    @property
    def end_frame(self) -> int:
        return self.frames[-1]

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame + 1

    def add(self, frame: int, box: np.ndarray, conf: float) -> None:
        self.frames.append(frame)
        self.boxes.append(box.astype(float))
        self.detections.append(conf)

    def mean_box(self) -> np.ndarray:
        """Mean bounding box across all frames."""
        return np.mean(np.array(self.boxes), axis=0)

    def crop_at_frame(self, frame_idx: int) -> np.ndarray | None:
        """Get the box coordinates at a specific frame index."""
        try:
            idx = self.frames.index(frame_idx)
            return self.boxes[idx]
        except ValueError:
            return None


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box in a (N,4) and b (M,4), both in x1y1x2y2 format."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    inter_x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    inter_y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    inter_x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    inter_y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(inter_x2 - inter_x1, 0, None) * np.clip(inter_y2 - inter_y1, 0, None)
    a_area = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    b_area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = a_area[:, None] + b_area[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def _greedy_match(iou: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """Greedy one-to-one row→col assignment by descending IoU."""
    pairs = []
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    flat = [(iou[i, j], i, j)
            for i in range(iou.shape[0]) for j in range(iou.shape[1])]
    for score, i, j in sorted(flat, key=lambda t: -t[0]):
        if score < threshold or i in used_rows or j in used_cols:
            continue
        pairs.append((i, j))
        used_rows.add(i)
        used_cols.add(j)
    return pairs


class VehicleDetector:
    """YOLO detection + greedy IoU tracker for a single camera."""

    def __init__(
        self,
        model_path: str = "yolo11s.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        min_tracklet_frames: int = 5,
        max_lost_frames: int = 8,
        imgsz: int = 1280,
    ):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.min_tracklet_frames = min_tracklet_frames
        self.max_lost_frames = max_lost_frames
        self.imgsz = imgsz

    def detect_frame(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run YOLO on one frame. Returns (boxes xyxy, confs)."""
        results = self.model(frame, imgsz=self.imgsz, conf=self.conf_threshold,
                             verbose=False)[0]
        if results.boxes is None or len(results.boxes) == 0:
            return np.zeros((0, 4)), np.zeros(0)

        boxes = results.boxes.xyxy.cpu().numpy()
        clss = results.boxes.cls.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()

        # Keep only vehicle classes
        mask = np.isin(clss.astype(int), list(VEHICLE_CLASSES))
        return boxes[mask], confs[mask]

    def track_video(self, video_path: str, max_frames: int | None = None,
                    skip: int = 1) -> list[Tracklet]:
        """Run detection + IoU tracking on a full video.

        Args:
            video_path: path to video file.
            max_frames: stop after this many frames (None = entire video).
            skip: process every N-th frame.

        Returns:
            List of completed tracklets.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if max_frames is None:
            max_frames = total_frames

        actual_frames = (max_frames + skip - 1) // skip
        print(f"  [detect] {video_path.split('/')[-1]}: {w}x{h} @ {fps:.1f}fps, "
              f"{total_frames} frames (processing every {skip}th, ~{actual_frames})")

        active: dict[int, Tracklet] = {}
        finished: list[Tracklet] = []
        next_id = 0
        last_seen: dict[int, int] = {}
        frame_idx = 0

        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % skip != 0:
                frame_idx += 1
                continue

            boxes, confs = self.detect_frame(frame)

            if active:
                # Build IoU matrix: current dets vs active tracks (last box)
                active_ids = list(active)
                prev_boxes = np.vstack([active[tid].boxes[-1] for tid in active_ids])
                iou = _iou_matrix(boxes, prev_boxes)
                matches = _greedy_match(iou, self.iou_threshold)

                matched_det: set[int] = set()
                matched_trk: set[int] = set()
                for di, ti in matches:
                    tid = active_ids[ti]
                    active[tid].add(frame_idx, boxes[di], confs[di])
                    last_seen[tid] = frame_idx
                    matched_det.add(di)
                    matched_trk.add(ti)

                # Age unmatched tracks
                for ti, tid in enumerate(active_ids):
                    if ti not in matched_trk:
                        if frame_idx - last_seen[tid] > self.max_lost_frames:
                            finished.append(active.pop(tid))
                            del last_seen[tid]
            else:
                matched_det = set()

            # Start new tracks for unmatched detections
            for di in range(len(boxes)):
                if di in matched_det:
                    continue
                t = Tracklet(track_id=next_id)
                next_id += 1
                t.add(frame_idx, boxes[di], confs[di])
                active[t.track_id] = t
                last_seen[t.track_id] = frame_idx

            frame_idx += 1

        # Flush remaining tracks
        finished.extend(active.values())
        cap.release()

        # Filter short tracklets
        kept = [t for t in finished if t.num_frames >= self.min_tracklet_frames]

        # Stats
        n_vehicles = len(kept)
        n_dropped = len(finished) - len(kept)
        if kept:
            avg_len = np.mean([t.num_frames for t in kept])
            print(f"  [detect]   {n_vehicles} tracklets kept, {n_dropped} dropped "
                  f"(avg {avg_len:.0f} frames)")
        else:
            print(f"  [detect]   0 tracklets! ({n_dropped} dropped)")

        return kept


def _plate_to_vehicle_box(plate_box: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """Expand a plate bounding box to approximate the vehicle bounding box.

    The plate sits at the bottom-center of the vehicle's rear.  We expand
    the plate box symmetrically in width and upward in height to cover the
    approximate vehicle extent.  Clamps to image bounds.

    Args:
        plate_box: [x1, y1, x2, y2] plate coordinates (px).
        img_w, img_h: frame dimensions for clamping.

    Returns:
        [x1, y1, x2, y2] approximate vehicle box.
    """
    px1, py1, px2, py2 = plate_box
    pw = px2 - px1
    ph = py2 - py1
    pcx = (px1 + px2) / 2
    pcy = (py1 + py2) / 2

    # Expand width
    vw = pw * PLATE_EXPAND_W
    vx1 = pcx - vw / 2
    vx2 = pcx + vw / 2

    # Expand height upward (plate is near bottom of vehicle)
    vh = ph * PLATE_EXPAND_H
    vy2 = pcy + ph * 0.2          # slight downward from plate center
    vy1 = vy2 - vh                 # expand upward

    # Clamp
    vx1 = max(0, vx1)
    vy1 = max(0, vy1)
    vx2 = min(img_w, vx2)
    vy2 = min(img_h, vy2)

    return np.array([vx1, vy1, vx2, vy2], dtype=float)


class PlateVehicleDetector:
    """Detect vehicles via plate detection.

    Each license plate → one vehicle.  Plate boxes are expanded to approximate
    vehicle bounding boxes for IoU tracking.  This is more reliable than the
    generic COCO detector because:
      - Plates are high-contrast, always present on vehicles
      - 81% mAP@50 on CCTV footage (fine-tuned)
      - Plate count = vehicle count (no missed vehicles from class confusion)
    """

    def __init__(
        self,
        model_path: str | Path = str(DEFAULT_PLATE_DETECTOR),
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        min_tracklet_frames: int = 5,
        max_lost_frames: int = 8,
        imgsz: int = 1280,
        expand_plates: bool = True,
        device: str | None = None,
    ):
        self.model = YOLO(str(model_path))
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.min_tracklet_frames = min_tracklet_frames
        self.max_lost_frames = max_lost_frames
        self.imgsz = imgsz
        self.expand_plates = expand_plates
        self.device = device

    def detect_frame(
        self, frame: np.ndarray, expand: bool | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run plate detection on one frame.

        Returns:
            plate_boxes:  (N, 4) xyxy plate bounding boxes.
            vehicle_boxes: (N, 4) xyxy expanded vehicle bounding boxes.
            confs:        (N,) confidence scores.
        """
        expand = expand if expand is not None else self.expand_plates
        h, w = frame.shape[:2]

        results = self.model(
            frame, imgsz=self.imgsz, conf=self.conf_threshold,
            verbose=False, device=self.device,
        )[0]

        if results.boxes is None or len(results.boxes) == 0:
            z = np.zeros((0, 4))
            return z, z, np.zeros(0)

        plate_boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()

        if expand:
            vehicle_boxes = np.array(
                [_plate_to_vehicle_box(pb, w, h) for pb in plate_boxes]
            )
        else:
            vehicle_boxes = plate_boxes.copy()

        return plate_boxes, vehicle_boxes, confs

    def count_vehicles(self, frame: np.ndarray) -> int:
        """Quick vehicle count from a single frame (no tracking)."""
        _, _, confs = self.detect_frame(frame)
        return len(confs)

    def track_video(
        self,
        video_path: str,
        max_frames: int | None = None,
        skip: int = 1,
        return_plates: bool = False,
        tracker_config: str = "bytetrack.yaml",
    ) -> list[Tracklet]:
        """Run plate detection + ByteTrack on a full video.

        Uses ultralytics built-in ByteTrack (Kalman filter motion model,
        two-stage IoU matching) instead of a hand-rolled greedy matcher.

        Args:
            video_path: path to video file.
            max_frames: stop after this many frames (None = entire video).
            skip: process every N-th frame.
            return_plates: if True, tracklet boxes are plate coords (not vehicle).
            tracker_config: ultralytics tracker config (bytetrack.yaml, botsort.yaml).

        Returns:
            List of completed tracklets.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if max_frames is not None:
            total_frames = min(total_frames, max_frames)
        actual_frames = (total_frames + skip - 1) // skip
        print(f"  [plate-detect] {Path(video_path).name}: {w}x{h} @ {fps:.1f}fps, "
              f"{total_frames} frames (every {skip}th, ~{actual_frames})")

        # Collect all track data from model.track()
        track_data: dict[int, dict[int, tuple[np.ndarray, float]]] = {}  # tid -> {frame: (box, conf)}
        total_plates = 0

        frame_counter = 0
        for result in self.model.track(
            source=video_path,
            imgsz=self.imgsz,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            tracker=tracker_config,
            stream=True,
            save=False,
            verbose=False,
            device=self.device,
        ):
            frame_counter += 1
            if max_frames and frame_counter > max_frames:
                break
            frame_idx = frame_counter
            if result.boxes is None or result.boxes.id is None:
                continue

            ids = result.boxes.id.cpu().numpy().astype(int)
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            total_plates += len(ids)

            # Convert xyxy to xywh (matching Tracklet format)
            for tid, box_xyxy, conf in zip(ids, boxes_xyxy, confs):
                x1, y1, x2, y2 = box_xyxy
                box_xywh = np.array([x1, y1, x2 - x1, y2 - y1])
                if tid not in track_data:
                    track_data[tid] = {}
                track_data[tid][frame_idx] = (box_xywh, float(conf))

        # Convert to Tracklet objects
        tracklets: list[Tracklet] = []
        for tid in sorted(track_data.keys()):
            frames_dict = track_data[tid]
            if len(frames_dict) < self.min_tracklet_frames:
                continue
            t = Tracklet(track_id=int(tid))
            for f in sorted(frames_dict.keys()):
                box, conf = frames_dict[f]
                t.add(f, box, conf)
            tracklets.append(t)

        n_vehicles = len(tracklets)
        n_dropped = len(track_data) - n_vehicles
        if tracklets:
            avg_len = np.mean([t.num_frames for t in tracklets])
            print(f"  [plate-detect]   {n_vehicles} tracklets, {total_plates} total plates, "
                  f"{n_dropped} dropped (avg {avg_len:.0f} frames)")
        else:
            print(f"  [plate-detect]   0 tracklets! ({total_plates} plates, {n_dropped} dropped)")

        return tracklets
