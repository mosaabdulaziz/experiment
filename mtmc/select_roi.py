#!/usr/bin/env python3
"""Interactive polygon ROI selector for each camera.

Shows a sample frame from each camera and lets you click points to define
a polygon that covers the road area. Detections outside this polygon will
be ignored during plate extraction.

Usage:
    .venv/bin/python -m mtmc.select_roi --video-dir /media/mmtayee/6A79-CA62
    .venv/bin/python -m mtmc.select_roi --video-dir /media/mmtayee/6A79-CA62 --camera D18
    .venv/bin/python -m mtmc.select_roi --video-dir /media/mmtayee/6A79-CA62 --camera ch23 --frame-num 300

Output: mtmc/rois.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_MODELS_DIR = Path(__file__).resolve().parent / "models"
_ROI_FILE = Path(__file__).resolve().parent / "rois.json"

# Windows to exclude by default (user previously chose these)
_DEFAULT_EXCLUDE = {"D01", "D05", "D09", "D11", "D20", "D32"}


def _extract_camera_id(filename: str) -> str | None:
    """Extract camera ID from video filename.  e.g. D18_20260726000000.mp4 -> D18"""
    m = re.match(r"^(D\d+|ch\d+)", filename)
    return m.group(1) if m else None


def _pick_representative_video(video_dir: Path, camera_id: str) -> Path | None:
    """Pick a representative video for this camera (prefer a longer one)."""
    candidates = []
    for f in video_dir.glob(f"{camera_id}_*.mp4"):
        size = f.stat().st_size
        candidates.append((size, f))
    if not candidates:
        return None
    # Pick the largest file (likely the longest/most representative)
    candidates.sort(key=lambda x: -x[0])
    return candidates[0][1]


def _grab_frame(video_path: Path, frame_num: int = 150) -> np.ndarray | None:
    """Grab a specific frame from a video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_num = min(frame_num, total - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


# ---------------------------------------------------------------------------
# Interactive polygon selector (OpenCV)
# ---------------------------------------------------------------------------
class PolygonSelector:
    """Click to add points, right-click/Enter to close, 'u' to undo, 'q' to skip."""

    def __init__(self, image: np.ndarray, camera_id: str):
        self.image = image.copy()
        self.display = image.copy()
        self.camera_id = camera_id
        self.points: list[tuple[int, int]] = []
        self.done = False
        self.cancelled = False

        # Scale image to fit screen if too large
        h, w = image.shape[:2]
        self.scale = 1.0
        max_w, max_h = 1600, 900
        if w > max_w or h > max_h:
            self.scale = min(max_w / w, max_h / h)
            new_w, new_h = int(w * self.scale), int(h * self.scale)
            self.display = cv2.resize(self.display, (new_w, new_h))

        self.window_name = f"[{camera_id}] Click to draw polygon | Right-click/Enter=close | u=undo | q=skip | s=save empty"

    def _scale_point(self, x: int, y: int) -> tuple[int, int]:
        """Convert display coordinates back to original image coordinates."""
        return int(x / self.scale), int(y / self.scale)

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            orig_x, orig_y = self._scale_point(x, y)
            self.points.append((orig_x, orig_y))
            self._redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(self.points) >= 3:
                self.done = True

    def _redraw(self):
        self.display = self.image.copy()
        if self.scale != 1.0:
            h, w = self.image.shape[:2]
            self.display = cv2.resize(self.display, (int(w * self.scale), int(h * self.scale)))

        # Draw polygon
        if len(self.points) >= 2:
            scaled = [(int(p[0] * self.scale), int(p[1] * self.scale)) for p in self.points]
            pts = np.array(scaled, dtype=np.int32)
            overlay = self.display.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 0, 50))
            cv2.addWeighted(overlay, 0.3, self.display, 0.7, 0, self.display)
            cv2.polylines(self.display, [pts], True, (0, 255, 0), 2)

        # Draw points
        for i, (px, py) in enumerate(self.points):
            sx, sy = int(px * self.scale), int(py * self.scale)
            cv2.circle(self.display, (sx, sy), 5, (0, 0, 255), -1)
            cv2.putText(self.display, str(i), (sx + 8, sy - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Instructions
        h = self.display.shape[0]
        info_lines = [
            f"Camera: {self.camera_id}  |  Points: {len(self.points)}",
            "Left-click = add point  |  Right-click = close polygon",
            "'u' = undo  |  'q' = skip  |  's' = save empty (no ROI)",
        ]
        for i, line in enumerate(info_lines):
            y = h - 60 + i * 20
            cv2.putText(self.display, line, (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow(self.window_name, self.display)

    def select(self) -> list[tuple[int, int]] | None:
        """Run interactive selection. Returns polygon points or None if skipped."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        self._redraw()

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                self.cancelled = True
                break
            elif key == ord("u") and self.points:
                self.points.pop()
                self._redraw()
            elif key == ord("s"):
                # Save empty (no ROI = accept all detections)
                self.points = []
                self.done = True
                break
            elif key in (13, 10):  # Enter
                if len(self.points) >= 3:
                    self.done = True
                    break
            elif key == 27:  # Escape
                self.cancelled = True
                break
            self._redraw()

        cv2.destroyWindow(self.window_name)
        if self.cancelled:
            return None
        return self.points


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video-dir", required=True, help="Directory containing camera videos")
    p.add_argument("--camera", default=None, help="Process only this camera ID (e.g. D18)")
    p.add_argument("--frame-num", type=int, default=150, help="Frame number to use as sample (default: 150)")
    p.add_argument("--roi-file", default=str(_ROI_FILE), help="Output JSON file for ROI polygons")
    p.add_argument("--exclude", nargs="*", default=list(_DEFAULT_EXCLUDE),
                   help="Camera IDs to skip (default: D01 D05 D09 D11 D20 D32)")
    args = p.parse_args()

    video_dir = Path(args.video_dir)
    if not video_dir.exists():
        print(f"Error: Video directory not found: {video_dir}")
        sys.exit(1)

    exclude = set(args.exclude)
    roi_file = Path(args.roi_file)

    # Load existing ROIs
    existing_rois = {}
    if roi_file.exists():
        existing_rois = json.loads(roi_file.read_text())
        print(f"Loaded {len(existing_rois)} existing ROIs from {roi_file}")

    # Discover cameras
    cameras: dict[str, list[Path]] = {}
    for f in sorted(video_dir.glob("*.mp4")):
        cam = _extract_camera_id(f.name)
        if cam and cam not in exclude:
            cameras.setdefault(cam, []).append(f)

    if args.camera:
        if args.camera not in cameras:
            print(f"Error: Camera '{args.camera}' not found (available: {sorted(cameras.keys())})")
            sys.exit(1)
        cameras = {args.camera: cameras[args.camera]}

    print(f"\nCameras to configure: {sorted(cameras.keys())}")
    print(f"Excluded: {sorted(exclude)}")
    print(f"Output: {roi_file}\n")

    for cam_id in sorted(cameras.keys()):
        vid = _pick_representative_video(video_dir, cam_id)
        if vid is None:
            print(f"  [{cam_id}] No videos found, skipping")
            continue

        frame = _grab_frame(vid, args.frame_num)
        if frame is None:
            print(f"  [{cam_id}] Cannot read frame from {vid.name}, skipping")
            continue

        # Check if ROI already exists
        if cam_id in existing_rois:
            pts = existing_rois[cam_id]["polygon"]
            n = len(pts) if pts else 0
            print(f"  [{cam_id}] Existing ROI: {n} points (from {vid.name})")
            resp = input("    Redraw? [y/N] > ").strip().lower()
            if resp != "y":
                continue

        print(f"  [{cam_id}] Drawing ROI on {vid.name} (frame {args.frame_num})...")
        selector = PolygonSelector(frame, cam_id)
        polygon = selector.select()

        if polygon is None:
            print(f"    Skipped")
            continue

        if len(polygon) < 3:
            print(f"    Too few points ({len(polygon)}), saving empty ROI")
            polygon = []

        existing_rois[cam_id] = {
            "polygon": polygon,
            "source_video": vid.name,
            "frame_num": args.frame_num,
            "image_size": {"w": frame.shape[1], "h": frame.shape[0]},
        }

        if polygon:
            print(f"    Saved {len(polygon)} points")
        else:
            print(f"    Saved empty ROI (all detections accepted)")

        # Save after each camera (in case of crash)
        roi_file.write_text(json.dumps(existing_rois, indent=2, ensure_ascii=False))
        print(f"    -> {roi_file}")

    print(f"\nDone. ROIs saved to {roi_file}")
    print(json.dumps({k: len(v["polygon"]) for k, v in existing_rois.items()}, indent=2))


if __name__ == "__main__":
    main()
