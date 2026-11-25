"""Data loading for the AICity22 Track1 MTMC dataset.

Loads the challenge-provided per-camera detections, ground truth, camera
start-time offsets, and frame counts. All formats are MOTChallenge-style.

GT format:      frame, id, x, y, w, h, 1, -1, -1, -1
det format:     frame, -1, x, y, w, h, conf, -1, -1, -1
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import DATASET_ROOT


@dataclass
class CameraInfo:
    cam_id: str          # e.g. "c001"
    frames: int          # total frames in the video
    start_offset_s: float  # wall-clock offset vs the scene's first camera
    det_path: Path
    gt_path: Path
    video_path: Path


def load_camera_infos(scene: str, cameras: list[str], det_file: str) -> dict[str, CameraInfo]:
    """Build per-camera info for a scene, reading cam_framenum and cam_timestamp."""
    root = DATASET_ROOT
    framenum_path = root / "cam_framenum" / f"{scene}.txt"
    timestamp_path = root / "cam_timestamp" / f"{scene}.txt"

    framenums: dict[str, int] = {}
    offsets: dict[str, float] = {}
    for p, out in ((framenum_path, framenums), (timestamp_path, offsets)):
        for line in p.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2:
                out[parts[0]] = float(parts[1])

    infos: dict[str, CameraInfo] = {}
    for cam in cameras:
        base = root / "train" / scene / cam
        infos[cam] = CameraInfo(
            cam_id=cam,
            frames=int(framenums.get(cam, 0)),
            start_offset_s=offsets.get(cam, 0.0),
            det_path=base / "det" / det_file,
            gt_path=base / "gt" / "gt.txt",
            video_path=base / "vdo.avi",
        )
    return infos


def load_detections(det_path: Path, conf_threshold: float) -> dict[int, np.ndarray]:
    """Load a det file -> {frame: Nx4 array of [x, y, w, h]} (filtered by conf)."""
    out: dict[int, np.ndarray] = {}
    if not det_path.exists():
        return out
    for line in det_path.read_text().splitlines():
        p = line.replace(",", " ").split()
        if len(p) < 7:
            continue
        frame = int(float(p[0]))
        x, y, w, h, conf = float(p[2]), float(p[3]), float(p[4]), float(p[5]), float(p[6])
        if conf < conf_threshold or w <= 0 or h <= 0:
            continue
        out.setdefault(frame, []).append([x, y, w, h])
    return {f: np.asarray(b, dtype=float).reshape(-1, 4) for f, b in out.items()}


def load_ground_truth(gt_path: Path) -> dict[int, np.ndarray]:
    """Load a gt.txt -> {frame: Nx5 array of [id, x, y, w, h]}."""
    out: dict[int, list] = {}
    if not gt_path.exists():
        return {}
    for line in gt_path.read_text().splitlines():
        p = line.replace(",", " ").split()
        if len(p) < 6:
            continue
        frame = int(float(p[0]))
        tid = int(float(p[1]))
        x, y, w, h = (float(p[i]) for i in (2, 3, 4, 5))
        out.setdefault(frame, []).append([tid, x, y, w, h])
    return {f: np.asarray(v, dtype=float) for f, v in out.items()}
