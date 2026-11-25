"""Shared fixtures and anatomical COCO-17 skeleton builders for concealment detection tests."""
from __future__ import annotations

import sys
from pathlib import Path

PRODUCT_DIR = Path(__file__).resolve().parents[1]
if str(PRODUCT_DIR) not in sys.path:
    sys.path.insert(0, str(PRODUCT_DIR))

from concealment.types import PoseFrame, TrackObservation  # noqa: E402


def make_skeleton_obs(
    track_id: int = 1,
    *,
    center_x: float = 960.0,
    shoulder_y: float = 400.0,
    hip_y: float = 540.0,
    ankle_y: float = 780.0,
    shoulder_half_w: float = 45.0,
    hip_half_w: float = 35.0,
    right_elbow_offset: tuple[float, float] = (50.0, 70.0),
    right_wrist_offset: tuple[float, float] = (52.0, 145.0),
    left_elbow_offset: tuple[float, float] = (-50.0, 70.0),
    left_wrist_offset: tuple[float, float] = (-52.0, 145.0),
    conf: float = 0.90,
    right_wrist_conf: float | None = None,
    left_wrist_conf: float | None = None,
    ankle_conf: float | None = None,
    box: tuple[float, float, float, float] | None = None,
    has_bag: bool = False,
    bag_box: tuple[float, float, float, float] | None = None,
) -> TrackObservation:
    """Build a realistic 17-keypoint TrackObservation in capture pixels (default 1920x1080).

    By default, the person stands upright with arms hanging straight at their sides:
    - shoulders at (center_x -/+ 45, 400)
    - elbows at (center_x -/+ 50, 470)
    - wrists at (center_x -/+ 52, 545) -- right next to the hips vertically, but with a straight ~177 deg elbow!
    """
    rw_c = conf if right_wrist_conf is None else right_wrist_conf
    lw_c = conf if left_wrist_conf is None else left_wrist_conf
    ak_c = conf if ankle_conf is None else ankle_conf
    knee_y = (hip_y + ankle_y) / 2.0

    kps = [
        (center_x, shoulder_y - 50.0, conf),                              # 0: nose
        (center_x - 10.0, shoulder_y - 58.0, conf),                       # 1: left_eye
        (center_x + 10.0, shoulder_y - 58.0, conf),                       # 2: right_eye
        (center_x - 20.0, shoulder_y - 52.0, conf),                       # 3: left_ear
        (center_x + 20.0, shoulder_y - 52.0, conf),                       # 4: right_ear
        (center_x - shoulder_half_w, shoulder_y, conf),                   # 5: left_shoulder
        (center_x + shoulder_half_w, shoulder_y, conf),                   # 6: right_shoulder
        (center_x + left_elbow_offset[0], shoulder_y + left_elbow_offset[1], conf),   # 7: left_elbow
        (center_x + right_elbow_offset[0], shoulder_y + right_elbow_offset[1], conf), # 8: right_elbow
        (center_x + left_wrist_offset[0], shoulder_y + left_wrist_offset[1], lw_c),   # 9: left_wrist
        (center_x + right_wrist_offset[0], shoulder_y + right_wrist_offset[1], rw_c), # 10: right_wrist
        (center_x - hip_half_w, hip_y, conf),                             # 11: left_hip
        (center_x + hip_half_w, hip_y, conf),                             # 12: right_hip
        (center_x - hip_half_w, knee_y, conf),                            # 13: left_knee
        (center_x + hip_half_w, knee_y, conf),                            # 14: right_knee
        (center_x - hip_half_w, ankle_y, ak_c),                           # 15: left_ankle
        (center_x + hip_half_w, ankle_y, ak_c),                           # 16: right_ankle
    ]
    if box is None:
        box = (center_x - 80.0, shoulder_y - 75.0, center_x + 80.0, ankle_y + 15.0)
    return TrackObservation(
        track_id=track_id,
        box=box,
        kps=tuple(kps),
        has_bag=has_bag,
        bag_box=bag_box,
    )


def make_frame(
    pts: float,
    tracks: list[TrackObservation] | tuple[TrackObservation, ...] = (),
    *,
    camera: str = "cam-1",
    frame_num: int = 0,
    shape: tuple[int, int] = (1920, 1080),
    observed: bool = True,
) -> PoseFrame:
    return PoseFrame(
        camera=camera,
        frame_num=frame_num,
        pts=pts,
        shape=shape,
        tracks=tuple(tracks),
        observed=observed,
    )
