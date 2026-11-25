"""SOTA tracker factory and unified interfaces."""

from __future__ import annotations

from typing import Any

from vema_tracker.core.config import VEMAConfig

from .base import BaseTracker, Detection, TrackOutput
from .bytetrack import ByteTrackBaseline
from .ocsort import OCSORTBaseline
from .sort_kalman import SORTTracker
from .vema_adapter import VEMABaselineAdapter
from .vema3d_adapter import VEMA3DBaselineAdapter

TRACKER_NAMES = ("vema3d", "vema", "bytetrack", "ocsort", "sort")


def get_tracker(
    name: str,
    config: VEMAConfig | None = None,
    **kwargs: Any,
) -> BaseTracker:
    """Instantiate a tracker by name ('vema3d', 'vema', 'bytetrack', 'ocsort', 'sort')."""
    key = name.strip().lower()
    cfg = config or VEMAConfig()

    if key == "vema3d":
        return VEMA3DBaselineAdapter(config=cfg, **kwargs)
    if key == "vema":
        return VEMABaselineAdapter(config=cfg)
    if key == "bytetrack":
        return ByteTrackBaseline(
            high_thresh=cfg.tau_high,
            low_thresh=cfg.tau_low,
            new_track_thresh=cfg.tau_high,
            track_buffer=cfg.max_lost_frames,
            check_class_match=cfg.check_class_match,
            dt_aware=cfg.dt_aware,
            ref_fps=cfg.ref_fps,
            **kwargs,
        )
    if key == "ocsort":
        return OCSORTBaseline(
            det_thresh=cfg.tau_high,
            low_thresh=cfg.tau_low,
            max_age=cfg.max_lost_frames,
            min_hits=cfg.probation_frames,
            check_class_match=cfg.check_class_match,
            dt_aware=cfg.dt_aware,
            ref_fps=cfg.ref_fps,
            **kwargs,
        )
    if key == "sort":
        return SORTTracker(
            conf_thresh=cfg.tau_high,
            iou_thresh=cfg.iou_thresh_high,
            max_age=cfg.max_lost_frames,
            min_hits=cfg.probation_frames,
            check_class_match=cfg.check_class_match,
            dt_aware=cfg.dt_aware,
            ref_fps=cfg.ref_fps,
            **kwargs,
        )
    raise ValueError(f"Unknown tracker {name!r}. Choose from {TRACKER_NAMES}.")


__all__ = [
    "BaseTracker",
    "Detection",
    "TrackOutput",
    "VEMA3DBaselineAdapter",
    "VEMABaselineAdapter",
    "ByteTrackBaseline",
    "OCSORTBaseline",
    "SORTTracker",
    "TRACKER_NAMES",
    "get_tracker",
]
