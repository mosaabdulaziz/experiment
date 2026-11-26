"""Configuration dataclass for VEMA v2 (Derived Continuous Polar-EMA Tracker)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from .config import VEMAConfig


@dataclass
class VEMA2Config:
    """Minimal configuration: shared detector/lifecycle contract + 1 physical time constant."""

    # --- Shared Detector & Lifecycle Contract (identical to ByteTrack / OC-SORT) ---
    tau_high: float = 0.60
    tau_low: float = 0.15
    iou_thresh_high: float = 0.25
    iou_thresh_low: float = 0.15
    probation_frames: int = 2
    max_lost_frames: int = 30
    early_term_lost: int = 2
    min_iou_diff_new: float = 0.70
    check_class_match: bool = True
    emit_shadow_tracks: bool = False
    ref_fps: float = 30.0
    dt_aware: bool = True

    # --- Single Intrinsic Physical Parameter ---
    tau_mem: float = 0.15  # Kinematic memory time constant (seconds)

    # --- Optional Track Spawn Confidence Floor (0 = fall back to tau_high) ---
    new_track_thresh: float = 0.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.tau_low < self.tau_high <= 1.0):
            raise ValueError(
                f"Expected 0 <= tau_low < tau_high <= 1, got tau_low={self.tau_low}, tau_high={self.tau_high}"
            )
        if not (0.0 <= self.new_track_thresh <= 1.0):
            raise ValueError(f"Expected 0 <= new_track_thresh <= 1, got {self.new_track_thresh}")
        if self.tau_mem <= 0.0:
            raise ValueError(f"tau_mem must be > 0, got {self.tau_mem}")
        if self.ref_fps <= 0.0:
            raise ValueError(f"ref_fps must be > 0, got {self.ref_fps}")
        if self.probation_frames < 1:
            raise ValueError(f"probation_frames must be >= 1, got {self.probation_frames}")
        if self.max_lost_frames < 1:
            raise ValueError(f"max_lost_frames must be >= 1, got {self.max_lost_frames}")

    @property
    def effective_new_track_thresh(self) -> float:
        """Minimum confidence required to spawn a new track (at least tau_high)."""
        return float(max(self.tau_high, self.new_track_thresh))

    @classmethod
    def from_vema_config(cls, cfg: VEMAConfig) -> "VEMA2Config":
        """Construct VEMA2Config from the shared contract fields of VEMAConfig."""
        return cls(
            tau_high=cfg.tau_high,
            tau_low=cfg.tau_low,
            iou_thresh_high=cfg.iou_thresh_high,
            iou_thresh_low=cfg.iou_thresh_low,
            probation_frames=cfg.probation_frames,
            max_lost_frames=cfg.max_lost_frames,
            early_term_lost=cfg.early_term_lost,
            min_iou_diff_new=cfg.min_iou_diff_new,
            check_class_match=cfg.check_class_match,
            emit_shadow_tracks=cfg.emit_shadow_tracks,
            ref_fps=cfg.ref_fps,
            dt_aware=cfg.dt_aware,
            new_track_thresh=float(getattr(cfg, "new_track_thresh", 0.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "VEMA2Config":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)
