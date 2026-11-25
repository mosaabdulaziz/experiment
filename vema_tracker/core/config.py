"""Configuration dataclass for VEMATracker with YAML serialization and ablation toggles."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass
class VEMAConfig:
    """All tunable hyperparameters and ablation flags for VEMATracker."""

    # --- Confidence regime thresholds ---
    tau_high: float = 0.60
    tau_low: float = 0.15

    # --- Kinematic EMA & damping ---
    alpha: float = 0.80           # EMA decay at ref_fps (higher = smoother velocity & size)
    gamma: float = 0.98           # Kinetic velocity damping per reference frame in Regime III
    eps_cal: float = 0.05         # Bounded velocity nudge ratio in Regime II
    v_var_alpha: float = 0.85     # EMA smoothing for online velocity variance sigma_v^2
    scale_alpha: float = 0.90     # EMA decay for scale-velocity v_d_ema = [dw/dt, dh/dt]
    scale_gamma: float = 0.0      # Zero scale-velocity during Regime III coasting (ByteTrack vh=0 parity)
    scale_max_rate: float = 0.05  # Max relative scale rate per ref-frame (rejects noisy box jumps)
    ref_fps: float = 30.0         # Reference frame rate for dt-invariant scaling
    dt_aware: bool = True         # Normalize velocity & decay by actual dt (seconds)

    # --- Regime II truncation / cropping sanity gate ---
    area_ratio_min: float = 0.75  # Reject degraded obs velocity if area_obs / area_ema < this
    cos_min: float = 0.0          # Reject degraded obs velocity if cos(v_obs, v_ema) < this
    min_speed_ratio: float = 0.05 # Stationary guard: speed < min_speed_ratio * min(w, h) per ref frame

    # --- Association & Stage 3 relocation cost ---
    iou_thresh_high: float = 0.25         # Stage 1 (active <-> high-conf) min IoU
    iou_thresh_low: float = 0.15          # Stage 2 (active <-> low-conf) min IoU
    relocation_cost_thresh: float = 1.60  # Stage 3 (lost <-> high-conf) max total cost
    biou_rate: float = 0.05               # Buffered IoU box expansion per lost ref-frame
    biou_max_expansion: float = 2.0       # Cap total BIoU expansion multiplier
    lambda_biou: float = 1.0              # Weight on (1 - BIoU)
    lambda_pos: float = 0.35              # Weight on anisotropic C_pos
    lambda_dir: float = 0.25              # Weight on observation-centric direction cosine
    sigma_par_scale: float = 1.5          # Longitudinal uncertainty multiplier
    sigma_perp_scale: float = 0.6         # Lateral uncertainty multiplier
    base_pos_sigma_ratio: float = 0.25    # Base position sigma as fraction of min(w, h)

    # --- Track lifecycle ---
    probation_frames: int = 2             # Hits before a tentative track becomes confirmed
    max_lost_frames: int = 30             # Max lost ref-frames before track termination
    early_term_lost: int = 2              # Max lost frames while still tentative
    min_iou_diff_new: float = 0.70        # Suppress new track if IoU with active track > this
    check_class_match: bool = True        # Gate association on matching class id
    emit_shadow_tracks: bool = False      # Emit coasting tracks (useful for live preview, False for MOT eval)

    # --- Ablation toggles (all True in full VEMA) ---
    enable_truncation_gate: bool = True
    enable_size_lock: bool = True
    enable_velocity_blending: bool = True
    enable_anisotropic_cost: bool = True
    enable_biou: bool = True
    enable_dir_cost: bool = True
    enable_scale_velocity: bool = True

    # --- VEMA-3D Pinhole Ground-Plane Parameters ---
    cam_height: float = 2.0               # Nominal camera height above ground (meters)
    cam_pitch_deg: float = 8.0            # Nominal downward camera pitch (degrees)
    min_ray_y: float = 0.02               # Horizon safety clamp on downward ground ray y
    max_depth: float = 80.0               # Maximum ground-plane depth clamp (meters)
    pos_3d_process_std: float = 0.22      # Ground-plane process noise std per 30Hz step (meters)
    lambda_mmd: float = 0.45              # Stage 1/2 3D Mapped Mahalanobis blend weight
    mmd_gate: float = 12.0                # Stage 1/2 MMD normalization scale (chi-squared 2-DOF)
    auto_pitch: bool = True               # Adapt pitch on frame 1 if boxes sit above nominal horizon

    def __post_init__(self) -> None:
        if not (0.0 <= self.tau_low < self.tau_high <= 1.0):
            raise ValueError(
                f"Expected 0 <= tau_low < tau_high <= 1, got tau_low={self.tau_low}, tau_high={self.tau_high}"
            )
        if not (0.0 <= self.alpha < 1.0):
            raise ValueError(f"alpha must be in [0, 1), got {self.alpha}")
        if not (0.0 < self.gamma <= 1.0):
            raise ValueError(f"gamma must be in (0, 1], got {self.gamma}")
        if self.ref_fps <= 0:
            raise ValueError(f"ref_fps must be > 0, got {self.ref_fps}")
        if self.probation_frames < 1:
            raise ValueError(f"probation_frames must be >= 1, got {self.probation_frames}")
        if self.max_lost_frames < 1:
            raise ValueError(f"max_lost_frames must be >= 1, got {self.max_lost_frames}")

    def with_overrides(self, **kwargs: Any) -> "VEMAConfig":
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "VEMAConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)
