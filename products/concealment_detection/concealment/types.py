"""Domain data structures and constants for pose-first concealment detection."""
from __future__ import annotations

from dataclasses import dataclass

COCO_KEYPOINTS = 17

# A short skipped inference is normal; longer observation gaps break evidence continuity.
MAX_OBSERVATION_GAP_S = 0.5

NOSE = 0
L_EYE = 1
R_EYE = 2
L_EAR = 3
R_EAR = 4
L_SHOULDER = 5
R_SHOULDER = 6
L_ELBOW = 7
R_ELBOW = 8
L_WRIST = 9
R_WRIST = 10
L_HIP = 11
R_HIP = 12
L_KNEE = 13
R_KNEE = 14
L_ANKLE = 15
R_ANKLE = 16

TIERS = ("high", "medium", "low")
TERMINAL_STAGES = ("RETURNED", "HIDDEN", "PARTIAL", "RETURNED_FAST", "HIDDEN_SPATIAL", "EXIT")
COOLDOWN_SCOPES = ("track", "track_zone")
TARGET_KINDS = ("clothing", "personal_bag")
REVIEW_STATUSES = ("candidate", "acknowledged", "concern_noted", "dismissed")


@dataclass(frozen=True)
class TrackObservation:
    track_id: int
    box: tuple[float, float, float, float]
    kps: tuple[tuple[float, float, float], ...]
    has_bag: bool = False
    bag_box: tuple[float, float, float, float] | None = None
    cart_boxes: tuple[tuple[float, float, float, float], ...] = ()


@dataclass(frozen=True)
class PoseFrame:
    camera: str
    frame_num: int
    pts: float
    shape: tuple[int, int]
    tracks: tuple[TrackObservation, ...]
    observed: bool = True


@dataclass(frozen=True)
class ZonePolygon:
    id: str
    points: tuple[tuple[float, float], ...]
    facing: tuple[float, float] | None = None
    kind: str = "shelf_front"


@dataclass(frozen=True)
class CameraZones:
    camera: str
    zone_free: bool = False
    zones: tuple[ZonePolygon, ...] = ()


@dataclass(frozen=True)
class DetectorSettings:
    min_track_s: float = 0.4
    min_keypoints: int = 6
    keypoint_conf_threshold: float = 0.30
    min_box_height_px: float = 48.0
    min_torso_px: float = 25.0
    min_core_conf_enter: float = 0.30
    min_core_conf_exit: float = 0.35
    degrade_max_s: float = 1.0
    recover_s: float = 0.4
    lost_s: float = 1.5
    camera_stale_s: float = 3.0
    reid_max_gap_s: float = 1.5
    reid_max_dist_torsos: float = 1.2
    dwell_min_s: float = 2.0
    short_track_grace_s: float = 1.2
    reach_excursion_torsos: float = 0.35
    reach_window_s: float = 2.5
    inward_min_torsos: float = 0.12
    body_zone_radius_torsos: float = 0.55
    bag_zone_radius_torsos: float = 0.50
    max_conceal_elbow_deg: float = 135.0
    min_hold_s: float = 0.35
    hidden_s: float = 0.45
    hidden_conf_drop_fraction: float = 0.45
    baseline_window_s: float = 2.0
    cooldown_s: float = 8.0
    emit_low_tier: bool = True
    # Opt-in FSM knobs; every default reproduces the legacy behaviour exactly.
    sequence_max_gap_s: float = MAX_OBSERVATION_GAP_S
    insufficient_grace_s: float = 0.0
    zone_persist_across_shelves: bool = False
    zone_exit_grace_s: float = 0.0
    reach_expiry_s: float = 0.0
    partial_min_near_body_s: float = 0.0
    reach_seeds_dwell: bool = False
    reach_seed_max_speed_torsos_s: float = 0.6
    at_body_requires_flex: bool = True
    hold_flicker_grace_s: float = 0.0
    fast_dropin_min_hold_s: float = 0.0
    spatial_hidden_s: float = 0.0
    exit_min_hidden_frac: float = 0.0
    hand_transfer: bool = False
    hand_transfer_meet_torsos: float = 0.35
    cooldown_scope: str = "track"


def validate_detector_settings(s: DetectorSettings) -> None:
    """Reject knob combinations that would break a stage invariant."""
    if not MAX_OBSERVATION_GAP_S <= s.sequence_max_gap_s < s.lost_s:
        raise ValueError("sequence_max_gap_s must be in [0.5, lost_s)")
    if s.sequence_max_gap_s > s.reid_max_gap_s:
        raise ValueError("sequence_max_gap_s must not exceed reid_max_gap_s")
    # Shorter than min_hold_s, so neither can stand in for a full hold.
    for name in ("hold_flicker_grace_s", "fast_dropin_min_hold_s"):
        value = getattr(s, name)
        if value < 0 or (value > 0 and value >= s.min_hold_s):
            raise ValueError(f"{name} must be 0 or below min_hold_s")
    # Long enough that a normal hold can still end as RETURNED first.
    floor = max(s.hidden_s, 2.0 * s.min_hold_s)
    if s.spatial_hidden_s < 0 or (s.spatial_hidden_s > 0 and s.spatial_hidden_s < floor - 1e-9):
        raise ValueError("spatial_hidden_s must be 0 or at least max(hidden_s, 2*min_hold_s)")
    if not 0.0 <= s.exit_min_hidden_frac <= 1.0:
        raise ValueError("exit_min_hidden_frac must be in [0, 1]")
    if s.hand_transfer_meet_torsos <= 0:
        raise ValueError("hand_transfer_meet_torsos must be positive")
    if s.cooldown_scope not in COOLDOWN_SCOPES:
        raise ValueError(f"cooldown_scope must be one of {COOLDOWN_SCOPES}")


@dataclass(frozen=True)
class QualityStatus:
    state: str
    reason: str = ""
    core_conf: float = 0.0
    box_height_px: float = 0.0
    confident_joints: int = 0


@dataclass(frozen=True)
class TrackFeatures:
    track_id: int
    pts: float
    torso_center: tuple[float, float]
    torso_length: float
    ground_point_norm: tuple[float, float]
    zone_id: str | None
    in_shelf_zone: bool
    left_elbow_deg: float
    right_elbow_deg: float
    min_elbow_deg: float
    active_hand: str
    wrist_to_body_torsos: float
    body_zone_name: str
    target_kind: str
    elbow_flexed_or_medial: bool
    in_at_body_pose: bool
    wrist_excursion_torsos: float
    reach_detected: bool
    active_wrist_conf: float
    wrist_conf_baseline: float
    wrist_hidden_now: bool
    wrist_geometry_valid: bool = True
    # Raw anatomical membership, independent of the finite inward-motion lookback.
    at_body_geometry: bool | None = None
    # Unknown speed (too little history) never passes a speed filter.
    torso_speed_torsos_s: float = float("inf")
    other_hand_visible: bool = False
    other_hand_in_at_body_pose: bool = False
    other_hand_at_body_geometry: bool = False
    other_hand_reach_detected: bool = False
    hands_met_recent: bool = False


@dataclass(frozen=True)
class StageTransition:
    stage: str
    entered_pts: float
    detail: str = ""


@dataclass(frozen=True)
class CandidateEvent:
    camera: str
    track_id: int
    pts: float
    tier: str
    score: float
    zone_id: str
    target_kind: str
    body_zone_name: str
    terminal_stage: str
    hold_duration_s: float
    dwell_duration_s: float
    stage_timeline: tuple[StageTransition, ...]
    clip_window: tuple[float, float]
    keyframes_pts: tuple[float, ...]


@dataclass(frozen=True)
class AssessmentRecord:
    camera: str
    track_id: int
    pts: float
    state: str
    reason: str


@dataclass(frozen=True)
class MonitorStepResult:
    camera: str
    pts: float
    camera_ok: bool
    candidates: tuple[CandidateEvent, ...]
    assessments: tuple[AssessmentRecord, ...]
