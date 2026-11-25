"""Per-camera and per-track quality gate with hysteresis, plus tracker re-entry grace.

Provenance: `ReentryGrace` adapted from the re-ID grace pattern in `products/fall_detection/fall_monitor.py`
and `perception/lib/reentry_grace.py`. Owned independently by `products/concealment_detection`.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from .types import (
    DetectorSettings,
    L_HIP,
    L_SHOULDER,
    PoseFrame,
    QualityStatus,
    R_HIP,
    R_SHOULDER,
    TrackObservation,
)


def compute_track_quality_metrics(
    obs: TrackObservation,
    settings: DetectorSettings,
) -> tuple[float, float, float, int]:
    """Return `(core_conf, box_height_px, torso_px, confident_joints)` for `obs`."""
    kps = obs.kps
    _, y1, _, y2 = obs.box
    box_height_px = max(0.0, float(y2 - y1))

    if len(kps) > R_HIP:
        ls = kps[L_SHOULDER]
        rs = kps[R_SHOULDER]
        lh = kps[L_HIP]
        rh = kps[R_HIP]
        core_conf = 0.25 * (ls[2] + rs[2] + lh[2] + rh[2])
        mid_sh_x = 0.5 * (ls[0] + rs[0])
        mid_sh_y = 0.5 * (ls[1] + rs[1])
        mid_hp_x = 0.5 * (lh[0] + rh[0])
        mid_hp_y = 0.5 * (lh[1] + rh[1])
        torso_px = math.hypot(mid_hp_x - mid_sh_x, mid_hp_y - mid_sh_y)
    else:
        core_conf = 0.0
        torso_px = 0.0

    confident_joints = sum(
        1 for kp in kps if float(kp[2]) >= settings.keypoint_conf_threshold
    )
    return (core_conf, box_height_px, torso_px, confident_joints)


def _approx_torso_from_obs(obs: TrackObservation) -> tuple[tuple[float, float], float]:
    kps = obs.kps
    x1, y1, x2, y2 = obs.box
    if len(kps) > R_HIP:
        ls, rs, lh, rh = kps[L_SHOULDER], kps[R_SHOULDER], kps[L_HIP], kps[R_HIP]
        mid_sh_x = 0.5 * (ls[0] + rs[0])
        mid_sh_y = 0.5 * (ls[1] + rs[1])
        mid_hp_x = 0.5 * (lh[0] + rh[0])
        mid_hp_y = 0.5 * (lh[1] + rh[1])
        center = (0.5 * (mid_sh_x + mid_hp_x), 0.5 * (mid_sh_y + mid_hp_y))
        length = max(math.hypot(mid_hp_x - mid_sh_x, mid_hp_y - mid_sh_y), 1.0)
        return center, length
    center = (0.5 * (x1 + x2), 0.5 * (y1 + y2))
    length = max(0.35 * (y2 - y1), 1.0)
    return center, length


@dataclass
class _GraceRecord:
    canonical_id: int
    last_pts: float
    lost_pts: float | None
    torso_center: tuple[float, float]
    torso_length: float
    raw_track_id: int


class ReentryGrace:
    """Bridges short tracker ID switches under shelf/body occlusion."""

    def __init__(self, settings: DetectorSettings) -> None:
        self.settings = settings
        self._records: dict[int, _GraceRecord] = {}
        self._alias_to_canonical: dict[int, int] = {}
        self._next_synthetic_id = -2

    def expire(self, pts: float, protected_ids: set[int] | None = None) -> None:
        protected_ids = protected_ids or set()
        self._records = {
            cid: rec for cid, rec in self._records.items()
            if cid in protected_ids or 0.0 <= pts - rec.last_pts <= self.settings.reid_max_gap_s
        }
        self._alias_to_canonical = {
            raw: cid for raw, cid in self._alias_to_canonical.items() if cid in self._records
        }

    def observe_active(
        self,
        canonical_id: int,
        obs: TrackObservation,
        pts: float,
        torso_center: tuple[float, float],
        torso_length: float,
    ) -> None:
        # Keep only the current raw owner, not an unbounded chain of stale aliases.
        self._alias_to_canonical = {
            raw: cid for raw, cid in self._alias_to_canonical.items()
            if cid != canonical_id or raw == obs.track_id
        }
        self._alias_to_canonical[obs.track_id] = canonical_id
        self._records[canonical_id] = _GraceRecord(
            canonical_id=canonical_id,
            last_pts=pts,
            lost_pts=None,
            torso_center=torso_center,
            torso_length=max(torso_length, 1.0),
            raw_track_id=obs.track_id,
        )

    def mark_lost(self, canonical_id: int, pts: float) -> None:
        rec = self._records.get(canonical_id)
        if rec is not None and rec.lost_pts is None:
            rec.lost_pts = pts

    def resolve_id(
        self,
        raw_track_id: int,
        obs: TrackObservation,
        pts: float,
        active_raw_ids: set[int],
    ) -> int:
        # Reserve canonical identities for their actual current raw owners, regardless
        # of iteration order. Comparing a canonical ID to raw IDs merges bystanders.
        reserved = {
            cid for raw, cid in self._alias_to_canonical.items()
            if raw != raw_track_id and raw in active_raw_ids
        }
        reserved.update(cid for cid, rec in self._records.items()
                        if rec.last_pts == pts and rec.raw_track_id != raw_track_id)
        if raw_track_id in self._alias_to_canonical:
            cid = self._alias_to_canonical[raw_track_id]
            rec = self._records.get(cid)
            if rec is not None and cid not in reserved and pts >= rec.last_pts:
                return cid

        center, length = _approx_torso_from_obs(obs)
        best_cid: int | None = None
        best_norm_dist = float("inf")

        for cid, rec in list(self._records.items()):
            gap = pts - rec.last_pts
            if gap < 0.0 or gap > self.settings.reid_max_gap_s:
                continue
            if cid in reserved or (cid in active_raw_ids and cid != raw_track_id):
                continue
            denom = max(rec.torso_length, length, 25.0)
            dist_torsos = math.hypot(
                center[0] - rec.torso_center[0],
                center[1] - rec.torso_center[1],
            ) / denom
            ratio = max(length, 1.0) / max(rec.torso_length, 1.0)
            if (
                dist_torsos <= self.settings.reid_max_dist_torsos
                and 0.5 <= ratio <= 2.0
                and dist_torsos < best_norm_dist
            ):
                best_norm_dist = dist_torsos
                best_cid = cid

        if best_cid is not None:
            self._alias_to_canonical[raw_track_id] = best_cid
            return best_cid

        cid = raw_track_id
        if cid in reserved:
            while self._next_synthetic_id in self._records or self._next_synthetic_id in active_raw_ids:
                self._next_synthetic_id -= 1
            cid = self._next_synthetic_id
            self._next_synthetic_id -= 1
        self._alias_to_canonical[raw_track_id] = cid
        return cid


@dataclass
class _TrackQualityState:
    first_pts: float
    last_seen_pts: float
    state: str = "UNKNOWN"
    degraded_since_pts: float | None = None
    recover_since_pts: float | None = None
    last_status: QualityStatus = QualityStatus(state="UNKNOWN")


class QualityGate:
    """Camera-level (`LIVE` / `STALE` / `NO_POSE`) and per-track (`UNKNOWN` / `OK` / `DEGRADED` / `INSUFFICIENT` / `GONE`) gate."""

    def __init__(self, settings: DetectorSettings) -> None:
        self.settings = settings
        self._last_observed_pts: float | None = None
        self._last_pts: float | None = None
        self._shape: tuple[int, int] | None = None
        self._tracks: dict[int, _TrackQualityState] = {}

    def invalidate(self) -> None:
        self._last_observed_pts = None
        self._tracks.clear()

    def forget_track(self, track_id: int) -> None:
        self._tracks.pop(track_id, None)

    def evaluate_camera(self, frame: PoseFrame) -> tuple[str, str]:
        if not math.isfinite(frame.pts):
            return ("STALE", "invalid_frame_clock")
        previous_pts, previous_shape = self._last_pts, self._shape
        self._last_pts = frame.pts
        w, h = frame.shape
        if w <= 0 or h <= 0:
            self.invalidate()
            return ("STALE", "invalid_frame_shape")
        self._shape = frame.shape
        if previous_pts is not None and frame.pts < previous_pts - 1e-3:
            self.invalidate()
            return ("STALE", "camera_clock_reset")
        if previous_shape is not None and frame.shape != previous_shape:
            self.invalidate()
            return ("STALE", "frame_shape_changed")
        if previous_pts is not None and frame.pts - previous_pts > self.settings.camera_stale_s:
            self.invalidate()
            return ("STALE", "camera_frame_gap")
        if frame.observed:
            self._last_observed_pts = frame.pts
            return ("LIVE", "")
        if self._last_observed_pts is None:
            return ("NO_POSE", "pose_not_observed")
        if frame.pts - self._last_observed_pts > self.settings.camera_stale_s:
            return ("NO_POSE", "pose_inference_stale")
        return ("LIVE", "")

    def _check_geometry_and_conf(
        self,
        obs: TrackObservation,
        for_recovery: bool,
    ) -> tuple[bool, str, float, float, int]:
        core_conf, box_h, torso_px, n_joints = compute_track_quality_metrics(
            obs, self.settings
        )
        min_conf = (
            self.settings.min_core_conf_exit
            if for_recovery
            else self.settings.min_core_conf_enter
        )
        reasons: list[str] = []
        if box_h < self.settings.min_box_height_px:
            reasons.append("small_person_box_height")
        if torso_px < self.settings.min_torso_px:
            reasons.append("small_torso_scale")
        if n_joints < self.settings.min_keypoints:
            reasons.append("insufficient_visible_keypoints")
        if core_conf < min_conf:
            reasons.append("low_core_keypoint_confidence")
        return (len(reasons) == 0, ",".join(reasons), core_conf, box_h, n_joints)

    def evaluate_track(
        self,
        track_id: int,
        obs: TrackObservation,
        pts: float,
    ) -> QualityStatus:
        st = self._tracks.get(track_id)
        if st is None:
            st = _TrackQualityState(first_pts=pts, last_seen_pts=pts, state="UNKNOWN")
            self._tracks[track_id] = st
        st.last_seen_pts = pts

        need_recovery_thresh = st.state in ("DEGRADED", "INSUFFICIENT")
        passes, reason, core_conf, box_h, n_joints = self._check_geometry_and_conf(
            obs, for_recovery=need_recovery_thresh
        )

        age = pts - st.first_pts
        if st.state == "UNKNOWN":
            if not passes:
                st.state = "INSUFFICIENT"
                st.degraded_since_pts = pts
                st.recover_since_pts = None
            elif age >= self.settings.min_track_s:
                st.state = "OK"
                reason = ""
            else:
                reason = "track_warmup"
        elif st.state == "OK":
            if not passes:
                st.state = "DEGRADED"
                st.degraded_since_pts = pts
                st.recover_since_pts = None
            else:
                reason = ""
        elif st.state == "DEGRADED":
            if passes:
                if st.recover_since_pts is None:
                    st.recover_since_pts = pts
                if (pts - st.recover_since_pts) >= self.settings.recover_s:
                    st.state = "OK"
                    st.degraded_since_pts = None
                    st.recover_since_pts = None
                    reason = ""
            else:
                st.recover_since_pts = None
                if st.degraded_since_pts is None:
                    st.degraded_since_pts = pts
                if (pts - st.degraded_since_pts) >= self.settings.degrade_max_s:
                    st.state = "INSUFFICIENT"
        elif st.state in ("INSUFFICIENT", "GONE"):
            if passes:
                if st.recover_since_pts is None:
                    st.recover_since_pts = pts
                if (pts - st.recover_since_pts) >= self.settings.recover_s:
                    st.state = "OK"
                    st.degraded_since_pts = None
                    st.recover_since_pts = None
                    reason = ""
            else:
                st.recover_since_pts = None

        status = QualityStatus(
            state=st.state,
            reason=reason if st.state in ("DEGRADED", "INSUFFICIENT") else "",
            core_conf=core_conf,
            box_height_px=box_h,
            confident_joints=n_joints,
        )
        st.last_status = status
        return status

    def step_missing_tracks(
        self,
        seen_track_ids: set[int],
        pts: float,
        frame_observed: bool,
    ) -> list[tuple[int, QualityStatus]]:
        if not frame_observed:
            return []
        updates: list[tuple[int, QualityStatus]] = []
        for tid, st in list(self._tracks.items()):
            if tid in seen_track_ids or st.state == "GONE":
                continue
            gap = pts - st.last_seen_pts
            if gap >= self.settings.lost_s:
                st.state = "GONE"
                status = QualityStatus(
                    state="GONE",
                    reason="track_lost",
                    core_conf=st.last_status.core_conf,
                    box_height_px=st.last_status.box_height_px,
                    confident_joints=st.last_status.confident_joints,
                )
                st.last_status = status
                updates.append((tid, status))
        return updates
