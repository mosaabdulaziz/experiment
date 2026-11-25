"""Per-track ordered concealment state machine and camera-level ConcealmentMonitor."""
from __future__ import annotations

from collections import Counter, deque
from collections.abc import Callable
from dataclasses import replace
import math

from .features import extract_track_features, torso_geometry
from .quality import QualityGate, ReentryGrace
from .types import (
    AssessmentRecord,
    CameraZones,
    CandidateEvent,
    DetectorSettings,
    MAX_OBSERVATION_GAP_S,
    MonitorStepResult,
    PoseFrame,
    QualityStatus,
    StageTransition,
    TIERS,
    TrackFeatures,
    TrackObservation,
    validate_detector_settings,
)


_TIER_SCORES = {"high": 0.85, "medium": 0.72, "low": 0.56}
_TERMINAL_TIER_CAPS = {"PARTIAL": "low", "RETURNED_FAST": "low",
                       "HIDDEN_SPATIAL": "medium", "EXIT": "medium"}


def _lower_tier(tier: str, cap: str) -> str:
    return TIERS[max(TIERS.index(tier), TIERS.index(cap))]


class TrackStateMachine:
    """Ordered per-track state machine.

    IDLE -> DWELL -> REACH -> AT_BODY -> (RETURNED | HIDDEN) -> COOLDOWN, and an expired
    REACH after full dwell may end as PARTIAL. Track loss alone never emits. Opt-in knobs
    add RETURNED_FAST, HIDDEN_SPATIAL and a hidden-gated EXIT (see DetectorSettings).
    """

    def __init__(
        self,
        track_id: int,
        camera: str,
        first_seen_pts: float,
        on_reset: Callable[[float, int, str, str], None] | None = None,
    ) -> None:
        self.track_id = track_id
        self._on_reset = on_reset
        self.camera = camera
        self.first_seen_pts = first_seen_pts
        self.stage: str = "IDLE"
        self.dwell_accum_s: float = 0.0
        self.had_full_dwell: bool = False
        self.short_track_grace_used: bool = False
        self.reach_pts: float | None = None
        self.at_body_entered_pts: float | None = None
        self.last_in_body_pts: float | None = None
        self.cooldown_until_pts: float = -1e9
        self._zone_cooldowns: dict[str, float] = {}
        self.last_pts: float | None = None
        self._last_observation_pts: float | None = None
        self.active_zone_id: str = "zone_free"
        self.active_target_kind: str = "clothing"
        self.active_body_zone: str = "waist"
        self.timeline: list[StageTransition] = []
        self.active_hand: str | None = None
        self._last_zone_id: str | None = None
        self.hold_accum_s = 0.0
        self.hidden_accum_s = 0.0
        self.body_run_s = 0.0
        self.near_body_s = 0.0
        self._previous_body = False
        self._previous_hidden = False
        self._previous_near_body = False
        self._tier_cap = "high"
        self._transferred = False
        self._bad_quality_since_pts: float | None = None
        self._zone_exit_since_pts: float | None = None

    def _reset_to_idle(self, pts: float, reason: str) -> None:
        if self._on_reset is not None:
            self._on_reset(pts, self.track_id, reason, self.stage)
        # Reset evidence, never the suppression deadline.
        self.stage = "COOLDOWN" if pts < self.cooldown_until_pts else "IDLE"
        self.had_full_dwell = False
        self.short_track_grace_used = False
        self.reach_pts = None
        self.at_body_entered_pts = None
        self.last_in_body_pts = None
        self.timeline.clear()
        self.active_hand = None
        self._last_zone_id = None
        self.dwell_accum_s = 0.0
        self.hold_accum_s = 0.0
        self.hidden_accum_s = 0.0
        self.body_run_s = 0.0
        self.near_body_s = 0.0
        self.last_pts = None
        self._last_observation_pts = None
        self._previous_body = False
        self._previous_hidden = False
        self._previous_near_body = False
        self._tier_cap = "high"
        self._transferred = False
        self._bad_quality_since_pts = None
        self._zone_exit_since_pts = None

    def _enter_reach(self, pts: float, feat: TrackFeatures) -> None:
        self.stage = "REACH"
        self.reach_pts = pts
        self.active_hand = feat.active_hand
        self.near_body_s = 0.0
        self._previous_near_body = False
        self.timeline.append(StageTransition("REACH", pts, f"hand={self.active_hand}"))

    def _enter_at_body(self, pts: float, feat: TrackFeatures, settings: DetectorSettings) -> None:
        self.stage = "AT_BODY"
        self.at_body_entered_pts = pts
        self.last_in_body_pts = pts
        self._previous_body = True
        self.body_run_s = 0.0
        self.active_target_kind = feat.target_kind
        self.active_body_zone = feat.body_zone_name
        if not settings.at_body_requires_flex and not feat.elbow_flexed_or_medial:
            self._tier_cap = _lower_tier(self._tier_cap, "medium")
        self.timeline.append(
            StageTransition(
                stage="AT_BODY",
                entered_pts=pts,
                detail=f"zone={feat.body_zone_name}",
            )
        )

    def _transfer_hand(self, pts: float) -> None:
        # The item changes hands; the switch frame itself earns no hold or hidden time.
        self.active_hand = "left" if self.active_hand == "right" else "right"
        self._transferred = True
        self._previous_body = self._previous_hidden = self._previous_near_body = False
        self.hidden_accum_s = 0.0
        self.timeline.append(StageTransition("TRANSFER", pts, f"hand={self.active_hand}"))

    def _can_transfer(self, settings: DetectorSettings) -> bool:
        return settings.hand_transfer and not self._transferred

    def _cooldown_deadline(self, zone: str | None, settings: DetectorSettings) -> float:
        if settings.cooldown_scope == "track_zone":
            return self._zone_cooldowns.get(zone if zone is not None else self.active_zone_id, -1e9)
        return self.cooldown_until_pts

    def _emit_candidate(
        self,
        pts: float,
        terminal_stage: str,
        hold_dur: float,
        settings: DetectorSettings,
    ) -> CandidateEvent | None:
        if terminal_stage == "PARTIAL":
            tier = "low"
        elif self.had_full_dwell:
            tier = "high"
        elif self.short_track_grace_used:
            tier = "medium"
        else:
            tier = "low"
        tier = _lower_tier(tier, self._tier_cap)
        tier = _lower_tier(tier, _TERMINAL_TIER_CAPS.get(terminal_stage, "high"))
        if tier == "low" and not settings.emit_low_tier:
            self._reset_to_idle(pts, "low_tier_suppressed")
            return None
        score = _TIER_SCORES[tier]

        self.timeline.append(
            StageTransition(
                stage=terminal_stage,
                entered_pts=pts,
                detail=f"target={self.active_target_kind}:{self.active_body_zone}",
            )
        )
        keyframes = tuple(t.entered_pts for t in self.timeline[-4:])
        cand = CandidateEvent(
            camera=self.camera,
            track_id=self.track_id,
            pts=pts,
            tier=tier,
            score=round(min(0.98, score + 0.05 * min(hold_dur, 1.5)), 3),
            zone_id=self.active_zone_id,
            target_kind=self.active_target_kind,
            body_zone_name=self.active_body_zone,
            terminal_stage=terminal_stage,
            hold_duration_s=round(hold_dur, 3),
            dwell_duration_s=round(self.dwell_accum_s, 3),
            stage_timeline=tuple(self.timeline),
            clip_window=(pts - 5.0, pts + 5.0),
            keyframes_pts=keyframes,
        )
        # Reported as COOLDOWN so an emission is not logged as a lost sequence.
        self.stage = "COOLDOWN"
        deadline = pts + settings.cooldown_s
        if settings.cooldown_scope == "track_zone":
            self._zone_cooldowns[self.active_zone_id] = deadline
            self.cooldown_until_pts = max(self.cooldown_until_pts, deadline)
        else:
            self.cooldown_until_pts = deadline
        self._reset_to_idle(pts, "emitted")
        return cand

    def step(
        self,
        pts: float,
        feat: TrackFeatures,
        quality: QualityStatus,
        settings: DetectorSettings,
    ) -> CandidateEvent | None:
        dt = pts - self.last_pts if self.last_pts is not None else 0.0
        observation_dt = (pts - self._last_observation_pts
                          if self._last_observation_pts is not None else 0.0)
        self._last_observation_pts = pts
        gap_limit = (settings.sequence_max_gap_s if self.stage in ("REACH", "AT_BODY")
                     else MAX_OBSERVATION_GAP_S)
        if observation_dt < 0.0 or observation_dt > gap_limit + 1e-6:
            self._reset_to_idle(pts, "gap")
            self._last_observation_pts = pts
            dt = 0.0
        elif observation_dt > MAX_OBSERVATION_GAP_S + 1e-6:
            # A bridged gap keeps the sequence but never credits unseen time.
            dt = 0.0
            self.body_run_s = 0.0
        if quality.state in ("INSUFFICIENT", "GONE"):
            if settings.insufficient_grace_s > 0 and self.stage in ("REACH", "AT_BODY"):
                if self._bad_quality_since_pts is None:
                    self._bad_quality_since_pts = pts
                if pts - self._bad_quality_since_pts <= settings.insufficient_grace_s + 1e-6:
                    # Freeze: no accumulator may advance across poor-quality frames.
                    self.last_pts = None
                    return None
            self._reset_to_idle(pts, "quality")
            return None
        if (self._bad_quality_since_pts is not None
                and pts - self._bad_quality_since_pts > settings.insufficient_grace_s + 1e-6):
            self._reset_to_idle(pts, "quality")
            return None
        if self._bad_quality_since_pts is not None:
            paused_s = pts - self._bad_quality_since_pts
            if self.reach_pts is not None:
                self.reach_pts += paused_s
            if self.at_body_entered_pts is not None:
                self.at_body_entered_pts += paused_s
            if self.last_in_body_pts is not None:
                self.last_in_body_pts += paused_s
            if self._zone_exit_since_pts is not None:
                self._zone_exit_since_pts += paused_s
        self._bad_quality_since_pts = None
        current_zone = (feat.zone_id or "zone_free") if feat.in_shelf_zone else None
        if pts < self._cooldown_deadline(current_zone, settings):
            self._reset_to_idle(pts, "cooldown")
            return None
        if self.stage == "COOLDOWN":
            self._reset_to_idle(pts, "cooldown")
            self._last_observation_pts = pts
            # Under zone-scoped cooldown another zone may still be suppressed.
            self.stage = "IDLE"
            dt = 0.0

        in_sequence = self.stage in ("REACH", "AT_BODY")
        if (current_zone is None and in_sequence and self._last_zone_id is not None
                and settings.zone_exit_grace_s > 0):
            if self._zone_exit_since_pts is None:
                self._zone_exit_since_pts = pts
            if pts - self._zone_exit_since_pts <= settings.zone_exit_grace_s + 1e-6:
                current_zone = self._last_zone_id
        else:
            self._zone_exit_since_pts = None
        if current_zone != self._last_zone_id:
            persist = (in_sequence and settings.zone_persist_across_shelves
                       and current_zone is not None and self._last_zone_id is not None)
            if not persist:
                self.dwell_accum_s = 0.0
                if self._last_zone_id is not None:
                    self._reset_to_idle(pts, "zone_change" if current_zone is not None else "zone_exit")
                    self._last_observation_pts = pts
                dt = 0.0
        self._last_zone_id = current_zone
        self.last_pts = pts
        if current_zone is not None:
            if not (settings.zone_persist_across_shelves and self.stage in ("REACH", "AT_BODY")):
                self.active_zone_id = current_zone
            if self.stage in ("IDLE", "DWELL"):
                self.dwell_accum_s += dt

        if quality.state != "OK" and self.stage not in ("REACH", "AT_BODY"):
            return None
        visible = (feat.active_wrist_conf >= settings.keypoint_conf_threshold
                   and not feat.wrist_hidden_now and feat.wrist_geometry_valid)
        if self.active_hand is not None and feat.active_hand != self.active_hand:
            self._previous_body = self._previous_hidden = False
            self.hidden_accum_s = 0.0
            return None

        if self.stage == "IDLE":
            if current_zone is not None and self.dwell_accum_s >= settings.dwell_min_s - 1e-6:
                self.stage = "DWELL"
                self.had_full_dwell = True
                self.timeline = [StageTransition("DWELL", pts, f"zone={self.active_zone_id}")]
            elif (current_zone is not None and visible and feat.reach_detected
                  and pts - self.first_seen_pts <= settings.short_track_grace_s):
                self.short_track_grace_used = True
                self._enter_reach(pts, feat)
            elif (settings.reach_seeds_dwell and current_zone is not None and visible
                  and feat.reach_detected
                  and feat.torso_speed_torsos_s <= settings.reach_seed_max_speed_torsos_s + 1e-6):
                # Without dwell or short-track grace the path tier is already low.
                self._enter_reach(pts, feat)
            else:
                return None

        if self.stage == "DWELL":
            if visible and feat.reach_detected:
                self._enter_reach(pts, feat)
            else:
                return None

        if self.stage == "REACH":
            if (self._can_transfer(settings) and feat.other_hand_visible
                    and feat.other_hand_in_at_body_pose
                    and (feat.other_hand_reach_detected or feat.hands_met_recent)):
                self._transfer_hand(pts)
                return None
            # Expiry wins over late contact. PARTIAL needs observed same-hand
            # geometry and full dwell, never disappearance alone.
            expiry_s = settings.reach_expiry_s or settings.reach_window_s
            if self.reach_pts is not None and pts - self.reach_pts > expiry_s + 1e-6:
                if (self.had_full_dwell and visible and quality.state == "OK"
                        and self.near_body_s >= settings.partial_min_near_body_s - 1e-6):
                    return self._emit_candidate(pts, "PARTIAL", 0.0, settings)
                self._reset_to_idle(pts, "reach_expired")
                return None
            if visible and feat.at_body_geometry:
                if self._previous_near_body:
                    self.near_body_s += dt
                self._previous_near_body = True
            else:
                self._previous_near_body = False
            if visible and feat.in_at_body_pose:
                self._enter_at_body(pts, feat, settings)
            return None

        if self.stage == "AT_BODY":
            if feat.wrist_hidden_now:
                if self._previous_hidden:
                    self.hidden_accum_s += dt
                else:
                    self.hidden_accum_s = 0.0
                self._previous_hidden = True
                self._previous_body = False
                if self.hidden_accum_s >= settings.hidden_s - 1e-6:
                    return self._emit_candidate(pts, "HIDDEN", self.hold_accum_s, settings)
                return None

            self.hidden_accum_s = 0.0
            self._previous_hidden = False
            if not visible:
                self._previous_body = False
                return None
            # Losing the old inward-history sample is not a physical exit.
            if feat.in_at_body_pose or feat.at_body_geometry:
                if self._previous_body:
                    self.hold_accum_s += dt
                    self.body_run_s += dt
                else:
                    self.body_run_s = 0.0
                self._previous_body = True
                self.last_in_body_pts = pts
                if settings.spatial_hidden_s > 0 and self.body_run_s >= settings.spatial_hidden_s - 1e-6:
                    return self._emit_candidate(pts, "HIDDEN_SPATIAL", self.hold_accum_s, settings)
                return None
            if self.hold_accum_s >= settings.min_hold_s - 1e-6:
                return self._emit_candidate(pts, "RETURNED", self.hold_accum_s, settings)
            if (self._can_transfer(settings) and feat.other_hand_visible
                    and feat.other_hand_at_body_geometry
                    and (feat.other_hand_reach_detected or feat.hands_met_recent)):
                self._transfer_hand(pts)
                return None
            if (settings.hold_flicker_grace_s > 0 and self.last_in_body_pts is not None
                    and pts - self.last_in_body_pts <= settings.hold_flicker_grace_s + 1e-6):
                self._previous_body = False
                return None
            if (settings.fast_dropin_min_hold_s > 0
                    and self.hold_accum_s >= settings.fast_dropin_min_hold_s - 1e-6):
                return self._emit_candidate(pts, "RETURNED_FAST", self.hold_accum_s, settings)
            self._reset_to_idle(pts, "brief_hold")
        return None

    def flush_on_track_end(
        self,
        pts: float,
        settings: DetectorSettings,
    ) -> CandidateEvent | None:
        """Called once per track loss with the last-sighting pts, before the monitor resets it.

        Track loss alone supplies no terminal hand evidence; EXIT needs the hand to have
        been confidence-drop hidden on the last sighting for long enough.
        """
        if (settings.exit_min_hidden_frac > 0 and self.stage == "AT_BODY" and self._previous_hidden
                and self.hidden_accum_s >= settings.exit_min_hidden_frac * settings.hidden_s - 1e-6):
            return self._emit_candidate(pts, "EXIT", self.hold_accum_s, settings)
        return None


class ConcealmentMonitor:
    """Orchestrates quality gating, re-entry grace, feature extraction, and per-track state machines for one camera."""

    def __init__(
        self,
        camera: str,
        settings: DetectorSettings,
        camera_zones: CameraZones | None = None,
    ) -> None:
        validate_detector_settings(settings)
        self.camera = camera
        self.settings = settings
        self.camera_zones = camera_zones or CameraZones(camera=camera, zone_free=True, zones=())
        self.quality_gate = QualityGate(settings)
        self.reentry_grace = ReentryGrace(settings)
        self._histories: dict[int, deque[tuple[float, TrackObservation, float]]] = {}
        self._machines: dict[int, TrackStateMachine] = {}
        self._last_quality_state: dict[int, str] = {}
        self._last_seen: dict[int, float] = {}
        self._last_pts: float | None = None
        # Diagnostics only: why evidence was discarded, keyed by (reason, stage lost).
        self.reset_counts: Counter[tuple[str, str]] = Counter()
        self.sequence_resets: list[tuple[float, int, str, str]] = []

    def _record_reset(self, pts: float, track_id: int, reason: str, stage: str) -> None:
        self.reset_counts[(reason, stage)] += 1
        if stage in ("REACH", "AT_BODY"):
            self.sequence_resets.append((pts, track_id, reason, stage))

    def _reset_track(self, cid: int, pts: float, reason: str) -> None:
        self._machines[cid]._reset_to_idle(pts, reason)
        self._histories.pop(cid, None)
        self.quality_gate.forget_track(cid)
        self._last_quality_state.pop(cid, None)

    def _expire(self, pts: float) -> list[CandidateEvent]:
        retention = max(self.settings.lost_s, self.settings.reid_max_gap_s)
        ended: list[CandidateEvent] = []
        for cid in list(self._machines):
            age = pts - self._last_seen[cid]
            limit = (self.settings.sequence_max_gap_s
                     if self._machines[cid].stage in ("REACH", "AT_BODY")
                     else MAX_OBSERVATION_GAP_S)
            if age > limit + 1e-6:
                # A history exists only until the first reset after the last sighting.
                if cid in self._histories:
                    cand = self._machines[cid].flush_on_track_end(self._last_seen[cid],
                                                                  self.settings)
                    if cand is not None:
                        ended.append(cand)
                self._reset_track(cid, pts, "expired_track")
            if age > retention and pts >= self._machines[cid].cooldown_until_pts:
                self._reset_track(cid, pts, "expired_track")
                del self._machines[cid]
                del self._last_seen[cid]
        protected = {cid for cid, sm in self._machines.items() if pts < sm.cooldown_until_pts}
        self.reentry_grace.expire(pts, protected)
        return ended

    def invalidate(self, pts: float) -> None:
        """Break observations after silence/malformed input; retain cooldown tombstones.

        Caller owns assessment logging. PTS is footage time, not wall-clock time.
        Clock rollback rebases the remaining cooldown instead of cancelling it.
        """
        if not math.isfinite(pts):
            pts = self._last_pts if self._last_pts is not None else 0.0
        if self._last_pts is not None and pts < self._last_pts:
            shift = pts - self._last_pts
            for cid, sm in self._machines.items():
                sm.cooldown_until_pts += shift
                for zone in sm._zone_cooldowns:
                    sm._zone_cooldowns[zone] += shift
                sm.first_seen_pts += shift
                self._last_seen[cid] += shift
            for rec in self.reentry_grace._records.values():
                rec.last_pts += shift
                if rec.lost_pts is not None:
                    rec.lost_pts += shift
        self._last_pts = pts
        for cid in self._machines:
            self._reset_track(cid, pts, "invalidate")
        self.quality_gate.invalidate()
        self._expire(pts)

    def update(self, frame: PoseFrame) -> MonitorStepResult:
        cam_state, cam_reason = self.quality_gate.evaluate_camera(frame)
        if cam_state != "LIVE":
            self.invalidate(frame.pts)
            return MonitorStepResult(
                camera=self.camera,
                pts=frame.pts,
                camera_ok=False,
                candidates=(),
                assessments=(
                    AssessmentRecord(
                        camera=self.camera,
                        track_id=-1,
                        pts=frame.pts,
                        state="insufficient_evidence",
                        reason=cam_reason or cam_state.lower(),
                    ),
                ),
            )

        self._last_pts = frame.pts
        ended = self._expire(frame.pts)
        if not frame.observed:
            if self.settings.spatial_hidden_s > 0:
                for sm in self._machines.values():
                    sm.body_run_s = 0.0
                    sm._previous_body = False
            return MonitorStepResult(
                camera=self.camera,
                pts=frame.pts,
                camera_ok=True,
                candidates=tuple(ended),
                assessments=(),
            )

        active_raw_ids = {t.track_id for t in frame.tracks}
        seen_canonical_ids: set[int] = set()
        candidates: list[CandidateEvent] = ended
        assessments: list[AssessmentRecord] = []
        max_hist_window = max(4.0, self.settings.reach_window_s, self.settings.baseline_window_s)

        for raw_obs in frame.tracks:
            cid = self.reentry_grace.resolve_id(
                raw_obs.track_id, raw_obs, frame.pts, active_raw_ids
            )
            if cid in seen_canonical_ids:
                continue
            seen_canonical_ids.add(cid)
            obs = raw_obs if cid == raw_obs.track_id else replace(raw_obs, track_id=cid)
            tc, tl, _ = torso_geometry(obs)
            self.reentry_grace.observe_active(cid, raw_obs, frame.pts, tc, tl)
            sm = self._machines.get(cid)
            if sm is None:
                sm = TrackStateMachine(cid, self.camera, frame.pts, on_reset=self._record_reset)
                self._machines[cid] = sm
            self._last_seen[cid] = frame.pts
            hist = self._histories.setdefault(cid, deque())
            if hist and frame.pts <= hist[-1][0]:
                continue

            q_status = self.quality_gate.evaluate_track(cid, obs, frame.pts)
            prev_q = self._last_quality_state.get(cid)
            self._last_quality_state[cid] = q_status.state
            if q_status.state == "INSUFFICIENT" and prev_q != "INSUFFICIENT":
                assessments.append(
                    AssessmentRecord(
                        camera=self.camera,
                        track_id=cid,
                        pts=frame.pts,
                        state="insufficient_evidence",
                        reason=q_status.reason or "insufficient_pose_quality",
                    )
                )

            while hist and (frame.pts - hist[0][0]) > max_hist_window:
                hist.popleft()
            hist.append((frame.pts, obs, tl))

            feat = extract_track_features(
                obs=obs,
                history=list(hist),
                camera_zones=self.camera_zones,
                shape=frame.shape,
                settings=self.settings,
                active_hand=sm.active_hand,
            )

            cand = sm.step(frame.pts, feat, q_status, self.settings)
            if cand is not None:
                candidates.append(cand)

        # Mark any missing tracks in ReentryGrace and step missing tracks in QualityGate
        for cid in list(self._machines.keys()):
            if cid not in seen_canonical_ids:
                self.reentry_grace.mark_lost(cid, frame.pts)
                # The sequence bridges a blip, but the interval the camera did not see the
                # track must not accrue as dwell, hold or hidden time when it returns.
                self._machines[cid].last_pts = None
                self._machines[cid].body_run_s = 0.0
                self._machines[cid]._previous_body = False

        missing_updates = self.quality_gate.step_missing_tracks(
            seen_track_ids=seen_canonical_ids,
            pts=frame.pts,
            frame_observed=frame.observed,
        )
        for cid, q_status in missing_updates:
            self._last_quality_state[cid] = q_status.state

        return MonitorStepResult(
            camera=self.camera,
            pts=frame.pts,
            camera_ok=True,
            candidates=tuple(candidates),
            assessments=tuple(assessments),
        )
