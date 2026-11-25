"""VEMATracker: 3-stage multi-object association pipeline."""

from __future__ import annotations

import numpy as np

from .assignment import linear_assignment
from .config import VEMAConfig
from .costs import iou_matrix, relocation_cost_matrix
from .track import VEMATrack


class VEMATracker:
    """Three-regime velocity-EMA multi-object tracker."""

    def __init__(self, config: VEMAConfig | None = None) -> None:
        self.cfg = config or VEMAConfig()
        self.tracks: list[VEMATrack] = []
        self._next_id = 1
        self.frame_idx = 0

    def reset(self, fps: float | None = None) -> None:
        if fps is not None and fps > 0:
            self.cfg = self.cfg.with_overrides(ref_fps=float(fps))
        self.tracks = []
        self._next_id = 1
        self.frame_idx = 0

    def _spawn_track(self, bbox_xyxy: np.ndarray, conf: float, cls_id: int) -> VEMATrack:
        trk = VEMATrack(
            track_id=self._next_id,
            bbox_xyxy=bbox_xyxy,
            conf=conf,
            cls_id=cls_id,
            cfg=self.cfg,
        )
        self._next_id += 1
        return trk

    def _iou_cost(
        self,
        tracks: list[VEMATrack],
        dets_xyxy: np.ndarray,
        dets_cls: np.ndarray,
    ) -> np.ndarray:
        if not tracks or len(dets_xyxy) == 0:
            return np.zeros((len(tracks), len(dets_xyxy)), dtype=np.float64)
        trk_boxes = np.stack([t.get_bbox(buffered=False) for t in tracks], axis=0)
        cost = 1.0 - iou_matrix(trk_boxes, dets_xyxy)
        if self.cfg.check_class_match:
            trk_cls = np.array([t.cls_id for t in tracks], dtype=np.int64)[:, None]
            cost = np.where(trk_cls != dets_cls[None, :], np.inf, cost)
        return cost

    def update(
        self,
        detections_xyxy: np.ndarray,
        scores: np.ndarray,
        cls_ids: np.ndarray | None = None,
        dt: float | None = None,
    ) -> list[tuple[int, np.ndarray, float, int, bool]]:
        """Process one frame and return active [(track_id, bbox_xyxy, conf, cls_id, is_shadow)]."""
        self.frame_idx += 1
        cfg = self.cfg

        dets = np.asarray(detections_xyxy, dtype=np.float64).reshape(-1, 4)
        confs = np.asarray(scores, dtype=np.float64).reshape(-1)
        if cls_ids is None:
            clss = np.zeros(len(dets), dtype=np.int64)
        else:
            clss = np.asarray(cls_ids, dtype=np.int64).reshape(-1)

        if cfg.dt_aware and dt is not None and dt > 0.0:
            dt_steps = float(np.clip(float(dt) * cfg.ref_fps, 1e-4, 30.0))
        else:
            dt_steps = 1.0

        # 1. Predict all existing tracks forward by dt_steps
        for trk in self.tracks:
            trk.predict(dt_steps)

        active_tracks = [t for t in self.tracks if t.state in ("tentative", "confirmed")]
        lost_tracks = [t for t in self.tracks if t.state == "lost"]

        high_idx = np.where(confs >= cfg.tau_high)[0]
        low_idx = np.where((confs >= cfg.tau_low) & (confs < cfg.tau_high))[0]

        matched_active: set[int] = set()
        matched_lost: set[int] = set()
        matched_high_local: set[int] = set()

        # 2. Stage 1: Active (confirmed + tentative) <-> High-confidence detections
        if active_tracks and len(high_idx) > 0:
            cost_s1 = self._iou_cost(active_tracks, dets[high_idx], clss[high_idx])
            m1, u_trk1, u_det1 = linear_assignment(cost_s1, 1.0 - cfg.iou_thresh_high)
            for ti, di in m1:
                g_di = int(high_idx[di])
                active_tracks[ti].update(dets[g_di], confs[g_di], clss[g_di])
                matched_active.add(ti)
                matched_high_local.add(di)
        else:
            u_trk1 = list(range(len(active_tracks)))
            u_det1 = list(range(len(high_idx)))

        # 3. Stage 2: Remaining CONFIRMED active tracks <-> Low-confidence detections
        rem_confirmed_idx = [ti for ti in u_trk1 if active_tracks[ti].state == "confirmed"]
        if rem_confirmed_idx and len(low_idx) > 0:
            rem_confirmed = [active_tracks[ti] for ti in rem_confirmed_idx]
            cost_s2 = self._iou_cost(rem_confirmed, dets[low_idx], clss[low_idx])
            m2, _, _ = linear_assignment(cost_s2, 1.0 - cfg.iou_thresh_low)
            for local_ti, di in m2:
                ti = rem_confirmed_idx[local_ti]
                g_di = int(low_idx[di])
                active_tracks[ti].update(dets[g_di], confs[g_di], clss[g_di])
                matched_active.add(ti)

        # 4. Stage 3: Lost tracks <-> Remaining High-confidence detections (Anisotropic Relocation)
        rem_high_idx = high_idx[u_det1] if len(u_det1) > 0 else np.empty(0, dtype=np.int64)
        unmatched_high_global: list[int] = [int(i) for i in rem_high_idx]
        if lost_tracks and len(rem_high_idx) > 0:
            cost_s3 = relocation_cost_matrix(
                lost_tracks, dets[rem_high_idx], clss[rem_high_idx], cfg
            )
            m3, _, u_det3 = linear_assignment(cost_s3, cfg.relocation_cost_thresh)
            for li, di in m3:
                g_di = int(rem_high_idx[di])
                lost_tracks[li].update(dets[g_di], confs[g_di], clss[g_di])
                matched_lost.add(li)
            unmatched_high_global = [int(rem_high_idx[di]) for di in u_det3]

        # 5. Transition unmatched tracks to Regime III & prune dead tracks
        for ti, trk in enumerate(active_tracks):
            if ti not in matched_active:
                trk.mark_missed(dt_steps)
        for li, trk in enumerate(lost_tracks):
            if li not in matched_lost:
                trk.mark_missed(dt_steps)

        kept_tracks: list[VEMATrack] = []
        for trk in self.tracks:
            if trk.state == "tentative" and trk.k_lost > cfg.early_term_lost:
                continue
            if trk.k_lost > cfg.max_lost_frames:
                continue
            kept_tracks.append(trk)
        self.tracks = kept_tracks

        # 6. Spawn new tracks for unmatched high-conf detections (suppress duplicates)
        if unmatched_high_global:
            existing_boxes = (
                np.stack([t.get_bbox(buffered=False) for t in self.tracks if t.state != "lost"], axis=0)
                if any(t.state != "lost" for t in self.tracks)
                else np.empty((0, 4), dtype=np.float64)
            )
            for g_di in unmatched_high_global:
                det_box = dets[g_di]
                if len(existing_boxes) > 0:
                    ious = iou_matrix(det_box[None, :], existing_boxes)[0]
                    if float(np.max(ious)) > cfg.min_iou_diff_new:
                        continue
                trk = self._spawn_track(det_box, float(confs[g_di]), int(clss[g_di]))
                if self.frame_idx == 1:
                    trk.state = "confirmed"
                self.tracks.append(trk)

        # 7. Collect outputs
        outputs: list[tuple[int, np.ndarray, float, int, bool]] = []
        for trk in self.tracks:
            if trk.state == "confirmed":
                outputs.append((trk.id, trk.get_bbox(buffered=False), trk.conf, trk.cls_id, False))
            elif cfg.emit_shadow_tracks and trk.state == "lost":
                outputs.append((trk.id, trk.get_bbox(buffered=False), trk.conf, trk.cls_id, True))
        return outputs
