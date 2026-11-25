"""PolarEMATracker: 4-stage parity association with continuous confidence-weighted Polar-EMA."""

from __future__ import annotations

import numpy as np

from .assignment import linear_assignment
from .costs import iou_matrix
from .polar_config import VEMA2Config
from .polar_costs import iou_polar_cost_matrix, relocation_polar_cost_matrix
from .polar_track import PolarEMATrack


class PolarEMATracker:
    """Derived continuous Polar-EMA tracker with ByteTrack-parity 4-stage association."""

    def __init__(self, config: VEMA2Config | None = None) -> None:
        self.cfg = config or VEMA2Config()
        self.tracks: list[PolarEMATrack] = []
        self._next_id = 1
        self.frame_idx = 0
        self.t_sec = 0.0

    def reset(self, fps: float | None = None) -> None:
        if fps is not None and fps > 0:
            self.cfg = VEMA2Config(
                tau_high=self.cfg.tau_high,
                tau_low=self.cfg.tau_low,
                iou_thresh_high=self.cfg.iou_thresh_high,
                iou_thresh_low=self.cfg.iou_thresh_low,
                probation_frames=self.cfg.probation_frames,
                max_lost_frames=self.cfg.max_lost_frames,
                early_term_lost=self.cfg.early_term_lost,
                min_iou_diff_new=self.cfg.min_iou_diff_new,
                check_class_match=self.cfg.check_class_match,
                emit_shadow_tracks=self.cfg.emit_shadow_tracks,
                ref_fps=float(fps),
                dt_aware=self.cfg.dt_aware,
                tau_mem=self.cfg.tau_mem,
            )
        self.tracks = []
        self._next_id = 1
        self.frame_idx = 0
        self.t_sec = 0.0

    def _spawn_track(self, bbox_xyxy: np.ndarray, conf: float, cls_id: int) -> PolarEMATrack:
        trk = PolarEMATrack(
            track_id=self._next_id,
            bbox_xyxy=bbox_xyxy,
            conf=conf,
            cls_id=cls_id,
            t_sec=self.t_sec,
            cfg=self.cfg,
        )
        self._next_id += 1
        return trk

    @staticmethod
    def _max_other_ious(
        matched_det_boxes: np.ndarray,
        all_confirmed: list[PolarEMATrack],
        matched_conf_indices: list[int],
    ) -> list[float]:
        """Compute max IoU of each matched detection against other confirmed predicted boxes."""
        if not matched_conf_indices or len(all_confirmed) <= 1:
            return [0.0] * len(matched_conf_indices)
        conf_boxes = np.stack([t.get_bbox() for t in all_confirmed], axis=0)
        det_vs_conf = iou_matrix(matched_det_boxes, conf_boxes)
        out: list[float] = []
        for row_k, ti in enumerate(matched_conf_indices):
            row = det_vs_conf[row_k]
            row[ti] = 0.0
            out.append(float(np.max(row)))
        return out

    def update(
        self,
        detections_xyxy: np.ndarray,
        scores: np.ndarray,
        cls_ids: np.ndarray | None = None,
        dt: float | None = None,
    ) -> list[tuple[int, np.ndarray, float, int, bool]]:
        """Process one frame and return [(track_id, bbox_xyxy, conf, cls_id, is_shadow)]."""
        cfg = self.cfg
        self.frame_idx += 1

        if cfg.dt_aware and dt is not None and dt > 0.0:
            dt_sec = float(np.clip(float(dt), 1e-4, 30.0 / cfg.ref_fps))
        else:
            dt_sec = 1.0 / cfg.ref_fps
        self.t_sec += dt_sec

        dets = np.asarray(detections_xyxy, dtype=np.float64).reshape(-1, 4)
        confs = np.asarray(scores, dtype=np.float64).reshape(-1)
        if cls_ids is None:
            clss = np.zeros(len(dets), dtype=np.int64)
        else:
            clss = np.asarray(cls_ids, dtype=np.int64).reshape(-1)

        high_idx = np.where(confs >= cfg.tau_high)[0]
        low_idx = np.where((confs >= cfg.tau_low) & (confs < cfg.tau_high))[0]

        # 1. Predict all existing tracks forward by dt_sec
        for trk in self.tracks:
            trk.predict(dt_sec)

        confirmed = [t for t in self.tracks if t.state == "confirmed"]
        tentative = [t for t in self.tracks if t.state == "tentative"]

        matched_confirmed: set[int] = set()
        matched_tentative: set[int] = set()
        u_conf_idx = list(range(len(confirmed)))
        rem_high_idx = [int(i) for i in high_idx]

        # 2. Stage 1: Confirmed + lost tracks <-> High-confidence detections
        if confirmed and rem_high_idx:
            cost1 = iou_polar_cost_matrix(
                confirmed,
                dets[rem_high_idx],
                clss[rem_high_idx],
                t_sec=self.t_sec,
                dt_sec=dt_sec,
                cfg=cfg,
                iou_thresh=cfg.iou_thresh_high,
            )
            m1, u_conf_idx, u_det1 = linear_assignment(cost1, cost_limit=1.0 + cfg.iou_thresh_high)
            if m1:
                m_det_boxes = np.stack([dets[rem_high_idx[di]] for _, di in m1], axis=0)
                ov_list = self._max_other_ious(m_det_boxes, confirmed, [ti for ti, _ in m1])
                for (ti, di), max_ov in zip(m1, ov_list):
                    g_di = rem_high_idx[di]
                    confirmed[ti].update(
                        dets[g_di],
                        float(confs[g_di]),
                        int(clss[g_di]),
                        t_sec=self.t_sec,
                        dt_sec=dt_sec,
                        max_other_iou=max_ov,
                    )
                    matched_confirmed.add(ti)
            rem_high_idx = [rem_high_idx[di] for di in u_det1]

        # 3. Stage 2: Remaining active (k_lost == 0) confirmed tracks <-> Low-confidence detections
        rem_active_idx = [ti for ti in u_conf_idx if confirmed[ti].k_lost == 0.0]
        if rem_active_idx and len(low_idx) > 0:
            rem_active = [confirmed[ti] for ti in rem_active_idx]
            cost2 = iou_polar_cost_matrix(
                rem_active,
                dets[low_idx],
                clss[low_idx],
                t_sec=self.t_sec,
                dt_sec=dt_sec,
                cfg=cfg,
                iou_thresh=cfg.iou_thresh_low,
            )
            m2, _, _ = linear_assignment(cost2, cost_limit=1.0 + cfg.iou_thresh_high)
            if m2:
                m_det_boxes = np.stack([dets[int(low_idx[di])] for _, di in m2], axis=0)
                matched_global_ti = [rem_active_idx[lti] for lti, _ in m2]
                ov_list = self._max_other_ious(m_det_boxes, confirmed, matched_global_ti)
                for (lti, di), max_ov in zip(m2, ov_list):
                    ti = rem_active_idx[lti]
                    g_di = int(low_idx[di])
                    confirmed[ti].update(
                        dets[g_di],
                        float(confs[g_di]),
                        int(clss[g_di]),
                        t_sec=self.t_sec,
                        dt_sec=dt_sec,
                        max_other_iou=max_ov,
                    )
                    matched_confirmed.add(ti)

        # 4. Stage 3: Remaining lost (k_lost > 0) confirmed tracks <-> Remaining high-conf detections
        rem_lost_idx = [ti for ti in u_conf_idx if ti not in matched_confirmed and confirmed[ti].k_lost > 0.0]
        if rem_lost_idx and rem_high_idx:
            rem_lost = [confirmed[ti] for ti in rem_lost_idx]
            cost3 = relocation_polar_cost_matrix(
                rem_lost,
                dets[rem_high_idx],
                clss[rem_high_idx],
                t_sec=self.t_sec,
                dt_sec=dt_sec,
                cfg=cfg,
            )
            m3, _, u_det3 = linear_assignment(cost3, cost_limit=1.0)
            if m3:
                m_det_boxes = np.stack([dets[rem_high_idx[di]] for _, di in m3], axis=0)
                matched_global_ti = [rem_lost_idx[lti] for lti, _ in m3]
                ov_list = self._max_other_ious(m_det_boxes, confirmed, matched_global_ti)
                for (lti, di), max_ov in zip(m3, ov_list):
                    ti = rem_lost_idx[lti]
                    g_di = rem_high_idx[di]
                    confirmed[ti].update(
                        dets[g_di],
                        float(confs[g_di]),
                        int(clss[g_di]),
                        t_sec=self.t_sec,
                        dt_sec=dt_sec,
                        max_other_iou=max_ov,
                    )
                    matched_confirmed.add(ti)
            rem_high_idx = [rem_high_idx[di] for di in u_det3]

        # 5. Stage 4: Tentative tracks <-> Remaining high-confidence detections (pure IoU)
        if tentative and rem_high_idx:
            t_boxes = np.stack([t.get_bbox() for t in tentative], axis=0)
            rem_dets = dets[rem_high_idx]
            rem_clss = clss[rem_high_idx]
            cost4 = 1.0 - iou_matrix(t_boxes, rem_dets)
            if cfg.check_class_match:
                t_cls = np.array([t.cls_id for t in tentative], dtype=np.int64)[:, None]
                cost4 = np.where(t_cls != rem_clss[None, :], np.inf, cost4)
            m4, _, u_det4 = linear_assignment(cost4, cost_limit=1.0 - cfg.iou_thresh_high)
            for ti, di in m4:
                g_di = rem_high_idx[di]
                tentative[ti].update(
                    dets[g_di],
                    float(confs[g_di]),
                    int(clss[g_di]),
                    t_sec=self.t_sec,
                    dt_sec=dt_sec,
                    max_other_iou=0.0,
                )
                matched_tentative.add(ti)
            rem_high_idx = [rem_high_idx[di] for di in u_det4]

        # 6. Mark unmatched tracks missed & prune expired tracks
        for ti, trk in enumerate(confirmed):
            if ti not in matched_confirmed:
                trk.mark_missed(dt_sec)
        for ti, trk in enumerate(tentative):
            if ti not in matched_tentative:
                trk.mark_missed(dt_sec)

        kept_tracks: list[PolarEMATrack] = []
        for trk in self.tracks:
            if trk.state == "tentative" and trk.k_lost > cfg.early_term_lost:
                continue
            if trk.k_lost > cfg.max_lost_frames:
                continue
            kept_tracks.append(trk)
        self.tracks = kept_tracks

        # 7. Spawn new tentative tracks from remaining high-conf detections (live suppression)
        for g_di in rem_high_idx:
            det_box = dets[g_di]
            det_conf = float(confs[g_di])
            det_cls = int(clss[g_di])

            suppress = False
            for existing in self.tracks:
                if existing.k_lost > 0.0:
                    continue
                if cfg.check_class_match and existing.cls_id != det_cls:
                    continue
                ov = float(iou_matrix(det_box[None, :], existing.get_bbox()[None, :])[0, 0])
                if ov > cfg.min_iou_diff_new:
                    suppress = True
                    break
            if suppress:
                continue

            trk = self._spawn_track(det_box, det_conf, det_cls)
            if self.frame_idx == 1 or cfg.probation_frames == 1:
                trk.state = "confirmed"
                trk.hits = max(trk.hits, cfg.probation_frames)
            self.tracks.append(trk)

        # 8. Emit outputs
        outputs: list[tuple[int, np.ndarray, float, int, bool]] = []
        for trk in self.tracks:
            if trk.state != "confirmed":
                continue
            if trk.k_lost == 0.0:
                outputs.append((trk.id, trk.get_bbox(), trk.conf, trk.cls_id, False))
            elif cfg.emit_shadow_tracks:
                outputs.append((trk.id, trk.get_bbox(), trk.conf, trk.cls_id, True))
        return outputs
