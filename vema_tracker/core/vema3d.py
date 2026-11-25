"""VEMA-3D: 3D Pole Projection + 2x2 Jacobian Depth Damping + Ground-Plane Mapped Mahalanobis Tracker."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .assignment import linear_assignment
from .camera3d import PinholeGroundCamera
from .config import VEMAConfig
from .costs import iou_matrix


def _inv_2x2_batch(s_mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact closed-form inverses of (..., 2, 2) symmetric matrices."""
    s00 = s_mat[..., 0, 0]
    s01 = s_mat[..., 0, 1]
    s11 = s_mat[..., 1, 1]
    det = np.maximum(s00 * s11 - s01 * s01, 1e-12)
    return det, np.stack(
        [
            np.stack([s11 / det, -s01 / det], axis=-1),
            np.stack([-s01 / det, s00 / det], axis=-1),
        ],
        axis=-2,
    )


def _mahalanobis_sq_2x2(diff: np.ndarray, cov: np.ndarray) -> float:
    """Squared Mahalanobis distance for a single 2D vector and 2x2 covariance."""
    s00 = float(cov[0, 0])
    s01 = float(cov[0, 1])
    s11 = float(cov[1, 1])
    det = max(s00 * s11 - s01 * s01, 1e-12)
    dx, dz = float(diff[0]), float(diff[1])
    return float((s11 * dx * dx - 2.0 * s01 * dx * dz + s00 * dz * dz) / det)


class VEMA3DTrack:
    """Single 3D pole tracked object with Dual-Anchor projection and Jacobian depth damping."""

    def __init__(
        self,
        track_id: int,
        bbox_xyxy: np.ndarray,
        conf: float,
        cls_id: int = 0,
        cfg: VEMAConfig | None = None,
        camera: PinholeGroundCamera | None = None,
    ) -> None:
        self.id = int(track_id)
        self.cfg = cfg or VEMAConfig()
        self.camera = camera or PinholeGroundCamera(
            cam_height=self.cfg.cam_height,
            pitch_deg=self.cfg.cam_pitch_deg,
            min_ray_y=self.cfg.min_ray_y,
            max_depth=self.cfg.max_depth,
        )
        self.cls_id = int(cls_id)
        self.conf = float(conf)

        obs_box = np.asarray(bbox_xyxy, dtype=np.float64).reshape(4)
        c2d = self._obs_cov_2d(obs_box, self.conf)

        pos_3d, pole_h, pole_w, cov_obs, base_elev = self.camera.bbox_to_pole_elev(obs_box, cov_2d=c2d)
        self.pos_3d = pos_3d.copy()
        self.v_3d = np.zeros(2, dtype=np.float64)
        self.pole_h = float(pole_h)
        self.pole_w = float(pole_w)
        self.base_elev = float(base_elev)
        self.cov_pos = cov_obs.copy()
        self.v_var_3d = 0.0

        self.has_vel = False
        self.last_obs_3d = self.pos_3d.copy()
        self.last_obs_bbox = obs_box.copy()
        self.last_obs_time = 0.0
        self.time_now = 0.0

        self.k_lost = 0.0
        self.k_degraded = 0
        self.last_anchor = "foot"
        self.hits = 1
        self.age = 1
        self.last_beta = 1.0
        self.state = "confirmed" if self.cfg.probation_frames <= 1 else "tentative"

        self._pre_predict_pole_h = self.pole_h
        self._pre_predict_pole_w = self.pole_w
        self._pre_predict_elev = self.base_elev
        self._consec_high_nis = 0

    def predict(self, dt_steps: float = 1.0) -> np.ndarray:
        """Advance 3D ground-plane pole position by dt_steps (reference-frame units)."""
        dt_steps = max(float(dt_steps), 1e-4)
        self._pre_predict_pole_h = self.pole_h
        self._pre_predict_pole_w = self.pole_w
        self._pre_predict_elev = self.base_elev

        if self.has_vel:
            damping = self.cfg.gamma ** self.k_lost
            self.pos_3d = self.pos_3d + damping * self.v_3d * dt_steps
            self.pos_3d[1] = float(np.clip(self.pos_3d[1], 0.3, self.camera.max_depth))

        fps_scale = 30.0 / max(self.cfg.ref_fps, 5.0)
        q_base = ((self.cfg.pos_3d_process_std * fps_scale) ** 2) * dt_steps
        q_vel = 0.25 * max(self.v_var_3d, 0.0) * (dt_steps ** 2)
        self.cov_pos = self.cov_pos + (q_base + q_vel) * np.eye(2, dtype=np.float64)
        # Keep covariance bounded and strictly symmetric PSD
        self.cov_pos = 0.5 * (self.cov_pos + self.cov_pos.T)
        self.cov_pos[0, 0] = min(max(float(self.cov_pos[0, 0]), 1e-6), 25.0)
        self.cov_pos[1, 1] = min(max(float(self.cov_pos[1, 1]), 1e-6), 25.0)
        max_cov = float(np.sqrt(self.cov_pos[0, 0] * self.cov_pos[1, 1])) * 0.99
        self.cov_pos[0, 1] = self.cov_pos[1, 0] = float(np.clip(self.cov_pos[0, 1], -max_cov, max_cov))

        self.time_now += dt_steps
        self.age += 1
        return self.get_bbox()

    def mark_missed(self, dt_steps: float = 1.0) -> None:
        """Regime III: damped 3D ground-plane coasting with locked pole_h/pole_w."""
        self.k_lost += max(float(dt_steps), 1e-4)
        if self.state == "confirmed":
            self.state = "lost"

    def _kalman_pos_update(
        self,
        obs_3d: np.ndarray,
        cov_obs: np.ndarray,
        allow_nis_boost: bool = True,
    ) -> np.ndarray:
        """2x2 Jacobian-weighted Kalman position update with persistent-NIS adaptation."""
        p_pred = self.cov_pos
        s_nom = p_pred + cov_obs
        innov = obs_3d - self.pos_3d
        d2 = _mahalanobis_sq_2x2(innov, s_nom)

        if allow_nis_boost and d2 > 6.0:
            self._consec_high_nis += 1
        else:
            self._consec_high_nis = 0

        if allow_nis_boost and self._consec_high_nis >= 2:
            # Sustained camera pan or physical turn across >=2 frames: adapt P_eff to avoid lag
            boost = min(max(d2 / 4.0, 1.0), 16.0)
            p_eff = p_pred * boost
        else:
            p_eff = p_pred

        s_mat = p_eff + cov_obs
        _, s_inv = _inv_2x2_batch(s_mat)
        k_gain = p_eff @ s_inv
        pos_filt = self.pos_3d + k_gain @ innov
        pos_filt[1] = float(np.clip(pos_filt[1], 0.3, self.camera.max_depth))

        i_mat = np.eye(2, dtype=np.float64)
        ik = i_mat - k_gain
        p_post = ik @ p_eff @ ik.T + k_gain @ cov_obs @ k_gain.T
        p_post = 0.5 * (p_post + p_post.T)
        p_post[0, 0] = max(float(p_post[0, 0]), 1e-6)
        p_post[1, 1] = max(float(p_post[1, 1]), 1e-6)
        self.cov_pos = p_post
        return pos_filt

    def _obs_cov_2d(self, obs_box: np.ndarray, obs_conf: float) -> np.ndarray:
        w_px = max(float(obs_box[2] - obs_box[0]), 1e-3)
        h_px = max(float(obs_box[3] - obs_box[1]), 1e-3)
        conf_scale = 1.0 + max(0.0, 1.0 - float(obs_conf)) * 0.8
        su = max(2.0, 0.008 * w_px) * conf_scale
        sv = max(2.0, 0.006 * h_px) * conf_scale
        return np.diag([su ** 2, sv ** 2])

    def _is_leg_occlusion(self, obs_box: np.ndarray, pred_box: np.ndarray) -> bool:
        """Detect bottom-edge leg occlusion: height shrunk, head y1 continuous, bottom y2 jumped up."""
        pred_h = max(float(pred_box[3] - pred_box[1]), 1.0)
        obs_h = max(float(obs_box[3] - obs_box[1]), 1e-3)
        h_ratio = obs_h / pred_h
        if h_ratio >= self.cfg.area_ratio_min:
            return False

        dy1 = abs(float(obs_box[1] - pred_box[1]))
        dy2_up = float(pred_box[3] - obs_box[3])
        return (
            dy1 <= max(8.0, 0.18 * pred_h)
            and dy2_up >= max(6.0, 0.12 * pred_h)
            and dy1 < dy2_up
        )

    def update(
        self,
        obs_bbox_xyxy: np.ndarray,
        obs_conf: float,
        cls_id: int | None = None,
    ) -> None:
        """Update 3D pole state from a matched 2D detection (Regime I, II, or III fallback)."""
        cfg = self.cfg
        obs_box = np.asarray(obs_bbox_xyxy, dtype=np.float64).reshape(4)
        obs_conf = float(obs_conf)
        self.conf = obs_conf
        if cls_id is not None:
            self.cls_id = int(cls_id)

        elapsed = max(self.time_now - self.last_obs_time, 1e-6)
        alpha_eff = cfg.alpha ** min(elapsed, 5.0)
        c2d = self._obs_cov_2d(obs_box, obs_conf)

        # --- REGIME I: Healthy Detection ---
        if obs_conf >= cfg.tau_high:
            obs_3d, obs_pole_h, obs_pole_w, cov_obs, obs_elev = self.camera.bbox_to_pole_elev(obs_box, cov_2d=c2d)
            self.pos_3d = self._kalman_pos_update(obs_3d, cov_obs, allow_nis_boost=True)

            v_obs_3d = (self.pos_3d - self.last_obs_3d) / elapsed
            if not self.has_vel:
                self.v_3d = v_obs_3d.copy()
                self.has_vel = True
            else:
                err_sq = float(np.sum((v_obs_3d - self.v_3d) ** 2))
                var_alpha = cfg.v_var_alpha ** min(elapsed, 5.0)
                self.v_var_3d = var_alpha * self.v_var_3d + (1.0 - var_alpha) * err_sq
                self.v_3d = alpha_eff * self.v_3d + (1.0 - alpha_eff) * v_obs_3d

            size_alpha = min(alpha_eff, 0.55)
            self.pole_h = max(size_alpha * self.pole_h + (1.0 - size_alpha) * obs_pole_h, 0.05)
            self.pole_w = max(size_alpha * self.pole_w + (1.0 - size_alpha) * obs_pole_w, 0.02)
            self.base_elev = size_alpha * self.base_elev + (1.0 - size_alpha) * obs_elev

            self.k_lost = 0.0
            self.k_degraded = 0
            self.last_anchor = "foot"
            self.last_obs_3d = self.pos_3d.copy()
            self.last_obs_bbox = obs_box.copy()
            self.last_obs_time = self.time_now
            self.last_beta = 1.0
            self.hits += 1
            if self.hits >= cfg.probation_frames:
                self.state = "confirmed"
            return

        # --- REGIME II: Degraded Detection (Partial Occlusion) ---
        if obs_conf >= cfg.tau_low:
            pred_box = self.get_bbox(buffered=False)
            pred_w = max(float(pred_box[2] - pred_box[0]), 1.0)
            pred_h = max(float(pred_box[3] - pred_box[1]), 1.0)
            obs_w = max(float(obs_box[2] - obs_box[0]), 1e-3)
            obs_h = max(float(obs_box[3] - obs_box[1]), 1e-3)

            min_speed_3d = max(0.012, 0.025 * max(self.pole_w, 0.3))
            norm_v_ema = float(np.linalg.norm(self.v_3d))

            if cfg.enable_truncation_gate and self._is_leg_occlusion(obs_box, pred_box):
                # Sub-case 2A: Bottom-edge leg occlusion -> Dual-Anchor Head Projection
                self.last_anchor = "head"
                obs_head_3d, cov_head = self.camera.head_bbox_to_pole(
                    obs_box,
                    self.pole_h + self.base_elev,
                    cov_2d=c2d,
                    fallback_z=float(self.pos_3d[1]),
                    base_elev=self.base_elev,
                )
                pos_head_filt = self._kalman_pos_update(obs_head_3d, cov_head, allow_nis_boost=False)
                v_obs_head = (pos_head_filt - self.last_obs_3d) / elapsed
                norm_obs = float(np.linalg.norm(v_obs_head))

                if self.has_vel and norm_v_ema > min_speed_3d and norm_obs > min_speed_3d:
                    cos_sim = float(np.dot(v_obs_head, self.v_3d) / (norm_obs * norm_v_ema))
                else:
                    cos_sim = 1.0

                if cos_sim < cfg.cos_min:
                    # Horizontal center shifted backward despite head anchor: hold predicted kinematics
                    self.last_beta = 0.0
                    self.k_degraded += 1
                else:
                    beta = (obs_conf - cfg.tau_low) / (cfg.tau_high - cfg.tau_low)
                    beta = float(np.clip(beta, 0.0, 1.0)) if cfg.enable_velocity_blending else 1.0
                    self.last_beta = beta

                    if not self.has_vel:
                        self.v_3d = (beta * v_obs_head).copy()
                        self.has_vel = True
                    else:
                        max_delta = cfg.eps_cal * (norm_v_ema + 0.05)
                        delta_v = np.clip(beta * (v_obs_head - self.v_3d), -max_delta, max_delta)
                        self.v_3d = self.v_3d + delta_v

                    self.pos_3d = pos_head_filt
                    self.k_lost = 0.0
                    self.k_degraded = 0
                    self.last_obs_3d = self.pos_3d.copy()
                    self.last_obs_bbox = obs_box.copy()
                    self.last_obs_time = self.time_now
                    self.hits += 1

                if cfg.enable_size_lock:
                    self.pole_h = self._pre_predict_pole_h
                    self.pole_w = self._pre_predict_pole_w
                    self.base_elev = self._pre_predict_elev
                if self.state == "lost":
                    self.state = "confirmed"
                return

            # Sub-case 2B: Non-leg-occlusion degraded detection -> Foot projection + truncation gate
            obs_3d, obs_pole_h, obs_pole_w, cov_obs, obs_elev = self.camera.bbox_to_pole_elev(obs_box, cov_2d=c2d)
            area_ratio = (obs_w * obs_h) / max(pred_w * pred_h, 1e-6)
            v_obs_raw = (obs_3d - self.last_obs_3d) / elapsed
            norm_obs_raw = float(np.linalg.norm(v_obs_raw))

            if self.has_vel and norm_v_ema > min_speed_3d and norm_obs_raw > min_speed_3d:
                cos_sim = float(np.dot(v_obs_raw, self.v_3d) / (norm_obs_raw * norm_v_ema))
            else:
                cos_sim = 1.0

            unoccluding = (obs_pole_h > 1.25 * self.pole_h) and (area_ratio >= 1.15)
            gated = cfg.enable_truncation_gate and (not unoccluding) and (
                area_ratio < cfg.area_ratio_min or cos_sim < cfg.cos_min
            )
            if gated:
                beta = 0.0
            elif not cfg.enable_velocity_blending:
                beta = 1.0
            else:
                beta = (obs_conf - cfg.tau_low) / (cfg.tau_high - cfg.tau_low)
                beta = float(np.clip(beta, 0.0, 1.0))

            self.last_beta = beta
            if beta > 0.0:
                pos_filt = self._kalman_pos_update(obs_3d, cov_obs, allow_nis_boost=False)
                v_obs_filt = (pos_filt - self.last_obs_3d) / elapsed

                if not self.has_vel:
                    v_blend = beta * v_obs_filt
                    self.v_3d = v_blend.copy()
                    self.has_vel = True
                else:
                    v_blend = beta * v_obs_filt + (1.0 - beta) * self.v_3d
                    max_delta = cfg.eps_cal * (norm_v_ema + 0.05)
                    delta_v = np.clip(beta * (v_obs_filt - self.v_3d), -max_delta, max_delta)
                    self.v_3d = self.v_3d + delta_v

                self.pos_3d = self.last_obs_3d + v_blend * elapsed
                self.pos_3d[1] = float(np.clip(self.pos_3d[1], 0.3, self.camera.max_depth))
                self.k_lost = 0.0
                self.k_degraded = 0
                self.last_anchor = "foot"
                self.last_obs_3d = self.pos_3d.copy()
                self.last_obs_bbox = obs_box.copy()
                self.last_obs_time = self.time_now
                self.hits += 1
            else:
                if cfg.enable_size_lock:
                    self.pole_h = self._pre_predict_pole_h
                    self.pole_w = self._pre_predict_pole_w
                    self.base_elev = self._pre_predict_elev
                self.k_degraded += 1

            expanding = beta > 0.0 and obs_pole_h > self.pole_h
            if not cfg.enable_size_lock or expanding:
                size_alpha = min(alpha_eff, 0.55) if expanding else alpha_eff
                self.pole_h = max(size_alpha * self.pole_h + (1.0 - size_alpha) * obs_pole_h, 0.05)
                self.pole_w = max(size_alpha * self.pole_w + (1.0 - size_alpha) * obs_pole_w, 0.02)
                if expanding:
                    self.base_elev = size_alpha * self.base_elev + (1.0 - size_alpha) * obs_elev

            if self.state == "lost":
                self.state = "confirmed"
            return

        # --- REGIME III fallback ---
        self.last_beta = 0.0
        self.mark_missed(0.0)

    def get_bbox(
        self,
        buffered: bool = False,
        biou_rate: float | None = None,
        max_expansion: float | None = None,
    ) -> np.ndarray:
        """Project 3D pole to 2D [x1, y1, x2, y2], optionally buffered for Stage 3 BIoU."""
        box = self.camera.project_pole_to_bbox(
            self.pos_3d,
            self.pole_h,
            self.pole_w,
            base_elev=self.base_elev,
        )
        if buffered and self.k_lost > 0.0:
            rate = self.cfg.biou_rate if biou_rate is None else biou_rate
            cap = self.cfg.biou_max_expansion if max_expansion is None else max_expansion
            expansion = min(1.0 + rate * self.k_lost, cap)
            if expansion != 1.0:
                cx = 0.5 * (box[0] + box[2])
                cy = 0.5 * (box[1] + box[3])
                hw = 0.5 * (box[2] - box[0]) * expansion
                hh = 0.5 * (box[3] - box[1]) * expansion
                return np.array([cx - hw, cy - hh, cx + hw, cy + hh], dtype=np.float64)
        return box


# ---------------------------------------------------------------------------
# Vectorized 3D Cost Matrices (MMD + Stage 3 Ground-Plane Relocation)
# ---------------------------------------------------------------------------

def mapped_mahalanobis_sq_matrix(
    tracks: Sequence[VEMA3DTrack],
    det_pos_3d: np.ndarray,
    det_cov_3d: np.ndarray,
    dets_xyxy: np.ndarray | None = None,
    cfg: VEMAConfig | None = None,
    check_leg_occ: bool = False,
) -> np.ndarray:
    """Compute (N, M) squared 3D Mapped Mahalanobis Distance d^2 = e^T (P_i + R_j)^(-1) e."""
    n = len(tracks)
    m = len(det_pos_3d)
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float64)

    trk_pos = np.stack([t.pos_3d for t in tracks], axis=0)         # (N, 2)
    trk_cov = np.stack([t.cov_pos for t in tracks], axis=0)        # (N, 2, 2)

    diff = det_pos_3d[None, :, :] - trk_pos[:, None, :]            # (N, M, 2)
    s_mat = trk_cov[:, None, :, :] + det_cov_3d[None, :, :, :]     # (N, M, 2, 2)
    _, s_inv = _inv_2x2_batch(s_mat)                               # (N, M, 2, 2)

    # d^2 = diff^T @ s_inv @ diff
    d2 = np.einsum("nmi,nmij,nmj->nm", diff, s_inv, diff)
    d2 = np.maximum(d2, 0.0)

    # Dual-anchor leg-occlusion refinement: if a candidate box looks leg-cropped relative to track i,
    # evaluate MMD using head_bbox_to_pole so foot teleportation does not penalize true match.
    if check_leg_occ and dets_xyxy is not None and cfg is not None and cfg.enable_truncation_gate:
        trk_boxes = np.stack([t.get_bbox(buffered=False) for t in tracks], axis=0)
        det_h = np.maximum(dets_xyxy[:, 3] - dets_xyxy[:, 1], 1e-3)
        trk_h = np.maximum(trk_boxes[:, 3] - trk_boxes[:, 1], 1.0)
        h_ratio = det_h[None, :] / trk_h[:, None]
        dy1 = np.abs(dets_xyxy[None, :, 1] - trk_boxes[:, None, 1])
        dy2_up = trk_boxes[:, None, 3] - dets_xyxy[None, :, 3]
        cand_mask = (
            (h_ratio < cfg.area_ratio_min)
            & (dy1 <= np.maximum(8.0, 0.18 * trk_h[:, None]))
            & (dy2_up >= np.maximum(6.0, 0.12 * trk_h[:, None]))
            & (dy1 < dy2_up)
        )
        ti_idx, di_idx = np.where(cand_mask)
        for ti, di in zip(ti_idx.tolist(), di_idx.tolist()):
            trk = tracks[ti]
            dbox = dets_xyxy[di]
            dw = max(float(dbox[2] - dbox[0]), 1e-3)
            dh = max(float(dbox[3] - dbox[1]), 1e-3)
            c2d = np.diag([max(2.0, 0.008 * dw) ** 2, max(2.0, 0.006 * dh) ** 2])
            h_pos, h_cov = trk.camera.head_bbox_to_pole(
                dbox,
                trk.pole_h + trk.base_elev,
                cov_2d=c2d,
                fallback_z=float(trk.pos_3d[1]),
                base_elev=trk.base_elev,
            )
            d2_head = _mahalanobis_sq_2x2(h_pos - trk.pos_3d, trk.cov_pos + h_cov)
            if d2_head < d2[ti, di]:
                d2[ti, di] = d2_head

    return d2


def stage12_blended_cost(
    tracks: Sequence[VEMA3DTrack],
    dets_xyxy: np.ndarray,
    dets_cls: np.ndarray,
    det_pos_3d: np.ndarray,
    det_cov_3d: np.ndarray,
    cfg: VEMAConfig,
    iou_thresh: float,
    check_leg_occ: bool = False,
) -> np.ndarray:
    """Blend 2D IoU cost with 3D Mapped Mahalanobis Distance (MMD) for Stage 1 and Stage 2."""
    n, m = len(tracks), len(dets_xyxy)
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float64)

    trk_boxes = np.stack([t.get_bbox(buffered=False) for t in tracks], axis=0)
    ious = iou_matrix(trk_boxes, dets_xyxy)

    w_mmd = float(np.clip(cfg.lambda_mmd, 0.0, 1.0))
    if w_mmd > 0.0:
        d2 = mapped_mahalanobis_sq_matrix(
            tracks,
            det_pos_3d,
            det_cov_3d,
            dets_xyxy=dets_xyxy,
            cfg=cfg,
            check_leg_occ=check_leg_occ,
        )
        c_mmd = np.clip(d2 / max(cfg.mmd_gate, 1e-6), 0.0, 1.0)
        cost = (1.0 - w_mmd) * (1.0 - ious) + w_mmd * c_mmd
    else:
        cost = 1.0 - ious

    cost = np.where(ious < iou_thresh, np.inf, cost)
    if cfg.check_class_match:
        trk_cls = np.array([t.cls_id for t in tracks], dtype=np.int64)[:, None]
        cost = np.where(trk_cls != np.asarray(dets_cls, dtype=np.int64)[None, :], np.inf, cost)
    return cost


def anisotropic_3d_kinematic_cost(
    tracks: Sequence[VEMA3DTrack],
    det_pos_3d: np.ndarray,
    det_cov_3d: np.ndarray,
    cfg: VEMAConfig,
) -> np.ndarray:
    """Compute 3D ground-plane anisotropic relocation cost C_pos_3d (N, M)."""
    n, m = len(tracks), len(det_pos_3d)
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float64)

    out = np.zeros((n, m), dtype=np.float64)
    for i, trk in enumerate(tracks):
        e = det_pos_3d - trk.pos_3d[None, :]  # (M, 2)
        min_dim_3d = max(float(trk.pole_w), 0.3)
        base = max(min_dim_3d * cfg.base_pos_sigma_ratio, 0.15)
        k = max(float(trk.k_lost), 1.0)
        v_std = float(np.sqrt(max(trk.v_var_3d, 0.0)))
        sigma_base = base + v_std * k

        norm_v = float(np.linalg.norm(trk.v_3d))
        min_speed_3d = max(0.012, 0.025 * min_dim_3d)
        if cfg.enable_anisotropic_cost and trk.has_vel and norm_v > min_speed_3d:
            u_hat = trk.v_3d / norm_v
            u_perp = np.array([-u_hat[1], u_hat[0]], dtype=np.float64)
            sig_par = (sigma_base * cfg.sigma_par_scale) ** 2
            sig_perp = (sigma_base * cfg.sigma_perp_scale) ** 2
            s_motion = sig_par * np.outer(u_hat, u_hat) + sig_perp * np.outer(u_perp, u_perp)
        else:
            s_motion = (sigma_base ** 2) * np.eye(2, dtype=np.float64)

        s_tot = s_motion[None, :, :] + det_cov_3d  # (M, 2, 2)
        _, s_inv = _inv_2x2_batch(s_tot)
        c_pos = np.einsum("mi,mij,mj->m", e, s_inv, e)
        out[i] = np.clip(c_pos, 0.0, 10.0)

    return out


def direction_3d_cosine_cost(
    tracks: Sequence[VEMA3DTrack],
    det_pos_3d: np.ndarray,
) -> np.ndarray:
    """Compute 3D ground-plane direction cosine cost (1 - cos(v_3d, det_3d - last_obs_3d))."""
    n, m = len(tracks), len(det_pos_3d)
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float64)

    out = np.zeros((n, m), dtype=np.float64)
    for i, trk in enumerate(tracks):
        min_dim_3d = max(float(trk.pole_w), 0.3)
        min_speed_3d = max(0.012, 0.025 * min_dim_3d)
        norm_v = float(np.linalg.norm(trk.v_3d))
        if not (trk.has_vel and norm_v > min_speed_3d):
            continue

        v_cand = det_pos_3d - trk.last_obs_3d[None, :]
        norm_c = np.linalg.norm(v_cand, axis=1)
        valid = norm_c > min_speed_3d
        if not np.any(valid):
            continue

        cos_theta = np.zeros(m, dtype=np.float64)
        cos_theta[valid] = (v_cand[valid] @ trk.v_3d) / (norm_c[valid] * norm_v)
        out[i] = np.where(valid, 1.0 - np.clip(cos_theta, -1.0, 1.0), 0.0)

    return out


def relocation_cost_matrix_3d(
    tracks: Sequence[VEMA3DTrack],
    dets_xyxy: np.ndarray,
    dets_cls: np.ndarray,
    det_pos_3d: np.ndarray,
    det_cov_3d: np.ndarray,
    cfg: VEMAConfig,
) -> np.ndarray:
    """Stage 3 3D relocation cost: l1*(1-BIoU) + l2*C_pos_3d + l3*C_dir_3d."""
    n, m = len(tracks), len(dets_xyxy)
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float64)

    trk_boxes = np.stack([t.get_bbox(buffered=cfg.enable_biou) for t in tracks], axis=0)
    biou = iou_matrix(trk_boxes, dets_xyxy)
    c_pos = anisotropic_3d_kinematic_cost(tracks, det_pos_3d, det_cov_3d, cfg)
    cost = cfg.lambda_biou * (1.0 - biou) + cfg.lambda_pos * c_pos

    if cfg.enable_dir_cost and cfg.lambda_dir > 0.0:
        c_dir = direction_3d_cosine_cost(tracks, det_pos_3d)
        cost = cost + cfg.lambda_dir * c_dir

    if cfg.check_class_match:
        trk_cls = np.array([t.cls_id for t in tracks], dtype=np.int64)[:, None]
        cls_mismatch = trk_cls != np.asarray(dets_cls, dtype=np.int64)[None, :]
        cost = np.where(cls_mismatch, np.inf, cost)

    return cost


class VEMA3DTracker:
    """Three-stage multi-object tracker operating in zero-calibration 3D pole coordinates."""

    def __init__(
        self,
        config: VEMAConfig | None = None,
        img_size: tuple[int, int] = (1920, 1080),
    ) -> None:
        self.cfg = config or VEMAConfig()
        self.img_size = (int(img_size[0]), int(img_size[1]))
        self.camera = self._build_camera(self.img_size)
        self.tracks: list[VEMA3DTrack] = []
        self._next_id = 1
        self.frame_idx = 0

    def _build_camera(self, img_size: tuple[int, int]) -> PinholeGroundCamera:
        return PinholeGroundCamera(
            width=img_size[0],
            height=img_size[1],
            cam_height=self.cfg.cam_height,
            pitch_deg=self.cfg.cam_pitch_deg,
            min_ray_y=self.cfg.min_ray_y,
            max_depth=self.cfg.max_depth,
        )

    def reset(
        self,
        fps: float | None = None,
        img_size: tuple[int, int] | None = None,
    ) -> None:
        if fps is not None and fps > 0:
            self.cfg = self.cfg.with_overrides(ref_fps=float(fps))
        if img_size is not None:
            self.img_size = (int(img_size[0]), int(img_size[1]))
        self.camera = self._build_camera(self.img_size)
        self.tracks = []
        self._next_id = 1
        self.frame_idx = 0

    def _spawn_track(self, bbox_xyxy: np.ndarray, conf: float, cls_id: int) -> VEMA3DTrack:
        trk = VEMA3DTrack(
            track_id=self._next_id,
            bbox_xyxy=bbox_xyxy,
            conf=conf,
            cls_id=cls_id,
            cfg=self.cfg,
            camera=self.camera,
        )
        self._next_id += 1
        return trk

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

        if self.frame_idx == 1 and cfg.auto_pitch and not self.tracks and len(dets) > 0:
            self.camera = self.camera.auto_pitch_for_bboxes(dets)

        if cfg.dt_aware and dt is not None and dt > 0.0:
            dt_steps = float(np.clip(float(dt) * cfg.ref_fps, 1e-4, 30.0))
        else:
            dt_steps = 1.0

        # 1. Predict all existing tracks forward in 3D ground coordinates
        for trk in self.tracks:
            trk.predict(dt_steps)

        active_tracks = [t for t in self.tracks if t.state in ("tentative", "confirmed")]
        lost_tracks = [t for t in self.tracks if t.state == "lost"]

        high_idx = np.where(confs >= cfg.tau_high)[0]
        low_idx = np.where((confs >= cfg.tau_low) & (confs < cfg.tau_high))[0]

        # Precompute 3D poles for candidate detections once per frame
        if len(dets) > 0:
            all_pos_3d, _, _, all_cov_3d = self.camera.bbox_to_pole_batch(dets, confs=confs)
        else:
            all_pos_3d = np.empty((0, 2), dtype=np.float64)
            all_cov_3d = np.empty((0, 2, 2), dtype=np.float64)

        matched_active: set[int] = set()
        matched_lost: set[int] = set()

        # 2. Stage 1: Active <-> High-confidence detections (2D IoU + 3D MMD)
        if active_tracks and len(high_idx) > 0:
            cost_s1 = stage12_blended_cost(
                active_tracks,
                dets[high_idx],
                clss[high_idx],
                all_pos_3d[high_idx],
                all_cov_3d[high_idx],
                cfg,
                iou_thresh=cfg.iou_thresh_high,
                check_leg_occ=False,
            )
            m1, u_trk1, u_det1 = linear_assignment(cost_s1, 1.0 - cfg.iou_thresh_high)
            for ti, di in m1:
                g_di = int(high_idx[di])
                active_tracks[ti].update(dets[g_di], confs[g_di], clss[g_di])
                matched_active.add(ti)
        else:
            u_trk1 = list(range(len(active_tracks)))
            u_det1 = list(range(len(high_idx)))

        # 3. Stage 2: Remaining CONFIRMED active tracks <-> Low-confidence detections (Dual-Anchor MMD + IoU)
        rem_confirmed_idx = [ti for ti in u_trk1 if active_tracks[ti].state == "confirmed"]
        if rem_confirmed_idx and len(low_idx) > 0:
            rem_confirmed = [active_tracks[ti] for ti in rem_confirmed_idx]
            cost_s2 = stage12_blended_cost(
                rem_confirmed,
                dets[low_idx],
                clss[low_idx],
                all_pos_3d[low_idx],
                all_cov_3d[low_idx],
                cfg,
                iou_thresh=cfg.iou_thresh_low,
                check_leg_occ=True,
            )
            m2, _, _ = linear_assignment(cost_s2, 1.0 - cfg.iou_thresh_low)
            for local_ti, di in m2:
                ti = rem_confirmed_idx[local_ti]
                g_di = int(low_idx[di])
                active_tracks[ti].update(dets[g_di], confs[g_di], clss[g_di])
                matched_active.add(ti)

        # 4. Stage 3: Lost tracks <-> Remaining High-confidence detections (3D Anisotropic Relocation)
        rem_high_idx = high_idx[u_det1] if len(u_det1) > 0 else np.empty(0, dtype=np.int64)
        unmatched_high_global: list[int] = [int(i) for i in rem_high_idx]
        if lost_tracks and len(rem_high_idx) > 0:
            cost_s3 = relocation_cost_matrix_3d(
                lost_tracks,
                dets[rem_high_idx],
                clss[rem_high_idx],
                all_pos_3d[rem_high_idx],
                all_cov_3d[rem_high_idx],
                cfg,
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

        kept_tracks: list[VEMA3DTrack] = []
        img_w, img_h = float(self.img_size[0]), float(self.img_size[1])
        for trk in self.tracks:
            if trk.state == "tentative" and trk.k_lost > cfg.early_term_lost:
                continue
            if trk.k_lost > cfg.max_lost_frames:
                continue
            if trk.state == "lost" and cfg.border_exit_lost > 0 and img_w > 0.0 and img_h > 0.0:
                pbox = trk.get_bbox(buffered=False)
                area = max(float(pbox[2] - pbox[0]), 0.0) * max(float(pbox[3] - pbox[1]), 0.0)
                vis_w = max(0.0, min(float(pbox[2]), img_w) - max(float(pbox[0]), 0.0))
                vis_h = max(0.0, min(float(pbox[3]), img_h) - max(float(pbox[1]), 0.0))
                vis_frac = (vis_w * vis_h) / area if area > 1e-6 else 0.0
                mostly_offscreen = (vis_frac < cfg.min_visible_frac) and (trk.k_lost > 1.0)
                m = cfg.border_exit_margin_px
                at_border = (
                    pbox[0] <= m or pbox[1] <= m or pbox[2] >= (img_w - m) or pbox[3] >= (img_h - m)
                )
                if mostly_offscreen or (at_border and trk.k_lost > cfg.border_exit_lost):
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
                if cfg.max_shadow_emit_lost <= 0 or trk.k_lost <= cfg.max_shadow_emit_lost:
                    out_box = trk.get_bbox(buffered=False)
                    if cfg.border_exit_lost > 0 and trk.k_lost > 2.0 and img_w > 0.0 and img_h > 0.0:
                        m = cfg.border_exit_margin_px
                        if out_box[0] <= m or out_box[1] <= m or out_box[2] >= (img_w - m) or out_box[3] >= (img_h - m):
                            continue
                    outputs.append((trk.id, out_box, trk.conf, trk.cls_id, True))
        return outputs
