"""PolarEMATrack: Continuous-health decoupled speed/heading EMA track state."""

from __future__ import annotations

import math

import numpy as np

from .polar_config import VEMA2Config


def _xyxy_to_cxcywh(box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cx = 0.5 * (box[0] + box[2])
    cy = 0.5 * (box[1] + box[3])
    w = max(float(box[2] - box[0]), 1.0)
    h = max(float(box[3] - box[1]), 1.0)
    return np.array([cx, cy], dtype=np.float64), np.array([w, h], dtype=np.float64)


class PolarEMATrack:
    """Single object track with decoupled speed/direction EMAs and continuous trust/health."""

    def __init__(
        self,
        track_id: int,
        bbox_xyxy: np.ndarray,
        conf: float,
        cls_id: int,
        t_sec: float,
        cfg: VEMA2Config,
    ) -> None:
        self.id = int(track_id)
        self.cls_id = int(cls_id)
        self.conf = float(conf)
        self.cfg = cfg

        pos, dim = _xyxy_to_cxcywh(bbox_xyxy)
        self.pos = pos
        self.d_ema = dim

        # Decoupled polar kinematics (speed in px/sec, mean resultant heading vector in unit disk)
        self.s_ema: float = 0.0
        self.m_vec: np.ndarray = np.zeros(2, dtype=np.float64)

        # Trust-weighted clean anchor (exact constant-velocity invariant)
        self.p_anchor: np.ndarray = pos.copy()
        self.t_anchor: float = float(t_sec)

        # Continuous visibility/health score H_t in [0, 1] and reference-frame lost counter
        denom = max(cfg.tau_high - cfg.tau_low, 1e-6)
        self.health: float = float(np.clip((conf - cfg.tau_low) / denom, 0.0, 1.0))
        self.k_lost: float = 0.0

        self.state: str = "tentative"
        self.hits: int = 1
        self.age: int = 1
        self.has_vel: bool = False

        # Online single-step innovation variance & speed-spread EMAs
        self.inn_sq: float = 0.0
        self.s_dev: float = 0.0
        self.p_var: float = (0.05 * float(np.min(self.d_ema))) ** 2

    @property
    def R(self) -> float:
        """Directional coherence ||m_vec|| in [0, 1]."""
        return float(min(1.0, np.linalg.norm(self.m_vec)))

    @property
    def u_hat(self) -> np.ndarray:
        """Unit heading vector derived from m_vec."""
        r = float(np.linalg.norm(self.m_vec))
        if r <= 1e-6:
            return np.zeros(2, dtype=np.float64)
        return self.m_vec / r

    @property
    def v_ema_sec(self) -> np.ndarray:
        """Effective velocity vector in px/sec: s_ema * m_vec."""
        return self.s_ema * self.m_vec

    @property
    def sigma_noise_px(self) -> float:
        """Capped per-step innovation std (px), immune to multi-frame coasting jumps."""
        raw = math.sqrt(max(self.inn_sq, 0.0))
        return float(min(raw, 0.10 * float(np.min(self.d_ema))))

    def sigma_noise_sec(self, dt_sec: float) -> float:
        """Detector noise floor in px/sec over timescale dt_sec."""
        px_floor = max(self.sigma_noise_px, 0.015 * float(np.min(self.d_ema)))
        return px_floor / max(float(dt_sec), 1e-3)

    @property
    def maneuver_ratio(self) -> float:
        """Compatibility property; direct confidence substitution avoids lag."""
        return 0.0

    def rho(self, dt_sec: float) -> float:
        """Kinematic reliability gate rho in [0, 1]."""
        s_noise = self.sigma_noise_sec(dt_sec)
        s2 = self.s_ema * self.s_ema
        return float(self.R * s2 / (s2 + s_noise * s_noise + 1e-8))

    def predict(self, dt_sec: float) -> np.ndarray:
        """Advance position forward by dt_sec seconds with analytically derived damping."""
        cfg = self.cfg
        t_max = max(cfg.max_lost_frames / cfg.ref_fps, 1e-6)
        t_lost_sec = self.k_lost / cfg.ref_fps
        damping = math.exp(-t_lost_sec / t_max)
        self.pos = self.pos + damping * self.v_ema_sec * dt_sec
        min_dim = float(np.min(self.d_ema))
        sigma_q = (0.15 * min_dim) * (dt_sec / cfg.tau_mem) + self.s_dev * dt_sec
        max_p = (0.5 * min_dim) ** 2
        self.p_var = min(self.p_var + sigma_q * sigma_q, max_p)
        self.age += 1
        return self.get_bbox()

    def mark_missed(self, dt_sec: float) -> None:
        """Decay continuous health and advance lost counter on unmatched frame."""
        alpha_h = 1.0 - math.exp(-dt_sec / self.cfg.tau_mem)
        self.health *= 1.0 - alpha_h
        self.k_lost += dt_sec * self.cfg.ref_fps

    def update(
        self,
        obs_xyxy: np.ndarray,
        obs_conf: float,
        obs_cls: int,
        t_sec: float,
        dt_sec: float,
        max_other_iou: float = 0.0,
    ) -> None:
        """Confidence-, overlap-, and truncation-weighted continuous state update."""
        cfg = self.cfg
        obs_pos, obs_d = _xyxy_to_cxcywh(obs_xyxy)

        denom = max(cfg.tau_high - cfg.tau_low, 1e-6)
        w_conf = float(np.clip((obs_conf - cfg.tau_low) / denom, 0.0, 1.0))
        w_occ = float(np.clip(1.0 - max_other_iou, 0.0, 1.0))
        area_ratio = (obs_d[0] * obs_d[1]) / (self.d_ema[0] * self.d_ema[1] + 1e-6)
        w_size = min(1.0, float(area_ratio)) ** 2
        w_ema = float(w_conf * w_occ * w_size)

        dt_anchor = max(float(t_sec) - self.t_anchor, 1e-4)
        v_obs = (obs_pos - self.p_anchor) / dt_anchor
        s_obs = float(np.linalg.norm(v_obs))

        # Single-step innovation relative to predicted position
        r_t = obs_pos - self.pos

        dt_mem_eff = min(dt_anchor, 5.0 * cfg.tau_mem)
        eta_v_base = 1.0 - math.exp(-dt_mem_eff / cfg.tau_mem)
        eta_d_base = eta_v_base
        alpha_h = 1.0 - math.exp(-dt_sec / cfg.tau_mem)

        # Update innovation variance ONLY on active (k_lost == 0) frames so coasting gaps don't spike it
        if self.has_vel and self.k_lost == 0.0:
            eta_inn = w_ema * eta_v_base
            self.inn_sq = (1.0 - eta_inn) * self.inn_sq + eta_inn * float(np.dot(r_t, r_t))

        # Scalar NSA Kalman gain + NIS maneuver boost (smooths steady jitter, zero lag on turns/re-entry)
        sigma_meas = max(self.sigma_noise_px, 0.02 * float(np.min(self.d_ema)))
        r_base = sigma_meas * sigma_meas
        r_t_var = r_base / (w_conf + 1e-3)
        nis = float(np.dot(r_t, r_t)) / (2.0 * (self.p_var + r_t_var) + 1e-8)
        p_eff = self.p_var * max(1.0, nis)
        kappa_t = p_eff / (p_eff + r_t_var + 1e-8)
        self.pos = (1.0 - kappa_t) * self.pos + kappa_t * obs_pos
        self.p_var = (1.0 - kappa_t) * p_eff

        # Speed reliability gate using derived detector noise floor
        s_noise = self.sigma_noise_sec(dt_anchor)
        g_s = (s_obs * s_obs) / (s_obs * s_obs + s_noise * s_noise + 1e-8)
        u_obs = (v_obs / (s_obs + 1e-8)) if s_obs > 1e-6 else np.zeros(2, dtype=np.float64)

        if not self.has_vel:
            if w_ema > 1e-3:
                self.s_ema = s_obs
                self.m_vec = g_s * u_obs
                self.has_vel = True
        else:
            eta_s = w_ema * eta_v_base
            eta_u = w_ema * g_s * eta_v_base
            eta_sd = w_ema * eta_d_base
            self.s_dev = (1.0 - eta_sd) * self.s_dev + eta_sd * abs(s_obs - self.s_ema)
            self.s_ema = (1.0 - eta_s) * self.s_ema + eta_s * s_obs
            self.m_vec = (1.0 - eta_u) * self.m_vec + eta_u * u_obs
            m_norm = float(np.linalg.norm(self.m_vec))
            if m_norm > 1.0:
                self.m_vec = self.m_vec / m_norm

        eta_d = w_ema * eta_d_base
        self.d_ema = (1.0 - eta_d) * self.d_ema + eta_d * obs_d

        # Exact trust-weighted anchor update (zero-bias constant-velocity invariant)
        self.p_anchor = (1.0 - w_ema) * self.p_anchor + w_ema * obs_pos
        self.t_anchor = (1.0 - w_ema) * self.t_anchor + w_ema * float(t_sec)

        # Continuous health & liveness driven by detection visibility
        self.health = float(np.clip((1.0 - alpha_h) * self.health + alpha_h * w_conf, 0.0, 1.0))
        self.conf = float(obs_conf)
        self.cls_id = int(obs_cls)
        self.k_lost = 0.0
        self.hits += 1
        if self.state == "tentative" and self.hits >= cfg.probation_frames:
            self.state = "confirmed"

    def get_bbox(self) -> np.ndarray:
        """Return [x1, y1, x2, y2] from current center and EMA dimensions."""
        hw = 0.5 * self.d_ema[0]
        hh = 0.5 * self.d_ema[1]
        return np.array(
            [self.pos[0] - hw, self.pos[1] - hh, self.pos[0] + hw, self.pos[1] + hh],
            dtype=np.float64,
        )
