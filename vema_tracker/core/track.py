"""VEMATrack: 3-regime kinematic-geometric state machine."""

from __future__ import annotations

import numpy as np

from .config import VEMAConfig


def xyxy_to_cxcywh(bbox: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    w = max(x2 - x1, 1e-3)
    h = max(y2 - y1, 1e-3)
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    return np.array([cx, cy], dtype=np.float64), np.array([w, h], dtype=np.float64)


def cxcywh_to_xyxy(pos: np.ndarray, dims: np.ndarray) -> np.ndarray:
    half_w = 0.5 * dims[0]
    half_h = 0.5 * dims[1]
    return np.array(
        [pos[0] - half_w, pos[1] - half_h, pos[0] + half_w, pos[1] + half_h],
        dtype=np.float64,
    )


class VEMATrack:
    """Single tracked object maintaining a compact Kinematic-Geometric State Vector."""

    def __init__(
        self,
        track_id: int,
        bbox_xyxy: np.ndarray,
        conf: float,
        cls_id: int = 0,
        cfg: VEMAConfig | None = None,
    ) -> None:
        self.id = int(track_id)
        self.cfg = cfg or VEMAConfig()
        self.cls_id = int(cls_id)
        self.conf = float(conf)

        pos, dims = xyxy_to_cxcywh(bbox_xyxy)
        self.pos = pos
        self.d_ema = dims
        self.v_ema = np.zeros(2, dtype=np.float64)
        self.v_d_ema = np.zeros(2, dtype=np.float64)
        self.v_var = 0.0
        self.has_vel = False
        self.has_scale_vel = False

        self.last_obs_pos = self.pos.copy()
        self.last_obs_d = self.d_ema.copy()
        self._pre_predict_d = self.d_ema.copy()
        self.last_obs_time = 0.0
        self.time_now = 0.0

        self.k_lost = 0.0
        self.k_degraded = 0
        self.hits = 1
        self.age = 1
        self.last_beta = 1.0
        self.state = "confirmed" if self.cfg.probation_frames <= 1 else "tentative"

    def predict(self, dt_steps: float = 1.0) -> np.ndarray:
        """Advance state forward by dt_steps (in reference-frame units)."""
        dt_steps = max(float(dt_steps), 1e-4)
        self._pre_predict_d = self.d_ema.copy()
        if self.has_vel:
            damping = self.cfg.gamma ** self.k_lost
            self.pos = self.pos + damping * self.v_ema * dt_steps
        if (
            self.cfg.enable_scale_velocity
            and self.has_scale_vel
            and self.k_lost == 0.0
            and self.k_degraded == 0
        ):
            self.d_ema = np.maximum(self.d_ema + self.v_d_ema * dt_steps, 4.0)
        elif (
            self.cfg.enable_scale_velocity
            and self.has_scale_vel
            and self.cfg.scale_gamma > 0.0
            and self.k_degraded == 0
        ):
            d_damp = self.cfg.scale_gamma ** self.k_lost
            self.d_ema = np.maximum(self.d_ema + d_damp * self.v_d_ema * dt_steps, 4.0)
        self.time_now += dt_steps
        self.age += 1
        return self.get_bbox()

    def mark_missed(self, dt_steps: float = 1.0) -> None:
        """Regime III: no reliable observation on this frame."""
        self.k_lost += max(float(dt_steps), 1e-4)
        if self.state == "confirmed":
            self.state = "lost"

    def update(
        self,
        obs_bbox_xyxy: np.ndarray,
        obs_conf: float,
        cls_id: int | None = None,
    ) -> None:
        """Update state from a matched detection (Regime I or Regime II)."""
        cfg = self.cfg
        obs_pos, obs_d = xyxy_to_cxcywh(obs_bbox_xyxy)
        obs_conf = float(obs_conf)
        self.conf = obs_conf
        if cls_id is not None:
            self.cls_id = int(cls_id)

        elapsed = max(self.time_now - self.last_obs_time, 1e-6)
        v_obs = (obs_pos - self.last_obs_pos) / elapsed
        alpha_eff = cfg.alpha ** min(elapsed, 5.0)

        # --- REGIME I: Healthy Detection ---
        if obs_conf >= cfg.tau_high:
            self.pos = obs_pos
            if not self.has_vel:
                self.v_ema = v_obs.copy()
                self.has_vel = True
            else:
                err_sq = float(np.sum((v_obs - self.v_ema) ** 2))
                var_alpha = cfg.v_var_alpha ** min(elapsed, 5.0)
                self.v_var = var_alpha * self.v_var + (1.0 - var_alpha) * err_sq
                self.v_ema = alpha_eff * self.v_ema + (1.0 - alpha_eff) * v_obs

            if cfg.enable_scale_velocity:
                v_d_raw = (obs_d - self.last_obs_d) / elapsed
                max_sd = cfg.scale_max_rate * self.d_ema
                v_d_obs = np.clip(v_d_raw, -max_sd, max_sd)
                if not self.has_scale_vel:
                    self.v_d_ema = v_d_obs.copy()
                    self.has_scale_vel = True
                else:
                    s_alpha_eff = cfg.scale_alpha ** min(elapsed, 5.0)
                    self.v_d_ema = s_alpha_eff * self.v_d_ema + (1.0 - s_alpha_eff) * v_d_obs

            self.d_ema = alpha_eff * self.d_ema + (1.0 - alpha_eff) * obs_d
            self.k_lost = 0.0
            self.k_degraded = 0
            self.last_obs_pos = obs_pos.copy()
            self.last_obs_d = obs_d.copy()
            self.last_obs_time = self.time_now
            self.last_beta = 1.0
            self.hits += 1
            if self.hits >= cfg.probation_frames:
                self.state = "confirmed"
            return

        # --- REGIME II: Degraded Detection (Partial Occlusion) ---
        if obs_conf >= cfg.tau_low:
            area_ratio = float(obs_d[0] * obs_d[1]) / float(self.d_ema[0] * self.d_ema[1] + 1e-6)
            norm_obs = float(np.linalg.norm(v_obs))
            norm_ema = float(np.linalg.norm(self.v_ema))
            min_speed = cfg.min_speed_ratio * float(min(self.d_ema[0], self.d_ema[1]))

            if self.has_vel and norm_ema > min_speed and norm_obs > min_speed:
                cos_sim = float(np.dot(v_obs, self.v_ema) / (norm_obs * norm_ema))
            else:
                cos_sim = 1.0

            gated = cfg.enable_truncation_gate and (
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
                if not self.has_vel:
                    v_blend = beta * v_obs
                    self.v_ema = v_blend.copy()
                    self.has_vel = True
                else:
                    v_blend = beta * v_obs + (1.0 - beta) * self.v_ema
                    max_delta = cfg.eps_cal * (norm_ema + 1.0)
                    delta_v = np.clip(beta * (v_obs - self.v_ema), -max_delta, max_delta)
                    self.v_ema = self.v_ema + delta_v

                self.pos = self.last_obs_pos + v_blend * elapsed
                self.k_lost = 0.0
                self.k_degraded = 0
                self.last_obs_pos = self.pos.copy()
                self.last_obs_time = self.time_now
                self.hits += 1
            else:
                # Truncated/reversed observation rejected: hold predicted pos, v_ema,
                # and pre-occlusion d_ema; do NOT inflate k_lost.
                if cfg.enable_size_lock:
                    self.d_ema = self._pre_predict_d.copy()
                self.k_degraded += 1

            if not cfg.enable_size_lock:
                self.d_ema = alpha_eff * self.d_ema + (1.0 - alpha_eff) * obs_d
                self.last_obs_d = obs_d.copy()

            if self.state == "lost":
                self.state = "confirmed"
            return

        # --- REGIME III fallback if called directly with conf < tau_low ---
        self.last_beta = 0.0
        self.mark_missed(0.0)

    def get_bbox(
        self,
        buffered: bool = False,
        biou_rate: float | None = None,
        max_expansion: float | None = None,
    ) -> np.ndarray:
        """Return [x1, y1, x2, y2] using locked EMA dimensions (optionally buffered)."""
        if buffered and self.k_lost > 0.0:
            rate = self.cfg.biou_rate if biou_rate is None else biou_rate
            cap = self.cfg.biou_max_expansion if max_expansion is None else max_expansion
            expansion = min(1.0 + rate * self.k_lost, cap)
        else:
            expansion = 1.0
        return cxcywh_to_xyxy(self.pos, self.d_ema * expansion)
