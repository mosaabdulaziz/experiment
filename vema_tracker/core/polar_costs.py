"""Vectorized association cost matrices for PolarEMATracker (VEMA v2)."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .costs import iou_matrix
from .polar_config import VEMA2Config
from .polar_track import PolarEMATrack


def _polar_kinematic_terms(
    tracks: Sequence[PolarEMATrack],
    det_boxes: np.ndarray,
    t_sec: float,
    dt_sec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute (c_kin, rho, sigma_noise_sec, det_centers) for (N, M) pairs."""
    det_centers = np.column_stack(
        [
            0.5 * (det_boxes[:, 0] + det_boxes[:, 2]),
            0.5 * (det_boxes[:, 1] + det_boxes[:, 3]),
        ]
    )
    t_mvec = np.stack([t.m_vec for t in tracks], axis=0)
    t_p_anchor = np.stack([t.p_anchor for t in tracks], axis=0)
    t_t_anchor = np.array([t.t_anchor for t in tracks], dtype=np.float64)
    t_s_ema = np.array([t.s_ema for t in tracks], dtype=np.float64)
    t_r = np.array([t.R for t in tracks], dtype=np.float64)
    t_noise_sec = np.array([t.sigma_noise_sec(dt_sec) for t in tracks], dtype=np.float64)

    dt_a = np.maximum(float(t_sec) - t_t_anchor, 1e-4)[:, None, None]
    v_cand = (det_centers[None, :, :] - t_p_anchor[:, None, :]) / dt_a
    v_ema = (t_s_ema[:, None] * t_mvec)[:, None, :]

    diff_norm = np.linalg.norm(v_cand - v_ema, axis=2)
    cand_norm = np.linalg.norm(v_cand, axis=2)
    ema_norm = np.linalg.norm(v_ema, axis=2)
    c_kin = diff_norm / (cand_norm + ema_norm + t_noise_sec[:, None] + 1e-6)

    rho = t_r * (t_s_ema ** 2) / (t_s_ema ** 2 + t_noise_sec ** 2 + 1e-8)
    return c_kin, rho, t_noise_sec, det_centers


def iou_polar_cost_matrix(
    tracks: Sequence[PolarEMATrack],
    det_boxes: np.ndarray,
    det_clss: np.ndarray,
    t_sec: float,
    dt_sec: float,
    cfg: VEMA2Config,
    iou_thresh: float,
) -> np.ndarray:
    """Stage 1/2 row-scale-invariant IoU + additive polar kinematic cost matrix (N, M)."""
    n, m = len(tracks), len(det_boxes)
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float64)

    t_boxes = np.stack([t.get_bbox() for t in tracks], axis=0)
    iou = iou_matrix(t_boxes, det_boxes)
    c_kin, rho, _, _ = _polar_kinematic_terms(tracks, det_boxes, t_sec, dt_sec)

    cost = (1.0 - iou) + cfg.iou_thresh_high * rho[:, None] * c_kin
    cost = np.where(iou >= iou_thresh, cost, np.inf)

    if cfg.check_class_match:
        t_cls = np.array([t.cls_id for t in tracks], dtype=np.int64)[:, None]
        cost = np.where(t_cls != det_clss[None, :], np.inf, cost)

    return cost


def relocation_polar_cost_matrix(
    lost_tracks: Sequence[PolarEMATrack],
    det_boxes: np.ndarray,
    det_clss: np.ndarray,
    t_sec: float,
    dt_sec: float,
    cfg: VEMA2Config,
) -> np.ndarray:
    """Stage 3 anisotropic distance + polar kinematic relocation cost for lost tracks (N, M)."""
    n, m = len(lost_tracks), len(det_boxes)
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float64)

    c_kin, rho, t_noise_sec, det_centers = _polar_kinematic_terms(
        lost_tracks, det_boxes, t_sec, dt_sec
    )

    t_pos = np.stack([t.pos for t in lost_tracks], axis=0)
    t_dims = np.stack([t.d_ema for t in lost_tracks], axis=0)
    t_uhat = np.stack([t.u_hat for t in lost_tracks], axis=0)
    t_s_ema = np.array([t.s_ema for t in lost_tracks], dtype=np.float64)
    t_s_dev = np.array([t.s_dev for t in lost_tracks], dtype=np.float64)
    t_r = np.array([t.R for t in lost_tracks], dtype=np.float64)
    t_k_lost = np.array([t.k_lost for t in lost_tracks], dtype=np.float64)

    e = det_centers[None, :, :] - t_pos[:, None, :]
    proj_scalar = np.sum(e * t_uhat[:, None, :], axis=2)
    e_par = np.abs(proj_scalar)
    e_perp = np.linalg.norm(e - proj_scalar[:, :, None] * t_uhat[:, None, :], axis=2)

    sigma_base = 0.5 * np.sqrt(np.maximum(t_dims[:, 0] * t_dims[:, 1], 1.0))
    t_lost_sec = t_k_lost / cfg.ref_fps
    ang_std = np.sqrt(2.0 * np.maximum(1.0 - t_r, 0.0))

    sigma_par = sigma_base + t_lost_sec * (t_s_dev + t_noise_sec)
    sigma_perp = sigma_base + t_lost_sec * (t_s_ema * ang_std + t_noise_sec)

    has_dir = t_r > 1e-3
    dist_aniso = np.sqrt(
        (e_par / (sigma_par[:, None] + 1e-6)) ** 2
        + (e_perp / (sigma_perp[:, None] + 1e-6)) ** 2
    )
    dist_iso = np.linalg.norm(e, axis=2) / (
        sigma_base[:, None] + t_lost_sec[:, None] * t_noise_sec[:, None] + 1e-6
    )
    c_dist = np.minimum(1.0, np.where(has_dir[:, None], dist_aniso, dist_iso))

    cost = c_dist + cfg.iou_thresh_high * rho[:, None] * c_kin

    feasible = (c_dist <= (1.0 - cfg.iou_thresh_high)) & ~((rho[:, None] > 0.3) & (c_kin > 0.5))
    cost = np.where(feasible, cost, np.inf)

    if cfg.check_class_match:
        t_cls = np.array([t.cls_id for t in lost_tracks], dtype=np.int64)[:, None]
        cost = np.where(t_cls != det_clss[None, :], np.inf, cost)

    return cost
