"""Vectorized association cost matrices for VEMATracker."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .config import VEMAConfig
from .track import VEMATrack


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between boxes_a (N, 4) and boxes_b (M, 4) in [x1, y1, x2, y2]."""
    a = np.asarray(boxes_a, dtype=np.float64)
    b = np.asarray(boxes_b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)

    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter = inter_w * inter_h

    area_a = np.maximum(0.0, a[:, 2] - a[:, 0]) * np.maximum(0.0, a[:, 3] - a[:, 1])
    area_b = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 1e-9, inter / union, 0.0)


def biou_matrix(
    tracks: Sequence[VEMATrack],
    dets_xyxy: np.ndarray,
    cfg: VEMAConfig,
) -> np.ndarray:
    """Compute Buffered IoU (BIoU) between lost/active tracks and candidate detections."""
    if not tracks or len(dets_xyxy) == 0:
        return np.zeros((len(tracks), len(dets_xyxy)), dtype=np.float64)
    trk_boxes = np.stack([t.get_bbox(buffered=cfg.enable_biou) for t in tracks], axis=0)
    return iou_matrix(trk_boxes, dets_xyxy)


def anisotropic_kinematic_cost(
    tracks: Sequence[VEMATrack],
    dets_xyxy: np.ndarray,
    cfg: VEMAConfig,
) -> np.ndarray:
    """Compute normalized parallel/perpendicular displacement cost C_pos (N, M)."""
    n, m = len(tracks), len(dets_xyxy)
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float64)

    det_pos = np.column_stack(
        [
            0.5 * (dets_xyxy[:, 0] + dets_xyxy[:, 2]),
            0.5 * (dets_xyxy[:, 1] + dets_xyxy[:, 3]),
        ]
    )
    out = np.zeros((n, m), dtype=np.float64)

    for i, trk in enumerate(tracks):
        e = det_pos - trk.pos[None, :]
        min_dim = max(float(min(trk.d_ema[0], trk.d_ema[1])), 1.0)
        base = max(min_dim * cfg.base_pos_sigma_ratio, 1.0)
        k = max(float(trk.k_lost), 1.0)
        v_std = float(np.sqrt(max(trk.v_var, 0.0)))
        sigma_base = base + v_std * k

        norm_v = float(np.linalg.norm(trk.v_ema))
        min_speed = cfg.min_speed_ratio * min_dim
        if cfg.enable_anisotropic_cost and trk.has_vel and norm_v > min_speed:
            u_hat = trk.v_ema / norm_v
            proj = e @ u_hat
            e_par = np.abs(proj)
            e_perp = np.linalg.norm(e - np.outer(proj, u_hat), axis=1)
            sigma_par = sigma_base * cfg.sigma_par_scale
            sigma_perp = sigma_base * cfg.sigma_perp_scale
            c_pos = (e_par / sigma_par) ** 2 + (e_perp / sigma_perp) ** 2
        else:
            c_pos = np.sum(e ** 2, axis=1) / (sigma_base ** 2)

        out[i] = np.clip(c_pos, 0.0, 10.0)

    return out


def direction_cosine_cost(
    tracks: Sequence[VEMATrack],
    dets_xyxy: np.ndarray,
    cfg: VEMAConfig,
) -> np.ndarray:
    """Compute observation-centric direction cosine cost (1 - cos(v_ema, p_det - p_last_obs))."""
    n, m = len(tracks), len(dets_xyxy)
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float64)

    det_pos = np.column_stack(
        [
            0.5 * (dets_xyxy[:, 0] + dets_xyxy[:, 2]),
            0.5 * (dets_xyxy[:, 1] + dets_xyxy[:, 3]),
        ]
    )
    out = np.zeros((n, m), dtype=np.float64)

    for i, trk in enumerate(tracks):
        min_dim = max(float(min(trk.d_ema[0], trk.d_ema[1])), 1.0)
        min_speed = cfg.min_speed_ratio * min_dim
        norm_v = float(np.linalg.norm(trk.v_ema))
        if not (trk.has_vel and norm_v > min_speed):
            continue

        v_cand = det_pos - trk.last_obs_pos[None, :]
        norm_c = np.linalg.norm(v_cand, axis=1)
        valid = norm_c > min_speed
        if not np.any(valid):
            continue

        cos_theta = np.zeros(m, dtype=np.float64)
        cos_theta[valid] = (v_cand[valid] @ trk.v_ema) / (norm_c[valid] * norm_v)
        out[i] = np.where(valid, 1.0 - np.clip(cos_theta, -1.0, 1.0), 0.0)

    return out


def relocation_cost_matrix(
    tracks: Sequence[VEMATrack],
    dets_xyxy: np.ndarray,
    dets_cls: np.ndarray,
    cfg: VEMAConfig,
) -> np.ndarray:
    """Total Stage-3 ID recovery cost C_total = l1*(1-BIoU) + l2*C_pos + l3*C_dir."""
    n, m = len(tracks), len(dets_xyxy)
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float64)

    biou = biou_matrix(tracks, dets_xyxy, cfg)
    c_pos = anisotropic_kinematic_cost(tracks, dets_xyxy, cfg)
    cost = cfg.lambda_biou * (1.0 - biou) + cfg.lambda_pos * c_pos

    if cfg.enable_dir_cost and cfg.lambda_dir > 0.0:
        c_dir = direction_cosine_cost(tracks, dets_xyxy, cfg)
        cost = cost + cfg.lambda_dir * c_dir

    if cfg.check_class_match:
        trk_cls = np.array([t.cls_id for t in tracks], dtype=np.int64)[:, None]
        cls_mismatch = trk_cls != np.asarray(dets_cls, dtype=np.int64)[None, :]
        cost = np.where(cls_mismatch, np.inf, cost)

    return cost
