"""Unit tests for VEMATrack, VEMATracker, costs, and pure-NumPy Hungarian solver."""

from __future__ import annotations

import numpy as np
import pytest

from vema_tracker.core.assignment import linear_assignment
from vema_tracker.core.config import VEMAConfig
from vema_tracker.core.costs import anisotropic_kinematic_cost, biou_matrix, direction_cosine_cost
from vema_tracker.core.track import VEMATrack
from vema_tracker.core.tracker import VEMATracker


def _box_from_cxcywh(cx: float, cy: float, w: float, h: float) -> np.ndarray:
    return np.array([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dtype=np.float64)


def test_hungarian_assignment_rectangular_and_inf():
    cost = np.array(
        [
            [0.1, 0.9, np.inf],
            [0.8, 0.2, 0.7],
            [np.inf, np.inf, 0.3],
            [0.9, 0.9, 0.9],
        ],
        dtype=np.float64,
    )
    matches, u_rows, u_cols = linear_assignment(cost, cost_limit=0.5)
    assert sorted(matches) == [(0, 0), (1, 1), (2, 2)]
    assert u_rows == [3]
    assert u_cols == []


def test_regime_i_ema_convergence_and_variance():
    cfg = VEMAConfig(alpha=0.8, probation_frames=2, dt_aware=False)
    trk = VEMATrack(1, _box_from_cxcywh(100.0, 200.0, 50.0, 100.0), conf=0.9, cfg=cfg)
    assert trk.state == "tentative"

    # Constant velocity (10, 0) per frame
    for step in range(1, 15):
        trk.predict(1.0)
        trk.update(_box_from_cxcywh(100.0 + 10.0 * step, 200.0, 50.0, 100.0), obs_conf=0.9)

    assert trk.state == "confirmed"
    assert trk.k_lost == 0.0
    assert np.allclose(trk.v_ema, [10.0, 0.0], atol=1e-2)
    assert np.allclose(trk.d_ema, [50.0, 100.0], atol=1e-2)
    assert trk.v_var < 1e-3


def test_regime_ii_truncation_gate_locks_size_and_rejects_backward_crop():
    cfg = VEMAConfig(tau_high=0.6, tau_low=0.15, area_ratio_min=0.75, dt_aware=False)
    trk = VEMATrack(1, _box_from_cxcywh(100.0, 200.0, 80.0, 160.0), conf=0.9, cfg=cfg)
    for step in range(1, 6):
        trk.predict(1.0)
        trk.update(_box_from_cxcywh(100.0 + 10.0 * step, 200.0, 80.0, 160.0), obs_conf=0.9)

    saved_d = trk.d_ema.copy()
    saved_v = trk.v_ema.copy()

    # Now enter partial occlusion: width shrinks to 40%, center shifts backward by 24 px, conf=0.30
    for step in range(6, 10):
        trk.predict(1.0)
        cx_true = 100.0 + 10.0 * step
        cropped_box = _box_from_cxcywh(cx_true - 24.0, 200.0, 32.0, 160.0)
        trk.update(cropped_box, obs_conf=0.30)
        assert trk.last_beta == 0.0
        assert trk.k_lost == 0.0  # must not inflate k_lost in Regime II
        assert np.allclose(trk.d_ema, saved_d)
        assert np.allclose(trk.v_ema, saved_v)


def test_regime_iii_coasting_and_reacquisition_no_velocity_spike():
    cfg = VEMAConfig(alpha=0.8, gamma=0.98, dt_aware=False)
    trk = VEMATrack(1, _box_from_cxcywh(100.0, 200.0, 60.0, 140.0), conf=0.9, cfg=cfg)
    for step in range(1, 5):
        trk.predict(1.0)
        trk.update(_box_from_cxcywh(100.0 + 10.0 * step, 200.0, 60.0, 140.0), obs_conf=0.9)

    # Full occlusion for 10 frames while target actually slowed slightly to 9 px/frame
    for _ in range(10):
        trk.predict(1.0)
        trk.mark_missed(1.0)

    assert trk.state == "lost"
    assert trk.k_lost == pytest.approx(10.0)

    # Re-acquire at frame 15 (11 frames since frame 4)
    trk.predict(1.0)
    reacq_cx = 140.0 + 9.0 * 11.0
    trk.update(_box_from_cxcywh(reacq_cx, 200.0, 60.0, 140.0), obs_conf=0.9)

    # Velocity must point forward (+x) near ~9-10 px/frame, NOT reverse/spike!
    assert trk.v_ema[0] > 8.0 and trk.v_ema[0] < 11.0
    assert trk.k_lost == 0.0
    assert trk.state == "confirmed"


def test_stationary_guard_prevents_cos_gate_freeze_or_nan():
    cfg = VEMAConfig(dt_aware=False)
    trk = VEMATrack(1, _box_from_cxcywh(500.0, 500.0, 80.0, 160.0), conf=0.9, cfg=cfg)
    trk.predict(1.0)
    trk.update(_box_from_cxcywh(501.0, 500.0, 80.0, 160.0), obs_conf=0.9)

    # Opposite 1.5px jitter under medium conf should NOT trigger cos reversal gate
    trk.predict(1.0)
    trk.update(_box_from_cxcywh(499.5, 500.0, 80.0, 160.0), obs_conf=0.35)
    assert np.all(np.isfinite(trk.pos))
    assert np.all(np.isfinite(trk.v_ema))
    assert trk.last_beta > 0.0


def test_dt_invariance_30fps_vs_10fps():
    # Identical physical motion: 300 px/s horizontally over 1.0 s
    cfg30 = VEMAConfig(ref_fps=30.0, dt_aware=True, probation_frames=1)
    cfg10 = VEMAConfig(ref_fps=30.0, dt_aware=True, probation_frames=1)
    t30 = VEMATracker(cfg30)
    t10 = VEMATracker(cfg10)

    for k in range(31):
        t_sec = k / 30.0
        box = _box_from_cxcywh(100.0 + 300.0 * t_sec, 300.0, 60.0, 140.0)[None, :]
        t30.update(box, np.array([0.9]), dt=1.0 / 30.0)

    for k in range(11):
        t_sec = k / 10.0
        box = _box_from_cxcywh(100.0 + 300.0 * t_sec, 300.0, 60.0, 140.0)[None, :]
        t10.update(box, np.array([0.9]), dt=1.0 / 10.0)

    # Both should infer ~10 px per 30Hz reference frame (= 300 px/s)
    assert t30.tracks[0].v_ema[0] == pytest.approx(10.0, abs=0.2)
    assert t10.tracks[0].v_ema[0] == pytest.approx(10.0, abs=0.2)


def test_anisotropic_cost_and_biou():
    cfg = VEMAConfig(dt_aware=False, biou_rate=0.1, enable_anisotropic_cost=True)
    trk = VEMATrack(1, _box_from_cxcywh(100.0, 200.0, 40.0, 80.0), conf=0.9, cfg=cfg)
    trk.predict(1.0)
    trk.update(_box_from_cxcywh(115.0, 200.0, 40.0, 80.0), obs_conf=0.9)
    trk.predict(1.0)
    trk.mark_missed(1.0)

    cx, cy = trk.pos
    # Two candidate boxes at equal Euclidean distance (20 px): one along velocity (+x), one lateral (+y)
    det_par = _box_from_cxcywh(cx + 20.0, cy, 40.0, 80.0)
    det_perp = _box_from_cxcywh(cx, cy + 20.0, 40.0, 80.0)
    dets = np.stack([det_par, det_perp], axis=0)

    c_pos = anisotropic_kinematic_cost([trk], dets, cfg)[0]
    assert c_pos[0] < c_pos[1]

    c_dir = direction_cosine_cost([trk], dets, cfg)[0]
    assert c_dir[0] < c_dir[1]

    biou = biou_matrix([trk], dets, cfg)[0]
    assert np.all(biou > 0.0)


def test_scale_velocity_looming_and_lock():
    cfg = VEMAConfig(
        alpha=0.8,
        scale_alpha=0.8,
        scale_max_rate=0.10,
        dt_aware=False,
        enable_scale_velocity=True,
    )
    trk = VEMATrack(1, _box_from_cxcywh(400.0, 400.0, 50.0, 100.0), conf=0.9, cfg=cfg)

    # Target approaching camera: +3 px/frame in width, +6 px/frame in height
    for step in range(1, 20):
        trk.predict(1.0)
        trk.update(
            _box_from_cxcywh(400.0, 400.0, 50.0 + 3.0 * step, 100.0 + 6.0 * step),
            obs_conf=0.9,
        )

    assert np.allclose(trk.v_d_ema, [3.0, 6.0], atol=0.1)
    # Prediction-correction eliminates steady-state EMA lag (unlike plain d_ema which lags by ~24 px)
    assert trk.d_ema[0] == pytest.approx(50.0 + 3.0 * 19, abs=0.2)
    assert trk.d_ema[1] == pytest.approx(100.0 + 6.0 * 19, abs=0.2)

    # Now enter Regime II truncation: size_lock keeps pre-occlusion d_ema
    saved_d = trk.d_ema.copy()
    trk.predict(1.0)
    trk.update(_box_from_cxcywh(400.0, 400.0, 30.0, 90.0), obs_conf=0.30)
    assert trk.last_beta == 0.0
    assert np.allclose(trk.d_ema, saved_d)
