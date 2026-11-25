"""Unit tests and real-world failure-mode tests for VEMA v2 (PolarEMATracker)."""

from __future__ import annotations

import numpy as np
import pytest

from vema_tracker.baselines import get_tracker
from vema_tracker.core.assignment import linear_assignment
from vema_tracker.core.polar_config import VEMA2Config
from vema_tracker.core.polar_track import PolarEMATrack
from vema_tracker.core.polar_tracker import PolarEMATracker
from vema_tracker.datasets.synthetic import (
    _scenario_high_conf_crossing_swap,
    _scenario_overtaking_same_direction,
    _scenario_standing_to_seated_posture,
    _scenario_unoccluded_uturn,
)
from vema_tracker.eval.runner import run_tracker_on_sequence


def _box_from_cxcywh(cx: float, cy: float, w: float, h: float) -> np.ndarray:
    return np.array([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dtype=np.float64)


def test_hungarian_pre_masks_finite_costs_above_limit():
    """Finite entries > cost_limit must not distort optimal assignments for valid pairs."""
    cost = np.array(
        [
            [0.10, 0.74],
            [0.80, 10.0],
        ],
        dtype=np.float64,
    )
    matches, u_rows, u_cols = linear_assignment(cost, cost_limit=0.75)
    assert matches == [(0, 0)]
    assert u_rows == [1]
    assert u_cols == [1]


def test_exact_constant_velocity_anchor_invariant():
    """Trust-weighted anchor recovers exact constant velocity for arbitrary weight sequences."""
    cfg = VEMA2Config(dt_aware=True, ref_fps=30.0)
    dt = 1.0 / 30.0
    v_true = np.array([300.0, -120.0], dtype=np.float64)  # px/sec
    p0 = np.array([200.0, 400.0], dtype=np.float64)

    trk = PolarEMATrack(1, _box_from_cxcywh(p0[0], p0[1], 60.0, 150.0), conf=0.9, cls_id=0, t_sec=0.0, cfg=cfg)
    rng = np.random.default_rng(123)
    t_sec = 0.0
    for step in range(1, 20):
        t_sec += dt
        trk.predict(dt)
        p_obs = p0 + v_true * t_sec
        # Vary confidence across [0.16, 0.95] and overlap across [0.0, 0.8]
        conf = float(rng.uniform(0.16, 0.95))
        other_iou = float(rng.uniform(0.0, 0.8))
        dt_anchor = t_sec - trk.t_anchor
        v_obs = (p_obs - trk.p_anchor) / dt_anchor
        assert np.allclose(v_obs, v_true, atol=1e-9)
        trk.update(_box_from_cxcywh(p_obs[0], p_obs[1], 60.0, 150.0), conf, 0, t_sec=t_sec, dt_sec=dt, max_other_iou=other_iou)


def test_jitter_continuity_across_tau_high():
    """Confidence jitter around tau_high (0.59 vs 0.61) causes smooth < 0.05 change."""
    cfg = VEMA2Config(tau_high=0.60, tau_low=0.15, ref_fps=30.0)
    dt = 1.0 / 30.0

    def _run_probe(probe_conf: float) -> tuple[np.ndarray, float, float]:
        trk = PolarEMATrack(1, _box_from_cxcywh(100.0, 200.0, 60.0, 150.0), conf=0.9, cls_id=0, t_sec=0.0, cfg=cfg)
        for k in range(1, 8):
            t_sec = k * dt
            trk.predict(dt)
            trk.update(_box_from_cxcywh(100.0 + 10.0 * k, 200.0, 60.0, 150.0), 0.9, 0, t_sec=t_sec, dt_sec=dt)
        t_sec = 8 * dt
        trk.predict(dt)
        trk.update(_box_from_cxcywh(180.0, 200.0, 60.0, 150.0), probe_conf, 0, t_sec=t_sec, dt_sec=dt)
        return trk.pos.copy(), trk.s_ema, trk.health

    pos_lo, s_lo, h_lo = _run_probe(0.59)
    pos_hi, s_hi, h_hi = _run_probe(0.61)
    assert np.linalg.norm(pos_hi - pos_lo) < 0.5
    assert abs(s_hi - s_lo) / max(s_hi, 1.0) < 0.05
    assert abs(h_hi - h_lo) < 0.05


def test_stationary_jitter_zero_drift():
    """Stationary noisy detections yield low directional coherence R and minimal coast drift."""
    cfg = VEMA2Config(ref_fps=30.0)
    dt = 1.0 / 30.0
    rng = np.random.default_rng(7)
    trk = PolarEMATrack(1, _box_from_cxcywh(500.0, 500.0, 75.0, 160.0), conf=0.9, cls_id=0, t_sec=0.0, cfg=cfg)

    t_sec = 0.0
    for _ in range(1, 40):
        t_sec += dt
        trk.predict(dt)
        jit = rng.normal(0.0, 2.0, size=2)
        trk.update(_box_from_cxcywh(500.0 + jit[0], 500.0 + jit[1], 75.0, 160.0), 0.88, 0, t_sec=t_sec, dt_sec=dt)

    assert trk.R < 0.25
    start_pos = trk.pos.copy()
    for _ in range(10):
        trk.predict(dt)
        trk.mark_missed(dt)
    drift = float(np.linalg.norm(trk.pos - start_pos))
    assert drift < 5.0


def test_standing_to_seated_posture_survival():
    """Standing-to-seated posture change keeps the same ID alive and adapts d_ema."""
    seq = _scenario_standing_to_seated_posture(np.random.default_rng(42))
    tracker = get_tracker("vema2")
    tracker.reset(fps=seq.fps)

    observed_ids: set[int] = set()
    final_outputs = []
    for fid in range(1, seq.num_frames + 1):
        outs = tracker.update(seq.detections[fid], dt=1.0 / seq.fps)
        for o in outs:
            observed_ids.add(o.track_id)
        final_outputs = outs

    assert len(observed_ids) == 1
    assert len(final_outputs) == 1
    h_final = final_outputs[0].bbox[3] - final_outputs[0].bbox[1]
    assert h_final < 105.0


def test_crossing_and_overtaking_zero_idsw():
    """VEMA v2 resolves high-conf crossing swap, overtaking, and unoccluded U-turn cleanly."""
    rng = np.random.default_rng(42)
    for builder in (
        _scenario_high_conf_crossing_swap,
        _scenario_overtaking_same_direction,
        _scenario_unoccluded_uturn,
    ):
        seq = builder(rng)
        _, metrics, _ = run_tracker_on_sequence("vema2", seq)
        d = metrics.to_dict()
        assert d["IDSW"] == 0, f"{seq.name} had {d['IDSW']} IDSW: {d}"
        assert d["IDF1"] >= 95.0, f"{seq.name} had low IDF1: {d}"
