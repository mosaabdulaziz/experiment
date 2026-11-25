"""Deterministic synthetic MOT stress-test scenarios with exact ground truth."""

from __future__ import annotations

import numpy as np

from vema_tracker.baselines.base import Detection

from .mot_format import SequenceData


def _cxcywh_to_xyxy(cx: float, cy: float, w: float, h: float) -> np.ndarray:
    return np.array([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dtype=np.float64)


def _scenario_partial_occlusion_truncation(rng: np.random.Generator) -> SequenceData:
    """Targets enter partial occluder (area drops ~55%, center shifts backward) then full occlusion."""
    num_frames = 120
    fps = 30.0
    dets: dict[int, list[Detection]] = {}
    gt: dict[int, list[tuple[int, np.ndarray]]] = {}

    # 6 pedestrians spaced vertically to isolate occlusion dynamics cleanly
    tracks_spec = [
        (1, 100.0, 140.0, 8.0, 0.5, 70.0, 170.0, True),
        (2, 140.0, 340.0, 9.5, -0.4, 72.0, 175.0, True),
        (3, 180.0, 560.0, 7.2, 0.3, 68.0, 165.0, True),
        (4, 900.0, 760.0, -8.5, 0.2, 74.0, 172.0, True),
        (5, 120.0, 920.0, 6.5, 0.0, 70.0, 168.0, False),
        (6, 980.0, 220.0, -7.0, 0.4, 70.0, 168.0, False),
    ]

    for f in range(1, num_frames + 1):
        f_dets: list[Detection] = []
        f_gt: list[tuple[int, np.ndarray]] = []
        t = f - 1
        for tid, x0, y0, vx, vy, w, h, has_occ in tracks_spec:
            cx = x0 + vx * t
            cy = y0 + vy * t
            gt_box = _cxcywh_to_xyxy(cx, cy, w, h)
            f_gt.append((tid, gt_box))

            jitter = rng.normal(0.0, 1.0, size=2)
            if has_occ and 35 <= f <= 42:
                # Partial occlusion: leading edge cropped -> width shrinks to 45%, center shifts backward
                crop_w = 0.45 * w
                shift = -np.sign(vx) * 0.275 * w
                det_box = _cxcywh_to_xyxy(cx + shift + jitter[0], cy + jitter[1], crop_w, h * 0.95)
                conf = float(rng.uniform(0.24, 0.38))
                f_dets.append(Detection(det_box, conf, 0))
            elif has_occ and 43 <= f <= 54:
                # Full occlusion: zero detections
                continue
            else:
                det_box = _cxcywh_to_xyxy(cx + jitter[0], cy + jitter[1], w, h)
                conf = float(rng.uniform(0.82, 0.94))
                f_dets.append(Detection(det_box, conf, 0))

        dets[f] = f_dets
        gt[f] = f_gt

    return SequenceData(
        name="partial_occlusion_truncation",
        fps=fps,
        width=1920,
        height=1080,
        num_frames=num_frames,
        detections=dets,
        gt=gt,
    )


def _scenario_crossing_crowds(rng: np.random.Generator) -> SequenceData:
    """Eight identical-size targets crossing obliquely with 8-frame occlusions at intersections."""
    num_frames = 100
    fps = 30.0
    dets: dict[int, list[Detection]] = {}
    gt: dict[int, list[tuple[int, np.ndarray]]] = {}

    w, h = 54.0, 128.0
    specs = [
        # Left-to-right diagonal group
        (1, 200.0, 220.0, 9.0, 3.0),
        (2, 200.0, 380.0, 9.0, 3.0),
        (3, 200.0, 560.0, 9.0, 2.5),
        (4, 200.0, 740.0, 9.0, -2.5),
        # Right-to-left crossing group (collides near frame 45-55)
        (5, 1080.0, 520.0, -9.0, -3.0),
        (6, 1080.0, 680.0, -9.0, -3.0),
        (7, 1080.0, 810.0, -9.0, -2.5),
        (8, 1080.0, 490.0, -9.0, 2.5),
    ]

    for f in range(1, num_frames + 1):
        f_dets: list[Detection] = []
        f_gt: list[tuple[int, np.ndarray]] = []
        t = f - 1
        for tid, x0, y0, vx, vy in specs:
            cx = x0 + vx * t
            cy = y0 + vy * t
            f_gt.append((tid, _cxcywh_to_xyxy(cx, cy, w, h)))
            # Crossing group 5..8 occluded behind group 1..4 near intersection frames 45..52
            if tid >= 5 and 45 <= f <= 52:
                continue
            if tid <= 4 and 46 <= f <= 48:
                # Brief medium-conf frame at crossing
                conf = float(rng.uniform(0.35, 0.50))
            else:
                conf = float(rng.uniform(0.80, 0.95))
            jitter = rng.normal(0.0, 1.2, size=2)
            f_dets.append(Detection(_cxcywh_to_xyxy(cx + jitter[0], cy + jitter[1], w, h), conf, 0))
        dets[f] = f_dets
        gt[f] = f_gt

    return SequenceData(
        name="crossing_crowds",
        fps=fps,
        width=1920,
        height=1080,
        num_frames=num_frames,
        detections=dets,
        gt=gt,
    )


def _scenario_stationary_jitter(rng: np.random.Generator) -> SequenceData:
    """Stationary seated/standing targets with detector jitter, low-conf periods, and dropouts."""
    num_frames = 120
    fps = 30.0
    dets: dict[int, list[Detection]] = {}
    gt: dict[int, list[tuple[int, np.ndarray]]] = {}

    specs = [
        (1, 320.0, 400.0, 75.0, 160.0),
        (2, 680.0, 420.0, 72.0, 155.0),
        (3, 1080.0, 450.0, 78.0, 165.0),
        (4, 1460.0, 410.0, 70.0, 150.0),
    ]

    for f in range(1, num_frames + 1):
        f_dets: list[Detection] = []
        f_gt: list[tuple[int, np.ndarray]] = []
        for tid, cx, cy, w, h in specs:
            f_gt.append((tid, _cxcywh_to_xyxy(cx, cy, w, h)))
            if tid in (1, 3) and 50 <= f <= 64:
                # 15-frame dropout
                continue
            jitter = rng.normal(0.0, 2.0, size=2)
            size_jit = rng.normal(0.0, 1.5, size=2)
            if tid in (2, 4) and 30 <= f <= 45:
                conf = float(rng.uniform(0.22, 0.35))
            else:
                conf = float(rng.uniform(0.78, 0.93))
            f_dets.append(
                Detection(
                    _cxcywh_to_xyxy(cx + jitter[0], cy + jitter[1], w + size_jit[0], h + size_jit[1]),
                    conf,
                    0,
                )
            )
        dets[f] = f_dets
        gt[f] = f_gt

    return SequenceData(
        name="stationary_jitter",
        fps=fps,
        width=1920,
        height=1080,
        num_frames=num_frames,
        detections=dets,
        gt=gt,
    )


def _scenario_variable_fps_drops(rng: np.random.Generator) -> SequenceData:
    """Continuous 30 Hz physical trajectories sampled with irregular frame drops (10-30 Hz)."""
    phys_frames = 120
    base_dt = 1.0 / 30.0
    specs = [
        (1, 150.0, 260.0, 300.0, 20.0, 66.0, 150.0),
        (2, 1750.0, 460.0, -280.0, 35.0, 68.0, 155.0),
        (3, 220.0, 720.0, 260.0, -30.0, 64.0, 148.0),
        (4, 1680.0, 840.0, -250.0, -20.0, 70.0, 160.0),
    ]

    # Sample pattern: irregular steps of 1, 2, or 3 base frames (30 fps -> 15 fps -> 10 fps)
    kept_steps: list[int] = [0]
    step_choices = [1, 1, 2, 3, 1, 2, 1, 3, 2, 1]
    idx = 0
    while True:
        nxt = kept_steps[-1] + step_choices[idx % len(step_choices)]
        if nxt >= phys_frames:
            break
        kept_steps.append(nxt)
        idx += 1

    dets: dict[int, list[Detection]] = {}
    gt: dict[int, list[tuple[int, np.ndarray]]] = {}
    frame_dts: dict[int, float] = {}

    for out_f, phys_step in enumerate(kept_steps, start=1):
        t_sec = phys_step * base_dt
        dt_sec = (phys_step - kept_steps[out_f - 2]) * base_dt if out_f > 1 else base_dt
        frame_dts[out_f] = dt_sec

        f_dets: list[Detection] = []
        f_gt: list[tuple[int, np.ndarray]] = []
        for tid, x0, y0, vx_s, vy_s, w, h in specs:
            cx = x0 + vx_s * t_sec
            cy = y0 + vy_s * t_sec
            f_gt.append((tid, _cxcywh_to_xyxy(cx, cy, w, h)))
            # Simulate brief 2-frame detection dropout mid-run while dt jumps
            if tid in (1, 2) and 22 <= out_f <= 24:
                continue
            jitter = rng.normal(0.0, 1.0, size=2)
            conf = float(rng.uniform(0.80, 0.94))
            f_dets.append(Detection(_cxcywh_to_xyxy(cx + jitter[0], cy + jitter[1], w, h), conf, 0))
        dets[out_f] = f_dets
        gt[out_f] = f_gt

    return SequenceData(
        name="variable_fps_drops",
        fps=30.0,
        width=1920,
        height=1080,
        num_frames=len(kept_steps),
        detections=dets,
        gt=gt,
        frame_dts=frame_dts,
    )


def _scenario_high_conf_crossing_swap(rng: np.random.Generator) -> SequenceData:
    """Two identical-size pedestrians cross in opposite directions with high-conf overlap."""
    num_frames = 70
    fps = 30.0
    dets: dict[int, list[Detection]] = {}
    gt: dict[int, list[tuple[int, np.ndarray]]] = {}

    w, h = 60.0, 150.0
    # Cross at t=30 (frame 31), x=700
    specs = [
        (1, 400.0, 500.0, 10.0, 0.0),
        (2, 1000.0, 500.0, -10.0, 0.0),
    ]

    for f in range(1, num_frames + 1):
        f_dets: list[Detection] = []
        f_gt: list[tuple[int, np.ndarray]] = []
        t = f - 1
        for tid, x0, y0, vx, vy in specs:
            cx = x0 + vx * t
            cy = y0 + vy * t
            f_gt.append((tid, _cxcywh_to_xyxy(cx, cy, w, h)))

            # During crossing frames 30..33, Ped 2 is occluded behind Ped 1 while Ped 1
            # keeps high detector confidence (0.80..0.90) with slight merged box jitter
            if tid == 2 and 30 <= f <= 33:
                continue
            jitter = rng.normal(0.0, 1.0, size=2)
            cx_obs = cx + jitter[0]
            if tid == 1 and 29 <= f <= 33:
                cx_obs -= 6.0  # slight merged-box center pull toward center
            conf = float(rng.uniform(0.80, 0.92))
            f_dets.append(Detection(_cxcywh_to_xyxy(cx_obs, cy + jitter[1], w, h), conf, 0))
        dets[f] = f_dets
        gt[f] = f_gt

    return SequenceData(
        name="high_conf_crossing_swap",
        fps=fps,
        width=1920,
        height=1080,
        num_frames=num_frames,
        detections=dets,
        gt=gt,
    )


def _scenario_overtaking_same_direction(rng: np.random.Generator) -> SequenceData:
    """Slow walker overtaken along same line by fast runner with overlap/occlusion."""
    num_frames = 80
    fps = 30.0
    dets: dict[int, list[Detection]] = {}
    gt: dict[int, list[tuple[int, np.ndarray]]] = {}

    w, h = 58.0, 148.0
    # Meet around t=33.3 (frame 34, x=533.3)
    specs = [
        (1, 450.0, 500.0, 2.5, 0.0),   # slow walker
        (2, 200.0, 500.0, 10.0, 0.0),  # fast runner
    ]

    for f in range(1, num_frames + 1):
        f_dets: list[Detection] = []
        f_gt: list[tuple[int, np.ndarray]] = []
        t = f - 1
        for tid, x0, y0, vx, vy in specs:
            cx = x0 + vx * t
            cy = y0 + vy * t
            f_gt.append((tid, _cxcywh_to_xyxy(cx, cy, w, h)))
            if tid == 1 and 33 <= f <= 36:
                continue
            jitter = rng.normal(0.0, 0.8, size=2)
            conf = float(rng.uniform(0.80, 0.92))
            f_dets.append(Detection(_cxcywh_to_xyxy(cx + jitter[0], cy + jitter[1], w, h), conf, 0))
        dets[f] = f_dets
        gt[f] = f_gt

    return SequenceData(
        name="overtaking_same_direction",
        fps=fps,
        width=1920,
        height=1080,
        num_frames=num_frames,
        detections=dets,
        gt=gt,
    )


def _scenario_unoccluded_uturn(rng: np.random.Generator) -> SequenceData:
    """Unoccluded pedestrian decelerates smoothly and reverses direction (U-turn)."""
    num_frames = 90
    fps = 30.0
    dets: dict[int, list[Detection]] = {}
    gt: dict[int, list[tuple[int, np.ndarray]]] = {}

    w, h = 65.0, 160.0
    cx = 300.0
    cy = 500.0

    for f in range(1, num_frames + 1):
        if f <= 35:
            vx = 9.0
        elif f <= 45:
            frac = (f - 35) / 10.0
            vx = 9.0 * (1.0 - 2.0 * frac)
        else:
            vx = -9.0
        if f > 1:
            cx += vx
        gt_box = _cxcywh_to_xyxy(cx, cy, w, h)
        gt[f] = [(1, gt_box)]
        jitter = rng.normal(0.0, 1.0, size=2)
        conf = float(rng.uniform(0.82, 0.94))
        dets[f] = [Detection(_cxcywh_to_xyxy(cx + jitter[0], cy + jitter[1], w, h), conf, 0)]

    return SequenceData(
        name="unoccluded_uturn",
        fps=fps,
        width=1920,
        height=1080,
        num_frames=num_frames,
        detections=dets,
        gt=gt,
    )


def _scenario_standing_to_seated_posture(rng: np.random.Generator) -> SequenceData:
    """Pedestrian stands then sits permanently at high confidence (AreaRatio ~ 0.50)."""
    num_frames = 110
    fps = 30.0
    dets: dict[int, list[Detection]] = {}
    gt: dict[int, list[tuple[int, np.ndarray]]] = {}

    cx, cy_stand = 600.0, 500.0
    for f in range(1, num_frames + 1):
        if f <= 30:
            w, h, cy = 70.0, 170.0, cy_stand
        else:
            w, h, cy = 75.0, 80.0, cy_stand + 45.0
        gt[f] = [(1, _cxcywh_to_xyxy(cx, cy, w, h))]
        jitter = rng.normal(0.0, 1.2, size=2)
        conf = float(rng.uniform(0.82, 0.92))
        dets[f] = [Detection(_cxcywh_to_xyxy(cx + jitter[0], cy + jitter[1], w, h), conf, 0)]

    return SequenceData(
        name="standing_to_seated_posture",
        fps=fps,
        width=1920,
        height=1080,
        num_frames=num_frames,
        detections=dets,
        gt=gt,
    )


def generate_synthetic_suite(seed: int = 42) -> list[SequenceData]:
    """Generate all 8 deterministic synthetic stress-test sequences."""
    rng = np.random.default_rng(seed)
    return [
        _scenario_partial_occlusion_truncation(rng),
        _scenario_crossing_crowds(rng),
        _scenario_stationary_jitter(rng),
        _scenario_variable_fps_drops(rng),
        _scenario_high_conf_crossing_swap(rng),
        _scenario_overtaking_same_direction(rng),
        _scenario_unoccluded_uturn(rng),
        _scenario_standing_to_seated_posture(rng),
    ]
