"""Unit and integration tests for PinholeGroundCamera and VEMA-3D."""

from __future__ import annotations

import numpy as np
import pytest

from vema_tracker.baselines import get_tracker
from vema_tracker.baselines.base import Detection
from vema_tracker.core.camera3d import PinholeGroundCamera
from vema_tracker.core.config import VEMAConfig
from vema_tracker.core.costs import iou_matrix
from vema_tracker.core.vema3d import VEMA3DTrack, VEMA3DTracker
from vema_tracker.datasets.synthetic import generate_synthetic_suite
from vema_tracker.eval.runner import run_tracker_on_sequence


def test_camera_default_intrinsics_and_roundtrip():
    cam = PinholeGroundCamera(1920, 1080)
    assert cam.fx == pytest.approx(0.95 * 1920)
    assert cam.fy == pytest.approx(0.95 * 1920)
    assert cam.cx == pytest.approx(960.0)
    assert cam.cy == pytest.approx(540.0)
    assert cam.cam_height == pytest.approx(2.0)
    assert cam.pitch_deg == pytest.approx(8.0)
    assert cam.min_ray_y == pytest.approx(0.02)
    assert cam.max_depth == pytest.approx(80.0)

    # Round-trip across near, mid, far, and off-center boxes below horizon
    test_boxes = [
        np.array([900.0, 420.0, 980.0, 680.0]),
        np.array([250.0, 360.0, 320.0, 560.0]),
        np.array([1500.0, 320.0, 1550.0, 430.0]),
        np.array([940.0, 620.0, 1060.0, 990.0]),
    ]
    for box in test_boxes:
        pos_3d, pole_h, pole_w, cov_3d = cam.bbox_to_pole(box)
        assert pos_3d.shape == (2,)
        assert cov_3d.shape == (2, 2)
        assert np.all(np.isfinite(pos_3d)) and pos_3d[1] > 0.0
        assert pole_h > 0.0 and pole_w > 0.0

        reproj = cam.project_pole_to_bbox(pos_3d, pole_h, pole_w)
        assert np.allclose(reproj, box, atol=1e-9)

        # Also verify batch matches single-box
        b_pos, b_h, b_w, b_cov = cam.bbox_to_pole_batch(box[None, :])
        assert np.allclose(b_pos[0], pos_3d, atol=1e-10)
        assert b_h[0] == pytest.approx(pole_h, abs=1e-10)
        assert b_w[0] == pytest.approx(pole_w, abs=1e-10)
        assert np.allclose(b_cov[0], cov_3d, atol=1e-10)


def test_head_to_ground_high_and_low_angle_safety():
    # Case 1: cam_height > pole_h (standard elevated surveillance camera)
    cam_high = PinholeGroundCamera(1920, 1080, cam_height=2.2, pitch_deg=8.0)
    true_pos = np.array([1.5, 12.0])
    pole_h, pole_w = 1.7, 0.6
    box = cam_high.project_pole_to_bbox(true_pos, pole_h, pole_w)
    u_cx = 0.5 * (box[0] + box[2])
    rec_high = cam_high.head_to_ground(u_cx, box[1], pole_h)
    assert np.isfinite(rec_high).all()
    assert rec_high[1] == pytest.approx(12.0, abs=1e-6)

    # Case 2: low-angle cam_height < pole_h (head lies above camera horizon!)
    cam_low = PinholeGroundCamera(1920, 1080, cam_height=1.4, pitch_deg=4.0)
    box_low = cam_low.project_pole_to_bbox(true_pos, pole_h, pole_w)
    assert box_low[1] < cam_low.v_horizon
    u_cx_low = 0.5 * (box_low[0] + box_low[2])
    rec_low = cam_low.head_to_ground(u_cx_low, box_low[1], pole_h)
    assert np.isfinite(rec_low).all()
    assert 0.3 <= rec_low[1] <= cam_low.max_depth
    assert rec_low[1] == pytest.approx(12.0, abs=1e-6)

    # Case 3: exact cam_height == pole_h and wildly noisy above/below horizon rays
    cam_eq = PinholeGroundCamera(1920, 1080, cam_height=1.7, pitch_deg=8.0)
    for v_test in [10.0, cam_eq.v_horizon, 900.0]:
        p_eq = cam_eq.head_to_ground(960.0, v_test, pole_h=1.7, fallback_z=14.0)
        assert np.isfinite(p_eq).all() and 0.3 <= p_eq[1] <= cam_eq.max_depth
        p_no_fb = cam_eq.head_to_ground(960.0, v_test, pole_h=1.7)
        assert np.isfinite(p_no_fb).all() and 0.3 <= p_no_fb[1] <= cam_eq.max_depth

    # Wrong-sign noise on cam_low (head pushed far below horizon) stays bounded & positive
    p_bad = cam_low.head_to_ground(960.0, 950.0, pole_h=1.7)
    assert np.isfinite(p_bad).all() and 0.3 <= p_bad[1] <= cam_low.max_depth


def test_jacobian_exactness_and_far_field_jitter_damping():
    cam = PinholeGroundCamera(1920, 1080)

    # Check analytical 2x2 Jacobian against central finite differences
    u, v = 1120.0, 460.0
    j_analytic = cam.jacobian_2d_to_3d(u, v)
    eps = 1e-4
    du = (cam.foot_to_ground(u + eps, v) - cam.foot_to_ground(u - eps, v)) / (2.0 * eps)
    dv = (cam.foot_to_ground(u, v + eps) - cam.foot_to_ground(u, v - eps)) / (2.0 * eps)
    j_fd = np.column_stack([du, dv])
    assert np.allclose(j_analytic, j_fd, atol=1e-6, rtol=1e-5)

    # Depth variance cov_3d[1, 1] grows strongly with depth Z
    box_near = cam.project_pole_to_bbox(np.array([0.0, 6.0]), 1.7, 0.6)
    box_far = cam.project_pole_to_bbox(np.array([0.0, 45.0]), 1.7, 0.6)
    cov_2d = np.diag([4.0, 4.0])
    _, _, _, cov_near = cam.bbox_to_pole(box_near, cov_2d=cov_2d)
    _, _, _, cov_far = cam.bbox_to_pole(box_far, cov_2d=cov_2d)
    assert cov_far[1, 1] > 100.0 * cov_near[1, 1]

    # Far-field jitter damping: steady target at Z = 45 m gets a 3-pixel y2 spike
    cfg = VEMAConfig(dt_aware=False, probation_frames=1)
    trk = VEMA3DTrack(1, box_far, conf=0.9, cfg=cfg, camera=cam)
    for _ in range(5):
        trk.predict(1.0)
        trk.update(box_far, obs_conf=0.9)

    z_before = trk.pos_3d[1]
    noisy_box = box_far.copy()
    noisy_box[3] -= 3.0  # noisy upward foot jump -> raw Z jumps far backward
    raw_noisy_pos, _, _, _ = cam.bbox_to_pole(noisy_box)
    raw_z_jump = abs(raw_noisy_pos[1] - z_before)

    trk.predict(1.0)
    trk.update(noisy_box, obs_conf=0.9)
    filtered_z_jump = abs(trk.pos_3d[1] - z_before)

    assert raw_z_jump > 1.0  # raw foot projection jumps by > 1m
    assert filtered_z_jump < 0.35 * raw_z_jump  # Jacobian covariance strongly damps Z jump
    assert abs(trk.v_3d[1]) < 0.35 * raw_z_jump  # velocity is not poisoned


def test_dual_anchor_leg_occlusion_prevents_depth_teleportation():
    cam = PinholeGroundCamera(1920, 1080, cam_height=2.0, pitch_deg=8.0)
    cfg = VEMAConfig(tau_high=0.60, tau_low=0.15, area_ratio_min=0.75, dt_aware=False, probation_frames=1)

    # Off-center pedestrian walking right (+X) at constant depth Z = 10.0 m
    z_true = 10.0
    pole_h, pole_w = 1.70, 0.60
    x0, vx = -1.5, 0.08
    init_box = cam.project_pole_to_bbox(np.array([x0, z_true]), pole_h, pole_w)
    trk = VEMA3DTrack(1, init_box, conf=0.9, cfg=cfg, camera=cam)

    for step in range(1, 8):
        trk.predict(1.0)
        box = cam.project_pole_to_bbox(np.array([x0 + vx * step, z_true]), pole_h, pole_w)
        trk.update(box, obs_conf=0.9)

    saved_h, saved_w = trk.pole_h, trk.pole_w
    assert trk.last_anchor == "foot"

    # Enter bottom-edge leg occlusion (legs cropped: y2 jumps up by 45%, y1 stays on head)
    for step in range(8, 13):
        trk.predict(1.0)
        full_box = cam.project_pole_to_bbox(np.array([x0 + vx * step, z_true]), pole_h, pole_w)
        cropped_box = full_box.copy()
        cropped_box[3] = full_box[1] + 0.55 * (full_box[3] - full_box[1])

        # Verify naive foot projection would have teleported Z backward to ~16+ m
        naive_foot, _, _, _ = cam.bbox_to_pole(cropped_box)
        assert naive_foot[1] > 15.0

        trk.update(cropped_box, obs_conf=0.35)
        assert trk.last_anchor == "head"
        assert trk.pole_h == pytest.approx(saved_h, abs=1e-9)
        assert trk.pole_w == pytest.approx(saved_w, abs=1e-9)
        # Depth Z stays tightly near true 10.0 m (NO teleport backward!)
        assert abs(trk.pos_3d[1] - z_true) < 0.35
        # Projected 2D box preserves full un-occluded legs with high IoU against true full box
        assert iou_matrix(trk.get_bbox()[None, :], full_box[None, :])[0, 0] > 0.90


def test_constant_velocity_3d_looming_and_tight_2d_output():
    cam = PinholeGroundCamera(1920, 1080, cam_height=2.0, pitch_deg=8.0)
    cfg = VEMAConfig(alpha=0.80, ref_fps=30.0, dt_aware=True, probation_frames=1)
    tracker = VEMA3DTracker(cfg, img_size=(1920, 1080))

    # Pedestrian approaching camera straight on from Z = 18 m at -1.5 m/s (-0.05 m/frame @ 30fps)
    pole_h, pole_w = 1.70, 0.60
    vz_mps = -1.5
    dt = 1.0 / 30.0
    ious = []
    heights = []

    for k in range(90):
        z_k = 18.0 + vz_mps * (k * dt)
        gt_box = cam.project_pole_to_bbox(np.array([0.4, z_k]), pole_h, pole_w)
        outs = tracker.update(gt_box[None, :], np.array([0.90]), dt=dt)
        assert len(outs) == 1
        pred_box = outs[0][1]
        ious.append(float(iou_matrix(pred_box[None, :], gt_box[None, :])[0, 0]))
        heights.append(float(pred_box[3] - pred_box[1]))

    trk = tracker.tracks[0]
    assert trk.v_3d[1] == pytest.approx(vz_mps / 30.0, abs=0.005)
    assert trk.pole_h == pytest.approx(pole_h, abs=0.02)
    # Tight 2D box output across looming trajectory (no systematic offset up to IoU 0.95+)
    assert min(ious) > 0.97
    # Height grows non-linearly as target approaches (18 m -> 13.55 m -> ~1.33x)
    assert heights[-1] > 1.25 * heights[0]


def test_vema3d_synthetic_stress_suite():
    seqs = generate_synthetic_suite(seed=42)
    cfg = VEMAConfig()
    for seq in seqs:
        tracker = get_tracker("vema3d", config=cfg)
        assert tracker.name == "vema3d"
        _, metrics, ms_pf = run_tracker_on_sequence("vema3d", seq, config=cfg)
        d = metrics.to_dict()
        assert 0.0 <= d["HOTA"] <= 100.0
        assert 0.0 <= d["AssA"] <= 100.0
        assert 0.0 <= d["IDF1"] <= 100.0
        assert np.isfinite(d["MOTA"])
        assert ms_pf >= 0.0
        if seq.name == "partial_occlusion_truncation":
            assert d["IDSW"] == 0
            assert d["IDF1"] >= 90.0
