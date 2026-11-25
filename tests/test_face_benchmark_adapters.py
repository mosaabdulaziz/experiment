"""Unit tests for Phase 1 face-detector benchmark adapters & evaluation metrics.

Pure NumPy only (no cv2, onnxruntime, or DVC model bytes required) so this runs
on CI and developer machines without GPU dependencies.
"""
from __future__ import annotations

import numpy as np
import pytest

from benchmarks.bench_face_detectors import (
    compute_far_frr_curve,
    landmark_nme,
    match_detections,
    size_bucket,
)
from benchmarks.bench_face_hf_subsets import (
    parse_voc_xml_boxes,
    parse_wflw_csv_row,
    parse_wider_val_gt,
)
from benchmarks.face_bench_adapters import (
    ARCFACE_DST_112,
    FACEMESH_LEFT_EYE_PAIR,
    FACEMESH_MOUTH_LEFT,
    FACEMESH_MOUTH_RIGHT,
    FACEMESH_NOSE_TIP,
    FACEMESH_RIGHT_EYE_PAIR,
    decode_face_det_lite,
    decode_mediapipe_face,
    decode_rfdetr_face,
    decode_tao_detectnet_v2,
    decode_yolov8_face,
    decode_yunet,
    estimate_arcface_affine,
    extract_5pt_from_facemesh,
    face_yaw_from_5pt,
    generate_blazeface_back_anchors,
    landmarks_plausible,
    letterbox_params,
    nms_xyxy,
    unmap_coords,
)


def test_letterbox_and_unmap_roundtrip():
    scale, pad_x, pad_y = letterbox_params(src_w=200, src_h=100, dst_w=640, dst_h=480)
    assert scale == pytest.approx(3.2)
    assert pad_x == pytest.approx(0.0)
    assert pad_y == pytest.approx(80.0)

    net_pts = np.array([[0.0, 80.0], [640.0, 400.0]], dtype=np.float32)
    src_pts = unmap_coords(net_pts, scale, pad_x, pad_y, max_w=200.0, max_h=100.0)
    assert np.allclose(src_pts, [[0.0, 0.0], [200.0, 100.0]], atol=1e-4)


def test_nms_suppresses_overlapping_lower_score_box():
    boxes = np.array(
        [
            [10.0, 10.0, 50.0, 50.0],
            [12.0, 12.0, 52.0, 52.0],
            [100.0, 100.0, 140.0, 140.0],
        ],
        dtype=np.float32,
    )
    scores = np.array([0.6, 0.9, 0.7], dtype=np.float32)
    assert nms_xyxy(boxes, scores, iou_thresh=0.45) == [1, 2]


def test_decode_yolov8_face_15_and_20_channel_layouts():
    # 20-channel layout: x1,y1,x2,y2,conf + 5*(x,y,c)
    row = np.zeros((2, 20), dtype=np.float32)
    row[0, :5] = [100.0, 120.0, 180.0, 220.0, 0.92]
    row[0, 5:20] = [120, 150, 0.9, 160, 150, 0.9, 140, 175, 0.9, 125, 200, 0.9, 155, 200, 0.9]
    row[1, :5] = [10.0, 10.0, 20.0, 20.0, 0.05]  # below conf_thresh

    decoded = decode_yolov8_face(row[None, ...], conf_thresh=0.18, iou_thresh=0.45)
    assert len(decoded) == 1
    box, score, kpts = decoded[0]
    assert np.allclose(box, [100.0, 120.0, 180.0, 220.0])
    assert score == pytest.approx(0.92)
    assert kpts.shape == (5, 2) and np.allclose(kpts[2], [140.0, 175.0])

    # 15-channel layout: x1,y1,x2,y2,conf + 5*(x,y)
    row15 = np.zeros((1, 15), dtype=np.float32)
    row15[0, :5] = [20.0, 30.0, 60.0, 80.0, 0.85]
    row15[0, 5:15] = [30, 45, 50, 45, 40, 60, 32, 70, 48, 70]
    decoded15 = decode_yolov8_face(row15, conf_thresh=0.18)
    assert len(decoded15) == 1
    assert np.allclose(decoded15[0][2][0], [30.0, 45.0])


def test_decode_face_det_lite_stride_8_peak_extraction():
    hm_h, hm_w = 60, 80  # 480x640 at stride 8
    hm = np.full((1, 1, hm_h, hm_w), -10.0, dtype=np.float32)
    box = np.zeros((1, 4, hm_h, hm_w), dtype=np.float32)
    lmk = np.zeros((1, 10, hm_h, hm_w), dtype=np.float32)

    cx, cy = 20, 15
    hm[0, 0, cy, cx] = 4.0       # sigmoid(4.0) ~ 0.982
    hm[0, 0, cy, cx + 1] = 1.0   # neighbor peak suppressed by 3x3 maxpool
    box[0, :, cy, cx] = [3.0, 4.0, 3.0, 5.0]  # box = (17*8, 11*8, 23*8, 20*8) = (136, 88, 184, 160)
    lmk[0, :5, cy, cx] = [-1.5, 1.5, 0.0, -1.0, 1.0]
    lmk[0, 5:, cy, cx] = [-1.0, -1.0, 0.5, 2.5, 2.5]

    decoded = decode_face_det_lite(hm, box, lmk, conf_thresh=0.55, nms_iou=0.45, stride=8)
    assert len(decoded) == 1
    b, score, kpts = decoded[0]
    assert score > 0.98
    assert np.allclose(b, [136.0, 88.0, 184.0, 160.0])
    assert np.allclose(kpts[0], [(20 - 1.5) * 8, (15 - 1.0) * 8])
    assert np.allclose(kpts[2], [20 * 8, 15.5 * 8])


def test_blazeface_back_anchors_and_mediapipe_decode():
    anchors = generate_blazeface_back_anchors()
    assert anchors.shape == (896, 4)
    assert np.allclose(anchors[0], [0.03125, 0.03125, 1.0, 1.0])
    assert np.allclose(anchors[512], [0.0625, 0.0625, 1.0, 1.0])

    coords = np.zeros((1, 896, 16), dtype=np.float32)
    scores = np.full((1, 896, 1), -10.0, dtype=np.float32)
    # Anchor 512 center is (0.0625 * 256, 0.0625 * 256) = (16.0, 16.0).
    scores[0, 512, 0] = 5.0
    coords[0, 512, 0:4] = [2.0, 4.0, 40.0, 40.0]  # cx=18, cy=20, w=40, h=40 -> [-2, 0, 38, 40]
    coords[0, 512, 4:6] = [-4.0, -4.0]            # kp0 -> (12, 12)

    decoded = decode_mediapipe_face(coords, scores, anchors, img_size=(256, 256), conf_thresh=0.5)
    assert len(decoded) == 1
    box, score, kpts = decoded[0]
    assert score > 0.99
    assert np.allclose(box, [-2.0, 0.0, 38.0, 40.0])
    assert np.allclose(kpts[0], [12.0, 12.0])


def test_extract_5pt_from_facemesh_orders_viewer_left_to_right():
    mesh = np.zeros((468, 3), dtype=np.float32)
    mesh[FACEMESH_LEFT_EYE_PAIR[0], :2] = [30.0, 40.0]
    mesh[FACEMESH_LEFT_EYE_PAIR[1], :2] = [46.0, 40.0]
    mesh[FACEMESH_RIGHT_EYE_PAIR[0], :2] = [70.0, 40.0]
    mesh[FACEMESH_RIGHT_EYE_PAIR[1], :2] = [78.0, 40.0]
    mesh[FACEMESH_NOSE_TIP, :2] = [56.0, 72.0]
    mesh[FACEMESH_MOUTH_LEFT, :2] = [42.0, 92.0]
    mesh[FACEMESH_MOUTH_RIGHT, :2] = [70.0, 92.0]

    pts = extract_5pt_from_facemesh(mesh)
    assert pts.shape == (5, 2)
    assert np.allclose(pts[0], [38.0, 40.0])
    assert np.allclose(pts[1], [74.0, 40.0])
    assert pts[0, 0] < pts[1, 0] and pts[3, 0] < pts[4, 0]


def test_umeyama_arcface_affine_inverts_known_similarity_transform():
    theta = np.deg2rad(15.0)
    scale = 1.8
    tx, ty = 120.0, 80.0
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float64)
    src = (scale * (ARCFACE_DST_112.astype(np.float64) @ rot.T)) + np.array([tx, ty])

    mat = estimate_arcface_affine(src, image_size=112)
    src_h = np.hstack([src, np.ones((5, 1))])
    mapped = src_h @ mat.T
    assert np.allclose(mapped, ARCFACE_DST_112, atol=1e-3)


def test_plausibility_yaw_and_evaluation_metrics():
    kpts = ARCFACE_DST_112.copy()
    assert landmarks_plausible(kpts, (20.0, 20.0, 95.0, 105.0))
    assert face_yaw_from_5pt(kpts) < 0.05

    # Inverted eyes below mouth -> implausible
    bad = kpts.copy()
    bad[0, 1] = 100.0
    assert not landmarks_plausible(bad, (20.0, 20.0, 95.0, 105.0))

    assert size_bucket(16.0) == "<24px"
    assert size_bucket(28.0) == "24-35px"
    assert size_bucket(48.0) == "35-64px"
    assert size_bucket(80.0) == ">64px"

    matched, unm_a, unm_c = match_detections(
        [(0, 0, 20, 20), (100, 100, 130, 130)],
        [(1, 1, 21, 21), (200, 200, 220, 220)],
        iou_thresh=0.4,
    )
    assert len(matched) == 1 and matched[0][:2] == (0, 0)
    assert unm_a == [1] and unm_c == [1]

    nme = landmark_nme(kpts, kpts + 1.0)
    assert nme > 0.0


def test_compute_far_frr_curve_with_open_set_margin():
    g_ids = ["emp1", "emp2"]
    g_embs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    # Probe 0: clearly emp1; Probe 1: ambiguous tie between emp1 and emp2 -> rejected by margin
    # Probe 2: unknown impostor slightly matching emp1 at 0.2
    p_ids = ["emp1", "emp2", "impostor"]
    p_embs = np.array(
        [
            [0.95, 0.05, 0.0],
            [0.50, 0.51, 0.0],
            [0.20, 0.00, 0.98],
        ],
        dtype=np.float32,
    )
    res = compute_far_frr_curve(g_ids, g_embs, p_ids, p_embs, thresholds=(0.15, 0.30), margin=0.05)
    pts = {p["threshold"]: p for p in res["operating_points"]}
    # At 0.15: emp1 accepted, emp2 rejected by margin (FR), impostor accepted (FA, sim ~0.20)
    assert pts[0.15]["false_accepts"] == 1
    assert pts[0.15]["false_rejects"] == 1
    # At 0.30: impostor rejected -> FA=0, emp2 still rejected by margin -> FR=1
    assert pts[0.30]["false_accepts"] == 0
    assert pts[0.30]["false_rejects"] == 1


def test_decode_yolov8_face_21col_and_5ch_layouts():
    # 21-column end2end layout: x1,y1,x2,y2,conf,cls + 5*(x,y,vis)
    row21 = np.zeros((1, 21), dtype=np.float32)
    row21[0, :6] = [10.0, 20.0, 90.0, 120.0, 0.91, 0.0]
    row21[0, 6:21] = [30, 50, 0.9, 70, 50, 0.9, 50, 75, 0.9, 35, 95, 0.9, 65, 95, 0.9]
    d21 = decode_yolov8_face(row21, conf_thresh=0.18)
    assert len(d21) == 1
    assert np.allclose(d21[0][0], [10.0, 20.0, 90.0, 120.0])
    assert d21[0][2] is not None and np.allclose(d21[0][2][2], [50.0, 75.0])

    # [5, N] channel-first box-only layout: cx, cy, w, h, conf
    ch5 = np.zeros((5, 3), dtype=np.float32)
    ch5[:, 0] = [100.0, 120.0, 40.0, 50.0, 0.85]
    d5 = decode_yolov8_face(ch5[None, ...], conf_thresh=0.18)
    assert len(d5) == 1
    assert np.allclose(d5[0][0], [80.0, 95.0, 120.0, 145.0])
    assert d5[0][2] is None


def test_hf_dataset_parsers_wider_wflw_and_voc():
    wider_txt = (
        "0--Parade/a.jpg\n"
        "2\n"
        "10 20 30 40 2 0 0 0 1 1\n"
        "50 60 15 15 0 0 0 1 0 0\n"
        "1--Empty/b.jpg\n"
        "0\n"
        "0 0 0 0 0 0 0 0 0 0\n"
    )
    parsed_w = parse_wider_val_gt(wider_txt)
    assert len(parsed_w["0--Parade/a.jpg"]) == 1  # invalid==1 row filtered out
    assert parsed_w["0--Parade/a.jpg"][0]["bbox"] == (10.0, 20.0, 40.0, 60.0)
    assert parsed_w["0--Parade/a.jpg"][0]["pose"] == 1
    assert parsed_w["1--Empty/b.jpg"] == []

    cols = ["0.0"] * 206
    cols[54 * 2], cols[54 * 2 + 1] = "0.50", "0.55"
    cols[76 * 2], cols[76 * 2 + 1] = "0.40", "0.75"
    cols[82 * 2], cols[82 * 2 + 1] = "0.60", "0.75"
    cols[96 * 2], cols[96 * 2 + 1] = "0.35", "0.40"
    cols[97 * 2], cols[97 * 2 + 1] = "0.65", "0.40"
    cols[196:202] = ["1", "0", "0", "0", "1", "0"]
    cols[202:205] = ["12.5", "-3.0", "1.0"]
    cols[205] = "/abs/path/sample_01.png"
    name, k5, attrs, euler = parse_wflw_csv_row(",".join(cols))
    assert name == "sample_01.png"
    assert np.allclose(k5[0], [0.35, 0.40]) and np.allclose(k5[2], [0.50, 0.55])
    assert attrs == [1, 0, 0, 0, 1, 0] and euler == (12.5, -3.0, 1.0)

    xml = (
        "<annotation><object><name>with_mask</name>"
        "<bndbox><xmin>10</xmin><ymin>20</ymin><xmax>50</xmax><ymax>80</ymax></bndbox>"
        "</object></annotation>"
    )
    assert parse_voc_xml_boxes(xml) == [("with_mask", (10.0, 20.0, 50.0, 80.0))]


def test_decode_tao_peoplenet_yunet_and_rfdetr():
    # TAO DetectNet_v2 (class 2 = face)
    cov = np.zeros((1, 3, 4, 6), dtype=np.float32)
    bbox = np.zeros((1, 12, 4, 6), dtype=np.float32)
    cov[0, 2, 1, 2] = 0.85
    bbox[0, 8:12, 1, 2] = [0.5, 0.5, 0.5, 0.5]  # cx=32.5, cy=16.5; half=17.5 -> [15, -1, 50, 34]
    tao_res = decode_tao_detectnet_v2(cov, bbox, class_idx=2, conf_thresh=0.2)
    assert len(tao_res) == 1
    assert np.allclose(tao_res[0][0], [15.0, -1.0, 50.0, 34.0])

    # YuNet multi-stride
    yn = {
        "cls_8": np.zeros((1, 6400, 1), dtype=np.float32),
        "obj_8": np.zeros((1, 6400, 1), dtype=np.float32),
        "bbox_8": np.zeros((1, 6400, 4), dtype=np.float32),
        "kps_8": np.zeros((1, 6400, 10), dtype=np.float32),
        "cls_16": np.zeros((1, 1600, 1), dtype=np.float32),
        "obj_16": np.zeros((1, 1600, 1), dtype=np.float32),
        "bbox_16": np.zeros((1, 1600, 4), dtype=np.float32),
        "kps_16": np.zeros((1, 1600, 10), dtype=np.float32),
        "cls_32": np.zeros((1, 400, 1), dtype=np.float32),
        "obj_32": np.zeros((1, 400, 1), dtype=np.float32),
        "bbox_32": np.zeros((1, 400, 4), dtype=np.float32),
        "kps_32": np.zeros((1, 400, 10), dtype=np.float32),
    }
    yn["cls_32"][0, 21, 0] = 0.81  # gx=1, gy=1 at grid_w=20
    yn["obj_32"][0, 21, 0] = 1.0
    yn["bbox_32"][0, 21] = [0.5, 0.5, 0.0, 0.0]  # cx=48, cy=48, bw=32, bh=32 -> [32, 32, 64, 64]
    yn["kps_32"][0, 21, :2] = [0.25, 0.25]  # -> [40, 40]
    yn_res = decode_yunet(yn, conf_thresh=0.35)
    assert len(yn_res) == 1
    assert np.allclose(yn_res[0][0], [32.0, 32.0, 64.0, 64.0], atol=1e-4)
    assert np.allclose(yn_res[0][2][0], [40.0, 40.0], atol=1e-4)

    # RF-DETR cxcywh + logits
    b_rf = np.array([[[0.5, 0.5, 0.25, 0.5]]], dtype=np.float32)
    l_rf = np.array([[[3.0, -5.0]]], dtype=np.float32)
    rf_res = decode_rfdetr_face(b_rf, l_rf, img_wh=(400, 200), face_col=0, conf_thresh=0.5)
    assert len(rf_res) == 1
    assert np.allclose(rf_res[0][0], [150.0, 50.0, 250.0, 150.0])
