"""Unit and ONNX-parity tests for perception/models/DeepStream-Yolo-Face/export_face_det_lite.py."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from benchmarks.face_bench_adapters import (  # noqa: E402
    QualcommFaceDetLiteDetector,
    decode_face_det_lite,
    decode_yolov8_face,
)

REPO = Path(__file__).resolve().parents[1]
EXPORT_SCRIPT = REPO / "perception" / "models" / "DeepStream-Yolo-Face" / "export_face_det_lite.py"
ONNX_PATH = REPO / "perception" / "models" / "DeepStream-Yolo-Face" / "face_det_lite.onnx"


def _load_exporter():
    spec = importlib.util.spec_from_file_location("export_face_det_lite", EXPORT_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_decode_maps_to_yolo_face_matches_reference_decoder():
    exp = _load_exporter()
    hm_h, hm_w = 60, 80
    hm = torch.zeros((2, 1, hm_h, hm_w), dtype=torch.float32)
    box = torch.zeros((2, 4, hm_h, hm_w), dtype=torch.float32)
    lmk = torch.zeros((2, 10, hm_h, hm_w), dtype=torch.float32)

    cx, cy = 25, 18
    hm[0, 0, cy, cx] = 0.91
    hm[0, 0, cy, cx + 1] = 0.72  # 8-neighbor suppressed by 3x3 maxpool
    box[0, :, cy, cx] = torch.tensor([2.5, 3.0, 2.5, 4.0])
    lmk[0, :5, cy, cx] = torch.tensor([-1.0, 1.0, 0.0, -0.75, 0.75])
    lmk[0, 5:, cy, cx] = torch.tensor([-0.5, -0.5, 0.25, 1.5, 1.5])

    hm[1, 0, 10, 12] = 0.64
    box[1, :, 10, 12] = torch.tensor([1.0, 1.0, 2.0, 2.0])
    lmk[1, :5, 10, 12] = torch.tensor([-0.5, 0.5, 0.0, -0.4, 0.4])
    lmk[1, 5:, 10, 12] = torch.tensor([-0.2, -0.2, 0.1, 0.8, 0.8])

    gy, gx = torch.meshgrid(
        torch.arange(hm_h, dtype=torch.float32),
        torch.arange(hm_w, dtype=torch.float32),
        indexing="ij",
    )
    out = exp.decode_maps_to_yolo_face(hm, box, lmk, gx.view(1, 1, hm_h, hm_w), gy.view(1, 1, hm_h, hm_w), stride=8)
    assert out.shape == (2, 4800, 15)

    # Non-peak neighbor cell must be zeroed out.
    neighbor_idx = cy * hm_w + (cx + 1)
    assert float(out[0, neighbor_idx, 4]) == 0.0

    for b in range(2):
        ref = decode_face_det_lite(
            hm[b : b + 1].numpy(),
            box[b : b + 1].numpy(),
            lmk[b : b + 1].numpy(),
            conf_thresh=0.25,
            nms_iou=0.45,
            stride=8,
            hm_is_logits=False,
        )
        got = decode_yolov8_face(out[b].numpy(), conf_thresh=0.25, iou_thresh=0.45)
        assert len(got) == len(ref) == 1
        assert np.allclose(got[0][0], ref[0][0], atol=1e-4)
        assert got[0][1] == pytest.approx(ref[0][1], abs=1e-6)
        assert np.allclose(got[0][2], ref[0][2], atol=1e-4)


def test_decode_maps_to_yolo_face_suppresses_near_ties_and_degenerate_boxes():
    exp = _load_exporter()
    hm_h, hm_w = 60, 80
    hm = torch.zeros((1, 1, hm_h, hm_w), dtype=torch.float32)
    box = torch.zeros((1, 4, hm_h, hm_w), dtype=torch.float32)
    lmk = torch.zeros((1, 10, hm_h, hm_w), dtype=torch.float32)

    # Near-tie neighbor (diff = 5e-7) must be zeroed by exact hm == pooled check.
    hm[0, 0, 20, 20] = 0.80
    hm[0, 0, 20, 21] = 0.80 - 5e-7
    box[0, :, 20, 20] = torch.tensor([2.0, 2.0, 2.0, 2.0])
    box[0, :, 20, 21] = torch.tensor([2.0, 2.0, 2.0, 2.0])

    # Standalone peak with degenerate (< 1px) box must also be zeroed so C++ addBBoxProposal never skips init.
    hm[0, 0, 40, 40] = 0.90
    box[0, :, 40, 40] = torch.tensor([0.02, 0.02, 0.02, 0.02])  # width = height = 0.32px < 1.0px

    gy, gx = torch.meshgrid(
        torch.arange(hm_h, dtype=torch.float32),
        torch.arange(hm_w, dtype=torch.float32),
        indexing="ij",
    )
    out = exp.decode_maps_to_yolo_face(hm, box, lmk, gx.view(1, 1, hm_h, hm_w), gy.view(1, 1, hm_h, hm_w), stride=8)
    assert float(out[0, 20 * hm_w + 20, 4]) == pytest.approx(0.80, abs=1e-6)
    assert float(out[0, 20 * hm_w + 21, 4]) == 0.0
    assert float(out[0, 40 * hm_w + 40, 4]) == 0.0


def test_wrapper_bt601_frontend_and_static_output_shape():
    exp = _load_exporter()
    core = exp.FaceDetLite(phase="test").eval()
    wrapper = exp.FaceDetLiteDeployWrapper(core).eval()

    weights = wrapper.rgb_to_gray.weight.detach().numpy().reshape(-1)
    assert np.allclose(weights, [0.299, 0.587, 0.114], atol=1e-6)
    assert not wrapper.rgb_to_gray.weight.requires_grad
    assert wrapper.rgb_to_gray.bias is None

    with torch.no_grad():
        out = wrapper(torch.zeros((2, 3, 480, 640), dtype=torch.float32))
    assert out.shape == (2, 4800, 15)


def test_exported_onnx_multibatch_and_adapter_parity():
    if not ONNX_PATH.is_file():
        pytest.skip(f"{ONNX_PATH} not present on this checkout")
    ort = pytest.importorskip("onnxruntime")

    assert not ONNX_PATH.with_suffix(".onnx.data").exists()

    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    out_meta = sess.get_outputs()[0]
    assert inp.name == "images" and inp.shape[1:] == [3, 480, 640]
    assert out_meta.name == "output" and out_meta.shape[1:] == [4800, 15]

    rng = np.random.default_rng(7)
    img_rgb = rng.random((1, 3, 480, 640), dtype=np.float32)
    b1 = sess.run(None, {inp.name: img_rgb})[0]
    b4 = sess.run(None, {inp.name: np.repeat(img_rgb, 4, axis=0)})[0]
    assert b1.shape == (1, 4800, 15)
    assert b4.shape == (4, 4800, 15)
    assert float(np.abs(b4 - np.repeat(b1, 4, axis=0)).max()) <= 1e-5

    # Adapter handles 3-ch deployed ONNX cleanly.
    det = QualcommFaceDetLiteDetector(ONNX_PATH, gpu=False, default_conf=0.25)
    assert det.channels == 3 and (det.dst_h, det.dst_w) == (480, 640)
    bgr_u8 = (img_rgb[0].transpose(1, 2, 0)[:, :, ::-1] * 255.0).astype(np.uint8)
    assert isinstance(det.detect(bgr_u8), list)

    sample = REPO / "scratch" / "face_eval_tmp" / "datasets" / "wider_val_imgs" / "0--Parade_0_Parade_marchingband_1_1004.jpg"
    if sample.is_file():
        exp = _load_exporter()
        report = exp.check(ONNX_PATH, sample)
        assert report["ok"] is True
        assert report["max_box_err_px"] <= 1.0
        assert report["max_landmark_err_px"] <= 1.0
        assert report["max_score_err"] <= 1e-4
