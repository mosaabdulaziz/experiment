"""C++ <-> Python parity test for libnvds_vema3d_tracker via ctypes."""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from vema_tracker.core.config import VEMAConfig
from vema_tracker.core.vema3d import VEMA3DTracker
from vema_tracker.datasets.synthetic import generate_synthetic_suite

REPO = Path(__file__).resolve().parents[1]


def _find_tracker_lib() -> Path | None:
    candidates = []
    env_lib = os.environ.get("VEMA3D_TRACKER_LIB")
    if env_lib:
        candidates.append(Path(env_lib))
    candidates.extend([
        Path("/opt/nvidia/deepstream/deepstream/custom_trackers/libnvds_vema3d_tracker.so"),
        REPO / "perception" / "trackers" / "vema3d" / "libnvds_vema3d_tracker.so",
    ])
    if sys.platform == "win32":
        candidates.append(REPO / "perception" / "trackers" / "vema3d" / "libnvds_vema3d_tracker.dll")
    for path in candidates:
        if path.is_file():
            return path
    return None


def _load_harness(lib_path: Path):
    lib = ctypes.CDLL(str(lib_path))
    lib.vema3d_test_create.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    lib.vema3d_test_create.restype = ctypes.c_void_p

    lib.vema3d_test_step.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
    ]
    lib.vema3d_test_step.restype = ctypes.c_int

    lib.vema3d_test_remove_stream.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    lib.vema3d_test_remove_stream.restype = ctypes.c_int

    lib.vema3d_test_destroy.argtypes = [ctypes.c_void_p]
    lib.vema3d_test_destroy.restype = None
    return lib


def _step_cpp(lib, handle, stream_id: int, dets: np.ndarray, scores: np.ndarray, clss: np.ndarray, dt: float):
    max_out = 256
    out_ids = (ctypes.c_uint64 * max_out)()
    out_boxes = (ctypes.c_double * (max_out * 4))()
    out_confs = (ctypes.c_double * max_out)()
    out_clss = (ctypes.c_int64 * max_out)()
    out_shadow = (ctypes.c_int * max_out)()

    n_det = len(dets)
    if n_det > 0:
        det_ptr = dets.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        sc_ptr = scores.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        cls_ptr = clss.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
    else:
        det_ptr = ctypes.POINTER(ctypes.c_double)()
        sc_ptr = ctypes.POINTER(ctypes.c_double)()
        cls_ptr = ctypes.POINTER(ctypes.c_int64)()

    n_cpp = lib.vema3d_test_step(
        handle,
        ctypes.c_uint64(stream_id),
        det_ptr,
        sc_ptr,
        cls_ptr,
        n_det,
        dt,
        out_ids,
        out_boxes,
        out_confs,
        out_clss,
        out_shadow,
        max_out,
    )
    results = []
    for i in range(n_cpp):
        box = np.array([out_boxes[4 * i + k] for k in range(4)], dtype=np.float64)
        results.append((int(out_ids[i]), box, float(out_confs[i]), int(out_clss[i]), bool(out_shadow[i])))
    return results


@pytest.mark.parametrize("emit_shadow", [False, True])
def test_cpp_python_parity_across_synthetic_suite(tmp_path: Path, emit_shadow: bool):
    lib_path = _find_tracker_lib()
    if lib_path is None:
        pytest.skip("libnvds_vema3d_tracker.so is not built in this environment")

    lib = _load_harness(lib_path)
    cfg = VEMAConfig(emit_shadow_tracks=emit_shadow)
    yaml_path = tmp_path / f"vema3d_parity_{emit_shadow}.yml"
    cfg.to_yaml(yaml_path)

    for seq_idx, seq in enumerate(generate_synthetic_suite(seed=42)):
        py_tracker = VEMA3DTracker(config=cfg, img_size=(seq.width, seq.height))
        handle = lib.vema3d_test_create(str(yaml_path).encode("utf-8"), seq.width, seq.height)
        assert handle
        try:
            for f in range(1, seq.num_frames + 1):
                f_dets = seq.detections.get(f, [])
                if f_dets:
                    det_arr = np.asarray([d.bbox for d in f_dets], dtype=np.float64).reshape(-1, 4)
                    score_arr = np.asarray([d.conf for d in f_dets], dtype=np.float64)
                    cls_arr = np.asarray([d.cls_id for d in f_dets], dtype=np.int64)
                else:
                    det_arr = np.empty((0, 4), dtype=np.float64)
                    score_arr = np.empty(0, dtype=np.float64)
                    cls_arr = np.empty(0, dtype=np.int64)

                dt = seq.frame_dts.get(f, 1.0 / seq.fps) if seq.frame_dts else (1.0 / seq.fps)
                py_out = py_tracker.update(det_arr, score_arr, cls_arr, dt=dt)
                cpp_out = _step_cpp(lib, handle, seq_idx + 1, det_arr, score_arr, cls_arr, dt)

                assert len(cpp_out) == len(py_out), f"{seq.name} frame {f}: cpp={len(cpp_out)} != py={len(py_out)}"
                for (cid, cbox, cconf, ccls, csh), (pid, pbox, pconf, pcls, psh) in zip(cpp_out, py_out):
                    assert cid == pid, f"{seq.name} frame {f}"
                    assert ccls == pcls
                    assert csh is psh
                    assert cconf == pytest.approx(pconf, abs=1e-9)
                    assert np.allclose(cbox, pbox, atol=0.05), (
                        f"{seq.name} frame {f} id={pid}: cpp={cbox} vs py={pbox}"
                    )

            assert lib.vema3d_test_remove_stream(handle, ctypes.c_uint64(seq_idx + 1)) == 1
            assert lib.vema3d_test_remove_stream(handle, ctypes.c_uint64(seq_idx + 1)) == 0
        finally:
            lib.vema3d_test_destroy(handle)


def test_cpp_multistream_isolation_and_removal(tmp_path: Path):
    lib_path = _find_tracker_lib()
    if lib_path is None:
        pytest.skip("libnvds_vema3d_tracker.so is not built in this environment")

    lib = _load_harness(lib_path)
    cfg_path = REPO / "perception" / "pipeline" / "config_tracker_vema3d_proximity.yml"
    handle = lib.vema3d_test_create(str(cfg_path).encode("utf-8"), 1280, 720)
    assert handle
    try:
        box0 = np.array([[200.0, 200.0, 280.0, 360.0]], dtype=np.float64)
        box1 = np.array([[800.0, 250.0, 900.0, 420.0]], dtype=np.float64)
        sc = np.array([0.9], dtype=np.float64)
        cls_car = np.array([2], dtype=np.int64)
        cls_truck = np.array([7], dtype=np.int64)

        out0 = _step_cpp(lib, handle, 0, box0, sc, cls_car, 1.0 / 30.0)
        out1 = _step_cpp(lib, handle, 1, box1, sc, cls_truck, 1.0 / 30.0)
        assert len(out0) == 1 and out0[0][0] == 1
        assert len(out1) == 1 and out1[0][0] == 1

        # Remove stream 0; stream 1 must remain intact and keep ID 1 across car<->truck flip
        assert lib.vema3d_test_remove_stream(handle, ctypes.c_uint64(0)) == 1
        out1_next = _step_cpp(lib, handle, 1, box1, sc, cls_car, 1.0 / 30.0)
        assert len(out1_next) == 1 and out1_next[0][0] == 1
    finally:
        lib.vema3d_test_destroy(handle)
