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


def test_border_exit_eviction_and_shadow_cap_parity(tmp_path: Path):
    """Border-exit eviction and max_shadow_emit_lost must clear edge ghosts while keeping interior ID recovery."""
    cfg = VEMAConfig(
        probation_frames=1,
        max_lost_frames=50,
        emit_shadow_tracks=True,
        border_exit_lost=4,
        border_exit_margin_px=16.0,
        min_visible_frac=0.25,
        max_shadow_emit_lost=6,
    )
    yaml_path = tmp_path / "vema3d_border.yml"
    cfg.to_yaml(yaml_path)

    py_tracker = VEMA3DTracker(config=cfg, img_size=(1280, 720))
    lib_path = _find_tracker_lib()
    lib = _load_harness(lib_path) if lib_path is not None else None
    handle = lib.vema3d_test_create(str(yaml_path).encode("utf-8"), 1280, 720) if lib else None

    try:
        empty_b = np.empty((0, 4), dtype=np.float64)
        empty_s = np.empty(0, dtype=np.float64)
        empty_c = np.empty(0, dtype=np.int64)
        sc = np.array([0.9], dtype=np.float64)
        cls_arr = np.array([2], dtype=np.int64)

        def step_both(dets: np.ndarray, scores: np.ndarray, clss: np.ndarray):
            py_out = py_tracker.update(dets, scores, clss, dt=1.0 / 30.0)
            if lib is not None and handle is not None:
                cpp_out = _step_cpp(lib, handle, 1, dets, scores, clss, 1.0 / 30.0)
                assert len(cpp_out) == len(py_out)
                for (cid, cbox, _cconf, _ccls, csh), (pid, pbox, _pconf, _pcls, psh) in zip(cpp_out, py_out):
                    assert cid == pid and csh is psh
                    assert np.allclose(cbox, pbox, atol=0.1)
            return py_out

        # Track 1 drives to the left border; Track 2 stays in interior and is temporarily occluded
        for f in range(5):
            dets = np.array([
                [40.0 - 9.0 * f, 300.0, 140.0 - 9.0 * f, 400.0],
                [600.0, 320.0, 700.0, 420.0],
            ], dtype=np.float64)
            out = step_both(dets, np.array([0.9, 0.9]), np.array([2, 2], dtype=np.int64))
            assert len(out) == 2

        # Both lose detections for 8 frames:
        # - Border track (ID 1) must be evicted cleanly once k_lost > border_exit_lost (4).
        # - Interior track (ID 2) emits shadow up to k_lost <= 6, then hides shadow while staying in memory.
        saw_id1_evicted = False
        saw_id2_shadow_hidden = False
        for _ in range(8):
            out = step_both(empty_b, empty_s, empty_c)
            ids = {r[0] for r in out}
            if 1 not in ids:
                saw_id1_evicted = True
            if 2 not in ids:
                saw_id2_shadow_hidden = True
        assert saw_id1_evicted
        assert saw_id2_shadow_hidden

        # Interior track 2 reappears after shadow stopped emitting -> must still keep ID 2!
        reappear = step_both(np.array([[602.0, 320.0, 702.0, 420.0]], dtype=np.float64), sc, cls_arr)
        assert len(reappear) == 1 and reappear[0][0] == 2 and reappear[0][4] is False

        # Regime II un-occlusion recovery: track shrinks under occlusion, then un-occludes at conf=0.57
        # followed by high-conf=0.75 full box -> must keep ID 2 (never trap pole_h tiny).
        for y2 in (400.0, 396.0, 392.0):
            step_both(np.array([[602.0, 320.0, 660.0, y2]], dtype=np.float64), np.array([0.45]), cls_arr)
        step_both(np.array([[602.0, 320.0, 662.0, 542.0]], dtype=np.float64), np.array([0.57]), cls_arr)
        recovered = step_both(np.array([[604.0, 322.0, 665.0, 545.0]], dtype=np.float64), np.array([0.75]), cls_arr)
        assert len(recovered) == 1 and recovered[0][0] == 2 and recovered[0][4] is False
    finally:
        if lib is not None and handle is not None:
            lib.vema3d_test_destroy(handle)
