"""perception/models/staff/staff_dinov2_s.onnx (project A), staff_dinov2_s_palm.onnx (Palm Hills, 2026-10-21)
and staff_dinov2_s_palm_r3.onnx (its 2026-10-14 predecessor, just barber-palm-staff r3): each committed staff
classifier loads, takes batches of 1 and 5, returns P(staff), and its metadata matches the preprocessing in
perception/modules/appearance/worker.py and its own bytes. Needs onnxruntime: run with
/data/IRIS/.venv-staff/bin/python."""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ort = pytest.importorskip("onnxruntime")
from perception.modules.appearance import worker as ap  # noqa: E402

STAFF = Path(__file__).resolve().parents[1] / "perception/models/staff"
MODELS = [STAFF / f for f in ("staff_dinov2_s.onnx", "staff_dinov2_s_palm.onnx", "staff_dinov2_s_palm_r3.onnx")]
pytestmark = pytest.mark.skipif(not MODELS[0].exists(), reason="staff classifier not in this checkout")
each_model = pytest.mark.parametrize("model", MODELS, ids=lambda m: m.stem)


def meta(model):
    return json.loads(model.with_suffix(".json").read_text())


@each_model
def test_metadata_matches_the_sidecar_preprocessing(model):
    m = meta(model)
    assert tuple(m["input_hw"]) == ap.CLASSIFIER_HW
    assert np.allclose(m["mean"], ap.IMAGENET_MEAN) and np.allclose(m["std"], ap.IMAGENET_STD)
    assert (m["resize"], m["interpolation"], m["color"]) == ("stretch", "area", "rgb")


@each_model
def test_metadata_sha256_is_the_files(model):
    assert hashlib.sha256(model.read_bytes()).hexdigest() == meta(model)["sha256"]


@each_model
@pytest.mark.parametrize("n", [1, 5])
def test_model_returns_one_probability_per_crop(model, n):
    sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    crops = [rng.integers(0, 256, (int(rng.integers(40, 400)), int(rng.integers(20, 200)), 3), dtype=np.uint8)
             for _ in range(n)]
    out = sess.run(["staff_p"], {"input": ap.classifier_batch(crops)})[0]
    assert out.shape == (n, 1) and out.dtype == np.float32
    assert np.all((out >= 0) & (out <= 1))
