"""The vendor's test-suite wiring, copied into the product (ruling A10).

The vendor's own tests are byte-identical to the vendor's; only this file is rewritten, because
it is the wiring: the vendor's checkout laid its code out as <root>/src and <root>/tests, and
the product's layout is <product>/fall_detection and <product>/tests/vendor.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

PRODUCT = Path(__file__).resolve().parents[2]
COPY = PRODUCT / "fall_detection"
HERE = Path(__file__).resolve().parent

for _p in (str(PRODUCT), str(COPY), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# One vendor test imports `src.omnifall_adapter`, so the copy answers to the vendor's package name.
_spec = importlib.util.spec_from_file_location("src", COPY / "__init__.py",
                                               submodule_search_locations=[str(COPY)])
_src = importlib.util.module_from_spec(_spec)
sys.modules["src"] = _src
_spec.loader.exec_module(_src)

WEIGHT_DIR = PRODUCT / "weights"
CLASS_NAMES = ('Standing', 'Walking', 'Sitting', 'Lying Down', 'Stand up', 'Sit down', 'Fall Down')
STAND_, WALK_, SITTING_, LYING_, STANDUP_, SITDOWN_, FALL_ = range(7)


def onehot7(idx, mass=0.9):
    p = np.full(7, (1.0 - mass) / 6.0, dtype=np.float64)
    p[idx] = mass
    return p


class FakeClassifier:
    """Stands in for TSSTGClassifier: callable(stream_tensors) -> (pred_idx_7, name_7, probs_7, ntu60_probs)."""
    streams = ('j', 'jm')

    def __init__(self, probs7=None):
        self.probs7 = onehot7(STAND_) if probs7 is None else np.asarray(probs7, dtype=np.float64)
        self.calls = 0

    def __call__(self, stream_tensors):
        self.calls += 1
        for name, x in stream_tensors.items():
            assert tuple(x.shape) == (1, 2, 100, 17, 3), (name, tuple(x.shape))
        p = self.probs7
        ntu = np.zeros(60); ntu[42] = p[FALL_]; ntu[7] = p[SITDOWN_]; ntu[10] = p[SITTING_]
        i = int(p.argmax())
        return i, CLASS_NAMES[i], p, ntu


@pytest.fixture
def fake_cls():
    return FakeClassifier()


def drive(mon, fake, t, kps, boxes, probs_fn=None, tid=1, frames=None):
    """Feed one track through FallMonitor.update and return every emitted event."""
    events = []
    for i in range(len(t)):
        if probs_fn is not None:
            fake.probs7 = probs_fn(float(t[i]))
        fr = None if frames is None else frames[i]
        events += mon.update(float(t[i]), [tid], [boxes[i]], [kps[i]], frame=fr) if fr is not None \
            else mon.update(float(t[i]), [tid], [boxes[i]], [kps[i]])
    return events


# ── the tests that need the pinned checkpoints skip cleanly when they are absent (AC-N8.1) ──────────────

_CHECKPOINTS = {
    ("test_tsstg_model.py", "test_coco_17_graph_matches_checkpoint_buffer_before_load"):
        ("mmaction2_joint.pth", "mmaction2_jm.pth", "pyskl_j.pth", "pyskl_b.pth"),
    ("test_tsstg_model.py", "test_mmaction2_and_pyskl_checkpoints_load_strictly_with_sha256"):
        ("mmaction2_joint.pth", "mmaction2_jm.pth", "mmaction2_bone.pth", "pyskl_j.pth", "pyskl_jm.pth"),
    ("test_tsstg_model.py", "test_tampered_checkpoint_is_rejected"):
        ("mmaction2_joint.pth",),
    ("test_transfer_gate.py", "test_transfer_gate_evaluates_norm_modes_and_finetune_step"):
        ("mmaction2_joint.pth",),
    ("test_transfer_gate.py", "test_transfer_gate_real_weights_fall_and_sit"):
        ("mmaction2_joint.pth", "mmaction2_jm.pth"),
    ("test_tsstg_model.py", "test_mmaction2_tsstg_discriminates_fall_sitdown_and_pickup_on_real_weights"):
        ("mmaction2_joint.pth", "mmaction2_jm.pth"),
}


def _checkpoints_for(item):
    # a parametrized test's name carries its id, so the original name is the one to match on
    key = (Path(str(item.fspath)).name, getattr(item, "originalname", None) or item.name)
    if key in _CHECKPOINTS:
        return _CHECKPOINTS[key]
    if "real_weights" in item.keywords:
        return ("mmaction2_joint.pth", "mmaction2_jm.pth")
    return ()


def pytest_collection_modifyitems(config, items):
    for item in items:
        needed = _checkpoints_for(item)
        if not needed:
            continue
        missing = [name for name in needed if not (WEIGHT_DIR / name).is_file()]
        if missing:
            item.add_marker(pytest.mark.skip(
                reason=f"the pinned ST-GCN checkpoints are absent: {', '.join(missing)}"))
