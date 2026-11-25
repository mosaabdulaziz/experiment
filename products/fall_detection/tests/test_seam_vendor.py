"""The vendored copy and its one additive edit (the seam).

ACs: AC-F7.1, AC-F8.1, AC-F8.2, AC-F9.1, AC-N8.1.
Design: FALL_DETECTION_DESIGN.md §4.1-4.2; rulings A4 (the physical score at packaging time),
A5 (only the IRIS copy is edited), A10 (the copied vendor tests are wiring).

The manifest (`tests/pristine_manifest.json`) pins every copied file's text and the exact seam
edits, so the faithfulness check runs on the deployment server too -- the pristine checkout is
not needed there. The parity run (pristine vs copy on one fixture) still needs that checkout and
torch/pandas, and skips where they are absent.
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

VENDOR_ROOT = Path(r"F:\fall_detection_module")
VENDOR_SRC = VENDOR_ROOT / "src"
VENDOR_TESTS = VENDOR_ROOT / "tests"
HERE = Path(__file__).resolve().parent
PRODUCT = HERE.parent
COPY_DIR = PRODUCT / "fall_detection"
MANIFEST = HERE / "pristine_manifest.json"
VENDOR_FIXTURES = HERE / "vendor"

needs_vendor = pytest.mark.skipif(not VENDOR_SRC.is_dir(),
                                  reason="the pristine vendor checkout is not on this machine")


def _need_torch():
    """The monitor runs need the product venv (torch/pandas); the file-text checks do not."""
    pytest.importorskip("torch", reason="the vendor package needs torch (product venv, slice S2)")
    pytest.importorskip("pandas", reason="the vendor package needs pandas (product venv, slice S2)")


def _text(path: Path) -> str:
    """The file's text with CRLF normalised to LF (both trees are checked out with CRLF)."""
    return path.read_bytes().replace(b"\r\n", b"\n").decode("utf-8")


def _load(name: str, path: Path):
    """Load a module by path under its own name, so no `conftest` in sys.modules can answer for it."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _pristine_monitor():
    """The vendor's fall_monitor, loaded from the pristine checkout — the F8.1 baseline."""
    for p in (str(VENDOR_TESTS), str(VENDOR_SRC)):
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location("pristine_fall_monitor", VENDOR_SRC / "fall_monitor.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _copy_monitor():
    from fall_detection import fall_monitor as fm
    return fm


def _vendor_helpers():
    """The copied vendor wiring and synth, loaded by path (the product's own conftest must not answer)."""
    conftest = _load("seam_vendor_conftest", VENDOR_FIXTURES / "conftest.py")
    synth = _load("seam_synth", VENDOR_FIXTURES / "synth.py")
    return conftest, synth


def _fixture():
    conftest, synth = _vendor_helpers()
    t, k, b = synth.track([synth.hold(synth.STAND, 3.0), (0.3, synth.STAND, synth.LIE),
                           synth.hold(synth.LIE, 5.0)], fps=25.0)
    return t, k, b, synth.IMG_SHAPE, conftest


def _run(fm, t, k, b, img_shape, conftest):
    fake = conftest.FakeClassifier(conftest.onehot7(conftest.FALL_))
    mon = fm.FallMonitor(fm.CFG, classifier=fake, img_shape=img_shape, eval_every_s=0.0)
    return conftest.drive(mon, fake, t, k, b)   # every candidate type — parity is not fall-only


# ── the copy is the pristine text plus the manifested seam (AC-F7.1) ───────────────────────────────────────

def test_every_copied_module_is_the_pristine_text_except_the_manifested_seam():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        text = _text(COPY_DIR / name)
        for edit in (manifest["seam"].get(name) or {}).get("edits", []):
            assert edit["copy"] in text, f"{name}: the manifested seam edit is missing -- stale manifest"
            text = text.replace(edit["copy"], edit["pristine"], 1)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert digest == expected, f"{name} differs from the pristine text beyond the seam"


def test_the_copy_imports_nothing_from_the_product():
    # `__init__.py` is the vendor's own path bootstrap (the manifest pins its text); our seam
    # file must not add anything like it, and no copied module may import the product.
    seam = (COPY_DIR / "fall_monitor.py").read_text(encoding="utf-8")
    assert "sys.path.insert" not in seam
    assert "\nfrom scoring" not in seam and "\nimport scoring" not in seam
    for path in sorted(COPY_DIR.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert "\nfrom scoring" not in text and "\nimport scoring" not in text, \
            f"{path.name} imports the product's scoring -- the copy must stay self-contained"


@needs_vendor
def test_the_copy_and_the_pristine_tree_hold_the_same_modules():
    assert COPY_DIR.is_dir(), "the vendor copy does not exist yet"
    pristine = {p.name for p in VENDOR_SRC.rglob("*.py")}
    copied = {p.name for p in COPY_DIR.rglob("*.py")}
    assert pristine <= copied
    assert copied - pristine <= {"__init__.py"}
    assert (PRODUCT / "docs" / "PROVENANCE.md").is_file()


# ── the seam is additive (AC-F8.1) ─────────────────────────────────────────────────────────────────────────

@needs_vendor
def test_the_same_fixture_produces_the_same_candidates_unpatched_and_patched():
    _need_torch()
    t, k, b, img_shape, conftest = _fixture()
    base = _run(_pristine_monitor(), t, k, b, img_shape, conftest)
    patched = _run(_copy_monitor(), t, k, b, img_shape, conftest)
    assert len(base) == len(patched) >= 1
    key = lambda evs: [(e["type"], e["track"], round(e["t_start"], 3), round(e["t_candidate"], 3))
                       for e in evs]
    assert key(base) == key(patched)                                    # the boolean gate is unchanged
    # every type takes part: the fixture's 5 s LIE hold also fires detect_lying, and the seam
    # must not change that detector's contract either (AC-F30.4, ruling A7)
    assert any(e["type"] == "fall" for e in base)
    assert any(e["type"] == "lying" for e in base), \
        "the fixture's 5 s LIE hold must exercise detect_lying or AC-F30.4 is not covered"
    for e in base:
        assert "reading" not in e and "fall_paths" not in e
    for e in patched:
        assert set(e["reading"]) == {"probs", "top"} and len(e["reading"]["probs"]) == 8
        assert "path_scores" not in e                                   # the product computes those
        if e["type"] == "fall":
            assert set(e["fall_paths"]) == {"a", "b", "c"}
            assert all(isinstance(v, bool) for v in e["fall_paths"].values())
        else:
            assert e["fall_paths"] is None                              # not a fall: no latch


# ── the physical score is the packaging margin over the latched path's own gate (AC-F8.2) ───────────────

def _fall_candidate(cls_idx):
    """One fall from the copy, with the classifier scripted to `cls_idx` (the path it latches)."""
    _need_torch()
    t, k, b, img_shape, conftest = _fixture()
    fake = conftest.FakeClassifier(conftest.onehot7(cls_idx))
    fm = _copy_monitor()
    mon = fm.FallMonitor(fm.CFG, classifier=fake, img_shape=img_shape, eval_every_s=0.0)
    falls = [e for e in conftest.drive(mon, fake, t, k, b) if e["type"] == "fall"]
    assert falls, f"the fall fixture produced no candidate (classifier class {cls_idx})"
    return falls[0]


def test_the_physical_score_is_the_packaging_margin_over_the_latched_path_s_own_gate():
    import monitor
    import scoring
    conftest, _synth = _vendor_helpers()

    rec = _fall_candidate(conftest.LYING_)
    # the kinematic path spoke: the classifier was not decisive, so the action path did not
    assert rec["fall_paths"] == {"a": False, "b": True, "c": False}
    assert "path_scores" not in rec                                     # the copy carries data only
    monitor.attach_path_scores([rec], monitor.vendor_blocks())
    assert rec["path_scores"]["kinematic"] is not None
    assert rec["path_scores"]["cold"] is None
    assert rec["path_scores"]["action"] is None                         # the service fills it
    gate = _copy_monitor().CFG["path_b"]
    assert rec["path_scores"]["kinematic"] == pytest.approx(
        scoring.collapse_score(rec["evidence"]["kinematics"], gate, scoring.spans_for(gate)),
        abs=1e-9)


def test_the_action_path_claims_no_physical_slot():
    import monitor
    conftest, _synth = _vendor_helpers()

    rec = _fall_candidate(conftest.FALL_)
    assert rec["fall_paths"] == {"a": True, "b": False, "c": False}
    monitor.attach_path_scores([rec], monitor.vendor_blocks())
    assert rec["path_scores"] == {"action": None, "kinematic": None, "cold": None}
    assert rec["reading"]["top"] == conftest.FALL_                      # the smoothed vector at packaging


def test_the_camera_monitor_carries_the_copy_s_latch_into_the_path_scores():
    """The wiring the worker actually runs: the copy's FallMonitor -> CameraMonitor -> path_scores.

    Every other test here calls `attach_path_scores` itself, so a `CameraMonitor.update` that
    stopped calling it would leave the suite green while every cold path scored 0.0. This drives
    the real class through the vendor's own fixture.
    """
    import monitor as product_monitor
    conftest, _synth = _vendor_helpers()
    t, k, b, img_shape, _conftest = _fixture()
    fake = conftest.FakeClassifier(conftest.onehot7(conftest.LYING_))
    mon = product_monitor.CameraMonitor(classifier=fake)
    mon.set_shape([img_shape[1], img_shape[0]])          # the fixture's (h, w) -> the frame's [w, h]

    falls = [e for e in conftest.drive(mon, fake, t, k, b) if e["type"] == "fall"]
    assert falls, "the fixture produced no fall through the real CameraMonitor"
    assert falls[0]["path_scores"]["kinematic"] is not None, \
        "CameraMonitor.update did not carry the copy's latch into the candidate"
    assert falls[0]["path_scores"]["action"] is None     # the service fills it from the reading


def test_the_policy_s_class_count_is_the_copy_s_classifier_count():
    """scoring.CLASS_COUNT is the policy's own number; the classifier's head is the vendor's.

    The two are separate declarations, so a copy whose class list changed would silently make
    every evaluation lookup wrong. This is the only place the two numbers meet.
    """
    import scoring
    fm = _copy_monitor()
    assert scoring.CLASS_COUNT == len(fm.CFG["action_classes"]["names"])


# ── the copy imports with the product's own config.yaml (AC-F9.1) ──────────────────────────────────────────

def test_the_copied_package_imports_and_reads_the_config_beside_it():
    _need_torch()
    fm = _copy_monitor()
    assert fm.ROOT == PRODUCT
    assert (PRODUCT / "config.yaml").is_file()
    assert fm.CFG["fall"]["drop"] == 0.14 and fm.CFG["sampling"]["conf_kp"] == 0.30


# ── the vendor's own tests came with the copy (AC-N8.1) ────────────────────────────────────────────────────

@needs_vendor
def test_the_nine_vendor_test_files_are_part_of_the_product_suite():
    vendor_tests = HERE / "vendor"
    assert sorted(p.name for p in vendor_tests.glob("test_*.py")) == sorted(
        p.name for p in VENDOR_TESTS.glob("test_*.py"))
    assert len(list(vendor_tests.glob("test_*.py"))) == 9
    assert (vendor_tests / "conftest.py").is_file() and (vendor_tests / "synth.py").is_file()


def test_the_fall_test_recipe_runs_the_product_suite_and_thus_the_copied_vendor_tests():
    # the operative half of AC-N8.1 (and the dependency ruling's AC-N11.2): the nine files pass
    # under `just fall-test` — they live under the product's tests/, so the recipe must run that
    # suite in the product's own environment
    import re
    just = (PRODUCT.parents[1] / "justfile").read_text(encoding="utf-8")
    match = re.search(r"^fall-test\b.*?(?=\n\n|\n[a-z_#]|\Z)", just, re.M | re.S)
    assert match, "the justfile has no fall-test recipe"
    body = match.group(0)
    assert "pytest" in body
    assert re.search(r"^[ \t]+[^#]*pytest", body, re.M), "pytest only appears in a comment"
