"""Unit tests for perception/modules/face_detect/idle.py — the face-SGIE auto-toggle policy.

Pure stdlib (no gi), so these run on the host: the rule (gallery non-empty OR
capture enabled -> run; otherwise skip) and the transition monitor (flips the
SGIE's enabled state — wired by the sidecar onto `infer-on-class-ids` — only
when the state actually changes).
"""

import pytest

from perception.modules.face_detect.idle import (
    FACE_GALLERY_KEY,
    SGIE_DISABLED_CLASS,
    SgieGalleryMonitor,
    sgie_should_run,
)


# ── the decision rule ──────────────────────────────────────────────────────

def test_empty_gallery_no_capture_disabled():
    assert sgie_should_run(0, False) is False


def test_registered_faces_enable():
    assert sgie_should_run(1, False) is True
    assert sgie_should_run(42, False) is True


def test_capture_keeps_sgie_on_even_with_empty_gallery():
    # Capture consumes SGIE detections to build the gallery — it must stay on.
    assert sgie_should_run(0, True) is True


def test_gallery_key_matches_sidecar_register():
    assert FACE_GALLERY_KEY == "iris:face:gallery"


def test_disabled_class_matches_no_pgie_class():
    # 999 is well past the yolo pgie's ~80 classes, so no object ever matches
    # and should_infer_object() skips every person -> zero inference.
    assert SGIE_DISABLED_CLASS == "999"


# ── the monitor (transition behavior) ─────────────────────────────────────

class FakeMonitor:
    """Harness: scripted gallery counts + a recorder of applied enabled states."""

    def __init__(self, counts, capture=False):
        self.counts = iter(counts)
        self.applied = []

        def hlen():
            return next(self.counts)

        def set_enabled(on):
            self.applied.append(on)

        self.mon = SgieGalleryMonitor(hlen, set_enabled, capture)


def test_boot_empty_gallery_applies_disabled():
    f = FakeMonitor([0])
    assert f.mon.poll() is False
    assert f.applied == [False]


def test_boot_with_faces_applies_enabled():
    f = FakeMonitor([5])
    assert f.mon.poll() is True
    assert f.applied == [True]


def test_registration_flips_disabled_to_enabled():
    # boots empty (disabled), then a face is registered
    f = FakeMonitor([0, 0, 1, 1])
    assert f.mon.poll() is False
    assert f.mon.poll() is None            # no change while still empty
    assert f.mon.poll() is True
    assert f.mon.poll() is None            # stays enabled
    assert f.applied == [False, True]


def test_unregister_flips_enabled_to_disabled():
    # boots with faces (enabled), then the last face is removed
    f = FakeMonitor([2, 1, 0, 0])
    assert f.mon.poll() is True
    assert f.mon.poll() is None            # still non-empty
    assert f.mon.poll() is False
    assert f.mon.poll() is None
    assert f.applied == [True, False]


def test_capture_on_never_disables():
    f = FakeMonitor([0, 0, 0], capture=True)
    assert f.mon.poll() is True
    assert f.mon.poll() is None
    assert f.applied == [True]


def test_poll_errors_are_non_fatal():
    # hlen raising must not crash the monitor (returns None, keeps running)
    def boom():
        raise RuntimeError("redis down")

    mon = SgieGalleryMonitor(boom, lambda on: None, False)
    assert mon.poll() is None
    assert mon.current_enabled is None


def test_no_write_until_state_changes():
    # idempotent polls must not keep writing the property
    f = FakeMonitor([7, 7, 7])
    f.mon.poll()
    f.mon.poll()
    f.mon.poll()
    assert f.applied == [True]
