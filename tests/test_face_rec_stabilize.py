"""The vote lock publishes a smoothed label: once a name is locked on a track,
every frame reports that name with the BEST score of the vote window, not its
own. products/barber counts published entries as evidence for securing a barber,
so it needs each frame's own score too -- `frame_confidence` (2026-09-20).

insightface isn't installed on the host, and the stabilizer needs none of it.
"""
import sys
import types
from pathlib import Path
from unittest import mock


for name in ("insightface", "insightface.app", "insightface.utils"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["insightface.app"].FaceAnalysis = object
sys.modules["insightface.utils"].face_align = types.SimpleNamespace()

from perception.modules.face_recognition.recognizer import FaceRecognizer  # noqa: E402

FRAME = (720, 1280)


def recognizer(min_votes=2, majority=0.6, vote_ttl=4.0):
    """The stabilizer alone: no models, no Redis."""
    r = FaceRecognizer.__new__(FaceRecognizer)
    r._names_lock = mock.MagicMock()
    r._names_lock.__enter__ = lambda *_a: None
    r._names_lock.__exit__ = lambda *_a: None
    r._names, r._track_state = {}, {}
    r.rec_thresh = 0.25
    r._track_vote_ttl = vote_ttl
    r._track_min_votes = min_votes
    r._track_majority = majority
    r._track_lock_seconds = 5.0
    r._track_reconfirm_seconds = 3.0
    return r


def face(left=400.0, top=200.0, w=28.0, h=30.0, parent_id=7):
    return {"left": left, "top": top, "width": w, "height": h, "parent_id": parent_id}


def match(name, confidence, parent_id=7, rec_id="barber-6"):
    return {"parent_id": parent_id, "name": name, "confidence": confidence, "rec_id": rec_id}


def test_a_locked_track_reports_the_windows_best_score_but_keeps_the_frames_own():
    # Measured on the barber clip (2026-09-20): one 0.601 frame was republished
    # as 0.601 for seconds, and three of those copies secured a barber.
    rec = recognizer()
    lucky = [match("suzy", 0.601)]
    rec._stabilize_names("cam", lucky, [face()], FRAME)
    weak = [match("suzy", 0.310)]
    rec._stabilize_names("cam", weak, [face()], FRAME)
    assert weak[0]["confidence"] == 0.601        # the lock's smoothed label, as before
    assert weak[0]["confirmed"] is True
    assert weak[0]["frame_confidence"] == 0.310  # what THIS frame actually scored


def test_an_unlocked_track_reports_its_own_score_as_both():
    rec = recognizer()
    res = [match("suzy", 0.487)]
    rec._stabilize_names("cam", res, [face()], FRAME)
    assert res[0]["confirmed"] is False          # one vote: not locked yet
    assert res[0]["frame_confidence"] == 0.487 == res[0]["confidence"]


def test_an_unknown_face_carries_its_own_score_too():
    rec = recognizer()
    res = [match("unknown", 0.142, rec_id=None)]
    rec._stabilize_names("cam", res, [face()], FRAME)
    assert res[0]["frame_confidence"] == 0.142
