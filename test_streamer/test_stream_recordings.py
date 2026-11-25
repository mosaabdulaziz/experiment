"""Unit tests for the pure logic behind Task 3 (replay tooling restarts all
three barber streams together): grouping, the group-restart decision, the
shared start-timestamp write, and (fix round 1) the per-group exponential
backoff that stops a repeatedly-dying member from restarting the whole
group -- and killing its healthy siblings -- on every ~2s poll tick.
No ffmpeg/subprocess/network involved.

Run: python3 -m pytest test_streamer/test_stream_recordings.py -q
(or /data/IRIS/.venv-test/bin/python -m pytest test_streamer/test_stream_recordings.py -q)
"""
import importlib.util
import json
import os

import pytest

_MODULE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stream_recordings.py")
_spec = importlib.util.spec_from_file_location("stream_recordings", _MODULE_PATH)
sr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sr)

BARBER_GROUP = ("barber-c", "barber-l", "barber-r")


class FakeProc:
    """Stand-in for subprocess.Popen: only .poll()/.returncode are used by
    the restart-decision logic under test."""

    def __init__(self, alive=True, returncode=0):
        self._alive = alive
        self.returncode = None if alive else returncode

    def poll(self):
        return self.returncode


# ── build_group_map ──────────────────────────────────────────────────────

def test_all_three_barber_cams_share_one_group():
    group_map = sr.build_group_map(["barber-c", "barber-l", "barber-r", "cam-1"])
    assert group_map["barber-c"] == BARBER_GROUP
    assert group_map["barber-l"] == BARBER_GROUP
    assert group_map["barber-r"] == BARBER_GROUP
    # an unrelated cam id restarts on its own, untouched by grouping
    assert group_map["cam-1"] == ("cam-1",)


def test_group_membership_independent_of_ids_order():
    group_map = sr.build_group_map(["barber-r", "cam-1", "barber-c", "barber-l"])
    assert group_map["barber-c"] == group_map["barber-l"] == group_map["barber-r"] == BARBER_GROUP


def test_two_of_three_present_still_group_together():
    group_map = sr.build_group_map(["barber-c", "barber-l"])
    assert group_map["barber-c"] == group_map["barber-l"] == ("barber-c", "barber-l")


def test_lone_barber_cam_is_not_treated_as_a_group():
    # only one of the three synced cams is running (e.g. a scoped --ids) --
    # nothing to desync, so it restarts on its own like any other stream.
    group_map = sr.build_group_map(["barber-c", "cam-1"])
    assert group_map["barber-c"] == ("barber-c",)


# ── group_needs_restart ──────────────────────────────────────────────────

def test_no_restart_when_every_member_is_alive():
    procs = {cid: FakeProc(alive=True) for cid in BARBER_GROUP}
    assert sr.group_needs_restart(BARBER_GROUP, procs) is False


def test_one_dead_member_flags_the_whole_group():
    """The ledger bug: barber-r's ffmpeg died and only barber-r was
    restarted, leaving it 97.6s behind barber-c/l. The fix's decision must
    flag the GROUP (all three), not just the dead member."""
    procs = {
        "barber-c": FakeProc(alive=True),
        "barber-l": FakeProc(alive=True),
        "barber-r": FakeProc(alive=False, returncode=1),
    }
    assert sr.group_needs_restart(BARBER_GROUP, procs) is True


def test_never_started_member_flags_the_group():
    procs = {"barber-c": FakeProc(alive=True), "barber-l": FakeProc(alive=True)}
    assert sr.group_needs_restart(BARBER_GROUP, procs) is True


# ── merge_group_start (offset / clock-sharing math) ──────────────────────

def test_merge_group_start_gives_every_member_the_identical_timestamp():
    updated = sr.merge_group_start({}, BARBER_GROUP, 1000.0)
    assert updated == {"barber-c": 1000.0, "barber-l": 1000.0, "barber-r": 1000.0}
    # the shared timestamp means every member's clip clock (elapsed = now -
    # start) agrees exactly (0.0 s apart), not just "within 1 s".
    values = {updated[cid] for cid in BARBER_GROUP}
    assert len(values) == 1


def test_merge_group_start_is_pure_and_leaves_other_ids_alone():
    starts = {"cam-1": 5.0, "barber-r": 42.0}
    updated = sr.merge_group_start(starts, ("barber-c", "barber-l", "barber-r"), 1000.0)
    assert starts == {"cam-1": 5.0, "barber-r": 42.0}  # input untouched
    assert updated["cam-1"] == 5.0  # unrelated id preserved
    assert updated["barber-r"] == 1000.0  # group id overwritten


# ── _record_start_many (start-file writing) ──────────────────────────────

def test_record_start_many_writes_one_shared_timestamp_for_the_group(tmp_path, monkeypatch):
    starts_file = tmp_path / "stream_starts.json"
    monkeypatch.setattr(sr, "STREAM_STARTS_FILE", str(starts_file))

    sr._record_start_many(BARBER_GROUP, ts=1234.5)

    with open(starts_file) as f:
        written = json.load(f)
    assert written == {"barber-c": 1234.5, "barber-l": 1234.5, "barber-r": 1234.5}
    # clock agreement check mirrored from Step 3's live check: every pair of
    # clip clocks (now - start) agrees within 1 s -- here exactly 0 s.
    diffs = [abs(written[a] - written[b])
             for a in BARBER_GROUP for b in BARBER_GROUP]
    assert max(diffs) < 1.0


def test_record_start_many_merges_with_existing_file(tmp_path, monkeypatch):
    starts_file = tmp_path / "stream_starts.json"
    starts_file.write_text(json.dumps({"cam-1": 1.0, "barber-c": 0.1}))
    monkeypatch.setattr(sr, "STREAM_STARTS_FILE", str(starts_file))

    sr._record_start_many(BARBER_GROUP, ts=99.0)

    with open(starts_file) as f:
        written = json.load(f)
    assert written["cam-1"] == 1.0  # untouched
    assert written["barber-c"] == written["barber-l"] == written["barber-r"] == 99.0


# ── backoff_seconds ───────────────────────────────────────────────────────

def test_backoff_seconds_no_wait_for_zero_or_negative_failures():
    assert sr.backoff_seconds(0) == 0.0
    assert sr.backoff_seconds(-1) == 0.0


def test_backoff_seconds_doubles_each_time():
    assert sr.backoff_seconds(1) == 2.0
    assert sr.backoff_seconds(2) == 4.0
    assert sr.backoff_seconds(3) == 8.0
    assert sr.backoff_seconds(4) == 16.0
    assert sr.backoff_seconds(5) == 32.0


def test_backoff_seconds_caps_at_60s():
    assert sr.backoff_seconds(6) == 60.0
    assert sr.backoff_seconds(20) == 60.0


# ── next_group_action (the fix-round-1 backoff state machine) ───────────

def _state():
    return {"failures": 0, "started_at": None, "next_allowed_ts": None}


def test_very_first_start_is_immediate():
    action, state = sr.next_group_action(_state(), now=100.0, needs_restart=True, forced=False)
    assert action == "start"
    assert state["started_at"] == 100.0
    assert state["failures"] == 0  # first-ever start isn't counted as a failure


def test_healthy_group_is_idle_and_keeps_its_failure_count():
    state = {"failures": 0, "started_at": 100.0, "next_allowed_ts": None}
    action, state = sr.next_group_action(state, now=105.0, needs_restart=False, forced=False)
    assert action == "idle"
    assert state["failures"] == 0


def test_first_failure_after_a_start_is_immediate_no_backoff():
    """The scenario this fix-round guards against: a member that dies right
    away must still get ONE immediate retry, not be punished from the very
    first failure."""
    state = {"failures": 0, "started_at": 100.0, "next_allowed_ts": None}
    action, state = sr.next_group_action(state, now=101.0, needs_restart=True, forced=False)
    assert action == "start"
    assert state["started_at"] == 101.0
    assert state["failures"] == 1


def test_repeated_immediate_deaths_back_off_instead_of_restarting_every_tick():
    """Regression for review round 1: a member that keeps dying right after
    each restart must NOT cause a restart on every ~2s poll forever."""
    state = {"failures": 1, "started_at": 101.0, "next_allowed_ts": None}
    # died again 1s later, well under the 60s reset window
    action, state = sr.next_group_action(state, now=102.0, needs_restart=True, forced=False)
    assert action == "wait"
    assert state["next_allowed_ts"] == pytest.approx(102.0 + 2.0)  # backoff_seconds(1) == 2

    # still within the backoff window a couple ticks later -- keep waiting,
    # WITHOUT touching the healthy siblings and without re-scheduling
    action2, state2 = sr.next_group_action(state, now=103.0, needs_restart=True, forced=False)
    assert action2 == "wait"
    assert state2["next_allowed_ts"] == state["next_allowed_ts"]  # unchanged, not re-armed
    assert state2["failures"] == 1  # not incremented again while still waiting

    # backoff elapsed -- now it may actually restart
    action3, state3 = sr.next_group_action(state2, now=104.0, needs_restart=True, forced=False)
    assert action3 == "start"
    assert state3["failures"] == 2
    assert state3["started_at"] == 104.0


def test_backoff_grows_with_each_consecutive_failure():
    # failures=2 already recorded -> the 3rd failure's wait is backoff_seconds(2) == 4s
    state = {"failures": 2, "started_at": 200.0, "next_allowed_ts": None}
    action, state = sr.next_group_action(state, now=200.5, needs_restart=True, forced=False)
    assert action == "wait"
    assert state["next_allowed_ts"] == pytest.approx(200.5 + 4.0)


def test_staying_up_60s_resets_the_failure_count():
    state = {"failures": 3, "started_at": 100.0, "next_allowed_ts": None}
    action, state = sr.next_group_action(state, now=160.0, needs_restart=False, forced=False)
    assert action == "idle"
    assert state["failures"] == 0


def test_staying_up_under_60s_does_not_reset_yet():
    state = {"failures": 3, "started_at": 100.0, "next_allowed_ts": None}
    action, state = sr.next_group_action(state, now=159.0, needs_restart=False, forced=False)
    assert action == "idle"
    assert state["failures"] == 3


def test_forced_resync_bypasses_backoff_and_resets_failures():
    # even mid-backoff-wait, an on-demand SIGUSR1 resync happens immediately
    state = {"failures": 4, "started_at": 100.0, "next_allowed_ts": 500.0}
    action, state = sr.next_group_action(state, now=150.0, needs_restart=False, forced=True)
    assert action == "start"
    assert state["failures"] == 0
    assert state["next_allowed_ts"] is None
    assert state["started_at"] == 150.0


def test_barber_group_streams_one_synced_recording_per_camera():
    """center / left / right are the same recording from three cameras, so they sit in one folder
    (CAM_FILES says which footage: it moved from the 39-minute clips to the 109-minute synced3 set)."""
    files = {cam: sr.CAM_FILES[cam] for cam in ("barber-c", "barber-l", "barber-r")}
    assert {cam: f.rsplit("/", 1)[-1] for cam, f in files.items()} == {
        "barber-c": "center.mp4", "barber-l": "left.mp4", "barber-r": "right.mp4"}
    assert len({f.rsplit("/", 1)[0] for f in files.values()}) == 1


PH = tuple(f"ph-{i}" for i in range(1, 8))


def test_palm_hills_cams_loop_as_one_group():
    group_map = sr.build_group_map(list(PH))
    assert all(group_map[c] == PH for c in PH)
    assert all(sr.CAM_FILES[c].startswith("/data/IRIS/data/BekkyBarber/palm_hills/Cam") for c in PH)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


def test_files_override_maps_ids_to_other_recordings():
    """2026-10-18 (user): the 16 Oct synced recordings play as ph-* (same ids, same zones) without editing CAM_FILES."""
    assert sr.parse_files("ph-1=/x/Cam1.mp4, ph-3=/x/Cam3.mp4") == {"ph-1": "/x/Cam1.mp4", "ph-3": "/x/Cam3.mp4"}
    assert sr.parse_files("") == {}
    with pytest.raises(ValueError, match="ph-2"):
        sr.parse_files("ph-1=/x/Cam1.mp4,ph-2")
