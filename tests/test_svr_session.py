"""SvrFileTracker: correlate each SVR session's start to its sr-done completion.

Pure Python, importable off-GPU (same convention as test_svr_trigger.py)."""

from perception.engine.svr_session import SvrFileTracker


def test_session_started_records_only_the_newly_appeared_name():
    t = SvrFileTracker()
    recorded = t.session_started(before_names=["cam-7_00001.mkv"],
                                  after_names=["cam-7_00001.mkv", "cam-7_00002.mkv"])
    assert recorded == ["cam-7_00002.mkv"]
    assert t.pending_count() == 1


def test_session_done_pops_in_fifo_order():
    t = SvrFileTracker()
    t.session_started([], ["cam-7_a.mkv"])
    t.session_started(["cam-7_a.mkv"], ["cam-7_a.mkv", "cam-7_b.mkv"])
    assert t.session_done() == "cam-7_a.mkv"
    assert t.session_done() == "cam-7_b.mkv"
    assert t.session_done() is None


def test_reproduces_the_traced_race_back_to_back_sessions():
    # Session A starts and is recorded. Before A's sr-done is processed, the
    # fire/smoke re-arm loop has already started session B, whose file has begun
    # appearing on disk. session_done() must still resolve to A, the OLDER session
    # -- not B, whose file has only ~1s of content at this point (the traced bug).
    t = SvrFileTracker()
    t.session_started(before_names=[], after_names=["cam-7_a.mkv"])          # A starts
    t.session_started(before_names=["cam-7_a.mkv"],
                       after_names=["cam-7_a.mkv", "cam-7_b.mkv"])           # B starts before A's sr-done
    assert t.session_done() == "cam-7_a.mkv"    # A's sr-done: correctly resolves to A
    assert t.session_done() == "cam-7_b.mkv"    # B's sr-done, later: resolves to B


def test_session_done_on_an_empty_tracker_returns_none():
    t = SvrFileTracker()
    assert t.session_done() is None


def test_multiple_names_in_one_diff_queue_in_sorted_order():
    # Defensive case: more than one prefix-matching name appearing between the
    # before/after snapshots (should not happen in normal single-session-per-source
    # operation, but must not silently drop or misorder any of them).
    t = SvrFileTracker()
    t.session_started(before_names=[], after_names=["cam-7_b.mkv", "cam-7_a.mkv"])
    assert t.session_done() == "cam-7_a.mkv"
    assert t.session_done() == "cam-7_b.mkv"


def test_names_present_in_both_snapshots_are_not_recorded_again():
    t = SvrFileTracker()
    t.session_started(before_names=["cam-7_a.mkv"], after_names=["cam-7_a.mkv"])
    assert t.pending_count() == 0
    assert t.session_done() is None
