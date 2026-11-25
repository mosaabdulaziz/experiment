from app.alert_clips import (
    contiguous_run_containing,
    find_coalescible,
    segments_cover,
)
from app.segment_store import SegmentRef


def _job(sid="cam0", anchor=100.0, state="queued"):
    return {"sid": sid, "anchor": anchor, "lo": anchor - 5.0, "hi": anchor + 5.0,
            "event_ids": [1], "state": state, "deadline": 0.0}


def _ref(start, end):
    return SegmentRef(start, end, f"k{start}", 0.0)


# ── find_coalescible ────────────────────────────────────────────────────────

def test_alert_within_radius_on_same_stream_coalesces():
    jobs = [_job(anchor=100.0)]
    assert find_coalescible(jobs, "cam0", 102.0, 2.5) is jobs[0]


def test_alert_outside_radius_gets_its_own_job():
    jobs = [_job(anchor=100.0)]
    assert find_coalescible(jobs, "cam0", 103.0, 2.5) is None


def test_other_streams_never_coalesce():
    jobs = [_job(sid="cam0", anchor=100.0)]
    assert find_coalescible(jobs, "cam1", 100.0, 2.5) is None


def test_a_job_already_building_is_not_coalescible():
    # Its footage is being cut; a late arrival must get its own clip rather than
    # silently attaching to a build it will not appear in.
    jobs = [_job(anchor=100.0, state="building")]
    assert find_coalescible(jobs, "cam0", 100.1, 2.5) is None


def test_nearest_anchor_wins_when_several_qualify():
    far, near = _job(anchor=100.0), _job(anchor=101.5)
    assert find_coalescible([far, near], "cam0", 101.4, 2.5) is near


def _zoned_job(zone_id, anchor=100.0):
    job = _job(anchor=anchor)
    job["zone"] = {"id": zone_id, "name": zone_id, "points": [],
                   "intrusion": {"mode": "restricted"}}
    return job


def test_alerts_on_the_same_zone_coalesce():
    jobs = [_zoned_job("z0")]
    assert find_coalescible(jobs, "cam0", 101.0, 2.5, "z0") is jobs[0]


def test_alerts_on_different_zones_never_coalesce():
    # A shared clip draws ONE zone outline, so mixing zones onto it would label some
    # alerts with a zone they did not fire on.
    jobs = [_zoned_job("z0")]
    assert find_coalescible(jobs, "cam0", 101.0, 2.5, "z1") is None


def test_zoneless_alerts_still_coalesce_with_each_other():
    # Proactive VLM alerts carry no zone; their behaviour must not change.
    jobs = [_job(anchor=100.0)]
    assert find_coalescible(jobs, "cam0", 101.0, 2.5, None) is jobs[0]


def test_a_zoned_alert_does_not_join_a_zoneless_job():
    jobs = [_job(anchor=100.0)]
    assert find_coalescible(jobs, "cam0", 101.0, 2.5, "z0") is None


def test_no_jobs_returns_none():
    assert find_coalescible([], "cam0", 100.0, 2.5) is None


# ── segments_cover ──────────────────────────────────────────────────────────

def test_contiguous_segments_spanning_the_window_cover_it():
    refs = [_ref(90.0, 98.0), _ref(98.0, 106.0)]
    assert segments_cover(refs, 95.0, 105.0) is True


def test_a_gap_inside_the_window_is_not_coverage():
    # A sidecar restart or a dropped upload leaves a hole; concatenating across it
    # would produce a time discontinuity.
    refs = [_ref(90.0, 98.0), _ref(101.0, 109.0)]
    assert segments_cover(refs, 95.0, 105.0) is False


def test_material_ending_before_the_window_is_not_coverage():
    refs = [_ref(90.0, 98.0)]
    assert segments_cover(refs, 95.0, 105.0) is False


def test_material_starting_after_the_window_is_not_coverage():
    refs = [_ref(97.0, 106.0)]
    assert segments_cover(refs, 95.0, 105.0) is False


def test_empty_refs_are_not_coverage():
    assert segments_cover([], 95.0, 105.0) is False


def test_unsorted_refs_are_handled():
    refs = [_ref(98.0, 106.0), _ref(90.0, 98.0)]
    assert segments_cover(refs, 95.0, 105.0) is True


# ── contiguous_run_containing ───────────────────────────────────────────────

def test_run_containing_the_anchor_is_returned():
    a, b, c = _ref(80.0, 88.0), _ref(88.0, 96.0), _ref(120.0, 128.0)
    assert contiguous_run_containing([a, b, c], 90.0) == [a, b]


def test_later_run_is_returned_when_it_holds_the_anchor():
    a, b, c = _ref(80.0, 88.0), _ref(88.0, 96.0), _ref(120.0, 128.0)
    assert contiguous_run_containing([a, b, c], 123.0) == [c]


def test_anchor_inside_the_gap_yields_nothing():
    # A clip that does not contain its own alert moment is worthless.
    a, c = _ref(80.0, 88.0), _ref(120.0, 128.0)
    assert contiguous_run_containing([a, c], 100.0) == []


def test_empty_refs_yield_nothing():
    assert contiguous_run_containing([], 100.0) == []


def test_nested_refs_do_not_split_a_gap_free_run():
    # 'b' is nested inside 'a', so the run's right edge must stay at a.end_pts
    # (100.0) rather than collapsing to b.end_pts (90.0) and making 'c' look
    # like it starts after a gap. segments_cover already agrees this span is
    # gap-free; these two helpers must not disagree on the same refs.
    a, b, c = _ref(80.0, 100.0), _ref(85.0, 90.0), _ref(99.0, 105.0)
    assert contiguous_run_containing([a, b, c], 97.0) == [a, b, c]
    assert segments_cover([a, b, c], 80.0, 105.0) is True
