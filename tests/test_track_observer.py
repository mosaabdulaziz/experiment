"""TrackObserver - app-measured presence and stationarity per tracked object.

These numbers reach the model labelled "exact, not estimates", so the maths is
the whole point of the module. Two failure modes would make the prompt lie with
authority, and each has a dedicated regression test below: a presence duration
capped by the bounded position deque, and a stationary clock derived from that
same deque instead of from a running anchor.
"""
import math

from app.track_observer import TrackObserver, format_duration

W, H = 1920, 1080
DIAG = math.hypot(W, H)


def _obs(**kw):
    return TrackObserver(frame_width=W, frame_height=H, **kw)


def _det(oid, label="person", cx=500.0, cy=500.0, w=40.0, h=80.0):
    """A parse_dets-shaped (cls, conf, track_id, bbox) tuple centred on (cx, cy)."""
    return (label, 0.9, oid, (cx - w / 2, cy - h / 2, w, h))


# --- duration maths -------------------------------------------------------

def test_durations_under_an_hour_render_as_minutes_and_padded_seconds():
    assert format_duration(22) == "0m22s"
    assert format_duration(663) == "11m03s"


def test_durations_at_or_above_an_hour_render_as_hours_and_padded_minutes():
    assert format_duration(3600) == "1h00m"
    assert format_duration(3840) == "1h04m"


# --- presence -------------------------------------------------------------

def test_present_for_is_not_capped_by_the_position_window():
    """Regression for design point 2: first_seen outlives the trimmed deque."""
    o = _obs(window_seconds=120.0, stale_seconds=3600.0)
    o.observe(0.0, [_det(7)])
    o.observe(700.0, [_det(7)])
    assert o.present_for(7, 700.0) == 700.0


def test_present_for_is_none_for_an_id_never_seen():
    assert _obs().present_for(7, 0.0) is None


# --- stationarity ---------------------------------------------------------

def test_stationary_for_is_not_capped_by_the_position_window():
    """Regression for design point 1: the anchor is running, not deque-derived."""
    o = _obs(window_seconds=120.0)
    for t in range(0, 701, 10):
        o.observe(float(t), [_det(7)])
    assert o.stationary_for(7, 700.0) == 700.0


def test_moving_beyond_the_radius_restarts_the_stationary_clock():
    o = _obs(stationary_radius=0.02)
    o.observe(0.0, [_det(7, cx=500.0)])
    o.observe(10.0, [_det(7, cx=500.0 + 0.05 * DIAG)])
    assert o.stationary_for(7, 10.0) == 0.0


def test_jitter_within_the_radius_does_not_restart_the_stationary_clock():
    o = _obs(stationary_radius=0.02)
    o.observe(0.0, [_det(7, cx=500.0, cy=500.0)])
    o.observe(10.0, [_det(7, cx=500.0 + 0.01 * DIAG, cy=500.0)])
    assert o.stationary_for(7, 10.0) == 10.0


def test_the_radius_is_a_fraction_of_the_frame_diagonal_in_pixels():
    """Design point 3: one pixel threshold, not a per-axis normalised one."""
    near = _obs(stationary_radius=0.02)
    near.observe(0.0, [_det(7, cx=500.0, cy=500.0)])
    near.observe(10.0, [_det(7, cx=500.0 + 0.019 * DIAG, cy=500.0)])
    assert near.stationary_for(7, 10.0) == 10.0

    far = _obs(stationary_radius=0.02)
    far.observe(0.0, [_det(7, cx=500.0, cy=500.0)])
    far.observe(10.0, [_det(7, cx=500.0 + 0.021 * DIAG, cy=500.0)])
    assert far.stationary_for(7, 10.0) == 0.0


# --- degenerate detections ------------------------------------------------

def test_untracked_detections_create_no_record():
    o = _obs()
    o.observe(0.0, [("person", 0.9, None, (10.0, 10.0, 40.0, 80.0))])
    assert o.render(0.0) == ""


def test_a_missing_bbox_keeps_the_track_alive_without_touching_the_anchor():
    o = _obs(stale_seconds=30.0)
    o.observe(0.0, [_det(7, cx=500.0)])
    o.observe(20.0, [("person", 0.9, 7, None)])
    o.observe(40.0, [("person", 0.9, 7, None)])
    assert o.present_for(7, 40.0) == 40.0      # last_seen advanced past the 30s cut
    assert o.stationary_for(7, 40.0) == 40.0   # anchor untouched


# --- pruning and bounds ---------------------------------------------------

def test_a_stale_track_is_dropped_and_reappears_as_a_fresh_record():
    o = _obs(stale_seconds=30.0)
    o.observe(0.0, [_det(7)])
    o.observe(100.0, [])
    assert o.present_for(7, 100.0) is None
    o.observe(110.0, [_det(7)])
    assert o.present_for(7, 110.0) == 0.0


def test_position_history_is_trimmed_to_the_window():
    o = _obs(window_seconds=60.0, stale_seconds=3600.0)
    for t in range(0, 301, 10):
        o.observe(float(t), [_det(7)])
    oldest = o._tracks[7].positions[0][0]
    assert oldest >= 300.0 - 60.0


# --- speed (a query only; deliberately never rendered) --------------------

def test_speed_is_pixels_per_second_across_the_query_window():
    o = _obs()
    o.observe(0.0, [_det(7, cx=100.0, cy=100.0)])
    o.observe(10.0, [_det(7, cx=200.0, cy=100.0)])
    assert o.speed(7, 10.0, window_seconds=20.0) == 10.0


def test_speed_is_none_without_two_samples_in_the_window():
    o = _obs()
    o.observe(0.0, [_det(7)])
    assert o.speed(7, 0.0) is None


# --- rendering ------------------------------------------------------------

def test_render_is_empty_when_nothing_is_tracked():
    assert _obs().render(0.0) == ""


def test_render_orders_longest_present_first():
    o = _obs()
    o.observe(0.0, [_det(23, label="truck", cx=300.0)])
    o.observe(5.0, [_det(23, label="truck", cx=300.0), _det(31, cx=800.0)])
    rows = [r.strip() for r in o.render(10.0).strip().splitlines()[1:]]
    assert [r.split()[0] for r in rows] == ["#23", "#31"]


def test_render_caps_the_row_count_at_max_rendered():
    o = _obs(max_rendered=3)
    o.observe(0.0, [_det(i, cx=100.0 * i) for i in range(1, 11)])
    assert len(o.render(1.0).strip().splitlines()) == 1 + 3


def test_render_reports_moving_for_a_track_that_just_moved():
    o = _obs()
    o.observe(0.0, [_det(7, cx=300.0)])
    o.observe(10.0, [_det(7, cx=300.0 + 0.5 * DIAG)])
    assert o.render(10.0).strip().splitlines()[1].endswith("moving")


def test_render_reports_a_measured_duration_for_a_still_track():
    o = _obs()
    for t in range(0, 121, 10):
        o.observe(float(t), [_det(7, cx=300.0)])
    assert "stationary 2m00s" in o.render(120.0)


def test_render_matches_the_documented_block_format():
    """Golden render, byte for byte, including the load-bearing parenthetical."""
    o = _obs(stale_seconds=3600.0)
    settled = 300.0 + 0.5 * DIAG
    o.observe(0.0, [_det(23, label="truck", cx=300.0, cy=300.0)])
    o.observe(5.0, [_det(23, label="truck", cx=settled, cy=300.0)])
    o.observe(411.0, [_det(23, label="truck", cx=settled, cy=300.0),
                      _det(17, cx=100.0, cy=100.0)])
    o.observe(641.0, [_det(23, label="truck", cx=settled, cy=300.0),
                      _det(17, cx=100.0, cy=100.0),
                      _det(31, cx=900.0, cy=900.0)])
    o.observe(663.0, [_det(23, label="truck", cx=settled, cy=300.0),
                      _det(17, cx=100.0 + 0.4 * DIAG, cy=100.0),
                      _det(31, cx=900.0, cy=900.0 - 0.3 * DIAG)])
    assert o.render(663.0) == (
        "\n## TRACKED OBJECTS (measured — these durations are exact, not estimates)\n"
        "  #23 truck    present 11m03s  stationary 10m58s\n"
        "  #17 person   present 4m12s   moving\n"
        "  #31 person   present 0m22s   moving\n"
    )


def test_render_is_wrapped_in_newlines_so_the_prompt_slot_spaces_correctly():
    o = _obs()
    o.observe(0.0, [_det(7)])
    block = o.render(1.0)
    assert block.startswith("\n") and block.endswith("\n")


def test_render_claims_no_motion_for_a_track_that_never_carried_a_bbox():
    """An older sidecar publishes no bbox and parse_dets maps it to None by design.
    Rendering "moving" there would assert motion the module has not measured — and
    against such a sidecar it would do so for every object in the frame."""
    o = TrackObserver(frame_width=1280, frame_height=720)
    o.observe(0.0, [("person", 0.9, 4, None)])
    row = o.render(30.0).strip().splitlines()[1]
    assert "moving" not in row
    assert "stationary" not in row
    assert row.rstrip() == row  # no dangling pad where the state column would have been
    assert "present 0m30s" in row


def test_live_ids_reports_what_is_tracked_now_and_forgets_the_stale():
    """SceneMemory keys entity notes on track id but owns no tracks, so this set is
    the only thing standing between it and a note about someone who left ten minutes
    ago rendering forever."""
    o = _obs(stale_seconds=30.0)
    assert o.live_ids() == set()
    o.observe(0.0, [_det(17), _det(23)])
    assert o.live_ids() == {17, 23}
    o.observe(40.0, [_det(23)])          # 17 last seen at 0.0 -> pruned at 40.0
    assert o.live_ids() == {23}
