from app.tracking import CameraAggregator


def test_note_assigned_places_late_sighting_on_the_wall_clock_and_flags_pending():
    agg = CameraAggregator(["cam-01", "cam-02"])
    agg.process_entry("cam-02", [], {}, pts=100.0)  # reference: pts 100 is "now"
    ref_wall = agg._ref["cam-02"][1]
    agg.note_assigned("cam-02", 5, pts_first=90.0, pts_last=95.0)
    v = agg.get_vehicle_states()["cam-02"][5]
    assert v["pending"] and abs(v["wall_last"] - (ref_wall - 5.0)) < 0.01
    assert abs(v["wall_first"] - (ref_wall - 10.0)) < 0.01
    # an already-known sighting is only extended, not re-flagged
    agg.note_assigned("cam-02", 5, pts_first=90.0, pts_last=99.0)
    assert agg.get_vehicle_states()["cam-02"][5]["wall_last"] > ref_wall - 5.0
    agg.note_assigned("cam-09", 5, 1.0, 2.0)  # unknown camera is ignored
    agg.process_entry("cam-01", [], {}, pts=2.0)
    agg.note_assigned("cam-01", -1, 1.0, 2.0)  # untracked id is ignored
    agg.note_assigned("cam-01", 0, 1.0, 2.0)  # id 0 is a real global id
    assert 0 in agg.get_vehicle_states()["cam-01"] and -1 not in agg.get_vehicle_states()["cam-01"]
    assert 5 not in agg.get_vehicle_states()["cam-01"]


def test_replayed_entries_keep_their_publish_time():
    import time
    agg = CameraAggregator(["cam-01"])
    old = time.time() - 600.0
    agg.process_entry("cam-01", [["car", 0.9, 1, 0, 0, 10, 10]], {"1": 3}, pts=10.0, wall=old)
    v = agg.get_vehicle_states()["cam-01"][3]
    assert v["wall_first"] == old and v["wall_last"] == old  # not stamped with "now"


def test_assigned_event_needs_a_reference_and_old_ones_are_not_news():
    import time
    agg = CameraAggregator(["cam-01"])
    agg.note_assigned("cam-01", 4, 1.0, 2.0)  # no frame seen yet: can't place it
    assert 4 not in agg.get_vehicle_states()["cam-01"]
    agg.process_entry("cam-01", [], {}, pts=1000.0, wall=time.time())
    agg.note_assigned("cam-01", 4, 10.0, 20.0)  # ~16 min before this run started
    assert agg.get_vehicle_states()["cam-01"][4]["pending"] is False
    agg.note_assigned("cam-01", 6, 995.0, 998.0)  # moments ago: confirmed late, news
    assert agg.get_vehicle_states()["cam-01"][6]["pending"] is True
