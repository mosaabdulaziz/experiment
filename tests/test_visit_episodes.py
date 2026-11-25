"""A vehicle that comes back after a long absence is a new stay (new visit), and a visit that was
completed by a long absence at the entrance is reported as an exit at its last sighting."""
import collections
import sys
import pathlib
import types

import numpy as np

from app.tracking import CameraAggregator
from app.truck_feed import TruckFeedBridge, _iso, gid_to_plate

from perception.modules.mtmc.core import MTMCManager
from perception.modules.mtmc.core._associate import GlobalVehicleAssociation

CFG = {"enabled": True, "company_id": "El-Mohandes", "branch_id": "BRANCH-001",
       "plates_dir": str(pathlib.Path(__file__).parent / "_no_plates"),   # not the checkout's data/plates
       "camera_ips": {"cam-01": "10.0.0.1", "cam-02": "10.0.0.2"}}
DET = [["vehicle", 0.9, 7, 10, 10, 50, 50]]


def _agg():
    return CameraAggregator(["cam-01", "cam-02"])


def test_a_return_after_a_long_absence_starts_a_new_stay_with_its_own_start_time():
    a = _agg()
    for t in (1000.0, 1002.0, 1004.0):
        a.process_entry("cam-02", DET, {"7": 0}, 1.0, wall=t)
    a.process_entry("cam-02", DET, {"7": 0}, 2.0, wall=1000.0 + 900)      # 15 minutes later
    a.process_entry("cam-02", DET, {"7": 0}, 3.0, wall=1000.0 + 902)
    v = a._camera_vehicles["cam-02"][0]
    assert v["wall_first"] == 1900.0 and v["wall_last"] == 1902.0 and v["episodes"] == [(1000.0, 1004.0)]
    eps = a.episodes_of(0)
    assert [(e["cam"], e["first"], e["last"]) for e in eps] == [("cam-02", 1000.0, 1004.0), ("cam-02", 1900.0, 1902.0)]


def test_a_short_gap_is_the_same_stay():
    a = _agg()
    a.process_entry("cam-02", DET, {"7": 0}, 1.0, wall=1000.0)
    a.process_entry("cam-02", DET, {"7": 0}, 2.0, wall=1060.0)
    assert a._camera_vehicles["cam-02"][0]["wall_first"] == 1000.0 and len(a.episodes_of(0)) == 1


def test_a_sighting_confirmed_late_after_a_long_absence_is_a_new_stay_not_a_longer_one():
    a = _agg()
    a._started = 0.0                             # the test's clock is fake: everything counts as this run's
    a.process_entry("cam-01", DET, {"7": 0}, 100.0, wall=1000.0)
    a._ref["cam-01"] = (1000.0, 2000.0)         # now: pts 1000 <-> wall 2000
    a.note_assigned("cam-01", 0, 990.0, 995.0)  # the sighting really happened ~10 s ago, 1000 s after the last
    eps = a.episodes_of(0)
    assert len(eps) == 2 and eps[0]["last"] == 1000.0 and eps[1]["first"] > 1900
    assert a._camera_vehicles["cam-01"][0].get("pending") is True


def _v(first, last):
    return {"wall_first": first, "wall_last": last}


def test_a_completed_visit_is_reported_at_its_last_sighting_and_the_return_starts_a_new_one():
    b = TruckFeedBridge(CFG, "redis://x", {"gate": ["cam-01", "cam-02"]})
    b.note_exit(7, "cam-02", 1004.0, reopens=True)
    states = {"cam-01": {}, "cam-02": {7: _v(1000.0, 1004.0)}}
    out = b.build_batch(states, now=1005.0)
    sent = [(c["ip"], c["observed_at"], c["plates"][0].get("exited")) for c in out["cameras"] if c["plates"]]
    assert sent == [("10.0.0.2", _iso(1004.0), True)]               # the exit, at the last sighting
    b._exit_marks and [b._exits[g].__setitem__(2, True) for g in b._exit_marks]   # what a successful push records
    # the vehicle is back 15 minutes later, with a new stay starting at 1900
    states = {"cam-01": {}, "cam-02": {7: {"wall_first": 1900.0, "wall_last": 1902.0, "episodes": [(1000.0, 1004.0)]}}}
    out = b.build_batch(states, now=1903.0)
    entry = [c for c in out["cameras"] if c["plates"]]
    assert len(entry) == 1 and entry[0]["plates"][0]["plate"] == gid_to_plate(7) and not entry[0]["plates"][0].get("exited")
    assert entry[0]["plates"][0]["first_seen_at"] == _iso(1900.0) and 7 not in b._exits


def test_a_real_exit_still_silences_the_vehicle():
    b = TruckFeedBridge(CFG, "redis://x")
    b.note_exit(7, "cam-02", 1004.0)               # leaving the premises: not reopened
    b.build_batch({"cam-01": {}, "cam-02": {7: _v(1000.0, 1004.0)}}, now=1005.0)
    b._exits[7][2] = True
    out = b.build_batch({"cam-01": {}, "cam-02": {7: _v(1900.0, 1902.0)}}, now=1903.0)
    assert all(c["plates"] == [] for c in out["cameras"])


def _manager(absence_s, history_cams):
    assoc = GlobalVehicleAssociation(topology=["cam-1", "cam-2", "cam-3"], min_disappearance_s=2.0,
                                     origin_cameras=["cam-1", "cam-2"])
    gid = assoc.get_or_create_global_id("cam-1", 0, 10.0)
    gtr = assoc.global_tracks[gid]
    for cam in history_cams:
        gtr.add_camera_observation(cam, 9, 20.0)
    assoc.mark_tracklet_inactive("cam-1", 0, 30.0)
    for cam in history_cams:
        assoc.mark_tracklet_inactive(cam, 9, 30.0)
    m = MTMCManager.__new__(MTMCManager)
    m.association, m._absence_exit_s = assoc, absence_s
    m._exit_events, m._exits_total = collections.deque(), 0
    return m, gid, gtr


def test_a_long_absence_then_a_return_at_the_entrance_closes_the_earlier_visit_but_keeps_the_vehicle():
    m, gid, gtr = _manager(300.0, [])
    gtr.add_camera_observation("cam-1", 5, 30.0 + 900.0)                 # back 15 minutes later
    m._close_if_reappeared(gid, "cam-1", 5)
    ev = m.pop_exit_events()
    assert len(ev) == 1 and ev[0]["global_id"] == gid and ev[0]["reopens"] == 1
    assert 910 < (__import__("time").time() - ev[0]["at"]) < 930        # stamped at the last sighting, 920 s of video ago
    assert not gtr.exited                                              # the vehicle stays a candidate


def test_a_short_absence_or_the_rule_off_does_nothing():
    m, gid, gtr = _manager(300.0, [])
    gtr.add_camera_observation("cam-1", 5, 30.0 + 120.0)
    m._close_if_reappeared(gid, "cam-1", 5)
    assert m.pop_exit_events() == []
    m, gid, gtr = _manager(0.0, [])
    gtr.add_camera_observation("cam-1", 5, 30.0 + 900.0)
    m._close_if_reappeared(gid, "cam-1", 5)
    assert m.pop_exit_events() == []


def test_a_vehicle_that_went_inside_closes_through_the_interior_rule_not_the_absence_rule():
    m, gid, gtr = _manager(300.0, ["cam-3"])
    gtr.add_camera_observation("cam-1", 5, 30.0 + 900.0)
    m._close_if_reappeared(gid, "cam-1", 5)
    ev = m.pop_exit_events()
    assert len(ev) == 1 and "reopens" not in ev[0] and gtr.exited
