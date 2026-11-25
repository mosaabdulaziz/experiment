import json
import uuid

from app.truck_feed import _iso, TruckFeedBridge, gid_to_plate, plate_to_gid, tenant_uuid

# plates_dir points nowhere: left to its default ("data/plates", relative to the working
# directory) these tests read whatever plates a real run left in this checkout, and fail
# because of them.
CFG = {"enabled": True, "company_id": "El-Mohandes", "branch_id": "BRANCH-001",
       "camera_ips": {"cam-01": "10.0.0.1", "cam-02": "10.0.0.2"},
       "plates_dir": "/nonexistent/plates"}


def _by_ip(batch):
    out = {}
    for c in batch["cameras"]:
        out.setdefault(c["ip"], []).extend(c["plates"])
    return out


def _v(first, last):
    return {"wall_first": first, "wall_last": last}


def test_batch_lists_only_recently_seen_tracked_vehicles():
    b = TruckFeedBridge(CFG, "redis://x")
    states = {"cam-01": {0: _v(100.0, 108.0), 7: _v(100.0, 108.0), 8: _v(90.0, 95.0), -1: _v(100.0, 108.0)},
              "cam-02": {}}
    batch = b.build_batch(states, now=110.0)
    cams = _by_ip(batch)
    assert [p["plate"] for p in cams["10.0.0.1"]] == [gid_to_plate(0), gid_to_plate(7)]  # 8 stale, -1 untracked, id 0 is a real id
    assert cams["10.0.0.2"] == []  # empty list keeps the feed alive
    assert batch["company_id"] == tenant_uuid("El-Mohandes")
    assert batch["branch_id"] == tenant_uuid("BRANCH-001")
    assert uuid.UUID(batch["company_id"]) and batch["company_id"] != batch["branch_id"]
    assert [c for c in batch["cameras"] if c["plates"]][0]["plates"][0]["first_seen_at"].endswith("Z")
    # each vehicle is stamped with when it was really last seen (108.0), not the push time
    e = [c for c in batch["cameras"] if c["plates"]][0]
    assert e["observed_at"] == _iso(108.0)
    json.dumps(batch)


def test_batch_ids_unique():
    b = TruckFeedBridge(CFG, "redis://x")
    assert b.build_batch({}, 1.0)["batch_id"] != b.build_batch({}, 1.0)["batch_id"]


def test_disabled_without_tenant_or_cameras():
    assert not TruckFeedBridge({**CFG, "company_id": ""}, "r").enabled
    assert not TruckFeedBridge({**CFG, "camera_ips": {}}, "r").enabled


def test_real_uuid_passes_through():
    u = "cfcc819d-bc99-4b20-9ed3-38ff3a983b59"
    assert tenant_uuid(u) == u and tenant_uuid("") == ""


def test_plate_encoding_is_unique_reversible_and_platform_parsable():
    import re
    seen = set()
    for g in [1, 22, 9999, 10000, 123456, 5831 * 10000 + 9999]:
        p = gid_to_plate(g)
        assert plate_to_gid(p) == g and p not in seen
        seen.add(p)
        # same rules as truck-tracking parsePlate: 2-3 Arabic letters + 3-4 digits
        assert re.search(r"[\u0621-\u064A](?:\s*[\u0621-\u064A]){1,2}", p)
        assert re.search(r"[0-9]{3,4}", p)


def test_every_camera_reported_oldest_first_and_only_when_advanced():
    b = TruckFeedBridge(CFG, "redis://x")
    # gid 7 seen on both cameras recently: both sightings are reported, oldest first,
    # so the newest (cam-02) is what places the vehicle now
    states = {"cam-01": {7: _v(100.0, 107.0)}, "cam-02": {7: _v(105.0, 108.0)}}
    batch = b.build_batch(states, now=109.0)
    assert [(c["ip"], c["observed_at"]) for c in batch["cameras"] if c["plates"]] == [
        ("10.0.0.1", _iso(107.0)), ("10.0.0.2", _iso(108.0))]
    b._sent.update(b._marks)  # what a successful push records
    # nothing advanced -> only keep-alives
    again = b.build_batch(states, now=110.0)
    assert all(c["plates"] == [] for c in again["cameras"])
    # cam-02 advances -> just that one goes out
    states["cam-02"][7]["wall_last"] = 110.0
    third = b.build_batch(states, now=110.5)
    assert [c["ip"] for c in third["cameras"] if c["plates"]] == ["10.0.0.2"]


def test_pending_late_sighting_goes_out_once_even_when_old():
    b = TruckFeedBridge(CFG, "redis://x")
    late = {"cam-01": {9: {**_v(50.0, 55.0), "pending": True}}, "cam-02": {}}
    cams = _by_ip(b.build_batch(late, now=109.0))
    assert [p["plate"] for p in cams["10.0.0.1"]] == [gid_to_plate(9)]
    b._sent.update(b._marks)
    assert _by_ip(b.build_batch(late, now=109.0))["10.0.0.1"] == []


def test_exit_is_reported_once_and_silences_the_vehicle_afterwards():
    b = TruckFeedBridge({**CFG, "exit_hold_s": 0}, "redis://x")
    states = {"cam-01": {7: _v(100.0, 108.0)}, "cam-02": {}}
    b.note_exit(7, "cam-01", 107.0)
    batch = b.build_batch(states, now=109.0)
    marks = [c for c in batch["cameras"] if c["plates"]]
    assert len(marks) == 1                       # the exit marker only; the sighting is dropped
    m = marks[0]
    assert m["ip"] == "10.0.0.1" and m["observed_at"] == _iso(108.0)     # when it was last seen
    assert m["plates"][0]["plate"] == gid_to_plate(7) and m["plates"][0]["exited"] is True
    b._sent.update(b._marks)
    for gid in b._exit_marks:
        b._exits[gid][2] = True                  # what a successful push records
    again = b.build_batch({"cam-01": {7: _v(100.0, 112.0)}, "cam-02": {}}, now=113.0)
    assert all(c["plates"] == [] for c in again["cameras"])      # never reported again
    # a vehicle that comes back in is a new one (new id) and is reported normally
    back = b.build_batch({"cam-01": {8: _v(120.0, 124.0)}, "cam-02": {}}, now=125.0)
    assert [p["plate"] for c in back["cameras"] for p in c["plates"]] == [gid_to_plate(8)]



def test_cameras_of_a_group_are_one_place_reported_under_the_camera_that_saw_it_first():
    b = TruckFeedBridge(CFG, "redis://x", {"gate": ["cam-01", "cam-02"]})
    # on both entrance cameras at once: one entry, under the one that saw it first
    states = {"cam-01": {7: _v(103.0, 107.0)}, "cam-02": {7: _v(100.0, 108.0)}}
    batch = b.build_batch(states, now=109.0)
    sent = [(c["ip"], c["observed_at"], c["plates"][0]["first_seen_at"]) for c in batch["cameras"] if c["plates"]]
    assert sent == [("10.0.0.2", _iso(108.0), _iso(100.0))]
    b._sent.update(b._marks)
    # later only cam-01 sees it: still the same place, still reported under cam-02
    states = {"cam-01": {7: _v(103.0, 112.0)}, "cam-02": {7: _v(100.0, 108.0)}}
    again = b.build_batch(states, now=113.0)
    assert [(c["ip"], c["observed_at"]) for c in again["cameras"] if c["plates"]] == [("10.0.0.2", _iso(112.0))]


def test_cameras_outside_any_group_stay_separate_places():
    b = TruckFeedBridge(CFG, "redis://x", {"gate": ["cam-09"]})
    states = {"cam-01": {7: _v(100.0, 107.0)}, "cam-02": {7: _v(105.0, 108.0)}}
    batch = b.build_batch(states, now=109.0)
    assert [c["ip"] for c in batch["cameras"] if c["plates"]] == ["10.0.0.1", "10.0.0.2"]


def test_a_vehicle_that_waits_at_the_gate_stays_seen_until_it_is_out_of_sight():
    b = TruckFeedBridge(CFG, "redis://x")                       # exit_hold_s defaults to 30
    b.note_exit(7, "cam-01", 107.0)                             # the direction check called it "leaving"
    # still in view 10 s later: its sighting goes on being reported and no exit is sent yet
    batch = b.build_batch({"cam-01": {7: _v(100.0, 117.0)}, "cam-02": {}}, now=118.0)
    sent = [p for c in batch["cameras"] for p in c["plates"]]
    assert [p["plate"] for p in sent] == [gid_to_plate(7)] and "exited" not in sent[0]
    b._sent.update(b._marks)
    # last seen at 117, 20 s later: still inside the hold
    batch = b.build_batch({"cam-01": {7: _v(100.0, 117.0)}, "cam-02": {}}, now=137.0)
    assert all(c["plates"] == [] for c in batch["cameras"])
    # 30 s of nothing: the exit is reported, stamped when it was last seen
    batch = b.build_batch({"cam-01": {7: _v(100.0, 117.0)}, "cam-02": {}}, now=148.0)
    marks = [c for c in batch["cameras"] if c["plates"]]
    assert len(marks) == 1 and marks[0]["observed_at"] == _iso(117.0) and marks[0]["plates"][0]["exited"] is True


def _alert(**kw):
    a = {"kind": "restricted_zone", "gid": None, "obj_id": 3, "cam_id": "cam-01", "zone_id": "z1",
         "zone_name": "Dock 3", "dwell_s": 61.4, "threshold_s": 60.0, "since": 100.0, "at": 161.0,
         "key": "restricted:z1:cam-01:3:100:1", "update": False}
    a.update(kw)
    return a


def _push_ok(b):
    b._alert_told.update(b._alert_marks)


def test_a_restricted_zone_alert_of_an_untracked_vehicle_goes_out_once():
    b = TruckFeedBridge(CFG, "redis://x")
    b.note_alert(_alert())
    batch = b.build_batch({"cam-01": {}, "cam-02": {}}, now=162.0)
    (a,) = batch["alerts"]
    assert a["ip"] == "10.0.0.1" and a["zone_name"] == "Dock 3" and "plate" not in a      # no truck known
    assert (a["dwell_seconds"], a["threshold_seconds"]) == (61, 60)
    assert a["dedup_key"] == "restricted:z1:cam-01:3:100:1" and a["observed_at"] == _iso(161.0)
    json.dumps(batch)
    _push_ok(b)
    assert "alerts" not in b.build_batch({"cam-01": {}, "cam-02": {}}, now=170.0)


def test_the_alert_is_sent_again_with_the_truck_and_then_its_plate_once_known(tmp_path):
    import json as _j
    b = TruckFeedBridge({**CFG, "plates_dir": str(tmp_path)}, "redis://x")
    b.note_alert(_alert())
    b.build_batch({"cam-01": {}, "cam-02": {}}, now=162.0); _push_ok(b)
    b.note_alert(_alert(gid=15, update=True))                                   # matched to truck 15
    (a,) = b.build_batch({"cam-01": {}, "cam-02": {}}, now=165.0)["alerts"]
    assert a["plate"] == gid_to_plate(15) and "read_plate" not in a and a["dedup_key"] == "restricted:z1:cam-01:3:100:1"
    _push_ok(b)
    assert "alerts" not in b.build_batch({"cam-01": {}, "cam-02": {}}, now=170.0)
    (tmp_path / "15_reads.json").write_text(_j.dumps([{"cam": "cam-02", "text": "7192طفب", "ocr_conf": 0.8}]))
    (b2,) = b.build_batch({"cam-01": {}, "cam-02": {}}, now=175.0)["alerts"]       # the plate got read
    assert b2["read_plate"] == "ب ف ط 7192"


class _FakeRedis:
    def __init__(self, sent=()):
        self.sent = set(sent)
    def smembers(self, key): return list(self.sent)
    def sadd(self, key, *vals): self.sent.update(vals)
    def srem(self, key, *vals): self.sent.difference_update(vals)


def test_after_a_restart_a_vehicle_that_left_stays_silent_and_an_unreported_exit_is_still_due():
    b = TruckFeedBridge(CFG, "redis://x")
    b._r = _FakeRedis(sent={7})                                    # exit 7 was reported before the restart; 8 was not
    b.note_exit(7, "cam-01", 50.0, replay=True)
    b.note_exit(8, "cam-01", 50.0, replay=True)
    b._resolve_replays()
    # 7 left: whatever the cameras still see of it is not news; 8 is still due and goes out (out of sight for > 30 s)
    batch = b.build_batch({"cam-01": {7: _v(100.0, 190.0), 8: _v(100.0, 100.0)}, "cam-02": {}}, now=200.0)
    sent = [p for c in batch["cameras"] for p in c["plates"]]
    assert [(p["plate"], p.get("exited")) for p in sent] == [(gid_to_plate(8), True)]


def test_a_plate_reading_does_not_bring_back_a_vehicle_that_has_left(tmp_path):
    import json as _j
    b = TruckFeedBridge({**CFG, "plates_dir": str(tmp_path)}, "redis://x")
    b._exits[7] = ["cam-01", 50.0, True, False]                    # left, reported
    b._sent[("cam-01", 7)] = 40.0
    (tmp_path / "7_reads.json").write_text(_j.dumps([{"cam": "cam-01", "text": "1234ط", "ocr_conf": 0.8}]))
    assert b._reading_updates(set()) == []


def test_aggregator_stays_survive_a_save_and_load():
    from app.tracking import CameraAggregator
    agg = CameraAggregator(["cam-01", "cam-02"])
    agg.process_entry("cam-01", [["car", 0.9, 7, 1, 2, 3, 4]], {"7": 5}, pts=10.0, wall=1000.0)
    agg.process_entry("cam-01", [["car", 0.9, 7, 1, 2, 3, 4]], {"7": 5}, pts=400.0, wall=1400.0)   # a second stay
    state = json.loads(json.dumps(agg.export_state()))
    fresh = CameraAggregator(["cam-01", "cam-02"])
    assert fresh.import_state(state) == 1
    assert [(e["first"], e["last"]) for e in fresh.episodes_of(5)] == [(1000.0, 1000.0), (1400.0, 1400.0)]
    # replayed (older) history only extends a restored stay, it never shortens it
    fresh.process_entry("cam-01", [["car", 0.9, 7, 1, 2, 3, 4]], {"7": 5}, pts=410.0, wall=1300.0)
    assert fresh.episodes_of(5)[-1]["last"] == 1400.0
    assert "pending" not in fresh._camera_vehicles["cam-01"][5]
