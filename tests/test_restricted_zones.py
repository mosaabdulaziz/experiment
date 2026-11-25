import os

from app.restricted_zones import RestrictedZoneMonitor, load_zones, save_zones

SQUARE = [(0.4, 0.4), (0.8, 0.4), (0.8, 0.9), (0.4, 0.9)]


def _mon(tmp_path, threshold=30, **kw):
    p = str(tmp_path / "rz.json")
    save_zones(p, "cam-01", [{"id": "z1", "name": "Dock 3", "threshold_s": threshold, "points": SQUARE}])
    return RestrictedZoneMonitor(p, 1000, 1000, **kw)


def _det(obj=3, x=500, y=600, w=200, h=250, label="vehicle"):
    """bottom-centre = (x + w/2, y + h) = (600, 850): inside the square."""
    return [label, 0.9, obj, x, y, w, h]


def test_save_and_load_round_trip(tmp_path):
    p = str(tmp_path / "rz.json")
    save_zones(p, "cam-01", [{"id": "a", "name": "A", "threshold_s": 20, "points": SQUARE}])
    z = load_zones(p)["cam-01"][0]
    assert z["id"] == "a" and z["threshold_s"] == 20 and len(z["points"]) == 4
    save_zones(p, "cam-01", [])
    assert load_zones(p) == {}


def test_a_vehicle_seen_by_this_camera_alone_alerts_without_any_global_id(tmp_path):
    m = _mon(tmp_path, 30)
    assert m.observe("cam-01", [_det()], {}, 100.0) == []
    assert m.observe("cam-01", [_det()], {}, 120.0) == []                # 20 s: not yet
    fired = m.observe("cam-01", [_det()], {}, 131.0)                     # 31 s
    assert len(fired) == 1
    a = fired[0]
    assert (a["gid"], a["obj_id"], a["cam_id"], a["zone_name"], a["update"]) == (None, 3, "cam-01", "Dock 3", False)
    assert 30 <= a["dwell_s"] <= 32 and a["key"].startswith("restricted:z1:cam-01:3:")
    assert m.observe("cam-01", [_det()], {}, 150.0) == []                # once per stay


def test_outside_the_polygon_or_another_camera_or_not_a_vehicle_never_alerts(tmp_path):
    m = _mon(tmp_path, 5)
    for t in range(100, 140):
        assert m.observe("cam-01", [_det(x=0, y=0, w=100, h=100)], {}, float(t)) == []
        assert m.observe("cam-02", [_det()], {}, float(t)) == []
        assert m.observe("cam-01", [_det(label="person")], {}, float(t)) == []


def test_a_short_dropout_keeps_the_clock_but_leaving_restarts_it(tmp_path):
    m = _mon(tmp_path, 30, grace_s=5.0)
    m.observe("cam-01", [_det()], {}, 100.0)
    m.observe("cam-01", [], {}, 103.0); m.expire(103.0)                  # 3 s without it: inside the grace
    assert len(m.observe("cam-01", [_det()], {}, 131.0)) == 1            # clock kept: 31 s since 100
    m.expire(140.0)                                                      # gone for 9 s: the stay ended
    m.observe("cam-01", [_det()], {}, 141.0)
    assert m.observe("cam-01", [_det()], {}, 160.0) == []                # a new stay, 19 s
    assert len(m.observe("cam-01", [_det()], {}, 172.0)) == 1            # alerts again after the threshold


def test_it_is_still_reported_again_after_the_cooldown_if_it_never_leaves(tmp_path):
    m = _mon(tmp_path, 10, cooldown_s=600.0)
    assert len(m.observe("cam-01", [_det()], {}, 100.0 + 11)) == 0       # first sight: clock starts at 111
    m.observe("cam-01", [_det()], {}, 111.0)
    assert len(m.observe("cam-01", [_det()], {}, 122.0)) == 1
    t = 122.0
    while t < 700.0:
        t += 4.0
        assert m.observe("cam-01", [_det()], {}, t) == []
    again = m.observe("cam-01", [_det()], {}, 725.0)
    assert len(again) == 1 and again[0]["key"].endswith(":2")


def test_the_alert_is_enriched_when_the_vehicle_turns_out_to_be_a_tracked_truck(tmp_path):
    m = _mon(tmp_path, 30)
    m.observe("cam-01", [_det()], {}, 100.0)
    (alert,) = m.observe("cam-01", [_det()], {}, 131.0)
    assert alert["gid"] is None
    # the sidecar matches it to a global id a little later: the SAME alert is sent again with the id
    (more,) = m.observe("cam-01", [_det()], {"3": 15}, 140.0)
    assert more["update"] is True and more["gid"] == 15 and more["key"] == alert["key"]
    assert m.observe("cam-01", [_det()], {"3": 15}, 150.0) == []         # and not again


def test_the_clock_survives_the_vehicle_getting_a_global_id_or_a_new_tracker_id(tmp_path):
    m = _mon(tmp_path, 30)
    m.observe("cam-01", [_det(obj=3)], {}, 100.0)
    assert m.observe("cam-01", [_det(obj=3)], {"3": 15}, 115.0) == []    # known as truck 15 now, still 15 s in
    # the tracker lost it and gave it a new id; the global id is the same: the stay goes on
    fired = m.observe("cam-01", [_det(obj=9)], {"9": 15}, 131.0)
    assert len(fired) == 1 and fired[0]["gid"] == 15 and 30 <= fired[0]["dwell_s"] <= 32


def test_edited_zone_file_is_picked_up_and_a_deleted_zone_forgets_its_stays(tmp_path):
    m = _mon(tmp_path, 30)
    m.observe("cam-01", [_det()], {}, 100.0)
    save_zones(m.path, "cam-01", [])
    os.utime(m.path, (2e9, 2e9)) if os.path.exists(m.path) else None
    assert m.observe("cam-01", [_det()], {}, 140.0) == []
    assert m._stays == {}
