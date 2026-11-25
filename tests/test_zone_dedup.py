"""ZoneReentryGrace: zone-alert dedup that survives tracker re-IDs.

The sidecar itself needs gi/pyds (GPU container only), but zone_dedup is pure
Python and importable off-GPU (same convention as test_sidecar_sources.py)."""

from perception.modules.zones.dedup import ZoneReentryGrace

# A person-sized bbox (mux coords); diag ≈ 206.
BOX = (600.0, 300.0, 50.0, 200.0)


def _near(box, dx=10.0, dy=10.0):
    left, top, w, h = box
    return (left + dx, top + dy, w, h)


def _far(box):
    left, top, w, h = box
    return (left + 900.0, top, w, h)


def test_first_appearance_alerts_once():
    g = ZoneReentryGrace()
    assert g.observe(0.0, [(1, BOX)]) == [1]
    assert g.observe(0.5, [(1, BOX)]) == []


def test_reid_near_old_spot_within_grace_does_not_realert():
    g = ZoneReentryGrace(grace_seconds=5.0)
    assert g.observe(0.0, [(1, BOX)]) == [1]
    # detector dropout: frames 1s..3s show nobody in-zone
    assert g.observe(1.0, []) == []
    # re-detected 3s later with a new tracker id, roughly in place
    assert g.observe(3.0, [(2, _near(BOX))]) == []
    # and the inherited id keeps not re-firing
    assert g.observe(3.5, [(2, _near(BOX))]) == []


def test_new_id_far_away_alerts():
    g = ZoneReentryGrace()
    assert g.observe(0.0, [(1, BOX)]) == [1]
    assert g.observe(1.0, []) == []
    assert g.observe(2.0, [(2, _far(BOX))]) == [2]


def test_new_id_after_grace_expiry_alerts():
    g = ZoneReentryGrace(grace_seconds=5.0)
    assert g.observe(0.0, [(1, BOX)]) == [1]
    assert g.observe(1.0, []) == []
    assert g.observe(7.0, [(2, _near(BOX))]) == [2]


def test_second_person_entering_while_first_present_alerts():
    g = ZoneReentryGrace()
    assert g.observe(0.0, [(1, BOX)]) == [1]
    # id 1 is still visible, so id 2 can't be a re-badge of it even nearby
    assert g.observe(1.0, [(1, BOX), (2, _near(BOX, dx=60.0))]) == [2]


def test_chained_reids_alert_once_total():
    g = ZoneReentryGrace(grace_seconds=5.0)
    assert g.observe(0.0, [(1, BOX)]) == [1]
    assert g.observe(2.0, [(2, _near(BOX))]) == []
    assert g.observe(3.0, []) == []
    assert g.observe(4.5, [(3, _near(BOX, dx=-10.0))]) == []


def test_one_vanished_entry_cannot_absolve_two_new_ids():
    g = ZoneReentryGrace()
    assert g.observe(0.0, [(1, BOX)]) == [1]
    assert g.observe(1.0, []) == []
    # two new ids appear near the vanished spot; only one inherits
    fired = g.observe(2.0, [(2, _near(BOX)), (3, _near(BOX, dx=20.0))])
    assert len(fired) == 1


def test_same_id_never_refires_even_after_long_absence():
    g = ZoneReentryGrace(grace_seconds=5.0)
    assert g.observe(0.0, [(1, BOX)]) == [1]
    assert g.observe(1.0, []) == []
    # well past grace, but the tracker kept the id -> today's semantics hold
    assert g.observe(60.0, [(1, BOX)]) == []


def test_fresh_instance_realerts():
    g = ZoneReentryGrace()
    assert g.observe(0.0, [(1, BOX)]) == [1]
    # zone redraw resets state by rebinding a new instance
    g = ZoneReentryGrace()
    assert g.observe(1.0, [(1, BOX)]) == [1]
