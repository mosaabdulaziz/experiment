from datetime import datetime

from chair_engine import ChairEngine
from domain import Appearance, FaceCandidate, Frame, TrackObs
from policy import DEFAULT_POLICY
from zones import FrameGeometry

P = {**DEFAULT_POLICY, "register_source": 0.0, "fr_after_s": 5.0}   # chair crops unless a test sets 1 (default 1 since 2026-10-06); 5 s face settle
T0 = datetime(2026, 10, 4, 12, 0).timestamp()


def sq(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


ZONES = {
    "barber-c": [{"id": "c-door", "name": "door", "type": "entrance_line", "points": [[0.4, 0.5], [0.6, 0.5]],
                  "inside": [0.5, 0.8]}],
    "barber-l": [
        {"id": "l-R1", "name": "R1", "type": "chair", "points": sq(0.10, 0.10, 0.30, 0.40), "pair": "R1", "owner": True},
        {"id": "l-R1-seat", "name": "R1 seat", "type": "seat", "points": sq(0.15, 0.25, 0.25, 0.35), "seat_of": "l-R1"}],
}
SITTER = tuple(v for p in sq(0.14, 0.05, 0.26, 0.36) for v in p)


def frames(t0, secs, tracks, cam="barber-l", step=0.5):
    return [Frame(cam, t0 + i * step, tracks) for i in range(int(secs / step))]


def eng():
    return ChairEngine(ZONES, FrameGeometry(1000, 1000), barber_for=lambda c, at: "staff-1")


def test_end_to_end_one_haircut():
    e = eng()
    seated = (TrackObs(4, (0, 0, 1, 1), 0.9, SITTER),)
    ups = [e.observe(frames(T0, 300, seated), [], P)]
    assert e.live()["chairs"]["R1"]["state"] == "sitting"
    ups.append(e.observe(frames(T0 + 300, 70, ()), [], P))
    assert e.live()["chairs"]["R1"]["state"] == "away"
    ups.append(e.observe(frames(T0 + 370, 400, ()), [], P))
    assert e.live()["chairs"]["R1"]["state"] == "empty"
    sessions = {k: v for u in ups for k, v in u.sessions.items()}
    (s,) = sessions.values()
    assert not s.open and abs(s.service_s - 299.5) < 1.0
    assert [r.visit_id for u in ups for r in u.registrations] == [s.visit_id]   # chair crops: one card per haircut


def test_a_still_object_on_the_seat_is_not_a_haircut_but_a_moving_sitter_is():
    """2026-10-18 (user: a bag on VIP2). Boxes come in mux pixels: the engine hands the haircut fractions of the
    frame (the first wiring compared pixels with 0.005 and let the bag through)."""
    policy = {**P, "ce_object_spread": 0.005}
    for jitter, haircut in ((2, False), (20, True)):           # 2 px of 1000 = 0.002 (the bag's), 20 px = 0.02
        e = eng()
        fr = [Frame("barber-l", T0 + i * 0.5, (TrackObs(4, (140 + jitter * (i % 2), 50, 120, 310), 0.9, SITTER),))
              for i in range(800)]
        u = e.observe(fr, [], policy)
        assert bool(u.sessions) is haircut



def test_the_sitters_detection_conf_reaches_the_object_check():
    """2026-10-22 (user: clothes on VIP2): each occupant box goes to the haircut with its track's conf."""
    policy = {**P, "ce_object_conf": 0.72}
    for conf, haircut in ((0.66, False), (0.9, True)):         # the clothes' median, a seated customer's
        e = eng()
        fr = [Frame("barber-l", T0 + i * 0.5, (TrackObs(4, (140 + 20 * (i % 2), 50, 120, 310), conf, SITTER),))
              for i in range(800)]
        u = e.observe(fr, [], policy)
        assert bool(u.sessions) is haircut

def test_staff_sitting_is_not_a_haircut():
    e = eng()
    seated = (TrackObs(4, (0, 0, 1, 1), 0.9, SITTER),)
    apps = [Appearance("barber-l", 4, T0, None, 0.95)]
    u = e.observe(frames(T0, 300, seated), apps, P)
    assert not u.sessions and e.live()["chairs"]["R1"]["state"] == "empty"


def test_door_crossing_reaches_the_update():
    e = eng()
    fr = [Frame("barber-c", T0, (TrackObs(9, (480, 300, 40, 100), 0.9, None),)),
          Frame("barber-c", T0 + 1, (TrackObs(9, (480, 500, 40, 100), 0.9, None),)),
          Frame("barber-c", T0 + 4, ())]      # the crossing is confirmed once the track is gone
    (c,) = e.observe(fr, [], P).door_crossings
    assert c.direction == "in"


def test_clear_forgets_everything():
    e = eng()
    e.observe(frames(T0, 300, (TrackObs(4, (0, 0, 1, 1), 0.9, SITTER),)), [], P)
    e.clear()
    assert e.live()["chairs"]["R1"]["state"] == "empty" and e.live()["customers"] == []


def test_scene_lists_the_latest_frame_boxes():
    e = eng()
    e.observe(frames(T0, 2, (TrackObs(4, (100, 100, 100, 300), 0.9, SITTER),)), [], P)
    (b,) = e.scene(P)["barber-l"]
    assert b["track_id"] == 4 and b["box"] == [0.1, 0.1, 0.1, 0.3]


def test_a_busy_frame_is_cheap():
    import time
    e = eng()
    tracks = tuple(TrackObs(i, (0, 0, 1, 1), 0.9, SITTER) for i in range(12))
    t = time.perf_counter()
    e.observe(frames(T0, 50, tracks), [], P)       # 100 frames x 12 people
    assert (time.perf_counter() - t) / 100 < 0.02


def _away_engine():
    e = eng()
    seated = (TrackObs(4, (0, 0, 1, 1), 0.9, SITTER),)
    e.observe(frames(T0, 300, seated), [], P)
    e.observe(frames(T0 + 300, 70, ()), [], P)
    assert e.live()["chairs"]["R1"]["state"] == "away"
    return e


def _renamed():
    z = {k: [dict(x) for x in v] for k, v in ZONES.items()}
    z["barber-l"][0].update(id="l-R2", name="R2", pair="R2")
    z["barber-l"][1].update(seat_of="l-R2")
    return z


def test_renaming_a_chair_closes_its_open_haircut():
    e = _away_engine()
    e.set_zones(_renamed())
    u = e.observe(frames(T0 + 370, 2, ()), [], P)
    (s,) = u.sessions.values()
    assert not s.open and [r.visit_id for r in u.registrations] == [s.visit_id]
    assert list(e.live()["chairs"]) == ["R2"] and e.live()["customers"] == []


def test_removing_a_chair_mid_sitting_closes_it():
    e = eng()
    e.observe(frames(T0, 300, (TrackObs(4, (0, 0, 1, 1), 0.9, SITTER),)), [], P)
    e.set_zones({"barber-c": ZONES["barber-c"]})
    u = e.observe([], [], P)
    (s,) = u.sessions.values()
    assert not s.open and abs(s.service_s - 299.5) < 1.0
    assert e.live()["chairs"] == {}


def test_a_stale_camera_shows_no_view():
    e = eng()
    e.observe(frames(T0, 2, (TrackObs(4, (0, 0, 1, 1), 0.9, SITTER),)), [], P)
    e.observe(frames(T0 + 10, 1, (), cam="barber-c"), [], P)
    assert e.live()["chairs"]["R1"]["state"] == "no_view"


def _sessions(ups):
    return {k: v for u in ups for k, v in u.sessions.items()}


def test_a_15s_sit_is_no_customer_even_through_the_end_gap():
    """I-1: the 60 s end gap must not count as sitting time (ce_min_haircut_s 60)."""
    e = eng()
    seated = (TrackObs(4, (0, 0, 1, 1), 0.9, SITTER),)
    ups = [e.observe(frames(T0, 15, seated), [], P)]
    assert e.live()["chairs"]["R1"]["state"] == "sitting"     # the running sit shows on the chair
    for k in range(10):
        ups.append(e.observe(frames(T0 + 15 + k * 50, 50, ()), [], P))
    assert not _sessions(ups) and not any(u.visits for u in ups)
    assert e.live()["customers"] == [] and e.live()["chairs"]["R1"]["state"] == "empty"


def test_a_return_at_the_end_of_the_resume_window_resumes():
    """M-1: a return whose start window straddles the resume window's end is the same haircut."""
    e = eng()
    seated = (TrackObs(4, (0, 0, 1, 1), 0.9, SITTER),)
    ups = [e.observe(frames(T0, 300, seated), [], P)]          # last occupied frame T0 + 299.5
    back = T0 + 299.5 + 355
    for k in range(int((back - T0 - 300) / 5)):
        ups.append(e.observe(frames(T0 + 300 + k * 5, 5, ()), [], P))
    for k in range(20):
        ups.append(e.observe(frames(back + k * 5, 5, seated), [], P))
    for k in range(10):
        ups.append(e.observe(frames(back + 100 + k * 50, 50, ()), [], P))
    (s,) = _sessions(ups).values()
    assert not s.open and s.wash_s > 300


def test_staff_scores_of_tracks_gone_for_long_are_pruned():
    """I-5: the per-track dicts do not grow without bound."""
    e = eng()
    e.observe(frames(T0, 2, (TrackObs(4, (0, 0, 1, 1), 0.9, SITTER),)),
              [Appearance("barber-l", 4, T0, None, 0.95), Appearance("barber-l", 5, T0 + 1, None, 0.1)], P)
    assert ("barber-l", 4) in e.staff and ("barber-l", 5) in e.staff
    other = (TrackObs(5, (900, 900, 10, 10), 0.9, None),)
    for k in range(13):
        e.observe(frames(T0 + 2 + k * 60, 60, other, step=5.0), [], P)
    assert ("barber-l", 4) not in e.staff          # unseen > 10 min on its camera
    assert ("barber-l", 5) in e.staff              # still in view


def test_a_stalled_camera_keeps_its_staff_scores():
    e = eng()
    e.observe(frames(T0, 2, (TrackObs(4, (0, 0, 1, 1), 0.9, SITTER),)), [Appearance("barber-l", 4, T0, None, 0.95)], P)
    for k in range(13):
        e.observe(frames(T0 + 2 + k * 60, 60, (), cam="barber-c", step=5.0), [], P)
    assert ("barber-l", 4) in e.staff              # barber-l sent nothing since: no time passed for it


def test_frames_without_masks_warn_once_a_minute(caplog):
    """M-4: a sidecar publishing no person_masks gives silent zero customers otherwise."""
    import logging
    e = eng()
    bare = (TrackObs(4, (0, 0, 1, 1)),)
    with caplog.at_level(logging.WARNING, logger="barber.chair_engine"):
        e.observe(frames(T0, 90, bare), [], P)
    warns = [r for r in caplog.records if "person_masks" in r.getMessage()]
    assert len(warns) == 2 and "barber-l" in warns[0].getMessage()


def _rules(tracks, staff=None, policy=P):
    e = eng()
    apps = [Appearance("barber-l", tid, T0, None, sp) for tid, sp in (staff or {}).items()]
    e.observe(frames(T0, 2, tuple(tracks)), apps, policy)
    return e, {r["track_id"]: r["rule"] for r in e.scene(policy)["barber-l"]}


def _tr(tid, poly=SITTER, conf=0.9):
    return TrackObs(tid, (0, 0, 1, 1), conf, poly)


def test_scene_rule_reaches_every_why():
    far = tuple(v for p in sq(0.6, 0.5, 0.7, 0.9) for v in p)
    small = tuple(v for p in sq(0.15, 0.20, 0.25, 0.36) for v in p)
    _, r = _rules([_tr(1), _tr(2, small), _tr(3, conf=0.1), TrackObs(4, (0, 0, 1, 1), 0.9, None),
                   _tr(5, far), _tr(6, far), _tr(7)], staff={6: 0.95, 7: 0.95})
    assert r[1]["why"] == "other_chair" or r[1]["why"] == "occupied"
    whys = {k: v["why"] for k, v in r.items()}
    assert whys[3] == "low_conf" and whys[4] == "no_mask" and whys[5] == "no_chair" and whys[6] == "staff"
    assert whys[7] == "staff_cover" and r[7]["chair"] == "R1"
    assert sorted([whys[1], whys[2]]) == ["occupied", "other_chair"]
    occ = r[1] if whys[1] == "occupied" else r[2]
    assert occ["chair"] == "R1" and 0 < occ["share"] <= 1


def test_scene_rule_below_min():
    _, r = _rules([_tr(1)], policy={**P, "ce_seat_min": 1.01})
    assert r[1]["why"] == "below_min" and r[1]["chair"] == "R1"


def test_scene_lower_y_is_the_lower_cut_of_the_mask():
    e, _ = _rules([_tr(1), TrackObs(2, (0, 0, 1, 1), 0.9, None)])
    rows = {r["track_id"]: r for r in e.scene(P)["barber-l"]}
    assert abs(rows[1]["lower_y"] - (0.36 - P["ce_lower_frac"] * 0.31)) < 0.003
    assert rows[2]["lower_y"] is None


def test_centre_camera_entries_have_lower_y_but_no_rule():
    e = eng()
    e.observe(frames(T0, 1, (_tr(9),), cam="barber-c"), [], P)
    (r,) = e.scene(P)["barber-c"]
    assert "rule" not in r and r["lower_y"] is not None


def test_live_chair_has_camera_occupant_cover_and_resume_until():
    e = eng()
    e.observe(frames(T0, 300, (_tr(4),)), [], P)
    c = e.live()["chairs"]["R1"]
    assert c["camera"] == "barber-l" and c["occupant"] == 4 and c["cover"] is False and c["resume_until"] is None
    e.observe(frames(T0 + 300, 70, ()), [], P)
    c = e.live()["chairs"]["R1"]
    assert c["state"] == "away" and c["occupant"] is None
    end = datetime.fromisoformat(c["resume_until"]).timestamp()
    assert abs(end - (T0 + 299.5 + P["ce_resume_window_s"])) < 2


def test_live_chair_cover_flag():
    e = eng()
    e.observe(frames(T0, 2, (_tr(5, tuple(v for p in sq(0.12, 0.05, 0.28, 0.38) for v in p)),)),
              [Appearance("barber-l", 5, T0, None, 0.95)], P)
    c = e.live()["chairs"]["R1"]
    assert c["cover"] is True and c["occupant"] is None


def _wzones():
    z = {cam: list(v) for cam, v in ZONES.items()}
    z["barber-c"] = z["barber-c"] + [{"id": "c-sofa", "name": "sofa", "type": "waiting",
                                       "points": [[0.5, 0.5], [0.9, 0.5], [0.9, 0.9], [0.5, 0.9]]}]
    return z


SOFA = tuple(v for p in [[0.6, 0.4], [0.7, 0.4], [0.7, 0.8], [0.6, 0.8]] for v in p)


def test_centre_camera_frames_are_counted():
    e = ChairEngine(_wzones(), FrameGeometry(1000, 1000))
    e.observe(frames(T0, 30, (TrackObs(7, (0, 0, 1, 1), 0.9, SOFA),), cam="barber-c"), [], P)
    live = e.live()
    assert live["customers_waiting"] == 1 and live["waiting"]["cameras"]["barber-c"] == 1


def test_a_seated_customer_is_not_waiting():
    z = _wzones()
    z["barber-l"] = z["barber-l"] + [{"id": "l-waiting-1", "name": "w", "type": "waiting",
                                       "points": [[0.10, 0.10], [0.30, 0.10], [0.30, 0.40], [0.10, 0.40]]}]
    e = ChairEngine(z, FrameGeometry(1000, 1000))
    e.observe(frames(T0, 60, (TrackObs(4, (0, 0, 1, 1), 0.9, SITTER),)), [], P)
    assert e.live()["waiting"]["cameras"].get("barber-l", 0) == 0


def test_no_frames_means_no_waiting_number():
    assert eng().live()["customers_waiting"] is None


def test_waiting_counts_only_the_entrance_line_camera():
    z = _wzones()
    z["barber-l"] = z["barber-l"] + [{"id": "l-waiting-1", "name": "w", "type": "waiting",
                                       "points": [[0.5, 0.5], [0.9, 0.5], [0.9, 0.9], [0.5, 0.9]]}]
    e = ChairEngine(z, FrameGeometry(1000, 1000))
    assert e.waiting.camera == e.door_cam
    e.observe(frames(T0, 30, (TrackObs(7, (0, 0, 1, 1), 0.9, SOFA),), cam="barber-l"), [], P)
    live = e.live()
    assert live["waiting"]["cameras"]["barber-l"] == 1 and live["customers_waiting"] is None


def _flicker_return(chain_s):
    """A 300 s haircut (last occupied T0 + 299.5), away, then back 340 s after leaving with a flickering start
    (2 s seen, 12 s hidden behind the barber, x4: the chair drops to empty between) and seated from +396 s for 90 s:
    the start confirms > ce_resume_window_s (360) after leaving, but the chain began within it."""
    pol = {**P, "ce_start_chain_s": chain_s}
    e = eng()
    seated = (TrackObs(4, (0, 0, 1, 1), 0.9, SITTER),)
    ups = [e.observe(frames(T0, 300, seated), [], pol)]
    back = T0 + 299.5 + 340
    t = T0 + 300
    while t < back:
        ups.append(e.observe(frames(t, 1, ()), [], pol))
        t += 1
    t = back
    for _ in range(4):
        for secs, tr in ((2, seated), (12, ())):
            for _ in range(secs):
                ups.append(e.observe(frames(t, 1, tr), [], pol))
                t += 1
    for _ in range(90):
        ups.append(e.observe(frames(t, 1, seated), [], pol))
        t += 1
    for k in range(10):
        ups.append(e.observe(frames(t + k * 50, 50, ()), [], pol))
    return _sessions(ups)


def test_start_chain_keeps_an_away_haircut_taken_and_resumes_it():
    (s,) = _flicker_return(45.0).values()
    assert not s.open and abs(s.wash_s - 340.0) < 1.5


def test_without_the_start_chain_the_flickering_return_is_a_new_customer():
    assert len(_flicker_return(0.0)) == 2


def test_scene_rule_outside_chair():
    by = tuple(v for p in [[0.0, 0.0], [0.12, 0.0], [0.12, 0.25], [0.18, 0.25], [0.18, 0.35], [0.0, 0.35]]
               for v in p)
    _, r = _rules([_tr(1, by)], policy={**P, "ce_seat_inside_min": 0.30})
    assert r[1]["why"] == "outside_chair" and r[1]["chair"] == "R1"


def test_stray_frames_do_not_keep_an_away_haircut_forever():
    """Review fix 1: lone occupied frames 30 s apart for 20 min after a customer left (a barber's track with a
    customer's staff_p) must not keep the haircut taken; the next real sit is a second haircut, the first closed."""
    e = eng()
    seated = (TrackObs(4, (0, 0, 1, 1), 0.9, SITTER),)
    ups = [e.observe(frames(T0, 300, seated), [], P)]
    ups.append(e.observe(frames(T0 + 300, 90, ()), [], P))      # the sitting ends (end gap 60 s): away
    assert e.live()["chairs"]["R1"]["state"] == "away"
    t = T0 + 390
    while t < T0 + 390 + 1200:
        ups.append(e.observe(frames(t, 0.5, seated), [], P))
        ups.append(e.observe(frames(t + 0.5, 29.5, ()), [], P))
        t += 30
    for k in range(10):
        ups.append(e.observe(frames(t + k * 10, 10, seated), [], P))
    first = [s for u in ups for s in u.sessions.values() if s.started_at.timestamp() < T0 + 1]
    assert first and not first[-1].open
    assert len(_sessions(ups)) == 2


def _rzones():
    z = _wzones()
    z["barber-c"] = z["barber-c"] + [{"id": "c-row-l", "name": "left row", "type": "row",
                                       "points": sq(0.0, 0.5, 0.3, 0.9)}]
    return z


def _waiter(tid=7, x0=0.6):
    return TrackObs(tid, (x0 * 1000, 400, 100, 400), 0.9,
                    tuple(v for p in sq(x0, 0.4, x0 + 0.1, 0.8) for v in p))


def test_a_waiting_customer_is_live_with_a_ticket_and_a_scene_marker():
    from domain import dt
    e = ChairEngine(_rzones(), FrameGeometry(1000, 1000))
    u = e.observe(frames(T0, 12, (_waiter(),), cam="barber-c"), [], P)
    (r,) = u.waits.values()
    live = e.live()
    (c,) = [c for c in live["customers"] if c["state"] == "waiting"]
    assert (c["visit_id"], c["ticket"], c["waiting_started_at"], c["wait_target_s"], c["chair_id"]) == \
        (r.id, 1, dt(T0).isoformat(timespec="seconds"), 900.0, None)
    assert live["waits"] == [{"id": r.id, "track_id": 7, "started_at": c["waiting_started_at"], "on_hold": None}]
    (row,) = [x for x in e.scene(P)["barber-c"] if x["track_id"] == 7]
    assert (row["person_id"], row["state"], row["ticket"]) == (r.id, "waiting", 1)


def test_reaching_a_row_ends_the_wait_into_the_update():
    e = ChairEngine(_rzones(), FrameGeometry(1000, 1000))
    e.observe(frames(T0, 12, (_waiter(),), cam="barber-c"), [], P)
    u = e.observe(frames(T0 + 12, 4, (_waiter(x0=0.1),), cam="barber-c"), [], P)
    (r,) = u.waits.values()
    assert r.ended_by == "chair"
    assert not [c for c in e.live()["customers"] if c["state"] == "waiting"]


def test_a_stalled_centre_camera_never_ends_a_wait():
    e = ChairEngine(_rzones(), FrameGeometry(1000, 1000))
    e.observe(frames(T0, 12, (_waiter(),), cam="barber-c"), [], P)
    u = e.observe(frames(T0 + 12, 30, ()), [], P)          # only barber-l frames for 30 s
    assert not u.waits and not u.alerts
    (w,) = e.waits.open.values()
    assert w.hold is None


def test_saving_zones_keeps_open_waits():
    e = ChairEngine(_rzones(), FrameGeometry(1000, 1000))
    e.observe(frames(T0, 12, (_waiter(),), cam="barber-c"), [], P)
    e.set_zones(_rzones())
    assert len(e.waits.open) == 1 and e.waits.camera == "barber-c"
    e.clear()
    assert not e.waits.open


def test_scene_survives_a_waiting_track_followed_by_others():
    e = ChairEngine(_rzones(), FrameGeometry(1000, 1000))
    other = TrackObs(9, (50, 400, 100, 400), 0.9, tuple(v for p in sq(0.05, 0.4, 0.15, 0.8) for v in p))
    e.observe(frames(T0, 12, (_waiter(), other), cam="barber-c"), [], P)
    e.observe(frames(T0, 12, ()), [], P)
    rows = e.scene(P)
    states = {x["track_id"]: x["state"] for x in rows["barber-c"]}
    assert states[7] == "waiting" and states.get(9) != "waiting"


def test_a_long_wait_alert_carries_its_subject():
    p = dict(P, wait_target_s=20.0)
    e = ChairEngine(_rzones(), FrameGeometry(1000, 1000))
    u = e.observe(frames(T0, 35, (_waiter(),), cam="barber-c"), [], p)
    (a,) = [a for a in u.alerts if a.kind == "excessive_wait"]
    assert a.subject_cam == "barber-c" and a.subject_box == (0.6, 0.4, 0.1, 0.4)


def test_a_long_haircut_alert_carries_its_chair_box():
    p = dict(P, service_target_s=120.0)
    e = eng()
    seated = (TrackObs(4, (0, 0, 1, 1), 0.9, SITTER),)
    u = e.observe(frames(T0, 200, seated), [], p)
    (a,) = [a for a in u.alerts if a.kind == "excessive_service"]
    x, y, w, h = a.subject_box
    assert a.subject_cam == "barber-l" and abs(x - 0.10) < 0.01 and abs(y - 0.10) < 0.01 and abs(w - 0.20) < 0.01


def test_an_alert_with_no_subject_is_warned_about(caplog):
    import logging
    from domain import Alert, EngineUpdate
    e = eng()
    upd = EngineUpdate(alerts=[Alert("a9", "excessive_wait", "shop", "nobody", None, datetime(2026, 10, 4, 12, 0))])
    with caplog.at_level(logging.WARNING, logger="barber.chair_engine"):
        e._attach_subjects(upd)
    assert "alert a9 (excessive_wait): no subject, no snapshot" in caplog.text


def test_with_chair_crops_door_faces_are_discarded_and_register_nothing():
    from domain import FaceCandidate
    e = eng()
    emb = (1.0,) + (0.0,) * 511
    fr = [Frame("barber-c", T0, (TrackObs(9, (480, 300, 40, 100), 0.9, None),)),
          Frame("barber-c", T0 + 1, (TrackObs(9, (480, 500, 40, 100), 0.9, None),)),
          Frame("barber-c", T0 + 4, ())]
    u = e.observe(fr, [], {**P, "register_source": 0.0},
                  faces=(FaceCandidate("barber-c", 9, T0 + 0.5, "f/a.jpg", emb, 0.9),))
    assert "f/a.jpg" in u.discard_crops
    u2 = e.observe([Frame("barber-c", T0 + 12, ())], [], {**P, "register_source": 0.0})
    assert not u.registrations and not u2.registrations and e.clear() == []


def test_a_door_entry_becomes_a_registration_and_its_crossing_carries_the_entry_id():
    from domain import FaceCandidate
    P1 = {**P, "register_source": 1.0}
    e = eng()
    emb = (1.0,) + (0.0,) * 511
    fr = [Frame("barber-c", T0, (TrackObs(9, (480, 300, 40, 100), 0.9, None),)),
          Frame("barber-c", T0 + 1, (TrackObs(9, (480, 500, 40, 100), 0.9, None),)),
          Frame("barber-c", T0 + 4, ())]
    faces = (FaceCandidate("barber-c", 9, T0 + 0.5, "f/a.jpg", emb, 0.9),
             FaceCandidate("barber-c", 9, T0 + 0.8, "f/b.jpg", emb, 0.9))
    u = e.observe(fr, [], P1, faces=faces)
    (c,) = u.door_crossings
    assert c.direction == "in" and c.entry_id
    u = e.observe([Frame("barber-c", T0 + 12, ())], [], P1)
    (reg,) = u.registrations
    assert reg.visit_id == c.entry_id and {f.crop_path for f in reg.faces} == {"f/a.jpg", "f/b.jpg"}


def test_clear_returns_pending_entry_face_crops():
    from domain import FaceCandidate
    e = eng()
    emb = (1.0,) + (0.0,) * 511
    e.observe([], [], {**P, "register_source": 1.0}, faces=(FaceCandidate("barber-c", 9, T0, "f/p.jpg", emb, 0.9),))
    assert "f/p.jpg" in e.clear()


def _occupancy_engine():
    z = _wzones()
    z["barber-l"] = z["barber-l"] + [
        {"id": "l-R2", "name": "R2", "type": "chair", "points": sq(0.40, 0.10, 0.60, 0.40), "pair": "R2", "owner": True},
        {"id": "l-R2-seat", "name": "R2 seat", "type": "seat", "points": sq(0.45, 0.25, 0.55, 0.35), "seat_of": "l-R2"}]
    z["barber-c"] = z["barber-c"] + [{"id": "c-wash", "name": "wash", "type": "washing",
                                       "points": sq(0.0, 0.6, 0.4, 1.0)}]
    return ChairEngine(z, FrameGeometry(1000, 1000), barber_for=lambda c, at: "staff-1")


SITTER2 = tuple(v for p in sq(0.44, 0.05, 0.56, 0.36) for v in p)
WASHER = tuple(v for p in sq(0.10, 0.60, 0.20, 0.95) for v in p)


def test_occupancy_counts_haircuts_away_waits_and_people_at_the_wash():
    e = _occupancy_engine()
    both = (TrackObs(4, (0, 0, 1, 1), 0.9, SITTER), TrackObs(5, (0, 0, 1, 1), 0.9, SITTER2))
    e.observe(frames(T0, 300, both), [], P)
    e.observe(frames(T0 + 300, 70, (both[0],)), [], P)
    e.observe(frames(T0 + 300, 70, (_waiter(7), TrackObs(8, (0, 0, 1, 1), 0.9, WASHER)), cam="barber-c"), [], P)
    occ = e.occupancy()
    assert occ == {"haircuts": 2, "away": 1, "waiting": 1, "washing_people": 1}
    assert max(0, occ["washing_people"] - occ["away"]) == 0


def test_occupancy_leaves_staff_out_of_the_wash_and_is_zero_without_frames():
    e = _occupancy_engine()
    assert e.occupancy() == {"haircuts": 0, "away": 0, "waiting": 0, "washing_people": 0}
    apps = [Appearance("barber-c", 8, T0, None, 0.95)]
    e.observe(frames(T0, 5, (TrackObs(8, (0, 0, 1, 1), 0.9, WASHER),), cam="barber-c"), apps, P)
    assert e.occupancy()["washing_people"] == 0
    assert ChairEngine(ZONES, FrameGeometry(1000, 1000)).occupancy()["washing_people"] == 0


def test_live_chair_shows_the_re_id_check_while_it_decides():
    on = {**P, "ce_resume_check": 1.0}
    e = eng()
    e.observe(frames(T0, 300, (_tr(4),)), [Appearance("barber-l", 4, T0 + t, (1.0, 0.0), 0.1) for t in range(0, 300, 10)], on)
    e.observe(frames(T0 + 300, 70, ()), [], on)
    e.observe(frames(T0 + 370, 20, (_tr(9),)), [], on)
    c = e.live()["chairs"]["R1"]
    assert c["state"] == "sitting" and c["checking"] is not None and c["person_id"] is None
    assert c["check_ticket"] == e.haircuts.open["R1"].ticket
    (cust,) = e.live()["customers"]
    assert cust["state"] == "washing"
    e.observe(frames(T0 + 390, 160, (_tr(9),)), [], on)      # no re-ID by the deadline: the time rule, resumed
    c = e.live()["chairs"]["R1"]
    assert c["checking"] is None and c["person_id"] is not None and e.live()["customers"][0]["state"] == "haircut"
    assert c["resume"]["resumed"] is True and c["resume"]["sim"] is None


def test_live_chair_says_how_the_re_id_check_decided():
    """2026-10-18 (user): the card and the stream say "resumed" / "new customer" with the score, while that sitting
    lasts, instead of leaving it to the timer going on or restarting."""
    on = {**P, "ce_resume_check": 1.0}
    e = eng()
    assert e.live()["chairs"]["R1"]["resume"] is None
    e.observe(frames(T0, 300, (_tr(4),)), [Appearance("barber-l", 4, T0 + t, (1.0, 0.0), 0.1) for t in range(0, 300, 10)], on)
    ticket = e.haircuts.open["R1"].ticket
    e.observe(frames(T0 + 300, 70, ()), [], on)
    e.observe(frames(T0 + 370, 60, (_tr(9),)), [Appearance("barber-l", 9, T0 + 370 + t, (1.0, 0.0), 0.1)
                                                 for t in range(0, 60, 10)], on)
    r = e.live()["chairs"]["R1"]["resume"]
    assert r["resumed"] is True and r["sim"] == 1.0 and r["threshold"] == on["ce_resume_sim_low"]
    assert r["ticket"] == ticket and 60 <= r["gap_s"] <= 80 and r["at"] is not None
    e.observe(frames(T0 + 430, 400, ()), [], on)                      # he leaves for good: that sitting is over
    assert e.live()["chairs"]["R1"]["resume"] is None


def test_a_starting_chair_says_since_when():
    """2026-10-07 review: the dashboard debounces "sitting down" on `since` -- None meant it never showed."""
    e = eng()
    e.observe(frames(T0, 4, (_tr(4),)), [], P)
    c = e.live()["chairs"]["R1"]
    assert c["state"] == "starting" and datetime.fromisoformat(c["since"]).timestamp() == int(T0)


SOFA_OBS = TrackObs(1, (600, 400, 100, 400), 0.9, tuple(v for p in sq(0.6, 0.4, 0.7, 0.8) for v in p))
PH_SOFA = {"ph-3": [{"id": "s", "name": "sofa", "type": "waiting", "points": sq(0.5, 0.5, 0.9, 0.9)}], "ph-5": []}


def test_waits_run_on_the_configured_wait_camera():
    e = ChairEngine(PH_SOFA, FrameGeometry(1000, 1000), wait_camera="ph-3")
    assert e.wait_cam == "ph-3" and e.waits.camera == "ph-3" and e.waiting.camera == "ph-3"
    assert ChairEngine(PH_SOFA, FrameGeometry(1000, 1000)).wait_cam is None    # no door line, no wait camera
    e.observe([Frame("ph-3", T0 + i * 0.5, (SOFA_OBS,)) for i in range(24)], [], {**P, "wt_end_mode": 1.0})
    assert len(e.waits.open) == 1


def test_a_wait_camera_without_waiting_zones_runs_and_starts_nothing():
    e = ChairEngine({"ph-3": []}, FrameGeometry(1000, 1000), wait_camera="ph-3")
    upd = e.observe([Frame("ph-3", T0 + i * 0.5, (SOFA_OBS,)) for i in range(40)], [], {**P, "wt_end_mode": 1.0})
    assert not e.waits.open and not upd.waits


def test_the_engine_feeds_appearances_to_the_wait_book():
    e = ChairEngine(PH_SOFA, FrameGeometry(1000, 1000), wait_camera="ph-3")
    seen = []
    e.waits.on_appearance = lambda app: seen.append((app.camera, app.track_id))
    e.observe([], [Appearance("ph-3", 4, T0, (1.0, 0.0), 0.0), Appearance("ph-5", 5, T0, (1.0, 0.0), 0.0)], P)
    assert seen == [("ph-3", 4), ("ph-5", 5)]          # the book itself keeps only its camera's


def test_confirmed_haircuts_reach_the_wait_book():
    e = eng()
    u = e.observe(frames(T0, 300, (TrackObs(4, (0, 0, 1, 1), 0.9, SITTER),)), [], P)
    (s,) = u.sessions.values()
    assert e.waits._starts == {s.id: s.started_at.timestamp()}


# ── register_source 2: seated faces (Task 13) ────────────────────────────────
P2 = {**P, "register_source": 2.0}
SITTER_R2 = tuple(v for p in sq(0.54, 0.05, 0.66, 0.36) for v in p)   # brief's SITTER2; renamed, SITTER2 exists above
FACE_ZONES = {**ZONES, "barber-l": ZONES["barber-l"] + [
    {"id": "l-R2", "name": "R2", "type": "chair", "points": sq(0.50, 0.10, 0.70, 0.40), "pair": "R2", "owner": True},
    {"id": "l-R2-seat", "name": "R2 seat", "type": "seat", "points": sq(0.55, 0.25, 0.65, 0.35), "seat_of": "l-R2"},
    {"id": "l-R1-face", "name": "R1 face", "type": "face_area", "points": sq(0.10, 0.05, 0.30, 0.40), "chair": "R1"},
    {"id": "l-R2-face", "name": "R2 face", "type": "face_area", "points": sq(0.25, 0.30, 0.70, 0.40), "chair": "R2"}]}
BOX1 = (140, 50, 120, 310)       # centre (0.20, 0.205): only in R1's face area
BOX2 = (540, 50, 120, 310)       # centre (0.60, 0.205): in neither face area
OVERLAP = (240, 300, 40, 60)     # centre (0.26, 0.33): in both face areas


def face(cam, tid, ts, path="f.jpg", q=0.9):
    return FaceCandidate(cam, tid, ts, path, (1.0,) + (0.0,) * 511, q)


def running(chairs=1):
    """R1 (track 4) -- and R2 (track 5) when chairs == 2 -- sitting with a running haircut for 300 s."""
    e = ChairEngine(FACE_ZONES, FrameGeometry(1000, 1000), barber_for=lambda c, at: "staff-1")
    people = (TrackObs(4, BOX1, 0.9, SITTER),) + ((TrackObs(5, BOX2, 0.9, SITTER_R2),) if chairs == 2 else ())
    e.observe(frames(T0, 300, people), [], P2)
    assert e.haircuts.open["R1"].last_end is None
    return e, T0 + 300


def test_a_seated_customers_face_joins_his_running_haircut():
    e, now = running()
    upd = e.observe([], [], P2, faces=(face("barber-l", 4, now),))
    assert [f.crop_path for f in e.haircuts.open["R1"].faces] == ["f.jpg"] and not upd.discard_crops


def test_faces_outside_the_area_from_staff_or_another_camera_are_dropped():
    e, now = running()
    e.boxes[("barber-l", 99)] = (900, 900, 50, 50)        # outside every face area
    e.boxes[("barber-l", 98)] = BOX1                       # a barber leaning over R1's customer
    e.staff[("barber-l", 98)] = 0.99
    upd = e.observe([], [], P2, faces=(face("barber-l", 99, now, "out.jpg"), face("barber-l", 98, now, "staff.jpg"),
                                       face("barber-c", 4, now, "nocam.jpg")))
    assert sorted(upd.discard_crops) == ["nocam.jpg", "out.jpg", "staff.jpg"]
    assert not e.haircuts.open["R1"].faces


def test_a_face_where_two_running_chairs_face_areas_overlap_is_dropped():
    e, now = running(chairs=2)
    e.boxes[("barber-l", 77)] = OVERLAP
    upd = e.observe([], [], P2, faces=(face("barber-l", 77, now, "both.jpg"),))
    assert upd.discard_crops == ["both.jpg"]


# Face areas as drawn on site (Task 11): around the seated customer's head. His box's centre (his chest) is below.
HEAD_ZONES = {**ZONES, "barber-l": ZONES["barber-l"] + [
    {"id": "l-R1-face", "name": "R1 face", "type": "face_area", "points": sq(0.15, 0.06, 0.25, 0.16), "chair": "R1"}]}
HEAD = (185, 70, 30, 40)         # his face's own box: centre (0.20, 0.09), inside R1's face area


def face_at(tid, box, path="f.jpg"):
    return FaceCandidate("barber-l", tid, T0 + 300, path, (1.0,) + (0.0,) * 511, 0.9, face_box=box)


def running_head():
    e = ChairEngine(HEAD_ZONES, FrameGeometry(1000, 1000), barber_for=lambda c, at: "staff-1")
    e.observe(frames(T0, 300, (TrackObs(4, BOX1, 0.9, SITTER),)), [], P2)   # BOX1: centre (0.20, 0.205), below it
    assert e.haircuts.open["R1"].last_end is None
    return e


def test_a_seated_face_counts_by_its_own_box_centre_not_its_persons():
    e = running_head()
    upd = e.observe([], [], P2, faces=(face_at(4, HEAD, "head.jpg"),))
    assert [f.crop_path for f in e.haircuts.open["R1"].faces] == ["head.jpg"] and not upd.discard_crops
    assert e.haircuts.open["R1"].faces[0].face_box == HEAD


def test_a_face_box_outside_the_area_is_dropped_though_its_persons_box_overlaps():
    e = running_head()
    upd = e.observe([], [], P2, faces=(face_at(4, (185, 250, 30, 40), "chest.jpg"),))   # centre (0.20, 0.27)
    assert upd.discard_crops == ["chest.jpg"] and not e.haircuts.open["R1"].faces


def test_a_face_without_its_box_falls_back_to_the_person_box_overlapping_the_area():
    e = running_head()
    e.boxes[("barber-l", 99)] = (600, 500, 100, 300)     # nowhere near the area
    upd = e.observe([], [], P2, faces=(face_at(4, None, "old.jpg"), face_at(99, None, "far.jpg")))
    assert [f.crop_path for f in e.haircuts.open["R1"].faces] == ["old.jpg"] and upd.discard_crops == ["far.jpg"]


def test_a_staff_face_inside_the_area_is_dropped_and_an_unknown_track_without_a_box_too():
    e = running_head()
    e.staff[("barber-l", 98)] = 0.99
    upd = e.observe([], [], P2, faces=(face_at(98, HEAD, "staff.jpg"), face_at(97, None, "nobox.jpg")))
    assert sorted(upd.discard_crops) == ["nobox.jpg", "staff.jpg"] and not e.haircuts.open["R1"].faces


def test_a_face_box_where_two_running_chairs_face_areas_overlap_is_dropped():
    e, now = running(chairs=2)
    upd = e.observe([], [], P2, faces=(face_at(4, (250, 320, 20, 20), "both.jpg"),))   # centre (0.26, 0.33)
    assert upd.discard_crops == ["both.jpg"]


def test_the_wait_book_asks_whether_a_haircut_still_sits():
    # final review I2: a wait linked to a haircut whose customer is still in the chair is not "served"
    e, now = running()
    sid = e.haircuts.open["R1"].id
    assert e.waits._sitting(sid, now - 60) and not e.waits._sitting("other", now - 60)
    e.observe(frames(now, 70, ()), [], P2)                       # he gets up: the sitting ends (away)
    assert e.haircuts.open["R1"].last_end is not None and not e.waits._sitting(sid, now - 60)


def test_register_source_0_is_unchanged():
    e, now = running()
    upd = e.observe([], [], P, faces=(face("barber-l", 4, now, "a.jpg"),))
    assert upd.discard_crops == ["a.jpg"]


def test_prune_drops_boxes_with_stale_seen_keys():
    e, now = running()
    e.boxes[("barber-l", 99)] = BOX2
    e.seen[("barber-l", 99)] = T0 - 1000.0
    e._pruned_at = None
    e.observe(frames(now, 1, (TrackObs(4, BOX1, 0.9, SITTER),)), [], P2)
    assert ("barber-l", 99) not in e.boxes and ("barber-l", 4) in e.boxes


# ── final re-review N1 (2026-10-15): "still sitting" = his chair occupied after the dwell began ──────────────────
def test_a_customer_who_stood_up_before_his_dwell_is_not_still_sitting():
    e, now = running()
    h = e.haircuts.open["R1"]
    e.observe(frames(now, 20, ()), [], P2)                    # R1 empty for 20 s: no SitEnd yet (end gap 60 s)
    assert h.last_end is None
    assert e._sitting_now(h.id, now - 100)                    # occupied after a dwell that began 100 s earlier
    assert not e._sitting_now(h.id, now + 15)                 # he stood up before he began to dwell
    assert not e._sitting_now("nope", now)


def test_a_staff_track_needs_n_non_staff_samples_in_a_row_to_sit():
    """2026-10-15 (Palm Hills L7 on ph-5): one 0.67 sample among 0.95s made a resting staff member a sitter for a
    second, and the start chain backdated the next customer's start to it. ce_staff_release_n 3: the seat rule keeps
    a staff track staff until 3 non-staff samples in a row; 1 = the latest sample decides (today)."""
    seated = (TrackObs(4, (0, 0, 1, 1), 0.9, SITTER),)
    for n, after_dip in ((1, "occupied"), (3, "staff_cover")):
        p = {**P, "ce_staff_release_n": n}
        e = eng()
        e.observe(frames(T0, 2, seated), [Appearance("barber-l", 4, T0, None, 0.95)], p)
        e.observe(frames(T0 + 2, 1, seated), [Appearance("barber-l", 4, T0 + 2, None, 0.67)], p)
        assert {r["track_id"]: r["rule"]["why"] for r in e.scene(p)["barber-l"]}[4] == after_dip
        e.observe(frames(T0 + 3, 1, seated), [Appearance("barber-l", 4, T0 + 3 + i * 0.2, None, 0.1) for i in range(2)], p)
        assert {r["track_id"]: r["rule"]["why"] for r in e.scene(p)["barber-l"]}[4] == "occupied"   # 3 in a row


def test_a_new_track_is_read_at_once_whatever_the_release_n():
    p = {**P, "ce_staff_release_n": 3}
    e = eng()
    e.observe(frames(T0, 2, (TrackObs(4, (0, 0, 1, 1), 0.9, SITTER),)), [Appearance("barber-l", 4, T0, None, 0.1)], p)
    assert {r["track_id"]: r["rule"]["why"] for r in e.scene(p)["barber-l"]}[4] == "occupied"


def test_live_chair_times_a_resumed_haircut_from_its_start():
    """2026-10-18 (user): after a resume the stream's chair counter restarted at 00:00 while the card went on --
    the chair state's since is the current sitting's start; haircut_since is the haircut's (the card's) start."""
    e = eng()
    seated = (TrackObs(4, (0, 0, 1, 1), 0.9, SITTER),)
    e.observe(frames(T0, 300, seated), [], P)
    first = e.live()["chairs"]["R1"]
    assert first["haircut_since"] == first["since"]
    e.observe(frames(T0 + 300, 70, ()), [], P)                      # away
    assert e.live()["chairs"]["R1"]["state"] == "away"
    e.observe(frames(T0 + 370, 60, seated), [], P)                  # back within the resume window
    c = e.live()["chairs"]["R1"]
    assert c["state"] == "sitting" and c["session_id"] is not None
    assert c["haircut_since"] == first["haircut_since"] and c["since"] != c["haircut_since"]


def test_live_chair_has_no_haircut_start_before_the_haircut_is_confirmed():
    e = eng()
    e.observe(frames(T0, 30, (TrackObs(4, (0, 0, 1, 1), 0.9, SITTER),)), [], P)     # under ce_min_haircut_s
    c = e.live()["chairs"]["R1"]
    assert c["state"] == "sitting" and c["haircut_since"] is None


def _chunks(e, t0, secs, tracks, policy, every=10):
    """observe() once per `every` seconds of frames, as the consumer's cycles do."""
    return [e.observe(frames(t0 + k, every, tracks), [], policy) for k in range(0, secs, every)]


def test_a_haircut_asks_for_a_start_photo_then_an_end_photo_every_two_minutes():
    """2026-10-22 (spec haircut photos): a close-up of the chair -- its outline joined with the sitter's box."""
    import pytest
    policy = {**P, "hp_photos": 1.0}
    seated = (TrackObs(4, (140, 50, 120, 310), 0.9, SITTER),)
    reqs = [r for u in _chunks(eng(), T0, 600, seated, policy) for r in u.photos]
    assert reqs[0].kind == "start" and [r.kind for r in reqs].count("start") == 1
    assert (reqs[0].camera, reqs[0].chair) == ("barber-l", "R1")
    assert reqs[0].crop == pytest.approx((0.05, 0.0, 0.35, 0.40))
    ends = [r.at.timestamp() for r in reqs if r.kind == "end"]
    assert len(ends) >= 3 and all(120 <= b - a < 130 for a, b in zip(ends, ends[1:]))


def test_no_photo_while_away_and_a_sitting_too_short_drops_its_photos():
    policy = {**P, "hp_photos": 1.0, "ce_min_haircut_s": 300.0}
    e = eng()
    seated = (TrackObs(4, (140, 50, 120, 310), 0.9, SITTER),)
    (start,) = [r for u in _chunks(e, T0, 100, seated, policy) for r in u.photos]      # 100 s: never a haircut
    later = _chunks(e, T0 + 100, 900, (), policy)
    assert not [r for u in later for r in u.photos]
    assert {hid for u in later for hid in u.dropped_photos} == {start.haircut_id}


def test_no_photos_with_the_switch_off():
    seated = (TrackObs(4, (140, 50, 120, 310), 0.9, SITTER),)
    assert not [r for u in _chunks(eng(), T0, 300, seated, P) for r in u.photos]


def test_a_photo_waits_for_the_customer_on_the_seat_in_the_newest_frame():
    """2026-10-22 (user: no start photo of an empty chair): the customer must be seen on the seat in the chair camera's
    newest frame, not only a few seconds before it."""
    policy = {**P, "hp_photos": 1.0}
    e = eng()
    seated = (TrackObs(4, (140, 50, 120, 310), 0.9, SITTER),)
    left = frames(T0, 30, seated)[:-1] + [Frame("barber-l", T0 + 29.5, ())]      # the newest frame: nobody on it
    assert not e.observe(left, [], policy).photos and e.haircuts.open
    assert [r.kind for r in e.observe(frames(T0 + 30, 10, seated), [], policy).photos] == ["start"]
