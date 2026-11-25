"""proximity.camera: the rule end to end on synthetic persons and vehicles."""
import random

import numpy as np
import pytest

from perception.modules.proximity.camera import Person, ProximityCamera, Vehicle, severity
from perception.modules.proximity.rules import ZoneRule
from perception.modules.proximity.scale import PersonScale
from perception.modules.proximity.shapes import Blob, PersonShape, VehicleShape, vehicle_shape, vehicle_shape_from_box

FULL = np.array([[0, 0], [1280, 0], [1280, 720], [0, 720]], np.float32)
TIMES = [round(i * 0.15, 3) for i in range(31)]            # 0 .. 4.5 s, ~6.7 analysed frames/s
SEEN_BEFORE = [round(-2.1 + i * 0.15, 3) for i in range(45)]  # -2.1 .. 4.5 s: people in view before the vehicle comes


def _rule(**kw):
    base = dict(zone_id="yard", name="Yard", polygon=FULL, person_classes=frozenset({"person"}),
                vehicle_classes=frozenset({"car", "truck"}), max_gap=0.75, stationary_window_s=2.0,
                stationary_motion=0.5, persist_s=1.0, grace_s=0.5, cooldown_s=10.0, depth_check=False,
                depth_max_gap=0.2, hidden_feet="unconfirmed")
    base.update(kw)
    return ZoneRule(**base)


def _camera(ready=True, **kw):
    scale = PersonScale(min_samples=50)
    if ready:
        scale.restore({"a": 0.0, "b": 100.0, "n": 100})    # every person 100 px tall
    return ProximityCamera((1280, 720), scale, **kw)


def _person(pid=1, x=450.0, foot_y=560.0, visible=100.0):
    """Standing with the feet on row foot_y; `visible` px of the 100 seen from the head down."""
    head = foot_y - 100.0
    return Person(pid, "person", 0.9, (x - 15, head, 30, visible), PersonShape((x, head + visible), (x, head), False))


def _truck(vid=10, x=0.0, top=350.0, bottom=500.0):
    box = (x, top, 200.0, bottom - top)
    return Vehicle(vid, "truck", box, vehicle_shape_from_box(box))


def _drive(t):
    """A truck driving right at 150 px/s: it passes 60 px above the person's feet between t ~ 0.7 and 2.6 s."""
    return 100.0 + 150.0 * t


def _run(cam, rules, make, times=TIMES, depth=None):
    alerts, steps = [], []
    for t in times:
        persons, vehicles = make(t)
        step = cam.step(t, persons, vehicles, rules, depth)
        alerts += step.alerts
        steps.append(step)
    return alerts, steps


def test_a_person_beside_a_moving_truck_alerts_once():
    alerts, steps = _run(_camera(), [_rule()], lambda t: ([_person()], [_truck(x=_drive(t))]))
    (a,) = alerts
    assert (a.zone.zone_id, a.person.id, a.vehicle.id, a.confirmation, a.severity) == ("yard", 1, 10, "image", "warning")
    assert a.gap == pytest.approx(0.6, abs=0.02) and a.held_s >= 1.0 and a.foot.source == "visible"
    assert any(s.lines for s in steps)


def test_a_parked_truck_never_alerts_and_is_reported_parked():
    alerts, steps = _run(_camera(), [_rule()], lambda t: ([_person()], [_truck(x=300.0)]))
    assert alerts == [] and steps[-1].parked == {10} and steps[-1].states == {10: "parked"}
    (line,) = steps[-1].lines                       # measured for the clips, never close: a parked truck does not count
    assert (line.style, line.vehicle_state, line.person_id, line.vehicle_id) == ("far", "parked", 1, 10)
    assert line.gap == pytest.approx(0.6, abs=0.02)


def test_a_rider_moving_with_the_truck_is_not_a_pedestrian():
    alerts, _ = _run(_camera(), [_rule()],
                     lambda t: ([_person(x=_drive(t) + 100, foot_y=480.0)], [_truck(x=_drive(t))]))
    assert alerts == []


def _hidden_behind_a_low_vehicle(t):
    """Feet in view until t = 1.0, then hidden behind a low vehicle (box top 450) passing in front of them."""
    return [_person(foot_y=480.0, visible=100.0 if t < 1.0 else 60.0)], [_truck(x=_drive(t), top=450.0)]


def test_hidden_feet_without_depth_alert_unconfirmed_at_info():
    alerts, _ = _run(_camera(), [_rule()], _hidden_behind_a_low_vehicle)
    (a,) = alerts
    assert (a.confirmation, a.severity, a.foot.hidden, a.foot.source) == ("none", "info", True, "track")


def test_a_zone_can_suppress_unconfirmed_hidden_feet():
    alerts, _ = _run(_camera(), [_rule(hidden_feet="suppress")], _hidden_behind_a_low_vehicle)
    assert alerts == []


@pytest.mark.parametrize("measured, expected", [(0.1, ["depth"]), (0.5, [])])
def test_a_depth_zone_alerts_only_on_a_same_depth_measurement(measured, expected):
    alerts, _ = _run(_camera(), [_rule(depth_check=True)], lambda t: ([_person()], [_truck(x=_drive(t))]),
                     depth=lambda key: (0.0, measured))
    assert [a.confirmation for a in alerts] == expected


def _masked_person():
    """The standard person, with a mask: a 100 x 30 blob over its box."""
    p = _person()
    p.shape.blob = Blob(np.ones((100, 30), bool), 435.0, 460.0, 1.0, 1.0)
    return p


def test_a_depth_zone_asks_for_measurements_and_falls_back_when_none_comes():
    alerts, steps = _run(_camera(), [_rule(depth_check=True)], lambda t: ([_masked_person()], [_truck(x=_drive(t))]),
                         depth=lambda key: None)
    requests = [r for s in steps for r in s.depth_requests]
    assert requests and requests[0].key == ("yard", 1, 10)
    assert requests[0].person_box == (435.0, 460.0, 30, 100.0) and requests[0].vehicle_blob is None
    (a,) = alerts                                          # the check never answered: as if it were off
    assert a.confirmation == "image" and a.held_s >= 1.5


def test_no_depth_is_asked_for_a_person_without_a_mask():
    # a box's lowest band holds whatever hides the feet: it would read as "same depth"
    _alerts, steps = _run(_camera(), [_rule(depth_check=True)], lambda t: ([_person()], [_truck(x=_drive(t))]),
                          depth=lambda key: None)
    assert not [r for s in steps for r in s.depth_requests]


def test_a_depth_zone_without_a_depth_model_alerts_on_the_image():
    alerts, _ = _run(_camera(), [_rule(depth_check=True)], lambda t: ([_person()], [_truck(x=_drive(t))]))
    assert [a.confirmation for a in alerts] == ["image"]


def test_both_the_person_and_the_vehicle_must_be_in_the_zone():
    around_the_person = np.array([[400, 540], [500, 540], [500, 600], [400, 600]], np.float32)
    alerts, _ = _run(_camera(), [_rule(polygon=around_the_person)], lambda t: ([_person()], [_truck(x=_drive(t))]))
    assert alerts == []


def test_a_crowd_beside_one_truck_is_one_alert():
    crowd = lambda t: ([_person(1, 400.0), _person(2, 450.0), _person(3, 520.0)], [_truck(x=_drive(t))])
    alerts, _ = _run(_camera(), [_rule()], crowd)
    assert len(alerts) == 1


def test_time_going_back_resets_the_memory():
    cam = _camera()
    make = lambda t: ([_person()], [_truck(x=_drive(t))])
    first, _ = _run(cam, [_rule()], make)
    again, _ = _run(cam, [_rule()], make)                  # the file looped: pts start from 0 again
    assert len(first) == 1 and len(again) == 1


def test_before_the_scale_map_is_ready_the_rule_uses_box_heights():
    alerts, _ = _run(_camera(ready=False), [_rule()], lambda t: ([_person()], [_truck(x=_drive(t))]))
    assert len(alerts) == 1 and alerts[0].gap == pytest.approx(0.6, abs=0.02)


def test_clear_standing_people_teach_the_scale_map():
    cam = _camera(ready=False)
    refits = []
    for i in range(60):
        v = 200.0 + 8 * i
        h = 0.5 * v - 50
        p = Person(i, "person", 0.9, (100.0, v - h, 20.0, h), PersonShape((110.0, v), (110.0, v - h), False))
        refits.append(cam.step(i * 0.1, [p], [], [_rule()]).refit)
    assert any(refits) and cam.scale.ready and cam.scale.a == pytest.approx(0.5, rel=0.05)


class _CountingScale(PersonScale):
    def __init__(self):
        super().__init__(min_samples=50)
        self.taught_by = []

    def add(self, foot_v, height, person_id):
        self.taught_by.append(person_id)
        return super().add(foot_v, height, person_id)


def test_a_person_who_lingers_teaches_the_scale_map_once_a_second():
    scale = _CountingScale()
    cam = ProximityCamera((1280, 720), scale)
    make = lambda t: ([_person(pid=1), _person(pid=2, x=800.0)], [])     # noqa: E731
    _run(cam, [_rule()], make)                             # 4.5 s at ~6.7 frames/s: t = 0, 1.05, 2.1, 3.15, 4.2
    assert scale.taught_by.count(1) == 5 and scale.taught_by.count(2) == 5
    _run(cam, [_rule()], make)                             # the file looped: they teach again from its start
    assert scale.taught_by.count(1) == 10


def test_severity():
    assert severity("none", 0.1, 0.75) == "info"
    assert severity("image", 0.3, 0.75) == "critical"
    assert severity("depth", 0.5, 0.75) == "warning"


def test_grace_shorter_than_the_frame_interval_still_alerts():
    alerts, _ = _run(_camera(), [_rule(grace_s=0.0)], lambda t: ([_person()], [_truck(x=_drive(t))]))
    assert len(alerts) == 1


def test_a_zone_without_cooldown_still_alerts_once_per_pass():
    alerts, _ = _run(_camera(), [_rule(cooldown_s=0.0)], lambda t: ([_person()], [_truck(x=_drive(t))]))
    assert len(alerts) == 1


def test_a_vehicle_id_switch_mid_approach_still_alerts():
    alerts, _ = _run(_camera(), [_rule()],
                     lambda t: ([_person()], [_truck(vid=10 if t < 1.5 else 11, x=_drive(t))]))
    assert len(alerts) == 1


def test_a_parked_truck_jumping_for_one_frame_never_alerts():
    alerts, _ = _run(_camera(), [_rule()],
                     lambda t: ([_person()], [_truck(x=360.0 if t == 1.5 else 300.0)]))
    assert alerts == []


def test_a_person_standing_in_front_of_a_box_only_truck_is_not_a_rider():
    alerts, _ = _run(_camera(), [_rule()], lambda t: ([_person(foot_y=495.0)], [_truck(x=_drive(t))]), SEEN_BEFORE)
    (a,) = alerts
    assert a.severity == "critical"


def _forklift(t):
    """A tall vehicle box (mast) whose mask is only its body, the lowest 70 px: 200 x 200 px, top at 250."""
    x = _drive(t)
    grid = np.zeros((200, 200), bool)
    grid[130:, :] = True
    return Vehicle(10, "truck", (x, 250.0, 200.0, 200.0), vehicle_shape(Blob(grid, x, 250.0, 1.0, 1.0)))


def test_a_pedestrian_hidden_behind_a_moving_vehicle_is_not_a_rider():
    # standing still behind it: only head and torso seen, the lowest of it 1 px into the vehicle's mask
    person = Person(1, "person", 0.9, (435.0, 311.0, 30.0, 70.0), PersonShape((450.0, 381.0), (450.0, 311.0), False))
    alerts, _ = _run(_camera(), [_rule()], lambda t: ([person], [_forklift(t)]), SEEN_BEFORE)
    (a,) = alerts
    assert a.foot.hidden and a.confirmation == "none"


def test_people_cut_by_the_frame_bottom_stay_in_their_zone():
    person = Person(1, "person", 0.9, (435.0, 650.0, 30.0, 70.0), PersonShape((450.0, 719.0), (450.0, 650.0), True))
    alerts, _ = _run(_camera(), [_rule()], lambda t: ([person], [_truck(x=_drive(t), top=600.0, bottom=700.0)]))
    (a,) = alerts
    assert a.foot.point[1] > 720 and a.foot.hidden


def test_an_unconfirmed_episode_escalates_when_a_confirmed_pair_joins():
    def scene(t):
        persons = [_person(1, foot_y=480.0, visible=60.0)]                  # feet hidden all along: info
        if t >= 1.95:
            persons.append(_person(2, x=480.0, foot_y=505.0))               # feet in view, 0.05 ph: critical
        return persons, [_truck(x=_drive(t), top=450.0)]
    alerts, _ = _run(_camera(), [_rule()], scene)
    assert [(a.severity, a.escalated) for a in alerts] == [("info", False), ("critical", True)]
    assert alerts[0].episode is alerts[1].episode


class _SlowDepth:
    """A depth worker that first answers 0.3 s after a pair was first asked about, then refreshes about once a second
    (its latest measurement is up to 0.9 s old)."""
    def __init__(self, gap):
        self.gap, self.first, self.now = gap, {}, 0.0

    def __call__(self, key):
        first = self.first.setdefault(key, self.now)
        if self.now < first + 0.3:
            return None
        return (max(first + 0.3, self.now - 0.9), self.gap)


def _run_with(depth, rules, make, times=TIMES):
    cam, alerts = _camera(), []
    for t in times:
        depth.now = t
        persons, vehicles = make(t)
        alerts += cam.step(t, persons, vehicles, rules, depth).alerts
    return alerts


def test_a_pair_the_depth_check_rejects_never_alerts_while_it_is_refreshed():
    alerts = _run_with(_SlowDepth(0.5), [_rule(depth_check=True)], lambda t: ([_person()], [_truck(x=_drive(t))]))
    assert alerts == []


def test_a_vehicle_id_switch_does_not_skip_the_depth_wait():
    alerts = _run_with(_SlowDepth(0.5), [_rule(depth_check=True)],
                       lambda t: ([_person()], [_truck(vid=10 if t < 1.5 else 11, x=_drive(t))]))
    assert alerts == []


def test_each_zone_judges_the_depth_reading_against_its_own_cut_off():
    # Ticket 6: a reading of 0.3 is "apart" in a zone cut at 0.25 and "the same depth" in one cut at 0.35
    scene = lambda t: ([_person()], [_truck(x=_drive(t))])
    strict = _run_with(_SlowDepth(0.3), [_rule(depth_check=True, depth_max_gap=0.25)], scene)
    loose = _run_with(_SlowDepth(0.3), [_rule(depth_check=True, depth_max_gap=0.35)], scene)
    assert strict == [] and [a.confirmation for a in loose] == ["depth"]


# Ticket 6: a close pair the depth check says is apart is not alerted, but recorded once, quietly


def _steps_with(depth, rules, make, times=TIMES):
    cam, steps = _camera(), []
    for t in times:
        depth.now = t
        persons, vehicles = make(t)
        steps.append(cam.step(t, persons, vehicles, rules, depth))
    return steps


def test_a_pair_the_depth_check_suppresses_is_recorded_once_quietly():
    steps = _steps_with(_SlowDepth(0.5), [_rule(depth_check=True)], lambda t: ([_person()], [_truck(x=_drive(t))]))
    assert [a for s in steps for a in s.alerts] == []
    (q,) = [q for s in steps for q in s.suppressed]
    assert (q.person.id, q.vehicle.id, q.depth_gap, q.zone.zone_id) == (1, 10, 0.5, "yard")
    assert q.severity in ("warning", "critical") and q.gap <= 0.75


class _ChangingDepth(_SlowDepth):
    """Says the pair is apart until `until`, then that it is at the same depth (the pair is close 0.75-2.55 s)."""
    def __init__(self, until):
        super().__init__(0.5)
        self.until = until

    def __call__(self, key):
        got = super().__call__(key)
        return None if got is None else (got[0], 0.5 if self.now < self.until else 0.1)


def test_a_suppressed_pair_the_depth_check_later_puts_at_the_same_depth_still_alerts():
    steps = _steps_with(_ChangingDepth(2.1), [_rule(depth_check=True)], lambda t: ([_person()], [_truck(x=_drive(t))]))
    assert len([q for s in steps for q in s.suppressed]) == 1
    assert [a.confirmation for s in steps for a in s.alerts] == ["depth"]


class _SameThenApart(_SlowDepth):
    """Says the pair is at the same depth until `until`, then that it is apart (a reading near the cut-off flips)."""
    def __init__(self, until):
        super().__init__(0.1)
        self.until = until

    def __call__(self, key):
        got = super().__call__(key)
        return None if got is None else (got[0], 0.1 if self.now < self.until else 0.5)


def test_a_pair_already_alerted_is_not_recorded_again_when_the_depth_check_flips_to_apart():
    steps = _steps_with(_SameThenApart(2.1), [_rule(depth_check=True)], lambda t: ([_person()], [_truck(x=_drive(t))]))
    assert [a.confirmation for s in steps for a in s.alerts] == ["depth"]
    assert [q for s in steps for q in s.suppressed] == []


def test_waiting_for_the_depth_check_or_getting_no_answer_is_no_suppression():
    steps = _steps_with(_NoDepth(), [_rule(depth_check=True)], lambda t: ([_person()], [_truck(x=_drive(t))]))
    assert [q for s in steps for q in s.suppressed] == []
    assert len([a for s in steps for a in s.alerts]) == 1           # the zone's own fallback: the image


class _NoDepth:
    now = 0.0

    def __call__(self, key):
        return None


def test_a_same_depth_pair_alerts_once_measured():
    alerts = _run_with(_SlowDepth(0.1), [_rule(depth_check=True)], lambda t: ([_person()], [_truck(x=_drive(t))]))
    assert [a.confirmation for a in alerts] == ["depth"]


def test_a_driver_whose_head_jitters_is_still_a_rider():
    rnd = random.Random(3)

    def scene(t):
        x = _drive(t)
        jx, jy = rnd.uniform(-35, 35), rnd.uniform(-35, 35)                 # mask jitter, +-0.35 ph
        head = (x + 100 + jx, 380.0 + jy)
        person = Person(1, "person", 0.9, (head[0] - 15, head[1], 30.0, 100.0),
                        PersonShape((head[0], head[1] + 100), head, False))
        return [person], [_truck(x=x)]
    alerts, _ = _run(_camera(), [_rule()], scene)
    assert alerts == []


def test_the_step_says_whether_a_zone_has_a_vehicle_that_may_move():
    cam, rules = _camera(), [_rule()]
    first = cam.step(0.0, [_person()], [_truck(x=300.0)], rules)
    assert first.active                                    # a new vehicle: not known to be parked yet
    for t in TIMES[1:10]:
        step = cam.step(t, [_person()], [_truck(x=300.0)], rules)
    assert not step.active                                 # parked
    assert not cam.step(1.5, [_person()], [], rules).active
    moving = _camera()
    for t in TIMES[:10]:
        step = moving.step(t, [_person()], [_truck(x=_drive(t))], rules)
    assert step.active


def _wide_truck(x):
    box = (x - 400, 350.0, 1100.0, 150.0)
    return Vehicle(10, "truck", box, vehicle_shape_from_box(box))


def _on_board(head_x, pid=1):
    head = (head_x, 380.0)
    return Person(pid, "person", 0.9, (head_x - 15, 380.0, 30.0, 100.0), PersonShape((head_x, 480.0), head, False))


def test_a_driver_seen_driving_straight_stays_a_rider_when_the_vehicle_turns():
    # in a turn the cab and the vehicle's reference point move differently: here the cab keeps 0.4 of its travel
    def scene(t):
        x = _drive(t)
        head_x = x + 300 if t < 1.5 else _drive(1.5) + 300 + 60 * (t - 1.5)
        return [_on_board(head_x)], [_wide_truck(x)]
    alerts, _ = _run(_camera(), [_rule()], scene, times=[round(i * 0.15, 3) for i in range(41)])
    assert alerts == []


def test_a_driver_who_sat_in_the_vehicle_while_it_was_parked_stays_a_rider():
    def scene(t):
        x = 300.0 if t < 2.0 else 300.0 + 150 * (t - 2.0)
        head_x = 600.0 if t < 2.0 else 600.0 - 150 * (t - 2.0)                  # it drives off turning
        return [_on_board(head_x)], [_wide_truck(x)]
    alerts, _ = _run(_camera(), [_rule()], scene, times=[round(i * 0.15, 3) for i in range(41)])
    assert alerts == []


def test_a_pedestrian_walking_toward_a_moving_vehicle_is_not_a_rider():
    # inside the vehicle's (wide) box by the geometry, walking against its motion at 2/3 of its speed
    def scene(t):
        head_x = 900.0 - 100 * t
        person = Person(1, "person", 0.9, (head_x - 15, 395.0, 30.0, 100.0), PersonShape((head_x, 495.0), (head_x, 395.0), False))
        return [person], [_wide_truck(_drive(t))]
    alerts, _ = _run(_camera(), [_rule()], scene)
    assert len(alerts) == 1


def test_a_person_beside_a_vehicle_crossing_the_other_way_with_hidden_feet_alerts():
    def scene(t):
        x = _drive(t)
        hx = 700.0 - 150 * t                                               # walking left, the truck drives right
        return [Person(1, "person", 0.9, (hx - 15, 400.0, 30.0, 60.0), PersonShape((hx, 460.0), (hx, 400.0), False))], [_truck(x=x)]
    alerts, _ = _run(_camera(), [_rule()], scene, SEEN_BEFORE)
    assert len(alerts) == 1


def test_a_cab_driver_whose_track_id_switches_stays_a_rider():
    def scene(t):
        x = _drive(t)
        pid = 1 if t < 2.4 else 2
        person = Person(pid, "person", 0.9, (x - 35, 400.0, 30.0, 60.0), PersonShape((x - 20, 460.0), (x - 20, 400.0), False))
        return [person], [_truck(x=x)]
    alerts, _ = _run(_camera(), [_rule()], scene, times=[round(i * 0.15, 3) for i in range(41)])
    assert alerts == []


def test_a_driver_in_a_cab_outside_the_vehicle_box_is_a_rider():
    # the detector boxed only the cargo body; the driver sits just left of it, feet hidden, travelling with it
    def scene(t):
        x = _drive(t)
        person = Person(1, "person", 0.9, (x - 35, 400.0, 30.0, 60.0), PersonShape((x - 20, 460.0), (x - 20, 400.0), False))
        return [person], [_truck(x=x)]
    alerts, _ = _run(_camera(), [_rule()], scene)
    assert alerts == []


def test_a_guide_walking_ahead_of_a_vehicle_with_feet_in_view_still_alerts():
    # walking right in front of it at its speed: travels with it, but the feet are in view -- a pedestrian
    def scene(t):
        x = _drive(t)
        return [_person(x=x + 230, foot_y=530.0)], [_truck(x=x)]
    alerts, _ = _run(_camera(), [_rule()], scene)
    assert len(alerts) == 1


def test_a_driver_first_seen_inside_a_turning_vehicle_is_its_rider():
    # the cab comes into view mid-turn: until t = 2.6 the head keeps only 0.4 of the vehicle's travel, then all of it;
    # feet hidden, inside the vehicle's box -- too new a track for co-motion to tell
    def scene(t):
        x = _drive(t)
        if t < 0.9:
            return [], [_truck(x=x)]
        hx = x + 175 - 90 * (min(t, 2.6) - 0.9)                  # the cab stays inside the vehicle's box
        return [Person(1, "person", 0.9, (hx - 15, 380.0, 30.0, 60.0), PersonShape((hx, 440.0), (hx, 380.0), False))], \
            [_truck(x=x)]
    alerts, _ = _run(_camera(), [_rule()], scene)
    assert alerts == []


def _loader(t):
    """A moving vehicle whose box (top 250) reaches the ground at 450 but whose mask stops at 430: the space in front of
    its body is inside its box, not on it."""
    x = _drive(t)
    grid = np.zeros((200, 200), bool)
    grid[100:180, :] = True
    return Vehicle(10, "truck", (x, 250.0, 200.0, 200.0), vehicle_shape(Blob(grid, x, 250.0, 1.0, 1.0)))


def test_a_new_pedestrian_with_feet_on_the_ground_inside_a_vehicle_box_alerts():
    # someone stepping out right in front of a passing vehicle: a brand-new track, feet in view, not on the vehicle
    alerts, _ = _run(_camera(), [_rule()], lambda t: ([_person(foot_y=445.0)], [_loader(t)]))
    assert len(alerts) == 1 and not alerts[0].foot.hidden


def test_a_rider_who_moved_with_the_vehicle_while_new_stays_its_rider_through_a_turn():
    # on board from the first frame; the vehicle turns at t = 0.5 and the cab keeps 0.4 of its travel from then on
    def scene(t):
        head_x = _drive(t) + 300 if t < 0.5 else _drive(0.5) + 300 + 60 * (t - 0.5)
        return [_on_board(head_x)], [_wide_truck(_drive(t))]
    alerts, _ = _run(_camera(), [_rule()], scene)
    assert alerts == []


def test_a_driver_who_walked_up_and_got_on_as_the_vehicle_set_off_is_its_rider():
    # walks left on the ground to the parked vehicle, out of view while climbing on, seen on it once it already moves
    # right: the co-motion window still holds the walk against its direction (C26, 11 Dec, 140 s)
    def scene(t):
        x = 300.0 if t < 1.2 else 300.0 + 150 * (t - 1.2)
        if t < 1.5:
            return [_person(x=1250.0 - 100 * t)], [_wide_truck(x)]
        if t < 1.95:
            return [], [_wide_truck(x)]
        return [_on_board(x + 600)], [_wide_truck(x)]
    alerts, _ = _run(_camera(), [_rule(persist_s=0.25)], scene, times=[round(i * 0.15, 3) for i in range(41)])
    assert alerts == []


def test_a_driver_seen_again_as_the_vehicle_sets_off_under_a_new_id_is_its_rider():
    # on vehicle 9 while it is parked; it is re-detected as 10 (which takes over its history) and sets off, fast, then
    # slower; the driver is out of view meanwhile and seen again on it once it moves: their short history against the
    # vehicle's whole window reads as keeping up too little (warehouse bay 4)
    def scene(t):
        x = 300.0 + 250 * min(max(t - 1.0, 0.0), 1.0) + 150 * max(t - 2.0, 0.0)
        truck = _wide_truck(x)
        vehicles = [] if 0.3 <= t < 0.6 else [Vehicle(9 if t < 0.3 else 10, "truck", truck.box, truck.shape)]
        if 0.45 <= t < 2.0:
            return [], vehicles
        return [_on_board(x + 300)], vehicles
    alerts, _ = _run(_camera(), [_rule(persist_s=0.25)], scene, times=[round(-2.1 + i * 0.15, 3) for i in range(55)])
    assert alerts == []


def test_someone_who_just_got_on_but_never_reads_as_riding_is_judged_after_1_5_s():
    # walking along beside the vehicle, out of reach, then on it from t = 0.6, staying put on its box; its ground point
    # runs ahead of its box (the mask's lowest third swinging in a turn): they never read as moving along with it
    def scene(t):
        x = _drive(t) - 300
        box = (x - 400, 350.0, 1100.0, 150.0)
        shape = vehicle_shape_from_box(box)
        truck = Vehicle(10, "truck", box, VehicleShape(shape.outline, (x + 150 + 225 * t, 500.0), shape.area))
        return [_person(x=x + 300, foot_y=600.0) if t < 0.6 else _on_board(x + 300)], [truck]
    times = [round(-2.1 + i * 0.15, 3) for i in range(55)]
    alerts, steps = _run(_camera(), [_rule(persist_s=0.25)], scene, times)
    first = next((t for t, s in zip(times, steps) if s.alerts), None)
    assert len(alerts) == 1 and first <= 0.6 + 1.5 + 0.25 + 0.15


def test_someone_standing_still_on_a_slow_vehicles_geometry_gets_no_benefit_of_the_doubt():
    # a truck creeping by at half a person height a second, over the person's feet: they stay put while it moves, so
    # they alert as soon as it reads moving -- not only once it has carried its box half a person height past them
    def scene(t):
        return [_person(foot_y=495.0)], ([] if t < 0 else [_truck(x=250.0 + 50 * t)])
    alerts, steps = _run(_camera(), [_rule(persist_s=0.25)], scene, SEEN_BEFORE)
    moving = next(t for t, s in zip(SEEN_BEFORE, steps) if s.states.get(10) == "moving")
    first = next(t for t, s in zip(SEEN_BEFORE, steps) if s.alerts)
    assert len(alerts) == 1 and first <= moving + 0.25 + 0.15


def test_a_pairs_boarding_memory_is_forgotten_once_they_are_out_of_view():
    cam = _camera()
    _run(cam, [_rule()], lambda t: ([_on_board(_drive(t) + 300)], [_wide_truck(_drive(t))]))
    assert cam._aboard
    _run(cam, [_rule()], lambda t: ([], [_wide_truck(_drive(t))]), times=[5.1])
    assert cam._aboard == {}


def test_a_new_episode_starts_from_its_own_frames_not_the_last_episodes_span():
    """A pair stays close while the depth check says "not at the same depth" for longer than the episode's hold: the
    episode ends, and when the pair is confirmed again a new one opens from its own frames -- not the old span's start,
    minimum gap and severity (a pedestrian walking with a truck at 0.6 ph, a dip to 0.31 ph at 3 s)."""
    now = {"t": 0.0}

    def depth(_key):
        return now["t"], (0.1 if now["t"] < 3.5 or now["t"] >= 14.5 else 0.5)

    cam, rule = _camera(), _rule(depth_check=True)
    opened, closed = [], []
    for i in range(112):
        t = round(i * 0.15, 3)
        now["t"] = t
        x = 60.0 * t
        foot_y = 530.0 if 2.9 <= t <= 3.1 else 560.0
        step = cam.step(t, [_person(x=x + 100.0, foot_y=foot_y)], [_truck(x=x)], [rule], depth)
        opened += [a.episode for a in step.alerts if not a.escalated]
        closed += step.closed
    first, second = opened
    assert closed and closed[0] is first
    assert second.start >= first.end and second.min_gap_t >= second.start
    assert second.min_gap > first.min_gap and second.severity == "warning"



def test_pairs_beyond_the_max_gap_are_measured_without_changing_the_alert():
    alerts, steps = _run(_camera(), [_rule()], lambda t: ([_person()], [_truck(x=_drive(t))]))
    (a,) = alerts                                                   # as without them: one alert, the same numbers
    assert a.gap == pytest.approx(0.6, abs=0.02) and a.held_s >= 1.0
    lines = [ln for s in steps for ln in s.lines]
    far = [ln for ln in lines if ln.vehicle_state == "moving" and ln.gap > 0.75]
    assert far and all(ln.style == "far" and not ln.latched and not ln.close for ln in far)
    assert all(ln.style in ("close", "latched") for ln in lines if ln.vehicle_state == "moving" and ln.gap <= 0.75)
    assert {ln.style for ln in lines} >= {"far", "close", "latched"}


def test_the_step_reports_each_zone_vehicles_state():
    _alerts, steps = _run(_camera(), [_rule()], lambda t: ([_person()], [_truck(x=_drive(t))]))
    assert steps[0].states == {10: "unknown"} and steps[-1].states == {10: "moving"}


def test_a_rider_gets_no_line():
    _alerts, steps = _run(_camera(), [_rule()],
                          lambda t: ([_person(x=_drive(t) + 100, foot_y=480.0)], [_truck(x=_drive(t))]))
    assert not any(ln.vehicle_state == "moving" for s in steps for ln in s.lines)


def test_a_pair_in_two_zones_gets_one_line_the_most_severe():
    rules = [_rule(), _rule(zone_id="yard2", name="Yard 2", max_gap=0.3)]
    _alerts, steps = _run(_camera(), rules, lambda t: ([_person()], [_truck(x=_drive(t))]))
    assert all(len(s.lines) <= 1 for s in steps)
    assert any(ln.style == "latched" for s in steps for ln in s.lines)



def _long_truck(left, bottom, vid=10, width=900.0, dips=()):
    """A wide truck whose lower outline is flat at `bottom` (a point every 2 px), with `dips` (x offset, depth px)."""
    xs = np.arange(left, left + width + 1.0, 2.0)
    ys = np.full_like(xs, bottom)
    for dx, depth in dips:
        ys[np.abs(xs - (left + dx)) < 1.0] = bottom + depth
    box = (left, bottom - 150.0, width, 150.0)
    return Vehicle(vid, "truck", box, VehicleShape(np.stack([xs, ys], axis=1), (left + width / 2, bottom), 150 * width))


WARM = [round(i * 0.15, 3) for i in range(20)]             # the truck is known to move after ~2 s


def test_a_pairs_gap_is_the_median_of_its_last_measurements():
    gaps = [1.0] * len(WARM) + [0.5, 0.9, 0.6, 0.8, 0.4, 0.7]
    times = [round(i * 0.15, 3) for i in range(len(gaps))]
    make = lambda t: ([_person()], [_long_truck(60.0 * t, 560.0 - 100.0 * gaps[times.index(t)])])
    _, steps = _run(_camera(gap_smoothing_frames=3), [_rule()], make, times=times)
    assert [round(s.lines[0].gap, 2) for s in steps[len(WARM):]] == [1.0, 0.9, 0.6, 0.8, 0.6, 0.7]


def test_one_bad_frame_does_not_make_a_pair_close_once_its_gap_is_smoothed():
    times = [round(i * 0.15, 3) for i in range(31)]
    make = lambda t: ([_person()], [_long_truck(60.0 * t, 530.0 if t == times[25] else 400.0)])
    _, raw = _run(_camera(), [_rule()], make, times=times)
    _, smooth = _run(_camera(gap_smoothing_frames=7), [_rule()], make, times=times)
    assert any(ln.close for s in raw for ln in s.lines) and not any(ln.close for s in smooth for ln in s.lines)


def test_after_a_pause_the_median_starts_afresh():
    # the person is out of view for 3 s while the truck drives on (still known to move): their pair's old gaps are stale
    pause = [round(WARM[-1] + 0.15 * (i + 1), 3) for i in range(20)]
    times = WARM + pause + [round(pause[-1] + 0.15, 3)]

    def make(t):
        truck = _long_truck(60.0 * t, 560.0 - 100.0 * (0.4 if t == times[-1] else 0.9))
        return ([] if t in pause else [_person()]), [truck]
    _, steps = _run(_camera(gap_smoothing_frames=7), [_rule()], make, times=times)
    assert steps[-1].states[10] == "moving" and round(steps[-1].lines[0].gap, 2) == 0.4


def test_the_line_end_does_not_jump_to_a_point_that_is_nearest_for_one_frame():
    # a person walking beside a moving truck; of two dips 30 px either side of their foot, the right one is nearest
    # only every third frame (mask noise)
    times = WARM + [round(WARM[-1] + 0.15 * (i + 1), 3) for i in range(9)]

    def make(t):
        a, b = (40.0, 42.0) if times.index(t) % 3 == 2 else (42.0, 40.0)
        return [_person(x=450.0 + 60.0 * t)], [_long_truck(60.0 * t, 480.0, dips=((420.0, a), (480.0, b)))]

    def run(cam):
        _, steps = _run(cam, [_rule()], make, times=times)
        lines = [s.lines[0] for s in steps[len(WARM):]]
        return [round(ln.outline_point[0] - ln.foot[0]) for ln in lines], [round(ln.gap, 3) for ln in lines]
    raw_ends, raw_gaps = run(_camera())
    steady_ends, steady_gaps = run(_camera(line_steady_frames=7))
    assert set(raw_ends) == {-30, 30} and set(steady_ends[3:]) == {-30}
    assert steady_gaps == raw_gaps          # the line is steadied for the eye only: the rule's gap is as measured


def test_a_gone_pairs_gap_history_is_forgotten():
    cam = _camera(gap_smoothing_frames=7)
    _run(cam, [_rule()], lambda t: ([_person()], [_long_truck(60.0 * t, 500.0)]), times=WARM + [3.0, 3.15])
    assert cam._gaps
    _run(cam, [_rule()], lambda t: ([], []), times=[10.0])
    assert cam._gaps == {}


def _masked_truck(left, vid=10):
    """A 200 x 150 truck with a full mask (10 px cells), its bottom at row 500."""
    blob = Blob(np.ones((15, 20), bool), left, 350.0, 10.0, 10.0)
    return Vehicle(vid, "truck", (left, 350.0, 200.0, 150.0), vehicle_shape(blob))


def _beside(foot_y):
    """A person standing 30 px right of a slowly moving truck's right side, feet on row `foot_y`."""
    return lambda t: ([_person(x=60.0 * t + 230.0, foot_y=foot_y)], [_masked_truck(60.0 * t)])


@pytest.mark.parametrize("foot_y, expected", [(480.0, 0.35), (420.0, 0.57)])   # 10 px cells: hypot(35, 5), hypot(35, 45)
def test_the_gap_runs_to_the_lowest_quarter_of_the_vehicles_mask(foot_y, expected):
    # beside its lower side (480): to the side itself, not down to the bottom edge; level with its body (420): down
    # to the band, not across to the body (Ticket 3: the lowest 25 % of the mask beat the bottom edge and the whole)
    times = WARM + [3.0]
    _, outline = _run(_camera(), [_rule()], _beside(foot_y), times=times)
    _, band = _run(_camera(vehicle_band=0.25), [_rule()], _beside(foot_y), times=times)
    assert round(band[-1].lines[0].gap, 2) == expected
    assert band[-1].lines[0].gap <= outline[-1].lines[0].gap


def test_a_vehicle_without_a_mask_is_measured_to_its_lower_outline():
    make = lambda t: ([_person()], [_long_truck(60.0 * t, 500.0)])
    _, plain = _run(_camera(), [_rule()], make, times=WARM + [3.0])
    _, band = _run(_camera(vehicle_band=0.25), [_rule()], make, times=WARM + [3.0])
    assert band[-1].lines[0].gap == plain[-1].lines[0].gap


def _behind_counter(pid=1, x=450.0, visible=40.0):
    """Standing with the feet on row 570, only the top `visible` px of their 100 in view (a counter hides the rest);
    their box is less than RIDER_OVERLAP inside the passing truck's, so they are no rider."""
    return Person(pid, "person", 0.9, (x - 15, 470.0, 30.0, visible), PersonShape((x, 470.0 + visible), (x, 470.0), False))


def test_a_person_showing_under_half_their_height_with_hidden_feet_gets_no_line_no_check_and_no_alert():
    # D41 (Ticket 4): a man behind the gatehouse counter, head and shoulders in view; his feet were put in a truck's path
    asked = []

    def depth(key):
        asked.append(key)
        return (0.0, 0.05)                                   # it would say "same depth"
    cam, alerts, lines = _camera(), [], []
    for t in TIMES:
        step = cam.step(t, [_behind_counter()], [_truck(x=_drive(t))], [_rule(depth_check=True)], depth)
        alerts += step.alerts
        lines += step.lines
        assert step.depth_requests == []
    assert alerts == [] and lines == [] and asked == []


def test_a_person_showing_over_half_their_height_with_hidden_feet_still_alerts_unconfirmed():
    alerts, _ = _run(_camera(), [_rule()], lambda t: ([_behind_counter(visible=60.0)], [_truck(x=_drive(t))]))
    assert [a.confirmation for a in alerts] == ["none"]


def _close_lines(depth, rule=None):
    cam, lines = _camera(), []
    for t in TIMES:
        lines += [ln for ln in cam.step(t, [_person()], [_truck(x=_drive(t))], [rule or _rule(depth_check=True)],
                                         depth).lines if ln.close]
    return lines


def test_a_close_pairs_line_carries_the_depth_checks_verdict():
    assert {ln.depth for ln in _close_lines(lambda key: (0.0, 0.05))} == {("same", 0.05)}
    assert {ln.depth for ln in _close_lines(lambda key: (0.0, 0.4))} == {("apart", 0.4)}


def test_an_unanswered_depth_check_waits_then_times_out():
    states = [ln.depth[0] for ln in _close_lines(lambda key: None)]
    assert states[0] == "pending" and states[-1] == "timeout" and set(states) == {"pending", "timeout"}


def test_without_the_depth_check_a_line_carries_no_verdict():
    lines = _close_lines(None, _rule())
    assert lines and all(ln.depth is None for ln in lines)


def test_a_line_from_a_hidden_foot_says_so_and_where_the_person_is_last_seen():
    cam, lines = _camera(), []
    for t in TIMES:
        lines += cam.step(t, [_behind_counter(visible=60.0)], [_truck(x=_drive(t))], [_rule()]).lines
    ln = lines[0]
    assert ln.foot_hidden and ln.seen_foot == (450.0, 530.0) and ln.foot == pytest.approx((450.0, 570.0))
    visible = []
    for t in TIMES:
        visible += _camera().step(t, [_person()], [_truck(x=_drive(t))], [_rule()]).lines
    assert visible and not visible[0].foot_hidden
