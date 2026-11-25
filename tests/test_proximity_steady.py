"""proximity.steady: the live preview's boxes, steadied (display only, Ticket 5)."""
import math

from perception.modules.proximity.steady import SteadyBoxes

FPS = 25


def _box(x, y=100.0):
    return (x, y, 40.0, 100.0)


def test_a_steady_box_moves_less_than_the_detections_it_follows():
    s = SteadyBoxes(tau_s=0.08, hold_s=0.5)
    raw_steps, steady_steps, last = [], [], None
    for i in range(50):                                 # a standing person whose detection jitters 10 px each frame
        raw = _box(200.0 + (10.0 if i % 3 == 0 else 0.0))
        steady, _held = s.update(i / FPS, {1: raw})
        if last is not None:
            raw_steps.append(abs(raw[0] - last[0][0]))
            steady_steps.append(abs(steady[1][0] - last[1][0]))
        last = (raw, steady[1])
    assert max(steady_steps) < 0.5 * max(raw_steps)


def test_a_steady_box_keeps_close_to_an_object_moving_steadily():
    s = SteadyBoxes(tau_s=0.08, hold_s=0.5)
    for i in range(50):                                 # 100 px/s
        steady, _held = s.update(i / FPS, {1: _box(100.0 + 100.0 * i / FPS)})
    lag = (100.0 + 100.0 * 49 / FPS) - steady[1][0]
    assert 0 < lag <= 100.0 * 0.08 + 1.0                # about speed x tau behind


def test_the_first_sighting_is_drawn_where_it_was_detected():
    s = SteadyBoxes(tau_s=0.08, hold_s=0.5)
    steady, held = s.update(0.0, {7: _box(300.0)})
    assert steady == {7: _box(300.0)} and held == {}


def test_without_smoothing_the_box_is_the_detection():
    s = SteadyBoxes(tau_s=0.0, hold_s=0.5)
    s.update(0.0, {1: _box(100.0)})
    steady, _held = s.update(0.04, {1: _box(160.0)})
    assert steady[1] == _box(160.0)


def test_a_vanished_box_is_held_where_it_was_then_dropped():
    s = SteadyBoxes(tau_s=0.08, hold_s=0.5)
    for i in range(10):
        steady, _held = s.update(i / FPS, {1: _box(200.0)})
    _steady, held = s.update(0.36 + 0.3, {})            # 0.3 s after its last detection
    assert set(held) == {1} and all(math.isclose(a, b) for a, b in zip(held[1], steady[1]))
    _steady, held = s.update(0.36 + 0.6, {})            # 0.6 s after: gone
    assert held == {}


def test_the_clock_going_back_starts_afresh():
    s = SteadyBoxes(tau_s=0.08, hold_s=0.5)
    for i in range(10):
        s.update(10.0 + i / FPS, {1: _box(200.0), 2: _box(600.0)})
    steady, held = s.update(0.0, {1: _box(400.0)})      # a looping file started over
    assert steady == {1: _box(400.0)} and held == {}
