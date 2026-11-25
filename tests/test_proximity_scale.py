"""proximity.scale: the per-camera person-scale map h(v) = a v + b."""
import pytest

from perception.modules.proximity.scale import PersonScale, is_clear_standing


def _feed(scale, a=0.5, b=-50.0, short_every=0):
    """100 standing people with their feet on rows 200..695; every `short_every`-th one half hidden."""
    for i, v in enumerate(range(200, 700, 5)):
        h = a * v + b
        if short_every and i % short_every == 0:
            h *= 0.5
        scale.add(v, h, i)


def test_not_ready_until_enough_samples():
    s = PersonScale(min_samples=50)
    for i, v in enumerate(range(200, 445, 5)):             # 49 people
        s.add(v, 0.5 * v - 50, i)
    assert not s.ready and s.height(400) == 5.0            # nothing fitted: the floor
    assert s.add(500, 200.0, 49) and s.ready


def test_one_walker_is_not_enough_however_many_frames():
    s = PersonScale(min_samples=50)
    for i, v in enumerate(range(200, 700, 5)):             # 100 frames of 4 people crossing the view
        s.add(v, 0.5 * v - 50, i % 4)
    assert not s.ready
    s.add(450, 175.0, 99)                                  # a fifth person
    assert s.ready


def test_the_fit_recovers_the_line_and_ignores_short_people():
    s = PersonScale(min_samples=50, refit_every=10)
    _feed(s, short_every=5)                                # 20 % half hidden
    assert s.a == pytest.approx(0.5, rel=0.02) and s.b == pytest.approx(-50.0, abs=3)


def test_standing_height_from_the_head_row():
    s = PersonScale(min_samples=50)
    _feed(s)
    assert s.standing_height_from_head(300.0) == pytest.approx(200.0, rel=0.01)   # feet on 500: 0.5 * 500 - 50
    assert s.height(500.0) == pytest.approx(200.0, rel=0.01)


def test_a_flat_view_fits_a_flat_map():
    s = PersonScale(min_samples=50)
    _feed(s, a=0.0, b=60.0)                                # bird's-eye: everyone about 60 px
    assert s.ready and s.height(250) == pytest.approx(60.0, abs=1)
    assert s.standing_height_from_head(250) == pytest.approx(60.0, abs=1)


def test_people_all_on_one_row_give_no_fit():
    s = PersonScale(min_samples=50)
    for i in range(60):
        s.add(400.0, 150.0, i)
    assert not s.ready


def test_a_saved_fit_is_ready_at_once_and_round_trips():
    s = PersonScale(min_samples=50)
    _feed(s)
    t = PersonScale(min_samples=50)
    t.restore(s.to_dict())
    assert t.ready and t.height(400) == pytest.approx(s.height(400))


@pytest.mark.parametrize("bad", [{"a": "x", "b": 0, "n": 60}, {"a": 1.2, "b": 0, "n": 60}, {"a": 0.1}])
def test_a_bad_saved_fit_is_refused(bad):
    s = PersonScale()
    with pytest.raises((KeyError, ValueError, TypeError)):
        s.restore(bad)
    assert not s.ready


def test_the_drift_signal_fires_with_the_default_parameters():
    s = PersonScale()
    rows = list(range(200, 700, 5))
    for i in range(2000):                                  # the camera as it was
        v = rows[i % 100]
        s.add(v, 0.5 * v - 50, i % 50)
    assert abs(s.drift) < 0.05
    drifts = []
    for i in range(400):                                   # zoomed in: people 50 % taller
        v = rows[i % 100]
        s.add(v, 0.75 * v - 75, 1000 + i % 50)
        drifts.append(s.drift)
    assert max(drifts) > 0.25


def test_clear_standing_means_confident_upright_and_clear_of_the_edges():
    assert is_clear_standing(0.8, (100, 100, 30, 80), 1280, 720)
    assert not is_clear_standing(0.5, (100, 100, 30, 80), 1280, 720)       # unsure
    assert not is_clear_standing(0.8, (100, 100, 40, 80), 1280, 720)       # squat or bent
    assert not is_clear_standing(0.8, (100, 640, 30, 79), 1280, 720)       # cut by the bottom edge


def test_a_restored_map_holds_until_as_many_new_samples_are_in_but_drift_still_shows():
    s = PersonScale(min_samples=50, refit_every=100)
    s.restore({"a": 0.4, "b": 0.0, "n": 2000})             # e.g. from the seed file
    rows = list(range(300, 500, 2))                        # a narrow band of rows, people 50 % taller than the map
    for i in range(300):
        v = rows[i % 100]
        s.add(v, 0.6 * v, i % 20)
    assert s.height(400) == pytest.approx(160.0) and s.drift > 0.25
    for i in range(300, 2000):
        v = rows[i % 100]
        s.add(v, 0.6 * v, i % 20)
    assert s.height(400) == pytest.approx(240.0, rel=0.02)  # a full window of new samples: refitted
