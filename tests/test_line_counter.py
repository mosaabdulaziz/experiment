import json

import pytest

from app.line_counter import CountLine, LineCounter, crossing_param, side_of


class FakeRedis:
    def __init__(self):
        self.h, self.l = {}, {}

    def hincrby(self, k, f, n):
        self.h.setdefault(k, {}); self.h[k][f] = self.h[k].get(f, 0) + n

    def lpush(self, k, v): self.l.setdefault(k, []).insert(0, v)
    def ltrim(self, k, a, b): self.l[k] = self.l[k][a:b + 1]
    def hgetall(self, k): return {f.encode(): str(v).encode() for f, v in self.h.get(k, {}).items()}
    def lrange(self, k, a, b): return self.l.get(k, [])[a:b + 1]
    def delete(self, *ks):
        for k in ks: self.h.pop(k, None); self.l.pop(k, None)


def det(oid, x, y, w=40, h=40):
    """A detection whose bottom-centre is (x, y)."""
    return ["vehicle", 0.9, oid, x - w / 2, y - h, w, h]


@pytest.fixture
def counter(tmp_path):
    r = FakeRedis()
    c = LineCounter(str(tmp_path / "lines.json"), lambda: r, frame_size=(1000, 1000))
    # vertical line at x=500, from top to bottom; normal points to +x? side_of > 0 for x < 500
    c.set_line("cam-01", {"a": [0.5, 0.0], "b": [0.5, 1.0], "enter_positive": True})
    return c, r


def test_geometry_helpers():
    a, b = (0, 0), (10, 0)
    assert side_of(a, b, (5, 3)) > 0 and side_of(a, b, (5, -3)) < 0
    assert crossing_param(a, b, (5, 3), (5, -3)) == pytest.approx(0.5)
    assert crossing_param(a, b, (5, 3), (5, 1)) is None          # doesn't reach the line
    assert crossing_param(a, b, (20, 3), (20, -3)) == pytest.approx(2.0)  # beyond the segment


def test_crossing_both_ways_is_counted_as_enter_and_exit(counter):
    c, r = counter
    # moving right->left: from x=700 to x=300. side_of for vertical a(.5,0)->b(.5,1): positive for x<500
    assert c.process("cam-01", [det(1, 700, 500)], {}, wall=100.0) == []
    ev = c.process("cam-01", [det(1, 300, 500)], {}, wall=101.0)
    assert [e["direction"] for e in ev] == ["enter"]      # towards the positive side = enter
    # a different vehicle goes the other way
    c.process("cam-01", [det(2, 300, 400)], {}, wall=110.0)
    ev = c.process("cam-01", [det(2, 700, 400)], {}, wall=111.0)
    assert [e["direction"] for e in ev] == ["exit"]
    t = c.totals("cam-01")
    assert (t["enter"], t["exit"]) == (1, 1) and len(t["recent"]) == 2


def test_flipping_the_entrance_direction_swaps_the_labels(counter):
    c, _ = counter
    c.set_line("cam-01", {"a": [0.5, 0.0], "b": [0.5, 1.0], "enter_positive": False})
    c.process("cam-01", [det(1, 700, 500)], {}, wall=100.0)
    ev = c.process("cam-01", [det(1, 300, 500)], {}, wall=101.0)
    assert [e["direction"] for e in ev] == ["exit"]


def test_jitter_on_the_line_and_passing_beside_it_do_not_count(counter):
    c, _ = counter
    c.process("cam-01", [det(1, 450, 500)], {}, wall=1.0)
    for x in (499, 501, 498, 502, 450):          # wobbles around the line, ends on the start side
        assert c.process("cam-01", [det(1, x, 500)], {}, wall=2.0) == []
    # crosses the infinite line but far past the end of the segment (y beyond the frame)
    c.process("cam-01", [det(3, 450, 2000)], {}, wall=3.0)
    assert c.process("cam-01", [det(3, 550, 2000)], {}, wall=4.0) == []


def test_same_vehicle_is_not_double_counted_when_its_local_id_changes(counter):
    c, _ = counter
    gids = {"1": 7, "9": 7}           # local ids 1 and 9 are both global vehicle 7
    c.process("cam-01", [det(1, 700, 500)], gids, wall=100.0)
    assert len(c.process("cam-01", [det(1, 300, 500)], gids, wall=101.0)) == 1
    c.process("cam-01", [det(9, 700, 500)], gids, wall=102.0)        # same vehicle re-detected
    assert c.process("cam-01", [det(9, 300, 500)], gids, wall=103.0) == []   # debounced
    # long after, it really does come through again
    c.process("cam-01", [det(9, 700, 500)], gids, wall=200.0)
    assert len(c.process("cam-01", [det(9, 300, 500)], gids, wall=201.0)) == 1


def test_config_validation_persistence_and_reset(counter, tmp_path):
    c, r = counter
    with pytest.raises(ValueError):
        c.set_line("cam-01", {"a": [0.1, 0.1], "b": [0.1, 0.1], "enter_positive": True})
    with pytest.raises(ValueError):
        c.set_line("cam-01", {"a": [0.1, 1.5], "b": [0.3, 0.3], "enter_positive": True})
    again = LineCounter(str(tmp_path / "lines.json"), lambda: r, frame_size=(1000, 1000))
    assert again.get_line("cam-01")["enter_positive"] is True          # survives a restart
    c.process("cam-01", [det(1, 700, 500)], {}, wall=1.0)
    c.process("cam-01", [det(1, 300, 500)], {}, wall=2.0)
    c.reset("cam-01")
    assert c.totals("cam-01")["enter"] == 0
    c.delete_line("cam-01")
    assert c.get_line("cam-01") is None and c.process("cam-01", [det(1, 300, 500)], {}) == []


def _run(c, oid, xs, y=500, t0=100.0):
    out = []
    for i, x in enumerate(xs):
        out += c.process("cam-01", [det(oid, x, y)], {}, wall=t0 + i)
    return [e["direction"] for e in out]


def test_track_first_detected_on_the_line_counts_when_it_moves_away(counter):
    c, _ = counter
    # first detection 0.5 px from the line, then clearly on the positive side (x < 500)
    assert _run(c, 1, [500.5, 470]) == ["enter"]
    # same, moving the other way
    assert _run(c, 2, [499.5, 530]) == ["exit"]


def test_track_first_detected_just_past_the_line_counts_only_if_moving_away(counter):
    c, _ = counter
    assert _run(c, 1, [480, 440]) == ["enter"]          # first seen 20 px past, moving away
    assert _run(c, 2, [480, 478]) == []                 # parked next to the line
    assert _run(c, 3, [300, 250]) == []                 # first seen far from the line
    # the rule is symmetric: first seen within 45 px of the line and moving away from it means
    # the vehicle came through the line from the other side, in whichever direction that is
    assert _run(c, 4, [520, 560]) == ["exit"]
    assert _run(c, 5, [520, 470]) == ["enter"]          # normal crossing, counted once
    assert _run(c, 6, [480, 520]) == ["exit"]           # first seen past the line, then back across


def test_late_first_detection_is_not_counted_twice(counter):
    c, _ = counter
    assert _run(c, 1, [500.5, 470, 440, 400, 520]) == ["enter", "exit"]   # enters, then drives back out


def test_recorded_d41_tracks_with_a_line_that_passes_where_tracks_begin(tmp_path):
    """Real detections from D41 (1280x720): the first truck starts 56 px before the line, the
    second is first detected 0.5 px from it (and was missed before). Another vehicle never
    reaches the line."""
    r = FakeRedis()
    c = LineCounter(str(tmp_path / "lines.json"), lambda: r, frame_size=(1280, 720))
    c.set_line("cam-01", {"a": [0.2002867775084497, 0.19981791697013837],
                          "b": [0.7456556621487829, 0.9363073561544064], "enter_positive": True})
    tracks = {
        0: [(819, 501), (794, 466), (777, 444), (762, 446), (744, 451), (737, 452), (574, 529)],
        19: [(753, 520), (719, 534), (667, 562), (611, 573)],
        88: [(795, 436), (770, 440), (735, 454)],
    }
    got = {}
    for oid, pts in tracks.items():
        got[oid] = []
        for i, (x, y) in enumerate(pts):
            got[oid] += [e["direction"] for e in c.process(
                "cam-01", [["vehicle", .9, oid, x - 20, y - 40, 40, 40]], {}, wall=100.0 + i * 0.5 + oid * 100)]
    assert got == {0: ["enter"], 19: ["enter"], 88: []}
