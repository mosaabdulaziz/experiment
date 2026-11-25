"""modules.ppe.reid: the optional re-ID grace (off by default) and the names alerts and the overlay use."""
import types

import pytest

from perception.engine.frame import FrameOutput, FrameView, ObjectView
from perception.engine.modules import EngineServices, ModuleConfigError, build_settings
from perception.modules.ppe import MODULE, Settings
from perception.modules.ppe.adapter import UNTRACKED
from perception.modules.ppe.reid import ReidConfig, ReidGrace

FRAME_H = 720


def box(x, y=200, h=100, w=40):
    """(x1, y1, x2, y2), h tall."""
    return (x, y, x + w, y + h)


def grace(**kw):
    return ReidGrace(ReidConfig(enabled=True, **kw), forget_after_s=120.0)


# --- the matching rule (re-ID on) ------------------------------------------------------------------------------------

def test_re_id_is_off_by_default():
    assert ReidConfig().enabled is False and Settings().reid_config().enabled is False
    g = ReidGrace(ReidConfig(), forget_after_s=120.0)
    g.observe(0.0, [(12, box(100))], FRAME_H)
    assert g.observe(0.5, [(40, box(100))], FRAME_H) == {40: 40}


def test_a_worker_lost_and_back_nearby_within_the_gap_keeps_the_old_id():
    g = grace()
    assert g.observe(0.0, [(12, box(100))], FRAME_H) == {12: 12}
    assert g.observe(1.0, [], FRAME_H) == {}
    assert g.observe(1.5, [(40, box(130))], FRAME_H) == {40: 12}         # 0.3 heights away after 1.5 s
    assert g.observe(1.6, [(40, box(132))], FRAME_H) == {40: 12}         # and keeps it
    assert [(m[1], m[2]) for m in g.matches] == [(40, 12)]


def test_too_far_for_the_gap_or_after_the_gap_is_a_new_worker():
    g = grace()
    g.observe(0.0, [(12, box(100))], FRAME_H)
    assert g.observe(0.4, [(40, box(160))], FRAME_H) == {40: 40}         # 0.6 heights in 0.4 s: max(0.5, 0.28)
    g = grace()
    g.observe(0.0, [(12, box(100))], FRAME_H)
    assert g.observe(2.5, [(40, box(100))], FRAME_H) == {40: 40}         # same place, but lost 2.5 s > 2 s


def test_the_allowed_distance_grows_with_the_gap_at_0_7_heights_per_second():
    for x, same in ((220, True), (240, False)):                          # after 1.8 s: up to 1.26 heights
        g = grace()
        g.observe(0.0, [(12, box(100))], FRAME_H)
        assert (g.observe(1.8, [(40, box(x))], FRAME_H)[40] == 12) is same, x


def test_the_height_ratio_limit_is_one_and_a_half_either_way():
    for new_h, same in ((140, True), (160, False), (70, True), (60, False)):
        g = grace()
        g.observe(0.0, [(12, box(100, h=100))], FRAME_H)
        assert (g.observe(0.5, [(40, box(100, h=new_h))], FRAME_H)[40] == 12) is same, new_h


def test_a_partial_last_box_is_judged_by_the_tallest_box_of_the_last_second():
    # test.mp4: the coverall worker's last box was 52 px (half hidden), against 97 px when they came back.
    g = grace()
    g.observe(0.0, [(12, box(100, h=100))], FRAME_H)
    g.observe(0.5, [(12, box(100, h=50))], FRAME_H)                       # walks behind something
    assert g.observe(1.0, [(40, box(110, h=100))], FRAME_H) == {40: 12}
    g = grace(recent_s=0.0)                                              # judged by the last box alone: no match
    g.observe(0.0, [(12, box(100, h=100))], FRAME_H)
    g.observe(0.5, [(12, box(100, h=50))], FRAME_H)
    assert g.observe(1.0, [(40, box(110, h=100))], FRAME_H) == {40: 40}


def test_two_lost_and_two_new_match_one_to_one_closest_first():
    # In arrival order 40 would take the closer-to-it B and leave 41 with nobody; closest first: 41-B, then 40-A.
    g = grace()
    g.observe(0.0, [(1, box(100)), (2, box(160))], FRAME_H)                # A at 100, B at 160
    assert g.observe(0.5, [(40, box(140)), (41, box(170))], FRAME_H) == {40: 1, 41: 2}


def test_one_lost_worker_is_taken_over_by_one_new_id_only():
    g = grace()
    g.observe(0.0, [(1, box(100))], FRAME_H)
    assert g.observe(0.5, [(40, box(105)), (41, box(120))], FRAME_H) == {40: 1, 41: 41}


def test_a_tall_close_up_box_is_never_matched():
    for lost_h, new_h in ((400, 400), (300, 400), (400, 300)):
        g = grace()
        g.observe(0.0, [(12, box(100, y=100, h=lost_h, w=150))], FRAME_H)
        assert g.observe(0.3, [(40, box(105, y=100, h=new_h, w=150))], FRAME_H) == {40: 40}, (lost_h, new_h)
    g = grace()                                                           # just under half the frame: matched
    g.observe(0.0, [(12, box(100, y=100, h=350, w=150))], FRAME_H)
    assert g.observe(0.3, [(40, box(105, y=100, h=350, w=150))], FRAME_H) == {40: 12}


def test_a_worker_still_in_the_frame_is_not_taken_over():
    g = grace()
    g.observe(0.0, [(1, box(100))], FRAME_H)
    assert g.observe(0.1, [(1, box(100)), (40, box(110))], FRAME_H) == {1: 1, 40: 40}


def test_an_old_id_back_from_shadow_while_its_alias_is_there_splits_them_again():
    g = grace()
    g.observe(0.0, [(1, box(100))], FRAME_H)
    assert g.observe(0.5, [(40, box(105))], FRAME_H) == {40: 1}
    assert g.observe(0.6, [(40, box(105)), (1, box(300))], FRAME_H) == {1: 1, 40: 40}


@pytest.mark.parametrize("raw", [{"max_gap": 2}, {"max_height_ratio": 0.8}, {"max_box_height": 0}, {"max_gap_s": -1},
                                 {"enabled": "yes"}, {"recent_s": -1}])
def test_unknown_reid_fields_and_bad_values_refuse_startup(raw):
    with pytest.raises(ModuleConfigError):
        build_settings(Settings, {"reid": raw}, "ppe")


# --- through the module: alerts and the overlay name the CURRENT tracker id ------------------------------------------

MUX = (1080, 720)
FPS = 20.0


def _module(**overrides):
    emitted = []
    services = EngineServices(
        redis=None, redis_factory=lambda: None, emit=lambda key, fields, maxlen=64: key.startswith("iris:alert:") and emitted.append(fields),  # alerts only (not the clip boxes)
        request_clip=lambda *a, **k: None, stream_ids=["cam-1"], timer=lambda *a: None, telemetry=None, batch_size=1,
        states={"cam-1": types.SimpleNamespace(module_state={})})
    # alert_hold_s 0: these tests are about when a violation starts; the hold has its own tests
    mod = MODULE(build_settings(Settings, {"alert_hold_s": 0, **overrides}, "ppe"), services)
    mod.on_built(None)
    return mod, emitted


def _worker(tid, x=100, top=200, height=300, width=100, head="NO-Hardhat"):
    return [ObjectView(tid, "person", 0.9, 0.9, x, top, width, height, gie_name="pgie"),
            ObjectView(UNTRACKED, head, 0.8, 0.8, x + width * 0.25, top - height * 0.05, width * 0.5,
                       height * 0.25, gie_name="pgie_ppe")]


def _play(mod, segments):
    """segments: [(start time, n evaluated frames, objects or None for an empty frame)]; returns the last frame."""
    frame = out = None
    for start, n, objs in segments:
        for i in range(n):
            t = start + i / FPS
            frame = FrameView("cam-1", None, 0, int(t * FPS), t, t, mux_size=MUX, _objects=objs or [])
            out = FrameOutput()
            mod.on_frame(frame, out)
            mod.transform(frame)
            mod.analyze(frame, out)
    return frame, out


REID_ON = {"reid": {"enabled": True}}


def test_with_re_id_on_one_episode_across_a_switch_and_the_overlay_names_the_box_seen():
    mod, emitted = _module(**REID_ON)
    frame, out = _play(mod, [(0.0, 10, _worker(12)), (0.5, 20, None), (1.5, 20, _worker(40, x=120))])
    assert [(a["message"], a["track_id"]) for a in emitted] == [("Person #12 without helmet on cam-1", "12")]
    assert out.annotations[frame.objects[0]].text == "#40 no helmet"     # the current id, not the inherited one
    assert "canonical" not in "".join(emitted[0])


def test_with_re_id_on_an_alert_after_the_switch_names_the_new_id():
    # The violation starts only after the switch: the alert is about the box seen, #40.
    mod, emitted = _module(**REID_ON)
    _play(mod, [(0.0, 10, _worker(12, head="Hardhat")), (0.5, 10, None), (1.0, 20, _worker(40, x=120))])
    assert [(a["message"], a["track_id"]) for a in emitted] == [("Person #40 without helmet on cam-1", "40")]


def test_with_re_id_off_a_switch_is_a_second_alert_named_after_the_new_box():
    mod, emitted = _module()
    frame, out = _play(mod, [(0.0, 10, _worker(12)), (0.5, 20, None), (1.5, 20, _worker(40, x=120))])
    assert [a["track_id"] for a in emitted] == ["12", "40"]
    assert out.annotations[frame.objects[0]].text == "#40 no helmet"


def test_with_re_id_on_two_workers_switching_ids_together_keep_one_alert_each():
    mod, emitted = _module(**REID_ON)
    both = lambda a, b, dx=0: _worker(a, x=100 + dx) + _worker(b, x=500 + dx)   # noqa: E731
    _play(mod, [(0.0, 10, both(1, 2)), (0.5, 10, None), (1.0, 20, both(41, 40, dx=20))])
    assert sorted(a["message"] for a in emitted) == ["Person #1 without helmet on cam-1",
                                                     "Person #2 without helmet on cam-1"]


def test_with_re_id_on_a_tall_close_up_worker_is_never_joined():
    mod, emitted = _module(**REID_ON)
    tall = lambda tid: _worker(tid, x=100, top=150, height=400, width=200)       # noqa: E731
    _play(mod, [(0.0, 10, tall(12)), (0.5, 4, None), (0.7, 10, tall(40))])
    assert [a["track_id"] for a in emitted] == ["12", "40"]
