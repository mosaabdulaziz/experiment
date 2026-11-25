"""modules.ppe: the possible-duplicate hint on alerts (grouping, not suppression; ppe-compliance issue #15), by the
track links of reid.py TrackLinks (lost track, gap <= 4 s, <= 1.2 heights from its last box, no height test)."""
import types

from perception.engine.frame import FrameOutput, FrameView, ObjectView
from perception.engine.modules import EngineServices, build_settings
from perception.modules.ppe import MODULE, Settings
from perception.modules.ppe.adapter import UNTRACKED

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


def _worker(tid, x=100, height=300, head="NO-Hardhat", torso=None, width=100, top=200):
    objs = [ObjectView(tid, "person", 0.9, 0.9, x, top, width, height, gie_name="pgie")]
    if head:
        objs.append(ObjectView(UNTRACKED, head, 0.8, 0.8, x + width * 0.25, top - height * 0.05, width * 0.5,
                               height * 0.25, gie_name="pgie_ppe"))
    if torso:
        objs.append(ObjectView(UNTRACKED, torso, 0.8, 0.8, x + width * 0.1, top + height * 0.17, width * 0.8,
                               height * 0.5, gie_name="pgie_ppe"))
    return objs


def _play(mod, segments):
    for start, n, objs in segments:
        for i in range(n):
            t = start + i / FPS
            mod.on_frame(FrameView("cam-1", None, 0, int(t * FPS), t, t, mux_size=MUX, _objects=objs or []),
                         FrameOutput())


def test_a_worker_re_found_under_a_new_id_alerts_again_and_names_the_earlier_alert():
    mod, emitted = _module()
    _play(mod, [(0.0, 10, _worker(22)), (0.5, 60, None), (3.5, 10, _worker(32, x=180))])     # 3 s later, 0.27 h
    first, second = emitted
    assert "possible_duplicate_of" not in first
    assert second["message"] == "Person #32 without helmet on cam-1 (possibly #22 again, alerted 4 s earlier)"
    assert (second["track_id"], second["possible_duplicate_of"], second["possible_duplicate_of_pts"]) == (
        "32", "22", "0.250")
    assert float(second["possible_duplicate_gap_s"]) == 3.05


def test_too_far_to_have_walked_there_gives_no_hint():
    mod, emitted = _module()
    _play(mod, [(0.0, 10, _worker(22)), (0.5, 10, None), (1.0, 10, _worker(32, x=800))])     # 2.3 h in 0.55 s
    assert [("possible_duplicate_of" in a) for a in emitted] == [False, False]


def test_two_workers_seen_together_are_not_hinted_as_one():
    mod, emitted = _module()
    _play(mod, [(0.0, 10, _worker(22) + _worker(32, x=180, head="Hardhat")),
                (0.5, 10, _worker(32, x=180))])                                             # 32 bare-headed later
    assert [a["track_id"] for a in emitted] == ["22", "32"]
    assert "possible_duplicate_of" not in emitted[1]


def test_a_different_item_gives_no_hint():
    mod, emitted = _module()
    _play(mod, [(0.0, 10, _worker(22)), (0.5, 10, None),
                (1.0, 10, _worker(32, x=120, head="Hardhat", torso="NO-Safety Vest"))])
    assert [(a["new"], "possible_duplicate_of" in a) for a in emitted] == [("helmet", False), ("vest", False)]


def test_a_re_find_more_than_4_s_after_the_loss_gives_no_hint():
    mod, emitted = _module()                                                                 # links max_gap_s 4
    _play(mod, [(0.0, 10, _worker(22)), (0.5, 10, None), (4.6, 10, _worker(32, x=120))])     # 4.15 s gap
    assert [("possible_duplicate_of" in a) for a in emitted] == [False, False]
    mod, emitted = _module()
    _play(mod, [(0.0, 10, _worker(22)), (0.5, 10, None), (4.3, 10, _worker(32, x=120))])     # 3.85 s gap
    assert emitted[1]["possible_duplicate_of"] == "22"


def test_the_distance_is_from_the_last_box_in_its_heights():
    mod, emitted = _module()
    # last box 300 px tall at x 100; centres 350 px (1.17 h) and 370 px (1.23 h) away
    _play(mod, [(0.0, 10, _worker(22)), (0.5, 10, None), (1.0, 10, _worker(32, x=450))])
    assert emitted[1]["possible_duplicate_of"] == "22"
    assert float(emitted[1]["possible_duplicate_distance"]) == 1.167
    mod, emitted = _module()
    _play(mod, [(0.0, 10, _worker(22)), (0.5, 10, None), (1.0, 10, _worker(32, x=470))])
    assert "possible_duplicate_of" not in emitted[1]


def test_no_height_test_a_close_up_half_box_is_still_the_same_worker():
    # test2.mp4: the jersey worker was re-found as #81 with a box 0.42 of their last height, 0.25 s later.
    mod, emitted = _module()
    _play(mod, [(0.0, 10, _worker(22, height=300)), (0.5, 5, None),
                (0.75, 10, _worker(32, x=110, top=230, height=130, width=60))])
    assert emitted[1]["possible_duplicate_of"] == "22"


def test_an_old_id_reused_for_another_worker_follows_the_lost_track_it_continues():
    # test2.mp4: nvtracker gave #10 (seen earlier on another worker) to the white-cap worker when #17 was lost.
    mod, emitted = _module()
    _play(mod, [(0.0, 10, _worker(10, x=800, head="Hardhat")),                              # 10 elsewhere, compliant
                (0.5, 10, _worker(17)),                                                    # 17 alerts
                (1.0, 2, None),
                (1.1, 10, _worker(10, x=120))])                                            # 10 back where 17 was
    assert [(a["track_id"], a.get("possible_duplicate_of")) for a in emitted] == [("17", None), ("10", "17")]


def test_the_hint_never_suppresses_an_alert():
    mod, emitted = _module()
    _play(mod, [(0.0, 10, _worker(22)), (0.5, 10, None), (1.0, 10, _worker(32, x=110)), (1.5, 10, None),
                (2.0, 10, _worker(41, x=120))])
    assert [a["track_id"] for a in emitted] == ["22", "32", "41"]
    assert [a.get("possible_duplicate_of") for a in emitted] == [None, "22", "32"]      # the most recent earlier one
