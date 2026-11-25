"""modules.ppe: candidate violation alerts (kind, severity, category, message, clip) from the smoothed rules."""
import types

from perception.engine.frame import FrameOutput, FrameView, ObjectView
from perception.engine.modules import EngineServices, build_settings
from perception.modules.ppe import MODULE, Settings
from perception.modules.ppe.adapter import UNTRACKED

MUX = (1080, 720)
FPS = 20.0                                   # the default eval_fps: feeding at this rate evaluates every frame


def _module(**overrides):
    emitted, clips = [], []
    services = EngineServices(
        redis=None, redis_factory=lambda: None, emit=lambda key, fields, maxlen=64: emitted.append((key, fields)),
        request_clip=lambda sid, now, age=None: clips.append((sid, round(now, 3))), stream_ids=["cam-1"],
        timer=lambda *a: None, telemetry=None, batch_size=1, states={"cam-1": types.SimpleNamespace(module_state={})},
        config={"overlay_boxes": True})                       # as the safety stack's PPE role (products/safety)
    # alert_hold_s 0: these tests are about when a violation starts; the hold has its own tests
    mod = MODULE(build_settings(Settings, {"alert_hold_s": 0, **overrides}, "ppe"), services)
    mod.on_built(None)
    return mod, emitted, clips


def _box(label, conf, left, top, width, height):
    return ObjectView(UNTRACKED, label, conf, conf, left, top, width, height, gie_name="pgie_ppe")


def _worker(track_id=12, head="NO-Hardhat", torso=None, x=100, top=200, height=300, width=100):
    """A worker in mux pixels: a head box in the head band, optionally a torso box in the torso band."""
    objs = [ObjectView(track_id, "person", 0.9, 0.9, x, top, width, height, gie_name="pgie")]
    if head:
        objs.append(_box(head, 0.8, x + width * 0.25, top - height * 0.05, width * 0.5, height * 0.25))
    if torso:
        objs.append(_box(torso, 0.8, x + width * 0.1, top + height * 0.17, width * 0.8, height * 0.5))
    return objs


def _feed(mod, frames, start=0.0):
    """frames: a list of object lists, one per evaluated frame at FPS from `start`; returns the last time used."""
    t = start
    for i, objs in enumerate(frames):
        t = start + i / FPS
        mod.on_frame(FrameView("cam-1", None, 0, i, t, t, mux_size=MUX, _objects=objs), FrameOutput())
    return t


def _alerts(emitted):
    return [f for key, f in emitted if key == "iris:alert:cam-1"]


def test_a_sustained_bare_head_gives_exactly_one_alert_with_a_clip():
    mod, emitted, clips = _module()
    _feed(mod, [_worker() for _ in range(40)])                 # 2 s of a bare head
    (a,) = _alerts(emitted)
    assert (a["kind"], a["severity"], a["category"]) == ("ppe_violation", "warning", "safety")
    assert a["message"] == "Person #12 without helmet on cam-1"
    assert (a["pts"], a["count"], a["track_id"], a["new"], a["missing"]) == ("0.250", "1", "12", "helmet", "helmet")
    assert clips == [("cam-1", 0.25)]                          # the 6th evaluated frame


def test_a_short_burst_gives_no_alert():
    # Like the open-vest worker of the evaluation clip: 4 bare-head frames between helmet frames.
    mod, emitted, _ = _module()
    _feed(mod, [_worker(head="Hardhat")] * 3 + [_worker()] * 4 + [_worker(head="Hardhat")] * 10)
    assert _alerts(emitted) == []


def test_a_head_cut_off_by_the_frame_edge_gives_no_alert():
    mod, emitted, _ = _module()
    _feed(mod, [_worker(top=0)] * 20)                          # person box touches the top of the mux frame
    assert _alerts(emitted) == []


def test_a_person_below_the_minimum_height_gives_no_alert():
    mod, emitted, _ = _module()
    _feed(mod, [_worker(height=30, width=12)] * 20)            # 30 px tall, under the 32 px default
    assert _alerts(emitted) == []
    mod, emitted, _ = _module()
    _feed(mod, [_worker(height=34, width=12)] * 20)            # just above it: the same bare head alerts
    assert len(_alerts(emitted)) == 1


def test_the_cooldown_holds_and_a_new_episode_after_it_alerts_again():
    mod, emitted, _ = _module()                                # cooldown_s 30 (evaluated default)
    # violation, helmet back on for 2.2 s (closes the episode), bare head again within the cooldown
    t = _feed(mod, [_worker()] * 6 + [_worker(head="Hardhat")] * 45 + [_worker()] * 12)
    assert len(_alerts(emitted)) == 1
    # still bare-headed once the cooldown has passed: the open absence now starts a new episode
    _feed(mod, [_worker()] * 3, start=t + 30.0)
    assert [a["pts"] for a in _alerts(emitted)] == ["0.250", f"{t + 30.0:.3f}"]


def test_a_short_false_helmet_does_not_close_the_episode_so_no_repeat_after_the_cooldown():
    # test2.mp4: a cap read as a hardhat for 0.4-1.45 s closed open episodes; 30 s later the same worker alerted again.
    mod, emitted, _ = _module()                                # close_after_present_s 2 (the module default)
    t = _feed(mod, [_worker()] * 6 + [_worker(head="Hardhat")] * 30 + [_worker()] * 12)   # 1.5 s of "helmet"
    _feed(mod, [_worker()] * 3, start=t + 30.0)
    assert [a["pts"] for a in _alerts(emitted)] == ["0.250"]
    mod, emitted, _ = _module(temporal={"close_after_present_s": 0})                      # the evaluated rule
    t = _feed(mod, [_worker()] * 6 + [_worker(head="Hardhat")] * 30 + [_worker()] * 12)
    _feed(mod, [_worker()] * 3, start=t + 30.0)
    assert len(_alerts(emitted)) == 2


def test_items_starting_within_the_hold_give_one_alert_naming_both():
    mod, emitted, clips = _module(alert_hold_s=1.0)            # the default hold
    _feed(mod, [_worker()] * 3 + [_worker(torso="NO-Safety Vest")] * 17)    # helmet at 0.25 s, vest at 0.40 s
    assert _alerts(emitted) == []                              # 0.95 s: still held
    _feed(mod, [_worker(torso="NO-Safety Vest")] * 10, start=1.0)
    (a,) = _alerts(emitted)
    assert (a["pts"], a["new"], a["missing"], a["count"]) == ("0.250", "helmet,vest", "helmet,vest", "2")
    assert a["message"] == "Person #12 without helmet and vest on cam-1"
    assert clips == [("cam-1", 0.25)]                          # requested when the violation started


def test_an_item_starting_after_the_hold_is_its_own_alert():
    mod, emitted, _ = _module(alert_hold_s=1.0)
    _feed(mod, [_worker()] * 30 + [_worker(torso="NO-Safety Vest")] * 40)   # vest from 1.5 s, 1.25 s after helmet
    assert [(a["pts"], a["new"]) for a in _alerts(emitted)] == [("0.250", "helmet"), ("1.750", "vest")]


def test_a_held_alert_is_raised_when_the_person_has_left_and_at_shutdown():
    mod, emitted, _ = _module(alert_hold_s=1.0)
    t = _feed(mod, [_worker()] * 8 + [[]] * 30)                # the worker leaves right after the violation starts
    assert [a["pts"] for a in _alerts(emitted)] == ["0.250"]
    mod, emitted, _ = _module(alert_hold_s=1.0)
    _feed(mod, [_worker()] * 8)                                # the stream ends while the alert is held
    assert _alerts(emitted) == []
    mod.stop()
    assert [a["pts"] for a in _alerts(emitted)] == ["0.250"]


def test_a_worker_missing_both_items_on_the_same_frame_gives_one_alert_naming_both():
    mod, emitted, clips = _module()
    _feed(mod, [_worker(torso="NO-Safety Vest")] * 20)
    (a,) = _alerts(emitted)
    assert a["message"] == "Person #12 without helmet and vest on cam-1"
    assert (a["count"], a["new"], a["missing"]) == ("2", "helmet,vest", "helmet,vest")
    assert len(clips) == 1


def test_items_that_start_on_different_frames_give_one_alert_each_naming_all_missing():
    # The rules decide each item on its own: an item becomes a violation on its own 6th absent frame.
    mod, emitted, _ = _module()
    _feed(mod, [_worker()] * 3 + [_worker(torso="NO-Safety Vest")] * 17)
    assert [(a["pts"], a["new"], a["message"]) for a in _alerts(emitted)] == [
        ("0.250", "helmet", "Person #12 without helmet on cam-1"),
        ("0.400", "vest", "Person #12 without helmet and vest on cam-1")]


def test_only_required_items_alert():
    mod, emitted, _ = _module(required_items=["vest"])
    _feed(mod, [_worker()] * 20)                               # bare head, no vest evidence
    assert _alerts(emitted) == []


def _zones(tmp_path, block):
    import json
    path = tmp_path / "zones_config.txt"
    path.write_text(json.dumps({"cam-1": {"zones": [
        {"id": "bay-a", "name": "Bay A", "points": [[0, 0], [400, 0], [400, 720], [0, 720]], **block}]}}))
    return str(path)


def test_with_ppe_zones_a_worker_inside_one_alerts_and_names_it(tmp_path):
    mod, emitted, clips = _module(zones_path=_zones(tmp_path, {"ppe": {"enabled": True}}))
    _feed(mod, [_worker(x=100) for _ in range(40)])
    (a,) = _alerts(emitted)
    assert a["zone"] == "bay-a" and clips


def test_with_ppe_zones_a_worker_outside_them_gives_no_alert_and_no_clip(tmp_path):
    mod, emitted, clips = _module(zones_path=_zones(tmp_path, {"ppe": {"enabled": True}}))
    _feed(mod, [_worker(x=700) for _ in range(40)])
    assert _alerts(emitted) == [] and clips == []


def test_without_ppe_zones_the_whole_frame_is_covered(tmp_path):
    mod, emitted, _ = _module(zones_path=_zones(tmp_path, {"intrusion": {"mode": "restricted"}}))
    _feed(mod, [_worker(x=700) for _ in range(40)])
    (a,) = _alerts(emitted)
    assert a["zone"] == ""


def test_a_worker_who_walks_into_a_ppe_zone_still_bare_headed_alerts_there(tmp_path):
    """The rules flag a missing item once, on the frame it starts. Started outside every PPE zone, it must still
    alert when the worker steps into one while it is still missing (the usual case: entering from elsewhere)."""
    mod, emitted, clips = _module(zones_path=_zones(tmp_path, {"ppe": {"enabled": True}}))
    t = _feed(mod, [_worker(x=700) for _ in range(40)])                      # bare-headed, outside the zone
    assert _alerts(emitted) == [] and clips == []
    _feed(mod, [_worker(x=100) for _ in range(200)], start=t + 1 / FPS)     # walks in, still bare-headed
    (a,) = _alerts(emitted)
    assert a["zone"] == "bay-a" and a["missing"] == "helmet" and len(clips) == 1


def test_a_worker_who_puts_the_helmet_on_before_walking_in_does_not_alert(tmp_path):
    mod, emitted, _ = _module(zones_path=_zones(tmp_path, {"ppe": {"enabled": True}}))
    t = _feed(mod, [_worker(x=700) for _ in range(40)])
    t = _feed(mod, [_worker(x=700, head="Hardhat") for _ in range(60)], start=t + 1 / FPS)
    _feed(mod, [_worker(x=100, head="Hardhat") for _ in range(100)], start=t + 1 / FPS)
    assert _alerts(emitted) == []


def _two_zones(tmp_path, left_items, right_items):
    """Bay A on the left (x 0-400) and the Dock on the right (x 600-1080), each checking the items given."""
    import json
    path = tmp_path / "zones_config.txt"
    path.write_text(json.dumps({"cam-1": {"zones": [
        {"id": "bay-a", "name": "Bay A", "points": [[0, 0], [400, 0], [400, 720], [0, 720]],
         "ppe": {"enabled": True, "items": left_items}},
        {"id": "dock", "name": "Dock", "points": [[600, 0], [1080, 0], [1080, 720], [600, 720]],
         "ppe": {"enabled": True, "items": right_items}}]}}))
    return str(path)


def test_a_hard_hat_only_zone_alerts_a_bare_head_and_names_only_the_helmet(tmp_path):
    mod, emitted, clips = _module(zones_path=_zones(tmp_path, {"ppe": {"enabled": True, "items": ["helmet"]}}))
    _feed(mod, [_worker(x=100, torso="NO-Safety Vest") for _ in range(40)])     # bare head AND no vest
    (a,) = _alerts(emitted)
    assert (a["missing"], a["new"], a["count"]) == ("helmet", "helmet", "1")
    assert a["message"] == "Person #12 without helmet on cam-1" and len(clips) == 1


def test_a_hard_hat_only_zone_lets_a_worker_without_a_vest_pass(tmp_path):
    mod, emitted, clips = _module(zones_path=_zones(tmp_path, {"ppe": {"enabled": True, "items": ["helmet"]}}))
    _feed(mod, [_worker(x=100, head="Hardhat", torso="NO-Safety Vest") for _ in range(40)])
    assert _alerts(emitted) == [] and clips == []


def test_a_vest_only_zone_lets_a_bare_head_pass(tmp_path):
    mod, emitted, _ = _module(zones_path=_zones(tmp_path, {"ppe": {"enabled": True, "items": ["vest"]}}))
    _feed(mod, [_worker(x=100, torso="Safety Vest") for _ in range(40)])
    assert _alerts(emitted) == []


def test_a_missing_vest_unchecked_where_it_started_alerts_in_the_zone_that_checks_it(tmp_path):
    mod, emitted, _ = _module(zones_path=_two_zones(tmp_path, ["helmet"], ["vest"]))
    t = _feed(mod, [_worker(x=100, head="Hardhat", torso="NO-Safety Vest") for _ in range(40)])   # in Bay A
    assert _alerts(emitted) == []
    _feed(mod, [_worker(x=700, head="Hardhat", torso="NO-Safety Vest") for _ in range(40)], start=t + 1 / FPS)
    (a,) = _alerts(emitted)
    assert (a["zone"], a["missing"]) == ("dock", "vest")


def test_where_ppe_zones_overlap_a_worker_is_held_to_what_any_of_them_checks(tmp_path):
    import json
    path = tmp_path / "zones_config.txt"
    path.write_text(json.dumps({"cam-1": {"zones": [
        {"id": "bay-a", "name": "Bay A", "points": [[0, 0], [400, 0], [400, 720], [0, 720]],
         "ppe": {"enabled": True, "items": ["helmet"]}},
        {"id": "aisle", "name": "Aisle", "points": [[100, 0], [700, 0], [700, 720], [100, 720]],
         "ppe": {"enabled": True, "items": ["vest"]}}]}}))
    mod, emitted, _ = _module(zones_path=str(path))
    _feed(mod, [_worker(x=200, head="Hardhat", torso="NO-Safety Vest") for _ in range(40)])   # in both zones
    (a,) = _alerts(emitted)
    assert (a["zone"], a["missing"]) == ("bay-a", "vest")


def test_a_zone_whose_items_the_module_does_not_check_says_so_once(tmp_path, caplog):
    mod, emitted, _ = _module(zones_path=_zones(tmp_path, {"ppe": {"enabled": True, "items": ["mask"]}}))
    with caplog.at_level("WARNING", logger="iris.sidecar"):
        _feed(mod, [_worker(x=100) for _ in range(40)])
    assert _alerts(emitted) == []
    warned = [r for r in caplog.records if "bay-a" in r.getMessage() and "checks none" in r.getMessage()]
    assert len(warned) == 1


def test_a_worker_s_clip_box_and_highlight_name_only_what_their_zone_checks(tmp_path):
    import json
    mod, emitted, _ = _module(zones_path=_zones(tmp_path, {"ppe": {"enabled": True, "items": ["helmet"]}}))
    frames = [_worker(x=100, head="Hardhat", torso="NO-Safety Vest") for _ in range(40)]
    _feed(mod, frames)
    (worker,) = [json.loads(f["shapes"]) for key, f in emitted if key == "iris:overlay:cam-1"][-1]
    assert "label" not in worker                                  # missing a vest the zone does not check
    out = FrameOutput()
    mod.analyze(FrameView("cam-1", None, 0, 99, 9.0, 9.0, mux_size=MUX, _objects=frames[-1]), out)
    assert not [a for a in out.annotations.values() if " no " in (a.text or "")]     # no yellow "#12 no vest"


def test_the_workers_boxes_go_to_the_overlay_stream_so_the_alert_clip_shows_who():
    """Each evaluated worker's box, by track id, labelled with what is missing (app/overlay_track.py draws the alert's
    own worker), at most every 0.1 s."""
    import json
    mod, emitted, _ = _module()
    _feed(mod, [_worker() for _ in range(40)])                 # 2 s at 20 fps
    entries = [json.loads(f["shapes"]) for key, f in emitted if key == "iris:overlay:cam-1"]
    assert 15 <= len(entries) <= 21
    (worker,) = entries[-1]
    assert (worker["style"], worker["id"], worker["rect"]) == ("alert", 12, [100.0, 200.0, 100.0, 300.0])
    assert worker["label"] == "no helmet"
