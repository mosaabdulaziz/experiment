"""modules.fire_alerts: the engine-facing behaviour (settings, GIE spec, alert stream, clip requests, colour)."""
import pathlib
import types

import pytest

from perception.engine.frame import FrameOutput, FrameView, ObjectView
from perception.engine.modules import EngineServices, ModuleConfigError, build_settings
from perception.modules.fire_alerts import BOX_HAZARD, HAZARD_CLASSES, MODULE, Settings
from perception.modules.zones.colors import BOX_GRAY, BOX_GREEN, BOX_RED, box_color


def _services(emitted, clips, batch=4):
    st = types.SimpleNamespace(module_state={}, zones=[])
    return EngineServices(
        redis=None, redis_factory=lambda: None, emit=lambda key, fields, maxlen=64: emitted.append((key, fields)),
        request_clip=lambda sid, now, age=0.0: clips.append((sid, round(now, 3))), stream_ids=["cam-1"],
        timer=lambda *a: None, telemetry=None, batch_size=batch, states={"cam-1": st},
        config={"overlay_boxes": True})                       # as the safety stack's fire role (products/safety)


def _module(**overrides):
    emitted, clips = [], []
    mod = MODULE(build_settings(Settings, overrides, "fire_alerts"), _services(emitted, clips))
    mod.on_built(None)
    return mod, emitted, clips


def _frame(now, objects):
    return FrameView("cam-1", None, 0, int(now * 10), now, now, _objects=objects)


def _fire(conf=0.9, oid=1, label="fire"):
    return ObjectView(oid, label, conf, conf, 0, 0, 10, 10)


def test_defaults_match_the_values_the_shared_yaml_used_to_carry():
    s = Settings()
    assert (s.interval, s.classes, s.fire_sustain_seconds, s.smoke_sustain_seconds, s.release_seconds,
            s.min_confidence, s.dropout_grace_seconds) == (0, ["fire", "smoke"], 2.0, 4.0, 10.0, 0.45, 1.0)


@pytest.mark.parametrize("bad", [{"min_confidence": 0}, {"min_confidence": -0.1}, {"classes": []}, {"interval": -1}])
def test_invalid_settings_refuse_startup(bad):
    with pytest.raises(ModuleConfigError):
        build_settings(Settings, bad, "fire_alerts")


def test_the_old_flat_keys_are_not_accepted():
    with pytest.raises(ModuleConfigError, match="unknown setting"):
        build_settings(Settings, {"fire_alert_enabled": True}, "fire_alerts")


def test_it_asks_for_one_full_frame_detector_with_its_own_engine():
    config = str(pathlib.Path(__file__).resolve().parents[1]
                 / "perception/models/DeepStream-Yolo-Fire/config_infer_fire_box.txt")
    mod, _, _ = _module(config=config)
    (spec,) = mod.gies()
    assert (spec.name, spec.role, spec.interval, spec.config_path) == ("pgie_fire", "primary", 0, config)
    assert "_b4_gpu0_" in spec.engine_path            # batch size 4 comes from the engine, not the module


def test_a_sustained_fire_alerts_once_requests_a_clip_and_resolves():
    mod, emitted, clips = _module(fire_sustain_seconds=2.0, release_seconds=3.0)
    out = FrameOutput()
    t = 0.0
    while t <= 2.5:                                            # confident fire on every frame
        mod.on_frame(_frame(t, [_fire()]), out)
        t += 0.1
    started = [f for _k, f in emitted if f.get("kind") == "fire_smoke"]
    assert len(started) == 1 and started[0]["class"] == "fire" and started[0]["count"] == 1
    assert all(k in ("iris:overcrowd_alert:cam-1", "iris:overlay:cam-1") for k, _f in emitted)   # alerts + clip boxes
    assert clips, "an alert starts a clip"
    while t <= 8.0:                                            # fire gone
        mod.on_frame(_frame(t, []), out)
        t += 0.1
    resolved = [f for _k, f in emitted if f.get("kind") == "fire_smoke_resolved"]
    assert len(resolved) == 1 and resolved[0]["class"] == "fire" and float(resolved[0]["duration"]) > 0


def test_a_low_confidence_detection_does_not_count():
    mod, emitted, _ = _module(min_confidence=0.5, fire_sustain_seconds=0.5)
    for i in range(40):
        mod.on_frame(_frame(i * 0.1, [_fire(conf=0.3)]), FrameOutput())
    assert emitted == []


def test_an_open_incident_rearms_the_clip_about_once_a_second():
    mod, _emitted, clips = _module(fire_sustain_seconds=0.5, release_seconds=30.0)
    for i in range(100):                                       # 10 s of fire at 10 fps
        mod.on_frame(_frame(i * 0.1, [_fire()]), FrameOutput())
    rearms = [c for c in clips if c[1] > 1.0]
    assert 6 <= len(rearms) <= 11


def test_fire_and_smoke_boxes_get_the_hazard_colour_that_beats_zone_colours():
    mod, _, _ = _module()
    out = FrameOutput()
    fire, smoke, person = _fire(oid=1), _fire(oid=2, label="smoke"), _fire(oid=3, label="person")
    out.annotate(fire, color=(94, 111, 255), priority=0)       # a zone module's "violating" red, set first
    mod.analyze(_frame(0.0, [fire, smoke, person]), out)
    assert out.annotations[fire].color == BOX_HAZARD and out.annotations[smoke].color == BOX_HAZARD
    assert person not in out.annotations


def test_hazard_classes_are_exactly_fire_and_smoke():
    assert set(HAZARD_CLASSES) == {"fire", "smoke"}


def test_hazard_is_distinct_from_every_zone_colour():
    # Red already means "person violating a zone"; an operator must tell a trespasser from a fire at a glance.
    assert BOX_HAZARD not in (BOX_GREEN, BOX_RED, BOX_GRAY)


def test_hazard_colour_does_not_depend_on_the_zone_colouring_being_on():
    # box_color alone says green (analytics off, no zones); the module's annotation still turns the box hazard-coloured.
    assert box_color(11, "fire", set(), False, {"person"}) == BOX_GREEN
    mod, _, _ = _module()
    out, fire = FrameOutput(), _fire(oid=11)
    mod.analyze(_frame(0.0, [fire]), out)
    assert out.annotations[fire].color == BOX_HAZARD


def test_every_setting_reaches_the_alert_state():
    mod, _, _ = _module(classes=["fire"], fire_sustain_seconds=1.5, smoke_sustain_seconds=6.0, release_seconds=7.0,
                        min_confidence=0.6, dropout_grace_seconds=2.5)
    a = mod.alert_settings
    assert a.enabled is True and a.classes == ("fire",) and a.sustain_seconds == {"fire": 1.5, "smoke": 6.0}
    assert (a.release_seconds, a.min_confidence, a.dropout_grace_seconds) == (7.0, 0.6, 2.5)


def test_role_scoped_keys_put_fire_alerts_under_the_role_key():
    mod, emitted, _ = _module(fire_sustain_seconds=0.5, release_seconds=1.0)
    mod.services.key_id = lambda sid: f"{sid}@fire"
    t = 0.0
    while t <= 1.0:
        mod.on_frame(_frame(t, [_fire()]), FrameOutput())
        t += 0.1
    while t <= 4.0:
        mod.on_frame(_frame(t, []), FrameOutput())
        t += 0.1
    assert [f["kind"] for k, f in emitted if "overlay" not in k] == ["fire_smoke", "fire_smoke_resolved"]
    assert {k for k, _f in emitted} == {"iris:overcrowd_alert:cam-1@fire", "iris:overlay:cam-1@fire"}


def _fire_zones(tmp_path):
    import json
    path = tmp_path / "zones_config.txt"
    path.write_text(json.dumps({"cam-1": {"zones": [
        {"id": "store", "name": "Store", "points": [[0, 0], [400, 0], [400, 720], [0, 720]],
         "fire": {"enabled": True}}]}}))
    return str(path)


def _fire_at(x, conf=0.9):
    return ObjectView(1, "fire", conf, conf, x, 100, 40, 40)


def test_with_fire_zones_a_fire_inside_one_alerts_and_names_it(tmp_path):
    mod, emitted, _ = _module(zones_path=_fire_zones(tmp_path), fire_sustain_seconds=0.5)
    for i in range(15):
        mod.on_frame(_frame(i * 0.1, [_fire_at(100)]), FrameOutput())
    (started,) = [f for _k, f in emitted if f.get("kind") == "fire_smoke"]
    assert started["zone"] == "store"


def test_with_fire_zones_a_fire_outside_them_does_not_count(tmp_path):
    mod, emitted, clips = _module(zones_path=_fire_zones(tmp_path), fire_sustain_seconds=0.5)
    for i in range(30):
        mod.on_frame(_frame(i * 0.1, [_fire_at(700)]), FrameOutput())
    assert emitted == [] and clips == []


def test_without_fire_zones_a_fire_anywhere_alerts_with_no_zone(tmp_path):
    mod, emitted, _ = _module(zones_path=str(tmp_path / "none.txt"), fire_sustain_seconds=0.5)
    for i in range(15):
        mod.on_frame(_frame(i * 0.1, [_fire_at(700)]), FrameOutput())
    (started,) = [f for _k, f in emitted if f.get("kind") == "fire_smoke"]
    assert "zone" not in started            # the entry is exactly what it was before fire zones (attendance's too)


def test_the_fire_and_smoke_boxes_go_to_the_overlay_stream_so_the_alert_clip_shows_them():
    import json
    mod, emitted, _ = _module()
    for i in range(5):                                         # every 0.05 s: written every 0.1 s
        mod.on_frame(_frame(i * 0.05, [_fire(conf=0.9), _fire(conf=0.8, oid=2, label="smoke")]), FrameOutput())
    entries = [json.loads(f["shapes"]) for k, f in emitted if k == "iris:overlay:cam-1"]
    assert len(entries) == 3
    assert entries[0] == [{"type": "box", "rect": [0.0, 0.0, 10.0, 10.0], "style": "fire", "label": "fire 0.90"},
                          {"type": "box", "rect": [0.0, 0.0, 10.0, 10.0], "style": "smoke", "label": "smoke 0.80"}]
