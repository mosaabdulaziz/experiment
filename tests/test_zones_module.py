"""perception/modules/zones: alerts, filter gating and colours, driven with stand-ins (no GStreamer)."""
import json
import os
import types

import pytest

from perception.engine.frame import FrameOutput, FrameView, ObjectView
from perception.engine.modules import EngineServices, ModuleConfigError, PipelineHandle, load_modules
from perception.modules.zones import APPLIED_KEY, VERSION_KEY, Zones, hazard_overlap_warnings
from perception.modules.zones.analytics import (
    read_zones, render_analytics_config, slot_map, write_worker_config)
from perception.modules.zones.colors import BOX_GRAY, BOX_GREEN, BOX_RED
from perception.modules.zones.runtime import ZoneRuntime
from perception.engine.sources import make_stream_state


class FakeRedis:
    def __init__(self):
        self.values, self.writes = {}, []

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value):
        self.values[key] = value
        self.writes.append((key, value))


def _services(tmp_path, streams=("cam-1",), emitted=None, clips=None, redis=None):
    states = {sid: make_stream_state(sid) for sid in streams}
    emitted = [] if emitted is None else emitted
    clips = [] if clips is None else clips
    return EngineServices(
        redis=redis or FakeRedis(), redis_factory=lambda: None,
        emit=lambda key, fields, maxlen=64: emitted.append((key, fields)) or True,
        request_clip=lambda sid, now, age=0.0: clips.append((sid, now)), stream_ids=list(states),
        timer=lambda i, f: None, telemetry=None, states=states, mux_size=(1000, 500),
        sources=[{"id": s} for s in streams], all_sources=[{"id": s} for s in streams], role=None,
        config={"overlay_boxes": True})                        # as the safety stack's roles (products/safety)


def _zones_file(tmp_path, zones, name="zones_config.txt"):
    p = tmp_path / name
    p.write_text(json.dumps({"cam-1": {"zones": zones}}))
    return str(p)


def _module(tmp_path, zones=None, **settings):
    emitted, clips = [], []
    svc = _services(tmp_path, emitted=emitted, clips=clips)
    path = _zones_file(tmp_path, zones if zones is not None else [
        {"id": "z0", "name": "Door", "points": [[0, 0], [1, 0], [1, 1]],
         "intrusion": {"mode": "restricted", "classes": ["person"]},
         "overcrowding": {"enabled": True, "threshold": 2, "classes": ["person"]}}])
    mod = Zones(Zones.Settings(config_path=path, analytics_config_path=str(tmp_path / "roi.txt"), **settings), svc)
    mod.on_built(PipelineHandle(pipeline=None, get=lambda n: None))
    return mod, emitted, clips


def _obj(oid, label="person", roi=("z0",), box=(100, 100, 50, 80), conf=0.9):
    return ObjectView(oid, label, conf, conf, *box, attrs={"roi": frozenset(roi)})


def _frame(now, objects, stream="cam-1", analyze=True):
    f = FrameView(stream_id=stream, state=None, pad_index=0, frame_num=int(now * 10), pts=now, now=now,
                  analyze=analyze, mux_size=(1000, 500))
    f.objects = objects
    return f


def test_an_object_entering_a_restricted_zone_alerts_once_and_asks_for_a_clip(tmp_path):
    mod, emitted, clips = _module(tmp_path, overcrowding_enabled=False)
    for i in range(5):
        mod.on_frame(_frame(i * 0.1, [_obj(7)]), FrameOutput())
    alerts = [e for e in emitted if e[0] == "iris:zone_alert:cam-1"]
    assert len(alerts) == 1 and alerts[0][1]["object_id"] == 7 and alerts[0][1]["zone"] == "z0"
    assert alerts[0][1]["object_label"] == "person" and len(clips) >= 1


def test_an_object_outside_the_zone_or_of_another_class_does_not_alert(tmp_path):
    mod, emitted, _ = _module(tmp_path)
    mod.on_frame(_frame(0.0, [_obj(1, roi=()), _obj(2, "car")]), FrameOutput())
    assert [e for e in emitted if e[0].startswith("iris:zone_alert")] == []


def test_a_valid_zone_alerts_when_the_object_is_outside_it(tmp_path):
    zone = {"id": "z0", "name": "Floor", "points": [[0, 0], [1, 0], [1, 1]],
            "intrusion": {"mode": "valid", "classes": ["person"]}}
    mod, emitted, _ = _module(tmp_path, zones=[zone])
    mod.on_frame(_frame(0.0, [_obj(1, roi=())]), FrameOutput())
    assert len([e for e in emitted if e[0].startswith("iris:zone_alert")]) == 1


def test_overcrowding_fires_at_the_zones_threshold_and_releases_with_a_debounce(tmp_path):
    mod, emitted, _ = _module(tmp_path, overcrowding_release_seconds=5.0)
    crowd = [_obj(1), _obj(2)]
    mod.on_frame(_frame(0.0, crowd), FrameOutput())
    mod.on_frame(_frame(0.1, crowd), FrameOutput())
    assert len([e for e in emitted if e[0] == "iris:overcrowd_alert:cam-1"]) == 1
    mod.on_frame(_frame(0.2, [_obj(1)]), FrameOutput())
    mod.on_frame(_frame(1.0, crowd), FrameOutput())                 # back over the threshold inside the debounce
    assert len([e for e in emitted if e[0] == "iris:overcrowd_alert:cam-1"]) == 1


def test_a_skipped_frame_raises_no_alert_and_keeps_the_last_colouring(tmp_path):
    mod, emitted, _ = _module(tmp_path)
    mod.on_frame(_frame(0.0, [_obj(7)]), FrameOutput())
    emitted.clear()
    out, person = FrameOutput(), _obj(7)
    skipped = _frame(0.1, [person], analyze=False)
    mod.on_frame(skipped, out)
    mod.analyze(skipped, out)
    assert emitted == [] and out.annotations[person].color == BOX_RED


def test_boxes_are_coloured_by_zone_status_and_a_face_box_beside_its_person_is_not(tmp_path):
    mod, _, _ = _module(tmp_path)
    violator, bystander, car = _obj(7), _obj(8, roi=()), _obj(9, "car", roi=())
    frame = _frame(0.0, [violator, bystander, car])
    mod.on_frame(frame, FrameOutput())
    out = FrameOutput()
    mod.analyze(frame, out)
    assert out.annotations[violator].color == BOX_RED
    assert out.annotations[bystander].color == BOX_GREEN
    assert out.annotations[car].color == BOX_GRAY


def test_without_an_intrusion_zone_nothing_is_coloured(tmp_path):
    zone = {"id": "z0", "name": "Count", "points": [[0, 0], [1, 0], [1, 1]],
            "intrusion": {"mode": "off"}, "overcrowding": {"enabled": True, "threshold": 3, "classes": ["person"]}}
    mod, _, _ = _module(tmp_path, zones=[zone])
    frame = _frame(0.0, [_obj(1)])
    mod.on_frame(frame, FrameOutput())
    out = FrameOutput()
    mod.analyze(frame, out)
    assert out.annotations == {}


def test_a_hazard_colour_set_by_another_module_beats_the_zone_colour(tmp_path):
    mod, _, _ = _module(tmp_path)
    fire = _obj(7, "fire")
    frame = _frame(0.0, [fire])
    out = FrameOutput()
    out.annotate(fire, color=(0, 140, 255), priority=10)
    mod.on_frame(frame, out)
    mod.analyze(frame, out)
    assert out.annotations[fire].color == (0, 140, 255)


def test_removing_the_zones_clears_the_cached_colouring(tmp_path):
    mod, _, _ = _module(tmp_path)
    mod.on_frame(_frame(0.0, [_obj(7)]), FrameOutput())
    mod.services.state("zones", "cam-1")["zones"] = []
    frame = _frame(0.1, [_obj(7)])
    mod.on_frame(frame, FrameOutput())
    out = FrameOutput()
    mod.analyze(frame, out)
    assert out.annotations == {}


def _filter_module(tmp_path):
    emitted = []
    svc = _services(tmp_path, emitted=emitted)
    (tmp_path / "filter_zones.json").write_text(json.dumps(
        {"cam-1": {"zones": [{"points": [[0, 0], [500, 0], [500, 500], [0, 500]]}]}}))   # left half of 1000x500
    path = _zones_file(tmp_path, [])
    mod = Zones(Zones.Settings(config_path=path, analytics_config_path=str(tmp_path / "roi.txt")), svc)
    mod.on_built(PipelineHandle(pipeline=None, get=lambda n: None))
    return mod


def test_filter_zones_drop_detections_whose_centre_is_outside_all_polygons(tmp_path):
    mod = _filter_module(tmp_path)
    inside, outside = _obj(1, box=(100, 100, 100, 100)), _obj(2, box=(800, 100, 100, 100))
    face = _obj(3, box=(900, 50, 20, 20))
    frame = _frame(0.0, [inside, outside, face])
    mod.transform(frame)
    assert (inside.draw, inside.publish) == (True, True)
    assert (outside.draw, outside.publish) == (False, False)
    assert (face.draw, face.publish) == (False, False)          # a drawn box outside is dropped as well


def test_a_camera_without_filter_polygons_is_not_filtered(tmp_path):
    mod, _, _ = _module(tmp_path)
    far = _obj(1, box=(900, 400, 50, 50))
    mod.transform(_frame(0.0, [far]))
    assert far.draw and far.publish


def test_a_new_zones_version_reloads_and_is_acknowledged(tmp_path):
    redis = FakeRedis()
    svc = _services(tmp_path, redis=redis)
    path = _zones_file(tmp_path, [])
    mod = Zones(Zones.Settings(config_path=path, analytics_config_path=str(tmp_path / "roi.txt")), svc)
    reloaded = []
    mod._analytics = types.SimpleNamespace(set_property=lambda k, v: reloaded.append((k, v)))
    mod._load()
    assert svc.state("zones", "cam-1")["zones"] == []
    _zones_file(tmp_path, [{"id": "z1", "name": "New", "points": [[0, 0], [1, 0], [1, 1]],
                            "intrusion": {"mode": "restricted"}}])
    redis.values[VERSION_KEY] = b"4"
    mod.control()[VERSION_KEY](b"4")
    assert [z.label for z in svc.state("zones", "cam-1")["zones"]] == ["z1"]
    assert reloaded and reloaded[0][0] == "config-file"
    assert (APPLIED_KEY, 4) in redis.writes


def test_a_failed_reload_is_retried_and_leaves_no_ack(tmp_path):
    """set_property fails once: the applied text (_config_text) and the acks must stay as they were, so the next
    poll re-renders the *new* config, retries the reload, and acks only after set_property succeeds. If the applied
    text were recorded before set_property, the retry would see nothing to do and ack the version the element never
    loaded -- the UI would show 'active' over the old ROIs."""
    redis = FakeRedis()
    svc = _services(tmp_path, redis=redis)
    path = _zones_file(tmp_path, [])
    mod = Zones(Zones.Settings(config_path=path, analytics_config_path=str(tmp_path / "roi.txt")), svc)
    mod.elements()                                      # the header-only render is the applied baseline
    baseline = mod._config_text
    assert baseline is not None

    calls = []

    def set_property(key, value):
        calls.append((key, open(value).read()))
        if len(calls) == 1:                            # the first reload fails, the retry succeeds
            raise RuntimeError("set_property boom")
    mod._analytics = types.SimpleNamespace(set_property=set_property)

    _zones_file(tmp_path, [{"id": "z1", "name": "New", "points": [[0, 0], [1, 0], [1, 1]],
                            "intrusion": {"mode": "restricted"}}])
    redis.values[VERSION_KEY] = b"1"

    # the reload fails: the bump propagates, nothing is acked, the applied text is untouched
    with pytest.raises(RuntimeError):
        mod.control()[VERSION_KEY](b"1")
    assert len(calls) == 1 and calls[0][0] == "config-file"     # a reload was attempted
    assert mod._config_text == baseline                         # still the previously applied text
    assert APPLIED_KEY not in redis.values                      # the version was not acknowledged...
    assert f"{APPLIED_KEY}:cam-1" not in redis.values           # ...on the global key or the camera's

    # the next poll retries the same version: it re-renders the new ROIs, reloads, and only then acks
    mod.control()[VERSION_KEY](b"1")
    assert len(calls) == 2 and "roi-z1=0;0;1;0;1;1" in calls[1][1]   # the new config reached the element
    assert mod._config_text != baseline                             # the applied text advanced after success
    assert redis.values[APPLIED_KEY] == 1
    assert redis.values[f"{APPLIED_KEY}:cam-1"] == 1


def test_startup_acknowledges_the_version_that_is_already_applied(tmp_path):
    redis = FakeRedis()
    redis.values[VERSION_KEY] = b"9"
    svc = _services(tmp_path, redis=redis)
    mod = Zones(Zones.Settings(config_path=_zones_file(tmp_path, []), analytics_config_path=str(tmp_path / "r.txt")), svc)
    mod.on_built(PipelineHandle(pipeline=None, get=lambda n: None))
    assert (APPLIED_KEY, 9) in redis.writes


def test_it_reads_regions_only_on_analyzed_frames_of_a_zoned_camera(tmp_path):
    mod, _, _ = _module(tmp_path)
    assert mod.extract(_frame(0.0, [])) is not None
    assert mod.extract(_frame(0.0, [], analyze=False)) is None
    nowhere = tmp_path / "empty"
    nowhere.mkdir()
    empty, _, _ = _module(nowhere, zones=[])
    assert empty.extract(_frame(0.0, [])) is None


def test_the_module_adds_the_analytics_element_with_its_own_rendered_config(tmp_path):
    svc = _services(tmp_path)
    cfg = tmp_path / "data" / "roi.txt"
    mod = Zones(Zones.Settings(config_path=_zones_file(tmp_path, [_ZONE]), analytics_config_path=str(cfg)), svc)
    (spec,) = mod.elements()
    assert spec.name == "analytics" and spec.factory == "nvdsanalytics"
    path = spec.properties["config-file"]
    assert path.endswith("roi_worker_single.txt")
    assert open(path).read() == _PROPERTY_HEADER_1000 + _section(0, "roi-z0=0;0;1;0;1;1")
    assert not cfg.exists()                         # the app's shared file is no longer read or needed


def test_a_role_process_writes_its_own_file(tmp_path):
    svc = _services(tmp_path)
    svc.role = "ppe"
    mod = Zones(Zones.Settings(config_path=_zones_file(tmp_path, []), analytics_config_path=str(tmp_path / "roi.txt")),
                svc)
    (spec,) = mod.elements()
    assert spec.properties["config-file"] == str(tmp_path / "roi_worker_ppe.txt")


def test_the_render_is_written_beside_the_configured_file_and_falls_back_to_it_when_unwritable(tmp_path):
    ok = write_worker_config(str(tmp_path / "shared.txt"), _PROPERTY_HEADER_1000, None, 1000, 500)
    assert ok == str(tmp_path / "shared_worker_single.txt")
    assert open(ok).read() == _PROPERTY_HEADER_1000
    blocker = tmp_path / "ro"
    blocker.write_text("")                          # a file where the output directory should be
    assert write_worker_config(str(blocker / "roi.txt"), "x", None, 1000, 500) == str(blocker / "roi.txt")


def test_the_fallback_is_header_only_and_overwrites_a_stale_config(tmp_path):
    """When this process cannot write its own render, the shared fallback file must be header-only -- never an old
    file left there with stale [roi-filtering-stream-N] sections that would keep applying wrong ROIs."""
    from perception.modules.zones.analytics import worker_config_path, write_worker_config
    analytics = tmp_path / "nvds.txt"
    analytics.write_text(_PROPERTY_HEADER_1000 + _section(5, "roi-z9=0;0;1;0;1;1"))   # an old file with a stale section
    os.mkdir(worker_config_path(str(analytics), None))                # a directory where this process's file goes:
    path = write_worker_config(str(analytics), _PROPERTY_HEADER_1000  # os.replace onto it fails -> the fallback runs
                               + _section(0, "roi-z0=0;0;1;0;1;1"), None, 1000, 500)
    assert path == str(analytics)                                     # fell back to the shared file
    assert analytics.read_text() == _PROPERTY_HEADER_1000             # header only; the stale section is gone
    assert "[roi-filtering-stream" not in analytics.read_text()


def test_the_reload_re_renders_from_the_zones_file(tmp_path):
    svc = _services(tmp_path)
    zones = _zones_file(tmp_path, [])
    mod = Zones(Zones.Settings(config_path=zones, analytics_config_path=str(tmp_path / "roi.txt")), svc)
    reloaded = []
    mod._analytics = types.SimpleNamespace(set_property=lambda k, v: reloaded.append(v))
    (spec,) = mod.elements()
    assert open(spec.properties["config-file"]).read() == _PROPERTY_HEADER_1000
    _zones_file(tmp_path, [_ZONE])
    mod.control()[VERSION_KEY](b"1")
    assert reloaded == [spec.properties["config-file"]]
    assert open(reloaded[0]).read() == _PROPERTY_HEADER_1000 + _section(0, "roi-z0=0;0;1;0;1;1")


def test_a_save_between_render_and_on_built_is_loaded_not_just_acked(tmp_path):
    """The pipeline is built from the config elements() renders; a zone saved before on_built must be applied, not
    just acknowledged, or the element would run the stale ROIs until the next save."""
    svc = _services(tmp_path)
    mod = Zones(Zones.Settings(config_path=_zones_file(tmp_path, []), analytics_config_path=str(tmp_path / "roi.txt")),
                svc)
    (spec,) = mod.elements()
    assert open(spec.properties["config-file"]).read() == _PROPERTY_HEADER_1000      # rendered with no zones
    _zones_file(tmp_path, [_ZONE])                                                    # saved during the build window
    reloaded = []
    element = types.SimpleNamespace(set_property=lambda k, v: reloaded.append((k, v)))
    mod.on_built(PipelineHandle(pipeline=None, get=lambda n: element))
    assert reloaded and reloaded[0][0] == "config-file"
    assert open(reloaded[0][1]).read() == _PROPERTY_HEADER_1000 + _section(0, "roi-z0=0;0;1;0;1;1")


def test_on_built_does_not_reload_when_the_render_is_unchanged(tmp_path):
    svc = _services(tmp_path)
    mod = Zones(Zones.Settings(config_path=_zones_file(tmp_path, [_ZONE]),
                               analytics_config_path=str(tmp_path / "roi.txt")), svc)
    mod.elements()
    reloaded = []
    element = types.SimpleNamespace(set_property=lambda k, v: reloaded.append((k, v)))
    mod.on_built(PipelineHandle(pipeline=None, get=lambda n: element))
    assert reloaded == []                                # the file is unchanged since the render: no set_property


def test_it_loads_by_name_with_defaults_and_rejects_unknown_settings(tmp_path):
    svc = _services(tmp_path)
    assert load_modules({"zones": {}}, svc).names() == ["zones"]
    with pytest.raises(ModuleConfigError, match="unknown setting"):
        load_modules({"zones": {"grace": 1}}, svc)


def test_zone_runtime_intrusion_enabled_reflects_mode():
    assert ZoneRuntime(label="z0", name="A", mode="restricted").intrusion_enabled is True
    assert ZoneRuntime(label="z0", name="A", mode="valid").intrusion_enabled is True
    assert ZoneRuntime(label="z0", name="A", mode="off").intrusion_enabled is False


# ── render_analytics_config (each process numbers sections by its own streammux slots) ───

_PROPERTY_HEADER = ("[property]\n"
                    "enable=1\n"
                    "config-width=1920\n"
                    "config-height=1080\n"
                    "osd-mode=0\n")
_PROPERTY_HEADER_1000 = _PROPERTY_HEADER.replace("1920", "1000").replace("1080", "500")
_ZONE = {"id": "z0", "name": "Door", "points": [[0, 0], [1, 0], [1, 1]], "intrusion": {"mode": "restricted"}}


def _section(slot, roi="roi-z1=0;0;10;0;10;10"):
    return (f"\n[roi-filtering-stream-{slot}]\n"
            "enable=1\n"
            f"{roi}\n"
            "inverse-roi=0\n"
            "class-id=-1\n")


def _zoned(*cams, roi_id="z1", points=((0, 0), (10, 0), (10, 10))):
    return {c: {"zones": [{"id": roi_id, "points": [list(p) for p in points], "intrusion": {"mode": "restricted"}}]}
            for c in cams}


def test_the_golden_zones_fixture_renders_byte_identically():
    """tests/configs/golden-zones-roi.txt is what the app rendered for the golden zones setup (one camera)."""
    zones = read_zones("tests/configs/golden_zones_config.json")
    expected = open("tests/configs/golden-zones-roi.txt").read()
    assert render_analytics_config(zones, slot_map([{"id": "cam-1"}]), 1280, 720) == expected


def test_sections_are_numbered_by_this_process_s_slots_not_the_global_list():
    # global list cam-1..cam-10; this process skipped the two fire cameras (cam-7, cam-8)
    zones = {**_zoned("cam-9"), **_zoned("cam-10", roi_id="z2", points=((1, 1), (2, 2), (3, 3)))}
    local = [{"id": f"cam-{i}"} for i in (1, 2, 3, 4, 5, 6, 9, 10)]
    out = render_analytics_config(zones, slot_map(local), 1920, 1080)
    assert out == _PROPERTY_HEADER + _section(6) + _section(7, "roi-z2=1;1;2;2;3;3")


def test_cameras_this_process_does_not_own_get_no_section():
    zones = {**_zoned("cam-7", roi_id="fire", points=((1, 1), (2, 2), (3, 3))), **_zoned("cam-9")}
    out = render_analytics_config(zones, {6: "cam-9"}, 1920, 1080)
    assert out == _PROPERTY_HEADER + _section(6)


def test_no_owned_zones_renders_the_property_section_only():
    assert render_analytics_config(_zoned("cam-7"), {0: "cam-1"}, 1920, 1080) == _PROPERTY_HEADER
    assert render_analytics_config({}, {}, 1920, 1080) == _PROPERTY_HEADER


def test_a_slot_map_with_gaps_keeps_each_camera_on_its_slot():
    """After a detach (#173) the freed slot is simply absent; the other cameras keep their pads."""
    out = render_analytics_config(_zoned("cam-1", "cam-3"), {0: "cam-1", 2: "cam-3"}, 1920, 1080)
    assert out == _PROPERTY_HEADER + _section(0) + _section(2)


def test_only_zones_nvdsanalytics_must_flag_get_an_roi():
    zones = {"cam-1": {"zones": [
        {"points": [[0, 0], [1, 0], [1, 1]], "intrusion": {"mode": "off"}},                       # nothing on
        {"points": [[0, 0], [2, 0], [2, 2]], "intrusion": {"mode": "off"}, "overcrowding": {"enabled": True}},
        {"name": "no points"},                                                                    # skipped
        {"points": [[0, 0], [3, 0], [3, 3]], "proximity": {"enabled": True}},                     # proximity only
        {"points": [[0, 0], [4, 0], [4, 4]], "type": "valid"},                                    # legacy shape
    ]}}
    out = render_analytics_config(zones, {0: "cam-1"}, 1920, 1080)
    assert out == _PROPERTY_HEADER + _section(0, "roi-z1=0;0;2;0;2;2\nroi-z4=0;0;4;0;4;4")


def test_a_legacy_single_zone_entry_renders_as_z0():
    zones = {"cam-1": {"points": [[0, 0], [5, 0], [5, 5]]}}
    assert render_analytics_config(zones, {0: "cam-1"}, 1920, 1080) == _PROPERTY_HEADER + _section(0, "roi-z0=0;0;5;0;5;5")


def test_an_unreadable_zones_file_reads_as_no_zones(tmp_path):
    bad = tmp_path / "zones.txt"
    bad.write_text("{not json")
    assert read_zones(str(bad)) == {}
    assert read_zones(str(tmp_path / "missing.txt")) == {}


# ── zone intrusion classes vs the fire module's classes (guidance, not enforcement) ───────────────────────

def _zone(mode="restricted", classes=("person",), name="Front Door"):
    return ZoneRuntime(label="z0", name=name, mode=mode, intrusion_classes=set(classes))


def test_overlap_warns_when_a_zone_intrusion_class_is_a_hazard_class():
    warnings = hazard_overlap_warnings({"cam-1": [_zone(classes={"fire", "person"})]}, {"fire", "smoke"})
    assert len(warnings) == 1 and "cam-1" in warnings[0] and "Front Door" in warnings[0] and "fire" in warnings[0]


def test_overlap_is_silent_without_overlap_with_intrusion_off_or_without_hazard_classes():
    assert hazard_overlap_warnings({"cam-1": [_zone(classes={"person"})]}, {"fire", "smoke"}) == []
    assert hazard_overlap_warnings({"cam-1": [_zone(mode="off", classes={"fire"})]}, {"fire", "smoke"}) == []
    assert hazard_overlap_warnings({"cam-1": [_zone(classes={"fire"})]}, ()) == []


def test_the_module_warns_at_boot_when_the_fire_module_is_configured_with_overlapping_classes(tmp_path, caplog):
    svc = _services(tmp_path)
    svc.config = {"modules": {"fire_alerts": {"classes": ["fire"]}}}
    path = _zones_file(tmp_path, [{"id": "z0", "name": "Door", "points": [[0, 0], [1, 0], [1, 1]],
                                   "intrusion": {"mode": "restricted", "classes": ["fire"]}}])
    mod = Zones(Zones.Settings(config_path=path, analytics_config_path=str(tmp_path / "roi.txt")), svc)
    with caplog.at_level("WARNING", logger="iris.sidecar"):
        mod.on_built(PipelineHandle(pipeline=None, get=lambda n: None))
    assert any("overlapping modules.fire_alerts.classes" in r.message for r in caplog.records)


def test_role_scoped_keys_put_zone_and_overcrowding_alerts_under_the_role_key(tmp_path):
    mod, emitted, _ = _module(tmp_path)
    mod.services.key_id = lambda sid: f"{sid}@zones"
    mod.on_frame(_frame(0.0, [_obj(1), _obj(2)]), FrameOutput())
    mod.on_frame(_frame(0.1, [_obj(1), _obj(2)]), FrameOutput())
    keys = {key for key, _fields in emitted}
    assert keys == {"iris:zone_alert:cam-1@zones", "iris:overcrowd_alert:cam-1@zones", "iris:overlay:cam-1@zones"}


def _run(mod, frames, stream="cam-1"):
    for now, objects in frames:
        mod.on_frame(_frame(now, objects, stream=stream), FrameOutput())


def _kinds(emitted, kind):
    return [f for _k, f in emitted if f.get("kind") == kind]


def test_no_end_markers_unless_switched_on(tmp_path):
    mod, emitted, _ = _module(tmp_path, overcrowding_release_seconds=1.0)
    _run(mod, [(i * 0.1, [_obj(1), _obj(2)]) for i in range(5)] + [(1.0 + i * 0.1, []) for i in range(40)])
    assert not _kinds(emitted, "overcrowding_resolved") and not _kinds(emitted, "intrusion_cleared")


def test_overcrowding_resolved_when_the_latch_releases(tmp_path):
    mod, emitted, clips = _module(tmp_path, end_markers=True, overcrowding_release_seconds=1.0)
    _run(mod, [(i * 0.1, [_obj(1), _obj(2)]) for i in range(10)])
    assert not _kinds(emitted, "overcrowding_resolved")
    clips.clear()
    _run(mod, [(1.0 + i * 0.1, [_obj(1)]) for i in range(20)])    # below the threshold of 2 for 2 s
    (resolved,) = _kinds(emitted, "overcrowding_resolved")
    assert resolved["zone"] == "z0" and float(resolved["duration"]) > 0
    assert [k for k, f in emitted if f is resolved] == ["iris:overcrowd_alert:cam-1"]
    assert clips == []                                                 # an end asks for no clip


def test_intrusion_cleared_once_the_zone_stays_empty(tmp_path):
    mod, emitted, _ = _module(tmp_path, end_markers=True, intrusion_clear_seconds=1.0, overcrowding_enabled=False)
    _run(mod, [(i * 0.1, [_obj(7)]) for i in range(5)])
    _run(mod, [(0.5 + i * 0.1, []) for i in range(5)])                 # empty for 0.4 s: not yet
    assert not _kinds(emitted, "intrusion_cleared")
    _run(mod, [(1.0 + i * 0.1, []) for i in range(10)])
    (cleared,) = _kinds(emitted, "intrusion_cleared")
    assert cleared["zone"] == "z0" and float(cleared["duration"]) >= 0.4
    assert [k for k, f in emitted if f is cleared] == ["iris:zone_alert:cam-1"]


def test_someone_back_in_the_zone_restarts_the_clear_countdown(tmp_path):
    mod, emitted, _ = _module(tmp_path, end_markers=True, intrusion_clear_seconds=1.0, overcrowding_enabled=False)
    _run(mod, [(0.0, [_obj(7)]), (0.5, []), (0.9, [_obj(7)]), (1.6, []), (2.0, [])])
    assert not _kinds(emitted, "intrusion_cleared")
    _run(mod, [(2.7, [])])
    assert len(_kinds(emitted, "intrusion_cleared")) == 1


_DOOR = {"id": "z0", "name": "Door", "points": [[0, 0], [1, 0], [1, 1]],
         "intrusion": {"mode": "restricted", "classes": ["person"]},
         "overcrowding": {"enabled": True, "threshold": 2, "classes": ["person"]}}


def test_a_zones_reload_ends_the_open_incidents_it_replaces(tmp_path):
    """Re-drawing a stream's zone rebuilds that stream's zones with fresh state; without an end marker the app's
    incident would stay 'ongoing' and swallow the next entry hours later."""
    mod, emitted, _ = _module(tmp_path, end_markers=True, intrusion_clear_seconds=5.0,
                              overcrowding_release_seconds=5.0)
    _run(mod, [(i * 0.1, [_obj(1), _obj(2)]) for i in range(5)])        # intrusion + overcrowding open in z0
    emitted.clear()
    _zones_file(tmp_path, [{**_DOOR, "points": [[0, 0], [2, 0], [2, 2]]}])
    mod._on_zones_version(2)
    kinds = sorted((f["kind"], f["zone"], f.get("reason")) for _k, f in emitted)
    assert kinds == [("intrusion_cleared", "z0", "zones_reloaded"), ("overcrowding_resolved", "z0", "zones_reloaded")]


def test_a_reload_without_open_incidents_or_end_markers_sends_nothing(tmp_path):
    mod, emitted, _ = _module(tmp_path)
    _run(mod, [(i * 0.1, [_obj(1), _obj(2)]) for i in range(5)])
    emitted.clear()
    mod._on_zones_version(2)
    assert emitted == []


def test_intrusion_clear_seconds_must_be_positive():
    from perception.engine.modules import ModuleConfigError, build_settings
    from perception.modules.zones import Settings
    with pytest.raises(ModuleConfigError):
        build_settings(Settings, {"intrusion_clear_seconds": 0}, "zones")


def test_the_people_in_a_zone_go_to_the_overlay_stream_so_the_alert_clip_shows_them(tmp_path):
    """An intrusion's clip draws its intruder (by id), an overcrowding's everyone counted (app/overlay_track.py)."""
    mod, emitted, _ = _module(tmp_path)
    mod.on_frame(_frame(0.0, [_obj(7), _obj(8, box=(300, 100, 50, 80)), _obj(9, roi=())]), FrameOutput())
    (entry,) = [json.loads(f["shapes"]) for k, f in emitted if k == "iris:overlay:cam-1"]
    assert entry == [{"type": "box", "rect": [100.0, 100.0, 50.0, 80.0], "style": "alert", "id": 7, "label": "Door"},
                     {"type": "box", "rect": [300.0, 100.0, 50.0, 80.0], "style": "alert", "id": 8, "label": "Door"}]


# ── targeted reload: only what changed is reloaded or rebuilt ──────────────────────────────────────────────

def _two_cameras(tmp_path, redis=None, end_markers=True):
    emitted = []
    svc = _services(tmp_path, streams=("cam-1", "cam-2"), emitted=emitted, redis=redis)
    zones = tmp_path / "zones_config.txt"
    zones.write_text(json.dumps({"cam-1": {"zones": [_DOOR]}, "cam-2": {"zones": [_DOOR]}}))
    mod = Zones(Zones.Settings(config_path=str(zones), analytics_config_path=str(tmp_path / "roi.txt"),
                               end_markers=end_markers), svc)
    reloads = []
    mod.elements()
    mod.on_built(PipelineHandle(pipeline=None, get=lambda n: types.SimpleNamespace(
        set_property=lambda k, v: reloads.append(open(v).read()))))
    return mod, svc, zones, emitted, reloads


def test_slots_changed_and_a_version_bump_do_not_interleave(tmp_path):
    """The engine's slot-change thread and the control thread's version bump both re-render, reload and rewrite
    stream state. Under the lock their critical sections run one at a time -- never a mixed render."""
    import threading
    import time

    redis = FakeRedis()
    mod, svc, zones, _, _ = _two_cameras(tmp_path, redis=redis)
    inside, max_concurrent = [], [0]
    guard, orig = threading.Lock(), mod._render

    def traced():
        with guard:
            inside.append(1)
            max_concurrent[0] = max(max_concurrent[0], len(inside))
        time.sleep(0.03)                                 # widen the window an interleave would show up in
        with guard:
            inside.pop()
        return orig()

    mod._render = traced
    redis.values[VERSION_KEY] = b"6"
    zones.write_text(json.dumps({"cam-1": {"zones": [{**_DOOR, "points": [[0, 0], [5, 0], [5, 5]]}]},
                                 "cam-2": {"zones": [_DOOR]}}))
    barrier = threading.Barrier(2)
    threads = [threading.Thread(target=lambda: (barrier.wait(), mod.slots_changed({0: "cam-1", 1: "cam-2"}))),
               threading.Thread(target=lambda: (barrier.wait(), mod._on_zones_version(6)))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert max_concurrent[0] == 1                         # the two renders never overlapped


def test_an_unchanged_render_does_not_reload_the_element_or_reset_any_stream(tmp_path):
    redis = FakeRedis()
    mod, svc, _, emitted, reloads = _two_cameras(tmp_path, redis=redis)
    before = {sid: svc.state("zones", sid)["zones"] for sid in ("cam-1", "cam-2")}
    redis.values[VERSION_KEY] = b"3"
    mod.control()[VERSION_KEY](b"3")                    # e.g. another process's camera was edited
    assert reloads == []
    assert all(svc.state("zones", sid)["zones"] is before[sid] for sid in before)
    assert emitted == []
    assert (APPLIED_KEY, 3) in redis.writes              # still acknowledged: nothing to apply here


def test_editing_one_camera_reloads_once_and_rebuilds_only_that_camera(tmp_path):
    mod, svc, zones, emitted, reloads = _two_cameras(tmp_path)
    _run(mod, [(i * 0.1, [_obj(1), _obj(2)]) for i in range(5)], stream="cam-2")   # incidents open on cam-2
    kept = svc.state("zones", "cam-2")["zones"]
    emitted.clear()
    zones.write_text(json.dumps({"cam-1": {"zones": [{**_DOOR, "points": [[0, 0], [5, 0], [5, 5]]}]},
                                 "cam-2": {"zones": [_DOOR]}}))
    mod._on_zones_version(4)
    assert len(reloads) == 1 and "roi-z0=0;0;5;0;5;5" in reloads[0]
    assert svc.state("zones", "cam-2")["zones"] is kept   # cam-2's alert state and open incidents are untouched
    assert emitted == []


def test_a_settings_only_change_rebuilds_the_stream_without_reloading_the_element(tmp_path):
    mod, svc, zones, _, reloads = _two_cameras(tmp_path)
    kept = svc.state("zones", "cam-2")["zones"]
    zones.write_text(json.dumps({"cam-1": {"zones": [{**_DOOR, "overcrowding": {"enabled": True, "threshold": 9}}]},
                                 "cam-2": {"zones": [_DOOR]}}))
    mod._on_zones_version(5)
    assert reloads == []                                  # same ROIs: nvdsanalytics has nothing new
    assert svc.state("zones", "cam-1")["zones"][0].overcrowding_threshold == 9
    assert svc.state("zones", "cam-2")["zones"] is kept


def test_each_camera_s_zones_are_acknowledged_on_its_own_key(tmp_path):
    redis = FakeRedis()
    redis.values[VERSION_KEY] = b"7"
    mod, *_ = _two_cameras(tmp_path, redis=redis)
    assert redis.values[f"{APPLIED_KEY}:cam-1"] == 7 and redis.values[f"{APPLIED_KEY}:cam-2"] == 7
    redis.values[VERSION_KEY] = b"8"
    mod.control()[VERSION_KEY](b"8")
    assert redis.values[f"{APPLIED_KEY}:cam-1"] == 8 and redis.values[f"{APPLIED_KEY}:cam-2"] == 8


# ── slot map changes (attach / detach / slot reuse, #173), simulated ───────────────────────────────────────

def _section_slots(text):
    return [int(line.split("-")[-1].rstrip("]")) for line in text.splitlines() if line.startswith("[roi-filtering")]


def test_attaching_a_camera_adds_its_section_and_leaves_the_others_alone(tmp_path):
    mod, svc, zones, emitted, reloads = _two_cameras(tmp_path)
    svc.stream_ids.append("cam-3")
    svc.states["cam-3"] = make_stream_state("cam-3")
    zones.write_text(json.dumps({c: {"zones": [_DOOR]} for c in ("cam-1", "cam-2", "cam-3")}))
    kept = {sid: svc.state("zones", sid)["zones"] for sid in ("cam-1", "cam-2")}
    mod.slots_changed({0: "cam-1", 1: "cam-2", 2: "cam-3"})
    assert _section_slots(reloads[-1]) == [0, 1, 2]
    assert [z.label for z in svc.state("zones", "cam-3")["zones"]] == ["z0"]
    assert all(svc.state("zones", sid)["zones"] is kept[sid] for sid in kept)
    assert emitted == []


def test_detaching_a_camera_drops_its_section_and_ends_its_incidents(tmp_path):
    mod, svc, _, emitted, reloads = _two_cameras(tmp_path)
    _run(mod, [(i * 0.1, [_obj(1), _obj(2)]) for i in range(5)], stream="cam-2")
    kept = svc.state("zones", "cam-1")["zones"]
    emitted.clear()
    svc.stream_ids.remove("cam-2")                        # the engine releases the pad (its state may linger)
    mod.slots_changed({0: "cam-1"})
    assert _section_slots(reloads[-1]) == [0]
    assert sorted(f["kind"] for _k, f in emitted) == ["intrusion_cleared", "overcrowding_resolved"]
    assert svc.state("zones", "cam-2")["zones"] == []
    assert svc.state("zones", "cam-1")["zones"] is kept


def test_a_detached_camera_still_in_stream_ids_keeps_no_zones(tmp_path):
    """_load follows the slot map, not stream_ids: a detached camera whose pad still lingers in stream_ids must keep
    no zones -- its saved zones are still in the file, but its ROI section is gone, so the state must match."""
    mod, svc, _, _, _ = _two_cameras(tmp_path)
    mod.slots_changed({0: "cam-1"})                       # cam-2 detached; stream_ids still lists it
    assert "cam-2" in svc.stream_ids
    assert svc.state("zones", "cam-2")["zones"] == []
    assert [z.label for z in svc.state("zones", "cam-1")["zones"]] == ["z0"]
    mod._on_zones_version(9)                              # a later bump must not resurrect it from the file
    assert svc.state("zones", "cam-2")["zones"] == []
    assert [z.label for z in svc.state("zones", "cam-1")["zones"]] == ["z0"]


def test_a_reused_slot_gets_the_new_camera_s_zones_with_fresh_state(tmp_path):
    mod, svc, zones, emitted, reloads = _two_cameras(tmp_path)
    _run(mod, [(i * 0.1, [_obj(1), _obj(2)]) for i in range(5)], stream="cam-2")
    svc.stream_ids.remove("cam-2")
    svc.stream_ids.append("cam-9")
    svc.states["cam-9"] = make_stream_state("cam-9")
    other = {**_DOOR, "id": "dock", "points": [[0, 0], [7, 0], [7, 7]]}
    zones.write_text(json.dumps({"cam-1": {"zones": [_DOOR]}, "cam-2": {"zones": [_DOOR]},
                                 "cam-9": {"zones": [other]}}))
    emitted.clear()
    mod.slots_changed({0: "cam-1", 1: "cam-9"})          # cam-2 detached, cam-9 attached on its slot
    assert "[roi-filtering-stream-1]\nenable=1\nroi-dock=0;0;7;0;7;7" in reloads[-1]
    assert "roi-z0" in reloads[-1].split("[roi-filtering-stream-1]")[0]       # slot 0 still cam-1's
    assert [z.label for z in svc.state("zones", "cam-9")["zones"]] == ["dock"]
    assert sorted(f["kind"] for _k, f in emitted) == ["intrusion_cleared", "overcrowding_resolved"]   # cam-2's


# ── replicas, the engine's pad map, the reload probe ───────────────────────────────────────────────────────

def test_shard_replicas_of_one_config_write_separate_files(tmp_path):
    """Shards share the role (None) and the data/ mount: without the shard in the name they overwrite each other."""
    paths = []
    for shard in (0, 1):
        svc = _services(tmp_path)
        svc.shard_id = shard
        mod = Zones(Zones.Settings(config_path=_zones_file(tmp_path, []), analytics_config_path=str(tmp_path / "r.txt")),
                    svc)
        (spec,) = mod.elements()
        paths.append(spec.properties["config-file"])
    assert paths == [str(tmp_path / "r_worker_single.txt"), str(tmp_path / "r_worker_single_shard1.txt")]


def test_slots_come_from_the_engine_s_pad_map(tmp_path):
    svc = _services(tmp_path, streams=("cam-1", "cam-2"))
    svc.index_to_id = {0: "cam-2", 3: "cam-1"}
    zones = tmp_path / "zones_config.txt"
    zones.write_text(json.dumps({"cam-1": {"zones": [_DOOR]}}))
    mod = Zones(Zones.Settings(config_path=str(zones), analytics_config_path=str(tmp_path / "roi.txt")), svc)
    (spec,) = mod.elements()
    assert "[roi-filtering-stream-3]" in open(spec.properties["config-file"]).read()


def test_the_roi_probe_reports_labels_on_every_analyzed_frame_even_without_zones(tmp_path):
    mod, emitted, _ = _module(tmp_path, zones=[], roi_probe=True)
    assert mod.extract(_frame(0.0, [])) is not None              # reads roiStatus although the camera has no zones
    mod.on_frame(_frame(1.0, [_obj(1, roi=("gone",)), _obj(2, roi=())]), FrameOutput())
    mod.on_frame(_frame(1.1, [], analyze=False), FrameOutput())
    probes = [(k, f) for k, f in emitted if k.startswith("iris:zones:roi_probe:")]
    assert probes == [("iris:zones:roi_probe:cam-1", {"pts": "1.000", "at": "0.000", "objects": 2, "labels": "gone"})]


def test_the_roi_probe_is_off_by_default(tmp_path):
    mod, emitted, _ = _module(tmp_path, zones=[])
    assert mod.extract(_frame(0.0, [])) is None
    mod.on_frame(_frame(1.0, [_obj(1)]), FrameOutput())
    assert not [k for k, _ in emitted if k.startswith("iris:zones:roi_probe:")]
