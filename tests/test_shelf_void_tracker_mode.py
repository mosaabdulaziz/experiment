"""modules.shelf_void with stability: tracker -- the accepted shelf-demo's per-region tracker, person pause and
situation (vendored in core/), the alerts each region's situation raises, overlay and meta."""
import json
import pathlib
import types

import numpy as np
import pytest

from perception.engine.frame import FrameOutput, FrameView, ObjectView
from perception.engine.modules import EngineServices, ModuleConfigError, build_settings
from perception.modules.shelf_void import BOX_VOID, MODULE, Settings
from perception.modules.shelf_void.adapter import SHELF_GIE
from perception.modules.shelf_void.stability import FpsProbe, RegionTrackers, TrackerConfig

REPO = pathlib.Path(__file__).resolve().parents[1]
UNTRACKED = 0xFFFFFFFFFFFFFFFF
MUX = (1280, 720)
FPS = 25.0
CONFIRM, MISS = 50, 25                     # 2 s and 1 s at 25 fps
CLEAR = 5.0                                # situation.clear_s in these tests (the default 30 s is just longer)


def _module(**overrides):
    emitted, clips = [], []
    services = EngineServices(
        redis=None, redis_factory=lambda: None, emit=lambda key, fields, maxlen=64: emitted.append((key, fields)),
        request_clip=lambda sid, now, age=None: clips.append((sid, round(now, 3))), stream_ids=["cam-1"],
        timer=lambda *a: None, telemetry=None, batch_size=1,
        states={"cam-1": types.SimpleNamespace(module_state={})})
    raw = {"stability": "tracker", "tracker": {"fps": FPS}, "situation": {"clear_s": CLEAR}, **overrides}
    mod = MODULE(build_settings(Settings, raw, "shelf_void"), services)
    mod.on_built(None)
    return mod, emitted, clips


def _void(left=100, top=100, width=80, height=40, conf=0.1):
    return ObjectView(UNTRACKED, "empty", conf, conf, left, top, width, height, gie_name=SHELF_GIE)


def _feed(mod, frames, start=0):
    """frames: object lists, one per frame at FPS, frame numbers from `start`."""
    for i, objs in enumerate(frames, start):
        t = i / FPS
        mod.on_frame(FrameView("cam-1", None, 0, i, t, t, mux_size=MUX, _objects=objs), FrameOutput())
    return start + len(frames)


def _alerts(emitted):
    return [f for key, f in emitted if key == "iris:alert:cam-1"]


def _rt(mod):
    return mod.services.state("shelf_void", "cam-1")["regions"]


# --- one region: confirm, raise, clear ---

def test_a_gap_raises_shelf_empty_on_its_confirm_th_frame_with_a_clip():
    mod, emitted, clips = _module()
    _feed(mod, [[_void()] for _ in range(CONFIRM - 1)])
    assert _alerts(emitted) == []
    _feed(mod, [[_void()]], start=CONFIRM - 1)
    (a,) = _alerts(emitted)
    assert (a["kind"], a["severity"], a["category"], a["region"], a["reason"], a["count"]) == \
        ("shelf_empty", "warning", "operations", "frame", "new", "1")
    assert a["message"] == "Empty shelf space on cam-1 in region frame: 1 spot at (100,100)-(180,140)"
    assert a["bbox"] == "100,100,180,140" and a["score"] == "0.100" and a["stability"] == "1.000" and a["alert_id"]
    assert json.loads(a["spots"]) == [{"bbox": [0.0781, 0.1389, 0.1406, 0.1944], "detectorScore": 0.1,
                                       "stability": 1.0}]
    assert a["occurred_at"].endswith("Z") and a["pts"] == f"{(CONFIRM - 1) / FPS:.3f}"
    assert clips == [("cam-1", round((CONFIRM - 1) / FPS, 3))]


def test_boxes_below_the_conf_filter_never_count():
    mod, emitted, _ = _module()
    _feed(mod, [[_void(conf=0.079)] for _ in range(3 * CONFIRM)])
    assert _alerts(emitted) == []
    mod, emitted, _ = _module(tracker={"fps": FPS, "conf": 0.05})
    _feed(mod, [[_void(conf=0.079)] for _ in range(CONFIRM)])
    assert len(_alerts(emitted)) == 1


def test_a_gap_seen_in_under_half_the_frames_never_alerts():
    mod, emitted, _ = _module()
    _feed(mod, [[_void()] if i % 3 == 0 else [] for i in range(10 * CONFIRM)])
    assert _alerts(emitted) == []


def test_the_region_clears_after_clear_s_without_spots_as_shelf_restocked_with_the_same_alert_id():
    mod, emitted, _ = _module()
    n = _feed(mod, [[_void()] for _ in range(3 * CONFIRM)])           # last seen on frame 149 (5.96 s)
    n = _feed(mod, [[] for _ in range(MISS + 1)], start=n)             # dropped by the tracker on frame 175 (7.0 s)
    assert len(_alerts(emitted)) == 1
    n = _feed(mod, [[] for _ in range(int(CLEAR * FPS) - 1)], start=n)  # 4.96 s of zero spots: still open
    assert len(_alerts(emitted)) == 1
    _feed(mod, [[]], start=n)                                           # 5 s: cleared
    raised, cleared = _alerts(emitted)
    assert (cleared["kind"], cleared["severity"], cleared["region"], cleared["reason"], cleared["count"]) == \
        ("shelf_restocked", "info", "frame", "gone", "0")
    assert cleared["alert_id"] == raised["alert_id"]
    assert cleared["empty_for_s"] == "10.0" and cleared["message"].endswith("(empty for 10 s)")
    assert cleared["clear_s"] == "5"


def test_more_spots_reraise_the_same_alert_and_a_new_situation_gets_a_new_id():
    mod, emitted, clips = _module()
    n = _feed(mod, [[_void()] for _ in range(CONFIRM)])
    n = _feed(mod, [[_void(), _void(left=600)] for _ in range(CONFIRM)], start=n)
    first, more = _alerts(emitted)
    assert (more["reason"], more["count"]) == ("more", "2") and more["alert_id"] == first["alert_id"]
    assert len(json.loads(more["spots"])) == 2 and len(clips) == 1        # one clip per situation
    n = _feed(mod, [[] for _ in range(MISS + 1 + int(CLEAR * FPS))], start=n)
    _feed(mod, [[_void()] for _ in range(CONFIRM)], start=n)
    *_, cleared, again = _alerts(emitted)
    assert cleared["kind"] == "shelf_restocked" and again["reason"] == "new"
    assert again["alert_id"] != first["alert_id"] and len(clips) == 2


def test_an_open_region_is_raised_again_every_reraise_s():
    mod, emitted, _ = _module(situation={"clear_s": CLEAR, "reraise_s": 4.0})
    _feed(mod, [[_void()] for _ in range(int(11 * FPS))])
    reasons = [(a["reason"], a["pts"]) for a in _alerts(emitted)]
    assert reasons == [("new", "1.960"), ("periodic", "5.960"), ("periodic", "9.960")]
    assert len({a["alert_id"] for a in _alerts(emitted)}) == 1


def test_an_unconfirmed_candidate_dropped_is_silent():
    mod, emitted, _ = _module()
    n = _feed(mod, [[_void()] for _ in range(20)])
    _feed(mod, [[] for _ in range(3 * MISS)], start=n)
    assert _alerts(emitted) == []


def test_a_single_roi_polygon_is_one_region_named_roi_and_boxes_outside_are_ignored():
    mod, emitted, _ = _module(roi={"cam-1": [[0, 0], [300, 0], [300, 720], [0, 720]]})
    _feed(mod, [[_void(), _void(left=600)] for _ in range(CONFIRM)])
    (a,) = _alerts(emitted)
    assert a["region"] == "roi" and a["bbox"].startswith("100,") and a["count"] == "1"


def test_without_an_fps_setting_the_rate_is_measured_from_the_pts_first():
    mod, emitted, _ = _module(tracker={})
    # frames 0-10 give the 10 intervals; frame 10 completes the measurement and is the tracker's first frame
    _feed(mod, [[_void()] for _ in range(10 + CONFIRM - 1)])
    assert _alerts(emitted) == []
    _feed(mod, [[_void()]], start=10 + CONFIRM - 1)
    assert len(_alerts(emitted)) == 1
    rt = _rt(mod)
    assert rt.fps == pytest.approx(FPS) and rt.trackers[0].interval == pytest.approx(1 / FPS)


def test_fps_probe_skips_a_pts_restart():
    p = FpsProbe(samples=3)
    assert [p.feed(t) for t in (20.0, 20.04, 0.0, 0.04)] == [None, None, None, None]
    assert p.feed(0.08) == pytest.approx(25.0)


def test_confirmed_gaps_are_drawn_and_published_with_their_region_and_alert_id():
    mod, emitted, _ = _module()
    _feed(mod, [[_void()] for _ in range(CONFIRM)])
    view = FrameView("cam-1", None, 0, CONFIRM, 2.0, 2.0, mux_size=MUX, need_meta=True, _objects=[])
    out = FrameOutput()
    mod.analyze(view, out)
    (f,) = out.fields["shelf_void"]
    assert f["bbox"] == [100, 100, 180, 140] and f["region"] == "frame"
    assert f["alert_id"] == _alerts(emitted)[0]["alert_id"]
    img = np.zeros((720, 1280, 3), np.uint8)
    (draw,) = out.draws
    draw(img, 1.0, 1.0)
    assert (img == np.array(BOX_VOID, np.uint8)).all(axis=2).any()


def test_region_trackers_drop_candidates_of_a_paused_region_and_keep_its_confirmed_spots():
    from datetime import datetime, timezone
    from perception.modules.shelf_void.stability import PersonConfig
    wall = datetime(2026, 11, 12, tzinfo=timezone.utc)
    rt = RegionTrackers([("a", [(0, 0), (500, 0), (500, 500), (0, 500)])], TrackerConfig(), FPS, (500, 500),
                        PersonConfig())
    for i in range(CONFIRM):
        rt.step([(10, 10, 50, 50, 0.2)], [], i / FPS, wall)
    rt.step([(300, 300, 340, 340, 0.2)], [(100, 100, 200, 200)], CONFIRM / FPS, wall)   # paused
    assert [t["box"] for _, _, t in rt.confirmed()] == [(10, 10, 50, 50)] and len(rt.trackers[0].tracks) == 1


# --- named regions (the demo's roi.json), scaled to the mux; one situation each ---

# ~/sarah/shelf-demo/roi.json, drawn on a 1920x1080 frame of the demo camera
DEMO_ROI = [[[1369, 328], [1684, 352], [1581, 931], [1294, 933], [1357, 327]],
            [[225, 346], [516, 358], [594, 835], [544, 928], [276, 880], [150, 451], [222, 343]]]


def _at(x, y, w=60, h=40, size=(1920, 1080)):
    """A void box centred on (x, y) of a `size` frame, in mux pixels."""
    sx, sy = MUX[0] / size[0], MUX[1] / size[1]
    return _void(left=x * sx - w / 2, top=y * sy - h / 2, width=w, height=h)


def test_roi_json_regions_are_named_like_the_demo_and_scaled_to_the_mux():
    s = build_settings(Settings, {"stability": "tracker", "roi": {"cam-1": {"size": [1920, 1080],
                                                                              "regions": DEMO_ROI}}}, "shelf_void")
    (n1, p1), (n2, p2) = s.regions("cam-1", MUX)
    assert (n1, n2) == ("R1", "R2")
    assert p1[0] == pytest.approx((1369 * 2 / 3, 328 * 2 / 3)) and p2[-1] == pytest.approx((222 * 2 / 3, 343 * 2 / 3))


def test_each_region_has_its_own_situation_and_voids_outside_every_region_are_ignored():
    mod, emitted, _ = _module(roi={"cam-1": {"size": [1920, 1080], "regions": DEMO_ROI}})
    _feed(mod, [[_at(1480, 600), _at(380, 600), _at(960, 600)] for _ in range(CONFIRM)])   # R1, R2, the aisle
    alerts = _alerts(emitted)
    assert sorted(a["region"] for a in alerts) == ["R1", "R2"]
    assert len({a["alert_id"] for a in alerts}) == 2 and all(a["count"] == "1" for a in alerts)
    assert all(a["message"].startswith(f"Empty shelf space on cam-1 in region {a['region']}: 1 spot at (")
               for a in alerts)


def test_one_region_clearing_leaves_the_others_situation_open():
    mod, emitted, _ = _module(roi={"cam-1": {"size": [1920, 1080], "regions": DEMO_ROI}})
    n = _feed(mod, [[_at(1480, 600), _at(380, 600)] for _ in range(CONFIRM)])
    _feed(mod, [[_at(380, 600)] for _ in range(MISS + 1 + int(CLEAR * FPS) + 1)], start=n)   # R1's spot gone
    assert [(a["kind"], a["region"]) for a in _alerts(emitted)] == \
        [("shelf_empty", "R1"), ("shelf_empty", "R2"), ("shelf_restocked", "R1")]


def test_named_regions_as_a_mapping_keep_their_names():
    sq = lambda x: [[x, 0], [x + 400, 0], [x + 400, 720], [x, 720]]   # noqa: E731
    mod, emitted, _ = _module(roi={"cam-1": {"size": list(MUX), "regions": {"Rice": sq(0), "Water": sq(600)}}})
    _feed(mod, [[_void(left=100), _void(left=700)] for _ in range(CONFIRM)])
    assert sorted(a["region"] for a in _alerts(emitted)) == ["Rice", "Water"]


def test_a_void_in_overlapping_regions_belongs_to_the_first_only():
    square = [[0, 0], [1000, 0], [1000, 1000], [0, 1000]]
    mod, emitted, _ = _module(roi={"cam-1": {"size": [1000, 1000], "regions": {"top": square, "all": square}}})
    _feed(mod, [[_at(500, 500, size=(1000, 1000))] for _ in range(CONFIRM)])
    (a,) = _alerts(emitted)
    assert a["region"] == "top"
    rt = _rt(mod)
    assert rt.names == ["top", "all"] and rt.trackers[1].tracks == []


def test_the_centre_test_uses_the_scaled_polygon_unrounded():
    # 3x3 source pixels scaled by 2/3 -> 2x2 mux pixels; a centre at 1.9 is inside only if the corner stays 2.0
    s = build_settings(Settings, {"stability": "tracker", "roi": {"cam-1": {"size": [3, 3], "regions": [
        [[0, 0], [3, 0], [3, 3], [0, 3]]]}}}, "shelf_void")
    rt = RegionTrackers(s.regions("cam-1", (2, 2)), TrackerConfig(), FPS, (2, 2))
    assert rt.owner((1.8, 1.8, 2.0, 2.0)) == 0


@pytest.mark.parametrize("roi", [
    {"size": [1920, 1080]},                                            # no regions
    {"size": [1920, 0], "regions": DEMO_ROI},                          # bad size
    {"size": [1920, 1080], "regions": []},                             # empty
    {"size": [1920, 1080], "regions": [[[0, 0], [1, 1]]]},             # a 2-point polygon
    {"size": [1920, 1080], "regions": {"a": [[0, 0], [1, "x"], [2, 2]]}},
    {"size": [1920, 1080], "regions": DEMO_ROI, "names": ["a", "b"]},  # unknown key
])
def test_invalid_regions_refuse_startup(roi):
    with pytest.raises(ModuleConfigError):
        build_settings(Settings, {"stability": "tracker", "roi": {"cam-1": roi}}, "shelf_void")


def test_named_regions_need_tracker_mode():
    with pytest.raises(ModuleConfigError, match="need stability: tracker"):
        build_settings(Settings, {"roi": {"cam-1": {"size": [1920, 1080], "regions": DEMO_ROI}}}, "shelf_void")


# --- person pause (the shared pgie's persons) ---

def _person(left, top, width=100, height=300, conf=0.6, oid=7):
    return ObjectView(oid, "person", conf, conf, left, top, width, height, gie_name="pgie")


TWO = {"cam-1": {"size": [1280, 720], "regions": {"left": [[0, 0], [600, 0], [600, 720], [0, 720]],
                                                   "right": [[700, 0], [1280, 0], [1280, 720], [700, 720]]}}}


def test_person_boxes_are_the_pgies_persons_grown_15_percent_and_clamped():
    from perception.modules.shelf_void.adapter import person_boxes
    objs = [_person(100, 100), _person(0, 0, conf=0.6), _person(500, 100, conf=0.3), _person(900, 100, conf=-0.5),
            _void(), ObjectView(9, "person", 0.9, 0.9, 1, 1, 10, 10, gie_name="pgie_ppe")]
    assert person_boxes(objs, MUX) == [(85, 55, 215, 445), (0, 0, 115, 345)]


def test_defaults_are_the_demos():
    s = build_settings(Settings, {"stability": "tracker"}, "shelf_void")
    p = s.person_config()
    assert (p.enabled, p.conf, p.pad, p.resume_s) == (True, 0.35, 0.15, 0.4)
    t = s.tracker_config()
    assert (t.conf, t.confirm_s, t.max_miss_s, t.min_iou, t.min_hit_ratio, t.fps) == (0.08, 2.0, 1.0, 0.3, 0.5, 0.0)
    c = s.situation_config()
    assert (c.clear_s, c.reraise_s) == (30.0, 300.0)                    # the demo catalog's clearSeconds, RERAISE


def test_a_person_in_front_pauses_only_the_region_they_touch():
    mod, emitted, _ = _module(roi=TWO)
    # a person in "left" the whole time: its void is never learned; "right" alerts as usual
    _feed(mod, [[_person(200, 100), _void(left=300), _void(left=900)] for _ in range(2 * CONFIRM)])
    (a,) = _alerts(emitted)
    assert a["region"] == "right"
    rt = _rt(mod)
    assert rt.trackers[0].tracks == [] and rt.paused_samples == 2 * CONFIRM


def test_the_pad_reaches_a_region_the_raw_box_does_not():
    mod, _, _ = _module(roi=TWO)
    # raw box ends at x 680 (< 700); grown by 15 px per side it reaches 695 -> still outside; at 690 + 15 = 705 inside
    _feed(mod, [[_person(580, 100)]])
    rt = _rt(mod)
    assert rt.active == [False, True]
    _feed(mod, [[_person(590, 100)]], start=1)
    assert rt.active == [False, False]


def test_a_region_resumes_after_resume_s_person_free_and_holds_its_confirmed_spot_meanwhile():
    mod, emitted, _ = _module(roi=TWO)
    n = _feed(mod, [[_void(left=300)] for _ in range(CONFIRM)])            # confirmed in "left"
    n = _feed(mod, [[_person(200, 100)] for _ in range(int(2 * CLEAR * FPS))], start=n)  # 10 s in front, void unseen
    assert len(_alerts(emitted)) == 1                                       # held, and person time never clears
    rt = _rt(mod)
    n = _feed(mod, [[] for _ in range(9)], start=n)                         # 0.36 s person-free: still paused
    assert rt.active[0] is False and len(rt.confirmed()) == 1
    n = _feed(mod, [[]], start=n)                                           # 0.4 s: active again
    assert rt.active[0] is True
    _feed(mod, [[] for _ in range(MISS + 1 + int(CLEAR * FPS) + 1)], start=n)   # now it can expire and clear
    assert [a["kind"] for a in _alerts(emitted)] == ["shelf_empty", "shelf_restocked"]


def test_a_candidate_made_while_a_person_stood_there_is_dropped():
    mod, emitted, _ = _module(roi=TWO)
    n = _feed(mod, [[_void(left=300)] for _ in range(CONFIRM // 2)])       # a candidate (e.g. a head)
    _feed(mod, [[_person(200, 100), _void(left=300)]], start=n)             # the person seen: candidate dropped
    assert _rt(mod).trackers[0].tracks == []


def test_person_pause_can_be_switched_off():
    mod, emitted, _ = _module(roi=TWO, person={"enabled": False})
    _feed(mod, [[_person(200, 100), _void(left=300)] for _ in range(CONFIRM)])
    assert [a["region"] for a in _alerts(emitted)] == ["left"]


def test_the_gate_masks_are_the_demo_services_rounded_polygons():
    import cv2
    s = build_settings(Settings, {"stability": "tracker", "roi": {"cam-1": {"size": [1920, 1080],
                                                                              "regions": DEMO_ROI}}}, "shelf_void")
    rt = RegionTrackers(s.regions("cam-1", (1920, 1080)), TrackerConfig(), FPS, (1920, 1080))
    for mask, poly in zip(rt.gate.masks, DEMO_ROI):
        demo = np.zeros((1080, 1920), np.uint8)
        cv2.fillPoly(demo, [np.array(poly, np.int32)], 1)                   # void_roi.py region_masks
        assert (mask == demo).all()


@pytest.mark.parametrize("key,bad", [("person", {"conf": 2}), ("person", {"pad": -0.1}),
                                     ("person", {"resume_s": "x"}), ("person", {"enabled": "yes"}),
                                     ("person", {"resume_frames": 10}), ("tracker", {"conf": 1.5}),
                                     ("tracker", {"confirm": 2}), ("situation", {"clear_s": -1}),
                                     ("situation", {"reraise_s": 0}), ("situation", {"clear": 3}),
                                     ("situation", [])])
def test_invalid_tracker_person_situation_settings_refuse_startup(key, bad):
    with pytest.raises(ModuleConfigError):
        build_settings(Settings, {"stability": "tracker", key: bad}, "shelf_void")


# --- the vendored files ---

SHELF_DEMO = pathlib.Path.home() / "sarah/shelf-demo"


@pytest.mark.parametrize("name", ["tracker", "person_gate", "situations"])
def test_vendored_files_name_their_source_commit_and_are_unchanged_below_the_header(name):
    import re
    import subprocess
    first, second, body = (REPO / f"perception/modules/shelf_void/core/{name}.py").read_text().split("\n", 2)
    m = re.match(r"# Vendored from shelf-demo .*commit ([0-9a-f]{40}),$", first)
    assert m and second.startswith(f"# service/{name}.py. Copied unchanged")
    if not (SHELF_DEMO / ".git").exists():
        pytest.skip("no shelf-demo checkout to compare with")
    src = subprocess.run(["git", "-C", str(SHELF_DEMO), "show", f"{m.group(1)}:service/{name}.py"],
                         capture_output=True, text=True)
    if src.returncode:
        pytest.skip(f"shelf-demo checkout lacks commit {m.group(1)}")
    assert body == src.stdout
