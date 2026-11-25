"""Overlay entries (iris:overlay:<camera>) placed on a clip, drawn, and handed to ffmpeg."""
import json

import cv2
import numpy as np
import pytest

from app import overlay_track as ot


def _entry(pts, label="0.42", p1=(100, 100), p2=(300, 100)):
    return {"pts": pts, "w": 1280, "h": 720,
            "shapes": [{"type": "line", "p1": list(p1), "p2": list(p2), "label": label, "style": "latched"},
                       {"type": "box", "rect": [10, 10, 50, 100], "style": "person", "id": 6}]}


class _Redis:
    def __init__(self, entries):
        self.entries = entries

    def xrange(self, key, lo, hi):
        return self.entries.get(key, [])


def _raw(pts, shapes='[]'):
    return (b"1-0", {b"v": b"1", b"module": b"proximity", b"pts": str(pts).encode(), b"w": b"1280", b"h": b"720",
                     b"shapes": shapes.encode()})


def test_each_entry_shows_until_the_next_or_the_hold_and_blanks_fill_the_rest():
    entries = [_entry(51.0), _entry(51.15), _entry(53.0)]
    # the clip's first frame is at frame time 45 and it lasts 10 s
    segs = ot.timeline(entries, clip_start_pts=45.0, clip_len=10.0)
    assert [(round(s, 2), round(d, 2), e is not None) for s, d, e in segs] == [
        (0.0, 6.0, False), (6.0, 0.15, True), (6.15, 0.5, True), (6.65, 1.35, False), (8.0, 0.5, True),
        (8.5, 1.5, False)]


def test_entries_outside_the_clip_are_dropped():
    segs = ot.timeline([_entry(30.0), _entry(80.0)], 45.0, 10.0)
    assert segs == [(0.0, 10.0, None)]


def test_entries_from_another_timeline_are_ignored():
    redis = _Redis({"iris:overlay:cam0": [_raw(50.0), _raw(500.0), _raw("bad"), (b"2-0", {b"pts": b"51"})]})
    got = ot.read_entries(redis, "cam0", around_pts=50.0, reach=10.0)
    assert [e["pts"] for e in got] == [50.0]


def test_lines_and_boxes_are_drawn_scaled_to_the_clip():
    img = ot.draw(np.zeros((360, 640, 4), np.uint8), _entry(0.0))
    # (100,100)-(300,100) at 1280x720 is row 50, columns 50..150 at 640x360: opaque red there
    assert img[50, 100, 3] > 200 and img[50, 100, 2] > 200
    # the person's box (10,10,50,100) is (5,5)-(30,55) at 640x360: its top edge, row 5, in green
    assert tuple(img[5, 15, :3]) == ot.PERSON_BGR


def test_a_sequence_lists_each_image_with_its_duration_and_repeats_the_last(tmp_path):
    segs = [(0.0, 1.0, None), (1.0, 0.5, _entry(0.0)), (1.5, 0.5, None)]
    path = ot.write_sequence(segs, 64, 36, str(tmp_path))
    lines = open(path).read().splitlines()
    assert lines[0] == "ffconcat version 1.0"
    assert [l for l in lines if l.startswith("duration")] == ["duration 1.000", "duration 0.500", "duration 0.500"]
    assert lines[-1] == lines[-3]                                      # the last file twice
    assert ot.write_sequence([(0.0, 2.0, None)], 64, 36, str(tmp_path)) is None


def test_one_malformed_shape_costs_only_itself():
    shapes = [{"type": "line", "p1": [100, 100], "label": "x", "style": "latched"},          # no p2
              {"type": "box", "rect": [1, 2], "style": "person", "id": 6},                      # a short rect
              {"type": "box", "rect": [1, 2, 3, 4], "style": "person", "id": 7, "mask": [[1, 2], "x"]},
              {"type": "line", "p1": [100, 100], "p2": [300, 100], "label": "0.42", "style": "latched"}]
    redis = _Redis({"iris:overlay:cam0": [_raw(50.0, json.dumps(shapes))]})
    (entry,) = ot.read_entries(redis, "cam0", around_pts=50.0, reach=1.0)
    assert entry["shapes"] == [shapes[3]]
    img = ot.draw(np.zeros((360, 640, 4), np.uint8), {**entry, "shapes": shapes})          # drawing never raises
    assert img[50, 100, 2] > 200


def _pairs_entry():
    """Person 6 far from truck 11, person 7 close to it, each with a mask."""
    square = lambda x, y: [[x, y], [x + 40, y], [x + 40, y + 40], [x, y + 40]]
    return {"pts": 0.0, "w": 1280, "h": 720, "shapes": [
        {"type": "line", "p1": [100, 600], "p2": [500, 500], "label": "3.10", "style": "far", "person": 6,
         "vehicle": 11},
        {"type": "line", "p1": [700, 600], "p2": [650, 500], "label": "0.40", "style": "close", "person": 7,
         "vehicle": 11},
        {"type": "box", "rect": [80, 500, 40, 100], "style": "person", "id": 6, "label": "person 6",
         "mask": square(80, 520)},
        {"type": "box", "rect": [680, 500, 40, 100], "style": "person", "id": 7, "label": "person 7",
         "mask": square(680, 520)},
        {"type": "box", "rect": [400, 200, 300, 300], "style": "vehicle", "id": 11, "label": "truck 11",
         "state": "moving", "mask": square(450, 300)}]}


def _kept(entry):
    return sorted((s["type"], s.get("person", s.get("id")), s.get("vehicle", s.get("style"))) for s in entry["shapes"])


def test_a_clip_draws_only_its_incidents_pair_close_or_not():
    got = ot.select(_pairs_entry(), ({6}, {11}))
    assert _kept(got) == [("box", 6, "person"), ("box", 11, "vehicle"), ("line", 6, 11)]
    assert ot.select(_pairs_entry(), ({99}, {98})) is None


def test_without_a_pair_only_the_close_lines_and_their_objects_are_drawn():
    got = ot.select(_pairs_entry(), None)
    assert _kept(got) == [("box", 7, "person"), ("box", 11, "vehicle"), ("line", 7, 11)]


def test_an_incidents_pair_comes_from_its_alert_details():
    assert ot.incident_pair({"person_ids": "6,8", "vehicle_ids": "11"}) == ({6, 8}, {11})
    assert ot.incident_pair({"person_ids": "x, 6"}) == ({6}, set())
    assert ot.incident_pair({"gap": "0.4"}) is None and ot.incident_pair(None) is None
    assert ot.merge_pairs(({6}, {11}), ({7}, {12})) == ({6, 7}, {11, 12})
    assert ot.merge_pairs(None, ({7}, {12})) == ({7}, {12}) and ot.merge_pairs(None, None) is None


def test_a_clip_draws_masks_without_boxes_by_default():
    entry = ot.select(_pairs_entry(), ({6}, {11}))
    clip = ot.draw(np.zeros((720, 1280, 4), np.uint8), entry)
    inside = clip[540, 100]                                   # inside person 6's mask, off its outline
    assert tuple(inside[:3]) == ot.PERSON_BGR and inside[3] == round(ot.MASK_ALPHA * 255)
    assert clip[200, 550, 3] == 0                             # the truck's box (top edge, row 200) is not drawn
    assert clip[300 - 30:300, 450:560, 3].max() == 255        # its label, just above its mask (top at row 300)
    shot = ot.draw(np.full((720, 1280, 3), 200, np.uint8), entry)                            # a snapshot: blended
    expected = np.array(ot.PERSON_BGR) * ot.MASK_ALPHA + 200 * (1 - ot.MASK_ALPHA)
    assert np.allclose(shot[540, 100], expected, atol=2)


def test_an_object_without_a_mask_gets_its_box_in_a_masks_clip():
    entry = ot.select(_pairs_entry(), ({6}, {11}))
    for s in entry["shapes"]:
        s.pop("mask", None) if s.get("id") == 6 else None
    clip = ot.draw(np.zeros((720, 1280, 4), np.uint8), entry)
    assert tuple(clip[500, 90, :3]) == ot.PERSON_BGR and clip[500, 90, 3] == 255   # person 6's box, top edge


def test_a_boxes_clip_draws_boxes_in_the_vehicles_state_colour_and_no_masks():
    entry = ot.select(_pairs_entry(), ({6}, {11}))
    clip = ot.draw(np.zeros((720, 1280, 4), np.uint8), entry, style="boxes")
    assert tuple(clip[200, 550, :3]) == ot.VEHICLE_BGR["moving"] and clip[200, 550, 3] == 255   # the box's top edge
    assert clip[540, 100, 3] == 0                             # no mask fill


class _Setting:
    def __init__(self, raw):
        self.raw = raw

    def get(self, key):
        assert key == ot.PREVIEW_KEY
        if isinstance(self.raw, Exception):
            raise self.raw
        return self.raw


@pytest.mark.parametrize("raw, style", [
    (json.dumps({"clip_style": "boxes"}), "boxes"), (json.dumps({"clip_style": "masks"}), "masks"),
    (None, "masks"), (json.dumps({"clip_style": "both"}), "masks"), ("{not json", "masks"), (b'[1]', "masks"),
    (ConnectionError("down"), "masks")])
def test_the_clip_style_is_the_ui_setting_else_masks(raw, style):
    assert ot.clip_style(_Setting(raw)) == style


def test_the_clips_colours_are_the_live_previews():
    from perception.modules import proximity as px
    assert (ot.PERSON_BGR, ot.VEHICLE_BGR, ot.MASK_ALPHA) == (px.PERSON_COLOR, px.VEHICLE_COLORS, px.MASK_ALPHA)
    assert ot.LINE_BGR == {"latched": px.LINE_LATCHED, "close": px.LINE_CLOSE, "far": px.LINE_FAR}
    assert ot.LINE_SHADOW == px.LINE_SHADOW


def test_a_box_at_the_top_of_the_frame_keeps_its_label_inside_it():
    entry = {"pts": 0.0, "w": 640, "h": 360, "shapes": [
        {"type": "box", "rect": [100, 0, 300, 200], "style": "vehicle", "id": 9, "label": "truck 9"}]}
    img = ot.draw(np.zeros((360, 640, 3), np.uint8), entry)
    assert img[4:20, 110:200].max() > 0                         # the label, under the top edge


def _clip_line(style):
    """A horizontal line drawn on a light grey 1280x720 clip frame: the column across it, and its rows in the line's
    colour."""
    entry = {"pts": 0.0, "w": 1280, "h": 720, "shapes": [
        {"type": "line", "p1": [100, 300], "p2": [600, 300], "label": "1.20", "style": style}]}
    col = ot.draw(np.full((720, 1280, 3), 200, np.uint8), entry)[:, 200].astype(int)
    return col, [r for r in range(720) if np.abs(col[r] - np.array(ot.LINE_BGR[style])).max() <= 30]


def test_a_clips_far_line_is_a_full_stroke_close_ones_thicker_over_a_dark_outline():
    thick = max(round(1280 / 400), 2)
    _col, far = _clip_line("far")
    col, close = _clip_line("close")
    assert len(far) >= thick and len(close) > len(far)
    assert col[min(close) - 1].max() < 150 and col[max(close) + 1].max() < 150


def _clip_with(line):
    entry = {"pts": 0.0, "w": 1280, "h": 720, "shapes": [
        {"type": "line", "p1": [300, 400], "p2": [600, 400], "label": "0.50", "style": "close", **line}]}
    return ot.draw(np.full((720, 1280, 3), 128, np.uint8), entry)


def _near(px, color, tol=40):
    return bool(np.abs(px.astype(int) - np.array(color)).max() <= tol)


def test_a_clip_rings_a_hidden_foot_with_a_dashed_leg_to_the_person():
    img = _clip_with({"hidden": True, "seen": [300, 320]})
    c = ot.LINE_BGR["close"]
    left = [_near(img[400, x], c) for x in range(285, 296)]          # the ring's left side
    assert any(left) and not _near(img[400, 300], c)                 # hollow
    leg = [_near(img[y, 300], c) for y in range(322, 380)]
    assert 0.3 < sum(leg) / len(leg) < 0.9


@pytest.mark.parametrize("depth, key", [({"state": "same", "gap": 0.05}, "same"),
                                        ({"state": "apart", "gap": 0.4}, "apart"),
                                        ({"state": "timeout", "gap": None}, "wait")])
def test_a_clip_shows_the_depth_verdict_after_the_gap(depth, key):
    region = _clip_with({"depth": depth})[360:420, 470:760]
    assert any(_near(p, ot.DEPTH_BGR[key], 30) for p in region.reshape(-1, 3))


def test_the_clips_depth_colours_are_the_live_previews():
    from perception.modules import proximity as px
    assert ot.DEPTH_BGR == {"same": px.DEPTH_SAME, "apart": px.DEPTH_APART, "wait": px.DEPTH_WAIT}


# ── the other features' boxes: the alert's own object (style "alert"), fire and smoke (perception/lib/overlay_boxes) ──

def _boxes(*shapes):
    return {"pts": 1.0, "w": 1280, "h": 720, "shapes": list(shapes)}


ALERT_7 = {"type": "box", "rect": [10, 10, 50, 90], "style": "alert", "id": 7, "label": "no helmet"}
ALERT_9 = {"type": "box", "rect": [300, 10, 50, 90], "style": "alert", "id": 9}
FIRE = {"type": "box", "rect": [600, 400, 40, 60], "style": "fire", "label": "fire 0.82"}


def test_an_alert_with_a_subject_draws_only_its_object():
    assert ot.select(_boxes(ALERT_7, ALERT_9), ({7}, set()))["shapes"] == [ALERT_7]
    assert ot.select(_boxes(ALERT_9), ({7}, set())) is None


def test_an_alert_naming_no_one_draws_every_object_it_could_be_about():
    assert ot.select(_boxes(ALERT_7, ALERT_9), None)["shapes"] == [ALERT_7, ALERT_9]      # overcrowding: everyone counted


def test_fire_and_smoke_are_always_drawn():
    assert ot.select(_boxes(FIRE), None)["shapes"] == [FIRE]
    assert ot.select(_boxes(FIRE, ALERT_9), ({7}, set()))["shapes"] == [FIRE]


def test_the_subject_of_a_ppe_or_intrusion_alert_comes_from_its_track_or_object_id():
    assert ot.incident_pair({"track_id": "7"}) == ({7}, set())
    assert ot.incident_pair({"object_id": "12"}) == ({12}, set())
    assert ot.incident_pair({"zone": "z1"}) is None


def test_the_alert_s_object_is_drawn_in_the_alert_colour_and_fire_in_its_own():
    img = ot.draw(np.zeros((720, 1280, 3), np.uint8), _boxes(ALERT_7, FIRE), "boxes")
    assert tuple(img[10, 30]) == ot.ALERT_BGR                    # the subject's top edge
    assert tuple(img[400, 620]) == ot.HAZARD_BGR["fire"]


def test_an_entry_s_wall_time_is_its_frame_s_arrival_when_it_carries_one():
    import fakeredis
    r = fakeredis.FakeStrictRedis()
    shapes = json.dumps([{"type": "box", "rect": [0, 0, 1, 1], "style": "fire"}])
    r.xadd("iris:overlay:cam0", {"pts": "1.0", "w": "10", "h": "10", "shapes": shapes, "at": "99.500"}, id="100000-0")
    r.xadd("iris:overlay:cam0", {"pts": "2.0", "w": "10", "h": "10", "shapes": shapes}, id="101000-0")
    assert [e["wall"] for e in ot.read_entries(r, "cam0", None, None)] == [99.5, 101.0]
