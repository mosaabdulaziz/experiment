"""perception/lib/overlay_boxes.py: what PPE, pose, fire and zones write so their alerts' clips show the object the
alert is about (app/overlay_track.py draws it)."""
import json
import types

from perception.engine.modules import EngineServices
from perception.lib.overlay_boxes import OverlayWriter, box, corners_box


def _services(emitted, overlay_boxes=True):
    st = types.SimpleNamespace(module_state={}, zones=[])
    return EngineServices(
        redis=None, redis_factory=lambda: None, emit=lambda key, fields, maxlen=64: emitted.append((key, fields)),
        request_clip=lambda *a, **k: None, stream_ids=["cam-1"], timer=lambda *a: None, telemetry=None,
        batch_size=1, states={"cam-1": st}, mux_size=(1280, 720), config={"overlay_boxes": overlay_boxes})


def _entries(emitted):
    return [(k, json.loads(f["shapes"]), float(f["pts"])) for k, f in emitted if k.startswith("iris:overlay:")]


def test_a_box_is_left_top_width_height_with_its_style_id_and_label():
    assert box((10.04, 20, 30, 40), "alert", 7, "no helmet") == {
        "type": "box", "rect": [10.0, 20.0, 30.0, 40.0], "style": "alert", "id": 7, "label": "no helmet"}
    assert box((0, 0, 5, 5), "fire") == {"type": "box", "rect": [0.0, 0.0, 5.0, 5.0], "style": "fire"}
    assert corners_box((10, 20, 40, 60), "alert", 3)["rect"] == [10.0, 20.0, 30.0, 40.0]


def test_it_writes_at_most_every_tenth_of_a_second_per_stream():
    emitted = []
    w = OverlayWriter(_services(emitted), "ppe")
    shapes = [box((0, 0, 1, 1), "alert", 1)]
    assert w.write("cam-1", 10.00, shapes) and not w.write("cam-1", 10.05, shapes) and w.write("cam-1", 10.10, shapes)
    ((key, written, pts), _) = _entries(emitted)
    assert key == "iris:overlay:cam-1" and written == shapes and pts == 10.0
    assert dict(emitted[0][1])["w"] == "1280" and dict(emitted[0][1])["module"] == "ppe"


def test_nothing_to_draw_writes_nothing_and_a_restarted_clock_writes_again():
    emitted = []
    w = OverlayWriter(_services(emitted), "fire_alerts")
    assert not w.write("cam-1", 5.0, [])
    assert w.write("cam-1", 50.0, [box((0, 0, 1, 1), "fire")])
    assert w.write("cam-1", 0.4, [box((0, 0, 1, 1), "fire")])          # a looping file starts its frame times over
    assert len(_entries(emitted)) == 2


def test_an_entry_carries_its_frame_s_arrival_time_so_a_clip_places_it_without_the_inference_delay():
    emitted = []
    w = OverlayWriter(_services(emitted), "pose")
    w.write("cam-1", 3.0, [box((0, 0, 1, 1), "alert", 4)], at=1794240000.25)
    assert emitted[0][1]["at"] == "1794240000.250"
    w.write("cam-1", 9.0, [box((0, 0, 1, 1), "alert", 4)])           # unknown arrival: none written
    assert "at" not in emitted[1][1]


def test_only_a_pipeline_that_asks_for_them_writes_the_boxes():
    """`overlay_boxes: true` (the safety stack's roles, products/safety/safety.yaml): every other product's pipeline
    writes no new overlay stream, so its Redis and its clips stay exactly as they were."""
    emitted = []
    w = OverlayWriter(_services(emitted, overlay_boxes=False), "ppe")
    assert not w.write("cam-1", 1.0, [box((0, 0, 1, 1), "alert", 1)]) and emitted == []

