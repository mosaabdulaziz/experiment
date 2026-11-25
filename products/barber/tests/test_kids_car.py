import json

from kids_car import KIDS, Detection, KidsCar, Place, merge, zones_for
from policy import DEFAULT_POLICY

P = {**DEFAULT_POLICY, "kc_enabled": 1.0}
BOX = (0.40, 0.40, 0.50, 0.50)


def det(cam, box=BOX, conf=0.9, ts=100.0):
    return Detection(cam, box, conf, ts)


def test_zones_for_a_place_are_an_unpaired_kids_chair_and_its_seat():
    ch, seat = zones_for(Place("ph-1", BOX, "auto", 0.9, 0.0))
    assert ch["id"] == KIDS and ch["name"] == KIDS and ch["type"] == "chair" and "pair" not in ch
    assert seat["seat_of"] == KIDS and seat["type"] == "seat"
    xs = [p[0] for p in ch["points"]]
    assert abs(min(xs) - 0.3925) < 1e-4 and abs(max(xs) - 0.5075) < 1e-4     # x 1.15 about the centre
    sx, sy = [p[0] for p in seat["points"]], [p[1] for p in seat["points"]]
    # 2026-10-18 (first real kids haircut, ph-1): the middle half of the width, 15-55 % of the height from the top --
    # the child's hips sit above the box centre (the box holds the car's nose, wheels and base under it)
    assert abs(min(sx) - 0.425) < 1e-4 and abs(max(sx) - 0.475) < 1e-4
    assert abs(min(sy) - 0.415) < 1e-4 and abs(max(sy) - 0.455) < 1e-4


def test_merge_replaces_the_stored_kids_chair_and_keeps_the_rest():
    stored = {"ph-2": [{"id": "ph2-Kids", "name": "Kids", "type": "chair", "points": []},
                       {"id": "ph2-Kids-seat", "name": "Kids", "type": "seat", "points": [], "seat_of": "ph2-Kids"},
                       {"id": "ph2-L9", "name": "L9", "type": "chair", "points": []}], "ph-1": []}
    out = merge(stored, Place("ph-1", BOX, "auto", 0.9, 0.0))
    assert [z["id"] for z in out["ph-2"]] == ["ph2-L9"]
    assert [z["id"] for z in out["ph-1"]] == [KIDS, f"{KIDS}-seat"]
    assert merge(stored, None)["ph-2"] == [stored["ph-2"][2]]                  # lost: no Kids chair
    assert stored["ph-2"][0]["id"] == "ph2-Kids"                               # the input is not changed


def test_a_move_needs_two_in_a_row_and_an_empty_chair():
    k = KidsCar(None, now=0.0)
    assert k.observe({"ph-1": det("ph-1", ts=20)}, 20, False, P) is None
    assert k.observe({"ph-1": det("ph-1", ts=40)}, 40, False, P).camera == "ph-1"
    far = (0.10, 0.70, 0.20, 0.80)
    k.observe({"ph-2": det("ph-2", far, ts=60)}, 60, False, P)
    assert k.observe({"ph-2": det("ph-2", far, ts=80)}, 80, True, P).camera == "ph-1"     # busy: waits
    assert k.observe({"ph-2": det("ph-2", far, ts=100)}, 100, False, P).camera == "ph-2"


def test_a_snapshot_counts_once_however_often_the_consumer_looks():
    """2026-10-16 bug (live): observe runs every consumer cycle with the locator's latest snapshot, so one detection
    counted kc_confirm_n times within a second and a single false hit moved the chair."""
    k = KidsCar(None, now=0.0)
    for now in (20.0, 20.1, 20.2, 25.0, 39.9):
        assert k.observe({"ph-1": det("ph-1", ts=20)}, now, False, P) is None
    assert k.observe({"ph-1": det("ph-1", ts=40)}, 40.0, False, P).camera == "ph-1"


def test_a_manual_pin_does_not_yield_to_one_snapshot_seen_many_times():
    k = KidsCar(None, now=0.0)
    k.pin("ph-3", 0.5, 0.5, (0.1, 0.2), now=0.0)
    far = (0.10, 0.70, 0.20, 0.80)
    for now in range(1000, 1040):                                              # 40 cycles, one snapshot
        assert k.observe({"ph-2": det("ph-2", far, ts=1000)}, float(now), False, P).source == "manual"


def test_the_best_camera_wins_and_edge_or_weak_or_stale_detections_do_not_count():
    k = KidsCar(None, now=0.0)
    small, big = (0.4, 0.4, 0.45, 0.45), (0.3, 0.3, 0.5, 0.5)
    for t in (20, 40):
        p = k.observe({"ph-1": det("ph-1", small, ts=t), "ph-2": det("ph-2", big, ts=t),
                       "ph-3": det("ph-3", (0.0, 0.3, 0.3, 0.5), ts=t),     # touches the edge
                       "ph-4": det("ph-4", big, conf=0.2, ts=t),            # weak
                       "ph-5": det("ph-5", big, ts=t - 100)}, t, False, P)   # stale
    assert p.camera == "ph-2"


def test_unseen_for_kc_lost_s_removes_the_chair():
    k = KidsCar(None, now=0.0)
    k.observe({"ph-1": det("ph-1", ts=20)}, 20, False, P)
    k.observe({"ph-1": det("ph-1", ts=40)}, 40, False, P)
    assert k.observe({"ph-1": None}, 300, False, P) is not None
    assert k.observe({"ph-1": None}, 341, False, P) is None


def test_a_saved_place_survives_a_restart(tmp_path):
    path = tmp_path / "kids_car.json"
    k = KidsCar(path, now=0.0)
    k.observe({"ph-1": det("ph-1", ts=20)}, 20, False, P)
    k.observe({"ph-1": det("ph-1", ts=40)}, 40, False, P)
    assert json.loads(path.read_text())["camera"] == "ph-1"
    k2 = KidsCar(path, now=1000.0)
    assert k2.place.camera == "ph-1"
    assert k2.observe({}, 1200.0, False, P) is not None and k2.observe({}, 1301.0, False, P) is None


def test_a_manual_pin_holds_until_cleared_or_the_car_is_seen_elsewhere_for_kc_release_s():
    k = KidsCar(None, now=0.0)
    k.pin("ph-3", 0.5, 0.5, (0.1, 0.2), now=0.0)
    assert k.place.source == "manual" and k.place.box == (0.45, 0.4, 0.55, 0.6)
    assert k.observe({}, 1000.0, False, P).source == "manual"                  # never lost
    far = (0.10, 0.70, 0.20, 0.80)
    for i, t in enumerate(range(1020, 1140, 20)):                              # 6 sightings x 20 s = 120 s
        p = k.observe({"ph-2": det("ph-2", far, ts=t)}, t, False, P)
        assert p.source == ("manual" if i < 5 else "auto")
    k.clear()
    assert k.place is None


def test_decode_takes_the_best_anchor_back_to_frame_coordinates():
    import numpy as np
    from kids_car import decode
    out = np.zeros((1, 5, 4), np.float32)
    out[0, :, 2] = [208, 208, 41.6, 41.6, 0.9]            # centre of the 416 input, 10 % of it
    out[0, 4, 1] = 0.3
    box, conf = decode(out, scale=416 / 1280, pad=(0.0, (416 - 234) / 2), w=1280, h=720)
    assert abs(conf - 0.9) < 1e-6
    assert abs(box[0] - 0.45) < 0.01 and abs(box[2] - 0.55) < 0.01                 # x 0.5 +- 0.05
    assert abs((box[1] + box[3]) / 2 - 0.5) < 0.01


def test_the_locator_keeps_the_latest_per_camera_and_survives_failures():
    from kids_car import CarLocator
    calls = []

    def fetch(cam):
        calls.append(cam)
        if cam == "ph-2":
            raise OSError("go2rtc down")
        return b"jpeg"

    loc = CarLocator(["ph-1", "ph-2", "ph-3"], fetch, lambda b: ((0.4, 0.4, 0.5, 0.5), 0.8), lambda: 20.0,
                     clock=lambda: 100.0)
    for cam in ("ph-1", "ph-2", "ph-3"):
        loc.step(cam)
    got = loc.latest()
    assert got["ph-1"].conf == 0.8 and got["ph-1"].ts == 100.0 and got["ph-2"] is None and got["ph-3"] is not None
    loc2 = CarLocator(["ph-1"], fetch, lambda b: None, lambda: 20.0, clock=lambda: 5.0)
    loc2.step("ph-1")
    assert loc2.latest()["ph-1"] is None


def test_the_onnx_detector_runs_a_one_class_head_on_a_jpeg():
    """tests/fixtures/kids_car_tiny.onnx: (1, 3, 416, 416) -> a fixed (1, 5, 2) one-class head (best box at the centre)."""
    import io
    from pathlib import Path
    from PIL import Image
    from kids_car import OnnxCarDetector
    buf = io.BytesIO()
    Image.new("RGB", (1280, 720), (90, 90, 90)).save(buf, "JPEG")
    model = Path(__file__).parent / "fixtures" / "kids_car_tiny.onnx"
    box, conf = OnnxCarDetector(str(model)).detect(buf.getvalue())
    assert abs(conf - 0.9) < 1e-6 and abs((box[0] + box[2]) / 2 - 0.5) < 0.01


# ── the consumer (Task 4) ───────────────────────────────────────────────────

def _consumer(tmp_path, kc_enabled):
    import time
    from config import CameraConfig, ServiceConfig
    from consumer import ConsumerThread
    from storage import connect
    from test_consumer import FakeRedis
    from zones import FrameGeometry
    sq = [[0.4, 0.4], [0.5, 0.4], [0.5, 0.5], [0.4, 0.5]]
    zones = {"ph-1": [], "ph-2": [{"id": "ph2-Kids", "name": "Kids", "type": "chair", "points": sq},
                                  {"id": "ph2-Kids-seat", "name": "Kids", "type": "seat", "points": sq, "seat_of": "ph2-Kids"}]}
    (tmp_path / "zones.json").write_text(json.dumps(zones))
    (tmp_path / "policy.json").write_text(json.dumps({"kc_enabled": kc_enabled}))
    c = ServiceConfig(redis_url="redis://unused", frame=FrameGeometry(1280, 720),
                      cameras=(CameraConfig("ph-1"), CameraConfig("ph-2")), data_dir=tmp_path,
                      stream_starts_file=tmp_path / "starts.json", crops_dir=tmp_path / "crops", engine="chair")
    consumer = ConsumerThread(c)
    consumer.attach(connect(str(c.db_path)), FakeRedis())

    class Stub:
        def latest(self):
            return {"ph-1": Detection("ph-1", BOX, 0.9, time.time()), "ph-2": None}
    consumer.locator = Stub()
    return consumer


def test_the_consumer_moves_the_kids_chair_onto_the_car(tmp_path):
    from zones import zone_set_problems
    consumer = _consumer(tmp_path, 1)
    consumer.step()
    consumer.step()
    cams = consumer.engine.camera_of
    assert cams.get(KIDS) == "ph-1" and "ph2-Kids" not in cams
    shown = consumer.shown_zones()
    assert [z["id"] for z in shown["ph-1"]] == [KIDS, f"{KIDS}-seat"] and shown["ph-2"] == []
    assert zone_set_problems(shown) == []
    assert json.loads((tmp_path / "zones.json").read_text())["ph-2"][0]["id"] == "ph2-Kids"   # stored zones untouched


def test_with_the_car_off_the_stored_kids_chair_stays(tmp_path):
    consumer = _consumer(tmp_path, 0)
    consumer.step()
    consumer.step()
    assert consumer.engine.camera_of.get("ph2-Kids") == "ph-2" and KIDS not in consumer.engine.camera_of


def test_the_current_camera_keeps_the_car_while_it_still_sees_it_there():
    """10 Oct GT hour: the parked car seen by ph-1 (big box) and ph-2 (small) flipped the chair six times whenever
    ph-1 missed twice; the owner keeps it while it still sees the car in its place, whatever the others score."""
    k = KidsCar(None, now=0.0)
    small, big = (0.40, 0.40, 0.45, 0.45), (0.30, 0.30, 0.50, 0.50)
    for t in (20, 40):
        k.observe({"ph-2": det("ph-2", small, ts=t)}, t, False, P)
    assert k.place.camera == "ph-2"
    for t in (60, 80, 100):                                   # ph-1 now scores far higher; ph-2 still sees it
        assert k.observe({"ph-1": det("ph-1", big, ts=t), "ph-2": det("ph-2", small, ts=t)}, t, False, P).camera == "ph-2"
    for t in (120, 140):                                      # ph-2 loses it: ph-1 takes over after 2 in a row
        p = k.observe({"ph-1": det("ph-1", big, ts=t), "ph-2": None}, t, False, P)
    assert p.camera == "ph-1"
