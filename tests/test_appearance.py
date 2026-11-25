"""perception/modules/appearance: crops, model inputs, staff probability, codec, the worker."""
import os

import numpy as np

from perception.modules.appearance import worker as ap


def frame(h=100, w=200, value=0):
    return np.full((h, w, 3), value, np.uint8)


def test_crop_clips_to_the_frame():
    c = ap.crop(frame(), (-10, 50, 40, 100))
    assert c.shape == (50, 30, 3)


def test_crop_too_small_is_none():
    assert ap.crop(frame(), (195, 10, 20, 20)) is None


def test_reid_batch_shape_and_mean_is_zero():
    mean_bgr = ap.REID_MEAN[::-1].round().astype(np.uint8)
    crop = np.tile(mean_bgr, (300, 100, 1))
    x = ap.reid_batch([crop, crop])
    assert x.shape == (2, 3, 256, 128) and x.dtype == np.float32
    assert np.abs(x).max() < 0.02


def test_reid_letterbox_pads_a_square_crop_to_one_by_two_with_black_bands():
    spec = ap.ReidSpec("t", "unused.onnx", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), letterbox=True)
    x = ap.reid_batch([np.full((100, 100, 3), 200, np.uint8)], spec)
    assert x.shape == (1, 3, 256, 128)
    # a 1:1 crop fills the middle 128 rows; the 64 rows above and below are black
    assert np.all(x[0, :, :60] == 0) and np.all(x[0, :, -60:] == 0)
    assert np.allclose(x[0, :, 70:186], 200)


def test_reid_stretch_fills_the_whole_input():
    spec = ap.ReidSpec("t", "unused.onnx", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    x = ap.reid_batch([np.full((100, 100, 3), 200, np.uint8)], spec)
    assert np.allclose(x, 200)


def test_swin_spec_maps_mid_grey_to_zero():
    x = ap.reid_batch([np.full((256, 128, 3), 127, np.uint8)], ap.TAO_SWIN_T)
    assert np.abs(x).max() < 0.01


def test_clip_batch_pads_to_square_keeping_the_whole_body():
    tall = np.full((200, 50, 3), 255, np.uint8)
    x = ap.clip_batch([tall])
    assert x.shape == (1, 3, 224, 224)
    # left/right margins are padding (black), the centre column is the white crop
    assert x[0, 0, 112, 2] < x[0, 0, 112, 112]


def test_classifier_batch_shape_and_imagenet_normalisation():
    mean_bgr = (ap.IMAGENET_MEAN[::-1] * 255).round().astype(np.uint8)
    crop = np.tile(mean_bgr, (300, 100, 1))
    x = ap.classifier_batch([crop, crop])
    assert x.shape == (2, 3, 252, 126) and x.dtype == np.float32 and x.flags["C_CONTIGUOUS"]
    assert np.abs(x).max() < 0.03          # the ImageNet mean maps to ~0 in every channel


def test_classifier_batch_turns_bgr_into_rgb():
    crop = np.zeros((200, 100, 3), np.uint8)
    crop[..., 0] = 255                      # pure blue in BGR
    x = ap.classifier_batch([crop])
    blue = (1.0 - ap.IMAGENET_MEAN[2]) / ap.IMAGENET_STD[2]
    red = (0.0 - ap.IMAGENET_MEAN[0]) / ap.IMAGENET_STD[0]
    assert np.allclose(x[0, 2], blue, atol=1e-4) and np.allclose(x[0, 0], red, atol=1e-4)


def test_classifier_batch_stretches_wide_and_tall_crops_without_padding():
    for h, w in ((100, 400), (400, 50)):
        x = ap.classifier_batch([np.full((h, w, 3), 255, np.uint8)])
        white = (1.0 - ap.IMAGENET_MEAN) / ap.IMAGENET_STD
        assert np.allclose(x[0], white[:, None, None], atol=1e-4)   # no black bands anywhere


def test_classifier_batch_takes_the_smallest_crop():
    x = ap.classifier_batch([np.full((ap.MIN_CROP_PX, ap.MIN_CROP_PX, 3), 90, np.uint8)])
    assert x.shape == (1, 3, 252, 126)


def test_staff_probability_follows_the_nearest_prompt():
    staff, customer = np.eye(4, dtype=np.float32)[0], np.eye(4, dtype=np.float32)[1]
    prompts = {"staff": [staff.tolist()], "customer": [customer.tolist()]}
    p = ap.staff_probability(np.stack([staff, customer]), prompts)
    assert p[0] > 0.99 and p[1] < 0.01


def test_class_probability_uses_the_named_classes():
    cape, no_cape = np.eye(4, dtype=np.float32)[2], np.eye(4, dtype=np.float32)[3]
    prompts = {"cape": [cape.tolist()], "no_cape": [no_cape.tolist()]}
    p = ap.class_probability(np.stack([cape, no_cape]), prompts, "cape", "no_cape")
    assert p[0] > 0.99 and p[1] < 0.01


def test_embedding_round_trip():
    v = ap.l2n(np.arange(1, 257, dtype=np.float32)[None])[0]
    back = ap.decode_embedding(ap.encode_embedding(v))
    assert back.shape == (256,) and np.allclose(back, v, atol=1e-3)


import json as _json


class FakeRedis:
    def __init__(self):
        self.added = []

    def xadd(self, key, fields, maxlen=None, approximate=True):
        self.added.append((key, fields))


def stub_worker(r, interval_s=1.0):
    reid = lambda crops: np.tile(np.eye(1, 256, dtype=np.float32), (len(crops), 1))
    staff = lambda crops: np.full(len(crops), 0.995, np.float32)
    return ap.AppearanceWorker(r, reid, staff, interval_s=interval_s, min_height_px=20)


def test_worker_samples_each_track_once_per_interval():
    r = FakeRedis()
    w = stub_worker(r)
    img = frame(200, 300, 128)
    boxes = [(10, 10, 40, 100, 7), (100, 10, 40, 100, 8), (200, 10, 40, 10, 9)]  # 9 is too short
    assert w.submit("barber-c", img, boxes, now=100.0) == 2
    assert w.submit("barber-c", img, boxes, now=100.5) == 0      # both sampled 0.5 s ago
    assert w.submit("barber-c", img, boxes, now=101.0) == 2
    w.process(w.queue.get_nowait())
    key, fields = r.added[0]
    items = _json.loads(fields["items"])
    assert key == "iris:appearance:barber-c"
    assert [i["track_id"] for i in items] == [7, 8]
    assert items[0]["staff_p"] > 0.99
    assert ap.decode_embedding(items[0]["reid"]).shape == (256,)


def test_worker_crops_are_copies_not_views_of_the_frame():
    w = stub_worker(FakeRedis())
    img = frame(200, 300, 10)
    w.submit("barber-c", img, [(10, 10, 40, 100, 7)], now=1.0)
    img[:] = 250
    job = w.queue.get_nowait()
    assert job.crops[0].max() == 10


def test_worker_drops_a_job_when_the_queue_is_full():
    w = stub_worker(FakeRedis(), interval_s=0.0)
    img = frame(200, 300, 10)
    for i in range(w.queue.maxsize + 3):
        w.submit("barber-c", img, [(10, 10, 40, 100, 7)], now=float(i))
    assert w.queue.qsize() == w.queue.maxsize and w.dropped == 3


def test_spec_by_name_resolves_known_specs():
    assert ap.spec_by_name("tao_resnet50") is ap.TAO_RESNET50
    assert ap.spec_by_name("tao_swin_t") is ap.TAO_SWIN_T
    assert ap.spec_by_name("clipreid_vitb16") is ap.CLIPREID_VITB16


def test_spec_by_name_rejects_unknown_name():
    try:
        ap.spec_by_name("nope")
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "tao_resnet50" in str(exc) and "tao_swin_t" in str(exc) and "clipreid_vitb16" in str(exc)


def test_worker_wants_a_frame_only_when_a_track_is_due():
    """The probe copies the whole frame off the GPU for submit(); asking first
    keeps that copy off the streaming thread when nothing would be sampled
    (the copy on every meta frame with people ran the pipeline down to ~10 fps
    on 2026-09-24)."""
    w = stub_worker(FakeRedis())
    boxes = [(10, 10, 40, 100, 7), (200, 10, 40, 10, 9)]   # 9 is too short to sample
    assert w.wants("barber-c", boxes, now=100.0)
    w.submit("barber-c", frame(200, 300, 128), boxes, now=100.0)
    assert not w.wants("barber-c", boxes, now=100.5)        # 7 sampled 0.5 s ago
    assert not w.wants("barber-c", [(200, 10, 40, 10, 9)], now=200.0)
    assert w.wants("barber-c", boxes, now=101.0)
    assert w.wants("barber-l", boxes, now=100.5)             # per camera


def test_worker_wants_nothing_while_its_queue_is_full():
    """A full queue drops the job anyway, so the copy would be wasted."""
    w = stub_worker(FakeRedis(), interval_s=0.0)
    for i in range(w.queue.maxsize):
        w.submit("barber-c", frame(200, 300, 10), [(10, 10, 40, 100, 7)], now=float(i))
    assert not w.wants("barber-c", [(10, 10, 40, 100, 8)], now=100.0)


def test_worker_saves_a_body_crop_per_track_every_interval(tmp_path):
    r = FakeRedis()
    reid = lambda crops: np.tile(np.eye(1, 256, dtype=np.float32), (len(crops), 1))
    staff = lambda crops: np.full(len(crops), 0.2, np.float32)
    w = ap.AppearanceWorker(r, reid, staff, interval_s=0.0, min_height_px=20,
                            crops_dir=str(tmp_path), body_interval_s=5.0)
    img = frame(300, 300, 128)
    for now in (100.0, 101.0, 106.0):
        w.submit("barber-l", img, [(10, 10, 60, 240, 7)], now=now)
        job = w.queue.get_nowait()
        w.process(job, now=now)
    items = [_json.loads(f["items"])[0] for _, f in r.added]
    assert "crop_path" in items[0] and "crop_path" not in items[1] and "crop_path" in items[2]
    assert (tmp_path / items[0]["crop_path"]).is_file() and items[0]["crop_path"].startswith("bodies/")
    assert items[0]["crop_quality"] == 0.5


def classifier(path, prompts="/nonexistent/prompts.json", clip="/nonexistent/clip.onnx"):
    return ("classifier", str(path), clip, prompts)


def test_worker_publishes_one_float_per_track_from_a_column_shaped_scorer():
    """The classifier ONNX returns [n, 1]; each item must carry a plain float."""
    r = FakeRedis()
    reid = lambda crops: np.tile(np.eye(1, 256, dtype=np.float32), (len(crops), 1))
    w = ap.AppearanceWorker(r, reid, lambda crops: np.full((len(crops), 1), 0.123456, np.float32), min_height_px=20)
    w.submit("barber-c", frame(200, 300, 128), [(10, 10, 40, 100, 7), (100, 10, 40, 100, 8)], now=1.0)
    w.process(w.queue.get_nowait())
    items = _json.loads(r.added[0][1]["items"])
    assert [i["staff_p"] for i in items] == [0.1235, 0.1235]


def test_classifier_scorer_loads_only_the_classifier(tmp_path, monkeypatch):
    model = tmp_path / "staff.onnx"
    model.write_bytes(b"x")
    loaded = []

    def fake_embedder(path, preprocess, output=None):
        loaded.append((path, preprocess, output))
        return lambda crops: np.full((len(crops), 1), 0.8, np.float32)   # the ONNX's [n, 1]
    monkeypatch.setattr(ap, "onnx_embedder", fake_embedder)
    scorer, desc = ap.build_staff_scorer(*classifier(model))
    assert loaded == [(str(model), ap.classifier_batch, "staff_p")]
    p = scorer([np.zeros((50, 20, 3), np.uint8)] * 3)
    assert p.shape == (3,) and np.allclose(p, 0.8)
    assert "classifier" in desc and str(model) in desc


def test_clip_scorer_is_the_default_and_keeps_the_prompts(tmp_path, monkeypatch):
    prompts = tmp_path / "prompts.json"
    prompts.write_text(_json.dumps({"staff": [[1, 0, 0, 0]], "customer": [[0, 1, 0, 0]]}))
    monkeypatch.setattr(ap, "onnx_embedder",
                        lambda path, preprocess, output=None: lambda crops: np.tile(np.eye(1, 4, dtype=np.float32),
                                                                                    (len(crops), 1)))
    scorer, desc = ap.build_staff_scorer("clip", "unused.onnx", "clip.onnx", str(prompts))
    assert scorer([np.zeros((50, 20, 3), np.uint8)])[0] > 0.99
    assert "clip" in desc


def test_unknown_staff_model_fails_instead_of_falling_back():
    for bad in ("Classifier", "dinov2", ""):
        try:
            ap.build_staff_scorer(bad, "c.onnx", "clip.onnx", "p.json")
            assert False, f"expected ValueError for {bad!r}"
        except ValueError as exc:
            assert "classifier" in str(exc) and "clip" in str(exc)


def test_missing_classifier_file_fails_and_names_the_path(tmp_path):
    missing = tmp_path / "nope.onnx"
    try:
        ap.build_staff_scorer(*classifier(missing))
        assert False, "expected FileNotFoundError"
    except FileNotFoundError as exc:
        assert str(missing) in str(exc)


def test_barber_config_selects_the_classifier():
    from pathlib import Path
    from perception.modules.appearance import Settings
    from perception.engine.sources import load_sidecar_config
    repo = Path(__file__).resolve().parents[1]
    barber = load_sidecar_config(str(repo / "products" / "barber" / "sidecar.yaml"))["modules"]["appearance"]
    assert Settings().staff_model == "clip"                  # the module's default is the zero-shot fallback
    assert Settings().staff_classifier == "/workspace/iris/perception/models/staff/staff_dinov2_s.onnx"
    assert Settings(**barber).staff_model == "classifier"
