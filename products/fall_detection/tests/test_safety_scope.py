"""The safety product's two asks of the fall service (safety integration, ticket 1).

1. Role-scoped keys: beside other role sidecars on the same camera, the fall sidecar writes its frames under
   iris:meta:<camera>@<role>; the service reads there and writes its alert and clip request there too.
2. Fall zones: a camera with zones whose `fall` block is on alerts only for a person inside one (box bottom centre),
   and names it; a camera with none covers the whole frame.
"""
import json

import acl
import config
from consumer import CameraWorker, Consumer
from fall_helpers import DECISIVE_FALL, POLICY, VENDOR_CFG, FakeRedis, FakeStorage, detailed, make_rec


class _NoCandidates:
    def update(self, *a, **k):
        return []


def _zones_file(tmp_path):
    path = tmp_path / "zones_config.txt"
    path.write_text(json.dumps({"cam-1": {"zones": [
        {"id": "bay-a", "name": "Bay A", "points": [[0, 0], [400, 0], [400, 720], [0, 720]],
         "fall": {"enabled": True}}]}}))
    return path


def _worker(redis, storage, policy=POLICY, **kw):
    return CameraWorker(camera="cam-1", monitor=_NoCandidates(), redis=redis, storage=storage,
                        policy=policy, vendor_cfg=VENDOR_CFG, **kw)


def _frame(track, left, shape=(1280, 720)):
    box = [left, 300, left + 120, 600]
    return {"pts": "10.0", "pose": json.dumps({"shape": list(shape), "items": [
        {"id": track, "box": box, "kps": [[0.0, 0.0, 1.0]] * 17}]})}


def test_a_role_scoped_worker_writes_under_the_key_id():
    redis, storage = FakeRedis(), FakeStorage()
    rec = make_rec(kind="fall", probs=DECISIVE_FALL, t_alert=12.0)
    _worker(redis, storage, key_id="cam-1@fall").handle_candidates([rec], t=12.0)
    assert [k for k, _f in redis.added] == ["iris:alert:cam-1@fall", "iris:svr_request:cam-1@fall"]


class _XRedis:
    def __init__(self):
        self.asked = []

    def xrevrange(self, key, count=1):
        self.asked.append(key)
        return []

    def xread(self, streams, count=None):
        self.asked.extend(streams)
        return []


def test_the_reader_reads_each_worker_s_key_id():
    redis = _XRedis()
    worker = _worker(FakeRedis(), FakeStorage(), key_id="cam-1@fall")
    Consumer(redis, {"cam-1": worker})._read_once()
    assert redis.asked == ["iris:meta:cam-1@fall"]


def test_a_fall_inside_a_fall_zone_alerts_and_names_it(tmp_path):
    redis, storage = FakeRedis(), FakeStorage()
    worker = _worker(redis, storage, zones=acl.ZonesFile(_zones_file(tmp_path), "fall"), zones_mux=(1280, 720))
    worker.handle_entry(_frame(track=1, left=100))
    worker.handle_candidates([make_rec(kind="fall", track=1, probs=DECISIVE_FALL, t_alert=12.0)], t=12.0)
    (alert,) = redis.entries("iris:alert:cam-1")
    assert alert["zone"] == "bay-a"


def test_a_fall_outside_every_fall_zone_writes_nothing(tmp_path):
    redis, storage = FakeRedis(), FakeStorage()
    worker = _worker(redis, storage, zones=acl.ZonesFile(_zones_file(tmp_path), "fall"), zones_mux=(1280, 720))
    worker.handle_entry(_frame(track=1, left=900))
    (result,) = worker.handle_candidates([make_rec(kind="fall", track=1, probs=DECISIVE_FALL, t_alert=12.0)], t=12.0)
    assert redis.added == [] and storage.rows == [] and storage.events == []
    assert result["out_of_zone"] is True


def test_capture_pixels_are_scaled_to_the_zones_mux(tmp_path):
    redis, storage = FakeRedis(), FakeStorage()
    worker = _worker(redis, storage, zones=acl.ZonesFile(_zones_file(tmp_path), "fall"), zones_mux=(1280, 720))
    worker.handle_entry(_frame(track=1, left=600, shape=(2560, 1440)))      # centre x 660 capture = 330 mux
    worker.handle_candidates([make_rec(kind="fall", track=1, probs=DECISIVE_FALL, t_alert=12.0)], t=12.0)
    assert redis.entries("iris:alert:cam-1")[0]["zone"] == "bay-a"


def test_a_malformed_fall_zone_is_skipped_and_never_loses_the_fall(tmp_path):
    """A zone point that is not an [x, y] pair (a hand-edited file) skips that zone; the candidate is still scored."""
    path = tmp_path / "zones_config.txt"
    path.write_text(json.dumps({
        "cam-1": {"zones": [{"id": "bad", "points": [[0, 0], [5], ["a", 1]], "fall": {"enabled": True}},
                            {"id": "bay-a", "points": [[0, 0], [400, 0], [400, 720], [0, 720]],
                             "fall": {"enabled": True}}]},
        "cam-2": {"zones": 7}}))
    redis, storage = FakeRedis(), FakeStorage()
    worker = _worker(redis, storage, zones=acl.ZonesFile(path, "fall"), zones_mux=(1280, 720))
    worker.handle_entry(_frame(track=1, left=100))
    worker.handle_candidates([make_rec(kind="fall", track=1, probs=DECISIVE_FALL, t_alert=12.0)], t=12.0)
    assert redis.entries("iris:alert:cam-1")[0]["zone"] == "bay-a"
    assert acl.ZonesFile(path, "fall").covering("cam-2", 1, 1) == (True, "")


def test_without_fall_zones_the_whole_frame_is_covered(tmp_path):
    redis, storage = FakeRedis(), FakeStorage()
    worker = _worker(redis, storage, zones=acl.ZonesFile(tmp_path / "none.txt", "fall"), zones_mux=(1280, 720))
    worker.handle_entry(_frame(track=1, left=900))
    worker.handle_candidates([make_rec(kind="fall", track=1, probs=DECISIVE_FALL, t_alert=12.0)], t=12.0)
    assert redis.entries("iris:alert:cam-1")[0]["zone"] == ""


def test_config_names_the_key_role_and_the_zones(tmp_path):
    site = tmp_path / "site.yaml"
    site.write_text(f"extends: {config.CONFIG_PATH}\nkey_role: fall\n"
                    f"zones: {{path: {tmp_path / 'z.txt'}, mux: [1280, 720]}}\n")
    cfg = config.load_config(site, env={})
    assert [c.key_id for c in cfg.cameras] == ["cam-1@fall"]
    assert cfg.zones_path == tmp_path / "z.txt" and cfg.zones_mux == (1280, 720)


def test_config_defaults_keep_today_s_keys_and_no_zones():
    cfg = config.load_config(env={})
    assert [c.key_id for c in cfg.cameras] == ["cam-1"]
    assert cfg.zones_path is None and cfg.zones_mux is None


def test_a_fall_alert_names_its_person_s_track_so_its_clip_draws_that_person(tmp_path):
    """The pose module writes each person's box by track id to the overlay stream; the app's clip builder draws the
    one the alert names (person_ids, app/overlay_track.py incident_pair)."""
    redis, storage = FakeRedis(), FakeStorage()
    worker = _worker(redis, storage, detailed(), zones=acl.ZonesFile(tmp_path / "none.txt", "fall"),
                     zones_mux=(1280, 720))
    worker.handle_entry(_frame(track=7, left=100))
    worker.handle_candidates([make_rec(kind="fall", track=7, probs=DECISIVE_FALL, t_alert=12.0)], t=12.0)
    assert redis.entries("iris:alert:cam-1")[0]["person_ids"] == "7"
