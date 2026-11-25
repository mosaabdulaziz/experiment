"""The app matches each stay of a vehicle on a camera with that stay's own crop."""
import asyncio
import json

import fakeredis
import pytest
from fastapi import HTTPException

from app import vehicle_gallery as vg
from app.tracking import CameraAggregator, TrackingService, pick_crop_url

KEY = "iris:vehicle:crops"


class _Manager:
    async def broadcast(self, *_a, **_k):
        pass


def _entry(cam, gid, stay, t, score=0.9):
    return {"cam_id": cam, "global_id": gid, "stay": stay, "updated_at": t, "score": score,
            "backend": "local", "path": "x"}


def test_the_crop_taken_inside_a_stay_is_the_stays_crop():
    entries = [_entry("cam-01", 0, 0, 1010.0), _entry("cam-01", 0, 1, 1950.0)]
    assert pick_crop_url(entries, 0, "cam-01", 1000.0, 1040.0).endswith("?stay=0&v=1010")
    assert pick_crop_url(entries, 0, "cam-01", 1900.0, 1990.0).endswith("?stay=1&v=1950")
    # a verdict that came 40 s after the stay ended still matches (slack), and nothing in range -> nearest
    assert "stay=0" in pick_crop_url(entries, 0, "cam-01", 900.0, 970.0)
    assert "stay=1" in pick_crop_url(entries, 0, "cam-01", 5000.0, 5100.0)
    assert pick_crop_url([], 0, "cam-01", 0.0, 1.0) is None


def _service_with_two_stays():
    r = fakeredis.FakeRedis()
    svc = TrackingService({"mtmc": {"cameras": ["cam-01"]}, "truck_feed": {}}, _Manager())
    svc._r = r
    svc._aggregator = CameraAggregator(["cam-01"])
    det = [["vehicle", 0.9, 7, 1, 1, 5, 5]]
    for t in (1000.0, 1003.0):
        svc._aggregator.process_entry("cam-01", det, {"7": 0}, 1.0, wall=t)
    for t in (1900.0, 1904.0):                                            # 15 minutes later
        svc._aggregator.process_entry("cam-01", det, {"7": 0}, 2.0, wall=t)
    r.hset(KEY, "cam-01:0", json.dumps(_entry("cam-01", 0, 0, 1002.0)))
    r.hset(KEY, "cam-01:0:1", json.dumps(_entry("cam-01", 0, 1, 1903.0)))
    return svc


def test_sightings_carry_a_different_crop_for_each_stay_on_the_same_camera():
    out = _service_with_two_stays().get_sightings(0)
    assert len(out) == 2 and out[0]["cam"] == out[1]["cam"] == "cam-01"
    assert out[0]["crop"].endswith("?stay=0&v=1002") and out[1]["crop"].endswith("?stay=1&v=1903")


def test_the_tracks_list_shows_the_newest_stay_per_camera():
    urls = _service_with_two_stays()._crop_urls()
    assert urls[0]["cam-01"].endswith("?stay=1&v=1903")


def test_the_image_endpoint_serves_the_asked_stay_and_defaults_to_the_first(tmp_path):
    r = fakeredis.FakeRedis()
    vg._redis_client, vg._local_dir, vg._storage_backend = r, str(tmp_path), "local"
    (tmp_path / "cam-01").mkdir()
    (tmp_path / "cam-01" / "0.jpg").write_bytes(b"first")
    (tmp_path / "cam-01" / "0-1.jpg").write_bytes(b"second")
    r.hset(KEY, "cam-01:0", json.dumps(_entry("cam-01", 0, 0, 1.0)))
    r.hset(KEY, "cam-01:0:1", json.dumps(_entry("cam-01", 0, 1, 2.0)))
    first = asyncio.run(vg.get_crop_image("cam-01", 0))
    second = asyncio.run(vg.get_crop_image("cam-01", 0, stay=1))
    assert pathlib_name(first) == "0.jpg" and pathlib_name(second) == "0-1.jpg"
    with pytest.raises(HTTPException):
        asyncio.run(vg.get_crop_image("cam-01", 0, stay=2))


def pathlib_name(resp):
    return resp.path.rsplit("/", 1)[-1]
