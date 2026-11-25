"""The incident card's depth image: kept per event, refused when it is not a small JPEG, served by event id."""
import base64
import logging

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.depth_images import DepthImages
from app.server import create_app
from tests.test_zone_proximity import _cfg, _shared_redis


def _jpeg_b64():
    ok, jpg = cv2.imencode(".jpg", np.zeros((90, 160, 3), np.uint8))
    return base64.b64encode(jpg.tobytes()).decode("ascii")


def test_a_depth_image_is_kept_by_event_id(tmp_path):
    store = DepthImages(str(tmp_path / "depth"))
    assert store.put(42, _jpeg_b64())
    with open(store.get(42), "rb") as fh:
        assert fh.read(2) == b"\xff\xd8"
    assert store.get(43) is None


def test_a_depth_image_expires_with_the_alert_clips(tmp_path):
    store = DepthImages(str(tmp_path / "depth"), ttl_seconds=60)
    store.put(42, _jpeg_b64())
    assert store.get(42, clock=lambda: 10**10) is None


def test_a_depth_image_that_is_not_a_small_jpeg_is_refused(tmp_path, caplog):
    store = DepthImages(str(tmp_path / "depth"))
    big = base64.b64encode(b"\xff\xd8" + b"0" * 300_000).decode()
    with caplog.at_level(logging.WARNING):
        for bad in ("not base64!", base64.b64encode(b"GIF89a....").decode(), big):
            assert not store.put(7, bad)
    assert store.get(7) is None and "depth image refused" in caplog.text


def test_the_card_fetches_the_depth_image_by_event_id(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _shared_redis(monkeypatch)
    DepthImages().put(5, _jpeg_b64())                       # data/alert_depth under the test's directory
    app = create_app(_cfg(tmp_path, {"proximity": {}}, backend="deepstream"))
    with TestClient(app) as client:
        r = client.get("/api/alerts/depth/5")
        assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg" and r.content[:2] == b"\xff\xd8"
        assert client.get("/api/alerts/depth/6").status_code == 404
