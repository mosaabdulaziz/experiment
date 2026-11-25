"""/api/safety/*: the safety product's settings, cameras, Apply and acknowledgement queue (safety integration,
ticket 2). Mounted only when IRIS_PRODUCT names the safety product."""
import pathlib

import fakeredis
import yaml
from fastapi.testclient import TestClient

from app.server import create_app

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _cfg(tmp_path):
    cams = tmp_path / "cameras.yaml"
    cams.write_text(yaml.safe_dump({"cameras": [
        {"id": "bay-a", "name": "Bay A", "clip": "a.mp4", "features": ["proximity", "zones", "ppe"]}]}))
    return {
        "perception": {"backend": "local"},
        "api": {"backend": "cloud", "api_key": "x", "model": "m"},
        "monitoring": {"mode": "proactive"}, "sampling": {},
        "events": {"db_path": str(tmp_path / "e.db")},
        "chat": {"tools_enabled": False}, "cv": {"enabled": False}, "camera": {"source": 0},
        "zones": {"file_path": str(tmp_path / "zones_config.txt"),
                  "sidecar_config_path": str(tmp_path / "none.yaml")},
        "safety": {"data_dir": str(tmp_path / "data"), "cameras": str(cams),
                   "product": str(ROOT / "products/safety/safety.yaml")},
    }


def _client(tmp_path, monkeypatch, product="safety"):
    monkeypatch.setenv("IRIS_PRODUCT", product)
    app = create_app(_cfg(tmp_path))
    if getattr(app.state, "safety", None) is not None:
        app.state.safety._redis = fakeredis.FakeStrictRedis()
    return TestClient(app)


def test_not_mounted_for_another_product(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch, product="iris") as client:
        assert client.get("/api/safety/settings").status_code == 404


def test_settings_read_and_change(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        assert client.get("/api/safety/settings").json()["features"]["fire"]["ack_required"] is True
        r = client.put("/api/safety/settings", json={"features": {"ppe": {"notify": True}}})
        assert r.status_code == 200 and r.json()["features"]["ppe"]["notify"] is True
        r = client.put("/api/safety/settings", json={"language": "fr"})
        assert r.status_code == 400 and "language" in r.json()["error"]


def test_cameras_are_listed_toggled_and_applied(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        (bay,) = client.get("/api/safety/cameras").json()["cameras"]
        assert bay["features"] == ["proximity", "zones", "ppe"]
        r = client.put("/api/safety/cameras/bay-a", json={"features": ["ppe"], "name": "Loading bay"})
        assert r.status_code == 200 and r.json()["features"] == ["ppe"]
        assert client.put("/api/safety/cameras/nope", json={"name": "x"}).status_code == 404
        assert client.put("/api/safety/cameras/bay-a", json={"features": ["teleport"]}).status_code == 400
        r = client.post("/api/safety/apply")
        assert r.json() == {"version": 1, "applied_version": 0, "applying": True}
        assert client.get("/api/safety/apply").json()["applying"] is True


def test_the_acknowledgement_queue(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        r = client.get("/api/safety/unacknowledged")
        assert r.status_code == 200 and r.json() == {"events": []}


class _Fall:
    def __init__(self):
        self.calls = []

    def __call__(self, method, path, body=None):
        self.calls.append((method, path, body))
        if path.startswith("/candidates"):
            return 200, {"candidates": [{"id": 3, "camera": "rest-room", "status": "candidate"}]}
        return 200, {"id": 3, "status": body["status"]}


def test_the_fall_review_queue_is_proxied_to_the_fall_service(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        fall = _Fall()
        client.app.state.safety.fall_http = fall
        r = client.get("/api/safety/fall/candidates?status=candidate")
        assert r.status_code == 200 and r.json()["candidates"][0]["camera"] == "rest-room"
        r = client.post("/api/safety/fall/review/3", json={"status": "confirmed", "note": "real fall"})
        assert r.status_code == 200 and r.json() == {"id": 3, "status": "confirmed"}
        assert fall.calls == [("GET", "/candidates?status=candidate", None),
                              ("POST", "/review/3", {"status": "confirmed", "note": "real fall"})]


def test_without_a_fall_service_the_proxy_says_so(tmp_path, monkeypatch):
    monkeypatch.delenv("SAFETY_FALL_URL", raising=False)
    with _client(tmp_path, monkeypatch) as client:
        r = client.get("/api/safety/fall/candidates")
        assert r.status_code == 503 and "fall service" in r.json()["error"]


def test_the_telegram_notifier_runs_with_the_app(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        notifier = client.app.state.safety_notifier
        assert notifier is not None and notifier._task is not None and not notifier._task.done()


def test_telegram_setup_test_message_and_chats(tmp_path, monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")
    with _client(tmp_path, monkeypatch) as client:
        calls = []

        def telegram(method, token, fields, photo=None):
            calls.append((method, fields.get("chat_id")))
            return [{"my_chat_member": {"chat": {"id": -1001, "title": "Site safety", "type": "group"}}}]
        client.app.state.safety_notifier._send = telegram
        r = client.post("/api/safety/telegram/test")
        assert r.status_code == 400 and "chat" in r.json()["error"]            # no chat saved yet
        assert client.get("/api/safety/telegram/chats").json() == {
            "chats": [{"id": "-1001", "title": "Site safety", "type": "group"}]}
        client.put("/api/safety/settings", json={"telegram": {"chat_id": "-1001"}})
        assert client.post("/api/safety/telegram/test").json() == {"ok": True}
        assert calls[-1] == ("sendMessage", "-1001")
        r = client.put("/api/safety/settings", json={"telegram": {"enabled": True}})
        assert r.json()["telegram"] == {"enabled": True, "chat_id": "-1001", "token_set": True}


def test_telegram_refusing_is_reported(tmp_path, monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")
    with _client(tmp_path, monkeypatch) as client:
        def refuse(method, token, fields, photo=None):
            raise OSError("telegram: Unauthorized")
        client.app.state.safety_notifier._send = refuse
        client.put("/api/safety/settings", json={"telegram": {"chat_id": "-1001"}})
        r = client.post("/api/safety/telegram/test")
        assert r.status_code == 502 and "Unauthorized" in r.json()["error"]


class _Go2rtc:
    """go2rtc's /api/frame.jpeg for the tests: a JPEG per call, or unreachable."""

    def __init__(self, down=False):
        self.calls, self.down = [], down

    def __call__(self, src):
        self.calls.append(src)
        if self.down:
            raise ConnectionError("connection refused")
        return b"\xff\xd8jpeg" + str(len(self.calls)).encode()


def test_a_camera_s_still_comes_from_go2rtc_without_any_feature_running(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        go2rtc = _Go2rtc()
        client.app.state.safety.go2rtc_frame = go2rtc
        r = client.get("/api/safety/cameras/bay-a/snapshot")
        assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
        assert r.content == b"\xff\xd8jpeg1" and r.headers["cache-control"] == "no-store"
        # the camera's raw stream (what the role sidecars pull); a second viewer within a moment shares the still
        assert client.get("/api/safety/cameras/bay-a/snapshot").content == b"\xff\xd8jpeg1"
        assert go2rtc.calls == ["raw-bay-a"]


def test_no_still_for_an_unknown_camera_or_from_an_unreachable_go2rtc(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        client.app.state.safety.go2rtc_frame = _Go2rtc(down=True)
        assert client.get("/api/safety/cameras/nope/snapshot").status_code == 404
        r = client.get("/api/safety/cameras/bay-a/snapshot")
        assert r.status_code == 502 and "connection refused" in r.json()["error"]


def test_a_failed_still_is_shared_too_so_viewers_do_not_queue_on_a_dead_camera(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        go2rtc = _Go2rtc(down=True)
        client.app.state.safety.go2rtc_frame = go2rtc
        assert [client.get("/api/safety/cameras/bay-a/snapshot").status_code for _ in range(3)] == [502, 502, 502]
        assert go2rtc.calls == ["raw-bay-a"]                    # one attempt per 1.5 s, whatever the viewers


def test_the_still_is_asked_of_go2rtc_for_the_camera_s_raw_stream(tmp_path, monkeypatch):
    import io
    import urllib.error
    from products.safety import service as service_mod
    asked = []

    def urlopen(url, timeout):
        asked.append((url, timeout))
        if "raw-dead" in url:
            raise urllib.error.URLError("connection refused")
        return io.BytesIO(b"" if "raw-blank" in url else b"\xff\xd8jpeg")

    monkeypatch.setattr(service_mod.urllib.request, "urlopen", urlopen)
    svc = service_mod.SafetyService({"safety": {"go2rtc_url": "http://go2rtc:1984/"}})
    assert svc._go2rtc_frame("raw-bay a") == b"\xff\xd8jpeg"
    assert asked[0] == ("http://go2rtc:1984/api/frame.jpeg?src=raw-bay%20a", 8)
    for src in ("raw-dead", "raw-blank"):
        try:
            svc._go2rtc_frame(src)
            raise AssertionError(src)
        except ConnectionError as exc:
            assert "go2rtc" in str(exc)


def test_only_the_safety_product_draws_on_its_smart_record_clips(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        assert client.app.state.alert_clip_service.svr_overlays is True
    with _client(tmp_path, monkeypatch, product="iris") as client:
        assert getattr(client.app.state, "alert_clip_service", None) is None     # the safety block never ran


def test_the_cameras_say_each_feature_s_frame_rate_when_camera_health_has_one(tmp_path, monkeypatch):
    import types
    with _client(tmp_path, monkeypatch) as client:
        client.app.state.stream_manager = types.SimpleNamespace(list_streams=lambda: [
            {"id": "bay-a@ppe", "camera": "bay-a", "feature": "ppe", "running": True},
            {"id": "bay-a@proximity", "camera": "bay-a", "feature": "proximity", "running": True}])
        client.app.state.safety_health = types.SimpleNamespace(fps={"bay-a@ppe": 4.94})
        (bay,) = client.get("/api/safety/cameras").json()["cameras"]
        assert {s["id"]: s["fps"] for s in bay["streams"]} == {"bay-a@ppe": 4.9, "bay-a@proximity": None}
