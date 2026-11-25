"""The hazard annotation core (app/hazards.py) and the Hazard Detection tab's API (app/hazards_api.py), without calling the VLM."""
import base64
import io

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from app import hazards, hazards_api

REPLY = {"scene": "A yard.", "hazards": [
    {"label": "Loose hose", "severity": "high", "description": "A hose lies across the aisle.", "bbox": [100, 200, 400, 600]},
    {"label": "No hard hat", "severity": "Medium", "description": "A worker without a hard hat.", "bbox": [600, 100, 700, 400]},
    {"label": "Broken box", "severity": "low", "description": "No usable box.", "bbox": "nope"},
]}


def _jpeg(w=400, h=300):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (20, 20, 20)).save(buf, "JPEG")
    return buf.getvalue()


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("QWEN_API_KEY", "k")
    monkeypatch.setenv("QWEN_BASE_URL", "http://vlm.invalid/v1")
    monkeypatch.setenv("QWEN_MODEL", "qwen-test")
    monkeypatch.setattr(hazards_api, "SAVE_DIR", tmp_path / "hazards")
    return tmp_path


@pytest.fixture
def client(env, monkeypatch):
    monkeypatch.setattr(hazards, "ask", lambda img, model, base, key, timeout=300: REPLY)
    monkeypatch.setattr(hazards, "translate_ar", lambda res, cfg, timeout=120: {
        "scene": "ساحة.", "hazards": [{"label": f"خطر {i}", "description": "وصف"} for i, _ in enumerate(res["hazards"], 1)]})
    app = FastAPI()
    app.state.stream_manager = None
    app.include_router(hazards_api.router)
    return TestClient(app)


def test_parse_reply_takes_the_json_out_of_prose_and_fences():
    assert hazards.parse_reply('Here you go:\n```json\n{"scene": "x", "hazards": []}\n```') == {"scene": "x", "hazards": []}
    for bad in ("no json here", "{broken json}", ""):
        with pytest.raises(hazards.HazardError):
            hazards.parse_reply(bad)


def test_annotate_draws_a_severity_coloured_box_and_skips_a_broken_bbox():
    img = Image.new("RGB", (1000, 1000), (0, 0, 0))
    out = hazards.annotate(img, REPLY)
    assert out.size == img.size
    # the high hazard's box edge (x1=100 of 1000 -> pixel 100) is red, the medium one's (x1=600) orange
    assert out.getpixel((100, 400)) == hazards.SEVERITY_COLOR["high"]
    assert out.getpixel((600, 250)) == hazards.SEVERITY_COLOR["medium"]
    assert img.getpixel((100, 400)) == (0, 0, 0)         # the input is not touched


def test_settings_need_all_three_variables(monkeypatch):
    monkeypatch.delenv("QWEN_MODEL", raising=False)
    monkeypatch.setenv("QWEN_API_KEY", "k")
    monkeypatch.setenv("QWEN_BASE_URL", "u")
    assert hazards.settings() is None
    monkeypatch.setenv("QWEN_MODEL", "m")
    assert hazards.settings() == hazards.Settings("k", "u", "m")


def test_status_says_whether_the_vlm_is_configured(client, monkeypatch):
    assert client.get("/api/hazards/status").json() == {"configured": True, "model": "qwen-test"}
    monkeypatch.delenv("QWEN_API_KEY")
    assert client.get("/api/hazards/status").json() == {"configured": False, "model": None}


def test_an_uploaded_image_comes_back_annotated_with_its_hazards(client, env):
    r = client.post("/api/hazards/analyze", files={"file": ("yard photo.jpg", _jpeg(), "image/jpeg")})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["scene"] == "A yard." and j["source"] == "upload" and j["model"] == "qwen-test"
    assert [h["label"] for h in j["hazards"]] == ["Loose hose", "No hard hat", "Broken box"]
    assert j["hazards"][1]["severity"] == "medium"                        # lower-cased
    img = Image.open(io.BytesIO(base64.b64decode(j["image"].split(",", 1)[1])))
    assert img.size == (400, 300) and (j["width"], j["height"]) == (400, 300)
    assert (env / "hazards" / j["saved"]["png"]).exists() and (env / "hazards" / j["saved"]["json"]).exists()
    assert "yard_photo" in j["saved"]["png"]


class _Src:
    def __init__(self, frame):
        self._f = frame

    def grab_display_frame(self):
        return self._f


class _Manager:
    def __init__(self, sources):
        self._s = sources

    def get_source(self, sid):
        return self._s.get(sid)


def test_a_camera_frame_can_be_analysed(client):
    client.app.state.stream_manager = _Manager({"cam-01": _Src(_jpeg(320, 240)), "cam-02": _Src(None)})
    r = client.post("/api/hazards/analyze", data={"stream_id": "cam-01"})
    assert r.status_code == 200 and r.json()["source"] == "camera" and r.json()["name"] == "cam-01"
    assert client.post("/api/hazards/analyze", data={"stream_id": "cam-02"}).status_code == 503     # no frame yet
    assert client.post("/api/hazards/analyze", data={"stream_id": "nope"}).status_code == 404
    client.app.state.stream_manager = None
    assert client.post("/api/hazards/analyze", data={"stream_id": "cam-01"}).status_code == 404     # no streams at all


def test_bad_requests_are_refused_before_the_vlm_is_asked(client, monkeypatch):
    called = []
    monkeypatch.setattr(hazards, "ask", lambda *a, **k: called.append(1) or REPLY)
    assert client.post("/api/hazards/analyze").status_code == 400                                        # nothing sent
    r = client.post("/api/hazards/analyze", files={"file": ("x.txt", b"not an image", "text/plain")})
    assert r.status_code == 400 and "not an image" in r.json()["error"]
    r = client.post("/api/hazards/analyze", files={"file": ("a.jpg", _jpeg(), "image/jpeg")}, data={"stream_id": "cam-01"})
    assert r.status_code == 400
    monkeypatch.setattr(hazards_api, "MAX_BYTES", 10)
    assert client.post("/api/hazards/analyze", files={"file": ("a.jpg", _jpeg(), "image/jpeg")}).status_code == 413
    assert called == []


def test_an_unconfigured_vlm_and_a_failing_vlm_are_reported(client, monkeypatch):
    def boom(*a, **k):
        raise hazards.HazardError("API error 429: The budget limit has been reached")
    monkeypatch.setattr(hazards, "ask", boom)
    r = client.post("/api/hazards/analyze", files={"file": ("a.jpg", _jpeg(), "image/jpeg")})
    assert r.status_code == 502 and "429" in r.json()["error"]
    monkeypatch.delenv("QWEN_BASE_URL")
    r = client.post("/api/hazards/analyze", files={"file": ("a.jpg", _jpeg(), "image/jpeg")})
    assert r.status_code == 503 and "not configured" in r.json()["error"]


def test_only_a_few_analyses_run_at_once(client, monkeypatch):
    import threading
    monkeypatch.setattr(hazards_api, "_slots", threading.BoundedSemaphore(0))
    r = client.post("/api/hazards/analyze", files={"file": ("a.jpg", _jpeg(), "image/jpeg")})
    assert r.status_code == 429


def test_saved_analyses_come_back_as_a_history_per_stream(client, env):
    client.app.state.stream_manager = _Manager({"cam-01": _Src(_jpeg()), "cam-02": _Src(_jpeg())})
    a = client.post("/api/hazards/analyze", data={"stream_id": "cam-01"}).json()
    client.post("/api/hazards/analyze", data={"stream_id": "cam-02"})
    client.post("/api/hazards/analyze", files={"file": ("yard photo.jpg", _jpeg(), "image/jpeg")})
    assert [e["name"] for e in client.get("/api/hazards/history").json()["items"]].count("cam-01") == 1
    one = client.get("/api/hazards/history", params={"stream_id": "cam-01"}).json()["items"]
    assert [e["id"] for e in one] == [a["id"]] and one[0]["source"] == "camera" and len(one[0]["hazards"]) == 3
    full = client.get(f"/api/hazards/history/{a['id']}").json()
    assert full["image"].startswith("data:image/jpeg;base64,") and full["scene"] == "A yard."
    assert client.get("/api/hazards/history/../../etc/passwd").status_code == 404
    assert client.get("/api/hazards/history/20200101_000000_nope").status_code == 404


def test_results_saved_before_the_history_existed_are_still_listed(client, env):
    d = env / "hazards"
    d.mkdir()
    (d / "20260101_101010_cam-07_hazards.json").write_text('{"scene": "old", "hazards": [{"label": "x", "severity": "High", "bbox": [0,0,1,1]}]}')
    items = client.get("/api/hazards/history", params={"stream_id": "cam-07"}).json()["items"]
    assert items[0]["scene"] == "old" and items[0]["created"] == "2026-01-01T10:10:10" and items[0]["source"] == "camera"
    assert client.get(f"/api/hazards/history/{items[0]['id']}").status_code == 404      # no png beside it


def test_the_arabic_wording_is_made_with_the_analysis_and_for_older_saves_when_opened(client, env, monkeypatch):
    j = client.post("/api/hazards/analyze", files={"file": ("a.jpg", _jpeg(), "image/jpeg")}).json()
    assert j["ar"]["scene"] == "ساحة." and len(j["ar"]["hazards"]) == 3
    assert client.get(f"/api/hazards/history/{j['id']}").json()["ar"]["hazards"][0]["label"] == "خطر 1"
    d = env / "hazards"
    (d / "20260101_101010_cam-07_hazards.json").write_text('{"scene": "old", "hazards": [{"label": "x", "severity": "low", "bbox": [0,0,1,1]}]}')
    Image.new("RGB", (10, 10)).save(d / "20260101_101010_cam-07_hazards.png")
    assert client.get("/api/hazards/history", params={"stream_id": "cam-07"}).json()["items"][0]["ar"] is None
    assert client.get("/api/hazards/history/20260101_101010_cam-07").json()["ar"]["scene"] == "ساحة."
    assert client.get("/api/hazards/history", params={"stream_id": "cam-07"}).json()["items"][0]["ar"]["scene"] == "ساحة."


def test_a_failing_translation_does_not_fail_the_analysis(client, monkeypatch):
    def boom(*a, **k):
        raise hazards.HazardError("down")
    monkeypatch.setattr(hazards, "translate_ar", boom)
    r = client.post("/api/hazards/analyze", files={"file": ("a.jpg", _jpeg(), "image/jpeg")})
    assert r.status_code == 200 and "ar" not in r.json()
