from fastapi.testclient import TestClient
from app.server import create_app


def _cfg(tmp_path):
    return {
        "perception": {"backend": "local"},
        "api": {"backend": "cloud", "api_key": "x", "model": "m"},
        "monitoring": {"mode": "proactive"}, "sampling": {},
        "events": {"db_path": str(tmp_path / "e.db")},
        "chat": {"tools_enabled": False}, "cv": {"enabled": False}, "camera": {"source": 0},
    }


def test_get_mode_defaults_to_proactive(tmp_path):
    app = create_app(_cfg(tmp_path))
    with TestClient(app) as client:
        r = client.get("/api/mode")
        assert r.status_code == 200
        assert r.json() == {"mode": "proactive"}


def test_post_mode_switches_to_chat(tmp_path):
    app = create_app(_cfg(tmp_path))
    with TestClient(app) as client:
        r = client.post("/api/mode", json={"mode": "chat"})
        assert r.status_code == 200
        assert r.json() == {"status": "updated", "mode": "chat"}
        assert client.get("/api/mode").json() == {"mode": "chat"}


def test_post_mode_rejects_unknown(tmp_path):
    app = create_app(_cfg(tmp_path))
    with TestClient(app) as client:
        r = client.post("/api/mode", json={"mode": "bogus"})
        assert r.status_code == 400


def test_ws_history_includes_mode(tmp_path):
    app = create_app(_cfg(tmp_path))
    with TestClient(app) as client:
        client.post("/api/mode", json={"mode": "chat"})
        with client.websocket_connect("/ws") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "history"
            assert msg["mode"] == "chat"
