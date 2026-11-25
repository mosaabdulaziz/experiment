from fastapi.testclient import TestClient
from app.server import create_app


def _cfg(tmp_path):
    return {
        "perception": {"backend": "local"},
        "api": {"backend": "cloud", "api_key": "x", "model": "m"},
        # vlm_enabled defaults to False in create_app, and /api/chat 503s when it is
        # off -- chat tests must opt in explicitly.
        "monitoring": {"vlm_enabled": True},
        "sampling": {}, "events": {"db_path": str(tmp_path / "e.db")},
        "chat": {"tools_enabled": False}, "cv": {"enabled": False}, "camera": {"source": 0},
    }


def test_chat_increments_and_clears_chat_in_flight(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    app = create_app(cfg)          # mutates cfg["monitoring"] in place (bootstrap)

    from app import analyzer as az_mod
    from app import local_chat_context as lcc_mod

    seen = {}

    async def fake_chat_multi(self, frames, message, **kw):
        seen["during"] = cfg["monitoring"]["chat_in_flight"]
        yield {"type": "token", "delta": "FOUND\nhi"}

    monkeypatch.setattr(az_mod.Analyzer, "chat_multi", fake_chat_multi)
    monkeypatch.setattr(lcc_mod.LocalChatContext, "gather_chat_context",
                        lambda self: ({"camera": [b"a"]}, {"camera": [0.0]}, ["camera"]))

    with TestClient(app) as client:
        r = client.post("/api/chat", json={"message": "hi"})
        assert r.status_code == 200

    assert seen["during"] == 1                       # paused during the turn
    assert cfg["monitoring"]["chat_in_flight"] == 0  # restored after the turn
