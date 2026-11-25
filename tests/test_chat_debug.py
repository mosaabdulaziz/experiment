import asyncio
import json
import os
from types import SimpleNamespace

from app.analyzer import Analyzer, _elide_image_payloads


def _analyzer(tmp_path, keep=3):
    return Analyzer({"api": {"backend": "cloud", "api_key": "x", "model": "m",
                             "chat_debug_path": str(tmp_path / "chat_calls"),
                             "chat_debug_keep": keep},
                     "monitoring": {}, "sampling": {}, "chat": {"tools_enabled": False}})


def _capturing_stream(text="ok"):
    calls = {}

    async def create(*a, **k):
        calls["kwargs"] = k

        async def gen():
            for ch in text:
                yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=ch))])
        return gen()
    create.calls = calls
    return create


def _drain(agen):
    async def go():
        async for _ in agen:
            pass
    asyncio.run(go())


def test_elide_image_payloads_replaces_base64_keeps_text():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": [
            {"type": "text", "text": "=== Camera cam0 @ 00:00:01 ==="},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}},
            {"type": "text", "text": "what?"},
        ]},
    ]
    out = _elide_image_payloads(messages)
    assert out[0] == {"role": "system", "content": "sys"}        # string content untouched
    blocks = out[1]["content"]
    assert blocks[1] == {"type": "image_url", "image_url": "<frame 0>"}
    assert blocks[0]["text"].startswith("=== Camera")            # text preserved
    # original messages are not mutated
    assert messages[1]["content"][1]["image_url"] == {"url": "data:image/jpeg;base64,AAAA"}


def test_chat_multi_appends_prompt_and_frames_to_debug_record(tmp_path):
    az = _analyzer(tmp_path)
    az._client.chat.completions.create = _capturing_stream("ok")
    record: list = []
    _drain(az.chat_multi({"cam0": [b"a", b"b"]}, "q", history=[], analytics=None,
                         prior_turns=[], per_camera_offsets={"cam0": [1.0, 2.0]},
                         sweep=True, debug_record=record))
    assert len(record) == 1
    entry = record[0]
    assert entry["frames"] == {"cam0": [b"a", b"b"]}
    assert entry["sweep"] is True
    # the captured messages are the real ones sent to the model (system + user)
    assert entry["messages"][0]["role"] == "system"
    assert "NOT_FOUND" in entry["messages"][0]["content"]   # sweep suffix injected


def _meta(slot_dir):
    with open(os.path.join(slot_dir, "meta.json"), encoding="utf-8") as fh:
        return json.load(fh)


def test_save_chat_debug_writes_frames_and_prompt(tmp_path):
    az = _analyzer(tmp_path)
    calls = [{
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,ZZ"}},
                {"type": "text", "text": "q"},
            ]},
        ],
        "frames": {"cam0": [b"jpeg-a"], "cam1": [b"jpeg-b"]},
        "offsets": {"cam0": [5.0], "cam1": [9.0]},
        "sweep": True,
        "use_tools": False,
    }]
    az.save_chat_debug("search", "find the car", calls, "FOUND it", session_id="s1")

    slot_dir = tmp_path / "chat_calls" / "search" / "0"
    assert slot_dir.is_dir()
    meta = _meta(slot_dir)
    assert meta["mode"] == "search" and meta["message"] == "find the car"
    assert meta["answer"] == "FOUND it" and meta["session_id"] == "s1"
    assert meta["model_calls"] == 1
    call = meta["calls"][0]
    assert call["frames_per_camera"] == {"cam0": 1, "cam1": 1}
    assert call["offsets"] == {"cam0": [5.0], "cam1": [9.0]}
    # image payload elided in the saved prompt; jpegs written as files
    assert call["messages"][1]["content"][0] == {"type": "image_url", "image_url": "<frame 0>"}
    assert set(call["frame_files"]) == {f for f in os.listdir(slot_dir) if f.endswith(".jpg")}
    assert len(call["frame_files"]) == 2


def test_save_chat_debug_rotates_and_separates_modes(tmp_path):
    az = _analyzer(tmp_path, keep=3)
    mk = lambda msg: [{"messages": [{"role": "system", "content": "s"}],
                       "frames": {"cam0": [b"x"]}, "offsets": {}, "sweep": False}]
    for i in range(4):
        az.save_chat_debug("normal", f"msg{i}", mk(i), f"a{i}")
    base = tmp_path / "chat_calls" / "normal"
    # only 3 slots, slot 0 was overwritten by the 4th request (msg3)
    assert {p.name for p in base.iterdir()} == {"0", "1", "2"}
    assert _meta(base / "0")["message"] == "msg3"
    # search mode keeps its own slots, untouched by normal-mode writes
    az.save_chat_debug("search", "smsg", mk(0), "sa")
    assert (tmp_path / "chat_calls" / "search" / "0").is_dir()


def test_chat_endpoint_writes_debug_dump(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from app.server import create_app
    from app import analyzer as az_mod
    from app import local_chat_context as lcc_mod

    cfg = {
        "perception": {"backend": "local"},
        "api": {"backend": "cloud", "api_key": "x", "model": "m",
                "chat_debug_path": str(tmp_path / "chat_calls")},
        # vlm_enabled defaults to False in create_app, and /api/chat 503s when it is
        # off -- chat tests must opt in explicitly.
        "monitoring": {"vlm_enabled": True},
        "sampling": {}, "events": {"db_path": str(tmp_path / "e.db")},
        "chat": {"tools_enabled": False}, "cv": {"enabled": False}, "camera": {"source": 0},
    }
    app = create_app(cfg)

    async def fake_chat_multi(self, frames, message, **kw):
        rec = kw.get("debug_record")
        if rec is not None:
            rec.append({"messages": [{"role": "system", "content": "sys"}],
                        "frames": frames, "offsets": kw.get("per_camera_offsets"),
                        "sweep": kw.get("sweep"), "use_tools": False})
        yield {"type": "token", "delta": "Two cars are visible."}

    monkeypatch.setattr(az_mod.Analyzer, "chat_multi", fake_chat_multi)
    monkeypatch.setattr(lcc_mod.LocalChatContext, "gather_chat_context",
                        lambda self: ({"camera": [b"a"]}, {"camera": [0.0]}, ["camera"]))

    with TestClient(app) as client:
        r = client.post("/api/chat", json={"message": "what do you see?", "mode": "normal"})
        assert r.status_code == 200

    slot = tmp_path / "chat_calls" / "normal" / "0"
    assert slot.is_dir()
    meta = _meta(slot)
    assert meta["mode"] == "normal"
    assert meta["answer"] == "Two cars are visible."
    assert meta["calls"][0]["frames_per_camera"] == {"camera": 1}


def test_clear_chat_debug_removes_everything(tmp_path):
    az = _analyzer(tmp_path)
    az.save_chat_debug("normal", "m", [{"messages": [], "frames": {"cam0": [b"x"]},
                                        "offsets": {}, "sweep": False}], "a")
    assert (tmp_path / "chat_calls").exists()
    az.clear_chat_debug()
    assert not (tmp_path / "chat_calls").exists()
    assert az._chat_debug_slots == {}
