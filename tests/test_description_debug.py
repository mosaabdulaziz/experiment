import asyncio
import json
import os
from types import SimpleNamespace

from app.analyzer import Analyzer


def _analyzer(tmp_path, keep=3):
    return Analyzer({"api": {"backend": "cloud", "api_key": "x", "model": "m",
                             "desc_debug_path": str(tmp_path / "description_calls"),
                             "desc_debug_keep": keep},
                     "monitoring": {}, "sampling": {}, "chat": {}})


def _capturing_completion(text="A person climbed the fence."):
    calls = {}

    async def create(*a, **k):
        calls["kwargs"] = k
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])
    create.calls = calls
    return create


def _meta(slot_dir):
    with open(os.path.join(slot_dir, "meta.json"), encoding="utf-8") as fh:
        return json.load(fh)


def test_save_description_debug_writes_frames_and_prompt(tmp_path):
    az = _analyzer(tmp_path)
    az._save_description_debug("intrusion", "cam3", [b"jpeg-a", b"jpeg-b"],
                               "sys prompt", "analyze this", "a person entered")

    slot_dir = tmp_path / "description_calls" / "0"
    assert slot_dir.is_dir()
    meta = _meta(slot_dir)
    assert meta["prompt_kind"] == "intrusion"
    assert meta["source_id"] == "cam3"
    assert meta["system_prompt"] == "sys prompt"
    assert meta["prompt_text"] == "analyze this"
    assert meta["answer"] == "a person entered"
    assert meta["count"] is None
    frame_files = sorted(f for f in os.listdir(slot_dir) if f.endswith(".jpg"))
    assert frame_files == ["frame_00.jpg", "frame_01.jpg"]


def test_save_description_debug_rotates_across_both_kinds(tmp_path):
    az = _analyzer(tmp_path, keep=3)
    for i in range(4):
        kind = "overcrowding" if i % 2 else "intrusion"
        az._save_description_debug(kind, f"cam{i}", [b"x"], "sys", "prompt", f"answer{i}")

    base = tmp_path / "description_calls"
    # only 3 slots total, shared across kinds; slot 0 overwritten by the 4th call
    assert {p.name for p in base.iterdir()} == {"0", "1", "2"}
    assert _meta(base / "0")["answer"] == "answer3"
    assert _meta(base / "0")["prompt_kind"] == "overcrowding"
    assert _meta(base / "1")["answer"] == "answer1"


def test_describe_intrusion_writes_debug_dump(tmp_path):
    az = _analyzer(tmp_path)
    az._client.chat.completions.create = _capturing_completion("A person entered the zone.")

    result = asyncio.run(az.describe_alert("intrusion", [b"frame0"], "cam1"))

    assert result == "A person entered the zone."
    slot_dir = tmp_path / "description_calls" / "0"
    meta = _meta(slot_dir)
    assert meta["prompt_kind"] == "intrusion"
    assert meta["source_id"] == "cam1"
    assert meta["answer"] == "A person entered the zone."
    assert "cam1" in meta["system_prompt"]


def test_describe_overcrowding_writes_debug_dump_with_count(tmp_path):
    az = _analyzer(tmp_path)
    az._client.chat.completions.create = _capturing_completion("A crowd has gathered.")

    result = asyncio.run(az.describe_alert("overcrowding", [b"frame0"], "cam2", 8))

    assert result == "A crowd has gathered."
    slot_dir = tmp_path / "description_calls" / "0"
    meta = _meta(slot_dir)
    assert meta["prompt_kind"] == "overcrowding"
    assert meta["source_id"] == "cam2"
    assert meta["count"] == 8
    assert meta["answer"] == "A crowd has gathered."


def test_describe_intrusion_empty_frames_writes_no_debug_dump(tmp_path):
    az = _analyzer(tmp_path)
    az._client.chat.completions.create = _capturing_completion()

    asyncio.run(az.describe_alert("intrusion", [], "cam1"))

    assert not (tmp_path / "description_calls").exists()


def test_clear_description_debug_removes_everything(tmp_path):
    az = _analyzer(tmp_path)
    az._save_description_debug("intrusion", "cam1", [b"x"], "sys", "prompt", "answer")
    assert (tmp_path / "description_calls").exists()

    az.clear_description_debug()

    assert not (tmp_path / "description_calls").exists()
    assert az._desc_debug_slot == 0
