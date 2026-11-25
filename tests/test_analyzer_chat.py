from app.analyzer import Analyzer


def test_format_history_tags_camera_when_enabled():
    history = [
        {"event_type": "alert", "message": "red-light runner", "camera": "cam1",
         "timestamp": "2026-06-23T10:00:00Z"},
        {"event_type": "comment", "message": "sparse traffic", "camera": "cam0",
         "timestamp": "2026-06-23T10:00:05Z"},
    ]
    out = Analyzer._format_history(history, include_camera=True)
    assert "[cam1] ALERT: red-light runner" in out
    assert "[cam0] COMMENT: sparse traffic" in out


def test_format_history_untagged_when_no_camera():
    history = [{"event_type": "comment", "message": "hi", "camera": None,
               "timestamp": "2026-06-23T10:00:00Z"}]
    out = Analyzer._format_history(history, include_camera=True)
    assert "[cam" not in out
    assert "COMMENT: hi" in out


def test_format_history_default_omits_camera_tag():
    # analyze() path: default include_camera=False -> no camera tag even if present
    history = [{"event_type": "comment", "message": "hi", "camera": "cam1",
               "timestamp": "2026-06-23T10:00:00Z"}]
    assert "[cam1]" not in Analyzer._format_history(history)


def test_format_history_empty():
    assert Analyzer._format_history([]) == "No previous observations."


import asyncio
from types import SimpleNamespace


def _analyzer():
    # tools disabled so chat_multi makes only the single streaming call
    return Analyzer({"api": {"backend": "cloud", "api_key": "x", "model": "m"},
                     "monitoring": {}, "sampling": {}, "chat": {"tools_enabled": False}})


def _capturing_stream(text="ok"):
    """Fake create() that records the kwargs of the streaming call and yields chunks."""
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
        out = []
        async for it in agen:
            out.append(it)
        return out
    return asyncio.run(go())


def _final_user_content(cap):
    return cap.calls["kwargs"]["messages"][-1]["content"]


def test_chat_multi_labels_each_camera_and_orders_content():
    az = _analyzer()
    cap = _capturing_stream("ok")
    az._client.chat.completions.create = cap
    frames = {"cam0": [b"a", b"b"], "cam1": [b"c"]}
    items = _drain(az.chat_multi(frames, "what's up?", history=[], analytics=None, prior_turns=[]))
    assert "".join(i["delta"] for i in items if i["type"] == "token") == "ok"
    content = _final_user_content(cap)
    texts = [c["text"] for c in content if c["type"] == "text"]
    images = [c for c in content if c["type"] == "image_url"]
    assert "=== Camera cam0 ===" in texts
    assert "=== Camera cam1 ===" in texts
    assert texts[-1] == "what's up?"     # question is last
    assert len(images) == 3              # 2 + 1 frames


def test_chat_multi_single_camera_omits_label():
    az = _analyzer()
    cap = _capturing_stream()
    az._client.chat.completions.create = cap
    _drain(az.chat_multi({"camera": [b"a"]}, "hi", history=[], analytics=None, prior_turns=[]))
    texts = [c["text"] for c in _final_user_content(cap) if c["type"] == "text"]
    assert not any(t.startswith("=== Camera") for t in texts)


def test_chat_multi_enforces_max_images():
    az = _analyzer()
    cap = _capturing_stream()
    az._client.chat.completions.create = cap
    frames = {"cam0": [b"0", b"1", b"2", b"3"], "cam1": [b"0", b"1", b"2", b"3"]}
    _drain(az.chat_multi(frames, "q", history=[], analytics=None, prior_turns=[], max_images=2))
    images = [c for c in _final_user_content(cap) if c["type"] == "image_url"]
    assert len(images) <= 2


def test_chat_multi_includes_prior_turns():
    az = _analyzer()
    cap = _capturing_stream()
    az._client.chat.completions.create = cap
    _drain(az.chat_multi({"cam0": [b"a"]}, "q", history=[], analytics=None,
                         prior_turns=[{"role": "user", "content": "earlier"}]))
    msgs = cap.calls["kwargs"]["messages"]
    assert {"role": "user", "content": "earlier"} in msgs


def test_tools_enabled_defaults_to_cloud():
    cloud = Analyzer({"api": {"backend": "cloud", "api_key": "x", "model": "m"},
                      "monitoring": {}, "sampling": {}, "chat": {}})
    local = Analyzer({"api": {"backend": "local"}, "monitoring": {}, "sampling": {}, "chat": {}})
    assert cloud._tools_enabled is True
    assert local._tools_enabled is False


def test_tools_enabled_explicit_override():
    a = Analyzer({"api": {"backend": "local"}, "monitoring": {}, "sampling": {},
                  "chat": {"tools_enabled": True, "max_tool_calls": 5}})
    assert a._tools_enabled is True
    assert a._max_tool_calls == 5


def test_chat_multi_adds_timestamp_range_header_when_offsets_given():
    az = _analyzer()
    cap = _capturing_stream("ok")
    az._client.chat.completions.create = cap
    frames = {"cam0": [b"a", b"b"]}
    offsets = {"cam0": [310.0, 330.0]}  # 00:05:10 to 00:05:30 into the footage
    _drain(az.chat_multi(frames, "when?", history=[], analytics=None, prior_turns=[],
                         per_camera_offsets=offsets))
    content = _final_user_content(cap)
    texts = [c["text"] for c in content if c["type"] == "text"]
    assert "=== Camera cam0 @ 00:05:10 to 00:05:30 ===" in texts
    # single header for the round, then the two frames, then the question
    kinds = [c["type"] for c in content]
    assert kinds == ["text", "image_url", "image_url", "text"]
    assert texts[-1] == "when?"


def test_chat_multi_timestamp_header_collapses_single_frame_range():
    az = _analyzer()
    cap = _capturing_stream("ok")
    az._client.chat.completions.create = cap
    _drain(az.chat_multi({"cam0": [b"a"]}, "when?", history=[], analytics=None,
                         prior_turns=[], per_camera_offsets={"cam0": [310.0]}))
    texts = [c["text"] for c in _final_user_content(cap) if c["type"] == "text"]
    assert "=== Camera cam0 @ 00:05:10 ===" in texts  # lo == hi -> single time


def test_chat_multi_omits_timestamps_without_offsets():
    az = _analyzer()
    cap = _capturing_stream("ok")
    az._client.chat.completions.create = cap
    _drain(az.chat_multi({"cam0": [b"a"]}, "q", history=[], analytics=None, prior_turns=[]))
    texts = [c["text"] for c in _final_user_content(cap) if c["type"] == "text"]
    assert texts == ["q"]  # only the question; no timestamp blocks


def test_search_mode_drops_analytics_block():
    az = _analyzer()
    cap = _capturing_stream("ok")
    az._client.chat.completions.create = cap
    analytics = {"total_alerts": 7, "total_comments": 2, "alerts_today": 1, "alerts_24h": 7,
                 "per_camera": [], "top_tags": [],
                 "most_recent_alert": {"camera": "cam0", "timestamp": "2026-06-24T12:45:36Z",
                                       "message": "runner"}}
    _drain(az.chat_multi({"cam0": [b"a"]}, "where is the car?", history=[], analytics=analytics,
                         prior_turns=[], sweep=True))
    sys = cap.calls["kwargs"]["messages"][0]["content"]
    assert "12:45:36" not in sys      # the confusing wall-clock time is gone
    assert "7 alerts" not in sys      # analytics counts dropped in search mode
    assert "No events recorded yet." in sys


def test_chat_prompt_excludes_recent_event_log_keeps_analytics():
    az = Analyzer({"api": {"backend": "cloud", "api_key": "x", "model": "m"},
                   "monitoring": {}, "sampling": {}, "chat": {}})
    history = [{"event_type": "alert", "message": "RED_LIGHT_RUNNER_XYZ", "camera": "cam1",
                "timestamp": "2026-06-23T10:00:00Z"}]
    analytics = {"total_alerts": 3, "total_comments": 1, "alerts_today": 2, "alerts_24h": 3,
                 "per_camera": [], "most_recent_alert": None, "top_tags": []}
    msgs = az._build_chat_messages({"cam0": [b"a"]}, "hi", history=history,
                                   analytics=analytics, prior_turns=[], max_images=12)
    sys = msgs[0]["content"]
    assert "RED_LIGHT_RUNNER_XYZ" not in sys   # recent events are not injected (may be stale)
    assert "RECENT EVENT LOG" not in sys
    assert "STREAM ANALYTICS" in sys           # aggregate analytics retained
    assert "3 alerts" in sys


def test_build_chat_messages_shape():
    az = Analyzer({"api": {"backend": "cloud", "api_key": "x", "model": "m"},
                   "monitoring": {}, "sampling": {}, "chat": {}})
    msgs = az._build_chat_messages({"cam0": [b"a"]}, "hello", history=[], analytics=None,
                                   prior_turns=[{"role": "user", "content": "earlier"}], max_images=12)
    assert msgs[0]["role"] == "system"
    assert {"role": "user", "content": "earlier"} in msgs
    user = msgs[-1]
    assert user["role"] == "user"
    assert user["content"][-1] == {"type": "text", "text": "hello"}


def _capturing_completion(text="A person climbed the fence."):
    """Fake non-streaming create() that records kwargs and returns a plain response."""
    calls = {}

    async def create(*a, **k):
        calls["kwargs"] = k
        msg = SimpleNamespace(content=text)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])
    create.calls = calls
    return create


def test_describe_intrusion_scopes_prompt_to_camera_and_returns_text():
    az = _analyzer()
    cap = _capturing_completion("A person in a dark jacket climbed the fence.")
    az._client.chat.completions.create = cap

    result = asyncio.run(az.describe_alert("intrusion", [b"frame0", b"frame1"], "cam3"))

    assert result == "A person in a dark jacket climbed the fence."
    system_prompt = cap.calls["kwargs"]["messages"][0]["content"]
    assert "cam3" in system_prompt
    assert "restricted zone" in system_prompt
    assert "ignore" in system_prompt.lower() or "not part of this intrusion" in system_prompt.lower()
    user_content = cap.calls["kwargs"]["messages"][1]["content"]
    images = [c for c in user_content if c["type"] == "image_url"]
    assert len(images) == 2


def test_describe_intrusion_strips_think_blocks():
    az = _analyzer()
    az._client.chat.completions.create = _capturing_completion(
        "<think>reasoning</think>A person entered the zone.")

    result = asyncio.run(az.describe_alert("intrusion", [b"frame0"], "cam1"))

    assert result == "A person entered the zone."


def test_describe_intrusion_empty_frames_returns_empty_without_calling_model():
    az = _analyzer()
    cap = _capturing_completion()
    az._client.chat.completions.create = cap

    result = asyncio.run(az.describe_alert("intrusion", [], "cam1"))

    assert result == ""
    assert "kwargs" not in cap.calls


def test_describe_intrusion_prompt_explains_zone_outline():
    az = _analyzer()
    cap = _capturing_completion("A person entered.")
    az._client.chat.completions.create = cap

    asyncio.run(az.describe_alert("intrusion", [b"frame0"], "cam1"))

    system_prompt = cap.calls["kwargs"]["messages"][0]["content"]
    assert "red polygon" in system_prompt.lower()
    assert "restricted zone" in system_prompt.lower()


def test_describe_overcrowding_scopes_prompt_to_camera_and_count():
    az = _analyzer()
    cap = _capturing_completion("A dense group of people fills the marked area.")
    az._client.chat.completions.create = cap

    result = asyncio.run(az.describe_alert("overcrowding", [b"frame0", b"frame1"], "cam3", 8))

    assert result == "A dense group of people fills the marked area."
    system_prompt = cap.calls["kwargs"]["messages"][0]["content"]
    assert "cam3" in system_prompt
    assert "8" in system_prompt
    images = [c for c in cap.calls["kwargs"]["messages"][1]["content"] if c["type"] == "image_url"]
    assert len(images) == 2


def test_describe_overcrowding_strips_think_blocks():
    az = _analyzer()
    az._client.chat.completions.create = _capturing_completion(
        "<think>reasoning</think>A crowd has gathered.")

    result = asyncio.run(az.describe_alert("overcrowding", [b"frame0"], "cam1", 6))

    assert result == "A crowd has gathered."


def test_describe_overcrowding_empty_frames_returns_empty_without_calling_model():
    az = _analyzer()
    cap = _capturing_completion()
    az._client.chat.completions.create = cap

    result = asyncio.run(az.describe_alert("overcrowding", [], "cam1", 6))

    assert result == ""
    assert "kwargs" not in cap.calls


def test_describe_overcrowding_prompt_explains_zone_outline_and_does_not_mention_intrusion():
    az = _analyzer()
    cap = _capturing_completion("A crowd has gathered.")
    az._client.chat.completions.create = cap

    asyncio.run(az.describe_alert("overcrowding", [b"frame0"], "cam1", 6))

    system_prompt = cap.calls["kwargs"]["messages"][0]["content"]
    assert "red polygon" in system_prompt.lower()
    assert "overcrowding" in system_prompt.lower()
    assert "intrusion" not in system_prompt.lower()


def test_describe_overcrowding_handles_missing_count():
    az = _analyzer()
    cap = _capturing_completion("A crowd has gathered.")
    az._client.chat.completions.create = cap

    result = asyncio.run(az.describe_alert("overcrowding", [b"frame0"], "cam1", None))

    assert result == "A crowd has gathered."
    system_prompt = cap.calls["kwargs"]["messages"][0]["content"]
    assert "None" not in system_prompt
