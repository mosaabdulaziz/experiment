import asyncio
import logging
from types import SimpleNamespace
from app.analyzer import Analyzer
from app.metrics import MetricsRegistry


def _analyzer(metrics=None):
    return Analyzer({"api": {"backend": "cloud", "api_key": "x", "model": "m"},
                     "monitoring": {}, "sampling": {}, "chat": {}}, metrics=metrics)


def test_analyze_logs_source_and_duration_and_records_metric(caplog):
    az = _analyzer(metrics=MetricsRegistry())

    async def create(*a, **k):
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="[]"))])
    az._client.chat.completions.create = create

    with caplog.at_level(logging.INFO, logger="iris.analyzer"):
        asyncio.run(az.analyze([b"a", b"b"], history=[], source_id="cam1"))

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "[cam1] analyze start" in text
    assert "[cam1] analyze done" in text
    # a metric row now exists for cam1
    assert any(row.stream == "cam1" for row in az._metrics.rows(now=1e9))


def _capturing_completion(text="[]"):
    calls = {}

    async def create(*a, **k):
        calls["kwargs"] = k
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])
    create.calls = calls
    return create


def test_analyze_prepends_current_clip_line_when_offsets_given():
    """I5: the model has no clock otherwise — history rows print [HH:MM:SS-HH:MM:SS]
    with nothing to compare against when judging whether a condition is still present."""
    az = _analyzer()
    cap = _capturing_completion()
    az._client.chat.completions.create = cap

    asyncio.run(az.analyze([b"a", b"b"], history=[], offset_start=116.0, offset_end=119.0))

    user_content = cap.calls["kwargs"]["messages"][1]["content"]
    prompt_text = next(c["text"] for c in user_content if c["type"] == "text")
    assert prompt_text.startswith("Current clip: [00:01:56-00:01:59].\n")


def test_analyze_omits_current_clip_line_without_offsets():
    az = _analyzer()
    cap = _capturing_completion()
    az._client.chat.completions.create = cap

    asyncio.run(az.analyze([b"a", b"b"], history=[]))

    user_content = cap.calls["kwargs"]["messages"][1]["content"]
    prompt_text = next(c["text"] for c in user_content if c["type"] == "text")
    assert "Current clip:" not in prompt_text
