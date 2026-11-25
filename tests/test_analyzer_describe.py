"""describe_alert: one prompt-kind-dispatched description entry point.

Replaces describe_intrusion / describe_exit / describe_overcrowding, which were
near-identical. The system prompt is built without any network call, so these tests
assert on the prompt text rather than on a model response."""

import pytest

from app.analyzer import _ALERT_SYSTEM_TEMPLATES


def test_every_prompt_kind_has_a_template():
    assert set(_ALERT_SYSTEM_TEMPLATES) == {"intrusion", "exit", "overcrowding", "fire_smoke"}


def test_fire_template_names_the_common_false_positives():
    tpl = " ".join(_ALERT_SYSTEM_TEMPLATES["fire_smoke"].lower().split())
    for confuser in ("steam", "vapour", "exhaust", "dust", "weld", "headlight"):
        assert confuser in tpl, f"fire prompt should ask the model to rule out {confuser}"
    # Listing the confusers is not enough on its own -- the prompt must actually put
    # the model in an adjudicating, not merely descriptive, stance toward them.
    assert "adjudicate" in tpl, "fire prompt should frame this as adjudication"
    assert "not to embellish" in tpl


def test_fire_template_permits_a_negative_verdict():
    # Normalize whitespace: the template line-wraps, so a phrase spanning a wrap
    # (e.g. "say\nthat clearly") must not evade this check by mere word-wrap luck.
    tpl = " ".join(_ALERT_SYSTEM_TEMPLATES["fire_smoke"].lower().split())
    # The load-bearing instruction: explicit permission (and example phrasing) to
    # report that this is NOT fire, not just an incidental "not" somewhere in the text.
    assert "this is not fire" in tpl
    assert "say that clearly" in tpl
    assert "{source_id}" in _ALERT_SYSTEM_TEMPLATES["fire_smoke"]


def test_only_overcrowding_template_takes_a_count():
    assert "{count_clause}" in _ALERT_SYSTEM_TEMPLATES["overcrowding"]
    for kind in ("intrusion", "exit", "fire_smoke"):
        assert "{count_clause}" not in _ALERT_SYSTEM_TEMPLATES[kind]


def test_unknown_prompt_kind_raises():
    import asyncio

    from app.analyzer import Analyzer

    analyzer = Analyzer.__new__(Analyzer)   # no network, no config needed for the guard
    with pytest.raises(ValueError, match="prompt_kind"):
        asyncio.run(analyzer.describe_alert("nonsense", [b"x"], "cam-1"))


def test_empty_frames_returns_empty_string_without_dispatch():
    import asyncio

    from app.analyzer import Analyzer

    analyzer = Analyzer.__new__(Analyzer)
    assert asyncio.run(analyzer.describe_alert("fire_smoke", [], "cam-1")) == ""
