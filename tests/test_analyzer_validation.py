"""Analyzer._validate_event: the gate between model output and the event store.

Severity is primary and event_type is derived. The clamp direction is a safety
property -- a malformed entry must never arrive as `critical`.
"""
from app.analyzer import Analyzer


def v(item):
    return Analyzer._validate_event(item)


def test_severity_drives_the_derived_event_type():
    assert v({"severity": "info", "message": "Quiet"})["type"] == "comment"
    assert v({"severity": "warning", "message": "Blocked route"})["type"] == "alert"
    assert v({"severity": "critical", "message": "Person down"})["type"] == "alert"


def test_legacy_entry_with_only_a_type_still_validates():
    """An older prompt emits type but no severity; it must keep working."""
    out = v({"type": "alert", "message": "Person in restricted zone"})
    assert out["type"] == "alert"
    assert out["severity"] == "warning"
    out = v({"type": "comment", "message": "Quiet yard"})
    assert out["type"] == "comment"
    assert out["severity"] == "info"


def test_bad_severity_on_an_alert_clamps_to_warning_never_critical():
    out = v({"type": "alert", "severity": "catastrophic", "message": "Smoke"})
    assert out["severity"] == "warning"
    assert out["type"] == "alert"


def test_bad_severity_with_no_alert_signal_falls_to_info():
    out = v({"severity": "urgent!", "message": "Something"})
    assert out["severity"] == "info"
    assert out["type"] == "comment"


def test_entry_with_neither_type_nor_severity_resolves_to_info_comment():
    """A missing severity is never a reason to drop a real observation -- it clamps
    to "info", it does not vanish (spec: 2026-08-30-cognitive-monitoring-design.md §4)."""
    out = v({"message": "No signal at all"})
    assert out is not None
    assert out["severity"] == "info"
    assert out["type"] == "comment"


def test_missing_severity_with_category_still_validates_as_info():
    out = v({"message": "A worker has collapsed near Bay 2", "category": "safety"})
    assert out is not None
    assert out["severity"] == "info"
    assert out["type"] == "comment"
    assert out["category"] == "safety"


def test_explicit_null_severity_resolves_to_info_not_dropped():
    out = v({"severity": None, "message": "Fire in the north aisle"})
    assert out is not None
    assert out["severity"] == "info"
    assert out["type"] == "comment"


def test_unparseable_severity_still_validates_and_logs_a_warning(caplog):
    with caplog.at_level("WARNING", logger="iris.analyzer"):
        out = v({"severity": "high", "message": "Fire in the north aisle"})
    assert out is not None
    assert out["severity"] == "info"
    assert any("high" in record.getMessage() for record in caplog.records)


def test_severity_overrides_a_contradicting_legacy_type_toward_comment():
    """Severity is primary: a contradicting legacy `type` never wins."""
    out = v({"type": "alert", "severity": "info", "message": "x"})
    assert out["type"] == "comment"


def test_severity_overrides_a_contradicting_legacy_type_toward_alert():
    out = v({"type": "comment", "severity": "critical", "message": "x"})
    assert out["type"] == "alert"
    assert out["severity"] == "critical"


def test_entry_with_no_message_is_dropped():
    assert v({"severity": "critical", "message": "   "}) is None
    assert v({"severity": "critical"}) is None


def test_non_dict_is_dropped():
    assert v("nope") is None
    assert v(None) is None


def test_optional_fields_are_carried_through():
    out = v({
        "severity": "critical", "category": "safety",
        "message": "Forklift reversing toward a worker",
        "tags": ["forklift"],
        "location": "by the bay door",
        "recommended_action": "Sound the horn",
        "description": "A forklift reverses while a worker walks behind it.",
    })
    assert out["category"] == "safety"
    assert out["location"] == "by the bay door"
    assert out["recommended_action"] == "Sound the horn"
    assert out["description"].startswith("A forklift reverses")


def test_unknown_category_becomes_none():
    assert v({"severity": "info", "message": "x", "category": "logistics"})["category"] is None


def test_blank_optional_fields_become_none_not_empty_strings():
    out = v({"severity": "info", "message": "x", "location": "  ",
             "recommended_action": "", "description": None})
    assert out["location"] is None
    assert out["recommended_action"] is None
    assert out["description"] is None


def test_non_list_tags_become_an_empty_list():
    assert v({"severity": "info", "message": "x", "tags": "forklift"})["tags"] == []
