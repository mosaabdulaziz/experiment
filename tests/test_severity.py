"""Severity/category vocabulary.

Severity is primary and event_type is derived from it, so these coercions decide
what every downstream consumer sees. The clamp direction is a safety property:
a bad value must never become `critical` and take over an operator's screen.
"""
from app.severity import (CATEGORIES, SEVERITIES, derive_event_type,
                          normalise_category, normalise_severity,
                          zone_alert_defaults)


def test_closed_sets_are_exactly_the_spec_values():
    assert SEVERITIES == ("critical", "warning", "info")
    assert CATEGORIES == ("safety", "security", "storage", "operations")


def test_info_derives_a_comment_and_everything_else_an_alert():
    assert derive_event_type("info") == "comment"
    assert derive_event_type("warning") == "alert"
    assert derive_event_type("critical") == "alert"


def test_valid_severities_pass_through():
    for s in SEVERITIES:
        assert normalise_severity(s) == s


def test_severity_is_case_and_whitespace_insensitive():
    assert normalise_severity("  CRITICAL ") == "critical"


def test_invalid_severity_without_alert_hint_falls_to_info():
    assert normalise_severity("catastrophic") == "info"
    assert normalise_severity(None) == "info"
    assert normalise_severity(7) == "info"


def test_invalid_severity_with_alert_hint_clamps_to_warning_not_critical():
    """The safety property: nothing reaches `critical` by accident."""
    assert normalise_severity("catastrophic", alert_hint=True) == "warning"
    assert normalise_severity(None, alert_hint=True) == "warning"
    assert normalise_severity("", alert_hint=True) == "warning"


def test_valid_categories_pass_through_and_invalid_become_none():
    for c in CATEGORIES:
        assert normalise_category(c) == c
    assert normalise_category("Safety") == "safety"
    assert normalise_category("logistics") is None
    assert normalise_category(None) is None


def test_zone_alert_defaults_per_behaviour():
    assert zone_alert_defaults("intrusion") == ("warning", "security")
    assert zone_alert_defaults("exit") == ("warning", "security")
    assert zone_alert_defaults("overcrowding") == ("warning", "safety")


def test_unknown_zone_kind_falls_back_without_raising():
    assert zone_alert_defaults("dwell") == ("warning", "security")
    assert zone_alert_defaults(None) == ("warning", "security")
