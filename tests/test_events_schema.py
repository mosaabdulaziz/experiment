"""The alert schema's new columns: severity, category, location, recommended_action.

Severity is primary and event_type is derived, but the store persists whatever it
is handed -- coercion lives in the analyzer. These tests pin persistence and the
back-compat bridge for callers that pass no severity at all.
"""
import asyncio

from app.events import EventStore


def _store(tmp_path):
    return EventStore({"events": {"db_path": str(tmp_path / "e.db"), "retention_hours": 48}})


def test_save_persists_all_new_fields(tmp_path):
    store = _store(tmp_path)

    async def run():
        await store.init()
        rec = await store.save(
            "alert", "Forklift reversing toward a worker", ["safety"],
            severity="critical", category="safety",
            location="by the bay door", recommended_action="Sound the horn",
            description="A forklift is reversing while a worker walks behind it.",
        )
        return rec, await store.get_recent(limit=10)

    rec, recent = asyncio.run(run())
    assert rec["severity"] == "critical"
    assert rec["category"] == "safety"
    assert rec["location"] == "by the bay door"
    assert rec["recommended_action"] == "Sound the horn"
    assert rec["description"] == "A forklift is reversing while a worker walks behind it."
    assert recent[0]["severity"] == "critical"
    assert recent[0]["category"] == "safety"
    assert recent[0]["location"] == "by the bay door"
    assert recent[0]["recommended_action"] == "Sound the horn"


def test_new_fields_default_to_none(tmp_path):
    store = _store(tmp_path)

    async def run():
        await store.init()
        return await store.save("comment", "Quiet yard", [])

    rec = asyncio.run(run())
    assert rec["category"] is None
    assert rec["location"] is None
    assert rec["recommended_action"] is None
    assert rec["description"] is None


def test_severity_is_back_derived_when_a_caller_omits_it(tmp_path):
    """Legacy call sites pass only event_type; the row must still be consistent."""
    store = _store(tmp_path)

    async def run():
        await store.init()
        comment = await store.save("comment", "Quiet yard", [])
        alert = await store.save("alert", "Person in restricted zone", [])
        return comment, alert, await store.get_recent(limit=10)

    comment, alert, recent = asyncio.run(run())
    assert comment["severity"] == "info"
    assert alert["severity"] == "warning"
    by_msg = {r["message"]: r for r in recent}
    assert by_msg["Quiet yard"]["severity"] == "info"
    assert by_msg["Person in restricted zone"]["severity"] == "warning"


def test_explicit_severity_wins_over_the_back_derivation(tmp_path):
    store = _store(tmp_path)

    async def run():
        await store.init()
        return await store.save("alert", "Fire in the north aisle", [], severity="critical")

    assert asyncio.run(run())["severity"] == "critical"


from app.events import save_kwargs_from_result
from app.severity import zone_alert_defaults


def test_save_kwargs_maps_every_schema_field():
    kw = save_kwargs_from_result({
        "type": "alert", "severity": "critical", "category": "safety",
        "message": "headline", "tags": [],
        "location": "by the bay door",
        "recommended_action": "Sound the horn",
        "description": "A forklift reverses while a worker crosses behind it.",
    })
    assert kw["severity"] == "critical"
    assert kw["category"] == "safety"
    assert kw["location"] == "by the bay door"
    assert kw["recommended_action"] == "Sound the horn"
    assert kw["description"].startswith("A forklift reverses")


def test_description_status_is_ready_only_when_a_description_exists():
    """A proactive alert reasons inline, so nothing resolves it later -- but an
    event with no description must not claim a status it will never fulfil."""
    with_desc = save_kwargs_from_result({"severity": "warning", "description": "why it matters"})
    without = save_kwargs_from_result({"severity": "warning"})
    assert with_desc["description_status"] == "ready"
    assert without["description_status"] is None


def test_minimal_result_maps_to_all_none():
    kw = save_kwargs_from_result({"severity": "info", "message": "x"})
    assert kw["category"] is None
    assert kw["location"] is None
    assert kw["recommended_action"] is None
    assert kw["description"] is None
    assert kw["description_status"] is None


def test_helper_never_raises_on_an_empty_dict():
    """Defensive: the emit loop must not die on an unexpected result shape."""
    kw = save_kwargs_from_result({})
    assert kw["severity"] is None
    assert kw["description_status"] is None


def test_zone_alert_defaults_cover_the_kinds_emit_uses():
    """_emit_zone_event maps prompt_kind -> (severity, category)."""
    assert zone_alert_defaults("intrusion") == ("warning", "security")
    assert zone_alert_defaults("overcrowding") == ("warning", "safety")
