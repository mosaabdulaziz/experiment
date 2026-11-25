"""Intrusion and overcrowding as incidents with an end (safety integration, ticket 2, decision D10/D36).

When the zones module sends end markers, the first entry into a zone opens an incident, later entries raise the same
incident in place, and `intrusion_cleared` resolves it; overcrowding opens on the alert and resolves on
`overcrowding_resolved`. Without end markers every entry stays its own alert (iris, multicam)."""
import asyncio
import types

from app.usecases import count_alerts, zone_alerts
from app.usecases.zone_incidents import ZoneIncidents


class _Alerts:
    def __init__(self):
        self.calls = []
        self.gone = set()           # event ids whose row was deleted

    async def emit(self, sid, message, tags, pts, describe, prompt_kind, count=None, **kw):
        self.calls.append(("emit", sid, tags, kw.get("zone_id"), kw.get("incident_started_at")))
        return len(self.calls)

    async def escalate(self, event_id, sid, pts, severity, message, details, zone_id=None, kind=None):
        self.calls.append(("escalate", event_id, details))
        return event_id not in self.gone

    async def resolve_incident(self, event_id, sid, duration_s, details):
        self.calls.append(("resolve", event_id, duration_s))


_DESCRIPTIONS = types.SimpleNamespace(describe_intrusions=False, describe_overcrowding=False, describe_fire=False)


def _run_zone(monkeypatch, entries, incidents):
    async def fake_read_stream(_redis, _key, _running):
        for entry in entries:
            yield entry
    monkeypatch.setattr(zone_alerts, "read_stream", fake_read_stream)
    alerts = _Alerts()
    reader = zone_alerts.ZoneAlerts({}, None, alerts, _DESCRIPTIONS, lambda: True,
                                    incidents=ZoneIncidents(alerts) if incidents else None)
    asyncio.run(reader.run_for("bay-a@zones"))
    return alerts.calls


def _entry(zone, oid):
    return {"zone": zone, "pts": "1.0", "object_id": str(oid), "object_label": "person"}


def test_entries_into_one_zone_are_one_incident_until_it_clears(monkeypatch):
    calls = _run_zone(monkeypatch, [_entry("z0", 1), _entry("z0", 2),
                                    {"kind": "intrusion_cleared", "zone": "z0", "duration": "6.5"}], incidents=True)
    (opened, again, ended) = calls
    assert opened[0] == "emit" and opened[3] == "z0" and opened[4] is not None     # incident_started_at
    assert again[:2] == ("escalate", 1) and again[2]["entries"] == 2
    assert ended == ("resolve", 1, 6.5)


def test_each_zone_has_its_own_incident(monkeypatch):
    calls = _run_zone(monkeypatch, [_entry("z0", 1), _entry("z1", 2)], incidents=True)
    assert [c[0] for c in calls] == ["emit", "emit"]


def test_after_a_clear_the_next_entry_opens_a_new_incident(monkeypatch):
    calls = _run_zone(monkeypatch, [_entry("z0", 1), {"kind": "intrusion_cleared", "zone": "z0", "duration": "3"},
                                    _entry("z0", 2)], incidents=True)
    assert [c[0] for c in calls] == ["emit", "resolve", "emit"]


def test_without_end_markers_every_entry_is_its_own_alert(monkeypatch):
    calls = _run_zone(monkeypatch, [_entry("z0", 1), _entry("z0", 2)], incidents=False)
    assert [(c[0], c[4]) for c in calls] == [("emit", None), ("emit", None)]


def test_overcrowding_is_an_incident_until_it_resolves(monkeypatch):
    async def fake_read_stream(_redis, _key, _running):
        for entry in [{"zone": "z0", "pts": "1.0", "count": "6"},
                      {"kind": "overcrowding_resolved", "zone": "z0", "pts": "9.0", "count": "2", "duration": "8"}]:
            yield entry
    monkeypatch.setattr(count_alerts, "read_stream", fake_read_stream)
    alerts = _Alerts()
    reader = count_alerts.CountAlerts({}, None, alerts, _DESCRIPTIONS, fire=None, running=lambda: True,
                                      incidents=ZoneIncidents(alerts))
    asyncio.run(reader.run_for("bay-a@zones"))
    (opened, ended) = alerts.calls
    assert opened[0] == "emit" and opened[2] == ["zone_overcrowding"] and opened[4] is not None
    assert ended == ("resolve", 1, 8.0)


def test_a_zone_incident_whose_row_was_deleted_reopens_on_the_next_entry():
    alerts, sid = _Alerts(), "bay-a@zones"
    alerts.gone.add(1)                              # incident 1's row was deleted (the console, or retention)
    incidents = ZoneIncidents(alerts)

    async def emit(started_at):
        return await alerts.emit(sid, "m", ["intrusion"], 1.0, False, "intrusion", zone_id="z0",
                                 incident_started_at=started_at)

    async def run():
        for _ in range(3):
            await incidents.alert(sid, "intrusion", "z0", 1.0, "intrusion", emit, lambda n: f"again ({n})")
    asyncio.run(run())
    assert [c[:2] for c in alerts.calls] == [("emit", sid), ("escalate", 1), ("emit", sid), ("escalate", 3)]
    assert incidents.event_id(sid, "intrusion", "z0") == 3


def test_a_sidecar_restart_forgets_the_camera_s_open_incidents():
    alerts = _Alerts()
    incidents = ZoneIncidents(alerts)
    incidents.opened("bay-a@zones", "intrusion", "z0", 7)
    incidents.opened("bay-b@zones", "intrusion", "z0", 8)
    asyncio.run(incidents.sidecar_restarted("bay-a@zones"))
    assert incidents.event_id("bay-a@zones", "intrusion", "z0") is None
    assert incidents.event_id("bay-b@zones", "intrusion", "z0") == 8
