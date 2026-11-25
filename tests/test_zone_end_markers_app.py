"""The app does not mistake the zones module's end markers for new alerts (safety integration, ticket 1).

`intrusion_cleared` (iris:zone_alert) and `overcrowding_resolved` (iris:overcrowd_alert) say an incident ENDED;
until the incident model consumes them (ticket 2) they must never become a fresh intrusion/overcrowding event."""
import asyncio
import types

from app.usecases import count_alerts, zone_alerts


class _Alerts:
    def __init__(self):
        self.emitted = []

    async def emit(self, sid, message, tags, *args, **kwargs):
        self.emitted.append((sid, message, tags))
        return len(self.emitted)


def _feed(monkeypatch, module, entries):
    async def fake_read_stream(_redis, _key, _running):
        for entry in entries:
            yield entry
    monkeypatch.setattr(module, "read_stream", fake_read_stream)


_DESCRIPTIONS = types.SimpleNamespace(describe_intrusions=False, describe_overcrowding=False, describe_fire=False)


def test_intrusion_cleared_is_not_a_new_intrusion(monkeypatch):
    _feed(monkeypatch, zone_alerts, [
        {"zone": "z0", "pts": "1.0", "object_id": "7", "object_label": "person"},
        {"kind": "intrusion_cleared", "zone": "z0", "pts": "9.0", "duration": "6.0"},
    ])
    alerts = _Alerts()
    reader = zone_alerts.ZoneAlerts({}, None, alerts, _DESCRIPTIONS, lambda: True)
    asyncio.run(reader.run_for("cam-1"))
    assert [tags for _sid, _msg, tags in alerts.emitted] == [["zone_intrusion"]]


def test_overcrowding_resolved_is_not_a_new_overcrowding(monkeypatch):
    _feed(monkeypatch, count_alerts, [
        {"zone": "z0", "pts": "1.0", "count": "6"},
        {"kind": "overcrowding_resolved", "zone": "z0", "pts": "9.0", "count": "2", "duration": "6.0"},
    ])
    alerts = _Alerts()
    reader = count_alerts.CountAlerts({}, None, alerts, _DESCRIPTIONS, fire=None, running=lambda: True)
    asyncio.run(reader.run_for("cam-1"))
    assert [tags for _sid, _msg, tags in alerts.emitted] == [["zone_overcrowding"]]
