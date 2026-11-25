"""Fire alert message dispatch, sampler gating, and sidecar-restart reconciliation.

The sampler gate is the silent-death case: the sidecar XADDs fire alerts and, if the
gate is wrong, nothing ever reads them while every component reports healthy."""

import asyncio
import json

import fakeredis

from app.stream_manager import StreamManager
from app.usecases.count_alerts import count_alert_message, fire_sampler_needed


def test_fire_kind_produces_fire_copy():
    message, tags, prompt_kind = count_alert_message("cam-1", None, kind="fire_smoke",
                                                     cls="fire", count=2)
    assert message == "Fire detected on cam-1"
    assert tags == ["fire_smoke"]
    assert prompt_kind == "fire_smoke"


def test_smoke_kind_produces_smoke_copy():
    message, tags, prompt_kind = count_alert_message("cam-1", None, kind="fire_smoke",
                                                     cls="smoke", count=1)
    assert message == "Smoke detected on cam-1"
    assert prompt_kind == "fire_smoke"


def test_absent_kind_falls_back_to_overcrowding():
    # Version skew: an app newer than its sidecar. Reporting a fire as overcrowding
    # would be wrong but visible; dropping it silently would not be.
    message, tags, prompt_kind = count_alert_message("cam-1", None, kind=None,
                                                     cls=None, count=6)
    assert "Overcrowding" in message
    assert tags == ["zone_overcrowding"]
    assert prompt_kind == "overcrowding"


def test_overcrowding_names_the_zone_when_present():
    zone = {"name": "Lobby"}
    message, _tags, _kind = count_alert_message("cam-1", zone, kind="overcrowding",
                                               cls=None, count=7)
    assert "Lobby" in message and "7" in message


def test_sampler_needed_when_only_analytics_enabled():
    assert fire_sampler_needed(analytics_enabled=True, fire_alert_enabled=False) is True


def test_sampler_needed_when_only_fire_enabled():
    # THE gate that matters: fire needs no ROI, so it must not require analytics.
    assert fire_sampler_needed(analytics_enabled=False, fire_alert_enabled=True) is True


def test_sampler_not_needed_when_neither_enabled():
    assert fire_sampler_needed(analytics_enabled=False, fire_alert_enabled=False) is False


# ── sidecar-restart reconciliation ──────────────────────────────────────────
# A crash-looping fire-role sidecar container restarts far more often than the
# app itself does, wiping that process's in-memory FireAlertState mid-incident.
# fire.check_sidecar_boot_id/reconcile_sidecar_restart is what stops the DB row
# that incident opened from staying "ongoing" forever (see
# EventStore.resolve_stale_incidents_for_camera).

def _cfg():
    return {
        "sampling": {"frames_per_batch": 4, "interval_seconds": 4},
        "monitoring": {"cooldown_seconds": 8, "alert_cooldown_seconds": 4, "max_history": 3},
        "dedup": {"enabled": False, "backend": "local"},
        "perception": {"redis_url": "redis://unused", "queue_depth": 3,
                       "discovery_interval_seconds": 5},
        "zones": {"sidecar_config_path": "/nonexistent/sidecar.yaml"},
    }


class _StubAnalyzer:
    async def analyze(self, *args, **kwargs):
        return []


class _StubConn:
    def __init__(self):
        self.events = []

    async def broadcast(self, payload):
        self.events.append(payload)


def test_check_sidecar_boot_id_first_sighting_does_not_reconcile(tmp_path):
    from app.events import EventStore
    r = fakeredis.FakeStrictRedis()
    conn = _StubConn()
    store = EventStore({"events": {"db_path": str(tmp_path / "e.db"), "retention_hours": 48}})
    sm = StreamManager(_cfg(), analyzer=_StubAnalyzer(), store=store, conn_manager=conn, redis_client=r)
    r.set("iris:status:cam0", json.dumps({"running": True, "boot_id": "boot-a"}))

    async def run():
        await store.init()
        rec = await store.save("alert", "Fire detected on cam0", ["fire_smoke"], camera="cam0",
                               incident_status="ongoing", incident_started_at="2026-09-04T10:00:00Z")
        sm.fire.open_incidents[("cam0", "fire")] = rec["id"]
        await sm.fire.check_sidecar_boot_id("cam0")
        return rec["id"]

    event_id = asyncio.run(run())
    assert sm.fire.sidecar_boot_ids["cam0"] == "boot-a"
    assert sm.fire.open_incidents.get(("cam0", "fire")) == event_id   # untouched
    assert conn.events == []                                          # no reconcile broadcast


def test_check_sidecar_boot_id_reconciles_on_change(tmp_path):
    from app.events import EventStore
    r = fakeredis.FakeStrictRedis()
    conn = _StubConn()
    store = EventStore({"events": {"db_path": str(tmp_path / "e.db"), "retention_hours": 48}})
    sm = StreamManager(_cfg(), analyzer=_StubAnalyzer(), store=store, conn_manager=conn, redis_client=r)

    async def run():
        await store.init()
        rec = await store.save("alert", "Fire detected on cam0", ["fire_smoke"], camera="cam0",
                               incident_status="ongoing", incident_started_at="2026-09-04T10:00:00Z")
        sm.fire.open_incidents[("cam0", "fire")] = rec["id"]
        r.set("iris:status:cam0", json.dumps({"running": True, "boot_id": "boot-a"}))
        await sm.fire.check_sidecar_boot_id("cam0")           # first sighting -> just records it
        r.set("iris:status:cam0", json.dumps({"running": True, "boot_id": "boot-b"}))
        await sm.fire.check_sidecar_boot_id("cam0")           # owning sidecar restarted mid-incident
        rows = {row["id"]: row for row in await store.get_recent(limit=10)}
        return rec["id"], rows

    event_id, rows = asyncio.run(run())
    assert sm.fire.sidecar_boot_ids["cam0"] == "boot-b"
    assert ("cam0", "fire") not in sm.fire.open_incidents
    assert rows[event_id]["incident_status"] == "unknown"
    assert conn.events == [{"type": "event_update", "id": event_id, "incident_status": "unknown"}]


def test_check_sidecar_boot_id_only_reconciles_the_restarted_camera(tmp_path):
    from app.events import EventStore
    r = fakeredis.FakeStrictRedis()
    conn = _StubConn()
    store = EventStore({"events": {"db_path": str(tmp_path / "e.db"), "retention_hours": 48}})
    sm = StreamManager(_cfg(), analyzer=_StubAnalyzer(), store=store, conn_manager=conn, redis_client=r)

    async def run():
        await store.init()
        rec = await store.save("alert", "Fire detected on cam1", ["fire_smoke"], camera="cam1",
                               incident_status="ongoing", incident_started_at="2026-09-04T10:00:00Z")
        sm.fire.open_incidents[("cam1", "fire")] = rec["id"]
        r.set("iris:status:cam0", json.dumps({"running": True, "boot_id": "boot-a"}))
        await sm.fire.check_sidecar_boot_id("cam0")
        r.set("iris:status:cam0", json.dumps({"running": True, "boot_id": "boot-b"}))
        await sm.fire.check_sidecar_boot_id("cam0")            # cam0's sidecar restarted, not cam1's
        rows = {row["id"]: row for row in await store.get_recent(limit=10)}
        return rec["id"], rows

    event_id, rows = asyncio.run(run())
    assert ("cam1", "fire") in sm.fire.open_incidents      # untouched: a different camera
    assert rows[event_id]["incident_status"] == "ongoing"
    assert conn.events == []


def test_check_sidecar_boot_id_unreadable_status_is_a_noop(tmp_path):
    # No iris:status:<id> key at all yet (a fresh camera) must not raise or record
    # a spurious boot_id that a later real one would then look like a "change" from.
    from app.events import EventStore
    r = fakeredis.FakeStrictRedis()
    store = EventStore({"events": {"db_path": str(tmp_path / "e.db"), "retention_hours": 48}})
    sm = StreamManager(_cfg(), analyzer=_StubAnalyzer(), store=store, conn_manager=_StubConn(),
                       redis_client=r)

    asyncio.run(sm.fire.check_sidecar_boot_id("cam0"))
    assert "cam0" not in sm.fire.sidecar_boot_ids


def _sm_with_ongoing(tmp_path, sid):
    from app.events import EventStore
    r = fakeredis.FakeStrictRedis()
    conn = _StubConn()
    store = EventStore({"events": {"db_path": str(tmp_path / "e.db"), "retention_hours": 48}})
    sm = StreamManager(_cfg(), analyzer=_StubAnalyzer(), store=store, conn_manager=conn, redis_client=r)
    return sm, store, conn, r


def test_a_feature_stopped_by_apply_closes_its_open_incidents(tmp_path):
    """Fire switched off on a camera during a fire: its role stream goes (the sidecar stopped, its status key with it),
    no "resolved" can ever come, so the incident is closed as "unknown" -- not left ongoing on the wall."""
    sm, store, conn, r = _sm_with_ongoing(tmp_path, "store@fire")

    async def run():
        await store.init()
        rec = await store.save("alert", "Fire detected", ["fire_smoke"], camera="store@fire",
                               incident_status="ongoing", incident_started_at="2026-11-10T10:00:00Z")
        sm.fire.open_incidents[("store@fire", "fire")] = rec["id"]
        await sm.fire.stream_gone("store@fire")
        return rec["id"], {row["id"]: row for row in await store.get_recent(limit=10)}

    event_id, rows = asyncio.run(run())
    assert rows[event_id]["incident_status"] == "unknown" and ("store@fire", "fire") not in sm.fire.open_incidents
    assert conn.events == [{"type": "event_update", "id": event_id, "incident_status": "unknown"}]


def test_a_stream_that_only_dropped_out_of_discovery_keeps_its_incidents(tmp_path):
    """Still publishing its status (a discovery hiccup), or a camera of another product (no role): nothing closes."""
    sm, store, conn, r = _sm_with_ongoing(tmp_path, "store@fire")
    r.set("iris:status:store@fire", json.dumps({"running": True, "boot_id": "b"}))

    async def run():
        await store.init()
        ids = []
        for sid in ("store@fire", "cam0"):
            rec = await store.save("alert", "Fire detected", ["fire_smoke"], camera=sid,
                                   incident_status="ongoing", incident_started_at="2026-11-10T10:00:00Z")
            ids.append(rec["id"])
            await sm.fire.stream_gone(sid)
        return ids, {row["id"]: row for row in await store.get_recent(limit=10)}

    ids, rows = asyncio.run(run())
    assert [rows[i]["incident_status"] for i in ids] == ["ongoing", "ongoing"] and conn.events == []


# ── an alert that arrives before discovery notices its sidecar restarted ─────────────────────────────────────────
# 2026-11-10 live demo: an Apply restarted the fire role; its first alert (the fire still burning) was saved one second
# before the discovery pass saw the new boot_id, and the reconcile then closed that NEW incident as "unknown". Every
# alert path checks the boot first, so a restart is reconciled before anything the new sidecar says is handled.

def _restarted(tmp_path):
    from app.events import EventStore
    r = fakeredis.FakeStrictRedis()
    store = EventStore({"events": {"db_path": str(tmp_path / "e.db"), "retention_hours": 48}})
    sm = StreamManager(_cfg(), analyzer=_StubAnalyzer(), store=store, conn_manager=_StubConn(), redis_client=r)
    return r, store, sm


async def _boot(r, sid, boot):
    r.set(f"iris:status:{sid}", json.dumps({"running": True, "boot_id": boot}))


def test_a_fire_alert_from_a_restarted_sidecar_reconciles_the_old_boot_first_and_keeps_its_own_incident(tmp_path):
    r, store, sm = _restarted(tmp_path)

    async def run():
        await store.init()
        old = await store.save("alert", "Fire detected on store@fire", ["fire_smoke"], camera="store@fire",
                               incident_status="ongoing", incident_started_at="2026-11-10T16:12:15Z")
        sm.fire.open_incidents[("store@fire", "fire")] = old["id"]
        await _boot(r, "store@fire", "boot-a")
        await sm.fire.check_sidecar_boot_id("store@fire")                  # discovery: first sighting
        await _boot(r, "store@fire", "boot-b")                         # an Apply restarted the role
        await sm.count_alerts.handle("store@fire", {"kind": "fire_smoke", "class": "fire", "pts": "3.0",
                                                    "count": "1", "onset_age": "2"})
        await sm.fire.check_sidecar_boot_id("store@fire")                  # discovery's next pass
        return old["id"], {row["id"]: row for row in await store.get_recent(limit=10)}

    old_id, rows = asyncio.run(run())
    (new_id,) = [i for i in rows if i != old_id]
    assert rows[old_id]["incident_status"] == "unknown"
    assert rows[new_id]["incident_status"] == "ongoing"
    assert sm.fire.open_incidents == {("store@fire", "fire"): new_id}


def test_a_module_alert_from_a_restarted_sidecar_reconciles_the_old_boot_first(tmp_path):
    r, store, sm = _restarted(tmp_path)
    opened = {"v": "1", "kind": "proximity_warning", "severity": "warning", "category": "safety", "message": "near",
              "pts": "3.0", "incident": "p-1", "incident_state": "opened"}

    async def run():
        await store.init()
        await _boot(r, "bay-a@proximity", "boot-a")
        await sm.fire.check_sidecar_boot_id("bay-a@proximity")
        await sm.module_alerts.handle("bay-a@proximity", opened)          # the old sidecar's incident p-1
        await _boot(r, "bay-a@proximity", "boot-b")
        await sm.module_alerts.handle("bay-a@proximity", opened)          # the new sidecar's own p-1
        await sm.fire.check_sidecar_boot_id("bay-a@proximity")
        return {row["id"]: row for row in await store.get_recent(limit=10)}

    rows = asyncio.run(run())
    old_id, new_id = sorted(rows)
    assert rows[old_id]["incident_status"] == "unknown" and rows[new_id]["incident_status"] == "ongoing"
    assert sm.module_alerts.open[("bay-a@proximity", "p-1")]["event_id"] == new_id


def test_a_zone_alert_from_a_restarted_sidecar_reconciles_the_old_boot_first(tmp_path):
    r, store, sm = _restarted(tmp_path)
    calls = []

    async def spy(sid):
        calls.append(sid)
    sm.zone_alerts._before = spy

    async def run():
        await store.init()
        await sm.zone_alerts.handle("office@zones", {"zone": "z1", "pts": "1.0", "object_id": "4",
                                                     "object_label": "person"})
    asyncio.run(run())
    assert calls == ["office@zones"]
