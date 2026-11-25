"""Module alerts with an incident: one event per incident, raised in place, closed when it ends."""
import asyncio

from app.events import EventStore
from app.usecases import alert_events as alert_events_module
from app.usecases.alert_events import AlertEvents
from app.usecases.module_alerts import ModuleAlerts


class _Alerts:
    def __init__(self):
        self.calls = []

    async def emit(self, *a, **k):
        self.calls.append(("emit", a, k))
        return 7

    async def escalate(self, *a):
        self.calls.append(("escalate", a))

    async def resolve_incident(self, *a):
        self.calls.append(("resolve", a))

    async def snapshot(self, *a):
        self.calls.append(("snapshot", a))


def _entry(state, incident="inc1", **extra):
    f = {"v": "1", "kind": "proximity", "severity": "warning", "category": "safety", "message": "Person near truck",
         "zone": "yard", "pts": "12.000", "count": "", "incident": incident, "incident_state": state}
    f.update(extra)
    return f


def _handle(uc, *entries, sid="cam-1"):
    async def run():
        for e in entries:
            await uc.handle(sid, e)
    asyncio.run(run())


def _uc():
    alerts = _Alerts()
    return ModuleAlerts(None, alerts, lambda: ["cam-1"], lambda: True), alerts


OPEN = dict(gap="0.42", duration_s="1.2", person_ids="1", vehicle_ids="10", min_gap_pts="11.500")


def test_an_opened_incident_is_one_event_with_its_details_and_an_ongoing_status():
    uc, alerts = _uc()
    _handle(uc, _entry("opened", **OPEN))
    (_, args, kwargs), = alerts.calls
    assert args[:3] == ("cam-1", "Person near truck", ["proximity"])
    assert kwargs["details"] == OPEN and kwargs["severity"] == "warning" and kwargs["incident_started_at"]
    assert uc.open[("cam-1", "inc1")] == {"event_id": 7, "min_gap_pts": 11.5, "persons": [1], "vehicles": [10]}


def test_an_escalation_raises_the_same_event_instead_of_making_a_new_one():
    uc, alerts = _uc()
    _handle(uc, _entry("opened", **OPEN), _entry("escalated", severity="critical", gap="0.12", min_gap_pts="12.000"))
    assert [c[0] for c in alerts.calls] == ["emit", "escalate"]
    assert alerts.calls[1][1] == (7, "cam-1", 12.0, "critical", "Person near truck",
                                  {"gap": "0.12", "min_gap_pts": "12.000"}, "yard", "proximity")
    assert uc.open[("cam-1", "inc1")]["min_gap_pts"] == 12.0


def test_an_escalation_the_app_never_saw_open_becomes_a_new_event():
    uc, alerts = _uc()
    _handle(uc, _entry("escalated", severity="critical"))
    assert [c[0] for c in alerts.calls] == ["emit"] and ("cam-1", "inc1") in uc.open


def test_a_resolved_incident_closes_its_event_queues_its_snapshot_and_is_forgotten():
    uc, alerts = _uc()
    done = dict(duration_s="3.4", min_gap="0.12", min_gap_pts="12.800", person_ids="1,2", vehicle_ids="10")
    _handle(uc, _entry("opened", **OPEN), _entry("resolved", **done))
    assert alerts.calls[1] == ("resolve", (7, "cam-1", 3.4, done))
    assert alerts.calls[2] == ("snapshot", (7, "cam-1", 12.8, [1, 2], [10]))
    assert uc.open == {}


def test_a_resolved_incident_the_app_never_saw_is_skipped():
    uc, alerts = _uc()
    _handle(uc, _entry("resolved", duration_s="3.4"))
    assert alerts.calls == []


def test_a_pipeline_restart_drops_the_cameras_incidents_with_a_snapshot_each():
    uc, alerts = _uc()
    _handle(uc, _entry("opened", **OPEN))
    _handle(uc, _entry("opened", incident="other", **OPEN), sid="cam-2")
    asyncio.run(uc.sidecar_restarted("cam-1"))
    assert alerts.calls[-1] == ("snapshot", (7, "cam-1", 11.5, [1], [10]))
    assert list(uc.open) == [("cam-2", "other")]


def test_an_alert_without_an_incident_is_unchanged():
    uc, alerts = _uc()
    plain = {"v": "1", "kind": "x", "severity": "info", "category": "", "message": "", "zone": "", "pts": "", "count": ""}
    _handle(uc, plain)
    (_, _args, kwargs), = alerts.calls
    assert kwargs["details"] is None and kwargs["incident_started_at"] is None and uc.open == {}


class _Conn:
    def __init__(self):
        self.events = []

    async def broadcast(self, payload):
        self.events.append(payload)


class _Rec:
    def clips_enabled(self):
        return True


class _ClipService:
    def __init__(self):
        self.submitted, self.snaps = [], []

    async def submit(self, *a, **k):
        self.submitted.append((a, k))
        return True

    def snapshot(self, *a):
        self.snaps.append(a)


async def _events(tmp_path):
    store = EventStore({"events": {"db_path": str(tmp_path / "e.db"), "retention_hours": 48}})
    await store.init()
    conn, clips = _Conn(), _ClipService()
    return AlertEvents({}, store, conn, clips, _Rec(), None), store, conn, clips


def test_escalating_a_deleted_event_raises_nothing_and_says_so(tmp_path):
    async def run():
        ev, store, conn, clips = await _events(tmp_path)
        rec = await store.save("alert", "m1", ["zone_intrusion"], camera="cam-1", severity="warning")
        live = await ev.escalate(rec["id"], "cam-1", 12.0, "warning", "m2", {"entries": 2})
        gone = await ev.escalate(rec["id"] + 99, "cam-1", 13.0, "warning", "m3", {"entries": 3})
        return live, gone, len(conn.events), len(clips.submitted)

    assert asyncio.run(run()) == (True, False, 1, 1)


def test_escalate_updates_the_event_in_place_and_queues_a_follow_up_clip(tmp_path):
    async def run():
        ev, store, conn, clips = await _events(tmp_path)
        rec = await store.save("alert", "m1", ["proximity"], camera="cam-1", severity="warning", details={"gap": "0.4"})
        await ev.escalate(rec["id"], "cam-1", 12.0, "critical", "m2", {"gap": "0.1"})
        return await store.get_event(rec["id"]), conn.events, clips.submitted

    row, broadcasts, submitted = asyncio.run(run())
    assert (row["severity"], row["message"], row["details"]["escalated_from"]) == ("critical", "m2", "warning")
    assert broadcasts[-1]["event"]["severity"] == "critical"
    (args, kwargs), = submitted
    assert args[:3] == (row["id"], "cam-1", 12.0) and kwargs["append"] is True and kwargs["pair"] is None


def test_an_incidents_clips_follow_its_pair(tmp_path):
    async def run():
        ev, store, conn, clips = await _events(tmp_path)
        eid = await ev.emit("cam-1", "m", ["proximity"], 12.0, False, "proximity", severity="warning",
                            category="safety", details={"person_ids": "1,4", "vehicle_ids": "10"})
        await ev.escalate(eid, "cam-1", 13.0, "critical", "m2", {"person_ids": "1", "vehicle_ids": "10,12"},
                          None, "proximity")
        return clips.submitted

    (_a1, opened), (_a2, escalated) = asyncio.run(run())
    assert opened["pair"] == ({1, 4}, {10}) and escalated["pair"] == ({1}, {10, 12})
    assert opened["kind"] == escalated["kind"] == "proximity"          # their clips leave the zone out


def test_resolve_incident_marks_it_resolved_with_its_duration(tmp_path):
    async def run():
        ev, store, conn, _ = await _events(tmp_path)
        rec = await store.save("alert", "m", ["proximity"], camera="cam-1", incident_status="ongoing",
                               incident_started_at="2026-10-29T10:00:00Z")
        await ev.resolve_incident(rec["id"], "cam-1", 3.4, {"min_gap": "0.12"})
        return await store.get_event(rec["id"]), conn.events

    row, broadcasts = asyncio.run(run())
    assert (row["incident_status"], row["incident_duration_seconds"]) == ("resolved", 3.4)
    assert row["details"]["min_gap"] == "0.12" and broadcasts[-1]["event"]["incident_status"] == "resolved"


def test_emit_names_the_zone_in_the_details(tmp_path, monkeypatch):
    monkeypatch.setattr(alert_events_module, "zone_by_id", lambda cfg, sid, zid: {"id": zid, "name": "Yard"})

    async def run():
        ev, store, *_ = await _events(tmp_path)
        eid = await ev.emit("cam-1", "m", ["proximity"], 12.0, False, "proximity", zone_id="z0",
                            severity="warning", category="safety", details={"gap": "0.4"})
        return await store.get_event(eid)

    assert asyncio.run(run())["details"] == {"gap": "0.4", "zone_name": "Yard"}


class _Store:
    def __init__(self, ok=True):
        self.kept, self.ok = {}, ok

    def put(self, event_id, b64):
        if self.ok:
            self.kept[event_id] = b64
        return self.ok


def test_a_depth_image_is_kept_apart_and_the_event_says_it_has_one():
    store = _Store()
    alerts = _Alerts()
    uc = ModuleAlerts(None, alerts, lambda: ["cam-1"], lambda: True, depth_images=store)
    _handle(uc, _entry("opened", depth_image="QUJD", depth_gap="0.050", **OPEN))
    (_, _args, kwargs), = alerts.calls
    assert kwargs["details"]["depth_image"] == "1" and kwargs["details"]["depth_gap"] == "0.050"
    assert store.kept == {7: "QUJD"}


def test_an_escalation_replaces_the_depth_image():
    store = _Store()
    uc = ModuleAlerts(None, _Alerts(), lambda: ["cam-1"], lambda: True, depth_images=store)
    _handle(uc, _entry("opened", depth_image="T0xE", **OPEN), _entry("escalated", depth_image="TkVX", **OPEN))
    assert store.kept == {7: "TkVX"}


def test_a_bad_depth_image_is_refused_and_the_event_kept():
    alerts = _Alerts()
    uc = ModuleAlerts(None, alerts, lambda: ["cam-1"], lambda: True, depth_images=_Store(ok=False))
    _handle(uc, _entry("opened", depth_image="???", **OPEN))
    assert [c[0] for c in alerts.calls] == ["emit"]


def test_a_pair_the_depth_check_held_back_is_kept_quietly():
    # Ticket 6: saved with its clip, tagged, never broadcast live; not an incident
    uc, alerts = _uc()
    quiet = {"v": "1", "kind": "proximity", "severity": "info", "category": "safety",
             "message": "Held back by the depth check", "zone": "yard", "pts": "12.000", "count": "",
             "suppressed": "depth", "would_be": "critical", "depth_gap": "0.500", "depth_max_gap": "0.25",
             "gap": "0.42", "duration_s": "1.2", "person_ids": "1", "vehicle_ids": "10", "min_gap_pts": "11.500"}
    _handle(uc, quiet)
    ((what, a, k),) = alerts.calls
    assert what == "emit" and a[2] == ["proximity", "depth-suppressed"]
    assert (k["severity"], k["broadcast"], k["incident_started_at"]) == ("info", False, None)
    assert (k["details"]["suppressed"], k["details"]["would_be"], k["details"]["depth_gap"]) == ("depth", "critical",
                                                                                                "0.500")


class _Notifier:
    def __init__(self):
        self.calls = []

    def notify(self, record, how):
        self.calls.append((record["id"], how, record["severity"]))


def test_an_incident_is_handed_to_the_notifier_when_it_opens_and_when_it_escalates(tmp_path):
    async def run():
        ev, _store, _conn, _clips = await _events(tmp_path)
        ev.notifier = _Notifier()
        eid = await ev.emit("cam-1", "m", ["proximity_warning"], 12.0, False, "proximity", severity="warning")
        await ev.escalate(eid, "cam-1", 13.0, "critical", "m2", None, None, "proximity")
        await ev.emit("cam-1", "quiet", ["proximity_warning"], 14.0, False, "proximity", broadcast=False)
        return eid, ev.notifier.calls

    eid, calls = asyncio.run(run())
    assert calls == [(eid, "opened", "warning"), (eid, "escalated", "critical")]    # never a quiet record


def test_an_update_at_the_same_severity_is_not_sent_as_an_escalation(tmp_path):
    """A second entry into an open zone incident raises it in place at the same severity: nothing to page about."""
    async def run():
        ev, _store, _conn, _clips = await _events(tmp_path)
        ev.notifier = _Notifier()
        eid = await ev.emit("cam-1", "m", ["zone_intrusion"], 12.0, False, "intrusion", severity="warning")
        await ev.escalate(eid, "cam-1", 13.0, "warning", "m (2 entries)", {"entries": 2}, None, "intrusion")
        return eid, ev.notifier.calls

    eid, calls = asyncio.run(run())
    assert calls == [(eid, "opened", "warning")]
