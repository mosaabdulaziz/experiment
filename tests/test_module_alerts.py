"""The one alert contract: a module calls EngineServices.alert, the app turns the entry into an event."""
import asyncio
import json

from app.redis_streams import read_streams
from app.usecases.alert_events import AlertEvents
from app.usecases.module_alerts import ModuleAlerts, alert_message
from perception.engine.modules import EngineServices


def _services(written):
    return EngineServices(redis=None, redis_factory=lambda: None,
                          emit=lambda key, fields, maxlen=64: written.append((key, fields, maxlen)) or True,
                          request_clip=lambda *a, **k: None, stream_ids=[], timer=lambda i, f: None, telemetry=None)


def test_a_module_alert_is_one_versioned_stream_entry_per_camera():
    written = []
    assert _services(written).alert("cam-1", "proximity_warning", severity="critical", category="safety", pts=12.3456,
                                    zone="z1", count=2, gap_m=1.5, nothing=None)
    (key, fields, _maxlen), = written
    assert key == "iris:alert:cam-1"
    assert fields == {"v": "1", "kind": "proximity_warning", "severity": "critical", "category": "safety",
                      "message": "", "zone": "z1", "pts": "12.346", "count": "2", "gap_m": "1.5"}


def test_the_text_defaults_to_the_kind():
    assert alert_message("cam-1", "proximity_warning") == "Proximity warning on cam-1"
    assert alert_message("cam-1", "x", "Truck near worker") == "Truck near worker"


class _Redis:
    def __init__(self, entries):
        self.entries = entries                      # key -> [(id, fields)]
        self.calls = []

    def xrevrange(self, key, count=1):
        e = self.entries.get(key, [])
        return e[-1:] if e else []

    def xread(self, streams, block=0, count=0):
        self.calls.append(dict(streams))
        out = []
        for key, last in streams.items():
            new = [e for e in self.entries.get(key, []) if e[0] > last]
            if new:
                out.append((key, new))
        return out


def test_one_reader_follows_every_camera_from_its_tail_and_never_repeats_an_entry():
    redis = _Redis({"iris:alert:a": [("1-0", {b"kind": b"old"})], "iris:alert:b": []})
    running = {"on": True}
    got = []

    async def go():
        async def read():
            async for key, f in read_streams(redis, lambda: ["iris:alert:a", "iris:alert:b"], lambda: running["on"]):
                got.append((key, f["kind"]))
                if len(got) == 2:
                    running["on"] = False
        task = asyncio.create_task(read())
        await asyncio.sleep(0.05)
        redis.entries["iris:alert:a"].append(("2-0", {b"kind": b"one"}))
        redis.entries["iris:alert:b"].append(("2-0", {b"kind": b"two"}))
        await asyncio.wait_for(task, 5)
    asyncio.run(go())
    assert sorted(got) == [("iris:alert:a", "one"), ("iris:alert:b", "two")]      # the entry before the tail is skipped


class _Alerts:
    def __init__(self):
        self.calls = []

    async def emit(self, *a, **k):
        self.calls.append((a, k))


def test_the_usecase_makes_an_event_with_the_modules_own_severity_and_category():
    redis = _Redis({"iris:alert:cam-1": []})
    alerts, running = _Alerts(), {"on": True}

    async def go():
        uc = ModuleAlerts(redis, alerts, lambda: ["cam-1"], lambda: running["on"])
        task = asyncio.create_task(uc.run())
        await asyncio.sleep(0.05)
        redis.entries["iris:alert:cam-1"].append(("2-0", {b"kind": b"proximity_warning", b"severity": b"critical",
                                                           b"category": b"safety", b"pts": b"4.5", b"count": b"",
                                                           b"zone": b"", b"message": b""}))
        for _ in range(100):
            if alerts.calls:
                break
            await asyncio.sleep(0.05)
        running["on"] = False
        await asyncio.wait_for(task, 5)
    asyncio.run(go())
    (args, kwargs), = alerts.calls
    assert args == ("cam-1", "Proximity warning on cam-1", ["proximity_warning"], 4.5, False, "proximity_warning", None)
    assert kwargs == {"zone_id": None, "severity": "critical", "category": "safety", "details": None,
                      "incident_started_at": None}


def test_an_event_keeps_a_stated_severity_and_falls_back_to_the_defaults():
    saved = []

    class Store:
        async def save(self, *a, **k):
            saved.append(k)
            return {"id": 1}

    class Conn:
        async def broadcast(self, *_):
            pass

    class Rec:
        def clips_enabled(self):
            return False

    ev = AlertEvents({}, Store(), Conn(), None, Rec(), None)
    asyncio.run(ev.emit("c", "m", ["t"], None, False, "intrusion"))
    asyncio.run(ev.emit("c", "m", ["t"], None, False, "proximity_warning", severity="critical", category="safety"))
    assert (saved[0]["severity"], saved[0]["category"]) == ("warning", "security")
    assert (saved[1]["severity"], saved[1]["category"]) == ("critical", "safety")


def test_a_module_overlay_is_one_versioned_entry_kept_a_few_minutes():
    written = []
    shapes = [{"type": "line", "p1": [1, 2], "p2": [3, 4], "label": "0.42", "style": "latched"}]
    assert _services(written).overlay("cam-1", "proximity", 12.3456, (1280, 720), shapes)
    (key, fields, maxlen), = written
    assert key == "iris:overlay:cam-1" and maxlen == 1500
    assert fields == {"v": "1", "module": "proximity", "pts": "12.346", "w": "1280", "h": "720",
                      "shapes": json.dumps(shapes, separators=(",", ":"))}


def test_a_quiet_event_is_saved_and_clipped_but_never_sent_to_live_viewers():
    # Ticket 6: a pair the depth check held back is kept for the Incidents page, not flashed on the dashboard
    saved, sent, clips = [], [], []

    class Store:
        async def save(self, *a, **k):
            saved.append(k)
            return {"id": 5}

    class Conn:
        async def broadcast(self, msg):
            sent.append(msg)

    class Rec:
        def clips_enabled(self):
            return True

    class Clips:
        async def submit(self, event_id, *a, **k):
            clips.append(event_id)

    ev = AlertEvents({}, Store(), Conn(), Clips(), Rec(), None)
    asyncio.run(ev.emit("c", "m", ["proximity", "depth-suppressed"], 3.0, False, "proximity", severity="info",
                        category="safety", broadcast=False))
    asyncio.run(ev.emit("c", "m", ["proximity"], 3.0, False, "proximity", severity="critical", category="safety"))
    assert len(saved) == 2 and clips == [5, 5] and len(sent) == 1
