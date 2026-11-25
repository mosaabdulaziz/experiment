"""The alert clip service burns the overlay stream's lines into the clip, with the zone, in one pass."""
import asyncio
import json

from app import alert_clips as alert_clips_module
from app.alert_clips import AlertClipService
from app.events import EventStore

_ZONE = {"id": "z0", "name": "Yard", "points": [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9)], "intrusion": {"mode": "off"}}
_SHAPES = json.dumps([{"type": "line", "p1": [100, 100], "p2": [300, 100], "label": "0.42", "style": "latched"}])


class _Redis:
    def __init__(self, entries, preview=None):
        self.entries, self.preview = entries, preview

    def xrange(self, key, lo, hi):
        return self.entries.get(key, [])

    def get(self, key):
        return self.preview if key == "iris:ctrl:proximity_preview" else None


def _raw(pts):
    return (b"1-0", {b"v": b"1", b"module": b"proximity", b"pts": str(pts).encode(), b"w": b"1280", b"h": b"720",
                     b"shapes": _SHAPES.encode()})


class _Conn:
    async def broadcast(self, payload):
        pass


class _Ref:
    def __init__(self, start, end):
        self.start_pts, self.end_pts, self.object_key, self.created = start, end, f"k{start}", 0.0


class _Segments:
    def list_range(self, source_id, lo, hi):
        return [_Ref(90.0, 110.0)]


class _Clips:
    def __init__(self):
        self.added = []

    def add(self, paths, meta=None):
        self.added.append((paths, meta))
        return f"clip{len(self.added)}"


async def _service(tmp_path, redis):
    store = EventStore({"events": {"db_path": str(tmp_path / "e.db"), "retention_hours": 48}})
    await store.init()
    cfg = {"alerts": {"clip": {"enabled": True, "pad_seconds": 5.0, "coalesce_seconds": 2.5,
                               "max_wait_seconds": 20.0, "queue_depth": 8}}}
    svc = AlertClipService(cfg, segment_store=_Segments(), clip_store=_Clips(), events_store=store,
                           conn_manager=_Conn(), build=lambda *a, **k: ("/tmp/clip.mp4", 95.0, 105.0),
                           redis_client=redis)
    ev = await store.save("alert", "m", ["proximity"], camera="cam0", clip_status="pending")
    return svc, ev["id"]


def _patch(monkeypatch, seen, overlays_result="lines"):
    monkeypatch.setattr(alert_clips_module, "probe_dimensions", lambda p: (640, 360))
    monkeypatch.setattr(alert_clips_module, "zone_overlay_png", lambda pts, w, h, dest, mode: dest)

    def fake_overlays(clip, still, timed, dest):
        seen.update(still=still, timed=open(timed).read())
        return None if overlays_result is None else clip + ".lines.mp4"

    def fake_overlay(clip, png, dest):
        seen["zone_only"] = True
        return clip + ".zone.mp4"

    monkeypatch.setattr(alert_clips_module, "apply_overlays", fake_overlays)
    monkeypatch.setattr(alert_clips_module, "apply_overlay", fake_overlay)


def test_a_clip_with_overlay_entries_gets_its_lines_burned_with_the_zone(tmp_path, monkeypatch):
    seen = {}
    _patch(monkeypatch, seen)

    async def run():
        svc, eid = await _service(tmp_path, _Redis({"iris:overlay:cam0": [_raw(50.0), _raw(50.15), _raw(500.0)]}))
        await svc.submit(eid, "cam0", 50.0, _ZONE, wall_time=100.0)
        await svc._build(svc.jobs[0])
        return svc._clips.added

    added = asyncio.run(run())
    assert added[0][0] == {"cam0": "/tmp/clip.mp4.lines.mp4"} and added[0][1]["pts"] == 50.0
    assert seen["still"].endswith("zone.png")
    assert "duration 5.000" in seen["timed"] and "duration 0.150" in seen["timed"]   # blank until 100 - 95


def test_without_overlay_entries_the_clip_gets_the_zone_only(tmp_path, monkeypatch):
    seen = {}
    _patch(monkeypatch, seen)

    async def run():
        svc, eid = await _service(tmp_path, _Redis({}))
        await svc.submit(eid, "cam0", 50.0, _ZONE, wall_time=100.0)
        await svc._build(svc.jobs[0])
        return svc._clips.added

    added = asyncio.run(run())
    assert added[0][0] == {"cam0": "/tmp/clip.mp4.zone.mp4"} and seen == {"zone_only": True}


def test_a_proximity_clip_draws_its_lines_without_the_zone(tmp_path, monkeypatch):
    seen = {}
    _patch(monkeypatch, seen)

    async def run():
        svc, eid = await _service(tmp_path, _Redis({"iris:overlay:cam0": [_raw(50.0)]}))
        await svc.submit(eid, "cam0", 50.0, _ZONE, wall_time=100.0, kind="proximity")
        await svc._build(svc.jobs[0])
        return svc._clips.added

    added = asyncio.run(run())
    assert added[0][0] == {"cam0": "/tmp/clip.mp4.lines.mp4"} and seen["still"] is None


def test_a_proximity_clip_without_overlay_entries_stays_plain(tmp_path, monkeypatch):
    seen = {}
    _patch(monkeypatch, seen)

    async def run():
        svc, eid = await _service(tmp_path, _Redis({}))
        await svc.submit(eid, "cam0", 50.0, _ZONE, wall_time=100.0, kind="proximity")
        await svc._build(svc.jobs[0])
        return svc._clips.added

    added = asyncio.run(run())
    assert added[0][0] == {"cam0": "/tmp/clip.mp4"} and seen == {}


def test_a_failed_lines_pass_still_delivers_the_zone_clip(tmp_path, monkeypatch):
    seen = {}
    _patch(monkeypatch, seen, overlays_result=None)

    async def run():
        svc, eid = await _service(tmp_path, _Redis({"iris:overlay:cam0": [_raw(50.0)]}))
        await svc.submit(eid, "cam0", 50.0, _ZONE, wall_time=100.0)
        await svc._build(svc.jobs[0])
        return svc._clips.added

    assert asyncio.run(run())[0][0] == {"cam0": "/tmp/clip.mp4.zone.mp4"}


class _TimedRedis:
    """Honours XRANGE's min/max on the entry ids (ms of the write), as Redis does."""
    def __init__(self, entries):
        self.entries = entries          # key -> [(ms, pts)]

    def xrange(self, key, lo, hi):
        lo = 0 if lo == "-" else int(lo)
        hi = 1 << 62 if hi == "+" else int(hi)
        return [(f"{ms}-0".encode(), _raw(pts)[1]) for ms, pts in self.entries.get(key, []) if lo <= ms <= hi]


def test_entries_written_long_before_the_clip_are_another_run_and_ignored(tmp_path, monkeypatch):
    seen = {}
    _patch(monkeypatch, seen)
    old_run, this_run = 1000 * 1000, 100 * 1000 * 1000      # ms: an earlier pipeline run vs. around this clip

    async def run():
        redis = _TimedRedis({"iris:overlay:cam0": [(old_run, 50.0), (old_run + 150, 50.15)]})
        svc, eid = await _service(tmp_path, redis)
        svc._build_fn = lambda *a, **k: ("/tmp/clip.mp4", 100095.0, 100105.0)
        svc._segments = type("Segs", (), {"list_range": lambda self, *a: [_Ref(100090.0, 100110.0)]})()
        await svc.submit(eid, "cam0", 50.0, _ZONE, wall_time=100100.0)
        await svc._build(svc.jobs[0])
        return svc._clips.added

    added = asyncio.run(run())
    assert seen == {"zone_only": True} and added[0][0] == {"cam0": "/tmp/clip.mp4.zone.mp4"}
    assert this_run                                          # (entries written near 100100 s would be drawn)


def test_burning_the_lines_does_not_block_the_event_loop(tmp_path, monkeypatch):
    import time as _time
    seen = {}
    _patch(monkeypatch, seen)

    def slow_overlays(clip, still, timed, dest):
        _time.sleep(0.5)                                     # ffmpeg's pass
        return clip + ".lines.mp4"

    monkeypatch.setattr(alert_clips_module, "apply_overlays", slow_overlays)

    async def run():
        svc, eid = await _service(tmp_path, _Redis({"iris:overlay:cam0": [_raw(50.0)]}))
        await svc.submit(eid, "cam0", 50.0, _ZONE, wall_time=100.0)
        ticks = []

        async def ticker():
            while True:
                ticks.append(1)
                await asyncio.sleep(0.05)

        task = asyncio.create_task(ticker())
        await svc._build(svc.jobs[0])
        task.cancel()
        return ticks

    assert len(asyncio.run(run())) >= 5                      # the loop kept running while the pass ran


class _LabelledRef:
    """A segment as the wall-mode index returns it: wall labels (indexed time minus duration) and its frame times."""
    def __init__(self, label_start, label_end, pts_start, recorded_at=0.0):
        self.start_pts, self.end_pts, self.pts_start = label_start, label_end, pts_start
        self.object_key, self.recorded_at = f"k{label_start}", recorded_at


def test_lines_are_placed_by_the_clips_own_frame_times_not_the_alerts_receipt_time(tmp_path, monkeypatch):
    """The receipt time (103) and the segment's wall label disagree by 3 s; the clip's frame 0 is pts 45 (label 95),
    so the entry at pts 50 shows 5 s in -- not 8 s, where the receipt time would put it."""
    seen = {}
    _patch(monkeypatch, seen)

    async def run():
        svc, eid = await _service(tmp_path, _Redis({"iris:overlay:cam0": [_raw(50.0)]}))
        svc._segments = type("Segs", (), {"list_range": lambda self, *a, **k: [_LabelledRef(90.0, 110.0, 40.0)]})()
        await svc.submit(eid, "cam0", 50.0, _ZONE, wall_time=103.0)
        await svc._build(svc.jobs[0])
        return svc._clips.added

    added = asyncio.run(run())
    assert added[0][1]["start_pts"] == 45.0
    assert seen["timed"].splitlines()[2] == "duration 5.000"


def test_a_clip_is_anchored_on_the_cameras_segment_clock_when_a_recent_segment_holds_it(tmp_path):
    import time as _time
    now = _time.time()

    async def run(refs, wall_time, svr=False):
        svc, eid = await _service(tmp_path, _Redis({}))
        svc._segments = type("Segs", (), {"list_range": lambda self, *a, **k: refs})()
        if svr:
            svc.mark_svr_source("cam0")
        await svc.submit(eid, "cam0", 50.0, None, wall_time=wall_time)
        return svc.jobs[-1]["anchor"]

    recent = [_LabelledRef(now - 20.0, now - 14.0, 40.0, recorded_at=now - 14.0)]     # label = pts + (now - 60)
    assert asyncio.run(run(recent, now)) == now - 60.0 + 50.0
    stale = [_LabelledRef(now - 400.0, now - 394.0, 900.0, recorded_at=now - 394.0)]   # an earlier run's segment
    assert asyncio.run(run(stale, now)) == now
    assert asyncio.run(run([], now)) == now
    assert asyncio.run(run(recent, now, svr=True)) == now



_PAIRS = json.dumps([
    {"type": "line", "p1": [100, 600], "p2": [500, 500], "label": "3.10", "style": "far", "person": 6, "vehicle": 11},
    {"type": "line", "p1": [700, 600], "p2": [650, 500], "label": "0.40", "style": "close", "person": 7,
     "vehicle": 11},
    {"type": "box", "rect": [80, 500, 40, 100], "style": "person", "id": 6, "label": "person 6"},
    {"type": "box", "rect": [680, 500, 40, 100], "style": "person", "id": 7, "label": "person 7"},
    {"type": "box", "rect": [400, 200, 300, 300], "style": "vehicle", "id": 11, "label": "truck 11",
     "state": "moving"}])


def _pairs_raw(pts):
    return (b"1-0", {b"v": b"1", b"module": b"proximity", b"pts": str(pts).encode(), b"w": b"1280", b"h": b"720",
                     b"shapes": _PAIRS.encode()})


def _record_segments(monkeypatch, drawn, styles=None):
    real = alert_clips_module.write_sequence

    def recording(segments, *a, **k):
        drawn.extend(segments)
        if styles is not None:
            styles.append(k.get("style"))
        return real(segments, *a, **k)
    monkeypatch.setattr(alert_clips_module, "write_sequence", recording)


def test_an_incident_clip_draws_its_pair_throughout_bridging_short_gaps(tmp_path, monkeypatch):
    seen, drawn = {}, []
    _patch(monkeypatch, seen)
    _record_segments(monkeypatch, drawn)

    async def run():
        svc, eid = await _service(tmp_path, _Redis({"iris:overlay:cam0": [_pairs_raw(50.0), _pairs_raw(50.9)]}))
        await svc.submit(eid, "cam0", 50.0, _ZONE, wall_time=100.0, pair=({6}, {11}))
        await svc._build(svc.jobs[0])

    asyncio.run(run())
    shown = [(round(start, 2), round(length, 2)) for start, length, e in drawn if e is not None]
    assert shown == [(5.0, 0.9), (5.9, 1.0)]                    # no blank between entries 0.9 s apart
    entry = next(e for _s, _d, e in drawn if e is not None)
    assert {(s["type"], s.get("person", s.get("id"))) for s in entry["shapes"]} == {
        ("line", 6), ("box", 6), ("box", 11)}


def test_coalesced_incidents_share_a_clip_that_draws_both_pairs(tmp_path, monkeypatch):
    async def run():
        svc, eid = await _service(tmp_path, _Redis({}))
        other = await svc._events.save("alert", "m2", ["proximity"], camera="cam0", clip_status="pending")
        await svc.submit(eid, "cam0", 50.0, _ZONE, wall_time=100.0, pair=({6}, {11}))
        await svc.submit(other["id"], "cam0", 50.5, _ZONE, wall_time=100.5, pair=({7}, {11}))
        return svc.jobs

    (job,) = asyncio.run(run())
    assert job["pair"] == ({6, 7}, {11})


def test_a_clip_is_drawn_in_the_style_chosen_in_the_ui(tmp_path, monkeypatch):
    for preview, expected in ((json.dumps({"clip_style": "boxes"}), "boxes"), (None, "masks")):
        seen, drawn, styles = {}, [], []
        _patch(monkeypatch, seen)
        _record_segments(monkeypatch, drawn, styles)

        async def run():
            svc, eid = await _service(tmp_path, _Redis({"iris:overlay:cam0": [_pairs_raw(50.0)]}, preview))
            await svc.submit(eid, "cam0", 50.0, _ZONE, wall_time=100.0, pair=({6}, {11}))
            await svc._build(svc.jobs[0])

        asyncio.run(run())
        assert styles == [expected]
