import asyncio
import subprocess
import time
from unittest import mock

import pytest

from app import alert_clips as alert_clips_module
from app.alert_clips import AlertClipService
from app.clip_store import ClipStore
from app.events import EventStore


def _cfg(**overrides):
    clip = {"enabled": True, "pad_seconds": 5.0, "coalesce_seconds": 2.5,
            "ttl_seconds": 21600, "max_wait_seconds": 20.0, "queue_depth": 8}
    clip.update(overrides)
    return {"alerts": {"clip": clip}}


class _StubConn:
    def __init__(self):
        self.events = []

    async def broadcast(self, payload):
        self.events.append(payload)


class _StubSegments:
    def __init__(self, refs=()):
        self.refs = list(refs)

    def list_range(self, source_id, lo, hi):
        return [r for r in self.refs if r.end_pts >= lo and r.start_pts <= hi]


class _StubClipStore:
    def __init__(self):
        self.added = []

    def add(self, cameras_mp4_paths, meta=None):
        self.added.append((cameras_mp4_paths, meta))
        return f"clip{len(self.added)}"


async def _service(tmp_path, **overrides):
    """Build a service wired to a real EventStore and stub collaborators."""
    store = EventStore({"events": {"db_path": str(tmp_path / "e.db"), "retention_hours": 48}})
    await store.init()
    conn = _StubConn()
    svc = AlertClipService(_cfg(**overrides), segment_store=_StubSegments(),
                           clip_store=_StubClipStore(), events_store=store,
                           conn_manager=conn)
    return svc, store, conn


def test_submit_opens_a_job(tmp_path):
    async def run():
        svc, _store, _conn = await _service(tmp_path)
        ok = await svc.submit(1, "cam0", 100.0)
        return ok, svc.jobs

    ok, jobs = asyncio.run(run())
    assert ok is True
    assert len(jobs) == 1
    assert jobs[0]["lo"] == 95.0 and jobs[0]["hi"] == 105.0
    assert jobs[0]["event_ids"] == [1]


def test_submit_with_wall_time_uses_wall_anchor(tmp_path):
    async def run():
        svc, _store, _conn = await _service(tmp_path)
        ok = await svc.submit(1, "cam0", pts=100.0, wall_time=1789224913.0)
        return ok, svc.jobs

    ok, jobs = asyncio.run(run())
    assert ok is True
    assert len(jobs) == 1
    assert jobs[0]["anchor"] == 1789224913.0
    assert jobs[0]["lo"] == 1789224913.0 - 5.0
    assert jobs[0]["hi"] == 1789224913.0 + 5.0
    assert jobs[0]["event_ids"] == [1]



def test_nearby_alerts_share_one_job(tmp_path):
    async def run():
        svc, _store, _conn = await _service(tmp_path)
        await svc.submit(1, "cam0", 100.0)
        await svc.submit(2, "cam0", 101.0)
        return svc.jobs

    jobs = asyncio.run(run())
    assert len(jobs) == 1
    assert jobs[0]["event_ids"] == [1, 2]
    # The window is anchored on the first alert and never widened.
    assert jobs[0]["hi"] == 105.0


def test_distant_alerts_get_separate_jobs(tmp_path):
    async def run():
        svc, _store, _conn = await _service(tmp_path)
        await svc.submit(1, "cam0", 100.0)
        await svc.submit(2, "cam0", 110.0)
        return svc.jobs

    assert len(asyncio.run(run())) == 2


def test_queue_full_rejects_and_resolves_unavailable(tmp_path):
    async def run():
        svc, store, conn = await _service(tmp_path, queue_depth=1)
        rec = await store.save("alert", "x", [], camera="cam0", clip_status="pending")
        await svc.submit(1, "cam0", 100.0)                 # fills the queue
        ok = await svc.submit(rec["id"], "cam0", 200.0)
        return ok, await store.get_recent(limit=10), conn.events, rec["id"]

    ok, rows, broadcasts, event_id = asyncio.run(run())
    assert ok is False
    assert rows[-1]["clip_status"] == "unavailable"
    assert broadcasts[-1] == {"type": "event_update", "id": event_id,
                              "clip_status": "unavailable", "clip_id": None, "clips": []}


def test_unknown_pts_resolves_unavailable_without_queuing(tmp_path):
    async def run():
        svc, store, _conn = await _service(tmp_path)
        rec = await store.save("alert", "x", [], camera="cam0", clip_status="pending")
        ok = await svc.submit(rec["id"], "cam0", None)
        return ok, svc.jobs, await store.get_recent(limit=10)

    ok, jobs, rows = asyncio.run(run())
    assert ok is False
    assert jobs == []
    assert rows[-1]["clip_status"] == "unavailable"


def test_disabled_service_never_queues(tmp_path):
    async def run():
        svc, _store, _conn = await _service(tmp_path, enabled=False)
        ok = await svc.submit(1, "cam0", 100.0)
        return svc.enabled, ok, svc.jobs

    enabled, ok, jobs = asyncio.run(run())
    assert enabled is False
    assert ok is False
    assert jobs == []


from app.segment_store import SegmentRef


def _ref(start, end):
    return SegmentRef(start, end, f"k{start}", 0.0)


async def _drain(svc, timeout=2.0):
    """Run the worker until every job is resolved, or give up."""
    task = asyncio.create_task(svc.run())
    deadline = asyncio.get_running_loop().time() + timeout
    while svc.jobs and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.05)
    svc.stop()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


def test_covered_job_builds_and_resolves_every_attached_event(tmp_path):
    built = {}

    async def run():
        svc, store, conn = await _service(tmp_path)
        svc._segments = _StubSegments([_ref(90.0, 98.0), _ref(98.0, 106.0)])

        def _fake_build(segment_store, source_id, lo, hi, padding, trim_enabled,
                        trim_seconds, dest_dir, window=None):
            built["args"] = (source_id, lo, hi, padding, window)
            path = tmp_path / "out.mp4"
            path.write_bytes(b"x")
            return (str(path), window[0], window[1])

        svc._build_fn = _fake_build

        a = await store.save("alert", "a", [], camera="cam0", clip_status="pending")
        b = await store.save("alert", "b", [], camera="cam0", clip_status="pending")
        await svc.submit(a["id"], "cam0", 100.0)
        await svc.submit(b["id"], "cam0", 101.0)          # coalesces

        await _drain(svc)

        rows = {r["id"]: r for r in await store.get_recent(limit=10)}
        return rows, conn.events, a["id"], b["id"]

    rows, broadcasts, a_id, b_id = asyncio.run(run())
    # Exact centred window, no extra retrieval padding, no heuristic trim.
    assert built["args"] == ("cam0", 95.0, 105.0, 0.0, (95.0, 105.0))
    assert rows[a_id]["clip_status"] == "ready"
    assert rows[b_id]["clip_status"] == "ready"
    assert rows[a_id]["clip_id"] == rows[b_id]["clip_id"]      # one clip, two events
    assert sum(1 for e in broadcasts if e["type"] == "event_update") == 2


def test_deadline_with_partial_footage_builds_the_run_holding_the_anchor(tmp_path):
    built = {}

    async def run():
        svc, store, _conn = await _service(tmp_path, max_wait_seconds=0.0)
        # Covers the anchor but stops short of hi=105.0.
        svc._segments = _StubSegments([_ref(90.0, 98.0), _ref(98.0, 102.0)])

        def _fake_build(segment_store, source_id, lo, hi, padding, trim_enabled,
                        trim_seconds, dest_dir, window=None):
            built["window"] = window
            path = tmp_path / "partial.mp4"
            path.write_bytes(b"x")
            return (str(path), window[0], window[1])

        svc._build_fn = _fake_build

        rec = await store.save("alert", "a", [], camera="cam0", clip_status="pending")
        await svc.submit(rec["id"], "cam0", 100.0)
        await _drain(svc)
        return await store.get_recent(limit=10)

    rows = asyncio.run(run())
    assert built["window"] == (95.0, 102.0)      # clamped to what exists
    assert rows[-1]["clip_status"] == "ready"


def test_deadline_with_no_footage_holding_the_anchor_resolves_unavailable(tmp_path):
    async def run():
        svc, store, _conn = await _service(tmp_path, max_wait_seconds=0.0)
        svc._segments = _StubSegments([_ref(10.0, 18.0)])    # nowhere near the anchor
        rec = await store.save("alert", "a", [], camera="cam0", clip_status="pending")
        await svc.submit(rec["id"], "cam0", 100.0)
        await _drain(svc)
        return await store.get_recent(limit=10)

    rows = asyncio.run(run())
    assert rows[-1]["clip_status"] == "unavailable"
    assert rows[-1]["clip_id"] is None


def test_build_failure_resolves_unavailable_and_the_worker_survives(tmp_path):
    async def run():
        svc, store, _conn = await _service(tmp_path, max_wait_seconds=0.0)
        svc._segments = _StubSegments([_ref(90.0, 106.0)])

        def _boom(*_args, **_kwargs):
            raise RuntimeError("ffmpeg exploded")

        svc._build_fn = _boom

        rec = await store.save("alert", "a", [], camera="cam0", clip_status="pending")
        await svc.submit(rec["id"], "cam0", 100.0)
        await _drain(svc)
        return await store.get_recent(limit=10), svc.jobs

    rows, jobs = asyncio.run(run())
    assert rows[-1]["clip_status"] == "unavailable"
    assert jobs == []                # job removed, loop did not wedge on it


def test_uncovered_job_waits_instead_of_building(tmp_path):
    calls = []

    async def run():
        svc, store, _conn = await _service(tmp_path)        # 20s deadline, far away
        svc._segments = _StubSegments([_ref(90.0, 98.0)])   # short of hi=105.0
        svc._build_fn = lambda *a, **k: calls.append(a)

        rec = await store.save("alert", "a", [], camera="cam0", clip_status="pending")
        await svc.submit(rec["id"], "cam0", 100.0)

        task = asyncio.create_task(svc.run())
        await asyncio.sleep(0.3)
        svc.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return svc.jobs

    jobs = asyncio.run(run())
    assert calls == []               # never built before coverage arrived
    assert len(jobs) == 1
    assert jobs[0]["state"] == "queued"


def test_worker_survives_a_failing_job_and_keeps_serving_later_ones(tmp_path):
    """The first build raises; the loop must catch it, resolve that job, and still
    be alive to build the next one. Asserting only on the failed job would pass
    even if the loop had broken out, because the finally clause still runs."""
    async def run():
        svc, store, _conn = await _service(tmp_path, max_wait_seconds=0.0)
        svc._segments = _StubSegments([_ref(90.0, 106.0)])

        calls = {"n": 0}

        def _first_call_explodes(segment_store, source_id, lo, hi, padding,
                                 trim_enabled, trim_seconds, dest_dir, window=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("ffmpeg exploded")
            path = tmp_path / "second.mp4"
            path.write_bytes(b"x")
            return (str(path), window[0], window[1])

        svc._build_fn = _first_call_explodes

        bad = await store.save("alert", "a", [], camera="cam0", clip_status="pending")
        await svc.submit(bad["id"], "cam0", 100.0)

        task = asyncio.create_task(svc.run())
        deadline = asyncio.get_running_loop().time() + 2.0
        while svc.jobs and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.05)

        # Same anchor as the failed job: the first job is already resolved and
        # removed by this point, so this opens a fresh job rather than coalescing,
        # and it stays within the one stub segment's coverage (90.0-106.0) so the
        # build actually runs instead of immediately resolving unavailable for
        # lack of footage (a separate, already-covered path).
        good = await store.save("alert", "b", [], camera="cam0", clip_status="pending")
        await svc.submit(good["id"], "cam0", 100.0)
        deadline = asyncio.get_running_loop().time() + 2.0
        while svc.jobs and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.05)

        svc.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        rows = {r["id"]: r for r in await store.get_recent(limit=10)}
        return rows, bad["id"], good["id"], calls["n"]

    rows, bad_id, good_id, n = asyncio.run(run())
    assert rows[bad_id]["clip_status"] == "unavailable"
    assert rows[good_id]["clip_status"] == "ready"     # loop was still alive
    assert n == 2


def test_nested_segments_do_not_truncate_the_cut_window(tmp_path):
    """refs are sorted by start_pts, so the LAST ref is not necessarily the one
    that ends furthest. Using its end would cut the clip short -- here it would
    end at 98.0, before the alert at 100.0 even happened."""
    built = {}

    async def run():
        svc, store, _conn = await _service(tmp_path)
        svc._segments = _StubSegments([_ref(90.0, 106.0), _ref(96.0, 98.0)])

        def _fake_build(segment_store, source_id, lo, hi, padding, trim_enabled,
                        trim_seconds, dest_dir, window=None):
            built["window"] = window
            path = tmp_path / "nested.mp4"
            path.write_bytes(b"x")
            return (str(path), window[0], window[1])

        svc._build_fn = _fake_build

        rec = await store.save("alert", "a", [], camera="cam0", clip_status="pending")
        await svc.submit(rec["id"], "cam0", 100.0)
        await _drain(svc)
        return await store.get_recent(limit=10)

    rows = asyncio.run(run())
    assert built["window"] == (95.0, 105.0)     # full window, not truncated to 98.0
    assert rows[-1]["clip_status"] == "ready"


def test_invalid_pad_seconds_is_rejected_when_enabled(tmp_path):
    async def run():
        store = EventStore({"events": {"db_path": str(tmp_path / "e.db"),
                                       "retention_hours": 48}})
        await store.init()
        return store

    store = asyncio.run(run())
    with pytest.raises(ValueError):
        AlertClipService(_cfg(pad_seconds=-1.0), segment_store=_StubSegments(),
                         clip_store=_StubClipStore(), events_store=store,
                         conn_manager=_StubConn())


def test_invalid_config_is_ignored_while_the_service_is_disabled(tmp_path):
    # A disabled service is inert; a stale or nonsensical alerts.clip block must
    # not stop the app from starting.
    async def run():
        store = EventStore({"events": {"db_path": str(tmp_path / "d.db"),
                                       "retention_hours": 48}})
        await store.init()
        return store

    store = asyncio.run(run())
    svc = AlertClipService(_cfg(enabled=False, pad_seconds=-1.0),
                           segment_store=_StubSegments(), clip_store=_StubClipStore(),
                           events_store=store, conn_manager=_StubConn())
    assert svc.enabled is False


# ── zone plumbing ───────────────────────────────────────────────────────────

_ZONE = {"id": "z0", "name": "Front Door", "points": [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9)],
         "intrusion": {"mode": "restricted", "classes": ["person"]}}


def test_submit_stores_the_zone_on_the_job(tmp_path):
    async def run():
        svc, _store, _conn = await _service(tmp_path)
        await svc.submit(1, "cam0", 100.0, _ZONE)
        return svc.jobs

    jobs = asyncio.run(run())
    assert jobs[0]["zone"] == _ZONE


def test_submit_without_a_zone_stores_none(tmp_path):
    async def run():
        svc, _store, _conn = await _service(tmp_path)
        await svc.submit(1, "cam0", 100.0)
        return svc.jobs

    jobs = asyncio.run(run())
    assert jobs[0]["zone"] is None


def test_alerts_from_two_zones_do_not_share_a_job(tmp_path):
    async def run():
        svc, _store, _conn = await _service(tmp_path)
        await svc.submit(1, "cam0", 100.0, _ZONE)
        await svc.submit(2, "cam0", 100.5, {**_ZONE, "id": "z1"})
        return svc.jobs

    jobs = asyncio.run(run())
    assert len(jobs) == 2


# ── zone overlay on the built clip ──────────────────────────────────────────

class _StubSegRef:
    def __init__(self, start, end):
        self.start_pts, self.end_pts = start, end
        self.object_key, self.created = f"k{start}", 0.0


async def _service_with_footage(tmp_path, build_result, **overrides):
    """A service whose segments cover the window and whose builder is stubbed.

    The real builder needs ffmpeg and recorded segments; these tests only care what
    happens to the clip path AFTER it is built. Saves a real alert row and returns its
    id too, because _resolve silently skips an event that does not exist -- asserting
    on clip_status without one would pass vacuously.
    """
    store = EventStore({"events": {"db_path": str(tmp_path / "e.db"), "retention_hours": 48}})
    await store.init()
    conn = _StubConn()
    segments = _StubSegments([_StubSegRef(90.0, 110.0)])
    svc = AlertClipService(_cfg(**overrides), segment_store=segments,
                           clip_store=_StubClipStore(), events_store=store,
                           conn_manager=conn, build=lambda *a, **k: build_result)
    record = await store.save("alert", "Zone intrusion detected on cam0",
                              ["zone_intrusion"], camera="cam0", clip_status="pending")
    return svc, store, conn, record["id"]


def test_overlay_is_applied_to_a_zoned_alert(tmp_path, monkeypatch):
    seen = {}

    def fake_apply(clip_path, png, dest_dir):
        seen["clip"], seen["png"] = clip_path, png
        return clip_path + ".overlaid.mp4"

    monkeypatch.setattr(alert_clips_module, "probe_dimensions", lambda p: (640, 480))
    monkeypatch.setattr(alert_clips_module, "zone_overlay_png",
                        lambda pts, w, h, dest, mode: dest)
    monkeypatch.setattr(alert_clips_module, "apply_overlay", fake_apply)

    async def run():
        svc, _store, _conn, eid = await _service_with_footage(
            tmp_path, ("/tmp/clip.mp4", 95.0, 105.0))
        await svc.submit(eid, "cam0", 100.0, _ZONE)
        await svc._build(svc.jobs[0])
        return svc._clips.added

    added = asyncio.run(run())
    assert seen["clip"] == "/tmp/clip.mp4"
    assert added[0][0] == {"cam0": "/tmp/clip.mp4.overlaid.mp4"}


def test_no_overlay_when_draw_zones_is_off(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(alert_clips_module, "apply_overlay",
                        lambda *a: calls.append(a) or "should-not-be-used")

    async def run():
        svc, _store, _conn, eid = await _service_with_footage(
            tmp_path, ("/tmp/clip.mp4", 95.0, 105.0), draw_zones=False)
        await svc.submit(eid, "cam0", 100.0, _ZONE)
        await svc._build(svc.jobs[0])
        return svc._clips.added

    added = asyncio.run(run())
    assert calls == []
    assert added[0][0] == {"cam0": "/tmp/clip.mp4"}


@pytest.mark.parametrize("zone_free, kind, drawn", [
    (None, "proximity", False), (None, "zone_intrusion", True), (None, None, True),
    ([], "proximity", True), (["fire"], "fire", False)])
def test_the_zone_free_kinds_get_no_zone_outline(tmp_path, monkeypatch, zone_free, kind, drawn):
    calls = []
    monkeypatch.setattr(alert_clips_module, "probe_dimensions", lambda p: (640, 480))
    monkeypatch.setattr(alert_clips_module, "zone_overlay_png", lambda pts, w, h, dest, mode: dest)
    monkeypatch.setattr(alert_clips_module, "apply_overlay", lambda *a: calls.append(a) or "/tmp/zoned.mp4")
    overrides = {} if zone_free is None else {"zone_free_kinds": zone_free}

    async def run():
        svc, _store, _conn, eid = await _service_with_footage(tmp_path, ("/tmp/clip.mp4", 95.0, 105.0), **overrides)
        await svc.submit(eid, "cam0", 100.0, _ZONE, kind=kind)
        await svc._build(svc.jobs[0])
        return svc._clips.added

    added = asyncio.run(run())
    assert added[0][0] == {"cam0": "/tmp/zoned.mp4" if drawn else "/tmp/clip.mp4"} and len(calls) == int(drawn)


def test_a_zone_free_alert_never_shares_a_clip_that_draws_its_zone(tmp_path):
    async def run():
        svc, _store, _conn = await _service(tmp_path)
        await svc.submit(1, "cam0", 100.0, _ZONE)
        await svc.submit(2, "cam0", 100.5, _ZONE, kind="proximity")
        await svc.submit(3, "cam0", 101.0, _ZONE, kind="proximity")
        return [j["event_ids"] for j in svc.jobs]

    assert asyncio.run(run()) == [[1], [2, 3]]


def test_no_overlay_for_an_alert_without_a_zone(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(alert_clips_module, "apply_overlay",
                        lambda *a: calls.append(a) or "should-not-be-used")

    async def run():
        svc, _store, _conn, eid = await _service_with_footage(
            tmp_path, ("/tmp/clip.mp4", 95.0, 105.0))
        await svc.submit(eid, "cam0", 100.0)
        await svc._build(svc.jobs[0])
        return svc._clips.added

    added = asyncio.run(run())
    assert calls == []
    assert added[0][0] == {"cam0": "/tmp/clip.mp4"}


def test_a_failed_overlay_still_delivers_the_plain_clip(tmp_path, monkeypatch):
    monkeypatch.setattr(alert_clips_module, "probe_dimensions", lambda p: (640, 480))
    monkeypatch.setattr(alert_clips_module, "zone_overlay_png",
                        lambda pts, w, h, dest, mode: dest)
    monkeypatch.setattr(alert_clips_module, "apply_overlay", lambda *a: None)

    async def run():
        svc, store, _conn, eid = await _service_with_footage(
            tmp_path, ("/tmp/clip.mp4", 95.0, 105.0))
        await svc.submit(eid, "cam0", 100.0, _ZONE)
        await svc._build(svc.jobs[0])
        return svc._clips.added, await store.get_recent(limit=5)

    added, rows = asyncio.run(run())
    assert added[0][0] == {"cam0": "/tmp/clip.mp4"}      # the plain clip, not lost
    assert rows[0]["clip_status"] == "ready"


def test_an_exception_in_the_overlay_still_delivers_the_plain_clip(tmp_path, monkeypatch):
    def boom(*_a):
        raise RuntimeError("overlay exploded")

    monkeypatch.setattr(alert_clips_module, "probe_dimensions", boom)

    async def run():
        svc, store, _conn, eid = await _service_with_footage(
            tmp_path, ("/tmp/clip.mp4", 95.0, 105.0))
        await svc.submit(eid, "cam0", 100.0, _ZONE)
        await svc._build(svc.jobs[0])
        return svc._clips.added, await store.get_recent(limit=5)

    added, rows = asyncio.run(run())
    assert added[0][0] == {"cam0": "/tmp/clip.mp4"}
    assert rows[0]["clip_status"] == "ready"


# ── Smart Video Record path ───────────────────────────────────────────────────
# RTSP sources record via nvurisrcbin's smart-record, which writes nothing until an
# alert fires. Their jobs never become "covered" in the segment index, so they are
# resolved by the sidecar's sr-done announcement instead of by the index poll.

def test_attach_svr_file_resolves_queued_jobs_for_that_source(tmp_path):
    mp4 = tmp_path / "cam0_20260809.mp4"
    mp4.write_bytes(b"\x00\x00\x00\x18ftypmp42")     # contents are never parsed here

    async def run():
        svc, store, conn = await _service(tmp_path)
        svc.mark_svr_source("cam0")
        rec = await store.save("alert", "x", [], camera="cam0", clip_status="pending")
        await svc.submit(rec["id"], "cam0", 100.0)
        ok = svc.attach_svr_file("cam0", str(mp4))
        await asyncio.gather(*svc._resolve_tasks)    # attach_svr_file schedules the resolve
        return ok, svc.jobs, await store.get_recent(limit=10), conn.events, svc._clips.added

    ok, jobs, rows, broadcasts, added = asyncio.run(run())
    assert ok is True
    assert jobs == [], "the job must be dequeued once resolved"
    assert rows[-1]["clip_status"] == "ready"
    assert rows[-1]["clip_id"] == "clip1"
    assert broadcasts[-1]["clip_status"] == "ready"
    assert added[0][0] == {"cam0": str(mp4)}
    assert added[0][1]["source"] == "svr"


def test_attach_svr_file_ignores_a_source_with_no_queued_job(tmp_path):
    mp4 = tmp_path / "cam1_x.mp4"
    mp4.write_bytes(b"\x00")

    async def run():
        svc, _store, _conn = await _service(tmp_path)
        svc.mark_svr_source("cam1")
        return svc.attach_svr_file("cam1", str(mp4))

    assert asyncio.run(run()) is False


def test_store_standalone_stores_a_clip_with_no_queued_job(tmp_path):
    # A fire incident's follow-up clip: no job is waiting on it (its first clip
    # already resolved), so it must still get stored rather than dropped.
    mp4 = tmp_path / "cam0_followup.mp4"
    mp4.write_bytes(b"\x00\x00\x00\x18ftypmp42")

    async def run():
        svc, _store, _conn = await _service(tmp_path)
        return await svc.store_standalone("cam0", str(mp4)), svc._clips.added

    (status, clip_id), added = asyncio.run(run())
    assert status == "ready"
    assert clip_id == "clip1"
    assert added[0][0] == {"cam0": str(mp4)}


def test_attach_svr_file_resolves_every_coalesced_event_on_the_job(tmp_path):
    mp4 = tmp_path / "cam0_y.mp4"
    mp4.write_bytes(b"\x00")

    async def run():
        svc, store, _conn = await _service(tmp_path)
        svc.mark_svr_source("cam0")
        a = await store.save("alert", "a", [], camera="cam0", clip_status="pending")
        b = await store.save("alert", "b", [], camera="cam0", clip_status="pending")
        await svc.submit(a["id"], "cam0", 100.0)
        await svc.submit(b["id"], "cam0", 100.5)     # coalesces onto the same job
        svc.attach_svr_file("cam0", str(mp4))
        await asyncio.gather(*svc._resolve_tasks)
        return await store.get_recent(limit=10)

    rows = asyncio.run(run())
    statuses = {r["message"]: r["clip_status"] for r in rows}
    assert statuses["a"] == "ready" and statuses["b"] == "ready"


def test_nearby_svr_alerts_share_one_job_like_every_other_source(tmp_path):
    """Seen in the field: two intrusions on one zone at the same pts opened two jobs
    with identical windows. find_coalescible matched state == "queued" only, and an
    SVR job is created "awaiting-svr", so SVR alerts never coalesced at all -- every
    alert of a burst opened its own job for the single file they all shared."""
    async def run():
        svc, _store, _conn = await _service(tmp_path)
        svc.mark_svr_source("cam0")
        await svc.submit(1, "cam0", 250.5)
        await svc.submit(2, "cam0", 250.5)     # same instant, same (absent) zone
        return svc.jobs

    jobs = asyncio.run(run())
    assert len(jobs) == 1, "identical anchors must not open two jobs"
    assert jobs[0]["event_ids"] == [1, 2]


def test_a_building_job_still_refuses_late_arrivals(tmp_path):
    """The state check must keep excluding builds -- that footage is already being
    cut, so a late alert needs its own clip."""
    async def run():
        svc, _store, _conn = await _service(tmp_path)
        await svc.submit(1, "cam0", 100.0)
        svc.jobs[0]["state"] = "building"
        await svc.submit(2, "cam0", 100.5)
        return svc.jobs

    jobs = asyncio.run(run())
    assert len(jobs) == 2
    assert jobs[0]["event_ids"] == [1] and jobs[1]["event_ids"] == [2]


def test_one_smart_record_file_resolves_every_outstanding_job_on_the_source(tmp_path):
    """The sidecar starts at most ONE session per source, so alerts landing inside a
    running one are deliberately given no recording of their own (SvrTrigger returns
    coalesce/skip). That single file is therefore the only footage they will ever
    have, and all of them must resolve from it -- sharing one clip.

    Uses the real ClipStore, not _StubClipStore: add() MOVES the file, and adding
    once per job found the path gone on the second one. The job had already been
    dequeued by then, so nothing was left to resolve its events or expire them at
    the deadline, and they sat on 'pending' for the life of the process.
    """
    mp4 = tmp_path / "cam0_20260812.mp4"
    mp4.write_bytes(b"\x00\x00\x00\x18ftypmp42")

    async def run():
        store = EventStore({"events": {"db_path": str(tmp_path / "e.db"),
                                       "retention_hours": 48}})
        await store.init()
        conn = _StubConn()
        svc = AlertClipService(
            _cfg(), segment_store=_StubSegments(),
            clip_store=ClipStore(ttl_seconds=600, root_dir=str(tmp_path / "clips")),
            events_store=store, conn_manager=conn)
        svc.mark_svr_source("cam0")
        a = await store.save("alert", "a", [], camera="cam0", clip_status="pending")
        b = await store.save("alert", "b", [], camera="cam0", clip_status="pending")
        await svc.submit(a["id"], "cam0", 100.0)
        await svc.submit(b["id"], "cam0", 104.0)   # past coalesce_seconds -> its own job
        assert len(svc.jobs) == 2, "precondition: two separate jobs"
        svc.attach_svr_file("cam0", str(mp4))
        await asyncio.gather(*svc._resolve_tasks)
        return svc, await store.get_recent(limit=10)

    svc, rows = asyncio.run(run())
    by_message = {r["message"]: r for r in rows}
    assert by_message["a"]["clip_status"] == "ready"
    assert by_message["b"]["clip_status"] == "ready"
    assert by_message["a"]["clip_id"] == by_message["b"]["clip_id"], \
        "one recording is one clip; the file can only be moved into the store once"
    assert svc.jobs == []
    assert svc._clips.get_clip_path(by_message["a"]["clip_id"], "cam0") is not None


def test_a_clip_store_failure_resolves_svr_events_unavailable_not_pending(tmp_path):
    """Dequeuing a job puts it past the deadline sweep, so the store failure that
    follows must resolve its events itself -- gate the pending on the resolver."""
    class _Boom:
        def add(self, *_a, **_k):
            raise OSError("disk full")

    async def run():
        store = EventStore({"events": {"db_path": str(tmp_path / "e.db"),
                                       "retention_hours": 48}})
        await store.init()
        conn = _StubConn()
        svc = AlertClipService(_cfg(), segment_store=_StubSegments(), clip_store=_Boom(),
                               events_store=store, conn_manager=conn)
        svc.mark_svr_source("cam0")
        rec = await store.save("alert", "x", [], camera="cam0", clip_status="pending")
        await svc.submit(rec["id"], "cam0", 100.0)
        svc.attach_svr_file("cam0", "/nonexistent/cam0.mp4")
        await asyncio.gather(*svc._resolve_tasks)
        return svc.jobs, await store.get_recent(limit=10)

    jobs, rows = asyncio.run(run())
    assert jobs == []
    assert rows[-1]["clip_status"] == "unavailable"


def test_svr_job_is_marked_awaiting_svr_not_queued(tmp_path):
    """An SVR job must not enter the segment-index path -- there are no segments to
    find, and _next_ready would hand it to _build the moment its deadline passed."""
    async def run():
        svc, _store, _conn = await _service(tmp_path)
        svc.mark_svr_source("cam0")
        await svc.submit(1, "cam0", 100.0)
        return svc.jobs[0]["state"]

    assert asyncio.run(run()) == "awaiting-svr"


def test_unmarked_source_still_uses_the_segment_index_path(tmp_path):
    async def run():
        svc, _store, _conn = await _service(tmp_path)
        await svc.submit(1, "cam0", 100.0)
        return svc.jobs[0]["state"]

    assert asyncio.run(run()) == "queued"


def test_svr_job_past_its_deadline_resolves_unavailable(tmp_path):
    """A lost sr-done (sidecar restart, dropped Redis entry) must not strand the
    event on 'pending' for the life of the process."""
    async def run():
        svc, store, _conn = await _service(tmp_path, max_wait_seconds=0.0)
        svc.mark_svr_source("cam0")
        rec = await store.save("alert", "x", [], camera="cam0", clip_status="pending")
        await svc.submit(rec["id"], "cam0", 100.0)
        picked = await svc._next_ready()             # the expiry sweep lives here
        return picked, svc.jobs, await store.get_recent(limit=10)

    picked, jobs, rows = asyncio.run(run())
    assert picked is None, "an SVR job must never be handed to the segment builder"
    assert jobs == []
    assert rows[-1]["clip_status"] == "unavailable"


def _write_mkv(path, duration=1.0):
    """A real Matroska recording — what Smart Record writes now that MP4 is out.

    Real bytes, not a stub: this path runs an actual ffmpeg rewrap, so a fake
    header would only prove the error branch.
    """
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i",
         f"testsrc=duration={duration}:size=64x64:rate=10", "-c:v", "libx264",
         "-g", "10", "-keyint_min", "10", "-sc_threshold", "0",
         "-pix_fmt", "yuv420p", "-f", "matroska", str(path)],
        check=True, capture_output=True,
    )


def test_attach_svr_file_rewraps_a_matroska_recording_before_storing(tmp_path):
    """ClipStore names every clip `<camera>.mp4` and the browser plays it straight
    from disk, so Matroska bytes stored under that name would be an unplayable
    clip — ready, attached, and silently broken in the UI."""
    mkv = tmp_path / "cam0_20260824.mkv"
    _write_mkv(mkv)

    async def run():
        svc, store, _conn = await _service(tmp_path)
        svc.mark_svr_source("cam0")
        rec = await store.save("alert", "x", [], camera="cam0", clip_status="pending")
        await svc.submit(rec["id"], "cam0", 100.0)
        svc.attach_svr_file("cam0", str(mkv))
        await asyncio.gather(*svc._resolve_tasks)
        return svc._clips.added, await store.get_recent(10)

    added, rows = asyncio.run(run())
    stored = added[0][0]["cam0"]
    assert stored.endswith(".mp4"), "a Matroska file must not be stored as the clip"
    assert stored != str(mkv)
    assert rows[-1]["clip_status"] == "ready"


def test_attach_svr_file_deletes_the_recording_it_rewrapped(tmp_path):
    """ClipStore.add MOVES what it is given, so before the rewrap the recording
    left svr_dir by itself. It no longer does — the move takes the temp copy — and
    an original left behind on every alert is how a 30-minute retention once
    accumulated three weeks of files."""
    mkv = tmp_path / "cam0_20260824.mkv"
    _write_mkv(mkv)

    async def run():
        svc, store, _conn = await _service(tmp_path)
        svc.mark_svr_source("cam0")
        rec = await store.save("alert", "x", [], camera="cam0", clip_status="pending")
        await svc.submit(rec["id"], "cam0", 100.0)
        svc.attach_svr_file("cam0", str(mkv))
        await asyncio.gather(*svc._resolve_tasks)

    asyncio.run(run())
    assert not mkv.exists()


def test_attach_svr_file_does_not_block_the_event_loop_while_rewrapping(tmp_path):
    """The rewrap is ffmpeg, and this runs on the Redis consumer task — the same
    loop serving SSE chat tokens and WS broadcasts. Measured at ~90ms for one
    recording, which is small but is exactly the hot-path ffmpeg this codebase
    already keeps on its own worker."""
    mkv = tmp_path / "cam0.mkv"
    _write_mkv(mkv)
    real_mp4 = tmp_path / "already.mp4"
    real_mp4.write_bytes(b"\x00\x00\x00\x18ftypmp42")

    def _slow_remux(src, dest_dir):
        time.sleep(0.3)                 # stands in for a large recording
        return str(real_mp4)

    async def run():
        svc, store, _conn = await _service(tmp_path)
        svc.mark_svr_source("cam0")
        rec = await store.save("alert", "x", [], camera="cam0", clip_status="pending")
        await svc.submit(rec["id"], "cam0", 100.0)

        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        t = asyncio.create_task(ticker())
        await asyncio.sleep(0)                      # let the ticker start
        svc.attach_svr_file("cam0", str(mkv))       # must return without blocking
        await asyncio.gather(*svc._resolve_tasks)
        t.cancel()
        return ticks

    with mock.patch.object(alert_clips_module, "remux_to_mp4", _slow_remux):
        ticks = asyncio.run(run())
    assert ticks > 10, (
        f"loop advanced only {ticks} ticks during a 0.3s rewrap — it was blocked")


def test_an_unexpected_rewrap_failure_still_resolves_the_events(tmp_path):
    """Moving the store into a task made a raised exception easy to lose: the
    events would sit on 'pending' for the life of the process, which is the exact
    failure this path's docstring exists to warn about."""
    mkv = tmp_path / "cam0.mkv"
    _write_mkv(mkv)

    def _boom(src, dest_dir):
        raise RuntimeError("ffmpeg vanished")

    async def run():
        svc, store, _conn = await _service(tmp_path)
        svc.mark_svr_source("cam0")
        rec = await store.save("alert", "x", [], camera="cam0", clip_status="pending")
        await svc.submit(rec["id"], "cam0", 100.0)
        svc.attach_svr_file("cam0", str(mkv))
        await asyncio.gather(*svc._resolve_tasks, return_exceptions=True)
        return await store.get_recent(10)

    with mock.patch.object(alert_clips_module, "remux_to_mp4", _boom):
        rows = asyncio.run(run())
    assert rows[-1]["clip_status"] == "unavailable", "an alert must never stay pending"


def test_a_smart_record_clip_draws_its_alert_s_object_placed_by_wall_time(tmp_path):
    """The safety stack records no continuous segments: its clips are smart-record files. Each overlay entry's id is
    its write time, and the file ends at the sidecar's sr-done (`recorded_at`) and lasts its duration -- so the entries
    are placed by wall time, and the alert's own object (its pair) is burned in."""
    import json
    import shutil
    import time

    import cv2
    import fakeredis

    mkv = tmp_path / "cam0_20261109.mkv"
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=128x72:d=2:r=10", "-c:v", "libx264",
                    "-g", "10", "-pix_fmt", "yuv420p", "-f", "matroska", str(mkv)], check=True, capture_output=True)
    recorded_at = time.time()
    redis = fakeredis.FakeStrictRedis()
    for back in (1.6, 1.1, 0.6, 0.1):                       # the subject (7) and someone else (9), all along
        shapes = [{"type": "box", "rect": [10, 10, 40, 40], "style": "alert", "id": 7},
                  {"type": "box", "rect": [70, 10, 40, 40], "style": "alert", "id": 9}]
        redis.xadd("iris:overlay:cam0", {"v": "1", "module": "ppe", "pts": f"{500 - back:.3f}", "w": "128",
                                         "h": "72", "shapes": json.dumps(shapes)},
                   id=f"{int((recorded_at - back) * 1000)}-0")

    async def run():
        store = EventStore({"events": {"db_path": str(tmp_path / "e.db"), "retention_hours": 48}})
        await store.init()
        class _KeepingClipStore(_StubClipStore):            # keeps a copy: the stored file lives in a temp dir
            def add(self, cameras_mp4_paths, meta=None):
                kept = {cam: shutil.copy(p, tmp_path / f"stored-{cam}.mp4") for cam, p in cameras_mp4_paths.items()}
                return super().add(kept, meta)
        svc = AlertClipService(_cfg(overlay_settle_seconds=0), segment_store=_StubSegments(),
                               clip_store=_KeepingClipStore(), events_store=store, conn_manager=_StubConn(),
                               redis_client=redis)
        svc.svr_overlays = True                                # the safety product (app/server.py)
        svc.mark_svr_source("cam0")
        rec = await store.save("alert", "x", [], camera="cam0", clip_status="pending")
        await svc.submit(rec["id"], "cam0", 499.0, pair=({7}, set()))
        svc.attach_svr_file("cam0", str(mkv), recorded_at=recorded_at)
        await asyncio.gather(*svc._resolve_tasks)
        return svc._clips.added

    added = asyncio.run(run())
    cap = cv2.VideoCapture(added[0][0]["cam0"])
    cap.set(cv2.CAP_PROP_POS_MSEC, 1200)
    ok, frame = cap.read()
    assert ok
    b, g, r = (int(v) for v in frame[10, 30])                # the subject's top edge: drawn, in the alert red
    assert r > 150 and g < 140
    assert int(frame[10, 90][2]) < 60                       # the other person is not the alert's subject


def test_a_smart_record_clip_draws_its_alert_s_zone_as_the_zone_editor_does(tmp_path):
    """A zone alert's clip shows its zone (outline and 20% fill, the editor's colour), as segment-built clips do."""
    import shutil

    import cv2

    mkv = tmp_path / "cam0_20261110.mkv"
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=128x72:d=2:r=10", "-c:v", "libx264",
                    "-g", "10", "-pix_fmt", "yuv420p", "-f", "matroska", str(mkv)], check=True, capture_output=True)
    zone = {"id": "z0", "name": "Lab door", "points": [(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)],
            "intrusion": {"mode": "restricted"}}

    async def run():
        store = EventStore({"events": {"db_path": str(tmp_path / "e.db"), "retention_hours": 48}})
        await store.init()

        class _KeepingClipStore(_StubClipStore):
            def add(self, cameras_mp4_paths, meta=None):
                kept = {cam: shutil.copy(p, tmp_path / f"stored-{cam}.mp4") for cam, p in cameras_mp4_paths.items()}
                return super().add(kept, meta)
        svc = AlertClipService(_cfg(overlay_settle_seconds=0), segment_store=_StubSegments(),
                               clip_store=_KeepingClipStore(), events_store=store, conn_manager=_StubConn())
        svc.svr_overlays = True                                # the safety product (app/server.py)
        svc.mark_svr_source("cam0")
        rec = await store.save("alert", "x", [], camera="cam0", clip_status="pending")
        await svc.submit(rec["id"], "cam0", 499.0, zone=zone)
        svc.attach_svr_file("cam0", str(mkv))
        await asyncio.gather(*svc._resolve_tasks)
        return svc._clips.added

    cap = cv2.VideoCapture(asyncio.run(run())[0][0]["cam0"])
    ok, frame = cap.read()
    assert ok
    b, g, r = (int(v) for v in frame[18, 64])                  # the zone's top edge: the editor's restricted colour
    assert r > 180 and 100 < g < 190 and b < 150
    assert int(frame[36, 64].max()) > 15 and int(frame[5, 5].max()) < 10   # filled inside, nothing outside


def _svr_clip(tmp_path, jobs, svr_overlays=True, redis=None, settle=0, recorded_at=None):
    """Submit `jobs` ({"zone"?, "pair"?, "at"?} each) on a smart-record camera, attach one 2 s black file to them all,
    and return the stored file's frame at 1.2 s."""
    import shutil

    import cv2

    mkv = tmp_path / "cam0_shared.mkv"
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=128x72:d=2:r=10", "-c:v", "libx264",
                    "-g", "10", "-pix_fmt", "yuv420p", "-f", "matroska", str(mkv)], check=True, capture_output=True)

    async def run():
        store = EventStore({"events": {"db_path": str(tmp_path / "e.db"), "retention_hours": 48}})
        await store.init()

        class _KeepingClipStore(_StubClipStore):
            def add(self, cameras_mp4_paths, meta=None):
                kept = {cam: shutil.copy(p, tmp_path / f"stored-{cam}.mp4") for cam, p in cameras_mp4_paths.items()}
                return super().add(kept, meta)
        svc = AlertClipService(_cfg(overlay_settle_seconds=settle), segment_store=_StubSegments(),
                               clip_store=_KeepingClipStore(), events_store=store, conn_manager=_StubConn(),
                               redis_client=redis)
        svc.svr_overlays = svr_overlays
        svc.mark_svr_source("cam0")
        for job in jobs:
            rec = await store.save("alert", "x", [], camera="cam0", clip_status="pending")
            await svc.submit(rec["id"], "cam0", job.get("at", 499.0), zone=job.get("zone"), pair=job.get("pair"))
        svc.attach_svr_file("cam0", str(mkv), recorded_at=recorded_at)
        await asyncio.gather(*svc._resolve_tasks)
        return svc._clips.added

    cap = cv2.VideoCapture(asyncio.run(run())[0][0]["cam0"])
    cap.set(cv2.CAP_PROP_POS_MSEC, 1200)
    ok, frame = cap.read()
    assert ok
    return frame


_LEFT = {"id": "z0", "name": "Door", "points": [(0.05, 0.25), (0.4, 0.25), (0.4, 0.75), (0.05, 0.75)],
         "intrusion": {"mode": "restricted"}}
_RIGHT = {"id": "z1", "name": "Dock", "points": [(0.6, 0.25), (0.95, 0.25), (0.95, 0.75), (0.6, 0.75)],
          "intrusion": {"mode": "restricted"}}


def test_another_product_s_smart_record_clip_is_stored_as_recorded(tmp_path):
    """Drawing on smart-record clips is the safety product's: every other product's stay as they were recorded."""
    frame = _svr_clip(tmp_path, [{"zone": _LEFT}], svr_overlays=False)
    assert int(frame[18, 30].max()) < 20                      # no zone outline


def test_a_shared_smart_record_file_outlines_every_alert_s_zone(tmp_path):
    """Two zone alerts close together share the one smart-record session: the clip both events get shows both zones,
    not only the first alert's."""
    frame = _svr_clip(tmp_path, [{"zone": _LEFT}, {"zone": _RIGHT, "at": 499.5}])
    assert int(frame[18, 28][2]) > 180 and int(frame[18, 100][2]) > 180


def test_an_alert_about_everyone_keeps_everyone_on_a_shared_clip(tmp_path):
    """An alert with no subject (overcrowding: everyone counted) sharing a file with one about #7: everyone is drawn,
    not only #7."""
    import json
    import time

    import fakeredis
    recorded_at = time.time()
    redis = fakeredis.FakeStrictRedis()
    for back in (1.6, 1.1, 0.6, 0.1):
        shapes = [{"type": "box", "rect": [10, 10, 40, 40], "style": "alert", "id": 7},
                  {"type": "box", "rect": [70, 10, 40, 40], "style": "alert", "id": 9}]
        redis.xadd("iris:overlay:cam0", {"v": "1", "module": "zones", "pts": f"{500 - back:.3f}", "w": "128",
                                         "h": "72", "shapes": json.dumps(shapes)},
                   id=f"{int((recorded_at - back) * 1000)}-0")
    frame = _svr_clip(tmp_path, [{"pair": ({7}, set())}, {"at": 499.5}], redis=redis, recorded_at=recorded_at)
    assert int(frame[10, 30][2]) > 150 and int(frame[10, 90][2]) > 150


def test_the_clip_waits_for_the_boxes_of_its_last_frames(tmp_path, monkeypatch):
    """An overlay entry is written once its frame's inference is done, after the frame arrived: the clip's last
    frames' boxes come up to that delay after the file ends, so the clip waits for them before drawing."""
    import time
    waited = []
    monkeypatch.setattr(alert_clips_module.time, "sleep", lambda s: waited.append(s))
    # a file that ends later than now (however long making the test clip took): the wait is capped at the settle time
    _svr_clip(tmp_path, [{"zone": _LEFT}], redis=__import__("fakeredis").FakeStrictRedis(), settle=1.5,
              recorded_at=time.time() + 30)
    assert waited == [1.5]

