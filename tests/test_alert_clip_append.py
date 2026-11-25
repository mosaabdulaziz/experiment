"""An incident's escalation clip is a follow-up: appended to its clips, never its first clip, never shared."""
import asyncio

from app.alert_clips import AlertClipService, find_coalescible
from app.events import EventStore


class _Conn:
    def __init__(self):
        self.events = []

    async def broadcast(self, payload):
        self.events.append(payload)


class _Ref:
    def __init__(self, start, end):
        self.start_pts, self.end_pts, self.object_key, self.created = start, end, f"k{start}", 0.0


class _Segments:
    def list_range(self, source_id, lo, hi):
        return [_Ref(80.0, 130.0)]


class _Clips:
    def __init__(self):
        self.n = 0

    def add(self, paths, meta=None):
        self.n += 1
        return f"clip{self.n}"


async def _service(tmp_path, result=("/tmp/c.mp4", 0.0, 10.0)):
    store = EventStore({"events": {"db_path": str(tmp_path / "e.db"), "retention_hours": 48}})
    await store.init()
    conn = _Conn()
    cfg = {"alerts": {"clip": {"enabled": True, "pad_seconds": 5.0, "coalesce_seconds": 2.5,
                               "max_wait_seconds": 20.0, "queue_depth": 8}}}
    svc = AlertClipService(cfg, segment_store=_Segments(), clip_store=_Clips(), events_store=store,
                           conn_manager=conn, build=lambda *a, **k: result)
    return svc, store, conn


def test_an_append_job_never_coalesces_either_way(tmp_path):
    jobs = [{"sid": "c", "state": "queued", "anchor": 100.0, "zone": None, "append": True}]
    assert find_coalescible(jobs, "c", 100.5, 2.5) is None             # nothing joins a follow-up job

    async def run():
        svc, store, _ = await _service(tmp_path)
        a = await store.save("alert", "a", [], camera="c", clip_status="pending")
        await svc.submit(a["id"], "c", 100.0, wall_time=100.0)
        await svc.submit(a["id"], "c", 101.0, wall_time=101.0, append=True)   # and it joins nothing
        return svc.jobs

    jobs = asyncio.run(run())
    assert len(jobs) == 2 and [j.get("append", False) for j in jobs] == [False, True]


def test_an_escalation_clip_is_appended_and_never_replaces_the_first(tmp_path):
    async def run():
        svc, store, conn = await _service(tmp_path)
        a = await store.save("alert", "a", [], camera="c", clip_status="pending")
        await svc.submit(a["id"], "c", 100.0, wall_time=100.0)
        await svc._build(svc.jobs.pop(0))
        await svc.submit(a["id"], "c", 106.0, wall_time=106.0, append=True)
        await svc._build(svc.jobs.pop(0))
        return await store.get_event(a["id"]), conn.events

    ev, broadcasts = asyncio.run(run())
    assert ev["clip_id"] == "clip1" and [c["clip_id"] for c in ev["clips"]] == ["clip1", "clip2"]
    assert broadcasts[-1] == {"type": "event_update", "id": ev["id"], "clips": ev["clips"]}


def test_a_failed_escalation_clip_leaves_the_event_alone(tmp_path):
    async def run():
        svc, store, conn = await _service(tmp_path, result=None)
        a = await store.save("alert", "a", [], camera="c", clip_status="ready")
        await svc.submit(a["id"], "c", 106.0, wall_time=106.0, append=True)
        await svc._build(svc.jobs.pop(0))
        return await store.get_event(a["id"]), conn.events

    ev, broadcasts = asyncio.run(run())
    assert ev["clip_status"] == "ready" and ev["clips"] == [] and broadcasts == []


def test_no_follow_up_clip_on_a_smart_record_source_or_with_clips_off(tmp_path):
    async def run():
        svc, store, conn = await _service(tmp_path)
        svc.mark_svr_source("c")
        a = await store.save("alert", "a", [], camera="c", clip_status="ready")
        first = await svc.submit(a["id"], "c", 106.0, wall_time=106.0, append=True)
        svc.enabled = False
        second = await svc.submit(a["id"], "d", 106.0, wall_time=106.0, append=True)
        return first, second, await store.get_event(a["id"]), conn.events

    first, second, ev, broadcasts = asyncio.run(run())
    assert (first, second) == (False, False) and ev["clip_status"] == "ready" and broadcasts == []


def test_the_first_clip_resolving_after_the_escalation_clip_announces_the_whole_clip_list(tmp_path):
    async def run():
        svc, store, conn = await _service(tmp_path)
        a = await store.save("alert", "a", [], camera="c", clip_status="pending")
        await store.append_clip(a["id"], "escalation")             # its follow-up clip finished first
        await svc.submit(a["id"], "c", 100.0, wall_time=100.0)
        await svc._build(svc.jobs.pop(0))
        return conn.events[-1]

    last = asyncio.run(run())
    assert last["clip_status"] == "ready" and [c["clip_id"] for c in last["clips"]] == ["clip1", "escalation"]

