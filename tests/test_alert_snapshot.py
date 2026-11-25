"""The snapshot at an incident's minimum gap: a frame of one of its clips, with the pair's boxes."""
import asyncio
import json
import subprocess

import cv2
import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient

from app import overlay_track as ot
from app.alert_clips import AlertClipService
from app.clip_store import ClipStore
from app.events import EventStore


def test_the_clip_holding_the_moment_is_picked_else_the_nearest_frame():
    clips = [("a", {"anchor": 100.0, "pts": 50.0, "range": [95.0, 105.0]}),
             ("b", {"anchor": 120.0, "pts": 70.0, "range": [115.0, 125.0]}),
             ("bad", {"anchor": 1.0, "pts": None, "range": [0.0, 1.0]})]
    assert ot.pick_clip(clips, 52.0) == ("a", 7.0, True)
    clip, offset, exact = ot.pick_clip(clips, 60.0)                  # 110 on the wall clock: past a's end, nearer a
    assert (clip, exact) == ("a", False) and offset == pytest.approx(9.9)    # 0.1 s inside its last frame
    assert ot.pick_clip([], 60.0) is None


def test_the_nearest_entry_within_the_hold_draws_the_boxes():
    entries = [{"pts": 50.0}, {"pts": 50.3}, {"pts": 52.0}]
    assert ot.nearest(entries, 50.2)["pts"] == 50.3 and ot.nearest(entries, 51.2) is None


class _Conn:
    def __init__(self):
        self.events = []

    async def broadcast(self, payload):
        self.events.append(payload)


class _Redis:
    def xrange(self, key, lo, hi):
        shapes = [{"type": "box", "rect": [0, 0, 640, 360], "style": "person", "id": 6}]
        return [(b"1-0", {b"pts": b"50.5", b"w": b"1280", b"h": b"720", b"shapes": json.dumps(shapes).encode()})]

    def get(self, key):
        return None


def _mp4(path):
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=2:r=10", "-pix_fmt", "yuv420p",
                    str(path)], check=True, capture_output=True)
    return str(path)


async def _setup(tmp_path, meta):
    store = EventStore({"events": {"db_path": str(tmp_path / "e.db"), "retention_hours": 48}})
    await store.init()
    clips = ClipStore(root_dir=str(tmp_path / "clips"))
    clip_id = clips.add({"cam0": _mp4(tmp_path / "c.mp4")}, meta=meta)
    ev = await store.save("alert", "m", ["proximity"], camera="cam0", clip_status="pending")
    await store.update_clip(ev["id"], "ready", clip_id)
    conn = _Conn()
    cfg = {"alerts": {"clip": {"enabled": True, "pad_seconds": 5.0, "max_wait_seconds": 1.0}}}
    svc = AlertClipService(cfg, segment_store=object(), clip_store=clips, events_store=store, conn_manager=conn,
                           redis_client=_Redis())
    return svc, store, clips, conn, ev["id"], clip_id


def test_a_snapshot_is_taken_stored_with_its_clip_and_announced(tmp_path):
    async def run():
        svc, store, clips, conn, eid, clip_id = await _setup(
            tmp_path, {"anchor": 100.0, "pts": 50.0, "range": [99.0, 101.0]})
        svc.snapshot(eid, "cam0", 50.5, [6], [10])
        await svc._take_snapshots()
        return await store.get_event(eid), clips, clip_id, conn.events, svc.snapshots

    ev, clips, clip_id, broadcasts, queue = asyncio.run(run())
    assert ev["snapshot_id"] == clip_id and ev["details"]["snapshot_exact"] == "1" and queue == []
    img = cv2.imread(clips.get_snapshot_path(clip_id, "cam0", ev["id"]))
    assert img is not None and img[1, 10, 1] > 180 and img[1, 10, 2] < 150      # the person's box, in green
    assert broadcasts[-1]["type"] == "event_update" and broadcasts[-1]["event"]["snapshot_id"] == clip_id


def test_a_moment_outside_every_clip_gets_the_closest_frame_flagged(tmp_path):
    async def run():
        svc, store, *_rest, eid, _cid = await _setup(tmp_path, {"anchor": 100.0, "pts": 50.0, "range": [99.0, 101.0]})
        svc.snapshot(eid, "cam0", 58.0, [6], [10])
        await svc._take_snapshots()
        return await store.get_event(eid)

    assert asyncio.run(run())["details"]["snapshot_exact"] == "0"


def test_an_event_without_clips_gets_no_snapshot_and_leaves_the_queue(tmp_path):
    async def run():
        svc, store, *_rest, _eid, _cid = await _setup(tmp_path, {"anchor": 100.0, "pts": 50.0, "range": [99.0, 101.0]})
        bare = await store.save("alert", "m", [], camera="cam0", clip_status="unavailable")
        svc.snapshot(bare["id"], "cam0", 50.5, [6], [10])
        await svc._take_snapshots()
        return await store.get_event(bare["id"]), svc.snapshots

    ev, queue = asyncio.run(run())
    assert ev["snapshot_id"] is None and queue == []


def test_the_snapshot_route_serves_the_jpeg(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)                                   # the app keeps alert clips under ./data/alert_clips
    sidecar = tmp_path / "sidecar.yaml"
    sidecar.write_text(yaml.safe_dump({"mux_width": 1280, "mux_height": 720, "sources": []}))
    cfg = {"perception": {"backend": "local"}, "api": {"backend": "cloud", "api_key": "x", "model": "m"},
           "monitoring": {}, "sampling": {}, "events": {"db_path": str(tmp_path / "e.db")},
           "chat": {"tools_enabled": False}, "cv": {"enabled": False}, "camera": {"source": 0},
           "zones": {"file_path": str(tmp_path / "z.txt"), "sidecar_config_path": str(sidecar)}}
    from app.server import create_app
    app = create_app(cfg)
    clips = ClipStore(root_dir="data/alert_clips")
    clip_id = clips.add({"cam0": _mp4(tmp_path / "c.mp4")}, meta={})
    jpg = tmp_path / "s.jpg"
    cv2.imwrite(str(jpg), np.zeros((8, 8, 3), np.uint8))
    assert clips.add_snapshot(clip_id, "cam0", str(jpg), 7)
    with TestClient(app) as client:
        ok = client.get(f"/api/alerts/snapshot/{clip_id}/cam0/7")
        missing = client.get(f"/api/alerts/snapshot/{clip_id}/cam0/8")
    assert ok.status_code == 200 and ok.headers["content-type"] == "image/jpeg"
    assert missing.status_code == 404


def test_two_incidents_sharing_a_clip_each_keep_their_own_snapshot(tmp_path):
    async def run():
        svc, store, clips, _conn, eid, clip_id = await _setup(
            tmp_path, {"anchor": 100.0, "pts": 50.0, "range": [99.0, 101.0]})
        other = await store.save("alert", "m2", ["proximity"], camera="cam0", clip_status="pending")
        await store.update_clip(other["id"], "ready", clip_id)          # coalesced onto the same clip
        svc.snapshot(eid, "cam0", 50.5, [6], [10])
        svc.snapshot(other["id"], "cam0", 49.5, [7], [11])
        await svc._take_snapshots()
        return clips, clip_id, eid, other["id"]

    clips, clip_id, first, second = asyncio.run(run())
    a, b = clips.get_snapshot_path(clip_id, "cam0", first), clips.get_snapshot_path(clip_id, "cam0", second)
    assert a and b and a != b
    assert cv2.imread(a)[1, 10, 1] > 180 and cv2.imread(b)[1, 10, 1] < 100    # only the first pair's box is in a


def test_one_failing_snapshot_does_not_hold_up_the_others(tmp_path):
    async def run():
        svc, store, clips, _conn, eid, clip_id = await _setup(
            tmp_path, {"anchor": 100.0, "pts": 50.0, "range": [99.0, 101.0]})
        broken = await store.save("alert", "m2", ["proximity"], camera="cam0", clip_status="pending")
        real = store.get_event

        async def get_event(event_id):
            if event_id == broken["id"]:
                raise RuntimeError("database hiccup")
            return await real(event_id)

        store.get_event = get_event
        svc.snapshot(broken["id"], "cam0", 50.5, [6], [10])
        svc.snapshot(eid, "cam0", 50.5, [6], [10])
        await svc._take_snapshots()
        return await real(eid), svc.snapshots

    ev, queue = asyncio.run(run())
    assert ev["snapshot_id"] is not None
    assert [s["event_id"] for s in queue] != [ev["id"]]          # the good one was taken, whatever became of the other

