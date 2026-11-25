"""The consumer against a fake Redis: frames in, rows out."""
import json

from config import CameraConfig, ServiceConfig
from consumer import ConsumerThread
from domain import EngineUpdate, Visit
from storage import apply_update, connect, list_visits
from zones import FrameGeometry

from datetime import datetime, timedelta


class FakeRedis:
    def __init__(self):
        self.streams = {}
        self.kv = {}

    def set(self, key, value):
        self.kv[key] = value

    def add(self, key, ms, fields):
        self.streams.setdefault(key, []).append((f"{ms}-0".encode(), fields))

    def xrevrange(self, key, count=1):
        return list(reversed(self.streams.get(key, [])))[:count]

    def xread(self, cursors, count=None, block=None):
        out = []
        for key, last in cursors.items():
            lo = tuple(int(p) for p in last.split("-"))
            rows = [(rid, f) for rid, f in self.streams.get(key, [])
                    if tuple(int(p) for p in rid.decode().split("-")) > lo][:count]
            if rows:
                out.append((key.encode(), rows))
        return out


def cfg(tmp_path):
    return ServiceConfig(redis_url="redis://unused", frame=FrameGeometry(1000, 1000),
                         cameras=(CameraConfig("barber-c"), CameraConfig("barber-l")),
                         data_dir=tmp_path, stream_starts_file=tmp_path / "starts.json",
                         crops_dir=tmp_path / "crops")


def box(pt, tid):
    return [pt[0] * 1000 - 50, pt[1] * 1000 - 120, 100, 150, tid]


def test_a_customer_waiting_on_the_sofa_reaches_the_database(tmp_path, monkeypatch):
    from shop_helpers import ZONES
    c = cfg(tmp_path)
    (tmp_path / "zones.json").write_text(json.dumps(ZONES))
    (tmp_path / "policy.json").write_text(json.dumps({"waiting_debounce_s": 2, "staff_min_samples": 1}))
    r = FakeRedis()
    now_ms = int(datetime.now().timestamp() * 1000)
    for cam in ("barber-c", "barber-l"):
        r.add(f"iris:meta:{cam}", now_ms - 10_000, {b"dets": b"[]"})
        r.add(f"iris:appearance:{cam}", now_ms - 10_000, {b"items": b"[]"})
    consumer = ConsumerThread(c)
    monkeypatch.setattr("consumer.live_watch.refresh", lambda redis_, cams: {k: datetime.now() for k in cams})
    consumer.attach(connect(str(c.db_path)), r)
    consumer.step()                                          # primes the cursors
    for i in range(6):
        ms = now_ms - 9_000 + i * 1000
        r.add("iris:meta:barber-c", ms, {b"dets": b"[]", b"person_boxes": json.dumps([box((0.5, 0.45), 2)]).encode()})
        r.add("iris:meta:barber-l", ms, {b"dets": b"[]"})
        r.add("iris:appearance:barber-c", ms + 500, {b"items": json.dumps([{"track_id": 2, "staff_p": 0.05}]).encode()})
    consumer.step()
    (v,) = list_visits(connect(str(c.db_path)), datetime.now() - timedelta(hours=1), datetime.now() + timedelta(hours=1))
    assert v["state"] == "waiting" and not v["entry_observed"]
    assert consumer.live()["customers_waiting"] == 1
    assert consumer.scene["barber-c"][0]["person_id"] == v["person_id"]


def test_attach_seeds_the_ticket_counter_from_the_days_max_ticket(tmp_path, monkeypatch):
    """A restart: the DB already has today's ticket #3. Once attach() has
    seeded the counter from the DB max, the first visit confirmed afterwards
    gets ticket #4."""
    from shop_helpers import ZONES
    c = cfg(tmp_path)
    (tmp_path / "zones.json").write_text(json.dumps(ZONES))
    (tmp_path / "policy.json").write_text(json.dumps({"waiting_debounce_s": 2, "staff_min_samples": 1}))
    now = datetime.now()
    conn = connect(str(c.db_path))
    apply_update(conn, EngineUpdate(visits={
        "v0": Visit("v0", "p0", now, True, state="finished", confirmed=True, ticket=3, open=False)}), now)
    r = FakeRedis()
    now_ms = int(now.timestamp() * 1000)
    for cam in ("barber-c", "barber-l"):
        r.add(f"iris:meta:{cam}", now_ms - 10_000, {b"dets": b"[]"})
        r.add(f"iris:appearance:{cam}", now_ms - 10_000, {b"items": b"[]"})
    consumer = ConsumerThread(c)
    monkeypatch.setattr("consumer.live_watch.refresh", lambda redis_, cams: {k: datetime.now() for k in cams})
    consumer.attach(conn, r)
    consumer.step()                                          # primes the cursors
    for i in range(6):
        ms = now_ms - 9_000 + i * 1000
        r.add("iris:meta:barber-c", ms, {b"dets": b"[]", b"person_boxes": json.dumps([box((0.5, 0.45), 2)]).encode()})
        r.add("iris:meta:barber-l", ms, {b"dets": b"[]"})
        r.add("iris:appearance:barber-c", ms + 500, {b"items": json.dumps([{"track_id": 2, "staff_p": 0.05}]).encode()})
    consumer.step()
    rows = list_visits(connect(str(c.db_path)), now - timedelta(hours=1), now + timedelta(hours=1))
    new = next(row for row in rows if row["id"] != "v0")
    assert new["ticket"] == 4


def test_face_candidates_reach_the_visit_pool(tmp_path, monkeypatch):
    from shop_helpers import ZONES
    import base64, struct
    c = cfg(tmp_path)
    (tmp_path / "zones.json").write_text(json.dumps(ZONES))
    (tmp_path / "policy.json").write_text(json.dumps({"waiting_debounce_s": 2, "staff_min_samples": 1}))
    r = FakeRedis()
    now_ms = int(datetime.now().timestamp() * 1000)
    for key in ("iris:meta:barber-c", "iris:meta:barber-l", "iris:appearance:barber-c", "iris:appearance:barber-l",
                "iris:customer_face:barber-c", "iris:customer_face:barber-l"):
        r.add(key, now_ms - 10_000, {b"dets": b"[]", b"items": b"[]", b"item": b"{}"})
    consumer = ConsumerThread(c)
    monkeypatch.setattr("consumer.live_watch.refresh", lambda redis_, cams: {k: datetime.now() for k in cams})
    consumer.attach(connect(str(c.db_path)), r)
    consumer.step()
    emb = base64.b64encode(struct.pack("<2e", 1.0, 0.0)).decode()
    for i in range(6):
        ms = now_ms - 9_000 + i * 1000
        r.add("iris:meta:barber-c", ms, {b"dets": b"[]", b"person_boxes": json.dumps([box((0.5, 0.45), 2)]).encode()})
        r.add("iris:meta:barber-l", ms, {b"dets": b"[]"})
        r.add("iris:appearance:barber-c", ms + 500, {b"items": json.dumps([{"track_id": 2, "staff_p": 0.05}]).encode()})
    r.add("iris:customer_face:barber-c", now_ms - 3_500, {b"item": json.dumps(
        {"track_id": 2, "crop_path": "faces/d/a.jpg", "embedding": emb, "quality": 0.7}).encode()})
    consumer.step()
    (pool,) = consumer.engine.pools.values()
    assert [f.crop_path for f in pool.faces] == ["faces/d/a.jpg"]


def test_start_sweeps_orphan_crops_older_than_a_day(tmp_path):
    import os, time
    c = cfg(tmp_path)
    old, fresh = c.crops_dir / "faces/2026-09-01/old.jpg", c.crops_dir / "bodies/2026-09-25/fresh.jpg"
    for p in (old, fresh):
        p.parent.mkdir(parents=True)
        p.write_bytes(b"x")
    os.utime(old, (time.time() - 2 * 86400,) * 2)
    ConsumerThread(c).attach(connect(str(c.db_path)), FakeRedis())
    assert not old.exists() and fresh.exists()


def test_a_failed_start_sweep_does_not_stop_the_consumer(tmp_path, monkeypatch):
    c = cfg(tmp_path)
    monkeypatch.setattr("consumer.crops.sweep_orphans", lambda *a: 1 / 0)
    consumer = ConsumerThread(c)
    consumer.attach(connect(str(c.db_path)), FakeRedis())
    assert consumer._conn is not None


def test_engine_class_follows_config(tmp_path):
    import dataclasses
    from chair_engine import ChairEngine
    from engine import ShopEngine
    base = cfg(tmp_path)
    assert type(ConsumerThread(base).engine) is ShopEngine
    chair = dataclasses.replace(base, engine="chair")
    assert isinstance(ConsumerThread(chair).engine, ChairEngine)


def test_attach_publishes_the_face_regions(tmp_path):
    from shop_helpers import ZONES
    (tmp_path / "zones.json").write_text(json.dumps(ZONES))
    r = FakeRedis()
    ConsumerThread(cfg(tmp_path)).attach(connect(str(tmp_path / "d.db")), r)
    assert "barber:face_regions" in r.kv
    json.loads(r.kv["barber:face_regions"])


def test_chair_mode_link_returns_follows_register_source(tmp_path, monkeypatch):
    """Chair crops (register_source 0): every haircut its own customer, link_returns 0; door faces (1): 1. Shop mode keeps the stored value."""
    import dataclasses
    from shop_helpers import ZONES
    (tmp_path / "zones.json").write_text(json.dumps(ZONES))
    monkeypatch.setattr("consumer.live_watch.refresh", lambda redis_, cams: {})
    seen = {}
    for source in (0, 1):
        (tmp_path / "policy.json").write_text(json.dumps({"link_returns": 0, "register_source": source}))
        for mode in ("shop", "chair"):
            consumer = ConsumerThread(dataclasses.replace(cfg(tmp_path), engine=mode))
            consumer.attach(connect(str(tmp_path / f"{mode}{source}.db")), FakeRedis())
            real = consumer.engine.observe

            def spy(frames, apps, policy, faces=(), real=real, mode=mode, source=source):
                seen[mode, source] = policy["link_returns"]
                return real(frames, apps, policy, faces=faces)
            monkeypatch.setattr(consumer.engine, "observe", spy)
            consumer.step()
    assert seen == {("shop", 0): 0.0, ("chair", 0): 0.0, ("shop", 1): 0.0, ("chair", 1): 1.0}


def _consumer_with_zones(tmp_path):
    from shop_helpers import ZONES
    (tmp_path / "zones.json").write_text(json.dumps(ZONES))
    return ConsumerThread(cfg(tmp_path))


def test_step_publishes_the_regions_when_zones_change(tmp_path, monkeypatch):
    monkeypatch.setattr("consumer.live_watch.refresh", lambda redis_, cams: {})
    consumer = ConsumerThread(cfg(tmp_path))
    r = FakeRedis()
    consumer.attach(connect(str(tmp_path / "d.db")), r)
    r.kv.clear()
    from shop_helpers import ZONES
    (tmp_path / "zones.json").write_text(json.dumps(ZONES))
    consumer.step()
    assert "barber:face_regions" in r.kv


def test_step_republishes_the_regions_when_register_source_changes(tmp_path, monkeypatch):
    monkeypatch.setattr("consumer.live_watch.refresh", lambda redis_, cams: {})
    face = {"id": "f1", "name": "f", "type": "face_area", "points": [[0, 0], [1, 0], [1, 1]], "chair": "c"}
    (tmp_path / "zones.json").write_text(json.dumps({"barber-c": [face]}))
    consumer = ConsumerThread(cfg(tmp_path))
    r = FakeRedis()
    consumer.attach(connect(str(tmp_path / "d.db")), r)
    consumer.step()
    assert json.loads(r.kv["barber:face_regions"]) == {}
    (tmp_path / "policy.json").write_text(json.dumps({"register_source": 2}))
    consumer.step()
    assert json.loads(r.kv["barber:face_regions"]) == {"match": "box", "regions": {"barber-c": [face["points"]]}}


def test_a_redis_error_on_publish_is_swallowed_and_retried(tmp_path, monkeypatch):
    import redis as redis_mod
    monkeypatch.setattr("consumer.live_watch.refresh", lambda redis_, cams: {})
    consumer = _consumer_with_zones(tmp_path)
    r = FakeRedis()
    real = r.set

    def broken(key, value):
        raise redis_mod.RedisError("down")
    r.set = broken
    consumer.attach(connect(str(tmp_path / "d.db")), r)
    consumer.step()   # zones changed + publish fails: must not raise
    assert "barber:face_regions" not in r.kv
    r.set = real
    consumer.step()
    assert "barber:face_regions" in r.kv


def test_regions_are_republished_after_30_seconds(tmp_path, monkeypatch):
    monkeypatch.setattr("consumer.live_watch.refresh", lambda redis_, cams: {})
    consumer = _consumer_with_zones(tmp_path)
    r = FakeRedis()
    consumer.attach(connect(str(tmp_path / "d.db")), r)
    consumer.step()
    r.kv.clear()
    consumer.step()
    assert "barber:face_regions" not in r.kv
    consumer._regions_at -= 31.0
    consumer.step()
    assert "barber:face_regions" in r.kv


# ── the start-of-day visit baseline ─────────────────────────────────────────

NOON = datetime(2026, 10, 6, 10, 0, 0)


def _baseline_consumer(tmp_path, monkeypatch, clock, cameras=None, settle_s=120.0):
    """settle_s: these tests pin the settle's timing at 120 s (the window length is BASELINE_SETTLE_S)."""
    from shop_helpers import ZONES
    import dataclasses
    monkeypatch.setattr("consumer.BASELINE_SETTLE_S", settle_s)
    c = dataclasses.replace(cfg(tmp_path), engine="chair")
    if cameras:
        c = dataclasses.replace(c, cameras=cameras)
        (tmp_path / "starts.json").write_text(json.dumps({k.id: datetime.now().timestamp() - 50 for k in cameras}))
    (tmp_path / "zones.json").write_text(json.dumps(ZONES))
    (tmp_path / "policy.json").write_text(json.dumps({"staff_min_samples": 1}))
    r = FakeRedis()
    now_ms = int(datetime.now().timestamp() * 1000)
    for cam in ("barber-c", "barber-l"):
        r.add(f"iris:meta:{cam}", now_ms - 200_000, {b"dets": b"[]"})
        r.add(f"iris:appearance:{cam}", now_ms - 200_000, {b"items": b"[]"})
    consumer = ConsumerThread(c)
    consumer.clock = lambda: clock[0]
    monkeypatch.setattr("consumer.live_watch.refresh", lambda redis_, cams: {k: datetime.now() for k in cams})
    consumer.attach(connect(str(c.db_path)), r)
    consumer.step()                                          # primes the cursors
    return consumer, r, now_ms


SOFA_MASK = [0.45, 0.35, 0.55, 0.35, 0.55, 0.5, 0.45, 0.5]


def _drain(consumer, n=4):
    for _ in range(n):               # a step reads a limited batch of frames
        consumer.step()


def _feed_waiter(r, start_ms, n=130):
    for i in range(n):
        ms = start_ms + i * 1000
        r.add("iris:meta:barber-c", ms, {b"dets": b"[]", b"person_boxes": json.dumps([box((0.5, 0.45), 2)]).encode(),
                                         b"person_masks": json.dumps([[2, 0.9, SOFA_MASK]]).encode()})
        r.add("iris:meta:barber-l", ms, {b"dets": b"[]"})
        r.add("iris:appearance:barber-c", ms + 500, {b"items": json.dumps([{"track_id": 2, "staff_p": 0.05}]).encode()})


def test_the_baseline_is_taken_two_minutes_after_start_not_before(tmp_path, monkeypatch):
    from storage import visit_baseline
    clock = [NOON]
    consumer, r, now_ms = _baseline_consumer(tmp_path, monkeypatch, clock)
    _feed_waiter(r, now_ms - 140_000)
    clock[0] = NOON + timedelta(seconds=119)
    _drain(consumer)
    assert visit_baseline(consumer._conn, "2026-10-06") is None
    clock[0] = NOON + timedelta(seconds=121)
    _drain(consumer)
    b = visit_baseline(consumer._conn, "2026-10-06")
    assert (b["count"], b["haircuts"], b["waiting"], b["washing"]) == (1, 0, 1, 0)
    assert b["taken_at"] == (NOON + timedelta(seconds=121)).isoformat(timespec="seconds")


def test_a_restart_the_same_day_keeps_the_stored_baseline(tmp_path, monkeypatch):
    from storage import set_visit_baseline, visit_baseline
    clock = [NOON]
    consumer, r, now_ms = _baseline_consumer(tmp_path, monkeypatch, clock)
    set_visit_baseline(consumer._conn, "2026-10-06", NOON - timedelta(hours=3), haircuts=4, waiting=3, washing=0)
    _feed_waiter(r, now_ms - 140_000)
    clock[0] = NOON + timedelta(seconds=300)
    _drain(consumer)
    assert visit_baseline(consumer._conn, "2026-10-06")["count"] == 7


def test_a_clips_loop_clear_removes_the_baseline_and_a_new_one_is_taken_later(tmp_path, monkeypatch):
    from storage import visit_baseline
    clock = [NOON]
    cams = (CameraConfig("barber-c", 100.0, True), CameraConfig("barber-l", 100.0, True))
    consumer, r, now_ms = _baseline_consumer(tmp_path, monkeypatch, clock, cams)
    _feed_waiter(r, now_ms - 140_000)
    clock[0] = NOON + timedelta(seconds=130)
    _drain(consumer)
    assert visit_baseline(consumer._conn, "2026-10-06") is not None
    restarted = datetime.now() - timedelta(seconds=10)             # the streamer restarted the clips
    (tmp_path / "starts.json").write_text(json.dumps({c.id: restarted.timestamp() for c in cams}))
    assert consumer._handle_loop_wraps({c.id: restarted for c in cams})
    assert visit_baseline(consumer._conn, "2026-10-06") is None
    _feed_waiter(r, now_ms + 5_000, 130)
    clock[0] = NOON + timedelta(seconds=200)                 # 70 s after the clear: too early
    _drain(consumer)
    assert visit_baseline(consumer._conn, "2026-10-06") is None
    clock[0] = NOON + timedelta(seconds=260)                 # 130 s after it
    _drain(consumer)
    assert visit_baseline(consumer._conn, "2026-10-06")["count"] == 1


def test_the_baseline_waits_for_two_minutes_of_frame_time_too(tmp_path, monkeypatch):
    from storage import visit_baseline
    clock = [NOON]
    consumer, r, now_ms = _baseline_consumer(tmp_path, monkeypatch, clock)
    _feed_waiter(r, now_ms - 40_000, 40)                     # the stream only just arrived: 40 s of frames
    clock[0] = NOON + timedelta(seconds=300)                 # plenty of wall time
    _drain(consumer)
    assert visit_baseline(consumer._conn, "2026-10-06") is None
    _feed_waiter(r, now_ms + 1_000, 90)                      # 130 s of frame time now
    _drain(consumer)
    assert visit_baseline(consumer._conn, "2026-10-06") is not None


def test_a_new_shop_day_takes_its_own_baseline(tmp_path, monkeypatch):
    from storage import visit_baseline
    clock = [NOON]
    consumer, r, now_ms = _baseline_consumer(tmp_path, monkeypatch, clock)
    _feed_waiter(r, now_ms - 140_000)
    clock[0] = NOON + timedelta(seconds=130)
    _drain(consumer)
    assert visit_baseline(consumer._conn, "2026-10-06") is not None
    assert visit_baseline(consumer._conn, "2026-10-07") is None
    clock[0] = datetime(2026, 10, 7, 5, 0, 10)               # day rolls over at 05:00
    _drain(consumer)
    assert visit_baseline(consumer._conn, "2026-10-07") is None
    _feed_waiter(r, now_ms + 1_000, 130)
    clock[0] = datetime(2026, 10, 7, 5, 2, 20)
    _drain(consumer)
    assert visit_baseline(consumer._conn, "2026-10-07") is not None


def test_a_failing_baseline_is_logged_once_a_minute(tmp_path, monkeypatch, caplog):
    clock = [NOON]
    consumer, r, now_ms = _baseline_consumer(tmp_path, monkeypatch, clock)
    monkeypatch.setattr("consumer.visit_baseline", lambda *a: 1 / 0)
    with caplog.at_level("ERROR", logger="barber.consumer"):
        for _ in range(5):
            consumer.step()
    assert sum("visit baseline failed" in m for m in caplog.messages) == 1


# ── 2026-10-11: a restart on a new run of the test clips starts clean ─────────
CLIP = 600.0


def _clip_consumer(tmp_path, start_ts, cameras=None):
    """A consumer attached to tmp_path's DB, the clips (re)started at start_ts."""
    import dataclasses
    from storage import connect
    cams = cameras or (CameraConfig("barber-c", CLIP, True), CameraConfig("barber-l", CLIP, True))
    c = dataclasses.replace(cfg(tmp_path), cameras=cams)
    (tmp_path / "starts.json").write_text(json.dumps({k.id: start_ts for k in cams}))
    consumer = ConsumerThread(c)
    consumer.attach(connect(str(c.db_path)), FakeRedis())
    return consumer


def _visit_ids(conn):
    return [r[0] for r in conn.execute("SELECT id FROM visit ORDER BY id")]


def _a_finished_visit(consumer, vid="v1"):
    now = datetime.now()
    apply_update(consumer._conn, EngineUpdate(visits={vid: Visit(vid, vid, now, True, open=False, ticket=7)}), now)


def test_a_restart_in_the_same_clip_run_keeps_the_rows(tmp_path):
    start = datetime.now().timestamp() - 100
    _a_finished_visit(_clip_consumer(tmp_path, start))
    again = _clip_consumer(tmp_path, start)                  # only the service restarted
    assert _visit_ids(again._conn) == ["v1"]


def test_a_restart_on_a_new_clip_run_clears_the_rows_and_the_tickets(tmp_path):
    first = _clip_consumer(tmp_path, datetime.now().timestamp() - 100)
    _a_finished_visit(first)
    first.tickets.next(datetime.now())
    again = _clip_consumer(tmp_path, datetime.now().timestamp() - 5)     # the streamer restarted the clips
    assert _visit_ids(again._conn) == []
    assert again.tickets.next(datetime.now()) == 1


def test_a_restart_after_the_clips_looped_while_down_clears_the_rows(tmp_path):
    from storage import get_state, set_state
    start = datetime.now().timestamp() - 1.5 * CLIP          # lap 1 now
    first = _clip_consumer(tmp_path, start)
    _a_finished_visit(first)
    run = json.loads(get_state(first._conn, "clip_run"))
    set_state(first._conn, "clip_run", json.dumps({k: [v[0], v[1], 0] for k, v in run.items()}))   # stored in lap 0
    assert _visit_ids(_clip_consumer(tmp_path, start)._conn) == []


def test_a_restart_on_new_footage_clears_the_rows(tmp_path):
    start = datetime.now().timestamp() - 100
    _a_finished_visit(_clip_consumer(tmp_path, start))
    longer = (CameraConfig("barber-c", 3 * CLIP, True), CameraConfig("barber-l", 3 * CLIP, True))
    assert _visit_ids(_clip_consumer(tmp_path, start, longer)._conn) == []


def test_live_cameras_keep_the_days_rows_across_a_restart(tmp_path):
    live = (CameraConfig("barber-c"), CameraConfig("barber-l"))
    _a_finished_visit(_clip_consumer(tmp_path, datetime.now().timestamp() - 100, live))
    assert _visit_ids(_clip_consumer(tmp_path, datetime.now().timestamp(), live)._conn) == ["v1"]


def test_a_streamer_restart_while_running_is_a_wrap_even_without_a_position_drop(tmp_path):
    consumer = _clip_consumer(tmp_path, datetime.now().timestamp() - 100)
    _a_finished_visit(consumer)
    starts = {c: datetime.fromtimestamp(v[0]) for c, v in consumer._run.items()}      # the same run
    assert not consumer._handle_loop_wraps(starts)
    moved = {c.id: datetime.now() - timedelta(seconds=300) for c in consumer.cfg.cameras}   # later position, new start
    assert consumer._handle_loop_wraps(moved)
    assert _visit_ids(consumer._conn) == []


def test_the_lap_end_hook_runs_when_the_clips_play_out_with_the_laps_rows_but_not_on_a_streamer_restart(tmp_path):
    """2026-10-18 (user): a run on recordings writes its report at the end of the lap -- before the rows go."""
    consumer = _clip_consumer(tmp_path, datetime.now().timestamp() - 300)
    _a_finished_visit(consumer)
    seen = []
    consumer.on_lap_end = lambda: seen.append(_visit_ids(consumer._conn))
    starts = {c: datetime.fromtimestamp(v[0]) for c, v in consumer._run.items()}
    moved = {c.id: datetime.now() - timedelta(seconds=310) for c in consumer.cfg.cameras}
    assert consumer._handle_loop_wraps(moved) and seen == []               # the streamer restarted: no report
    _a_finished_visit(consumer)
    consumer._last_position = {c: CLIP - 0.5 for c in moved}               # it had played to the end...
    assert consumer._handle_loop_wraps(moved)                              # ...and looped
    assert seen == [["v1"]] and _visit_ids(consumer._conn) == []           # the hook saw the lap; then cleared


def test_a_hard_restart_clears_the_rows_even_for_live_cameras(tmp_path, monkeypatch):
    live = (CameraConfig("barber-c"), CameraConfig("barber-l"))
    first = _clip_consumer(tmp_path, datetime.now().timestamp() - 100, live)
    _a_finished_visit(first)
    first.tickets.next(datetime.now())
    monkeypatch.setenv("BARBER_CLEAR_ON_START", "1")         # just barber-restart hard
    again = _clip_consumer(tmp_path, datetime.now().timestamp(), live)
    assert _visit_ids(again._conn) == [] and again.tickets.next(datetime.now()) == 1
    monkeypatch.setenv("BARBER_CLEAR_ON_START", "0")         # a soft restart keeps them
    _a_finished_visit(again, "v2")
    assert _visit_ids(_clip_consumer(tmp_path, datetime.now().timestamp(), live)._conn) == ["v2"]


def _feed_empty(r, start_ms, n):
    for i in range(n):
        for cam in ("barber-c", "barber-l"):
            r.add(f"iris:meta:{cam}", start_ms + i * 1000, {b"dets": b"[]"})


def test_the_baseline_counts_who_was_seen_in_the_window_not_one_instant(tmp_path, monkeypatch):
    """2026-10-11: a snapshot at 120 s read 1-7 on the same replay; the baseline is now the window's most
    haircuts + most open waits + median wash-area count. Someone waiting early who left before its end counts."""
    from consumer import BASELINE_SETTLE_S
    from storage import visit_baseline
    assert BASELINE_SETTLE_S == 180.0
    clock = [NOON]
    consumer, r, now_ms = _baseline_consumer(tmp_path, monkeypatch, clock, settle_s=BASELINE_SETTLE_S)
    _feed_waiter(r, now_ms - 200_000, 60)                    # on the sofa for the first minute, then gone
    _feed_empty(r, now_ms - 140_000, 140)
    clock[0] = NOON + timedelta(seconds=100)
    _drain(consumer, 8)
    clock[0] = NOON + timedelta(seconds=181)
    _drain(consumer, 8)
    b = visit_baseline(consumer._conn, "2026-10-06")
    assert b is not None and (b["count"], b["waiting"]) == (1, 1)


def test_the_engines_photo_requests_and_drops_go_to_the_worker(tmp_path):
    """2026-10-22 (spec haircut photos): fetched on the worker's thread, at the site's size."""
    from datetime import datetime as _dt
    from domain import EngineUpdate as _U, PhotoRequest
    from policy import DEFAULT_POLICY

    class Worker:
        def __init__(self):
            self.calls = []

        def request(self, r, px):
            self.calls.append(("photo", r.haircut_id, px))

        def drop(self, hid):
            self.calls.append(("drop", hid))

    consumer = ConsumerThread(cfg(tmp_path))
    consumer.photos = Worker()
    upd = _U(photos=[PhotoRequest("h1", "R1", "ph-2", "start", _dt(2026, 10, 22, 12), (0, 0, 1, 1))],
             dropped_photos={"h0"})
    consumer._photos_out(upd, {**DEFAULT_POLICY, "hp_photo_px": 800.0})
    assert consumer.photos.calls == [("photo", "h1", 800), ("drop", "h0")]
