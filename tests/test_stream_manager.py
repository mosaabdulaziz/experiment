import asyncio
import time
from app.redis_streams import read_stream
from app.stream_manager import pick_source
from app.usecases.vlm_analysis import enqueue_drop_oldest, should_emit
from app.usecases.zone_alerts import zone_alert_message


def test_pick_source_least_recently_serviced():
    assert pick_source(["cam0", "cam1"], {"cam0": 5.0, "cam1": 2.0}) == "cam1"
    assert pick_source(["cam0", "cam1"], {}) == "cam0"        # unseen → 0.0, first wins ties
    assert pick_source([], {"cam0": 1.0}) is None


def test_enqueue_drop_oldest():
    async def run():
        q = asyncio.Queue(maxsize=2)
        enqueue_drop_oldest(q, "a")
        enqueue_drop_oldest(q, "b")
        dropped = enqueue_drop_oldest(q, "c")            # full → drop "a"
        items = [q.get_nowait(), q.get_nowait()]
        return dropped, items, q.empty()
    dropped, items, empty = asyncio.run(run())
    assert dropped is True
    assert items == ["b", "c"]
    assert empty


def test_should_emit_per_type_cooldown():
    times = {"comment": 0.0, "alert": 0.0}
    assert should_emit("comment", times, cooldown=8, alert_cooldown=4, now=100) is True
    times["comment"] = 100
    assert should_emit("comment", times, cooldown=8, alert_cooldown=4, now=105) is False  # within 8s
    assert should_emit("comment", times, cooldown=8, alert_cooldown=4, now=109) is True   # past 8s
    assert should_emit("alert", times, cooldown=8, alert_cooldown=4, now=101) is True


_RESTRICTED_NAMED = {"name": "Front Door",
                     "intrusion": {"mode": "restricted", "classes": ["person"]}}
_RESTRICTED_UNNAMED = {"name": "",
                       "intrusion": {"mode": "restricted", "classes": ["person"]}}
_VALID_NAMED = {"name": "Ward A", "intrusion": {"mode": "valid", "classes": ["person"]}}
_VALID_UNNAMED = {"name": "", "intrusion": {"mode": "valid", "classes": ["person"]}}


def test_zone_alert_message_names_violator_on_intrusion():
    msg, tags, kind = zone_alert_message("cam0", _RESTRICTED_NAMED, "person", "17")
    assert msg == "Person #17 entered zone Front Door on cam0"
    assert tags == ["zone_intrusion"]
    assert kind == "intrusion"


def test_zone_alert_message_names_violator_on_exit():
    msg, tags, kind = zone_alert_message("cam0", _VALID_NAMED, "person", "17")
    assert msg == "Person #17 left zone Ward A on cam0"
    assert tags == ["zone_exit"]
    assert kind == "exit"


def test_zone_alert_message_unnamed_zones_use_generic_clause():
    assert (zone_alert_message("cam0", _RESTRICTED_UNNAMED, "person", "17")[0]
            == "Person #17 entered a restricted zone on cam0")
    assert (zone_alert_message("cam0", _VALID_UNNAMED, "person", "17")[0]
            == "Person #17 left the valid zone on cam0")


def test_zone_alert_message_reports_the_detected_class_not_person():
    # A zone watching vehicles must not report a car as a person.
    zone = {"name": "Loading Bay", "intrusion": {"mode": "restricted", "classes": ["car"]}}
    assert (zone_alert_message("cam0", zone, "car", "4")[0]
            == "Car #4 entered zone Loading Bay on cam0")


def test_zone_alert_message_unresolved_zone_takes_intrusion_branch():
    msg, tags, kind = zone_alert_message("cam0", None, "person", "17")
    assert msg == "Person #17 entered a restricted zone on cam0"
    assert tags == ["zone_intrusion"]
    assert kind == "intrusion"


def test_zone_alert_message_falls_back_when_label_missing():
    # An app newer than its sidecar sees no object_label field. Drop the id too rather
    # than emit a half-named "Object #17".
    assert (zone_alert_message("cam0", _RESTRICTED_NAMED, None, "17")[0]
            == "Zone intrusion detected on cam0 — Front Door")
    assert (zone_alert_message("cam0", _VALID_NAMED, "", "17")[0]
            == "Person left valid zone Ward A on cam0")


def test_zone_alert_message_falls_back_when_id_unusable():
    for bad in (None, "", "not-an-int"):
        assert (zone_alert_message("cam0", _RESTRICTED_NAMED, "person", bad)[0]
                == "Zone intrusion detected on cam0 — Front Door")


def test_zone_alert_message_falls_back_on_untracked_sentinel():
    # DeepStream's UNTRACKED_OBJECT_ID survives int() as 2**64-1; rendered verbatim it
    # would read "Person #18446744073709551615".
    assert (zone_alert_message("cam0", _RESTRICTED_NAMED, "person", str(0xFFFFFFFFFFFFFFFF))[0]
            == "Zone intrusion detected on cam0 — Front Door")


import json as _json
import fakeredis
from app.stream_manager import StreamManager


def _cfg():
    return {
        "sampling": {"frames_per_batch": 4, "interval_seconds": 4},
        "monitoring": {"cooldown_seconds": 8, "alert_cooldown_seconds": 4, "max_history": 3},
        "dedup": {"enabled": False, "backend": "local"},
        "perception": {"redis_url": "redis://unused", "queue_depth": 3,
                       "discovery_interval_seconds": 5},
        # Point at a nonexistent sidecar.yaml so analytics defaults to off deterministically,
        # rather than leaking the repo's real perception/pipeline/sidecar.yaml into these unit tests.
        "zones": {"sidecar_config_path": "/nonexistent/sidecar.yaml"},
    }


class _StubAnalyzer:
    def __init__(self):
        self.calls = []
        self.tracked = []
        self.scene = []
        self.scene_memories = []
        self.filters = []

    async def analyze(self, frames, history, cv_metadata=None, offset_start=None, offset_end=None,
                      source_id=None, tracked="", scene="", scene_memory=None,
                      live_track_ids=None, known_zones=None):
        self.calls.append((source_id, len(frames)))
        self.tracked.append(tracked)
        self.scene.append(scene)
        self.scene_memories.append(scene_memory)
        self.filters.append({"live_track_ids": live_track_ids, "known_zones": known_zones})
        return [{"type": "comment", "message": "seen", "tags": [],
                 "severity": "info", "category": "operations"}]


class _StubConn:
    def __init__(self):
        self.events = []

    async def broadcast(self, payload):
        self.events.append(payload)


def test_discover_reads_status_keys():
    r = fakeredis.FakeStrictRedis()
    r.set("iris:status:cam0", _json.dumps({"running": True}))
    r.set("iris:status:cam1", _json.dumps({"running": True}))
    sm = StreamManager(_cfg(), analyzer=_StubAnalyzer(), store=None,
                       conn_manager=_StubConn(), redis_client=r)
    assert sm.discover() == {"cam0", "cam1"}


def test_list_streams_includes_offset_and_status():
    r = fakeredis.FakeStrictRedis()
    r.set("iris:status:cam0", _json.dumps({"running": True, "is_file": True, "source": "clip.mp4"}))
    r.xadd("iris:meta:cam0", {"pts": "12.5", "frame_num": "3", "dets": "[]"})
    sm = StreamManager(_cfg(), analyzer=_StubAnalyzer(), store=None,
                       conn_manager=_StubConn(), redis_client=r)
    sm._ensure_stream_state("cam0")
    # No preview fields in status -> pipeline mode defaults (annotated JPEG polling).
    assert sm.list_streams() == [
        {"id": "cam0", "running": True, "is_file": True, "source": "clip.mp4",
         "offset": 12.5, "preview_uri": None, "preview_mode": "pipeline", "roles": [],
         "label": None, "zone": None, "camera": "cam0", "feature": None}
    ]


def test_list_streams_exposes_raw_preview_uri_and_mode():
    """pipeline_preview: false publishes preview_uri + preview_mode=raw in status,
    which list_streams must pass through so the frontend plays the raw stream
    instead of polling annotated JPEGs."""
    r = fakeredis.FakeStrictRedis()
    r.set("iris:status:cam0", _json.dumps({
        "running": True, "is_file": False, "source": "cam-1",
        "preview_uri": "rtsp://go2rtc:8554/cam-1", "preview_mode": "raw"}))
    r.xadd("iris:meta:cam0", {"pts": "12.5", "frame_num": "3", "dets": "[]"})
    sm = StreamManager(_cfg(), analyzer=_StubAnalyzer(), store=None,
                       conn_manager=_StubConn(), redis_client=r)
    sm._ensure_stream_state("cam0")
    assert sm.list_streams() == [
        {"id": "cam0", "running": True, "is_file": False, "source": "cam-1",
         "offset": 12.5, "preview_uri": "rtsp://go2rtc:8554/cam-1",
         "preview_mode": "raw", "roles": [], "label": None, "zone": None, "camera": "cam0", "feature": None}
    ]


def test_list_streams_exposes_roles_from_sidecar_yaml(tmp_path):
    sidecar_yaml = tmp_path / "sidecar.yaml"
    sidecar_yaml.write_text(
        "sources:\n"
        "  - {id: cam-7, uri: rtsp://x, roles: [fire]}\n"
        "  - {id: cam-1, uri: rtsp://x}\n"
    )
    cfg = _cfg()
    cfg["zones"] = {"sidecar_config_path": str(sidecar_yaml)}
    r = fakeredis.FakeStrictRedis()
    r.set("iris:status:cam-7", _json.dumps({"running": True, "is_file": False, "source": "cam-7"}))
    r.set("iris:status:cam-1", _json.dumps({"running": True, "is_file": False, "source": "cam-1"}))
    sm = StreamManager(cfg, analyzer=_StubAnalyzer(), store=None,
                       conn_manager=_StubConn(), redis_client=r)
    sm._ensure_stream_state("cam-7")
    sm._ensure_stream_state("cam-1")
    by_id = {s["id"]: s["roles"] for s in sm.list_streams()}
    assert by_id == {"cam-7": ["fire"], "cam-1": []}


def test_process_saves_event_tagged_with_camera(tmp_path):
    from app.events import EventStore
    r = fakeredis.FakeStrictRedis()
    analyzer, conn = _StubAnalyzer(), _StubConn()
    store = EventStore({"events": {"db_path": str(tmp_path / "e.db"), "retention_hours": 48}})
    sm = StreamManager(_cfg(), analyzer=analyzer, store=store, conn_manager=conn, redis_client=r)

    async def run():
        await store.init()
        sm._ensure_stream_state("cam1")                 # set up filter/cooldown/last_serviced
        await sm.analysis.process("cam1", [b"j0", b"j1"], [[("car", 0.9)], [("car", 0.9)]], [0.0, 0.5])
        return await store.get_recent(limit=10)

    rows = asyncio.run(run())
    assert analyzer.calls == [("cam1", 2)]               # analyzed 2 frames, tagged source cam1
    assert rows[-1]["camera"] == "cam1"                  # event tagged
    assert conn.events and conn.events[-1].get("camera") == "cam1"
    # The analyzer's own severity/category must reach the saved row via
    # save_kwargs_from_result -- not just save()'s type-derived back-compat bridge.
    assert rows[-1]["severity"] == "info"
    assert rows[-1]["category"] == "operations"


class _BatchSrc:
    is_running = True

    def grab_annotated_batch(self):
        return ([b"j"], [[("car", 0.9)]], [0.0])


def _producer_queue_empty(active_mode):
    cfg = _cfg()
    cfg["sampling"]["interval_seconds"] = 0.01
    cfg["monitoring"]["active_mode"] = active_mode
    cfg["monitoring"]["chat_in_flight"] = 0
    r = fakeredis.FakeStrictRedis()
    sm = StreamManager(cfg, analyzer=_StubAnalyzer(), store=None,
                       conn_manager=_StubConn(), redis_client=r)

    async def run():
        sm._running = True
        sm._ensure_stream_state("cam0")
        sm._sources["cam0"] = _BatchSrc()          # stub source: always has a batch
        task = asyncio.create_task(sm.analysis.run_for("cam0", sm._sources["cam0"]))
        await asyncio.sleep(0.06)
        sm._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return sm.analysis.queues["cam0"].empty()

    return asyncio.run(run())


def test_producer_skips_enqueue_in_chat_mode():
    assert _producer_queue_empty("chat") is True


def test_producer_enqueues_in_proactive_mode():
    assert _producer_queue_empty("proactive") is False


def test_sidecar_analytics_enabled_reads_flag(tmp_path, monkeypatch):
    # Was _sidecar_analytics_enabled, a wrapper removed in the MR !25 review;
    # the same behaviour, read through the one settings loader.
    from app.sidecar_config import module_enabled, running_sidecar_config
    monkeypatch.delenv("SIDECAR_CONFIG", raising=False)
    missing = {"zones": {"sidecar_config_path": str(tmp_path / "missing.yaml")}}
    assert module_enabled(running_sidecar_config(missing), "zones") is False

    present = tmp_path / "sidecar.yaml"
    present.write_text("modules:\n  zones: {}\n")
    cfg = {"zones": {"sidecar_config_path": str(present)}}
    assert module_enabled(running_sidecar_config(cfg), "zones") is True


def test_add_stream_zone_alert_sampler_gated_by_analytics_enabled():
    async def run():
        r = fakeredis.FakeStrictRedis()
        sm_off = StreamManager(_cfg(), analyzer=_StubAnalyzer(), store=None,
                               conn_manager=_StubConn(), redis_client=r)
        sm_off._running = True
        sm_off._add_stream("cam0")
        off_has_sampler = ("zone_alerts", "cam0") in sm_off._tasks
        await sm_off.stop()

        sm_on = StreamManager(_cfg(), analyzer=_StubAnalyzer(), store=None,
                              conn_manager=_StubConn(), redis_client=r)
        sm_on._analytics_enabled = True
        sm_on._running = True
        sm_on._add_stream("cam1")
        on_has_sampler = ("zone_alerts", "cam1") in sm_on._tasks
        await sm_on.stop()
        return off_has_sampler, on_has_sampler

    off_has_sampler, on_has_sampler = asyncio.run(run())
    assert off_has_sampler is False
    assert on_has_sampler is True


def test_stream_readers_never_poll_with_a_dollar_cursor():
    """"$" re-resolves to the stream's tail on EVERY xread, so an entry XADDed between
    two calls is skipped permanently. Seen in the field as smart-record files the
    sidecar announced (`smart-record complete`) that the clip service never received,
    leaving the alert to time out 10s later with footage sitting on disk."""
    r = fakeredis.FakeStrictRedis()
    cursors = []
    real_xread = r.xread

    def spy(streams, **kwargs):
        cursors.extend(streams.values())
        return real_xread(streams, **kwargs)

    r.xread = spy
    sm = StreamManager(_cfg(), analyzer=_StubAnalyzer(), store=None,
                       conn_manager=_StubConn(), redis_client=r)

    async def run():
        sm._running = True
        seen = []

        async def drain():
            async for f in read_stream(sm._redis, "iris:svr_done:cam0", lambda: sm._running):
                seen.append(f)

        task = asyncio.create_task(drain())
        await asyncio.sleep(0.1)
        r.xadd("iris:svr_done:cam0", {"path": "data/alert_svr/a.mp4"})
        await asyncio.sleep(1.3)          # covers the reader's 1s blocking window
        sm._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return seen

    seen = asyncio.run(run())
    assert "$" not in cursors, f"a re-resolving cursor loses entries: {cursors}"
    assert seen == [{"path": "data/alert_svr/a.mp4"}]


def test_a_stream_reader_does_not_replay_entries_that_predate_it():
    """Starting from the tail, not 0-0: a file announced before this process booted
    describes an alert that already resolved."""
    r = fakeredis.FakeStrictRedis()
    r.xadd("iris:svr_done:cam0", {"path": "old.mp4"})
    sm = StreamManager(_cfg(), analyzer=_StubAnalyzer(), store=None,
                       conn_manager=_StubConn(), redis_client=r)

    async def run():
        sm._running = True
        seen = []

        async def drain():
            async for f in read_stream(sm._redis, "iris:svr_done:cam0", lambda: sm._running):
                seen.append(f)

        task = asyncio.create_task(drain())
        await asyncio.sleep(1.3)
        sm._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return seen

    assert asyncio.run(run()) == []


def test_discovery_restarts_a_sampler_that_died():
    """_add_stream guards every task with `.done()` so a crashed one is replaced, but
    discovery only ever called it for newly-appeared streams -- so for a stream already
    known those guards could never fire and a dead sampler stayed dead."""
    async def run():
        r = fakeredis.FakeStrictRedis()
        r.set("iris:status:cam0", "running")
        cfg = _cfg()
        cfg["zones"] = {"sidecar_config_path": "perception/pipeline/sidecar.yaml"}
        sm = StreamManager(cfg, analyzer=_StubAnalyzer(), store=None,
                           conn_manager=_StubConn(), redis_client=r)
        sm._analytics_enabled = True
        sm._running = True
        sm._add_stream("cam0")
        dead = sm._tasks[("zone_alerts", "cam0")]
        dead.cancel()
        try:
            await dead
        except asyncio.CancelledError:
            pass
        assert dead.done()

        sm._add_stream("cam0")            # what the discovery loop now does every tick
        revived = sm._tasks[("zone_alerts", "cam0")]
        sm._running = False
        revived.cancel()
        try:
            await revived
        except asyncio.CancelledError:
            pass
        return dead, revived

    dead, revived = asyncio.run(run())
    assert revived is not dead, "a dead sampler must be replaced, not left dead"


def test_zone_alert_sampler_saves_event_and_broadcasts(tmp_path):
    from app.events import EventStore
    r = fakeredis.FakeStrictRedis()
    conn = _StubConn()
    store = EventStore({"events": {"db_path": str(tmp_path / "e.db"), "retention_hours": 48}})
    sm = StreamManager(_cfg(), analyzer=_StubAnalyzer(), store=store, conn_manager=conn, redis_client=r)

    async def run():
        await store.init()
        sm._running = True
        task = asyncio.create_task(sm.zone_alerts.run_for("cam0"))
        await asyncio.sleep(0.1)
        r.xadd("iris:zone_alert:cam0", {"frame_num": "5", "pts": "1.250"})
        await asyncio.sleep(1.3)   # covers the sampler's 1s blocking xread window
        sm._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return await store.get_recent(limit=10)

    rows = asyncio.run(run())
    assert len(rows) == 1
    assert rows[0]["event_type"] == "alert"
    assert rows[0]["tags"] == ["zone_intrusion"]
    assert rows[0]["camera"] == "cam0"
    assert conn.events and conn.events[-1]["camera"] == "cam0"
    # Back-compat guard: an alert carrying neither zone nor object (an older sidecar)
    # must still render the original wording, never a half-named subject.
    assert rows[0]["message"] == "Zone intrusion detected on cam0"
    # Deterministic zone-alert defaults (plan completion criterion): hardcoded
    # literals, not zone_alert_defaults() itself, so a swapped tuple in
    # app/severity.py or reversed kwargs in _emit_zone_event can't go unnoticed.
    assert rows[0]["severity"] == "warning"
    assert rows[0]["category"] == "security"


def test_zone_alert_sampler_names_the_zone_that_fired(tmp_path):
    from app.events import EventStore
    from app.zones import _save_zones
    r = fakeredis.FakeStrictRedis()
    conn = _StubConn()
    store = EventStore({"events": {"db_path": str(tmp_path / "e.db"), "retention_hours": 48}})
    zones_path = tmp_path / "zones_config.txt"
    _save_zones(str(zones_path), {"cam0": {"zones": [
        {"id": "z0", "name": "Front Door", "points": [[1, 1], [2, 2], [3, 3]],
         "intrusion": {"mode": "restricted", "classes": ["person"]}},
    ]}})
    cfg = _cfg()
    cfg["zones"] = {"file_path": str(zones_path)}
    sm = StreamManager(cfg, analyzer=_StubAnalyzer(), store=store, conn_manager=conn, redis_client=r)

    async def run():
        await store.init()
        sm._running = True
        task = asyncio.create_task(sm.zone_alerts.run_for("cam0"))
        await asyncio.sleep(0.1)
        r.xadd("iris:zone_alert:cam0", {"frame_num": "5", "pts": "1.250", "zone": "z0",
                                        "object_id": "17", "object_label": "person"})
        await asyncio.sleep(1.3)
        sm._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return await store.get_recent(limit=10)

    rows = asyncio.run(run())
    assert rows[0]["tags"] == ["zone_intrusion"]
    assert rows[0]["message"] == "Person #17 entered zone Front Door on cam0"


def test_zones_bump_increments_version_key():
    r = fakeredis.FakeStrictRedis()
    sm = StreamManager(_cfg(), analyzer=_StubAnalyzer(), store=None,
                       conn_manager=_StubConn(), redis_client=r)
    assert sm.zones_bump() == 1
    assert sm.zones_bump() == 2
    assert r.get("iris:ctrl:zones_version") == b"2"


def test_zones_bump_returns_none_on_redis_error():
    import redis as redis_mod

    class _RaisingRedis:
        def incr(self, *a, **kw):
            raise redis_mod.RedisError("down")

    sm = StreamManager(_cfg(), analyzer=_StubAnalyzer(), store=None,
                       conn_manager=_StubConn(), redis_client=_RaisingRedis())
    assert sm.zones_bump() is None


def test_zones_status_defaults_to_zeros():
    r = fakeredis.FakeStrictRedis()
    sm = StreamManager(_cfg(), analyzer=_StubAnalyzer(), store=None,
                       conn_manager=_StubConn(), redis_client=r)
    assert sm.zones_status() == {"enabled": False, "version": 0, "applied_version": 0, "applied_by_camera": {}}


def test_zones_status_reflects_redis_keys_and_enabled_flag(tmp_path):
    import yaml as yaml_mod
    sidecar_yaml = tmp_path / "sidecar.yaml"
    sidecar_yaml.write_text(yaml_mod.safe_dump({"modules": {"zones": {}}}))
    cfg = _cfg()
    cfg["zones"] = {"sidecar_config_path": str(sidecar_yaml)}
    r = fakeredis.FakeStrictRedis()
    r.set("iris:ctrl:zones_version", 3)
    r.set("iris:analytics:zones_applied", 2)
    sm = StreamManager(cfg, analyzer=_StubAnalyzer(), store=None,
                       conn_manager=_StubConn(), redis_client=r)
    assert sm.zones_status() == {"enabled": True, "version": 3, "applied_version": 2, "applied_by_camera": {}}


def test_zones_status_reports_each_camera_s_applied_version():
    """Each zones process acknowledges the cameras it runs; a role stream reports under its camera, and a camera no
    process acknowledged (an older sidecar) is left out so the editor falls back to applied_version."""
    r = fakeredis.FakeStrictRedis()
    r.set("iris:analytics:zones_applied", 5)
    r.set("iris:analytics:zones_applied:office", 4)
    r.set("iris:analytics:zones_applied:bay-a", 5)
    sm = StreamManager(_cfg(), analyzer=_StubAnalyzer(), store=None, conn_manager=_StubConn(), redis_client=r)
    sm._sources = {"office@zones": object(), "bay-a@zones": object(), "dock@zones": object()}
    assert sm.zones_status()["applied_by_camera"] == {"bay-a": 5, "office": 4}


def test_beside_role_sidecars_zones_count_as_applied_only_while_a_zones_sidecar_runs(tmp_path):
    """The safety stack: the zones module is in its config whether or not a camera runs it. With no zones sidecar
    streaming, nothing will acknowledge a save, so the zone editor must not wait for it (status "enabled": False)."""
    import yaml as yaml_mod
    sidecar_yaml = tmp_path / "sidecar.yaml"
    sidecar_yaml.write_text(yaml_mod.safe_dump({"modules": {"zones": {}}}))
    cfg = _cfg()
    cfg["zones"] = {"sidecar_config_path": str(sidecar_yaml)}
    sm = StreamManager(cfg, analyzer=_StubAnalyzer(), store=None, conn_manager=_StubConn(),
                       redis_client=fakeredis.FakeStrictRedis())
    sm._sources = {"bay-a@proximity": object(), "store@fire": object()}
    assert sm.zones_status()["enabled"] is False
    sm._sources["office@zones"] = object()
    assert sm.zones_status()["enabled"] is True
    sm._sources = {"cam-1": object()}                        # another product's stream: as before
    assert sm.zones_status()["enabled"] is True


def _cfg_with_zones(describe=True):
    cfg = _cfg()
    cfg["zones"] = {
        "describe_intrusions": describe,
        "describe_frame_count": 4,
        "describe_window_padding_seconds": 1.0,
    }
    return cfg


def test_zone_alert_sampler_sets_pending_status_when_describe_enabled(tmp_path):
    from app.events import EventStore
    r = fakeredis.FakeStrictRedis()
    conn = _StubConn()
    store = EventStore({"events": {"db_path": str(tmp_path / "e.db"), "retention_hours": 48}})
    sm = StreamManager(_cfg_with_zones(), analyzer=_StubAnalyzer(), store=store, conn_manager=conn,
                       redis_client=r)

    async def run():
        await store.init()
        sm._running = True
        task = asyncio.create_task(sm.zone_alerts.run_for("cam0"))
        await asyncio.sleep(0.1)
        r.xadd("iris:zone_alert:cam0", {"frame_num": "5", "pts": "12.500", "object_id": "7"})
        # The buffer lookup + fallback resolve are near-instant, but a fixed sleep failed once on a loaded CI runner
        # (rows was empty): wait for the stored event instead, up to 5 s.
        for _ in range(50):
            await asyncio.sleep(0.1)
            seen = await store.get_recent(limit=10)
            if seen and seen[0]["description_status"] == "ready":
                break
        sm._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return await store.get_recent(limit=10)

    rows = asyncio.run(run())
    assert rows[0]["description_status"] == "ready"           # no chat buffer for cam0 -> fallback resolved it
    assert rows[0]["description"] == "Description unavailable."
    assert conn.events[-1]["type"] == "event_update"


def test_zone_alert_sampler_no_description_status_when_describe_disabled(tmp_path):
    from app.events import EventStore
    r = fakeredis.FakeStrictRedis()
    conn = _StubConn()
    store = EventStore({"events": {"db_path": str(tmp_path / "e.db"), "retention_hours": 48}})
    sm = StreamManager(_cfg(), analyzer=_StubAnalyzer(), store=store, conn_manager=conn,
                       redis_client=r)

    async def run():
        await store.init()
        sm._running = True
        task = asyncio.create_task(sm.zone_alerts.run_for("cam0"))
        await asyncio.sleep(0.1)
        r.xadd("iris:zone_alert:cam0", {"frame_num": "5", "pts": "1.250", "object_id": "1"})
        await asyncio.sleep(1.3)
        sm._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return await store.get_recent(limit=10)

    rows = asyncio.run(run())
    assert rows[0]["description_status"] is None
    assert conn.events == [{"type": "event", "event": rows[0], "camera": "cam0"}]


def test_describe_intrusions_is_forced_off_when_there_is_no_analyzer(tmp_path):
    """vlm_enabled: false builds analyzer=None, but zones.describe_intrusions is read
    from a different config section and used to be trusted on its own. The alert then
    saved as 'pending' and its job went into _desc_jobs -- which only the VLM worker
    drains, and that worker is never started without an analyzer. Every alert sat on
    "preparing description" for the life of the process."""
    from app.events import EventStore
    r = fakeredis.FakeStrictRedis()
    conn = _StubConn()
    store = EventStore({"events": {"db_path": str(tmp_path / "e.db"), "retention_hours": 48}})
    cfg = _cfg_with_zones(describe=True)
    cfg["zones"]["sidecar_config_path"] = "/nonexistent/sidecar.yaml"
    sm = StreamManager(cfg, analyzer=None, store=store, conn_manager=conn, redis_client=r)

    async def run():
        await store.init()
        sm._running = True
        sm._ensure_stream_state("cam0")
        # Fill the chat buffer so the description window FINDS frames. With an empty
        # buffer the no-frames fallback resolves the event and hides the bug.
        for i in range(20):
            sm.chat.buffers["cam0"].append(10.0 + i * 0.5, b"\xff\xd8jpeg")
        task = asyncio.create_task(sm.zone_alerts.run_for("cam0"))
        await asyncio.sleep(0.1)
        r.xadd("iris:zone_alert:cam0", {"frame_num": "5", "pts": "12.500", "object_id": "7"})
        await asyncio.sleep(0.3)
        sm._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return await store.get_recent(limit=10)

    rows = asyncio.run(run())
    assert rows[0]["description_status"] is None      # never promises a description
    assert sm.descriptions.jobs == {}                        # nothing queued for a worker that won't run
    assert [m["type"] for m in conn.events] == ["event"]


def test_describe_overcrowding_is_forced_off_when_there_is_no_analyzer(tmp_path):
    from app.events import EventStore
    r = fakeredis.FakeStrictRedis()
    conn = _StubConn()
    store = EventStore({"events": {"db_path": str(tmp_path / "e.db"), "retention_hours": 48}})
    cfg = _cfg_with_zones(describe=False)
    cfg["zones"]["describe_overcrowding"] = True
    cfg["zones"]["sidecar_config_path"] = "/nonexistent/sidecar.yaml"
    sm = StreamManager(cfg, analyzer=None, store=store, conn_manager=conn, redis_client=r)

    async def run():
        await store.init()
        sm._running = True
        sm._ensure_stream_state("cam0")
        for i in range(20):
            sm.chat.buffers["cam0"].append(10.0 + i * 0.5, b"\xff\xd8jpeg")
        task = asyncio.create_task(sm.count_alerts.run_for("cam0"))
        await asyncio.sleep(0.1)
        r.xadd("iris:overcrowd_alert:cam0", {"frame_num": "5", "pts": "12.500", "count": "8"})
        await asyncio.sleep(0.3)
        sm._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return await store.get_recent(limit=10)

    rows = asyncio.run(run())
    assert rows[0]["description_status"] is None
    assert sm.descriptions.jobs == {}


def test_describe_fire_is_forced_off_when_there_is_no_analyzer(tmp_path):
    """alerts.describe_fire is read from yet another config section (alerts:, not
    zones:) and would be just as trusted-on-its-own as describe_intrusions/
    describe_overcrowding without this guard -- same failure mode: every fire alert
    saved as 'pending' with nothing to ever resolve it."""
    from app.events import EventStore
    r = fakeredis.FakeStrictRedis()
    conn = _StubConn()
    store = EventStore({"events": {"db_path": str(tmp_path / "e.db"), "retention_hours": 48}})
    cfg = _cfg_with_zones(describe=False)
    cfg["zones"]["sidecar_config_path"] = "/nonexistent/sidecar.yaml"
    cfg["alerts"] = {"describe_fire": True}
    sm = StreamManager(cfg, analyzer=None, store=store, conn_manager=conn, redis_client=r)

    async def run():
        await store.init()
        sm._running = True
        sm._ensure_stream_state("cam0")
        for i in range(20):
            sm.chat.buffers["cam0"].append(10.0 + i * 0.5, b"\xff\xd8jpeg")
        task = asyncio.create_task(sm.count_alerts.run_for("cam0"))
        await asyncio.sleep(0.1)
        r.xadd("iris:overcrowd_alert:cam0", {"frame_num": "5", "pts": "12.500",
                                             "kind": "fire_smoke", "class": "fire"})
        await asyncio.sleep(0.3)
        sm._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return await store.get_recent(limit=10)

    rows = asyncio.run(run())
    assert rows[0]["description_status"] is None
    assert sm.descriptions.jobs == {}


def test_describe_flags_are_untouched_when_an_analyzer_is_present():
    cfg = _cfg_with_zones(describe=True)
    cfg["zones"]["describe_overcrowding"] = True
    cfg["zones"]["sidecar_config_path"] = "/nonexistent/sidecar.yaml"
    cfg["alerts"] = {"describe_fire": True}
    sm = StreamManager(cfg, analyzer=_StubAnalyzer(), store=None, conn_manager=_StubConn(),
                       redis_client=fakeredis.FakeStrictRedis())
    assert sm.descriptions.describe_intrusions is True
    assert sm.descriptions.describe_overcrowding is True
    assert sm.descriptions.describe_fire is True


def test_remove_stream_cancels_only_that_streams_description_tasks():
    r = fakeredis.FakeStrictRedis()
    sm = StreamManager(_cfg_with_zones(), analyzer=_StubAnalyzer(), store=None, conn_manager=_StubConn(),
                       redis_client=r)

    async def run():
        sm._ensure_stream_state("cam0")
        sm._ensure_stream_state("cam1")
        t0 = asyncio.create_task(asyncio.sleep(10))
        t1 = asyncio.create_task(asyncio.sleep(10))
        sm.descriptions.tasks[1] = ("cam0", t0)
        sm.descriptions.tasks[2] = ("cam1", t1)
        sm._remove_stream("cam0")
        await asyncio.sleep(0)  # let cancellation propagate
        return t0.cancelled(), t1.cancelled(), list(sm.descriptions.tasks)

    t0_cancelled, t1_cancelled, remaining = asyncio.run(run())
    assert t0_cancelled is True
    assert t1_cancelled is False
    assert remaining == [2]


class _StubAnalyzerWithDescribe(_StubAnalyzer):
    def __init__(self, text="A person entered the zone.", raise_exc=False,
                overcrowding_text="A crowd has gathered.", exit_text="A person left the zone."):
        super().__init__()
        self._text = text
        self._overcrowding_text = overcrowding_text
        self._exit_text = exit_text
        self._raise = raise_exc
        self.describe_calls = []
        self.describe_overcrowding_calls = []
        self.describe_exit_calls = []

    async def describe_alert(self, prompt_kind, frames, source_id, count=None):
        if prompt_kind == "overcrowding":
            self.describe_overcrowding_calls.append((source_id, len(frames), count))
            if self._raise:
                raise RuntimeError("VLM down")
            return self._overcrowding_text
        if prompt_kind == "exit":
            self.describe_exit_calls.append((source_id, len(frames)))
            if self._raise:
                raise RuntimeError("VLM down")
            return self._exit_text
        self.describe_calls.append((source_id, len(frames)))
        if self._raise:
            raise RuntimeError("VLM down")
        return self._text


def test_worker_dispatches_description_job_and_resolves_ready(tmp_path):
    from app.events import EventStore
    r = fakeredis.FakeStrictRedis()
    conn = _StubConn()
    analyzer = _StubAnalyzerWithDescribe("A person in a red jacket entered the yard.")
    store = EventStore({"events": {"db_path": str(tmp_path / "e.db"), "retention_hours": 48}})
    sm = StreamManager(_cfg_with_zones(), analyzer=analyzer, store=store, conn_manager=conn,
                       redis_client=r)

    async def run():
        await store.init()
        record = await store.save("alert", "Zone intrusion detected on cam0", ["zone_intrusion"],
                                  camera="cam0", description_status="pending")
        sm.descriptions.jobs[f"__desc__:{record['id']}"] = ("cam0", [b"f0", b"f1"], "intrusion", None)
        sm._running = True
        task = asyncio.create_task(sm._worker())
        await asyncio.sleep(0.2)
        sm._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return record["id"], await store.get_recent(limit=10)

    event_id, rows = asyncio.run(run())
    assert f"__desc__:{event_id}" not in sm.descriptions.jobs
    assert f"__desc__:{event_id}" not in sm._last_serviced
    assert analyzer.describe_calls == [("cam0", 2)]
    assert rows[0]["description_status"] == "ready"
    assert rows[0]["description"] == "A person in a red jacket entered the yard."
    assert conn.events[-1] == {"type": "event_update", "id": event_id,
                               "description_status": "ready",
                               "description": "A person in a red jacket entered the yard."}


def test_worker_dispatches_overcrowding_description_job_to_the_right_prompt(tmp_path):
    from app.events import EventStore
    r = fakeredis.FakeStrictRedis()
    conn = _StubConn()
    analyzer = _StubAnalyzerWithDescribe(overcrowding_text="Eight people are clustered near the gate.")
    store = EventStore({"events": {"db_path": str(tmp_path / "e.db"), "retention_hours": 48}})
    sm = StreamManager(_cfg_with_zones(), analyzer=analyzer, store=store, conn_manager=conn,
                       redis_client=r)

    async def run():
        await store.init()
        record = await store.save("alert", "Overcrowding detected on cam0", ["zone_overcrowding"],
                                  camera="cam0", description_status="pending")
        sm.descriptions.jobs[f"__desc__:{record['id']}"] = ("cam0", [b"f0", b"f1"], "overcrowding", 8)
        sm._running = True
        task = asyncio.create_task(sm._worker())
        await asyncio.sleep(0.2)
        sm._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return record["id"], await store.get_recent(limit=10)

    event_id, rows = asyncio.run(run())
    assert analyzer.describe_overcrowding_calls == [("cam0", 2, 8)]
    assert analyzer.describe_calls == []   # intrusion prompt was NOT called
    assert rows[0]["description"] == "Eight people are clustered near the gate."


def test_worker_dispatches_exit_description_job_to_the_right_prompt(tmp_path):
    from app.events import EventStore
    r = fakeredis.FakeStrictRedis()
    conn = _StubConn()
    analyzer = _StubAnalyzerWithDescribe(exit_text="A person in a blue coat left the yard.")
    store = EventStore({"events": {"db_path": str(tmp_path / "e.db"), "retention_hours": 48}})
    sm = StreamManager(_cfg_with_zones(), analyzer=analyzer, store=store, conn_manager=conn,
                       redis_client=r)

    async def run():
        await store.init()
        record = await store.save("alert", "Person left the valid zone on cam0", ["zone_exit"],
                                  camera="cam0", description_status="pending")
        sm.descriptions.jobs[f"__desc__:{record['id']}"] = ("cam0", [b"f0", b"f1"], "exit", None)
        sm._running = True
        task = asyncio.create_task(sm._worker())
        await asyncio.sleep(0.2)
        sm._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return record["id"], await store.get_recent(limit=10)

    event_id, rows = asyncio.run(run())
    assert analyzer.describe_exit_calls == [("cam0", 2)]
    assert analyzer.describe_calls == []              # intrusion prompt was NOT called
    assert analyzer.describe_overcrowding_calls == []  # overcrowding prompt was NOT called
    assert rows[0]["description"] == "A person in a blue coat left the yard."


def test_zone_alert_sampler_emits_exit_event_for_valid_zone(tmp_path):
    from app.events import EventStore
    from app.zones import _save_zones
    r = fakeredis.FakeStrictRedis()
    conn = _StubConn()
    store = EventStore({"events": {"db_path": str(tmp_path / "e.db"), "retention_hours": 48}})
    zones_path = tmp_path / "zones_config.txt"
    _save_zones(str(zones_path), {"cam0": {"points": [[1, 1], [2, 2], [3, 3]], "type": "valid"}})
    cfg = _cfg()
    cfg["zones"] = {"file_path": str(zones_path)}
    sm = StreamManager(cfg, analyzer=_StubAnalyzer(), store=store, conn_manager=conn, redis_client=r)

    async def run():
        await store.init()
        sm._running = True
        task = asyncio.create_task(sm.zone_alerts.run_for("cam0"))
        await asyncio.sleep(0.1)
        # The sidecar tags each alert with the zone that fired; the legacy entry above
        # normalizes to id "z0", whose mode ("valid") drives the exit copy.
        r.xadd("iris:zone_alert:cam0", {"frame_num": "5", "pts": "1.250", "zone": "z0",
                                        "object_id": "17", "object_label": "person"})
        await asyncio.sleep(1.3)   # covers the sampler's 1s blocking xread window
        sm._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return await store.get_recent(limit=10)

    rows = asyncio.run(run())
    assert len(rows) == 1
    assert rows[0]["tags"] == ["zone_exit"]
    # The legacy entry carries no name, so _normalize_zone assigns the default "Zone 1".
    assert rows[0]["message"] == "Person #17 left zone Zone 1 on cam0"
    assert rows[0]["camera"] == "cam0"


def test_process_description_falls_back_when_overcrowding_vlm_raises(tmp_path):
    from app.events import EventStore
    r = fakeredis.FakeStrictRedis()
    conn = _StubConn()
    analyzer = _StubAnalyzerWithDescribe(raise_exc=True)
    store = EventStore({"events": {"db_path": str(tmp_path / "e.db"), "retention_hours": 48}})
    sm = StreamManager(_cfg_with_zones(), analyzer=analyzer, store=store, conn_manager=conn,
                       redis_client=r)

    async def run():
        await store.init()
        record = await store.save("alert", "Overcrowding detected on cam0", ["zone_overcrowding"],
                                  camera="cam0", description_status="pending")
        await sm.descriptions.process(record["id"], "cam0", [b"f0"], "overcrowding", 5)
        return await store.get_recent(limit=10)

    rows = asyncio.run(run())
    assert rows[0]["description_status"] == "ready"
    assert rows[0]["description"] == "Description unavailable."


def test_process_description_falls_back_when_vlm_raises(tmp_path):
    from app.events import EventStore
    r = fakeredis.FakeStrictRedis()
    conn = _StubConn()
    analyzer = _StubAnalyzerWithDescribe(raise_exc=True)
    store = EventStore({"events": {"db_path": str(tmp_path / "e.db"), "retention_hours": 48}})
    sm = StreamManager(_cfg_with_zones(), analyzer=analyzer, store=store, conn_manager=conn,
                       redis_client=r)

    async def run():
        await store.init()
        record = await store.save("alert", "Zone intrusion detected on cam0", ["zone_intrusion"],
                                  camera="cam0", description_status="pending")
        await sm.descriptions.process(record["id"], "cam0", [b"f0"])
        return await store.get_recent(limit=10)

    rows = asyncio.run(run())
    assert rows[0]["description_status"] == "ready"
    assert rows[0]["description"] == "Description unavailable."
    assert conn.events[-1]["description"] == "Description unavailable."


def test_worker_still_processes_proactive_batches_alongside_description_jobs(tmp_path):
    from app.events import EventStore
    r = fakeredis.FakeStrictRedis()
    conn = _StubConn()
    analyzer = _StubAnalyzerWithDescribe()
    store = EventStore({"events": {"db_path": str(tmp_path / "e.db"), "retention_hours": 48}})
    sm = StreamManager(_cfg_with_zones(), analyzer=analyzer, store=store, conn_manager=conn,
                       redis_client=r)

    async def run():
        await store.init()
        sm._ensure_stream_state("cam1")
        sm.analysis.queues["cam1"].put_nowait(("cam1", time.monotonic(), [b"j0"], [[("car", 0.9)]], [0.0]))
        record = await store.save("alert", "Zone intrusion detected on cam0", ["zone_intrusion"],
                                  camera="cam0", description_status="pending")
        sm.descriptions.jobs[f"__desc__:{record['id']}"] = ("cam0", [b"f0"], "intrusion", None)
        sm._running = True
        task = asyncio.create_task(sm._worker())
        await asyncio.sleep(0.2)
        sm._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return await store.get_recent(limit=10)

    rows = asyncio.run(run())
    assert {r["event_type"] for r in rows} == {"alert", "comment"}  # zone alert + proactive comment
    assert any(r.get("description_status") == "ready" for r in rows)


def test_worker_dispatches_description_job_even_when_analysis_paused(tmp_path):
    from app.events import EventStore
    r = fakeredis.FakeStrictRedis()
    conn = _StubConn()
    analyzer = _StubAnalyzerWithDescribe("A person entered the yard.")
    store = EventStore({"events": {"db_path": str(tmp_path / "e.db"), "retention_hours": 48}})
    cfg = _cfg_with_zones()
    cfg["monitoring"]["active_mode"] = "chat"  # proactive analysis paused
    cfg["monitoring"]["chat_in_flight"] = 0
    sm = StreamManager(cfg, analyzer=analyzer, store=store, conn_manager=conn,
                       redis_client=r)

    async def run():
        await store.init()
        record = await store.save("alert", "Zone intrusion detected on cam0", ["zone_intrusion"],
                                  camera="cam0", description_status="pending")
        sm.descriptions.jobs[f"__desc__:{record['id']}"] = ("cam0", [b"f0"], "intrusion", None)
        sm._running = True
        task = asyncio.create_task(sm._worker())
        await asyncio.sleep(0.2)
        sm._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return await store.get_recent(limit=10)

    rows = asyncio.run(run())
    assert rows[0]["description_status"] == "ready"
    assert rows[0]["description"] == "A person entered the yard."


def test_zone_alert_sampler_resolves_with_fallback_when_pts_unparseable(tmp_path):
    from app.events import EventStore
    r = fakeredis.FakeStrictRedis()
    conn = _StubConn()
    store = EventStore({"events": {"db_path": str(tmp_path / "e.db"), "retention_hours": 48}})
    sm = StreamManager(_cfg_with_zones(), analyzer=_StubAnalyzer(), store=store, conn_manager=conn,
                       redis_client=r)

    async def run():
        await store.init()
        sm._running = True
        task = asyncio.create_task(sm.zone_alerts.run_for("cam0"))
        await asyncio.sleep(0.1)
        r.xadd("iris:zone_alert:cam0", {"frame_num": "5", "pts": "not-a-number", "object_id": "1"})
        await asyncio.sleep(1.2)
        sm._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return await store.get_recent(limit=10)

    rows = asyncio.run(run())
    assert rows[0]["description_status"] == "ready"
    assert rows[0]["description"] == "Description unavailable."
    assert conn.events[-1]["type"] == "event_update"


def test_emit_zone_event_saves_and_broadcasts_with_given_message_and_tags(tmp_path):
    from app.events import EventStore
    r = fakeredis.FakeStrictRedis()
    conn = _StubConn()
    store = EventStore({"events": {"db_path": str(tmp_path / "e.db"), "retention_hours": 48}})
    sm = StreamManager(_cfg(), analyzer=_StubAnalyzer(), store=store, conn_manager=conn, redis_client=r)

    async def run():
        await store.init()
        await sm.alerts.emit("cam0", "Overcrowding detected on cam0 (7 people)",
                                  ["zone_overcrowding"], pts=1.0, describe_enabled=False,
                                  prompt_kind="overcrowding")
        return await store.get_recent(limit=10)

    rows = asyncio.run(run())
    assert rows[0]["message"] == "Overcrowding detected on cam0 (7 people)"
    assert rows[0]["tags"] == ["zone_overcrowding"]
    assert rows[0]["description_status"] is None
    assert conn.events == [{"type": "event", "event": rows[0], "camera": "cam0"}]
    # Deterministic zone-alert defaults (plan completion criterion): hardcoded
    # literals, not zone_alert_defaults() itself, so a swapped tuple in
    # app/severity.py or reversed kwargs in _emit_zone_event can't go unnoticed.
    assert rows[0]["severity"] == "warning"
    assert rows[0]["category"] == "safety"


def test_emit_zone_event_spawns_description_when_enabled(tmp_path):
    from app.events import EventStore
    r = fakeredis.FakeStrictRedis()
    conn = _StubConn()
    store = EventStore({"events": {"db_path": str(tmp_path / "e.db"), "retention_hours": 48}})
    sm = StreamManager(_cfg_with_zones(), analyzer=_StubAnalyzer(), store=store, conn_manager=conn,
                       redis_client=r)

    async def run():
        await store.init()
        await sm.alerts.emit("cam0", "Overcrowding detected on cam0", ["zone_overcrowding"],
                                  pts=1.0, describe_enabled=True, prompt_kind="overcrowding", count=9)
        # Check immediately, before yielding control -- asyncio.create_task only *schedules*
        # the spawned _prepare_description task; it hasn't run yet at this point, since nothing
        # after task creation inside _emit_zone_event awaits. Any await/sleep here would let the
        # (buffer-less, fast) task run to completion and remove itself from _description_tasks
        # in its own finally block, making this assertion racy.
        return list(sm.descriptions.tasks)

    tasks = asyncio.run(run())
    assert len(tasks) == 1


def test_emit_zone_event_resolves_fallback_immediately_when_pts_is_none(tmp_path):
    from app.events import EventStore
    r = fakeredis.FakeStrictRedis()
    conn = _StubConn()
    store = EventStore({"events": {"db_path": str(tmp_path / "e.db"), "retention_hours": 48}})
    sm = StreamManager(_cfg_with_zones(), analyzer=_StubAnalyzer(), store=store, conn_manager=conn,
                       redis_client=r)

    async def run():
        await store.init()
        await sm.alerts.emit("cam0", "Overcrowding detected on cam0", ["zone_overcrowding"],
                                  pts=None, describe_enabled=True, prompt_kind="overcrowding")
        return await store.get_recent(limit=10)

    rows = asyncio.run(run())
    assert rows[0]["description_status"] == "ready"
    assert rows[0]["description"] == "Description unavailable."
    assert conn.events[-1]["type"] == "event_update"


def test_prepare_description_annotates_frames_when_zone_id_resolves(tmp_path, monkeypatch):
    from app.events import EventStore
    import app.usecases.descriptions as desc_mod

    monkeypatch.setattr(desc_mod, "zone_by_id",
                        lambda config, sid, zone_id: {"points": [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9)],
                                                      "intrusion": {"mode": "restricted", "classes": ["person"]}})
    monkeypatch.setattr(desc_mod, "annotate_zone",
                        lambda frame, points, zone_type: frame + b"-annotated")

    r = fakeredis.FakeStrictRedis()
    store = EventStore({"events": {"db_path": str(tmp_path / "e.db"), "retention_hours": 48}})
    sm = StreamManager(_cfg_with_zones(), analyzer=_StubAnalyzer(), store=store, conn_manager=_StubConn(),
                       redis_client=r)
    sm._ensure_stream_state("cam0")
    buf = sm.chat.buffers["cam0"]
    buf.append(11.0, b"raw0")
    buf.append(12.0, b"raw1")
    buf.append(50.0, b"raw-far-away")  # outside the +-1.0s window (_cfg_with_zones padding), must be excluded

    async def run():
        await store.init()
        record = await store.save("alert", "Zone intrusion detected on cam0", ["zone_intrusion"],
                                  camera="cam0", description_status="pending")
        await sm.descriptions.prepare("cam0", record["id"], pts=12.0, zone_id="z0")
        return record["id"]

    event_id = asyncio.run(run())
    pid = f"__desc__:{event_id}"
    assert sm.descriptions.jobs[pid] == ("cam0", [b"raw0-annotated", b"raw1-annotated"], "intrusion", None)


def test_prepare_description_sends_raw_frames_when_no_zone_id(tmp_path):
    from app.events import EventStore

    r = fakeredis.FakeStrictRedis()
    store = EventStore({"events": {"db_path": str(tmp_path / "e.db"), "retention_hours": 48}})
    sm = StreamManager(_cfg_with_zones(), analyzer=_StubAnalyzer(), store=store, conn_manager=_StubConn(),
                       redis_client=r)
    sm._ensure_stream_state("cam0")
    buf = sm.chat.buffers["cam0"]
    buf.append(11.0, b"raw0")
    buf.append(12.0, b"raw1")

    async def run():
        await store.init()
        record = await store.save("alert", "Zone intrusion detected on cam0", ["zone_intrusion"],
                                  camera="cam0", description_status="pending")
        await sm.descriptions.prepare("cam0", record["id"], pts=12.0)  # zone_id defaults to None
        return record["id"]

    event_id = asyncio.run(run())
    pid = f"__desc__:{event_id}"
    assert sm.descriptions.jobs[pid] == ("cam0", [b"raw0", b"raw1"], "intrusion", None)


def test_prepare_description_zone_id_none_never_annotates_even_with_zones_configured(tmp_path, monkeypatch):
    # Regression guard for the dead-code-turned-live fallback: before this guard,
    # zone_id=None fell back to stream_zones(...)[0] and burned an arbitrary, unrelated
    # zone's polygon onto the frames sent to the VLM. That path was never reachable
    # before fire/smoke existed (every intrusion/overcrowding alert always resolves a
    # real zone_id), but fire alerts always call _prepare_description with zone_id=None
    # by design -- so with describe_fire: true on a stream that has zones configured,
    # every fire adjudication got a random ROI drawn on it, undermining the entire point
    # of the prompt (telling real fire apart from steam/glare/headlights). The stream
    # here genuinely has a saved zone (not mocked away) to prove there is no fallback
    # left to find one.
    from app.events import EventStore
    from app.zones import _save_zones
    import app.usecases.descriptions as desc_mod

    annotate_calls = []
    monkeypatch.setattr(desc_mod, "annotate_zone",
                        lambda frame, points, zone_type: annotate_calls.append(1) or frame + b"-annotated")

    zones_path = tmp_path / "zones_config.txt"
    _save_zones(str(zones_path), {"cam0": {"zones": [
        {"id": "z0", "name": "Front Door", "points": [[1, 1], [2, 2], [3, 3]],
         "intrusion": {"mode": "restricted", "classes": ["person"]}},
    ]}})

    r = fakeredis.FakeStrictRedis()
    store = EventStore({"events": {"db_path": str(tmp_path / "e.db"), "retention_hours": 48}})
    cfg = _cfg_with_zones()
    cfg["zones"]["file_path"] = str(zones_path)
    sm = StreamManager(cfg, analyzer=_StubAnalyzer(), store=store, conn_manager=_StubConn(),
                       redis_client=r)
    sm._ensure_stream_state("cam0")
    sm.chat.buffers["cam0"].append(11.0, b"raw0")

    async def run():
        await store.init()
        record = await store.save("alert", "Fire detected on cam0", ["fire_smoke"],
                                  camera="cam0", description_status="pending")
        await sm.descriptions.prepare("cam0", record["id"], pts=11.0, prompt_kind="fire_smoke",
                                      zone_id=None)
        return record["id"]

    event_id = asyncio.run(run())
    pid = f"__desc__:{event_id}"
    assert sm.descriptions.jobs[pid] == ("cam0", [b"raw0"], "fire_smoke", None)
    assert annotate_calls == []


def test_prepare_description_threads_prompt_kind_and_count_into_desc_jobs(tmp_path):
    from app.events import EventStore
    r = fakeredis.FakeStrictRedis()
    store = EventStore({"events": {"db_path": str(tmp_path / "e.db"), "retention_hours": 48}})
    sm = StreamManager(_cfg_with_zones(), analyzer=_StubAnalyzer(), store=store, conn_manager=_StubConn(),
                       redis_client=r)
    sm._ensure_stream_state("cam0")
    buf = sm.chat.buffers["cam0"]
    buf.append(11.0, b"raw0")

    async def run():
        await store.init()
        record = await store.save("alert", "Overcrowding detected on cam0", ["zone_overcrowding"],
                                  camera="cam0", description_status="pending")
        await sm.descriptions.prepare("cam0", record["id"], pts=11.0, prompt_kind="overcrowding", count=4)
        return record["id"]

    event_id = asyncio.run(run())
    pid = f"__desc__:{event_id}"
    assert sm.descriptions.jobs[pid] == ("cam0", [b"raw0"], "overcrowding", 4)


def test_prepare_description_defaults_to_intrusion_kind_when_unspecified(tmp_path):
    # Regression guard: existing call sites (_zone_alert_sampler's intrusion path via
    # _emit_zone_event) call _prepare_description without prompt_kind/count -- must still
    # default to the intrusion prompt with no count.
    from app.events import EventStore
    r = fakeredis.FakeStrictRedis()
    store = EventStore({"events": {"db_path": str(tmp_path / "e.db"), "retention_hours": 48}})
    sm = StreamManager(_cfg_with_zones(), analyzer=_StubAnalyzer(), store=store, conn_manager=_StubConn(),
                       redis_client=r)
    sm._ensure_stream_state("cam0")
    sm.chat.buffers["cam0"].append(11.0, b"raw0")

    async def run():
        await store.init()
        record = await store.save("alert", "Zone intrusion detected on cam0", ["zone_intrusion"],
                                  camera="cam0", description_status="pending")
        await sm.descriptions.prepare("cam0", record["id"], pts=11.0)
        return record["id"]

    event_id = asyncio.run(run())
    pid = f"__desc__:{event_id}"
    assert sm.descriptions.jobs[pid] == ("cam0", [b"raw0"], "intrusion", None)


def test_prepare_description_resolves_fallback_when_no_frames_in_window(tmp_path):
    from app.events import EventStore

    r = fakeredis.FakeStrictRedis()
    conn = _StubConn()
    store = EventStore({"events": {"db_path": str(tmp_path / "e.db"), "retention_hours": 48}})
    sm = StreamManager(_cfg_with_zones(), analyzer=_StubAnalyzer(), store=store, conn_manager=conn,
                       redis_client=r)
    sm._ensure_stream_state("cam0")
    buf = sm.chat.buffers["cam0"]
    buf.append(50.0, b"raw-far-away")  # nowhere near pts=12.0's +-1.0s window

    async def run():
        await store.init()
        record = await store.save("alert", "Zone intrusion detected on cam0", ["zone_intrusion"],
                                  camera="cam0", description_status="pending")
        await sm.descriptions.prepare("cam0", record["id"], pts=12.0)
        return record["id"], await store.get_recent(limit=10)

    event_id, rows = asyncio.run(run())
    pid = f"__desc__:{event_id}"
    assert pid not in sm.descriptions.jobs
    assert rows[0]["description_status"] == "ready"
    assert rows[0]["description"] == "Description unavailable."
    assert conn.events[-1]["type"] == "event_update"


def _sidecar_yaml_analytics(tmp_path, name: str, analytics_enabled: bool):
    import yaml as yaml_mod
    p = tmp_path / name
    p.write_text(yaml_mod.safe_dump({"modules": {"zones": {}} if analytics_enabled else {}}))
    return str(p)


def test_add_stream_overcrowd_alert_sampler_gated_by_analytics_enabled(tmp_path):
    # Overcrowding is now enabled per-zone (in the sidecar), so the app runs the sampler
    # whenever analytics is on — not gated by a global overcrowding_enabled flag.
    async def run():
        r = fakeredis.FakeStrictRedis()
        cfg_off = _cfg()
        cfg_off["zones"] = {"sidecar_config_path": _sidecar_yaml_analytics(tmp_path, "off.yaml", False)}
        sm_off = StreamManager(cfg_off, analyzer=_StubAnalyzer(), store=None,
                               conn_manager=_StubConn(), redis_client=r)
        sm_off._running = True
        sm_off._add_stream("cam0")
        off_has_sampler = ("count_alerts", "cam0") in sm_off._tasks
        await sm_off.stop()

        cfg_on = _cfg()
        cfg_on["zones"] = {"sidecar_config_path": _sidecar_yaml_analytics(tmp_path, "on.yaml", True)}
        sm_on = StreamManager(cfg_on, analyzer=_StubAnalyzer(), store=None,
                              conn_manager=_StubConn(), redis_client=r)
        sm_on._running = True
        sm_on._add_stream("cam1")
        on_has_sampler = ("count_alerts", "cam1") in sm_on._tasks
        await sm_on.stop()
        return off_has_sampler, on_has_sampler

    off_has_sampler, on_has_sampler = asyncio.run(run())
    assert off_has_sampler is False
    assert on_has_sampler is True


def test_overcrowd_alert_sampler_saves_event_with_count_in_message(tmp_path):
    from app.events import EventStore
    r = fakeredis.FakeStrictRedis()
    conn = _StubConn()
    store = EventStore({"events": {"db_path": str(tmp_path / "e.db"), "retention_hours": 48}})
    sm = StreamManager(_cfg(), analyzer=_StubAnalyzer(), store=store, conn_manager=conn, redis_client=r)

    async def run():
        await store.init()
        sm._running = True
        task = asyncio.create_task(sm.count_alerts.run_for("cam0"))
        await asyncio.sleep(0.1)
        r.xadd("iris:overcrowd_alert:cam0", {"frame_num": "5", "pts": "1.250", "count": "9"})
        await asyncio.sleep(1.3)
        sm._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return await store.get_recent(limit=10)

    rows = asyncio.run(run())
    assert len(rows) == 1
    assert rows[0]["event_type"] == "alert"
    assert rows[0]["tags"] == ["zone_overcrowding"]
    assert rows[0]["message"] == "Overcrowding detected on cam0 (9 people)"
    assert rows[0]["camera"] == "cam0"
    assert conn.events and conn.events[-1]["camera"] == "cam0"


def test_overcrowd_alert_sampler_falls_back_to_generic_message_when_count_unparseable(tmp_path):
    from app.events import EventStore
    r = fakeredis.FakeStrictRedis()
    conn = _StubConn()
    store = EventStore({"events": {"db_path": str(tmp_path / "e.db"), "retention_hours": 48}})
    sm = StreamManager(_cfg(), analyzer=_StubAnalyzer(), store=store, conn_manager=conn, redis_client=r)

    async def run():
        await store.init()
        sm._running = True
        task = asyncio.create_task(sm.count_alerts.run_for("cam0"))
        await asyncio.sleep(0.1)
        r.xadd("iris:overcrowd_alert:cam0", {"frame_num": "5", "pts": "1.250", "count": "not-a-number"})
        await asyncio.sleep(1.3)
        sm._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return await store.get_recent(limit=10)

    rows = asyncio.run(run())
    assert rows[0]["message"] == "Overcrowding detected on cam0"


def _cfg_with_overcrowd_description(tmp_path):
    import yaml as yaml_mod
    p = tmp_path / "sidecar.yaml"
    p.write_text(yaml_mod.safe_dump({"analytics_enabled": True, "overcrowding_enabled": True}))
    cfg = _cfg()
    cfg["zones"] = {
        "sidecar_config_path": str(p),
        "describe_overcrowding": True,
        "describe_frame_count": 4,
        "describe_window_padding_seconds": 1.0,
    }
    return cfg


def test_overcrowd_alert_sampler_sets_pending_status_when_describe_enabled(tmp_path):
    from app.events import EventStore
    r = fakeredis.FakeStrictRedis()
    conn = _StubConn()
    store = EventStore({"events": {"db_path": str(tmp_path / "e.db"), "retention_hours": 48}})
    sm = StreamManager(_cfg_with_overcrowd_description(tmp_path), analyzer=_StubAnalyzer(),
                       store=store, conn_manager=conn, redis_client=r)

    async def run():
        await store.init()
        sm._running = True
        task = asyncio.create_task(sm.count_alerts.run_for("cam0"))
        await asyncio.sleep(0.1)
        r.xadd("iris:overcrowd_alert:cam0", {"frame_num": "5", "pts": "12.500", "count": "7"})
        await asyncio.sleep(0.2)
        sm._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return await store.get_recent(limit=10)

    rows = asyncio.run(run())
    assert rows[0]["description_status"] == "ready"   # no chat buffer for cam0 -> fallback resolved it
    assert rows[0]["description"] == "Description unavailable."


class _StubClipService:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.submitted = []
        self.zones = []
        self.standalone_result = ("ready", "standalone-clip")

    async def submit(self, event_id, source_id, pts, zone=None, wall_time=None, pair=None, kind=None):
        self.submitted.append((event_id, source_id, pts))
        self.zones.append(zone)
        self.wall_times = getattr(self, 'wall_times', [])
        self.wall_times.append(wall_time)
        return True

    async def store_standalone(self, source_id, path, recorded_at=None):
        return self.standalone_result


def test_continue_fire_clip_appends_to_the_open_incident(tmp_path):
    from app.events import EventStore

    async def run():
        r = fakeredis.FakeStrictRedis()
        conn = _StubConn()
        store = EventStore({"events": {"db_path": str(tmp_path / "cf.db"), "retention_hours": 48}})
        await store.init()
        rec = await store.save("alert", "Fire detected on cam0", ["fire_smoke"], camera="cam0",
                               clip_status="pending", incident_status="ongoing")
        await store.update_clip(rec["id"], "ready", "clip-1")
        clips = _StubClipService()
        sm = StreamManager(_cfg(), analyzer=_StubAnalyzer(), store=store,
                           conn_manager=conn, redis_client=r, clip_service=clips)
        sm.fire.open_incidents[("cam0", "fire")] = rec["id"]

        await sm.fire.continue_clip("cam0", "/svr/cam0_2.mkv")

        saved = await store.get_recent(limit=10)
        return saved[-1], conn.events

    record, events = asyncio.run(run())
    assert record["clips"] == [
        {"clip_id": "clip-1", "status": "ready"},
        {"clip_id": "standalone-clip", "status": "ready"},
    ]
    assert record["clip_id"] == "clip-1"      # primary clip untouched
    assert {"type": "event_update", "id": record["id"], "clips": record["clips"]} in events


def test_continue_fire_clip_drops_a_file_with_no_open_incident(tmp_path):
    async def run():
        r = fakeredis.FakeStrictRedis()
        conn = _StubConn()
        clips = _StubClipService()
        sm = StreamManager(_cfg(), analyzer=_StubAnalyzer(), store=None,
                           conn_manager=conn, redis_client=r, clip_service=clips)

        await sm.fire.continue_clip("cam0", "/svr/orphan.mkv")
        return conn.events

    assert asyncio.run(run()) == []


def test_zone_alert_submits_a_clip_job_at_the_fired_pts(tmp_path):
    from app.events import EventStore

    async def run():
        r = fakeredis.FakeStrictRedis()
        conn = _StubConn()
        store = EventStore({"events": {"db_path": str(tmp_path / "z.db"), "retention_hours": 48}})
        await store.init()
        clips = _StubClipService()
        sm = StreamManager(_cfg(), analyzer=_StubAnalyzer(), store=store,
                           conn_manager=conn, redis_client=r, clip_service=clips)

        await sm.alerts.emit("cam0", "Zone intrusion detected on cam0",
                                  ["zone_intrusion"], 42.5, False, "intrusion")

        saved = await store.get_recent(limit=10)
        return clips.submitted, saved[-1]

    submitted, record = asyncio.run(run())
    assert submitted == [(record["id"], "cam0", 42.5)]
    assert record["clip_status"] == "pending"


def test_zone_alert_forwards_the_resolved_zone_to_the_clip_service(tmp_path):
    # The clip draws the zone that fired, so the geometry has to reach the clip job.
    # Resolved at fire time on purpose: a zone redrawn during the build must not
    # change what the clip outlines.
    from app.events import EventStore
    from app.zones import _save_zones

    async def run():
        r = fakeredis.FakeStrictRedis()
        store = EventStore({"events": {"db_path": str(tmp_path / "fz.db"), "retention_hours": 48}})
        await store.init()
        zones_path = tmp_path / "zones_config.txt"
        _save_zones(str(zones_path), {"cam0": {"zones": [
            {"id": "z0", "name": "Front Door", "points": [[1, 1], [2, 2], [3, 3]],
             "intrusion": {"mode": "restricted", "classes": ["person"]}},
        ]}})
        cfg = _cfg()
        cfg["zones"] = {"file_path": str(zones_path)}
        clips = _StubClipService()
        sm = StreamManager(cfg, analyzer=_StubAnalyzer(), store=store,
                           conn_manager=_StubConn(), redis_client=r, clip_service=clips)

        await sm.alerts.emit("cam0", "Zone intrusion detected on cam0",
                                  ["zone_intrusion"], 42.5, False, "intrusion",
                                  zone_id="z0")
        return clips.zones

    zones = asyncio.run(run())
    assert len(zones) == 1 and zones[0]["id"] == "z0"
    assert zones[0]["name"] == "Front Door"
    assert zones[0]["intrusion"]["mode"] == "restricted"


def test_proactive_alert_forwards_no_zone_to_the_clip_service(tmp_path):
    # A VLM alert is not zone-scoped; it must not inherit some zone's outline.
    from app.events import EventStore

    async def run():
        r = fakeredis.FakeStrictRedis()
        store = EventStore({"events": {"db_path": str(tmp_path / "pz.db"), "retention_hours": 48}})
        await store.init()
        clips = _StubClipService()
        sm = StreamManager(_cfg(), analyzer=_StubAnalyzer(), store=store,
                           conn_manager=_StubConn(), redis_client=r, clip_service=clips)

        await sm.alerts.emit("cam0", "Zone intrusion detected on cam0",
                                  ["zone_intrusion"], 42.5, False, "intrusion")
        return clips.zones

    assert asyncio.run(run()) == [None]


def test_zone_alert_marks_no_clip_status_when_the_service_is_off(tmp_path):
    from app.events import EventStore

    async def run():
        r = fakeredis.FakeStrictRedis()
        store = EventStore({"events": {"db_path": str(tmp_path / "o.db"), "retention_hours": 48}})
        await store.init()
        clips = _StubClipService(enabled=False)
        sm = StreamManager(_cfg(), analyzer=_StubAnalyzer(), store=store,
                           conn_manager=_StubConn(), redis_client=r, clip_service=clips)

        await sm.alerts.emit("cam0", "msg", ["zone_intrusion"], 42.5, False, "intrusion")

        saved = await store.get_recent(limit=10)
        return clips.submitted, saved[-1]

    submitted, record = asyncio.run(run())
    assert submitted == []
    assert record["clip_status"] is None


def test_proactive_alert_submits_a_clip_job_at_the_batch_midpoint(tmp_path):
    from app.events import EventStore

    class _AlertAnalyzer:
        async def analyze(self, frames, history, cv_metadata=None, offset_start=None,
                          offset_end=None, source_id=None, tracked="", **_kw):
            return [{"type": "alert", "message": "collision risk", "tags": []}]

    async def run():
        r = fakeredis.FakeStrictRedis()
        store = EventStore({"events": {"db_path": str(tmp_path / "p.db"), "retention_hours": 48}})
        await store.init()
        clips = _StubClipService()
        sm = StreamManager(_cfg(), analyzer=_AlertAnalyzer(), store=store,
                           conn_manager=_StubConn(), redis_client=r, clip_service=clips)
        sm._ensure_stream_state("cam0")

        await sm.analysis.process("cam0", [b"f"], [[]], [10.0])
        # Single-frame batch: midpoint of [10.0, 10.0] is 10.0.
        first = list(clips.submitted)

        sm.analysis.cooldown_times["cam0"]["alert"] = 0.0
        await sm.analysis.process("cam0", [b"f", b"g"], [[], []], [20.0, 30.0])
        return first, clips.submitted

    first, submitted = asyncio.run(run())
    assert first[0][1:] == ("cam0", 10.0)
    assert submitted[-1][1:] == ("cam0", 25.0)      # midpoint of [20.0, 30.0]


class _AlertAnalyzer:
    async def analyze(self, frames, history, cv_metadata=None, offset_start=None,
                      offset_end=None, source_id=None, tracked="", **_kw):
        return [{"type": "alert", "message": "collision risk", "tags": []}]


def test_a_proactive_alert_on_an_svr_stream_asks_the_sidecar_to_record(tmp_path):
    """The sidecar's probe decides zone alerts and starts their recording itself, but
    a VLM alert is decided here -- so nothing would ever open a smart-record session
    for it and its clip job resolved 'unavailable' at the deadline, every time."""
    from app.events import EventStore

    async def run():
        r = fakeredis.FakeStrictRedis()
        store = EventStore({"events": {"db_path": str(tmp_path / "sv.db"), "retention_hours": 48}})
        await store.init()
        sm = StreamManager(_cfg(), analyzer=_AlertAnalyzer(), store=store,
                           conn_manager=_StubConn(), redis_client=r,
                           clip_service=_StubClipService())
        sm._ensure_stream_state("cam0")
        sm.recordings.svr_sources.add("cam0")

        await sm.analysis.process("cam0", [b"f", b"g"], [[], []], [20.0, 30.0])
        return r.xrange("iris:svr_request:cam0")

    entries = asyncio.run(run())
    assert len(entries) == 1, "the alert must reach the sidecar's SVR trigger"
    # The batch midpoint, in the sidecar's own pts clock -- so it can age the request
    # by subtraction rather than comparing wall clocks across containers.
    assert float(entries[0][1][b"pts"]) == 25.0


def test_a_proactive_alert_on_an_encode_stream_asks_for_nothing(tmp_path):
    """file:// sources record continuously; a request would open a redundant session."""
    from app.events import EventStore

    async def run():
        r = fakeredis.FakeStrictRedis()
        store = EventStore({"events": {"db_path": str(tmp_path / "se.db"), "retention_hours": 48}})
        await store.init()
        sm = StreamManager(_cfg(), analyzer=_AlertAnalyzer(), store=store,
                           conn_manager=_StubConn(), redis_client=r,
                           clip_service=_StubClipService())
        sm._ensure_stream_state("cam0")

        await sm.analysis.process("cam0", [b"f"], [[]], [10.0])
        return r.keys("iris:svr_request:*")

    assert asyncio.run(run()) == []


def test_proactive_comment_does_not_submit_a_clip_job(tmp_path):
    from app.events import EventStore

    async def run():
        r = fakeredis.FakeStrictRedis()
        store = EventStore({"events": {"db_path": str(tmp_path / "c.db"), "retention_hours": 48}})
        await store.init()
        clips = _StubClipService()
        # _StubAnalyzer returns a comment, not an alert.
        sm = StreamManager(_cfg(), analyzer=_StubAnalyzer(), store=store,
                           conn_manager=_StubConn(), redis_client=r, clip_service=clips)
        sm._ensure_stream_state("cam0")
        await sm.analysis.process("cam0", [b"f"], [[]], [10.0])
        return clips.submitted

    assert asyncio.run(run()) == []

from app.cv_enricher import Detection
from app.stream_manager import build_frame_metas


def test_build_frame_metas_maps_wire_tuples_to_detections():
    metas = build_frame_metas([1.5], [[("person", 0.9, 17, (1.0, 2.0, 3.0, 4.0))]])
    assert len(metas) == 1
    assert metas[0].offset == 1.5
    assert metas[0].detections == [Detection("person", 0.9, 17, (1.0, 2.0, 3.0, 4.0))]


def test_build_frame_metas_handles_missing_track_id_and_bbox():
    metas = build_frame_metas([0.0], [[("car", 0.5, None, None)]])
    assert metas[0].detections == [Detection("car", 0.5, None, None)]


def test_build_frame_metas_pairs_each_offset_with_its_frame():
    metas = build_frame_metas([0.0, 1.0], [[], [("person", 0.7, 1, None)]])
    assert [m.offset for m in metas] == [0.0, 1.0]
    assert metas[0].detections == []
    assert len(metas[1].detections) == 1


# ── measured memory: the TrackObserver fed from the live detection stream ──────

def _memory_cfg(enabled=True):
    """_cfg() plus a memory block and a fast chat cadence (the sampler that feeds
    the observer ticks at 1/history.fps, so a slow fps makes the tests sleep)."""
    cfg = _cfg()
    cfg["monitoring"]["memory"] = {
        "enabled": enabled, "entity_window_seconds": 120, "entity_stale_seconds": 30,
        "max_entities_rendered": 20, "stationary_radius": 0.02,
    }
    cfg["chat"] = {"history": {"fps": 100, "window_seconds": 10}}
    return cfg


class _DetSrc:
    """Fake PerceptionSource for the sampler: scripted grab_detections replies."""
    is_running = True
    current_offset = 0.0

    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = 0

    def grab_detections(self):
        self.calls += 1
        return self._replies[min(self.calls - 1, len(self._replies) - 1)]

    def grab_frame(self):
        return b"JPG"


class _RecordingObserver:
    def __init__(self, raises=False):
        self.observed = []
        self._raises = raises

    def observe(self, now, dets):
        self.observed.append((now, dets))
        if self._raises:
            raise RuntimeError("observer exploded")


def _run_sampler(cfg, src, observer=None, seconds=0.15):
    """Drive one stream's _chat_sampler for a moment; returns (sm, task_alive)."""
    r = fakeredis.FakeStrictRedis()
    sm = StreamManager(cfg, analyzer=_StubAnalyzer(), store=None,
                       conn_manager=_StubConn(), redis_client=r)

    async def run():
        sm._running = True
        sm._ensure_stream_state("cam0")
        sm._sources["cam0"] = src
        if observer is not None:
            sm._track_observers["cam0"] = observer
        task = asyncio.create_task(sm.chat.run_for("cam0", sm._sources["cam0"]))
        await asyncio.sleep(seconds)
        alive = not task.done()
        sm._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return alive

    return sm, asyncio.run(run())


def test_ensure_stream_state_registers_an_observer_when_memory_enabled():
    r = fakeredis.FakeStrictRedis()
    sm = StreamManager(_memory_cfg(True), analyzer=_StubAnalyzer(), store=None,
                       conn_manager=_StubConn(), redis_client=r)
    sm._ensure_stream_state("cam0")
    assert "cam0" in sm._track_observers


def test_ensure_stream_state_registers_no_observer_when_memory_disabled():
    r = fakeredis.FakeStrictRedis()
    sm = StreamManager(_memory_cfg(False), analyzer=_StubAnalyzer(), store=None,
                       conn_manager=_StubConn(), redis_client=r)
    sm._ensure_stream_state("cam0")
    assert sm._track_observers == {}


def test_memory_block_absent_means_disabled():
    """No monitoring.memory key at all (an older config.yaml) is off, not a crash."""
    r = fakeredis.FakeStrictRedis()
    sm = StreamManager(_cfg(), analyzer=_StubAnalyzer(), store=None,
                       conn_manager=_StubConn(), redis_client=r)
    sm._ensure_stream_state("cam0")
    assert sm._track_observers == {}


def test_remove_stream_drops_the_observer():
    r = fakeredis.FakeStrictRedis()
    sm = StreamManager(_memory_cfg(True), analyzer=_StubAnalyzer(), store=None,
                       conn_manager=_StubConn(), redis_client=r)
    sm._ensure_stream_state("cam0")
    sm._remove_stream("cam0")
    assert "cam0" not in sm._track_observers


def test_chat_sampler_feeds_the_observer_real_detections():
    """End to end through a real TrackObserver: the track it saw is measurable."""
    src = _DetSrc([("1-0", [("person", 0.9, 17, (100.0, 100.0, 40.0, 80.0))])])
    sm, alive = _run_sampler(_memory_cfg(True), src)
    observer = sm._track_observers["cam0"]
    assert observer.present_for(17, time.monotonic()) is not None
    assert alive


def test_chat_sampler_skips_an_unchanged_entry_id():
    """The sampler polls faster than the sidecar publishes, so the newest entry is
    often one already consumed. Re-observing it would let a stalled sidecar accrue
    duration on tracks nobody has actually seen since."""
    src = _DetSrc([("1-0", [("person", 0.9, 17, (10.0, 10.0, 20.0, 20.0))])])  # same id forever
    obs = _RecordingObserver()
    _sm, alive = _run_sampler(_memory_cfg(True), src, observer=obs)
    assert src.calls > 1, "the sampler must have ticked more than once for this to prove anything"
    assert len(obs.observed) == 1
    assert alive


def test_chat_sampler_observes_each_new_entry_id():
    src = _DetSrc([("1-0", [("person", 0.9, 1, None)]),
                   ("2-0", [("person", 0.9, 2, None)]),
                   ("3-0", [("person", 0.9, 3, None)])])
    obs = _RecordingObserver()
    _run_sampler(_memory_cfg(True), src, observer=obs)
    assert [d[0][2] for _now, d in obs.observed] == [1, 2, 3]


def test_chat_sampler_survives_an_observer_that_raises():
    """Memory is an aid; the frame buffer is the sampler's actual product."""
    src = _DetSrc([("1-0", []), ("2-0", []), ("3-0", [])])
    sm, alive = _run_sampler(_memory_cfg(True), src, observer=_RecordingObserver(raises=True))
    assert alive, "an observer exception must not kill the chat sampler"
    assert len(sm.chat.buffers["cam0"]) > 0, "frames must still reach the chat buffer"


def test_chat_sampler_tolerates_a_source_without_grab_detections():
    """The local/fake-source path: no observer registered, no attribute reached for."""
    class _NoDetsSrc:
        is_running = True
        current_offset = 0.0

        def grab_frame(self):
            return b"JPG"

    sm, alive = _run_sampler(_memory_cfg(False), _NoDetsSrc())
    assert alive
    assert len(sm.chat.buffers["cam0"]) > 0


def _process_tracked(cfg, db_path, seed_observer):
    """Run one _process cycle and return what reached the analyzer's `tracked` arg."""
    from app.events import EventStore

    async def run():
        r = fakeredis.FakeStrictRedis()
        store = EventStore({"events": {"db_path": str(db_path), "retention_hours": 48}})
        await store.init()
        analyzer = _StubAnalyzer()
        sm = StreamManager(cfg, analyzer=analyzer, store=store,
                           conn_manager=_StubConn(), redis_client=r)
        sm._ensure_stream_state("cam0")
        observer = sm._track_observers.get("cam0")
        if observer is not None and seed_observer:
            observer.observe(time.monotonic(),
                             [("truck", 0.9, 23, (100.0, 100.0, 40.0, 40.0))])
        await sm.analysis.process("cam0", [b"f"], [[]], [0.0])
        return analyzer.tracked

    return asyncio.run(run())


def test_process_passes_the_rendered_track_block_to_the_analyzer(tmp_path):
    """The measurement is worthless unless the model actually sees it."""
    tracked = _process_tracked(_memory_cfg(True), tmp_path / "t.db", seed_observer=True)
    assert len(tracked) == 1
    assert "## TRACKED OBJECTS" in tracked[0]
    assert "#23 truck" in tracked[0]


def test_process_passes_an_empty_block_when_no_observer_is_registered(tmp_path):
    """memory disabled (and the local/fake-source path) must reach analyze with "",
    which is what keeps the rendered prompt byte-identical to the pre-memory one."""
    tracked = _process_tracked(_memory_cfg(False), tmp_path / "n.db", seed_observer=False)
    assert tracked == [""]


def test_process_passes_an_empty_block_when_the_observer_has_seen_nothing(tmp_path):
    """An observer with no tracks renders "" — memory on, but an empty scene must
    not start injecting an empty heading into the prompt."""
    tracked = _process_tracked(_memory_cfg(True), tmp_path / "e.db", seed_observer=False)
    assert tracked == [""]


# ── interpreted memory: one SceneMemory per stream, and the two merge filters ──

def _scene_cfg(enabled=True, scene_delta=True, zones_file=None):
    """_memory_cfg plus the interpreted half's own switch and bounds."""
    cfg = _memory_cfg(enabled)
    cfg["monitoring"]["memory"].update({
        "scene_delta": scene_delta, "max_situations": 8,
        "situation_ttl_cycles": 3, "scene_notes_max_chars": 400,
    })
    if zones_file is not None:
        cfg["zones"]["file_path"] = str(zones_file)
    return cfg


def _sm(cfg, analyzer=None):
    return StreamManager(cfg, analyzer=analyzer or _StubAnalyzer(), store=None,
                         conn_manager=_StubConn(), redis_client=fakeredis.FakeStrictRedis())


def test_ensure_stream_state_registers_a_scene_memory_when_both_switches_are_on():
    sm = _sm(_scene_cfg(enabled=True, scene_delta=True))
    sm._ensure_stream_state("cam0")
    assert "cam0" in sm._scene_memories


def test_ensure_stream_state_registers_no_scene_memory_when_the_master_is_off():
    """monitoring.memory.enabled is the master for both halves; scene_delta alone is
    inert, because the interpreted half costs output tokens on every single call."""
    sm = _sm(_scene_cfg(enabled=False, scene_delta=True))
    sm._ensure_stream_state("cam0")
    assert sm._scene_memories == {}


def test_ensure_stream_state_registers_no_scene_memory_when_scene_delta_is_off():
    """Measured memory on, interpreted memory off is the shipped configuration."""
    sm = _sm(_scene_cfg(enabled=True, scene_delta=False))
    sm._ensure_stream_state("cam0")
    assert "cam0" in sm._track_observers
    assert sm._scene_memories == {}


def test_remove_stream_drops_the_scene_memory():
    sm = _sm(_scene_cfg())
    sm._ensure_stream_state("cam0")
    sm._remove_stream("cam0")
    assert "cam0" not in sm._scene_memories


def test_each_stream_gets_its_own_scene_memory():
    """A delta describes the scene in front of ONE camera. Share one instance and
    cam0's situations render into cam1's prompt."""
    sm = _sm(_scene_cfg())
    sm._ensure_stream_state("cam0")
    sm._ensure_stream_state("cam1")
    assert sm._scene_memories["cam0"] is not sm._scene_memories["cam1"]
    sm._scene_memories["cam0"].merge(
        {"situations": [{"key": "bay2", "description": "truck at Bay 2", "state": "ongoing"}]},
        0.0)
    assert "truck at Bay 2" in sm._scene_memories["cam0"].render(1.0)
    assert sm._scene_memories["cam1"].render(1.0) == ""


def _process_once(cfg, db_path, seed=None, analyzer=None, cycles=1):
    """Run _process cycles on cam0; returns (sm, analyzer)."""
    from app.events import EventStore

    async def run():
        r = fakeredis.FakeStrictRedis()
        store = EventStore({"events": {"db_path": str(db_path), "retention_hours": 48}})
        await store.init()
        az = analyzer or _StubAnalyzer()
        sm = StreamManager(cfg, analyzer=az, store=store,
                           conn_manager=_StubConn(), redis_client=r)
        sm._ensure_stream_state("cam0")
        if seed is not None:
            seed(sm)
        for _ in range(cycles):
            sm.analysis.cooldown_times["cam0"] = {"comment": 0.0, "alert": 0.0}
            await sm.analysis.process("cam0", [b"f"], [[]], [0.0])
        return sm, az

    return asyncio.run(run())


def test_process_passes_the_rendered_scene_block_and_the_streams_own_memory(tmp_path):
    def seed(sm):
        sm._scene_memories["cam0"].merge(
            {"situations": [{"key": "bay2", "description": "truck at Bay 2 being unloaded",
                             "state": "ongoing"}]}, time.monotonic())

    sm, az = _process_once(_scene_cfg(), tmp_path / "s.db", seed=seed)
    assert "## SCENE MEMORY" in az.scene[0]
    assert "truck at Bay 2 being unloaded" in az.scene[0]
    assert az.scene_memories[0] is sm._scene_memories["cam0"]


def test_process_passes_an_empty_scene_block_when_no_memory_is_registered(tmp_path):
    """scene_delta off must reach analyze with "" and None -- which is what keeps the
    rendered prompt byte-identical to the pre-memory one."""
    _sm_, az = _process_once(_scene_cfg(scene_delta=False), tmp_path / "n.db")
    assert az.scene == [""]
    assert az.scene_memories == [None]


def test_process_supplies_the_live_track_ids_and_the_configured_zone_names(tmp_path):
    zones = tmp_path / "zones.json"
    zones.write_text(_json.dumps({"cam0": {"zones": [
        {"name": "Loading Bay", "points": [[0, 0], [100, 0], [100, 100]]},
        {"name": "", "points": [[0, 0], [10, 0], [10, 10]]},
    ]}}))

    def seed(sm):
        sm._track_observers["cam0"].observe(
            time.monotonic(), [("truck", 0.9, 23, (100.0, 100.0, 40.0, 40.0))])

    _sm_, az = _process_once(_scene_cfg(zones_file=zones), tmp_path / "f.db", seed=seed)
    # A zone saved without a name still HAS one: app/zones.py fills in "Zone <n>" for the
    # editor, and that is the name the operator sees, so it is a real zone name here too.
    assert az.filters[0] == {"live_track_ids": {23},
                             "known_zones": {"Loading Bay", "Zone 2"}}


def test_process_passes_no_track_filter_when_the_stream_has_no_observer(tmp_path):
    """None means "skip the filter"; an empty set would mean "drop every entity"."""
    _sm_, az = _process_once(_scene_cfg(enabled=False), tmp_path / "o.db")
    assert az.filters[0]["live_track_ids"] is None


# ── end to end through the real Analyzer: what the filters actually prevent ────

def _real_analyzer(tmp_path, content, record=None):
    """A real Analyzer whose single HTTP call returns `content`."""
    from types import SimpleNamespace
    from app.analyzer import Analyzer
    az = Analyzer({"api": {"backend": "cloud", "api_key": "x", "model": "m",
                           "prompt_log_path": str(tmp_path / "p.json"),
                           "response_log_path": str(tmp_path / "r.json"),
                           "debug_calls_path": str(tmp_path / "calls")},
                   "monitoring": {"memory": {"enabled": True, "scene_delta": True}},
                   "sampling": {}, "chat": {}})

    async def create(*a, **k):
        if record is not None:
            record.append(k)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    az._client.chat.completions.create = create
    return az


def test_an_entity_note_for_a_dead_track_never_reaches_the_memory(tmp_path):
    """The filter lives in SceneMemory.merge; this proves _process actually feeds it.
    Without the plumbing the note about #99 renders for the life of the process."""
    content = ('{"events":[],"memory":{"entities":['
               '{"id":23,"note":"carrying a long bag"},'
               '{"id":99,"note":"left the yard ten minutes ago"}]}}')

    def seed(sm):
        sm._track_observers["cam0"].observe(
            time.monotonic(), [("person", 0.9, 23, (100.0, 100.0, 40.0, 40.0))])

    sm, _az = _process_once(_scene_cfg(), tmp_path / "e2e1.db", seed=seed,
                            analyzer=_real_analyzer(tmp_path, content))
    block = sm._scene_memories["cam0"].render(time.monotonic())
    assert "carrying a long bag" in block
    assert "left the yard ten minutes ago" not in block


def test_an_area_for_an_unconfigured_zone_never_reaches_the_memory(tmp_path):
    zones = tmp_path / "zones.json"
    zones.write_text(_json.dumps({"cam0": {"zones": [
        {"name": "Loading Bay", "points": [[0, 0], [100, 0], [100, 100]]}]}}))
    content = ('{"events":[],"memory":{"areas":['
               '{"zone":"Loading Bay","condition":"unloading in progress"},'
               '{"zone":"Invented Wing","condition":"quiet"}]}}')

    sm, _az = _process_once(_scene_cfg(zones_file=zones), tmp_path / "e2e2.db",
                            analyzer=_real_analyzer(tmp_path, content))
    block = sm._scene_memories["cam0"].render(time.monotonic())
    assert "unloading in progress" in block
    assert "Invented Wing" not in block


def test_the_rendered_duration_uses_the_same_clock_the_merge_stamped(tmp_path):
    """analyze stamps the merge with time.monotonic(); rendering with time.time()
    would print the seconds since 1970 as a duration and raise nothing at all."""
    import re
    content = ('{"events":[],"memory":{"situations":[{"key":"bay2",'
               '"description":"truck at Bay 2","state":"ongoing"}]}}')
    record = []
    _sm_, _az = _process_once(_scene_cfg(), tmp_path / "clock.db", cycles=2,
                              analyzer=_real_analyzer(tmp_path, content, record=record))
    # Cycle 1 merged the situation; cycle 2's prompt renders its accumulated duration.
    system = record[1]["messages"][0]["content"]
    assert "truck at Bay 2" in system
    stamp = re.search(r"\[(\d+)h(\d+)m\]|\[(\d+)m(\d+)s\]", system)
    assert stamp is not None, "no duration stamp in the rendered scene block"
    assert stamp.group(1) is None, "an hours-wide duration means the clocks disagree"
    assert int(stamp.group(3)) * 60 + int(stamp.group(4)) < 3600


# ── chat: the block rides normal turns, never search turns ─────────────────────

def test_the_chat_scene_block_labels_every_stream_that_has_one():
    sm = _sm(_scene_cfg())
    sm._ensure_stream_state("cam0")
    sm._ensure_stream_state("cam1")
    sm._scene_memories["cam0"].merge(
        {"situations": [{"key": "bay2", "description": "truck at Bay 2", "state": "ongoing"}]},
        time.monotonic())
    block = sm.chat_scene_memory()
    assert "=== Camera cam0 ===" in block
    assert "truck at Bay 2" in block
    assert "cam1" not in block, "a stream with nothing to say must not get a bare heading"


def test_the_chat_scene_block_is_empty_when_nothing_has_been_observed():
    sm = _sm(_scene_cfg())
    sm._ensure_stream_state("cam0")
    assert sm.chat_scene_memory() == ""


def test_the_analyzer_is_given_the_chat_provider_only_when_scene_delta_is_on():
    class _Az(_StubAnalyzer):
        chat_scene_provider = None

    on = _Az()
    sm_on = _sm(_scene_cfg(), analyzer=on)
    assert on.chat_scene_provider == sm_on.chat_scene_memory

    off = _Az()
    _sm(_scene_cfg(scene_delta=False), analyzer=off)
    assert off.chat_scene_provider is None



def test_stop_cancels_the_smart_record_sampler_too():
    # stop() used to list the per-camera task dicts by hand and missed the
    # smart-record sampler, which kept running after shutdown (found in the
    # MR !25 refactor, 2026-10-01). Every registered task is cancelled now.
    async def run():
        r = fakeredis.FakeStrictRedis()
        sm = StreamManager(_cfg(), analyzer=None, store=None,
                           conn_manager=_StubConn(), redis_client=r)
        sm.recordings.svr_sources = {"cam0"}
        sm._running = True
        sm._add_stream("cam0")
        svr = sm._tasks[("svr_done", "cam0")]
        await sm.stop()
        try:
            await svr
        except asyncio.CancelledError:
            pass
        assert svr.cancelled()
        assert all(t.done() for t in sm._tasks.values())

    asyncio.run(run())


def test_a_role_sidecar_stream_says_its_camera_and_feature():
    """Beside several role sidecars (the safety product) a stream is `<camera>@<role>`; the UI groups by camera."""
    r = fakeredis.FakeStrictRedis()
    r.set("iris:status:bay-a@ppe", _json.dumps({"running": True}))
    sm = StreamManager(_cfg(), analyzer=_StubAnalyzer(), store=None, conn_manager=_StubConn(), redis_client=r)
    sm._ensure_stream_state("bay-a@ppe")
    (stream,) = sm.list_streams()
    assert (stream["id"], stream["camera"], stream["feature"]) == ("bay-a@ppe", "bay-a", "ppe")


def _cfg_with_zones_module(tmp_path, zones_settings):
    import yaml as yaml_mod
    p = tmp_path / "sidecar.yaml"
    p.write_text(yaml_mod.safe_dump({"modules": {"zones": zones_settings}}))
    cfg = _cfg()
    cfg["zones"] = {"sidecar_config_path": str(p)}
    return cfg


def test_zone_incidents_are_on_exactly_when_the_zones_module_sends_end_markers(tmp_path):
    r = fakeredis.FakeStrictRedis()
    on = StreamManager(_cfg_with_zones_module(tmp_path, {"end_markers": True}), analyzer=_StubAnalyzer(),
                       store=None, conn_manager=_StubConn(), redis_client=r)
    assert on.zone_alerts._incidents is on.count_alerts._incidents is on.zone_incidents
    assert on.zone_incidents.sidecar_restarted in on.fire.on_restart
    off = StreamManager(_cfg_with_zones_module(tmp_path, {}), analyzer=_StubAnalyzer(),
                        store=None, conn_manager=_StubConn(), redis_client=r)
    assert off.zone_alerts._incidents is None and off.count_alerts._incidents is None


def test_a_zone_intrusion_names_its_tracked_object_for_the_clip():
    """The intruder's id goes into the event's details, so the clip draws that person (app/overlay_track.py)."""
    from app.usecases.zone_alerts import zone_alert_details
    assert zone_alert_details("17") == {"object_id": "17"}
    # what crossed, as data too: a console words the alert in its own language (the safety console's Arabic)
    assert zone_alert_details("17", "car") == {"object_id": "17", "object_label": "car"}
    assert zone_alert_details(None, "person") == {"object_label": "person"}
    assert zone_alert_details(str(0xFFFFFFFFFFFFFFFF)) is None          # never tracked
    assert zone_alert_details("") is None and zone_alert_details(None) is None
