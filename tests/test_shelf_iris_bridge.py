"""products/shelf_iris_bridge: shelf_void alerts (iris:alert:<camera>, or <camera>@<role> with role_scoped_keys) ->
iris-platform messages, its own Redis consumer group, the alert stream names, the mapping file, settings and its compose
service. No broker and no pika: publish is a list."""
import json
import pathlib
import types
import uuid

import fakeredis
import pytest
import yaml

from products.shelf_iris_bridge.alerts import AlertReader
from products.shelf_iris_bridge.bridge import EVENT_NS, Bridge
from products.shelf_iris_bridge.catalog import EVENT_TYPE, PIPELINE_ID
from perception.engine.modules import EngineServices
from perception.engine.sources import stream_key_id
from products.shelf_iris_bridge.config import Settings
from products.shelf_iris_bridge.keys import KeyScopeError, alert_keys, camera_of, load_alert_keys
from products.shelf_iris_bridge.mapping import Mapping, MappingError

REPO = pathlib.Path(__file__).resolve().parents[1]
BRIDGE = REPO / "products/shelf_iris_bridge"
P1, P2 = "11111111-1111-4111-8111-111111111111", "22222222-2222-4222-8222-222222222222"
CAM, OTHER = "c0c0c0c0-0000-4000-8000-0000000000c1", "c0c0c0c0-0000-4000-8000-0000000000c2"
MAPPING = {"cam-shelf-test": {"R1": {"pipelineId": P1, "cameraId": CAM}, "R2": {"pipelineId": P2, "cameraId": CAM}}}


def _bridge(**settings):
    sent = []
    b = Bridge(Settings(amqp_url="amqp://ai:secret@rabbit:5672/iris", **settings), Mapping(MAPPING),
               lambda rk, body: sent.append((rk, body)), mapping_name="map.yaml")
    return b, sent


def _upsert(pid=P1, cams=(CAM,), state="active", catalog=PIPELINE_ID, **extra):
    return {"pipelineId": pid, "catalogPipelineId": catalog, "state": state, "config": {},
            "cameras": [{"cameraId": c, "name": f"cam {c[-2:]}", "streamUrl": "rtsp://x", "password": "pw"}
                        for c in cams], **extra}


def _raised(region="R1", reason="new", spots=None, alert_id="a1"):
    spots = spots or [{"bbox": [0.69, 0.66, 0.72, 0.76], "detectorScore": 0.15, "stability": 1.0}]
    return {"v": "1", "kind": "shelf_empty", "severity": "warning", "category": "operations", "region": region,
            "reason": reason, "alert_id": alert_id, "occurred_at": "2026-11-12T10:00:47.080Z", "count": str(len(spots)),
            "spots": json.dumps(spots), "score": "0.150", "stability": "1.000", "bbox": "1,2,3,4", "pts": "47.080"}


def _cleared(region="R1", alert_id="a1"):
    return {"v": "1", "kind": "shelf_restocked", "region": region, "reason": "gone", "alert_id": alert_id,
            "occurred_at": "2026-11-12T10:01:30.000Z", "clear_s": "30", "count": "0"}


# --- startup and iris-platform -> bridge ---

def test_startup_sends_the_catalog_then_a_state_sync_request():
    b, _ = _bridge()
    (rk1, cat), (rk2, sync) = b.startup_messages()
    assert (rk1, rk2, sync) == ("catalog.snapshot", "state.sync.request", {"target": "cloud"})
    assert [p["id"] for p in cat["pipelines"]] == [PIPELINE_ID]


@pytest.mark.parametrize("settings", [{"publish_catalog": False}, {"target": "edge.acme"}])
def test_no_catalog_when_switched_off_or_on_an_edge(settings):
    b, _ = _bridge(**settings)
    assert [rk for rk, _ in b.startup_messages()] == ["state.sync.request"]


def test_a_mapped_pipeline_is_synced_with_its_deepstream_regions():
    b, sent = _bridge()
    b.handle("pipeline.upsert.cloud", _upsert())
    assert sent == [("pipeline.status", {"pipelineId": P1, "status": "synced",
                                         "detail": "active on 1 camera (DeepStream cam-shelf-test/R1)"})]


def test_a_pipeline_with_an_unmapped_camera_fails_and_names_it():
    b, sent = _bridge()
    b.handle("pipeline.upsert.cloud", _upsert(cams=(CAM, OTHER)))
    ((rk, st),) = sent
    assert st["status"] == "failed" and OTHER in st["detail"] and "map.yaml" in st["detail"]
    assert "pw" not in json.dumps(st)


def test_another_catalog_pipeline_fails_and_zones_or_config_are_acknowledged_as_not_applied():
    b, sent = _bridge()
    b.handle("pipeline.upsert.cloud", _upsert(catalog="fire-detection"))
    b.handle("pipeline.upsert.cloud", _upsert(config={"clearSeconds": 10}))
    (_, a), (_, c) = sent
    assert a["status"] == "failed" and "fire-detection" in a["detail"]
    assert c["status"] == "synced" and c["detail"].endswith("config and zones are set in the DeepStream shelf preset")


def test_remove_is_always_synced():
    b, sent = _bridge()
    b.handle("pipeline.remove.cloud", {"pipelineId": P2})
    assert sent == [("pipeline.status", {"pipelineId": P2, "status": "synced", "detail": "removed"})]


# --- shelf alerts -> events ---

def test_shelf_empty_is_event_raised_with_a_correlation_key_per_region():
    b, sent = _bridge()
    assert b.on_alert("cam-shelf-test", "1700000000000-0", _raised())
    ((rk, ev),) = sent
    assert rk == "event.raised"
    assert (ev["pipelineId"], ev["cameraId"], ev["type"], ev["state"]) == (P1, CAM, EVENT_TYPE, "raised")
    assert ev["correlationKey"] == f"shelf-void:{CAM}:R1"
    assert ev["occurredAt"] == "2026-11-12T10:00:47.080Z" and ev["confidence"] == 1.0
    assert ev["eventId"] == str(uuid.uuid5(EVENT_NS, "cam-shelf-test/1700000000000-0"))
    m = ev["metadata"]
    assert (m["reason"], m["region"], m["alertId"], m["detectorScore"], m["spotCount"]) == ("new", "R1", "a1", 0.15, 1)
    assert m["spots"] == [{"bbox": [0.69, 0.66, 0.72, 0.76], "detectorScore": 0.15, "stability": 1.0}]
    assert ev["description"] == "1 empty spot on the shelf"


def test_regions_of_one_camera_report_to_their_own_pipeline():
    b, sent = _bridge()
    b.on_alert("cam-shelf-test", "1-0", _raised("R1"))
    b.on_alert("cam-shelf-test", "2-0", _raised("R2", alert_id="a2"))
    assert [(e["pipelineId"], e["correlationKey"]) for _, e in sent] == \
        [(P1, f"shelf-void:{CAM}:R1"), (P2, f"shelf-void:{CAM}:R2")]


def test_detector_confidence_mode():
    b, sent = _bridge(confidence_mode="detector")
    b.on_alert("cam-shelf-test", "1-0", _raised())
    assert sent[0][1]["confidence"] == 0.15


def test_shelf_restocked_is_event_cleared_with_the_same_key():
    b, sent = _bridge()
    b.on_alert("cam-shelf-test", "1-0", _raised())
    b.on_alert("cam-shelf-test", "2-0", _cleared())
    (_, r), (rk, c) = sent
    assert rk == "event.cleared" and c["correlationKey"] == r["correlationKey"]
    assert c["description"] == "No empty spots for 30 s" and "confidence" not in c
    assert b.open == {}


@pytest.mark.parametrize("fields", [_raised(region="R9"), {"kind": "ppe_violation", "region": "R1"},
                                    {**_raised(), "spots": "[]"}, {**_raised(), "occurred_at": "2026-11-12T10:00"},
                                    {**_cleared(), "reason": "weird"}])
def test_unmapped_regions_other_kinds_and_malformed_alerts_send_nothing(fields):
    b, sent = _bridge()
    assert b.on_alert("cam-shelf-test", "1-0", fields) is False and sent == []


def test_nothing_is_sent_for_a_paused_pipeline_and_pausing_clears_an_open_region():
    b, sent = _bridge()
    b.handle("pipeline.upsert.cloud", _upsert())
    b.on_alert("cam-shelf-test", "1-0", _raised())
    b.handle("pipeline.upsert.cloud", _upsert(state="paused"))
    assert [(rk, e.get("state"), e.get("metadata", {}).get("reason")) for rk, e in sent] == [
        ("pipeline.status", None, None), ("event.raised", "raised", "new"),
        ("event.cleared", "cleared", "paused"), ("pipeline.status", None, None)]
    assert sent[2][1]["correlationKey"] == f"shelf-void:{CAM}:R1"
    assert b.on_alert("cam-shelf-test", "2-0", _raised()) is False


def test_a_camera_removed_from_the_pipeline_clears_its_open_region_and_its_alerts_are_skipped():
    b, sent = _bridge()
    b.handle("pipeline.upsert.cloud", _upsert())
    b.on_alert("cam-shelf-test", "1-0", _raised())
    b.handle("pipeline.upsert.cloud", _upsert(cams=()))
    assert sent[2][0] == "event.cleared" and sent[2][1]["metadata"]["reason"] == "camera_removed"
    assert b.on_alert("cam-shelf-test", "2-0", _raised()) is False


def test_the_shelf_void_modules_own_alerts_become_valid_events():
    """The module (stability: tracker) feeds the bridge directly: a confirmed spot in R1, then cleared."""
    from perception.engine.frame import FrameOutput, FrameView, ObjectView
    from perception.engine.modules import EngineServices, build_settings
    from perception.modules.shelf_void import MODULE, Settings as ShelfSettings
    emitted = []
    services = EngineServices(redis=None, redis_factory=lambda: None,
                              emit=lambda key, f, maxlen=64: emitted.append((key, f)),
                              request_clip=lambda *a, **k: None, stream_ids=["cam-shelf-test"],
                              timer=lambda *a: None, telemetry=None, batch_size=1,
                              states={"cam-shelf-test": types.SimpleNamespace(module_state={})})
    roi = {"cam-shelf-test": {"size": [1920, 1080], "regions": [[[1369, 328], [1684, 352], [1581, 931],
                                                                  [1294, 933], [1357, 327]]]}}
    mod = MODULE(build_settings(ShelfSettings, {"stability": "tracker", "tracker": {"fps": 25},
                                                "situation": {"clear_s": 2}, "roi": roi}, "shelf_void"), services)
    mod.on_built(None)
    void = ObjectView(0xFFFFFFFFFFFFFFFF, "empty", 0.15, 0.15, 740, 400, 30, 60, gie_name="pgie_shelf")
    for i in range(200):
        t = i / 25
        mod.on_frame(FrameView("cam-shelf-test", None, 0, i, t, t, mux_size=(1080, 720),
                               _objects=[void] if i < 100 else []), FrameOutput())
    alerts = [f for k, f in emitted if k == "iris:alert:cam-shelf-test"]
    b, sent = _bridge()
    for n, f in enumerate(alerts):
        b.on_alert("cam-shelf-test", f"{n}-0", f)
    assert [rk for rk, _ in sent] == ["event.raised", "event.cleared"]
    raised, cleared = sent[0][1], sent[1][1]
    assert raised["metadata"]["region"] == "R1" and raised["confidence"] == 1.0
    assert raised["metadata"]["spots"][0]["bbox"] == [round(740 / 1080, 4), round(400 / 720, 4),
                                                      round(770 / 1080, 4), round(460 / 720, 4)]
    assert cleared["description"] == "No empty spots for 2 s"
    for ev in (raised, cleared):                          # the IRIS zod contract, the parts we set
        assert set(ev) >= {"eventId", "pipelineId", "cameraId", "type", "state", "occurredAt", "correlationKey"}
        assert ev["occurredAt"].endswith("Z") and len(ev["correlationKey"]) <= 200


# --- Redis: the bridge's own consumer group ---

def _redis():
    return fakeredis.FakeRedis(decode_responses=True)


def test_the_group_reads_every_new_alert_and_the_apps_plain_xread_still_sees_them_all():
    r = _redis()
    got = []
    reader = AlertReader(r, ["iris:alert:cam-shelf-test"], "shelf-iris-bridge", "c1", lambda *a: got.append(a),
                         block_ms=1)
    r.xadd("iris:alert:cam-shelf-test", {"kind": "old"})                   # before the group existed: not sent
    reader.ensure_groups()
    ids = [r.xadd("iris:alert:cam-shelf-test", _raised()), r.xadd("iris:alert:cam-shelf-test", _cleared())]
    reader.poll()                                                           # no pending entries yet
    reader.poll()
    assert [(cam, eid, f["kind"]) for cam, eid, f in got] == [
        ("cam-shelf-test", ids[0], "shelf_empty"), ("cam-shelf-test", ids[1], "shelf_restocked")]
    assert [f["kind"] for _, f in r.xread({"iris:alert:cam-shelf-test": "0"})[0][1]] == \
        ["old", "shelf_empty", "shelf_restocked"]                           # the app's reader is unaffected
    assert r.xpending("iris:alert:cam-shelf-test", "shelf-iris-bridge")["pending"] == 0


def test_after_a_restart_unacked_entries_are_sent_again_first():
    r = _redis()
    key = "iris:alert:cam-shelf-test"
    AlertReader(r, ["iris:alert:cam-shelf-test"], "g", "c1", None).ensure_groups()
    eid = r.xadd(key, _raised())
    r.xreadgroup("g", "c1", {key: ">"})                                     # read, then the bridge died
    got = []
    reader = AlertReader(r, ["iris:alert:cam-shelf-test"], "g", "c1", lambda *a: got.append(a[1]), block_ms=1)
    reader.ensure_groups()                                                  # the group already exists: kept
    reader.poll()
    assert got == [eid] and r.xpending(key, "g")["pending"] == 0


def test_a_handler_error_is_logged_acked_and_the_next_entry_still_handled(capsys):
    r = _redis()
    got = []

    def handler(cam, eid, f):
        if f["kind"] == "boom":
            raise RuntimeError("bad")
        got.append(f["kind"])
    reader = AlertReader(r, ["iris:alert:a"], "g", "c", handler, block_ms=1)
    reader.ensure_groups()
    r.xadd("iris:alert:a", {"kind": "boom"})
    r.xadd("iris:alert:a", {"kind": "ok"})
    reader.poll()
    reader.poll()
    assert got == ["ok"] and "alert handler error" in capsys.readouterr().out


# --- mapping, settings, packaging ---

def test_the_example_mapping_loads():
    m = Mapping.load(str(BRIDGE / "mapping.example.yaml"))
    assert m.cameras == ["cam-shelf-test"] and set(r for _, r in m.targets) == {"R1", "R2"}


@pytest.mark.parametrize("raw", [None, {}, {"cam": {}}, {"cam": {"R1": {"pipelineId": P1}}},
                                 {"cam": {"R1": {"pipelineId": "p1", "cameraId": CAM}}},
                                 {"cam": {"R1": {"pipelineId": P1, "cameraId": CAM, "zone": "x"}}}])
def test_a_bad_mapping_refuses_startup(raw):
    with pytest.raises(MappingError):
        Mapping(raw)


def test_a_missing_mapping_file_says_which(tmp_path):
    with pytest.raises(MappingError, match="cannot read"):
        Mapping.load(str(tmp_path / "nope.yaml"))


def test_settings_from_env_and_no_password_in_the_safe_urls():
    s = Settings.from_env({"AMQP_URL": "amqp://ai-shelf:s3cret@rabbit:5672/iris", "REDIS_URL": "redis://:pw@r:6379/0",
                           "PUBLISH_CATALOG": "false", "CONSUMER_NAME": "b1"})
    assert (s.amqp_user, s.inbound_queue, s.publish_catalog, s.group, s.consumer) == \
        ("ai-shelf", "ai.cloud.inbound", False, "shelf-iris-bridge", "b1")
    assert "s3cret" not in s.safe_url and "pw" not in s.safe_redis_url
    with pytest.raises(ValueError):
        Settings(target="edge.")


def test_vendored_files_name_their_shelf_demo_commit():
    for name in ("broker", "messages", "catalog", "log"):
        first = (BRIDGE / f"{name}.py").read_text().split("\n", 1)[0]
        assert first.startswith("# Vendored from shelf-demo") and "commit " in first


def test_the_compose_service_is_opt_in_with_its_env_file_and_mapping():
    class Loader(yaml.SafeLoader):                         # compose's own !override / !reset tags
        pass
    Loader.add_multi_constructor("!", lambda loader, tag, node: loader.construct_sequence(node)
                                 if isinstance(node, yaml.SequenceNode) else None)
    compose = yaml.load((REPO / "docker-compose.shelf-live.yml").read_text(), Loader=Loader)
    svc = compose["services"]["shelf-iris-bridge"]
    assert svc["profiles"] == ["iris-platform"] and svc["build"] == "./products/shelf_iris_bridge"
    assert svc["env_file"] == [{"path": "./products/shelf_iris_bridge/.env", "required": False}]
    assert any(v.endswith(":/config/shelf-iris-mapping.yaml:ro") for v in svc["volumes"])
    # the sidecar's own preset and the engine's key helper, from the checkout: the streams it reads are the sidecar's
    assert {"SIDECAR_CONFIG=${SIDECAR_CONFIG:-}", "IRIS_ROOT=/workspace/iris", "PYTHONPATH=/workspace/iris"} \
        <= set(svc["environment"])
    assert {"./configs:/workspace/iris/configs:ro", "./perception:/workspace/iris/perception:ro"} <= set(svc["volumes"])


def test_the_env_example_holds_no_secret():
    text = (BRIDGE / ".env.example").read_text()
    assert "AMQP_URL=amqp://<user>:<password>@" in text
    assert not [ln for ln in text.splitlines() if "PASSWORD=" in ln.upper() and not ln.startswith("#")]


# --- which streams: the sidecar's own role_scoped_keys and role, the engine's own key helper ---

def test_streams_are_per_camera_by_default_and_carry_the_role_with_role_scoped_keys():
    cams = ["cam-a", "cam-b"]
    off = alert_keys(cams, {"sidecar_role": "shelf"})
    assert (off.mode, off.scoped, off.streams) == ("camera", False, ("iris:alert:cam-a", "iris:alert:cam-b"))
    on = alert_keys(cams, {"sidecar_role": "shelf", "role_scoped_keys": True})
    assert (on.mode, on.role) == ("role-scoped", "shelf")
    assert on.streams == ("iris:alert:cam-a@shelf", "iris:alert:cam-b@shelf")
    assert [camera_of(s) for s in on.streams + off.streams] == cams + cams


def test_the_role_is_resolved_as_the_engine_does_env_first_then_its_overrides():
    cfg = {"sidecar_role": "yaml-role", "role_scoped_keys": True}
    assert alert_keys(["c"], cfg, env_role="env-role").streams == ("iris:alert:c@env-role",)
    assert alert_keys(["c"], cfg).streams == ("iris:alert:c@yaml-role",)
    # role_scoped_keys switched on by the role's own override entry
    cfg = {"sidecar_role": "shelf", "role_overrides": {"shelf": {"role_scoped_keys": True}}}
    assert alert_keys(["c"], cfg).streams == ("iris:alert:c@shelf",)
    # set-but-empty IRIS_SIDECAR_ROLE means no role, as in perception/engine/main.py
    assert alert_keys(["c"], {"sidecar_role": "shelf"}, env_role="").role is None


@pytest.mark.parametrize("cfg, cams, match", [({"role_scoped_keys": True}, ["c"], "needs a role"),
                                              ({"role_scoped_keys": True, "sidecar_role": "r"}, ["c@x"], "contains")])
def test_role_scoped_keys_the_engine_would_refuse_refuse_the_bridge_too(cfg, cams, match):
    with pytest.raises(KeyScopeError, match=match):
        alert_keys(cams, cfg)


def test_the_shelf_preset_loads_with_extends_and_reads_camera_keyed_streams():
    keys = load_alert_keys(["cam-shelf-test"], "configs/presets/shelf/live-test.yaml", root=str(REPO))
    assert (keys.mode, keys.streams) == ("camera", ("iris:alert:cam-shelf-test",))


def test_a_missing_or_unreadable_sidecar_config_refuses_startup(tmp_path):
    with pytest.raises(KeyScopeError, match="SIDECAR_CONFIG is required"):
        load_alert_keys(["c"], "")
    with pytest.raises(KeyScopeError, match="cannot read"):
        load_alert_keys(["c"], str(tmp_path / "nope.yaml"))


def test_settings_read_the_sidecar_config_and_role():
    s = Settings.from_env({"SIDECAR_CONFIG": " configs/x.yaml ", "IRIS_ROOT": "/w"})
    assert (s.sidecar_config, s.sidecar_role, s.iris_root) == ("configs/x.yaml", None, "/w")
    assert Settings.from_env({"IRIS_SIDECAR_ROLE": ""}).sidecar_role == ""


def _deliver(sidecar_cfg):
    """The sidecar side as the engine wires it (EngineServices.alert, key_id = stream_key_id of its role and
    role_scoped_keys) -> Redis -> the bridge's reader on the streams it resolved from the same yaml -> iris-platform.
    Returns (the streams written, what was published)."""
    r = _redis()
    role, scoped = sidecar_cfg["sidecar_role"], bool(sidecar_cfg.get("role_scoped_keys"))
    ids = iter(["1762941647080-0", "1762941690000-0"])
    services = EngineServices(redis=r, redis_factory=lambda: r,
                              emit=lambda key, fields, maxlen=64: bool(r.xadd(key, fields, id=next(ids),
                                                                              maxlen=maxlen)),
                              request_clip=lambda *a, **k: None, stream_ids=["cam-shelf-test"],
                              timer=lambda i, f: None, telemetry=None, role=role,
                              key_id=lambda sid: stream_key_id(sid, role, scoped))
    b, sent = _bridge()
    b.handle("pipeline.upsert.cloud", _upsert())
    sent.clear()
    reader = AlertReader(r, alert_keys(Mapping(MAPPING).cameras, sidecar_cfg).streams, "shelf-iris-bridge", "c1",
                         b.on_alert, block_ms=1)
    reader.ensure_groups()
    for fields in (_raised(), _cleared()):
        extra = {k: v for k, v in fields.items() if k not in ("v", "kind", "severity", "category", "count", "pts")}
        assert services.alert("cam-shelf-test", fields["kind"], severity=fields.get("severity", "info"),
                              category=fields.get("category", "operations"), count=int(fields["count"]), **extra)
    reader.poll()
    reader.poll()
    return sorted(r.keys("iris:alert:*")), sent


def test_off_and_on_deliver_the_same_event_to_iris_platform():
    off_streams, off = _deliver({"sidecar_role": "shelf"})
    on_streams, on = _deliver({"sidecar_role": "shelf", "role_scoped_keys": True})
    assert (off_streams, on_streams) == (["iris:alert:cam-shelf-test"], ["iris:alert:cam-shelf-test@shelf"])
    assert [rk for rk, _ in off] == ["event.raised", "event.cleared"]
    raised = off[0][1]
    assert raised["metadata"]["deepstreamCamera"] == "cam-shelf-test"            # the @role suffix stripped
    assert raised["eventId"] == str(uuid.uuid5(EVENT_NS, "cam-shelf-test/1762941647080-0"))
    assert (raised["pipelineId"], raised["cameraId"], raised["correlationKey"]) == (P1, CAM, f"shelf-void:{CAM}:R1")
    assert on == off                                                              # byte for byte the same events
