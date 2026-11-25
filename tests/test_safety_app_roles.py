"""The app beside several role sidecars (safety integration, ticket 2): it reads every role's config, its streams are
the roles' key ids (<camera>@<role>), and what it shows says which camera and which feature."""
import asyncio
import json

import yaml

from app.events import EventStore
from app.features import camera_of, feature_of
from app.sidecar_config import running_sidecar_config
from app.zones import zone_by_id


def _role(tmp_path, role, cameras, modules):
    path = tmp_path / f"{role}.yaml"
    path.write_text(yaml.safe_dump({
        "redis_url": "redis://redis:6379/0", "role_scoped_keys": True, "sidecar_role": role,
        "mux_width": 1280, "mux_height": 720, "modules": modules,
        "sources": [{"id": c, "uri": f"rtsp://go2rtc:8554/raw-{c}", "roles": [role]} for c in cameras]}))
    return str(path)


def test_one_plain_config_is_read_as_before(tmp_path, monkeypatch):
    plain = tmp_path / "plain.yaml"
    plain.write_text(yaml.safe_dump({"sidecar_role": "fire", "sources": [{"id": "cam-1", "uri": "x"}]}))
    monkeypatch.setenv("SIDECAR_CONFIG", str(plain))
    assert [s["id"] for s in running_sidecar_config({})["sources"]] == ["cam-1"]     # as written


def test_one_role_scoped_config_names_its_streams_as_its_sidecar_does(tmp_path, monkeypatch):
    monkeypatch.setenv("SIDECAR_CONFIG", _role(tmp_path, "ppe", ["bay-a"], {"ppe": {}}))
    assert [s["id"] for s in running_sidecar_config({})["sources"]] == ["bay-a@ppe"]


def test_several_role_configs_merge_with_key_id_sources(tmp_path, monkeypatch):
    paths = [_role(tmp_path, "proximity", ["bay-a", "bay-b"], {"proximity": {"preview_masks": True}}),
             _role(tmp_path, "zones", ["bay-a"], {"zones": {"end_markers": True}})]
    monkeypatch.setenv("SIDECAR_CONFIG", ",".join(paths))
    cfg = running_sidecar_config({})
    assert set(cfg["modules"]) == {"proximity", "zones"}
    assert cfg["modules"]["zones"] == {"end_markers": True}
    assert [(s["id"], s["roles"]) for s in cfg["sources"]] == [
        ("bay-a@proximity", ["proximity"]), ("bay-b@proximity", ["proximity"]), ("bay-a@zones", ["zones"])]
    assert (cfg["mux_width"], cfg["mux_height"]) == (1280, 720)


def test_a_role_without_scoped_keys_keeps_its_camera_ids(tmp_path, monkeypatch):
    plain = tmp_path / "plain.yaml"
    plain.write_text(yaml.safe_dump({"modules": {"zones": {}}, "sources": [{"id": "cam-1", "uri": "x"}]}))
    monkeypatch.setenv("SIDECAR_CONFIG", f"{plain},{_role(tmp_path, 'ppe', ['cam-2'], {'ppe': {}})}")
    assert [s["id"] for s in running_sidecar_config({})["sources"]] == ["cam-1", "cam-2@ppe"]


def test_camera_and_feature_of_a_stream():
    assert camera_of("bay-a@ppe") == "bay-a" and camera_of("cam-1") == "cam-1"
    assert feature_of("bay-a@ppe", ["ppe_violation"]) == "ppe"
    assert feature_of("cam-1", ["fire_smoke"]) == "fire"                  # unscoped: from the event's tag
    assert feature_of("cam-1", ["zone_intrusion"]) == "zones"
    assert feature_of("cam-1", ["zone_overcrowding"]) == "zones"
    assert feature_of("cam-1", ["proximity_critical"]) == "proximity"
    assert feature_of("cam-1", ["fall"]) == "fall"
    assert feature_of("cam-1", ["camera_health"]) == "health"
    assert feature_of("cam-1", ["something_else"]) is None


def test_a_zone_is_found_by_the_camera_of_a_role_stream(tmp_path, monkeypatch):
    zones = tmp_path / "zones_config.txt"
    zones.write_text(json.dumps({"bay-a": {"zones": [
        {"id": "z1", "name": "Dock", "points": [[0, 0], [100, 0], [100, 100]]}]}}))
    monkeypatch.setenv("SIDECAR_CONFIG", _role(tmp_path, "zones", ["bay-a"], {"zones": {}}))
    config = {"zones": {"file_path": str(zones)}}
    assert zone_by_id(config, "bay-a@zones", "z1")["name"] == "Dock"
    assert zone_by_id(config, "bay-a", "z1")["name"] == "Dock"


def test_an_event_says_its_camera_and_feature(tmp_path):
    store = EventStore({"events": {"db_path": str(tmp_path / "events.db"), "retention_hours": 48}})

    async def run():
        await store.init()
        saved = await store.save("alert", "Person #3 without helmet", ["ppe_violation"], camera="bay-a@ppe")
        return saved, await store.get_event(saved["id"])

    saved, got = asyncio.run(run())
    for event in (saved, got):
        assert (event["camera"], event["camera_id"], event["feature"]) == ("bay-a@ppe", "bay-a", "ppe")


def test_the_roi_file_is_rendered_for_the_zones_sidecar_s_own_cameras(tmp_path):
    """The ROI file must be numbered by the zones sidecar's own camera order and found by camera id, not by the merged
    list of every role's streams: the zones sidecar renders it itself, from its own sources (camera ids)."""
    from perception.modules.zones.analytics import read_zones, render_analytics_config, slot_map

    zones = tmp_path / "zones_config.txt"
    zones.write_text(json.dumps({"office": {"zones": [
        {"id": "z0", "name": "Server room", "points": [[10, 10], [200, 10], [200, 200]],
         "intrusion": {"mode": "restricted", "classes": ["person"]}}]}}))
    own = [{"id": "bay-a"}, {"id": "office"}]
    text = render_analytics_config(read_zones(str(zones)), slot_map(own), 1280, 720)
    assert "[roi-filtering-stream-1]" in text and "roi-z0=10;10;200;10;200;200" in text   # office: 2nd zones camera
    assert text.count("[roi-filtering-stream-") == 1


# ── after an Apply the app follows the new role configs: clips, camera roles, the ROI file ─────────────────────────

def test_the_pipeline_settings_are_parsed_once_until_a_file_changes(tmp_path, monkeypatch):
    import os
    import app.sidecar_config as sc_mod
    path = _role(tmp_path, "ppe", ["bay-a"], {"ppe": {}})
    monkeypatch.setenv("SIDECAR_CONFIG", path)
    calls = []
    real = sc_mod.load_sidecar_config
    monkeypatch.setattr(sc_mod, "load_sidecar_config", lambda p: calls.append(p) or real(p))
    sc_mod.running_sidecar_config({})
    sc_mod.running_sidecar_config({})
    assert len(calls) == 1
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    assert [s["id"] for s in sc_mod.running_sidecar_config({})["sources"]] == ["bay-a@ppe"] and len(calls) == 2


def test_a_change_in_the_role_configs_refreshes_clips_roles_and_tells_the_server(tmp_path, monkeypatch):
    import fakeredis
    from app.stream_manager import StreamManager
    zones = _role(tmp_path, "zones", ["bay-a"], {"zones": {}})
    monkeypatch.setenv("SIDECAR_CONFIG", zones)
    cfg = {"perception": {"redis_url": "redis://unused"}, "zones": {}, "sampling": {}, "monitoring": {}}
    sm = StreamManager(cfg, analyzer=None, store=None, conn_manager=None, redis_client=fakeredis.FakeStrictRedis())
    told = []
    sm.on_sources_changed.append(lambda: told.append(True))
    assert not sm.refresh_config()                                   # nothing changed since it started
    _role(tmp_path, "zones", ["bay-a", "office"], {"zones": {}})       # an Apply added office to the zones role
    import os
    st = os.stat(zones)
    os.utime(zones, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    assert sm.refresh_config() and told == [True]
    assert "office@zones" in sm.recordings.svr_sources


def test_after_a_source_change_the_zones_are_announced_again():
    """Each zones process re-renders its ROI config for its own cameras on the bump."""
    from app.server import _follow_source_changes
    bumps = []
    hook = _follow_source_changes(type("SM", (), {"zones_bump": lambda self: bumps.append(1) or 7})())
    hook()
    assert bumps == [1]
