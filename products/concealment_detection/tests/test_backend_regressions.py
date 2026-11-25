"""Regression coverage for configuration, Redis boundaries, and review persistence."""
from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import fields
import json
import sqlite3

import pytest
import yaml

from concealment.types import DetectorSettings
from config import load_config
from evidence import build_candidate_record
from storage import ReviewStore, connect
from test_evidence_and_clock import _sample_candidate


def write_config(root, **overrides):
    raw = {"cameras": ["cam-1"], "zones_path": "zones.yaml"}
    raw.update(overrides)
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    (root / "zones.yaml").write_text(
        "cameras:\n  cam-1:\n    zone_free: true\n", encoding="utf-8"
    )
    return path


def test_config_extends_merges_sections_and_resolves_declaring_file_paths(tmp_path):
    base = tmp_path / "base"
    child = tmp_path / "child"
    base.mkdir()
    child.mkdir()
    parent = write_config(
        base, data_dir="records", stream_starts_path="starts.json",
        clip_source_dir="clips", mux_shape=[1280, 720],
        detector={"dwell_min_s": 4.0, "min_hold_s": 0.6},
        retention={"review_days": 14, "queue_maxsize": 12},
    )
    override = child / "override.yaml"
    override.write_text(yaml.safe_dump({
        "extends": "../base/config.yaml", "data_dir": "local-data",
        "detector": {"min_hold_s": 0.8}, "retention": {"queue_maxsize": 24},
    }), encoding="utf-8")
    cfg = load_config(override, env={})
    assert cfg.zones_path == (base / "zones.yaml").resolve()
    assert cfg.data_dir == (child / "local-data").resolve()
    assert cfg.stream_starts_path == (base / "starts.json").resolve()
    assert cfg.clip_source_dir == (base / "clips").resolve()
    assert cfg.mux_shape == (1280, 720)
    assert cfg.detector.dwell_min_s == 4.0
    assert cfg.detector.min_hold_s == 0.8
    assert cfg.review_days == 14
    assert cfg.queue_maxsize == 24
    assert parent.exists()


def test_config_environment_selects_file_but_explicit_path_wins(tmp_path):
    selected = write_config(tmp_path, port=9012)
    assert load_config(env={"CONCEALMENT_CONFIG": str(selected)}).port == 9012
    alternate = tmp_path / "alternate.yaml"
    alternate.write_text("extends: config.yaml\nport: 9013\n", encoding="utf-8")
    cfg = load_config(alternate, env={
        "CONCEALMENT_CONFIG": str(selected), "CONCEALMENT_BACKEND_PORT": "9014",
        "CONCEALMENT_DATA_DIR": "env-records",
    })
    assert cfg.port == 9014
    assert cfg.data_dir == (tmp_path / "env-records").resolve()
    assert load_config(alternate, env={"CONCEALMENT_CONFIG": str(selected)}).port == 9013


def test_config_inheritance_cycle_rejected(tmp_path):
    path = write_config(tmp_path, extends="other.yaml")
    (tmp_path / "other.yaml").write_text("extends: config.yaml\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cycl"):
        load_config(path, env={})


def test_config_optional_service_fields_have_compatible_defaults(tmp_path):
    cfg = load_config(write_config(tmp_path), env={})
    assert cfg.mux_shape == (1920, 1080)
    assert cfg.stream_starts_path is None
    assert cfg.clip_source_dir is None


@pytest.mark.parametrize("cameras", [[], [""], ["   "], ["cam-1", "cam-1"],
                                      [{"id": "cam-1"}, {"id": "cam-1"}], [{"id": " "}]])
def test_config_rejects_empty_or_duplicate_camera_ids(tmp_path, cameras):
    with pytest.raises(ValueError, match="camera|Camera"):
        load_config(write_config(tmp_path, cameras=cameras), env={})


NUMERIC_THRESHOLDS = [f.name for f in fields(DetectorSettings)
                      if not isinstance(f.default, (bool, str))]


@pytest.mark.parametrize("name", NUMERIC_THRESHOLDS)
@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), "invalid", True])
def test_config_rejects_invalid_numeric_detector_thresholds(tmp_path, name, value):
    with pytest.raises(ValueError, match=name):
        load_config(write_config(tmp_path, detector={name: value}), env={})


@pytest.mark.parametrize("detector", [
    {"dwel_min_s": 2.0}, {"keypoint_conf_threshold": 1.01},
    {"min_core_conf_enter": 1.01}, {"min_core_conf_exit": 1.01},
    {"hidden_conf_drop_fraction": 1.01}, {"min_keypoints": 18},
    {"min_keypoints": 1.5}, {"max_conceal_elbow_deg": 181},
    {"min_core_conf_enter": 0.8, "min_core_conf_exit": 0.7},
    {"emit_low_tier": "false"},
])
def test_config_rejects_unknown_or_out_of_range_detector_settings(tmp_path, detector):
    with pytest.raises(ValueError):
        load_config(write_config(tmp_path, detector=detector), env={})


@pytest.mark.parametrize("name", ["queue_maxsize", "review_days", "keep_candidates"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_config_requires_positive_integer_retention_settings(tmp_path, name, value):
    with pytest.raises(ValueError, match=name):
        load_config(write_config(tmp_path, retention={name: value}), env={})


@pytest.mark.parametrize("shape", [[0, 1080], [-1, 10], [1], [1, 2, 3], [1.5, 2], "1920x1080"])
def test_config_rejects_invalid_mux_shape(tmp_path, shape):
    with pytest.raises(ValueError, match="mux_shape"):
        load_config(write_config(tmp_path, mux_shape=shape), env={})


@pytest.mark.parametrize("zones", [None, "cameras: {}\n", "cameras:\n  cam-1: {}\n"])
def test_config_missing_zones_never_silently_enables_zone_free(tmp_path, zones):
    path = write_config(tmp_path)
    zone_path = tmp_path / "zones.yaml"
    if zones is None:
        zone_path.unlink()
    else:
        zone_path.write_text(zones, encoding="utf-8")
    with pytest.raises(ValueError, match="zone|Zone"):
        load_config(path, env={})


def test_config_loads_optional_camera_clip_duration_for_runtime(tmp_path):
    cfg = load_config(write_config(tmp_path, cameras=[{"id": "cam-1", "clip_duration_s": 12.5}]), env={})
    assert cfg.cameras[0].clip_duration_s == 12.5
    assert load_config(write_config(tmp_path), env={}).cameras[0].clip_duration_s is None


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf"), "bad", True])
def test_config_rejects_invalid_camera_clip_duration(tmp_path, duration):
    path = write_config(tmp_path, cameras=[{"id": "cam-1", "clip_duration_s": duration}])
    with pytest.raises(ValueError, match="clip_duration_s"):
        load_config(path, env={})


@pytest.mark.parametrize("zone_free", ["false", "true", 1])
def test_config_requires_explicit_boolean_zone_free(tmp_path, zone_free):
    path = write_config(tmp_path)
    (tmp_path / "zones.yaml").write_text(yaml.safe_dump({
        "cameras": {"cam-1": {"zone_free": zone_free}},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="zone_free"):
        load_config(path, env={})


def test_config_accepts_explicit_zone_free_and_shipped_configuration(tmp_path):
    assert load_config(write_config(tmp_path), env={}).camera_zones["cam-1"].zone_free
    assert load_config(env={}).detector.dwell_min_s == 2.0


def pose_entry(**overrides):
    payload = {
        "shape": [960, 540],
        "items": [{"id": 7, "box": [430, 150, 530, 400], "kps": [[480, 270, 0.9]] * 17}],
    }
    payload.update(overrides)
    return {"frame_num": "5", "pts": "1.5", "pose": json.dumps(payload)}


def test_acl_skips_binary_jpeg_in_reader_and_parser():
    import acl
    from test_config_acl_storage_api import FakeRedis

    class UndecodableJPEG(bytes):
        def decode(self, *args, **kwargs):
            raise AssertionError("binary JPEG must not be decoded")

    entry = pose_entry()
    entry[b"jpeg"] = UndecodableJPEG(b"\xff\xd8\xff")
    assert acl.parse_pose_frame("cam-1", entry).observed
    redis = FakeRedis()
    reader = acl.MetaStreamReader(redis)
    assert reader.read("cam-1") == []
    redis.xadd("iris:meta:cam-1", entry)
    entries = reader.read("cam-1")
    assert "jpeg" not in entries[0]["fields"]
    assert acl.parse_pose_frame("cam-1", entries[0]).tracks[0].track_id == 7


def test_acl_scales_only_actual_seven_field_mux_detections():
    import acl
    entry = pose_entry()
    entry["dets"] = json.dumps([
        ["handbag", 0.9, 21, 940, 500, 40, 80],
        ["shopping_cart", 0.9, 22, 1000, 500, 80, 100],
    ])
    obs = acl.parse_pose_frame("cam-1", entry, mux_shape=(1920, 1080)).tracks[0]
    assert obs.bag_box == (470.0, 250.0, 490.0, 290.0)
    assert obs.cart_boxes == ((500.0, 250.0, 540.0, 300.0),)
    for detection in (["handbag", 0.9, 470, 250, 20, 40],
                      {"label": "handbag", "box": [470, 250, 490, 290]}):
        entry["dets"] = json.dumps([detection])
        assert acl.parse_pose_frame("cam-1", entry, mux_shape=(1920, 1080)).tracks[0].bag_box == (
            470.0, 250.0, 490.0, 290.0
        )


@pytest.mark.parametrize("label", ["bagpipe", "cartoon", "sandbag", "not_a_handbag"])
def test_acl_context_labels_match_exactly(label):
    import acl
    entry = pose_entry()
    entry["dets"] = json.dumps([[label, 0.9, 470, 250, 20, 40]])
    obs = acl.parse_pose_frame("cam-1", entry).tracks[0]
    assert not obs.has_bag
    assert obs.cart_boxes == ()


@pytest.mark.parametrize("pts", ["NaN", "Infinity", "-Infinity"])
def test_acl_rejects_nonfinite_timestamp_even_without_pose(pts):
    import acl
    with pytest.raises(ValueError):
        acl.parse_pose_frame("cam-1", {"frame_num": "1", "pts": pts})


@pytest.mark.parametrize("payload", [None, {}, [], {"shape": [960, 540]},
    {"shape": [0, 540], "items": []}, {"shape": [960, 540], "items": [{}]},
    {"shape": [960, 540], "items": [{"id": 1, "box": [1, 2, 3, 4], "kps": []}]},
])
def test_acl_rejects_invalid_pose_instead_of_synthesizing_observations(payload):
    import acl
    entry = pose_entry()
    entry["pose"] = json.dumps(payload)
    with pytest.raises(ValueError):
        acl.parse_pose_frame("cam-1", entry)


@pytest.mark.parametrize("field,value", [
    ("box", [1, 2, float("nan"), 4]), ("box", [4, 2, 1, 4]),
    ("kps", [[480, float("inf"), 0.9]] * 17),
    ("kps", [[480, 270, 1.1]] * 17), ("kps", [[480, 270, -0.1]] * 17),
    ("bag_box", [1, 2, float("inf"), 4]),
])
def test_acl_rejects_nonfinite_or_invalid_track_geometry(field, value):
    import acl
    entry = pose_entry()
    payload = json.loads(entry["pose"])
    payload["items"][0][field] = value
    entry["pose"] = json.dumps(payload)
    with pytest.raises(ValueError):
        acl.parse_pose_frame("cam-1", entry)


@pytest.mark.parametrize("detection", [
    ["handbag", 0.9, 470, 250, float("nan"), 40],
    ["cart", 0.9, 22, 940, 500, float("inf"), 80],
    {"label": "bag", "box": [470, 250, 490, float("nan")]},
    ["bag", 0.9, 470, 250, -20, 40],
])
def test_acl_rejects_invalid_context_geometry(detection):
    import acl
    entry = pose_entry()
    entry["dets"] = json.dumps([detection])
    with pytest.raises(ValueError):
        acl.parse_pose_frame("cam-1", entry)


def test_acl_rejects_present_but_empty_pose_field():
    import acl
    with pytest.raises(ValueError):
        acl.parse_pose_frame("cam-1", {"frame_num": "1", "pts": "1.0", "pose": ""})


def test_acl_empty_observed_pose_and_boxless_detections_are_valid():
    import acl
    entry = pose_entry(items=[])
    entry["dets"] = json.dumps([["handbag", 0.9, 1]])
    frame = acl.parse_pose_frame("cam-1", entry)
    assert frame.observed and frame.tracks == ()


@pytest.fixture
def store(tmp_path):
    value = ReviewStore(connect(tmp_path / "review.db"))
    yield value
    value.close()


def test_storage_before_cursor_paginates_without_duplicates_and_preserves_since(store):
    for _ in range(5):
        store.add_candidate(build_candidate_record(_sample_candidate()))
    assert [r["id"] for r in store.list_candidates(limit=2)] == [5, 4]
    assert [r["id"] for r in store.list_candidates(before=4, limit=2)] == [3, 2]
    assert [r["id"] for r in store.list_candidates(before=2, limit=2)] == [1]
    assert [r["id"] for r in store.list_candidates(since=2, before=5)] == [4, 3]
    assert store.list_candidates(before=0) == []
    assert store.list_candidates(camera="other", before=5) == []


def test_storage_roundtrips_evidence_and_migrates_legacy_database(tmp_path):
    path = tmp_path / "legacy.db"
    conn = connect(path)
    record = build_candidate_record(_sample_candidate())
    store = ReviewStore(conn)
    first = store.add_candidate(record)
    # Emulate an existing pre-migration SQLite database, including a saved row.
    columns = {r[1] for r in conn.execute("PRAGMA table_info(concealment_candidates)")}
    for name in ("terminal_stage", "hold_duration_s", "dwell_duration_s", "thumbnail_path"):
        if name in columns:
            conn.execute(f"ALTER TABLE concealment_candidates DROP COLUMN {name}")
    conn.commit()
    store.close()
    for _ in range(2):
        reopened = ReviewStore(connect(path))
        try:
            legacy = reopened.get_candidate(first)
            assert legacy["track_id"] == 42
            assert legacy["thumbnail_path"] is None
            record["thumbnail_path"] = "thumbnails/candidate.jpg"
            cid = reopened.add_candidate(record)
            row = reopened.get_candidate(cid)
            assert row["terminal_stage"] == "RETURNED"
            assert row["hold_duration_s"] == 1.4
            assert row["dwell_duration_s"] == 3.2
            assert row["thumbnail_path"] == "thumbnails/candidate.jpg"
            assert reopened.list_candidates()[0]["terminal_stage"] == "RETURNED"
        finally:
            reopened.close()


def test_storage_close_releases_connection(tmp_path):
    conn = connect(tmp_path / "close.db")
    store = ReviewStore(conn)
    store.close()
    store.close()
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")


def test_api_exposes_before_cursor_and_evidence_fields(store):
    from fastapi.testclient import TestClient
    from server import create_app
    for _ in range(4):
        store.add_candidate(build_candidate_record(_sample_candidate()))
    with TestClient(create_app({"ready": True}, store)) as client:
        response = client.get("/candidates", params={"before": 4, "since": 1, "limit": 2})
        assert response.status_code == 200
        assert [r["id"] for r in response.json()] == [3, 2]
        assert response.json()[0]["hold_duration_s"] == 1.4


def test_api_dynamic_status_health_and_lifespan(store):
    from fastapi.testclient import TestClient
    from server import create_app
    runtime = {"ready": False, "redis_connected": False}
    lifecycle = []

    @asynccontextmanager
    async def lifespan(app):
        lifecycle.append("started")
        yield
        lifecycle.append("stopped")

    with TestClient(create_app(lambda: dict(runtime), store, lifespan=lifespan)) as client:
        assert lifecycle == ["started"]
        assert client.get("/status").json() == runtime
        assert client.get("/healthz").status_code == 503
        runtime.update(ready=True, redis_connected=True)
        assert client.get("/status").json() == runtime
        health = client.get("/healthz")
        assert health.status_code == 200
        assert health.json() == runtime
    assert lifecycle == ["started", "stopped"]
