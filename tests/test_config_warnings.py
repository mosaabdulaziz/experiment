"""Startup warnings for config.yaml / sidecar.yaml key pairs that fail silently.

The two files are read by different processes, so a mismatch produces no error --
just clips that never build. create_app() must say so out loud.
"""
import logging

import yaml

from app.server import create_app


def _cfg(tmp_path, *, sidecar=None, segments=None, clip=None, backend="deepstream",
         sidecar_written=True, monitoring=None):
    sidecar_yaml = tmp_path / "sidecar.yaml"
    if sidecar_written:
        base = {"mux_width": 1000, "mux_height": 500,
                "sources": [{"id": "cam0", "uri": "file:///a.mp4"}],
                "record_enabled": True, "record_seconds": 8, "storage_backend": "local"}
        base.update(sidecar or {})
        sidecar_yaml.write_text(yaml.safe_dump(base))

    seg = {"enabled": True, "storage_backend": "local",
           "local_dir": str(tmp_path / "seg")}
    seg.update(segments or {})
    cl = {"enabled": False, "pad_seconds": 5.0, "max_wait_seconds": 20.0}
    cl.update(clip or {})

    return {
        "perception": {"backend": backend, "redis_url": "redis://localhost:6379/0"},
        "api": {"backend": "cloud", "api_key": "x", "model": "m"},
        "monitoring": {"mode": "proactive", **(monitoring or {})}, "sampling": {},
        "events": {"db_path": str(tmp_path / "e.db")},
        "chat": {"tools_enabled": False}, "cv": {"enabled": False}, "camera": {"source": 0},
        "segments": seg, "alerts": {"clip": cl},
        "zones": {"file_path": str(tmp_path / "z.txt"),
                  "sidecar_config_path": str(sidecar_yaml)},
    }


def _warnings(config, caplog):
    with caplog.at_level(logging.WARNING, logger="app.server"):
        create_app(config)
    return [r.getMessage() for r in caplog.records]


def _matching(msgs, needle):
    return [m for m in msgs if needle in m]


def test_consistent_config_warns_about_nothing(tmp_path, caplog):
    msgs = _warnings(_cfg(tmp_path, clip={"enabled": True}), caplog)
    assert not _matching(msgs, "record_enabled")
    assert not _matching(msgs, "storage_backend")
    assert not _matching(msgs, "max_wait_seconds")


def test_recorder_off_warns_that_nothing_is_ever_written(tmp_path, caplog):
    msgs = _warnings(_cfg(tmp_path, sidecar={"record_enabled": False}), caplog)
    hits = _matching(msgs, "record_enabled is false in sidecar.yaml")
    assert len(hits) == 1
    # Clips are off here, so the alert-specific consequence must not be claimed.
    assert "resolving 'unavailable'" not in hits[0]


def test_recorder_off_with_clips_on_also_names_the_wasted_wait(tmp_path, caplog):
    msgs = _warnings(_cfg(tmp_path, sidecar={"record_enabled": False},
                          clip={"enabled": True, "max_wait_seconds": 20.0}), caplog)
    hits = _matching(msgs, "record_enabled is false in sidecar.yaml")
    assert len(hits) == 1
    assert "20s" in hits[0] and "resolving 'unavailable'" in hits[0]


def test_storage_backend_mismatch_is_reported(tmp_path, caplog):
    msgs = _warnings(_cfg(tmp_path, sidecar={"storage_backend": "minio"},
                          segments={"storage_backend": "local"}), caplog)
    hits = _matching(msgs, "sidecar.yaml's storage_backend")
    assert len(hits) == 1
    assert "'local'" in hits[0] and "'minio'" in hits[0]


def test_storage_backend_mismatch_is_not_reported_when_nothing_records(tmp_path, caplog):
    """The recorder-off warning already names the root cause; a backend
    mismatch on top of it is noise, not a second problem."""
    msgs = _warnings(_cfg(tmp_path, sidecar={"record_enabled": False,
                                             "storage_backend": "minio"},
                          segments={"storage_backend": "local"}), caplog)
    assert not _matching(msgs, "sidecar.yaml's storage_backend")


def test_clips_enabled_without_segments_warns(tmp_path, caplog):
    msgs = _warnings(_cfg(tmp_path, segments={"enabled": False},
                          clip={"enabled": True}), caplog)
    assert len(_matching(msgs, "segments.enabled is false -- alert clips")) == 1


def test_max_wait_not_outlasting_pad_plus_record_seconds_warns(tmp_path, caplog):
    # pad 5 + record 8 = 13; a 13s wait can expire exactly as the segment closes.
    msgs = _warnings(_cfg(tmp_path, sidecar={"record_seconds": 8},
                          clip={"enabled": True, "pad_seconds": 5.0,
                                "max_wait_seconds": 13.0}), caplog)
    hits = _matching(msgs, "max_wait_seconds")
    assert len(hits) == 1
    assert "5 + 8 = 13" in hits[0]


def test_max_wait_check_accounts_for_a_longer_sidecar_segment(tmp_path, caplog):
    """20s is ample at record_seconds 8 and hopeless at 30 -- the check has to
    read the sidecar's value, not assume the default."""
    ok = _warnings(_cfg(tmp_path, sidecar={"record_seconds": 8},
                        clip={"enabled": True, "max_wait_seconds": 20.0}), caplog)
    assert not _matching(ok, "max_wait_seconds")
    caplog.clear()
    bad = _warnings(_cfg(tmp_path, sidecar={"record_seconds": 30},
                         clip={"enabled": True, "max_wait_seconds": 20.0}), caplog)
    assert len(_matching(bad, "max_wait_seconds")) == 1


def test_unreadable_sidecar_yaml_invents_no_mismatch(tmp_path, caplog):
    """An unreadable sidecar.yaml loads as {}; comparing config.yaml against its
    defaults would manufacture a mismatch that does not exist."""
    msgs = _warnings(_cfg(tmp_path, sidecar_written=False,
                          segments={"storage_backend": "local"},
                          clip={"enabled": True, "max_wait_seconds": 1.0}), caplog)
    assert not _matching(msgs, "record_enabled")
    assert not _matching(msgs, "sidecar.yaml's storage_backend")
    assert not _matching(msgs, "max_wait_seconds")


def test_local_backend_has_no_sidecar_to_disagree_with(tmp_path, caplog):
    msgs = _warnings(_cfg(tmp_path, backend="local", segments={"storage_backend": "minio"},
                          clip={"enabled": True, "max_wait_seconds": 1.0}), caplog)
    assert not _matching(msgs, "record_enabled")
    assert not _matching(msgs, "sidecar.yaml's storage_backend")
    assert not _matching(msgs, "max_wait_seconds")
    assert not _matching(msgs, "alert clips")


# ── Mixed recorder fleets ─────────────────────────────────────────────────────
# RTSP sources record via smart-record, which writes no segments. "recording is on"
# stopped being one boolean, so the checks must know which sources use which.

def test_all_svr_fleet_does_not_warn_about_record_enabled(tmp_path, caplog):
    """An all-RTSP deployment records via smart-record, so record_enabled: false is
    correct for it and must not warn."""
    msgs = _warnings(_cfg(tmp_path,
                          sidecar={"record_enabled": False,
                                   "sources": [{"id": "cam0", "uri": "rtsp://a/s"}]},
                          clip={"enabled": True}), caplog)
    assert not _matching(msgs, "record_enabled is false")


def test_file_source_with_recording_off_still_warns(tmp_path, caplog):
    msgs = _warnings(_cfg(tmp_path,
                          sidecar={"record_enabled": False,
                                   "sources": [{"id": "cam0", "uri": "file:///a.mp4"}]},
                          clip={"enabled": True}), caplog)
    hits = _matching(msgs, "record_enabled is false")
    assert hits and "cam0" in hits[0]


def test_mixed_fleet_names_only_the_encode_sources(tmp_path, caplog):
    msgs = _warnings(_cfg(tmp_path,
                          sidecar={"record_enabled": False,
                                   "sources": [{"id": "cam0", "uri": "rtsp://a/s"},
                                               {"id": "cam1", "uri": "file:///b.mp4"}]},
                          clip={"enabled": True}), caplog)
    hits = _matching(msgs, "record_enabled is false")
    assert hits and "cam1" in hits[0] and "cam0" not in hits[0]


def test_svr_cache_below_pad_seconds_warns(tmp_path, caplog):
    """start-sr(startTime=pad) cannot reach further back than the cache holds, so a
    cache shorter than pad_seconds silently truncates every clip's pre-roll."""
    msgs = _warnings(_cfg(tmp_path,
                          sidecar={"sources": [{"id": "cam0", "uri": "rtsp://a/s"}],
                                   "svr_cache_seconds": 3},
                          clip={"enabled": True, "pad_seconds": 5.0}), caplog)
    assert _matching(msgs, "svr_cache_seconds")


def test_adequate_svr_cache_does_not_warn(tmp_path, caplog):
    msgs = _warnings(_cfg(tmp_path,
                          sidecar={"sources": [{"id": "cam0", "uri": "rtsp://a/s"}],
                                   "svr_cache_seconds": 15},
                          clip={"enabled": True, "pad_seconds": 5.0}), caplog)
    assert not _matching(msgs, "svr_cache_seconds")


def test_svr_cache_is_not_checked_when_alert_clips_are_off(tmp_path, caplog):
    msgs = _warnings(_cfg(tmp_path,
                          sidecar={"sources": [{"id": "cam0", "uri": "rtsp://a/s"}],
                                   "svr_cache_seconds": 1},
                          clip={"enabled": False}), caplog)
    assert not _matching(msgs, "svr_cache_seconds")


def test_warns_when_vlm_on_but_pipeline_preview_off(tmp_path, caplog):
    """pipeline_preview: false means no JPEG reaches iris:meta -- the VLM starves."""
    msgs = _warnings(_cfg(tmp_path, monitoring={"vlm_enabled": True},
                          sidecar={"pipeline_preview": False}), caplog)
    assert _matching(msgs, "will receive no frames")


def test_warns_when_vlm_on_and_preview_is_viewer_gated(tmp_path, caplog):
    """An unwatched stream encodes no JPEG, so analysis goes blind for it."""
    msgs = _warnings(_cfg(tmp_path, monitoring={"vlm_enabled": True},
                          sidecar={"pipeline_preview": True,
                                   "viewer_gated_preview": True}), caplog)
    assert _matching(msgs, "will receive no frames")


def test_no_frame_supply_warning_when_preview_is_on_and_ungated(tmp_path, caplog):
    msgs = _warnings(_cfg(tmp_path, monitoring={"vlm_enabled": True},
                          sidecar={"pipeline_preview": True}), caplog)
    assert not _matching(msgs, "will receive no frames")


def test_no_frame_supply_warning_when_vlm_is_off(tmp_path, caplog):
    """Nothing consumes the frames, so an absent preview is not a problem."""
    msgs = _warnings(_cfg(tmp_path, monitoring={"vlm_enabled": False},
                          sidecar={"pipeline_preview": False}), caplog)
    assert not _matching(msgs, "will receive no frames")


def test_no_frame_supply_warning_when_sidecar_yaml_is_unreadable(tmp_path, caplog):
    """Comparing against sidecar defaults would invent a mismatch."""
    msgs = _warnings(_cfg(tmp_path, monitoring={"vlm_enabled": True},
                          sidecar_written=False), caplog)
    assert not _matching(msgs, "will receive no frames")


def test_app_only_imports_engine_modules_the_image_actually_ships():
    """The iris image copies a hand-listed subset of perception/engine/ (see the COPY in
    Dockerfile). An app-side import of any module outside that list passes every
    test here and then crash-loops the container in Docker -- which is exactly what
    adding `from perception.engine.recorder_kind import ...` did. Keep the two in step."""
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parent.parent
    dockerfile = (root / "Dockerfile").read_text()

    shipped = set()
    for line in dockerfile.splitlines():
        if line.startswith("COPY perception/engine/") and line.rstrip().endswith("./perception/engine/"):
            shipped.update(re.findall(r"perception/engine/(\w+)\.py", line))
    assert shipped, "could not parse the engine COPY line out of the Dockerfile"

    imported = set()
    for py in (root / "app").rglob("*.py"):
        imported.update(re.findall(r"from perception\.engine\.(\w+) import", py.read_text()))
    assert imported, "expected app/ to import at least one perception.engine module"

    missing = imported - shipped
    assert not missing, (
        f"app/ imports engine modules the iris image does not ship: {sorted(missing)}. "
        f"Add them to the COPY perception/engine/... line in Dockerfile, or the container will "
        f"fail at startup with ModuleNotFoundError.")
