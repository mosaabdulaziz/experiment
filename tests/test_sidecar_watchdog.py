"""The sidecar watchdog must watch the cameras the pipeline actually runs
(2026-09-24). It read the shared base config, which lists no cameras since
the restructure, so it watched nothing and a wedged pipeline sat frozen for
an hour in production. Nothing failed -- only a test like this notices."""
import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load_watchdog():
    spec = importlib.util.spec_from_file_location("sidecar_watchdog", REPO / "tools" / "sidecar_watchdog.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_it_reads_the_config_the_pipeline_runs(monkeypatch):
    wd = _load_watchdog()
    monkeypatch.setenv("SIDECAR_CONFIG", "products/attendance/sidecar.yaml")
    assert wd.sidecar_config_path() == REPO / "products" / "attendance" / "sidecar.yaml"


def test_attendance_config_gives_it_the_three_live_cameras(monkeypatch):
    wd = _load_watchdog()
    monkeypatch.setenv("SIDECAR_CONFIG", "products/attendance/sidecar.yaml")
    assert wd.enabled_cameras(wd.sidecar_config_path()) == {
        "cam-1": "raw-cam-1", "cam-2": "raw-cam-2", "cam-3": "raw-cam-3"}


def test_without_sidecar_config_it_falls_back_to_the_base_file(monkeypatch):
    wd = _load_watchdog()
    monkeypatch.delenv("SIDECAR_CONFIG", raising=False)
    path = wd.sidecar_config_path()
    assert path == REPO / "perception" / "pipeline" / "sidecar.yaml"
    assert wd.enabled_cameras(path) == {}   # the base declares no cameras -- the old bug
