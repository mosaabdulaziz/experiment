import os

import yaml

from app.server import (
    _load_zones, _save_zones, mux_resolution,
    _zones_file_path,
)


def test_zones_file_path_defaults_when_config_has_no_zones_section():
    assert _zones_file_path({}) == "data/zones_config.txt"


def test_zones_file_path_reads_config_override():
    config = {"zones": {"file_path": "/tmp/custom_zones.txt"}}
    assert _zones_file_path(config) == "/tmp/custom_zones.txt"


def test_mux_resolution_reads_sidecar_yaml(tmp_path):
    sidecar_yaml = tmp_path / "sidecar.yaml"
    sidecar_yaml.write_text("mux_width: 1920\nmux_height: 1080\n")
    config = {"zones": {"sidecar_config_path": str(sidecar_yaml)}}
    assert mux_resolution(config) == (1920, 1080)


def test_mux_resolution_falls_back_to_1280x720_when_file_missing(tmp_path):
    config = {"zones": {"sidecar_config_path": str(tmp_path / "does-not-exist.yaml")}}
    assert mux_resolution(config) == (1280, 720)


def test_mux_resolution_falls_back_when_config_has_no_zones_section(tmp_path, monkeypatch):
    # No "zones" key at all -> running_sidecar_config's default path
    # ("perception/pipeline/sidecar.yaml") is relative, so it resolves against cwd. Chdir
    # into an empty tmp_path so that default path genuinely doesn't exist and
    # this exercises the hardcoded fallback -- unpatched, this ran from the
    # repo root and silently read the REAL perception/pipeline/sidecar.yaml instead
    # (whose mux_width/height are tuned for this deployment's cameras, not a
    # stable constant), coupling this test to that file's current values.
    monkeypatch.chdir(tmp_path)
    assert mux_resolution({}) == (1280, 720)


def test_load_zones_returns_empty_dict_when_file_missing(tmp_path):
    path = str(tmp_path / "zones_config.txt")
    assert _load_zones(path) == {}


def test_save_then_load_round_trips(tmp_path):
    path = str(tmp_path / "zones_config.txt")
    zones = {"cam0": {"points": [[10, 20], [30, 20], [30, 40]]}}
    _save_zones(path, zones)
    assert _load_zones(path) == zones


def test_save_zones_leaves_no_tmp_file_behind(tmp_path):
    path = str(tmp_path / "zones_config.txt")
    _save_zones(path, {"cam0": {"points": [[1, 2], [3, 4], [5, 6]]}})
    assert os.path.isfile(path)
    assert not os.path.isfile(path + ".tmp")


def test_save_zones_overwrites_existing_file(tmp_path):
    path = str(tmp_path / "zones_config.txt")
    _save_zones(path, {"cam0": {"points": [[1, 2], [3, 4], [5, 6]]}})
    _save_zones(path, {"cam0": {"points": [[7, 8], [9, 10], [11, 12]]}})
    assert _load_zones(path) == {"cam0": {"points": [[7, 8], [9, 10], [11, 12]]}}


