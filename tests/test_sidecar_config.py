"""app.sidecar_config: the one place the app reads the pipeline's settings
(MR !25 review, 2026-10-01). It replaced a three-layer chain whose middle
layer silently dropped `extends:` and swallowed every error."""
import logging

import pytest

from app import sidecar_config
from app.sidecar_config import DEFAULT_PATH, running_sidecar_config, running_sidecar_path


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("SIDECAR_CONFIG", raising=False)
    sidecar_config._warned.clear()


def test_sidecar_config_env_wins_over_config_yaml(monkeypatch, tmp_path):
    monkeypatch.setenv("SIDECAR_CONFIG", "products/attendance/sidecar.yaml")
    cfg = {"zones": {"sidecar_config_path": str(tmp_path / "other.yaml")}}
    assert running_sidecar_path(cfg) == "products/attendance/sidecar.yaml"


def test_falls_back_to_config_yaml_then_the_base_file(tmp_path):
    assert running_sidecar_path({"zones": {"sidecar_config_path": "x.yaml"}}) == "x.yaml"
    assert running_sidecar_path({}) == DEFAULT_PATH
    assert running_sidecar_path({"zones": None}) == DEFAULT_PATH


def test_extends_is_followed(tmp_path):
    (tmp_path / "base.yaml").write_text("mux_width: 1280\nanalytics_enabled: false\n")
    (tmp_path / "product.yaml").write_text("extends: base.yaml\nanalytics_enabled: true\n")
    got = running_sidecar_config({"zones": {"sidecar_config_path": str(tmp_path / "product.yaml")}})
    assert got["mux_width"] == 1280            # from the base
    assert got["analytics_enabled"] is True    # the product's override


def test_a_missing_file_is_empty_and_logged_once(tmp_path, caplog):
    cfg = {"zones": {"sidecar_config_path": str(tmp_path / "missing.yaml")}}
    with caplog.at_level(logging.WARNING, logger="app.sidecar_config"):
        assert running_sidecar_config(cfg) == {}
        assert running_sidecar_config(cfg) == {}   # read on every request
    warnings = [r for r in caplog.records if "could not be read" in r.getMessage()]
    assert len(warnings) == 1


def test_an_extends_loop_is_empty_not_a_crash(tmp_path, caplog):
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n")
    with caplog.at_level(logging.WARNING, logger="app.sidecar_config"):
        assert running_sidecar_config({"zones": {"sidecar_config_path": str(tmp_path / "a.yaml")}}) == {}
    assert any("cycle" in r.getMessage() for r in caplog.records)
