"""The port the deployment binds: FALL_BACKEND_PORT over the yaml, the repo's product pattern.

The recipe (`just fall-up`) and the dashboard's dev server (frontend/vite.config.ts) both take the
port from that variable; a service that ignored it would come up on 8004 while the proxy pointed
somewhere else, and nothing would answer.
"""
from pathlib import Path

import config

_MINIMAL = """
scoring:
  confirm_threshold: 0.70
  margin_span: 0.80
  trust: {action: 1.0, kinematic: 0.5, cold: 0.4}
  evaluations: {falling: [6], lying: [3], immobile: [3, 2], unstable: [7], upright: [0, 1, 2, 4, 5]}
classifier: {device: "cuda:0", weights_dir: "weights"}
cameras: [{id: cam-1}]
"""


def _config(tmp_path: Path, text: str = _MINIMAL) -> Path:
    path = tmp_path / "fall.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_the_env_names_the_port_and_the_yaml_is_the_default(tmp_path):
    path = _config(tmp_path, _MINIMAL + "port: 8104\n")
    assert config.load_config(path, env={}).port == 8104
    assert config.load_config(path, env={"FALL_BACKEND_PORT": "9001"}).port == 9001


def test_the_shipped_yaml_binds_the_port_the_recipe_and_the_proxy_assume():
    assert config.load_config(config.CONFIG_PATH, env={}).port == 8004


def test_a_yaml_is_read_as_utf8_not_as_the_platform_default(tmp_path):
    # the repo's yaml files carry em dashes and box drawing; read with the platform's default
    # encoding they mojibake (or fail to decode) on Windows, and the config's own values then
    # differ from the file on disk
    path = _config(tmp_path, _MINIMAL + 'label: "\u2014"\n')
    assert config._read(path)["label"] == "\u2014"


def test_the_alert_floor_is_optional_and_a_confidence_below_one(tmp_path):
    assert "alert_above" not in config.load_config(_config(tmp_path), env={}).scoring
    with_floor = _MINIMAL.replace("  margin_span: 0.80\n", "  margin_span: 0.80\n  alert_above: 0.0\n")
    assert config.load_config(_config(tmp_path, with_floor), env={}).scoring["alert_above"] == 0.0
    for bad in ("-0.1", "1.0"):
        try:
            config.load_config(_config(tmp_path, with_floor.replace("alert_above: 0.0", f"alert_above: {bad}")),
                               env={})
        except ValueError as exc:
            assert "alert_above" in str(exc)
        else:
            raise AssertionError(f"alert_above {bad} was accepted")


def test_the_safety_stack_alerts_only_above_zero_confidence():
    safety = Path(__file__).resolve().parents[2] / "safety" / "fall_detection.yaml"
    assert config.load_config(safety, env={}).scoring["alert_above"] == 0.0


def test_the_detailed_alert_is_off_by_default_and_the_safety_stack_s(tmp_path):
    assert config.load_config(_config(tmp_path), env={}).alert_details is False
    safety = Path(__file__).resolve().parents[2] / "safety" / "fall_detection.yaml"
    assert config.load_config(safety, env={}).alert_details is True
    try:
        config.load_config(_config(tmp_path, _MINIMAL + "alert_details: yes please\n"), env={})
    except ValueError as exc:
        assert "alert_details" in str(exc)
    else:
        raise AssertionError("alert_details must be a boolean")
