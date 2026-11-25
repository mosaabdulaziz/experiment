"""The exclusivity preflight: `just fall-up` refuses to start beside a live sidecar holding a fall camera.

ACs: AC-F28.1.
Design: FALL_DETECTION_DESIGN.md §9 — the `roles` tag names the recording's owner but does NOT
exclude a second process (perception/engine/sources.py:52-74), so the enforcement is the preflight
in `just fall-up`, and it names the sidecar it found.
The pure check is unit-tested with an inventory; collecting the inventory from docker is the
recipe's job (asserted by the justfile scan, not by a live docker call).
"""
import re
from pathlib import Path

import pytest
import yaml

import preflight

REPO = Path(__file__).resolve().parents[3]
FALL_CAMERAS = ["cam-1"]


# ---------------------------------------------------------------- the pure check
def test_a_live_sidecar_holding_a_fall_camera_is_refused_and_named():
    live = [
        {"service": "deepstream-sidecar-attendance", "cameras": ["cam-1", "cam-2"]},
        {"service": "deepstream-sidecar-fire", "cameras": ["cam-9"]},
    ]
    with pytest.raises(preflight.PreflightConflict) as exc:
        preflight.check_cameras(FALL_CAMERAS, live)
    message = str(exc.value)
    assert "deepstream-sidecar-attendance" in message       # names the sidecar it found
    assert "cam-1" in message                              # names the contested camera
    assert "deepstream-sidecar-fire" not in message        # an unrelated sidecar is not accused


def test_disjoint_cameras_pass():
    live = [{"service": "deepstream-sidecar-attendance", "cameras": ["cam-5", "cam-6"]}]
    preflight.check_cameras(FALL_CAMERAS, live)            # no raise


def test_no_live_sidecars_passes():
    preflight.check_cameras(FALL_CAMERAS, [])


def test_every_contested_sidecar_is_named_not_just_the_first():
    live = [
        {"service": "deepstream-sidecar-attendance", "cameras": ["cam-1"]},
        {"service": "deepstream-sidecar-barber", "cameras": ["cam-1", "cam-3"]},
    ]
    with pytest.raises(preflight.PreflightConflict) as exc:
        preflight.check_cameras(["cam-1", "cam-3"], live)
    assert "deepstream-sidecar-attendance" in str(exc.value)
    assert "deepstream-sidecar-barber" in str(exc.value)


# ---------------------------------------------------------------- the inventory's rules
def test_the_overlay_extends_its_base_and_the_role_rule_is_the_engine_s(tmp_path):
    (tmp_path / "base.yaml").write_text(
        "sources:\n  - {id: cam-1, roles: [attendance]}\n  - {id: cam-9}\n", encoding="utf-8")
    overlay = tmp_path / "fall.yaml"
    overlay.write_text(
        "extends: base.yaml\n"
        "sources:\n"
        "  - {id: cam-1, roles: [fall_detection]}\n"
        "  - {id: cam-2, roles: [attendance]}\n"
        "  - {id: cam-3}\n", encoding="utf-8")
    cfg = preflight._read(overlay)
    # `sources` is a list, so the overlay REPLACES the base's -- exactly as the engine reads it
    assert preflight.cameras_for_role(cfg, "fall_detection") == ["cam-1", "cam-3"]
    assert preflight.cameras_for_role(cfg, "attendance") == ["cam-2", "cam-3"]
    assert preflight.cameras_for_role(cfg, None) == ["cam-1", "cam-2", "cam-3"]


def test_the_cli_refuses_and_names_the_conflict(monkeypatch, capsys):
    # the recipe's entry point; `--cameras` skips reading the product yaml
    monkeypatch.setattr(preflight, "live_sidecars",
                        lambda: [{"service": "deepstream-sidecar-attendance", "cameras": ["cam-1"]}])
    assert preflight.main(["--cameras", "cam-1"]) == 1
    err = capsys.readouterr().err
    assert "deepstream-sidecar-attendance" in err and "cam-1" in err


def test_a_sidecar_yaml_is_read_as_utf8_not_as_the_platform_default(tmp_path):
    # the sidecar yamls carry em dashes and box drawing; read with the platform's default
    # encoding they mojibake (or fail to decode) on Windows
    path = tmp_path / "sidecar.yaml"
    path.write_text("sources:\n  - {id: cam-1}\nlabel: \"\u2014\"\n", encoding="utf-8")
    assert preflight._read(path)["label"] == "\u2014"


def test_the_shipped_configs_name_the_same_camera():
    # the preflight guards the cameras the SERVICE config asks for, and the sidecar overlay it
    # runs must hold exactly those; if the two files drifted, the guard would protect a camera
    # the sidecar never takes, or miss one it does
    assert preflight.fall_cameras() == ["cam-1"]
    sidecar = preflight._read(Path(preflight.__file__).with_name("sidecar.yaml"))
    assert preflight.cameras_for_role(sidecar, "fall_detection") == ["cam-1"]


# ---------------------------------------------------------------- the recipe wiring
def _recipe(name):
    just = (REPO / "justfile").read_text(encoding="utf-8")
    match = re.search(rf"^{name}\b.*?(?=\n\n|\n[a-z_#]|\Z)", just, re.M | re.S)
    assert match, f"the justfile has no {name} recipe"
    # command lines only: a preflight mentioned in a comment enforces nothing
    lines = [line.strip() for line in match.group(0).splitlines()
             if line[:1].isspace() and not line.lstrip().startswith("#")]
    return "\n".join(lines)


def test_the_fall_up_recipe_runs_the_preflight_before_anything_starts():
    body = _recipe("fall-up")
    assert "preflight" in body
    assert "compose" in body
    # the guard runs before compose starts the sidecar
    assert body.index("preflight") < body.index("compose")


def test_the_fall_up_recipe_starts_the_service_before_the_sidecar():
    # the boot gate refuses in seconds (two sha256s, a CUDA op); a first sidecar start spends
    # minutes building the TensorRT engines. The fast gate goes first, so a bad deploy fails
    # before the expensive half and /status is already answering when the sidecar starts.
    body = _recipe("fall-up")
    assert "main.py" in body
    assert "deepstream-sidecar-fall" in body
    assert body.index("main.py") < body.index("deepstream-sidecar-fall")


def test_the_fall_up_recipe_waits_for_the_sidecar_and_fails_loudly():
    body = _recipe("fall-up")
    # a bounded wait: 100 attempts, 3 s apart, then a non-zero exit
    assert re.search(r"seq 1 100", body)
    assert re.search(r"sleep 3", body)
    assert "pipeline PLAYING" in body
    assert "exit 1" in body
    # and the failure names where to look, and how to recover
    assert "deepstream-sidecar-fall" in body
    assert "just fall-down" in body


def test_just_down_stops_what_fall_up_starts():
    body = _recipe("down")
    # the service's own pid file is in the loop `just down` walks
    assert re.search(r"for name in .*\bfall\b", body)
    # the orphan sweep is scoped to THIS checkout's product folder, so a sibling deployment's
    # identically-named process is not killed (the sweep is shared with concealment_detection: one alternation)
    assert re.search(r'\$repo_root/products/\((?:\w+\|)*fall_detection(?:\|\w+)*\)', body)
    # and the sidecar's compose profile goes down with it
    assert "--profile fall" in body


# ---------------------------------------------------------------- the compose service
def test_the_compose_service_carries_its_own_profile_port_role_and_config():
    compose = yaml.safe_load((REPO / "docker-compose.yml").read_text(encoding="utf-8"))
    svc = compose["services"]["deepstream-sidecar-fall"]
    assert svc["profiles"] == ["fall"]
    assert svc["ports"] == ["${SIDECAR_FALL_TELEMETRY_PORT:-9105}:9100"]
    assert "IRIS_SIDECAR_ROLE=fall_detection" in svc["environment"]
    # the sidecar runs the product's OWN overlay, and nothing else
    assert svc["command"] == ["python3", "-m", "perception.engine.main",
                              "${FALL_SIDECAR_CONFIG:-products/fall_detection/sidecar.yaml}"]


def test_the_compose_service_writes_no_core_files():
    # nvstreammux segfaults wrote 12 GB cores into the mounted repo root; a service's ulimits
    # replaces the anchor's whole map, so the anchor's nofile has to be restated too
    compose = yaml.safe_load((REPO / "docker-compose.yml").read_text(encoding="utf-8"))
    ulimits = compose["services"]["deepstream-sidecar-fall"]["ulimits"]
    assert ulimits["core"] == {"soft": 0, "hard": 0}
    assert ulimits["nofile"] == {"soft": 65536, "hard": 65536}


def test_a_role_scoped_sidecar_on_the_same_camera_holds_other_keys(tmp_path):
    """Beside other role sidecars (the safety product), a sidecar with role_scoped_keys writes cam-1@<role>: it does
    not collide with the fall sidecar's cam-1, so it is not a conflict; a scoped one in the SAME role still is."""
    cfg = {"role_scoped_keys": True, "sources": [{"id": "cam-1", "roles": ["ppe", "fall"]}]}
    assert preflight.keys_for_role(cfg, "ppe") == ["cam-1@ppe"]
    preflight.check_cameras(["cam-1"], [{"service": "safety-ppe", "cameras": preflight.keys_for_role(cfg, "ppe")}])
    with pytest.raises(preflight.PreflightConflict):
        preflight.check_cameras(["cam-1@fall"], [{"service": "safety-fall",
                                                  "cameras": preflight.keys_for_role(cfg, "fall")}])


def test_an_unscoped_sidecar_s_keys_are_its_camera_ids():
    cfg = {"sources": [{"id": "cam-1", "roles": ["fall_detection"]}, {"id": "cam-2", "roles": ["attendance"]}]}
    assert preflight.keys_for_role(cfg, "fall_detection") == ["cam-1"]


def test_the_fall_product_s_keys_follow_its_key_role(tmp_path):
    site = tmp_path / "site.yaml"
    site.write_text("cameras: [{id: cam-1}, {id: cam-4}]\nkey_role: fall\n")
    assert preflight.fall_cameras(site) == ["cam-1@fall", "cam-4@fall"]
