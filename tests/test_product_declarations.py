"""What each product declares for itself, and what the shared file must not decide.

The rule these tests encode (2026-09-23): the shared settings file holds
MECHANISM -- where the models live, driver-level knobs -- and each product
declares its own POLICY. A shared default that switches a detector ON costs
every future product GPU time and alerts it never asked for, and the only sign
is someone eventually adding `false` to their own file (barber had to).
"""
from pathlib import Path

import pytest
import yaml

from perception.engine.sources import load_sidecar_config

REPO = Path(__file__).resolve().parents[1]
SHARED = REPO / "perception" / "pipeline" / "sidecar.yaml"
PRODUCTS = REPO / "products"


def product_files():
    return sorted(p / "sidecar.yaml" for p in PRODUCTS.iterdir() if (p / "sidecar.yaml").is_file())


def test_the_shared_file_starts_no_module_a_product_did_not_ask_for():
    # A module is on exactly when a product lists it. Nothing is inherited, so a new product cannot end up
    # running a second detector and raising alerts nobody asked for (2026-09-23).
    assert yaml.safe_load(SHARED.read_text())["modules"] == {}


def test_a_product_declaring_only_a_camera_gets_no_modules(tmp_path):
    overlay = tmp_path / "sidecar.yaml"
    overlay.write_text(f"extends: {SHARED}\nsources: [{{id: x, uri: 'rtsp://x'}}]\n")
    cfg = load_sidecar_config(str(overlay))
    assert cfg["modules"] == {}


def test_attendance_declares_fire_with_the_values_it_was_tuned_with():
    own = yaml.safe_load((PRODUCTS / "attendance" / "sidecar.yaml").read_text())["modules"]["fire_alerts"]
    assert own["classes"] == ["fire", "smoke"]
    # Tuned on real footage: smoke is noisier, so it must persist longer than fire.
    assert own["smoke_sustain_seconds"] > own["fire_sustain_seconds"]
    assert 0 < own["min_confidence"] < 1
    assert own["interval"] == 0          # the detector runs before the tracker: a skipped frame reads as "no fire"


def test_barber_runs_no_fire():
    cfg = load_sidecar_config(str(PRODUCTS / "barber" / "sidecar.yaml"))
    assert "fire_alerts" not in (cfg.get("modules") or {})


@pytest.mark.parametrize("path", product_files(), ids=lambda p: p.parent.name)
def test_every_product_extends_the_shared_file_and_names_its_cameras(path):
    own = yaml.safe_load(path.read_text())
    assert "extends" in own, f"{path} must inherit the shared mechanism, not copy it"
    assert own.get("sources"), f"{path} must name the cameras this product owns"


def test_no_product_enables_overcrowding_without_the_analytics_element():
    """Overcrowding is evaluated on nvdsanalytics. Turning it on without that element
    can never alert, and the app warns about it at every start."""
    for path in product_files():
        cfg = load_sidecar_config(str(path))
        if cfg.get("overcrowding_enabled"):
            assert cfg.get("analytics_enabled"), f"{path.parent.name}: overcrowding_enabled needs analytics_enabled"
