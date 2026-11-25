"""The multi-camera tracking use case must not change what the other products run.

The rule (MULTI_PRODUCT_PLAN.md decisions 1 and 4): the shared files stay neutral and each
product declares its own policy. Multi-camera tracking brings an MTMC tracker, a VLM arbiter,
a plate detector, box stabilization, a tracking service and its API; none of that may be on
in the shared pipeline file, in attendance's or barber's, or in the shared app config. These
tests fail the moment a default drifts, which is how a use case "silently" starts costing the
others GPU time or opening routes they never asked for.
"""
import re
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from app.server import create_app
from perception.engine.sources import load_sidecar_config

REPO = Path(__file__).resolve().parents[1]

# Everything that is NOT multi-camera tracking.
OTHER_PRODUCTS = [
    "perception/pipeline/sidecar.yaml",
    "products/attendance/sidecar.yaml",
    "products/barber/sidecar.yaml",
    "tests/configs/attendance-clip.yaml",
]


def _resolved(rel):
    return load_sidecar_config(str(REPO / rel))


@pytest.mark.parametrize("rel", OTHER_PRODUCTS)
def test_other_products_run_none_of_the_multicam_pipeline(rel):
    cfg = _resolved(rel)
    assert not {"mtmc", "plates"} & set(cfg.get("modules") or {})
    assert "box_stabilizer" not in (cfg.get("modules") or {})
    # the publisher keeps today's encoder count, and the opt-in behaviours stay opt-in
    assert int(cfg.get("jpeg_encoders", 1)) == 1
    assert not cfg.get("clock_drift_tracking", False)
    assert float(cfg.get("preview_label_scale", 0.5)) == 0.5
    assert not cfg.get("flight_native_dump", False)


def test_the_multicam_overlay_declares_its_own_policy():
    cfg = _resolved("products/multicam/sidecar.yaml")
    modules = cfg["modules"]
    assert modules["mtmc"]["vlm_arbiter"]["enabled"] is True
    assert "plates" in modules and modules["mtmc"]["plates"]["ocr_model"]
    assert modules["box_stabilizer"]["labels"]
    # the order is the policy: stabilize, then filter, then match plates, then follow vehicles
    assert list(modules) == ["box_stabilizer", "zones", "plates", "mtmc"]
    assert cfg["source_stall_seconds"] == 60
    # the shared file's 500 ms batch window must not leak into this product
    assert not cfg.get("mux_push_timeout_us")
    # and it does not start the fire or face detectors the shared file leaves off
    assert not cfg.get("pgie_fire_enabled", False)


def test_the_shared_app_config_starts_none_of_it():
    cfg = yaml.safe_load((REPO / "config.yaml").read_text())
    assert cfg["mtmc"]["enabled"] is False
    assert cfg["truck_feed"]["enabled"] is False
    assert cfg["hazards"]["enabled"] is False


@pytest.mark.parametrize("preset,expected", [("multicam", True), ("scale13", True), ("vlm", False)])
def test_presets_turn_the_tracking_service_on_only_where_it_belongs(preset, expected):
    cfg = yaml.safe_load((REPO / "configs" / "presets" / preset / "config.yaml").read_text())
    assert bool((cfg.get("mtmc") or {}).get("enabled", False)) is expected


# ── the app: routes exist only when the use case is switched on ────────────────

def _cfg(tmp_path, **extra):
    cfg = {
        "perception": {"backend": "local"},
        "api": {"backend": "cloud", "api_key": "x", "model": "m"},
        "monitoring": {"mode": "proactive"}, "sampling": {},
        "events": {"db_path": str(tmp_path / "e.db")},
        "chat": {"tools_enabled": False}, "cv": {"enabled": False}, "camera": {"source": 0},
    }
    cfg.update(extra)
    return cfg


def _paths(app):
    """Every served path, with {param} names flattened (clients never see them).

    Read from the OpenAPI schema: this FastAPI wraps an included router in a lazy object,
    so walking app.routes would not see the routes of an included router at all."""
    return {re.sub(r"\{[^}]*\}", "{}", p) for p in app.openapi()["paths"]}


MULTICAM_ROUTES = [
    "/api/tracking/tracks", "/api/tracking/stats", "/api/tracking/cameras", "/api/tracking/config",
    "/api/vehicles/{}/sightings", "/api/count/{}", "/api/hazards/status",
]


def test_an_app_without_the_use_case_exposes_none_of_its_routes(tmp_path):
    app = create_app(_cfg(tmp_path))
    paths = _paths(app)
    assert not [p for p in MULTICAM_ROUTES if p in paths]
    assert not any(p.startswith(("/api/tracking", "/api/vehicles", "/api/count", "/api/hazards")) for p in paths)


def test_an_app_with_the_use_case_serves_the_routes_truck_tracking_calls(tmp_path):
    """The truck-tracking UI's contract with the camera system (kept byte for byte)."""
    app = create_app(_cfg(tmp_path, mtmc={"enabled": True}, hazards={"enabled": True},
                          truck_feed={"enabled": False}))
    paths = _paths(app)
    contract = [
        "/stream/{}/frame",
        "/api/vehicles/{}/sightings",
        "/api/vehicles/crops/{}/{}/image",
        "/api/vehicles/plates/{}/readings",
        "/api/vehicles/plates/{}/{}/image",
        "/api/tracking/config",
        "/api/streams/{}/restricted-zones",
        "/api/streams/{}/filter-zones",
        "/api/hazards/analyze", "/api/hazards/status", "/api/hazards/history",
    ]
    missing = [p for p in contract if p not in paths]
    assert not missing, f"truck-tracking calls these and they are gone: {missing}"


def test_the_tracking_service_is_not_even_built_when_off(tmp_path):
    """No handler answers. Without a built frontend that is a 404; with one (a checkout that has `static/`)
    the single-page-app fallback answers with the HTML shell instead -- never with JSON from a tracking route."""
    with TestClient(create_app(_cfg(tmp_path))) as client:
        for path in ("/api/tracking/stats", "/api/hazards/status"):
            r = client.get(path)
            assert r.status_code == 404 or r.headers["content-type"].startswith("text/html"), path
