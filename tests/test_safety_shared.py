"""The facts the safety product states in more than one place agree (MULTI_PRODUCT_PLAN.md decision 5): the
feature list, and the Redis keys the app and the host controller talk through (one copy: products/safety/config.py)."""
import importlib.util
import pathlib
import sys

import yaml

from products.safety import service as app_safety
from app.features import FEATURES
from products.safety import camera_health
from products.safety import config as shared

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _controller():
    spec = importlib.util.spec_from_file_location("safety_controller", ROOT / "tools" / "safety_controller.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["safety_controller"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_the_features_are_the_product_s_plus_camera_health():
    product = yaml.safe_load((ROOT / "products/safety/safety.yaml").read_text())
    assert list(FEATURES) == product["features"] + ["health"]
    assert list(app_safety.DEFAULT_FEATURES) == list(FEATURES)


def test_the_app_and_the_controller_use_the_one_copy_of_the_keys():
    ctl = _controller()
    for name in ("APPLY_VERSION_KEY", "APPLY_APPLIED_KEY", "APPLYING_PREFIX", "APPLY_HOLD_S"):
        assert getattr(ctl, name) is getattr(shared, name), name
    assert app_safety.APPLY_VERSION_KEY is shared.APPLY_VERSION_KEY
    assert app_safety.APPLY_APPLIED_KEY is shared.APPLY_APPLIED_KEY
    assert camera_health.APPLYING_PREFIX is shared.APPLYING_PREFIX


def test_the_app_and_the_controller_merge_cameras_with_the_one_function():
    assert _controller().effective_cameras is shared.effective_cameras
    assert app_safety.effective_cameras is shared.effective_cameras
