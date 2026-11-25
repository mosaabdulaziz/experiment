"""The safety product's configuration and the names the app and the host controller share -- one copy, imported by
both (products/safety/service.py, tools/safety_controller.py), so the two can never disagree about a camera's features.

  safety.yaml          the features, the role each extends, the settings every role shares
  cameras.yaml         the cameras (SAFETY_CAMERAS), each with its default features
  cameras_state.json   what the UI saved, in <data>/safety/: {camera: {"name", "features"}}
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

# A camera id must not contain it: a role sidecar's stream id is <camera>@<role> (perception/engine/sources.py).
ROLE_KEY_SEPARATOR = "@"
# The UI's Apply bumps the first; the controller writes the version it applied to the second.
APPLY_VERSION_KEY = "iris:ctrl:safety_apply"
APPLY_APPLIED_KEY = "iris:analytics:safety_applied"
# While a role sidecar (re)starts for an Apply, camera health holds its "not running" incidents
# (products/safety/camera_health.py): the controller sets iris:safety:applying:<role> for at most APPLY_HOLD_S -- long
# enough for a first engine build at a new camera count -- and deletes it as soon as the role is live again.
APPLYING_PREFIX = "iris:safety:applying:"
APPLY_HOLD_S = 300


def load_product(path) -> dict:
    with open(path, encoding="utf-8") as f:
        product = yaml.safe_load(f) or {}
    missing = [feature for feature in product.get("features") or [] if feature not in (product.get("roles") or {})]
    if missing:
        raise ValueError(f"{path}: no role for feature(s) {missing}")
    return product


def load_cameras(path) -> list:
    with open(path, encoding="utf-8") as f:
        return (yaml.safe_load(f) or {}).get("cameras") or []


def load_state(path) -> dict:
    """The UI's saved names and toggles; none saved yet (or unreadable) is an empty state."""
    path = Path(path)
    try:
        return json.loads(path.read_text(encoding="utf-8")) or {} if path.is_file() else {}
    except ValueError:
        return {}


def effective_cameras(cameras: list, state: dict, product: dict) -> list:
    """Each camera with the UI's name and features applied over the file's: [{id, name, features, clip|rtsp}],
    features in the product's order. ValueError: a camera id with the separator, or a feature the product lacks."""
    known = list(product.get("features") or [])
    out = []
    for cam in cameras:
        cid = str(cam["id"])
        if ROLE_KEY_SEPARATOR in cid:
            raise ValueError(f"camera id {cid!r} cannot contain {ROLE_KEY_SEPARATOR!r}")
        saved = state.get(cid) or {}
        features = list(saved["features"] if "features" in saved else cam.get("features") or [])
        unknown = [f for f in features if f not in known]
        if unknown:
            raise ValueError(f"camera {cid}: unknown feature(s) {unknown}; known: {known}")
        merged = {k: v for k, v in cam.items() if k in ("clip", "rtsp")}
        merged.update(id=cid, name=str(saved.get("name") or cam.get("name") or cid),
                      features=[f for f in known if f in features])
        out.append(merged)
    return out
