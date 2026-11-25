"""Which camera and which safety feature a stream or an event belongs to.

Beside several role sidecars (the safety product, role-scoped keys) a stream id is `<camera>@<role>` and the role IS
the feature. Elsewhere a stream id is the camera id, and the feature is read off the event's tag.
"""
from __future__ import annotations

from perception.engine.sources import split_stream_key_id

# The safety product's features, in its UI's order (products/safety/safety.yaml), and camera health.
FEATURES = ("proximity", "zones", "ppe", "fall", "fire", "health")

# The safety features whose zones are plain on/off blocks on the shared zones file (proximity's carries its own
# settings; intrusion and overcrowding are the zone's own sections).
ZONE_FEATURES = ("ppe", "fall", "fire")

# The equipment the PPE module checks unless its config says otherwise (perception/modules/ppe Settings.required_items):
# a zone's `ppe` block may narrow it to some of them ({"enabled": true, "items": ["helmet"]}).
PPE_DEFAULT_ITEMS = ("helmet", "vest")


def ppe_items(sidecar_cfg: dict) -> list:
    """The equipment a zone can ask PPE to check: the module's required_items as the running config states them."""
    settings = (sidecar_cfg.get("modules") or {}).get("ppe")
    items = settings.get("required_items") if isinstance(settings, dict) else None
    return list(items) if isinstance(items, list) and items else list(PPE_DEFAULT_ITEMS)

_TAG_FEATURE = {
    "zone_intrusion": "zones", "zone_exit": "zones", "zone_overcrowding": "zones",
    "ppe_violation": "ppe", "fall": "fall", "fire_smoke": "fire", "camera_health": "health",
}


def camera_of(stream_id: "str | None") -> "str | None":
    """The camera a stream id names: `bay-a@ppe` -> `bay-a`; an unscoped id is the camera."""
    return split_stream_key_id(stream_id)[0] if stream_id else stream_id


def feature_of(stream_id: "str | None", tags) -> "str | None":
    """The feature of an event on `stream_id` with these tags: the stream's role when it has one, else the tag's."""
    if stream_id:
        role = split_stream_key_id(stream_id)[1]
        if role in FEATURES:
            return role
    for tag in tags or ():
        if tag in _TAG_FEATURE:
            return _TAG_FEATURE[tag]
        if isinstance(tag, str) and tag.startswith("proximity"):
            return "proximity"
    return None


def features_where(features) -> "tuple[str, list]":
    """A SQL condition on the events table for "belongs to one of these features": a role sidecar's stream
    (`<camera>@<feature>`), or a tag of the feature (plus proximity's tag prefix). ("0", []) for none."""
    clauses, params = [], []
    for feature in features:
        clauses.append("camera LIKE ?")
        params.append(f"%@{feature}")
        for tag in [t for t, f in _TAG_FEATURE.items() if f == feature]:
            clauses.append("tags LIKE ?")
            params.append(f'%"{tag}"%')
        if feature == "proximity":
            clauses.append("tags LIKE ?")
            params.append('%"proximity%')
    return ("(" + " OR ".join(clauses) + ")", params) if clauses else ("0", [])
