"""The fall product's own configuration: fall_detection.yaml.

The loader's shape is the repo's product pattern -- SERVICE_DIR computed from __file__, the
`extends:` overlay, environment overrides for what differs per checkout, a frozen dataclass and
a fail-fast start-up -- written for this product alone.

Nothing here reads another product's files: the product is autonomous (design §5).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

import acl
from scoring import validate_evaluations

SERVICE_DIR = Path(__file__).parent
CONFIG_PATH = SERVICE_DIR / "fall_detection.yaml"


REPO_DIR = SERVICE_DIR.parents[1]


@dataclass(frozen=True)
class CameraConfig:
    id: str
    # The id the camera's streams carry: the id, or `<id>@<key_role>` beside other role sidecars (acl.key_id).
    key_id: str = ""

    def __post_init__(self):
        if not self.key_id:
            object.__setattr__(self, "key_id", self.id)


@dataclass(frozen=True)
class ServiceConfig:
    redis_url: str
    host: str
    port: int
    data_dir: Path
    cameras: tuple[CameraConfig, ...]
    scoring: dict
    classifier: dict
    suppress_after_fall_s: float
    model_manifest: str | None = None      # the product's model manifest (the sidecar yaml names the same file)
    # The zones file and the mux size its points are in; None = no zones (every camera's whole frame).
    zones_path: Path | None = None
    zones_mux: tuple | None = None
    # The safety stack's alert: a sentence ("Person fell (confidence 82%)"), the person's track and the evaluation,
    # confidence and review flag as fields. Off: the platform's alert as before ("falling 0.82").
    alert_details: bool = False

    @property
    def db_path(self) -> Path:
        return self.data_dir / "fall_detection.db"


def _merge(base: dict, overlay: dict) -> dict:
    out = dict(base)
    for key, value in overlay.items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def _read(path: Path, seen: tuple = ()) -> dict:
    """A site yaml; `extends: <other.yaml>` (beside it) is an overlay on that file: nested dicts
    merge key by key, anything else is replaced."""
    if path.resolve() in seen:
        raise ValueError(f"fall config extends cycle: {' -> '.join(p.name for p in seen + (path.resolve(),))}")
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    base = raw.pop("extends", None)
    return _merge(_read(path.parent / base, seen + (path.resolve(),)), raw) if base else raw


def load_config(path: Path | None = None, env: dict | None = None) -> ServiceConfig:
    """Load the product's yaml and fail fast on anything the service cannot run with."""
    env = os.environ if env is None else env
    path = Path(env.get("FALL_DETECTION_CONFIG") or path or CONFIG_PATH)
    raw = _read(path)

    key_role = str(raw.get("key_role") or "")
    if acl.ROLE_KEY_SEPARATOR in key_role:
        raise ValueError(f"{path}: key_role {key_role!r} cannot contain {acl.ROLE_KEY_SEPARATOR!r}")
    cameras = tuple(CameraConfig(id=str(c["id"]), key_id=acl.key_id(str(c["id"]), key_role))
                    for c in raw.get("cameras") or [])
    if not cameras:
        raise ValueError(f"{path}: no cameras configured")
    if len({c.id for c in cameras}) != len(cameras):
        raise ValueError(f"{path}: a camera id is listed twice")

    scoring = raw.get("scoring") or {}
    _check_scoring(path, scoring)
    classifier = raw.get("classifier") or {}
    if not classifier.get("device"):
        raise ValueError(f"{path}: classifier.device is required (the boot contract refuses to start without a GPU)")
    weights_dir = Path(env.get("FALL_DETECTION_WEIGHTS_DIR") or classifier.get("weights_dir") or "weights")
    classifier = {**classifier, "weights_dir": str(weights_dir if weights_dir.is_absolute()
                                                   else SERVICE_DIR / weights_dir)}

    data_dir = Path(env.get("FALL_DETECTION_DATA_DIR") or raw.get("data_dir") or "data")
    suppress = float(scoring.get("suppress_after_fall_s", 10.0))
    if suppress < 0:
        raise ValueError(f"{path}: scoring.suppress_after_fall_s cannot be negative")

    alert_details = raw.get("alert_details", False)
    if not isinstance(alert_details, bool):
        raise ValueError(f"{path}: alert_details must be true or false")

    zones = raw.get("zones") or {}
    zones_path = zones_mux = None
    if zones.get("path"):
        zones_path = Path(zones["path"])
        zones_path = zones_path if zones_path.is_absolute() else REPO_DIR / zones_path
        mux = zones.get("mux")
        if not (isinstance(mux, (list, tuple)) and len(mux) == 2 and all(float(v) > 0 for v in mux)):
            raise ValueError(f"{path}: zones.mux must be [width, height], the mux size the zone points are in")
        zones_mux = (float(mux[0]), float(mux[1]))

    return ServiceConfig(
        redis_url=env.get("REDIS_URL") or raw.get("redis_url", "redis://localhost:6379/0"),
        host=str(raw.get("host", "0.0.0.0")),
        port=int(env.get("FALL_BACKEND_PORT") or raw.get("port", 8004)),
        data_dir=data_dir if data_dir.is_absolute() else SERVICE_DIR / data_dir,
        cameras=cameras,
        scoring=scoring,
        classifier=classifier,
        suppress_after_fall_s=suppress,
        model_manifest=str(raw["model_manifest"]) if raw.get("model_manifest") else None,
        zones_path=zones_path,
        zones_mux=zones_mux,
        alert_details=alert_details,
    )


def _check_scoring(path: Path, scoring: dict) -> None:
    """The policy is the product's own: it is checked at start-up, not at the first candidate."""
    for key in ("confirm_threshold", "margin_span", "trust", "evaluations"):
        if key not in scoring:
            raise ValueError(f"{path}: scoring.{key} is required")
    threshold = float(scoring["confirm_threshold"])
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"{path}: scoring.confirm_threshold must be in (0, 1]")
    if float(scoring["margin_span"]) <= 0:
        raise ValueError(f"{path}: scoring.margin_span must be positive")
    if "alert_above" in scoring and not 0.0 <= float(scoring["alert_above"]) < 1.0:
        raise ValueError(f"{path}: scoring.alert_above must be in [0, 1) -- the confidence an alert must exceed")
    trust = scoring["trust"] or {}
    for path_name in ("action", "kinematic", "cold"):
        if path_name not in trust:
            raise ValueError(f"{path}: scoring.trust.{path_name} is required")
        if not 0.0 < float(trust[path_name]) <= 1.0:
            raise ValueError(f"{path}: scoring.trust.{path_name} must be in (0, 1] -- it is the path's ceiling")
    evaluations = scoring["evaluations"] or {}
    if not evaluations:
        raise ValueError(f"{path}: scoring.evaluations is empty")
    try:
        validate_evaluations(evaluations)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from None
