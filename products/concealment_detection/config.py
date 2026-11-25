"""Product configuration loader for `concealment_detection.yaml` and `zones.yaml`."""
from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime
import math
import os
from pathlib import Path
import yaml

from concealment.types import CameraZones, DetectorSettings, validate_detector_settings
from concealment.zones import load_zones

SERVICE_DIR = Path(__file__).parent
CONFIG_PATH = SERVICE_DIR / "concealment_detection.yaml"


@dataclass(frozen=True)
class CameraConfig:
    id: str
    location: str
    clip_duration_s: float | None = None


@dataclass(frozen=True)
class ServiceConfig:
    redis_url: str
    host: str
    port: int
    data_dir: Path
    zones_path: Path
    cameras: tuple[CameraConfig, ...]
    detector: DetectorSettings
    camera_zones: dict[str, CameraZones]
    footage_start: datetime | None
    keep_candidates: int
    review_days: int
    queue_maxsize: int
    mux_shape: tuple[int, int] = (1920, 1080)
    stream_starts_path: Path | None = None
    clip_source_dir: Path | None = None

    @property
    def db_path(self) -> Path:
        return self.data_dir / "concealment_detection.db"


def _resolve_path(value, parent: Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else parent / path).resolve()


def _merge(base: dict, overrides: dict) -> dict:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_config(path: Path, ancestors: tuple[Path, ...] = ()) -> dict:
    path = path.resolve()
    if path in ancestors:
        raise ValueError(f"Configuration extends cycle at {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a YAML mapping")
    base = {}
    if "extends" in raw:
        parent = raw.pop("extends")
        if not isinstance(parent, str) or not parent.strip():
            raise ValueError("'extends' must be a nonempty configuration path")
        base = _read_config(_resolve_path(parent, path.parent), (*ancestors, path))
    # Resolve paths before merging so inherited paths retain their declaring directory.
    for key in ("data_dir", "zones_path", "stream_starts_path", "clip_source_dir"):
        if raw.get(key) is not None:
            raw[key] = _resolve_path(raw[key], path.parent)
    return _merge(base, raw)


def _detector_settings(raw: dict) -> DetectorSettings:
    if not isinstance(raw, dict):
        raise ValueError("'detector' section must be a mapping")
    unknown = set(raw) - {f.name for f in fields(DetectorSettings)}
    if unknown:
        raise ValueError(f"Unknown detector settings: {', '.join(sorted(unknown))}")
    defaults = DetectorSettings()
    for name, value in raw.items():
        default = getattr(defaults, name)
        if isinstance(default, bool):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a boolean")
            continue
        if isinstance(default, str):
            if not isinstance(value, str):
                raise ValueError(f"{name} must be a string")
            continue
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0):
            raise ValueError(f"{name} must be a finite nonnegative number")
        if name in ("keypoint_conf_threshold", "min_core_conf_enter", "min_core_conf_exit",
                    "hidden_conf_drop_fraction") and value > 1:
            raise ValueError(f"{name} must be in [0, 1]")
        if name == "min_keypoints" and (not isinstance(value, int) or value > 17):
            raise ValueError("min_keypoints must be an integer in [0, 17]")
        if name == "max_conceal_elbow_deg" and value > 180:
            raise ValueError("max_conceal_elbow_deg must be in [0, 180]")
    detector = DetectorSettings(**raw)
    if detector.min_core_conf_exit < detector.min_core_conf_enter:
        raise ValueError("min_core_conf_exit must be >= min_core_conf_enter")
    validate_detector_settings(detector)
    return detector


def _positive_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def load_config(path: Path | None = None, env: dict | None = None) -> ServiceConfig:
    """Load config (explicit path, CONCEALMENT_CONFIG, then default) and validate it."""
    env_map = os.environ if env is None else env
    cfg_path = Path(path if path is not None else env_map.get("CONCEALMENT_CONFIG") or CONFIG_PATH).resolve()
    raw = _read_config(cfg_path)

    cameras_raw = raw.get("cameras") or []
    if not isinstance(cameras_raw, list) or len(cameras_raw) == 0:
        raise ValueError("At least one camera must be declared in 'cameras'")

    cameras: list[CameraConfig] = []
    seen: set[str] = set()
    for item in cameras_raw:
        cid = item.get("id") if isinstance(item, dict) else item
        if not isinstance(cid, str) or not cid.strip():
            raise ValueError(f"Invalid camera entry: {item!r}")
        cid = cid.strip()
        if cid in seen:
            raise ValueError(f"Duplicate camera id: {cid}")
        seen.add(cid)
        loc = str(item.get("location") or cid) if isinstance(item, dict) else cid
        duration = item.get("clip_duration_s") if isinstance(item, dict) else None
        if duration is not None and (
            isinstance(duration, bool) or not isinstance(duration, (int, float))
            or not math.isfinite(duration) or duration <= 0
        ):
            raise ValueError(f"Camera '{cid}' clip_duration_s must be finite and positive")
        cameras.append(CameraConfig(id=cid, location=loc, clip_duration_s=duration))

    redis_url = str(env_map.get("REDIS_URL") or raw.get("redis_url") or "redis://localhost:6379/0")
    host = str(env_map.get("CONCEALMENT_BACKEND_HOST") or raw.get("host") or "0.0.0.0")
    port = int(env_map.get("CONCEALMENT_BACKEND_PORT") or raw.get("port") or 8005)
    if not (1 <= port <= 65535):
        raise ValueError(f"Invalid port: {port}")

    data_dir = _resolve_path(env_map.get("CONCEALMENT_DATA_DIR") or raw.get("data_dir") or "data", cfg_path.parent)
    zones_path = _resolve_path(env_map.get("CONCEALMENT_ZONES_PATH") or raw.get("zones_path") or "zones.yaml", cfg_path.parent)
    if not zones_path.is_file():
        raise ValueError(f"Zone configuration not found: {zones_path}")
    zones_yaml = yaml.safe_load(zones_path.read_text(encoding="utf-8"))
    camera_zones = load_zones(zones_yaml)
    for cam in cameras:
        zones = camera_zones.get(cam.id)
        if zones is None or (not zones.zone_free and not zones.zones):
            raise ValueError(f"Camera '{cam.id}' needs zones or explicit zone_free: true")

    detector = _detector_settings(raw.get("detector", {}))

    fs_raw = env_map.get("CONCEALMENT_FOOTAGE_START") or raw.get("footage_start")
    footage_start = datetime.fromisoformat(str(fs_raw)) if fs_raw else None

    ret_raw = raw.get("retention", {})
    if not isinstance(ret_raw, dict):
        raise ValueError("'retention' section must be a mapping")
    keep_candidates = _positive_int(ret_raw.get("keep_candidates", 5000), "keep_candidates")
    review_days = _positive_int(ret_raw.get("review_days", 30), "review_days")
    queue_maxsize = _positive_int(ret_raw.get("queue_maxsize", 128), "queue_maxsize")
    mux_raw = raw.get("mux_shape", (1920, 1080))
    if not isinstance(mux_raw, (list, tuple)) or len(mux_raw) != 2:
        raise ValueError("mux_shape must contain width and height")
    mux_shape = (_positive_int(mux_raw[0], "mux_shape width"), _positive_int(mux_raw[1], "mux_shape height"))

    return ServiceConfig(
        redis_url=redis_url,
        host=host,
        port=port,
        data_dir=data_dir,
        zones_path=zones_path,
        cameras=tuple(cameras),
        detector=detector,
        camera_zones=camera_zones,
        footage_start=footage_start,
        keep_candidates=keep_candidates,
        review_days=review_days,
        queue_maxsize=queue_maxsize,
        mux_shape=mux_shape,
        stream_starts_path=raw.get("stream_starts_path"),
        clip_source_dir=raw.get("clip_source_dir"),
    )
