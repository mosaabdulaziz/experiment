"""Service configuration: barber.yaml, with environment overrides for what
differs per checkout (Redis URL, data dir, compose project name).

Replaces products/attendance/cameras.py's hard-coded camera table: which
cameras this shop has, and which of them are looping test clips, is
deployment data, not code.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

from model_files import KIDS_CAR_DEFAULT
from saved_reports import SavedReports, parse_saved_reports
from policy import DEFAULT_POLICY, validate_policy, validate_policy_update
from zones import FrameGeometry

SERVICE_DIR = Path(__file__).parent
REPO_DIR = SERVICE_DIR.parents[1]   # products/barber/ -> the repo root


@dataclass(frozen=True)
class CameraConfig:
    id: str
    clip_duration_s: float | None = None   # set for a looping test clip; None = live camera
    clear_on_loop: bool = False            # wipe this camera's rows when its clip loops
    name: str | None = None                # shown on the dashboard (2026-10-14); None = its translated / plain id


@dataclass(frozen=True)
class ServiceConfig:
    redis_url: str
    frame: FrameGeometry
    cameras: tuple[CameraConfig, ...]
    data_dir: Path
    stream_starts_file: Path
    crops_dir: Path
    go2rtc_url: str = "http://localhost:1984"
    engine: str = "shop"                   # "shop" (ShopEngine) | "chair" (ChairEngine, project 3)
    wait_camera: str | None = None         # the camera customer waits run on; None = the entrance-line camera
    policy: dict = field(default_factory=dict)   # the site's policy defaults, validated (over DEFAULT_POLICY)
    place: str = "BekkyBarber"            # the business on reports (2026-10-15)
    branch: str = ""                      # the branch on reports, e.g. "Palm Hills"
    branch_slug: str = ""                 # its file-name form, e.g. "palmhills"
    kids_car_model: Path | None = None    # the kids car detector (ONNX, 2026-10-15); None = the locator stays off
    kids_car_path: str = KIDS_CAR_DEFAULT  # the kids car model the yaml names (repo-relative), present or not
    model_manifest: str | None = None     # the site's model manifest (the sidecar yaml names the same file)
    logo: Path | None = None              # the shop's logo on reports (png / jpg / svg); None = a text wordmark
    saved_reports: SavedReports | None = None   # each finished period's reports on disk (2026-10-25); None = off
    # a run on recordings (2026-10-18): when its clips' first frame was filmed -- its reports give the footage's date
    # and times (footage_clock.py); None = the device's clock
    footage_start: datetime | None = None

    @property
    def db_path(self) -> Path:
        return self.data_dir / "barber-mc.db"

    def camera(self, cam_id: str) -> CameraConfig | None:
        return next((c for c in self.cameras if c.id == cam_id), None)


def _merge(base: dict, overlay: dict) -> dict:
    out = dict(base)
    for key, value in overlay.items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def _read(path: Path, seen: tuple = ()) -> dict:
    """A site yaml; `extends: <other.yaml>` (beside it) is an overlay on that file: nested dicts merge key by key,
    anything else (cameras) is replaced -- the sidecar overlay's rule (perception/engine/sources.py)."""
    if path.resolve() in seen:
        raise ValueError(f"barber config extends cycle: {' -> '.join(p.name for p in seen + (path.resolve(),))}")
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    base = raw.pop("extends", None)
    return _merge(_read(path.parent / base, seen + (path.resolve(),)), raw) if base else raw


def load_config(path: Path | None = None, env: dict | None = None) -> ServiceConfig:
    env = os.environ if env is None else env
    path = Path(env.get("BARBER_CONFIG", path or SERVICE_DIR / "barber.yaml"))
    raw = _read(path)
    frame = raw.get("frame") or {}
    cameras = tuple(
        CameraConfig(id=str(c["id"]),
                     clip_duration_s=float(c["clip_duration_s"]) if c.get("clip_duration_s") else None,
                     clear_on_loop=bool(c.get("clear_on_loop", False)),
                     name=str(c["name"]) if c.get("name") else None)
        for c in raw.get("cameras") or [])
    if not cameras:
        raise ValueError(f"{path}: no cameras configured")
    engine = str(raw.get("engine", "shop"))
    if engine not in ("shop", "chair"):
        raise ValueError(f"{path}: engine must be 'shop' or 'chair', not {engine!r}")
    data_dir = Path(env.get("BARBER_DATA_DIR") or raw.get("data_dir") or SERVICE_DIR / "data")
    crops = Path(env.get("BARBER_CROPS_DIR") or raw.get("crops_dir") or REPO_DIR / "data" / "barber" / "crops")
    wait_camera = raw.get("wait_camera")
    if wait_camera is not None and str(wait_camera) not in {c.id for c in cameras}:
        raise ValueError(f"{path}: wait_camera {wait_camera} is not one of the cameras")
    try:
        site_policy = validate_policy_update(raw.get("policy") or {})
        validate_policy({**DEFAULT_POLICY, **site_policy})
    except ValueError as exc:
        raise ValueError(f"{path}: policy: {exc}") from None
    branch = str(raw.get("branch") or "")
    kids_car_path = str(raw.get("kids_car_model") or KIDS_CAR_DEFAULT)
    model = REPO_DIR / kids_car_path
    logo = Path(str(raw["logo"])) if raw.get("logo") else None
    logo = (logo if logo.is_absolute() else SERVICE_DIR / logo) if logo else None
    if raw.get("daily_report") is not None:
        raise ValueError(f"{path}: daily_report was renamed saved_reports (spec 2026-10-25), without 'at': a period "
                         "is saved when its last shop day ends")
    try:
        saved_reports = parse_saved_reports(raw.get("saved_reports"))
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from None
    slug = str(raw.get("branch_slug") or "") or "".join(ch for ch in branch.lower() if ch.isascii() and ch.isalnum())
    footage = raw.get("footage_start")                 # yaml reads an unquoted ISO time as a datetime
    try:
        footage = (footage.replace(tzinfo=None) if isinstance(footage, datetime)
                   else datetime.fromisoformat(str(footage)) if footage else None)
    except ValueError:
        raise ValueError(f"{path}: footage_start must be an ISO date and time, e.g. 2026-10-16T09:29:58") from None
    return ServiceConfig(
        redis_url=env.get("REDIS_URL") or raw.get("redis_url", "redis://localhost:6379/0"),
        frame=FrameGeometry(float(frame.get("width", 1080)), float(frame.get("height", 720))),
        cameras=cameras,
        data_dir=data_dir if data_dir.is_absolute() else SERVICE_DIR / data_dir,
        stream_starts_file=REPO_DIR / "data" / "stream_starts.json",
        crops_dir=crops if crops.is_absolute() else SERVICE_DIR / crops,
        engine=engine,
        wait_camera=str(wait_camera) if wait_camera is not None else None,
        policy=site_policy,
        go2rtc_url=str(env.get("GO2RTC_URL") or raw.get("go2rtc_url")
                       or f"http://localhost:{env.get('GO2RTC_API_PORT', '1984')}"),
        place=str(raw.get("place") or "BekkyBarber"), branch=branch, branch_slug=slug,
        kids_car_model=model if model.is_file() else None,
        kids_car_path=kids_car_path,
        model_manifest=str(raw["model_manifest"]) if raw.get("model_manifest") else None,
        logo=logo if logo is not None and logo.is_file() else None,
        saved_reports=saved_reports,
        footage_start=footage,
    )


def load_stream_starts(path: Path) -> dict[str, datetime]:
    """cam_id -> when the test streamer (re)started that camera's clip
    (test_streamer writes the file). Missing file/camera = not streaming."""
    try:
        with open(path) as f:
            raw = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return {cam_id: datetime.fromtimestamp(ts) for cam_id, ts in raw.items()}
