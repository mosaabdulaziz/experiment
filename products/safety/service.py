"""The safety product's own state (products/safety/README.md): per-feature settings, the cameras and their toggles,
the Apply handshake with the host controller, and the incidents still waiting for an acknowledgement.

Files, in <data>/safety/ (config `safety.data_dir` is the stack's data dir, default data/), where
tools/safety_controller.py reads them too:
  settings.json         per feature {ack_required, notify, min_severity}; Telegram on/off and its chat; the UI's default
                        language
  cameras_state.json    the UI's names and toggles per camera, which tools/safety_controller.py applies
The cameras themselves are products/safety/cameras.yaml (config `safety.cameras`, else SAFETY_CAMERAS): the UI renames
and toggles them, never adds one. The Telegram bot token is TELEGRAM_BOT_TOKEN in .env: never stored, never returned.
"""
from __future__ import annotations

import copy
import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from products.safety.config import (APPLY_APPLIED_KEY, APPLY_VERSION_KEY, effective_cameras, load_cameras,
                                    load_product, load_state)

SEVERITIES = ("info", "warning", "critical")
# How long one camera's still from go2rtc serves every viewer: each one is a decode in go2rtc's ffmpeg.
SNAPSHOT_TTL_S = 1.5
LANGUAGES = ("en", "ar")

# What a fresh stack does before anyone touches Settings: falls and fire demand an acknowledgement. Telegram is off
# (`telegram.enabled`) until Settings turns it on; then falls, fire and camera health page, a proximity alert only when
# critical, and zones and PPE (frequent, mostly handled on the spot) only show.
DEFAULT_FEATURES = {
    "proximity": {"ack_required": False, "notify": True, "min_severity": "critical"},
    "zones": {"ack_required": False, "notify": False, "min_severity": "warning"},
    "ppe": {"ack_required": False, "notify": False, "min_severity": "warning"},
    "fall": {"ack_required": True, "notify": True, "min_severity": "warning"},
    "fire": {"ack_required": True, "notify": True, "min_severity": "warning"},
    "health": {"ack_required": False, "notify": True, "min_severity": "warning"},
}


def _atomic_write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> dict:
    return load_state(path)


class SafetyService:
    def __init__(self, config: dict, store=None, redis_client=None):
        cfg = config.get("safety") or {}
        self._dir = Path(cfg.get("data_dir", "data")) / "safety"
        self._cameras_path = Path(cfg.get("cameras") or os.environ.get("SAFETY_CAMERAS")
                                  or "products/safety/cameras.yaml")
        self._product_path = Path(cfg.get("product", "products/safety/safety.yaml"))
        self._store = store
        self._redis = redis_client
        # The fall service (products/fall_detection, a host process): its review queue is reached through this app,
        # so the UI needs no second origin. SAFETY_FALL_URL, e.g. http://host.docker.internal:58014.
        self.fall_url = cfg.get("fall_url") or os.environ.get("SAFETY_FALL_URL", "")
        self.fall_http = self._fall_http
        # go2rtc's API (the stack's relay): a still of a camera's raw stream, for a camera no feature runs on.
        self.go2rtc_url = cfg.get("go2rtc_url") or os.environ.get("SAFETY_GO2RTC_URL", "http://go2rtc:1984")
        self.go2rtc_frame = self._go2rtc_frame
        self._stills: dict = {}
        self._still_locks: dict = {}

    # ── settings ──────────────────────────────────────────────────────────────────────────────────────────────
    def _saved_settings(self) -> dict:
        return _read_json(self._dir / "settings.json")

    def settings(self) -> dict:
        saved = self._saved_settings()
        features = copy.deepcopy(DEFAULT_FEATURES)
        for name, values in (saved.get("features") or {}).items():
            if name in features:
                features[name].update({k: v for k, v in values.items() if k in features[name]})
        return {"features": features,
                "telegram": {"enabled": (saved.get("telegram") or {}).get("enabled") is True,
                             "chat_id": str((saved.get("telegram") or {}).get("chat_id", "")),
                             "token_set": bool(os.environ.get("TELEGRAM_BOT_TOKEN"))},
                "language": saved.get("language", "en")}

    def update_settings(self, change: dict) -> dict:
        """Merge a partial change (feature by feature) into the saved settings; ValueError names what is wrong."""
        saved = self._saved_settings()
        features = saved.setdefault("features", {})
        for name, values in (change.get("features") or {}).items():
            if name not in DEFAULT_FEATURES:
                raise ValueError(f"unknown feature {name!r}")
            for key, value in (values or {}).items():
                if key in ("ack_required", "notify"):
                    if not isinstance(value, bool):
                        raise ValueError(f"{name}.{key} must be true or false")
                elif key == "min_severity":
                    if value not in SEVERITIES:
                        raise ValueError(f"{name}.min_severity must be one of {', '.join(SEVERITIES)}")
                else:
                    raise ValueError(f"unknown setting {name}.{key}")
                features.setdefault(name, {})[key] = value
        if "telegram" in change:
            telegram = dict(saved.get("telegram") or {})
            for key, value in (change.get("telegram") or {}).items():
                if key == "enabled":
                    if not isinstance(value, bool):
                        raise ValueError("telegram.enabled must be true or false")
                    telegram["enabled"] = value
                elif key == "chat_id":
                    telegram["chat_id"] = str(value).strip()
                else:
                    raise ValueError(f"unknown setting telegram.{key}")
            saved["telegram"] = telegram
        if "language" in change:
            if change["language"] not in LANGUAGES:
                raise ValueError(f"language must be one of {', '.join(LANGUAGES)}")
            saved["language"] = change["language"]
        _atomic_write(self._dir / "settings.json", saved)
        return self.settings()

    def requires_ack(self, feature: "str | None") -> bool:
        return bool(feature) and self.settings()["features"].get(feature, {}).get("ack_required", False)

    def should_notify(self, feature: "str | None", severity: "str | None") -> bool:
        rule = self.settings()["features"].get(feature or "")
        if not rule or not rule["notify"]:
            return False
        rank = {s: i for i, s in enumerate(SEVERITIES)}
        return rank.get(severity or "warning", 1) >= rank[rule["min_severity"]]

    # ── cameras ───────────────────────────────────────────────────────────────────────────────────────────────
    def _product(self) -> dict:
        return load_product(self._product_path)

    def _camera_file(self) -> list:
        try:
            return load_cameras(self._cameras_path)
        except OSError:
            return []

    def _state(self) -> dict:
        return load_state(self._dir / "cameras_state.json")

    def cameras(self, streams: "list | None" = None, fps: "dict | None" = None) -> list:
        """Each camera with its name and features (the UI's toggles over the file's, merged by the same function the
        controller uses), and the role streams the app sees running for it (StreamManager.list_streams()), each with
        the frames a second camera health last measured on it (`fps`, by stream; None when it has no reading)."""
        by_camera: dict = {}
        for s in streams or []:
            rate = (fps or {}).get(s["id"])
            by_camera.setdefault(s.get("camera"), []).append(
                {"id": s["id"], "feature": s.get("feature"), "running": bool(s.get("running")),
                 "fps": None if rate is None else round(float(rate), 1)})
        return [{"id": c["id"], "name": c["name"], "features": c["features"], "streams": by_camera.get(c["id"], [])}
                for c in effective_cameras(self._camera_file(), self._state(), self._product())]

    def update_camera(self, cid: str, name: "str | None" = None, features: "list | None" = None) -> dict:
        """Rename a camera and/or set its features; saved for the controller, which applies them on Apply.
        KeyError: no such camera; ValueError: an unknown feature."""
        if cid not in {str(c["id"]) for c in self._camera_file()}:
            raise KeyError(cid)
        state = self._state()
        entry = dict(state.get(cid) or {})
        if name is not None:
            entry["name"] = name.strip()
        if features is not None:
            entry["features"] = list(features)
        known = self._product().get("features") or []
        effective_cameras([{"id": cid}], {cid: entry}, {"features": known})      # ValueError on an unknown feature
        if "features" in entry:
            entry["features"] = [f for f in known if f in entry["features"]]
        state[cid] = entry
        _atomic_write(self._dir / "cameras_state.json", state)
        return next(c for c in self.cameras() if c["id"] == cid)

    # ── Apply ─────────────────────────────────────────────────────────────────────────────────────────────────
    def request_apply(self) -> int:
        """Ask the host controller to apply the saved toggles (it restarts only the role sidecars that changed)."""
        return int(self._redis.incr(APPLY_VERSION_KEY))

    def apply_status(self) -> dict:
        def _int(key):
            value = self._redis.get(key)
            return int(value) if value is not None else 0
        version, applied = _int(APPLY_VERSION_KEY), _int(APPLY_APPLIED_KEY)
        return {"version": version, "applied_version": applied, "applying": applied < version}

    # ── the fall service's review queue ────────────────────────────────────────────────────────────────────────
    def snapshot(self, camera_id: str) -> bytes:
        """A JPEG still of the camera from go2rtc's raw-<id> (what the role sidecars pull), so a camera shows with no
        feature running on it. One fetch per camera at a time, shared for SNAPSHOT_TTL_S. KeyError: no such camera;
        ConnectionError: go2rtc gave none."""
        if camera_id not in {str(c["id"]) for c in self._camera_file()}:
            raise KeyError(camera_id)
        with self._still_locks.setdefault(camera_id, threading.Lock()):
            hit = self._stills.get(camera_id)
            if not (hit and time.monotonic() - hit[0] < SNAPSHOT_TTL_S):
                try:
                    hit = (time.monotonic(), self.go2rtc_frame(f"raw-{camera_id}"))
                except ConnectionError as exc:      # a failure is shared too: viewers do not queue on a dead camera
                    hit = (time.monotonic(), exc)
                self._stills[camera_id] = hit
            if isinstance(hit[1], ConnectionError):
                raise ConnectionError(str(hit[1]))
            return hit[1]

    def _go2rtc_frame(self, src: str) -> bytes:
        url = f"{self.go2rtc_url.rstrip('/')}/api/frame.jpeg?src={urllib.parse.quote(src)}"
        try:
            with urllib.request.urlopen(url, timeout=8) as resp:
                data = resp.read()
        except (urllib.error.URLError, OSError) as exc:
            raise ConnectionError(f"go2rtc: {getattr(exc, 'reason', exc)}") from exc
        if not data:
            raise ConnectionError("go2rtc sent an empty picture")
        return data

    def _fall_http(self, method: str, path: str, body: "dict | None" = None) -> tuple:
        """(status, json) from the fall service. LookupError: none configured; ConnectionError: unreachable."""
        if not self.fall_url:
            raise LookupError("no fall service configured (SAFETY_FALL_URL)")
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.fall_url.rstrip("/") + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"} if data is not None else {})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, json.loads(resp.read() or b"{}")
        except urllib.error.HTTPError as exc:
            try:
                return exc.code, json.loads(exc.read() or b"{}")
            except ValueError:
                return exc.code, {"error": str(exc)}
        except OSError as exc:
            raise ConnectionError(str(exc)) from exc

    # ── acknowledgement ───────────────────────────────────────────────────────────────────────────────────────
    async def unacknowledged(self, limit: int = 200) -> list:
        """Incidents of features that require an acknowledgement and have none yet -- ended ones too: someone must
        see every one (the banner stays until acked)."""
        features = [f for f, rule in self.settings()["features"].items() if rule["ack_required"]]
        events = await self._store.unacknowledged(limit, features=features)
        return [e for e in events if self.requires_ack(e.get("feature"))]
