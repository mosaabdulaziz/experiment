"""The exclusivity preflight: the fall deployment must not run beside a live sidecar that
already holds one of its cameras.

The `roles` tag names the recording's owner but does not exclude a second process
(perception/engine/sources.py:52-74), so the enforcement is here, in `just fall-up`, and it
names the sidecar it found.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

import acl

REPO = Path(__file__).resolve().parents[2]
PRODUCT_CONFIG = Path(__file__).with_name("fall_detection.yaml")


class PreflightConflict(RuntimeError):
    """A live sidecar already holds one of the cameras the fall product wants."""


def check_cameras(fall_cameras, live_sidecars) -> None:
    """Refuse when a live sidecar holds one of the fall cameras.

    fall_cameras -- the camera ids the fall product is about to take.
    live_sidecars -- [{"service": <name>, "cameras": [<id>, ...]}, ...] as collected from docker.
    """
    wanted = set(fall_cameras)
    conflicts = []
    for sidecar in live_sidecars:
        held = wanted & set(sidecar.get("cameras") or [])
        if held:
            conflicts.append(f"{sidecar.get('service')} holds {', '.join(sorted(held))}")
    if conflicts:
        raise PreflightConflict(
            "the fall product's cameras are already held by a live sidecar: " + "; ".join(conflicts)
            + " -- stop it first (just fall-up refuses to run two processes on one camera)")


# ── the inventory, as docker reports it ────────────────────────────────────────────────────────────────────

def _deep_merge(base: dict, overlay: dict) -> dict:
    out = dict(base)
    for key, value in overlay.items():
        out[key] = _deep_merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def _read(path: Path, seen: tuple = ()) -> dict:
    """A sidecar yaml with its `extends` chain applied, the engine's own semantics
    (perception/engine/sources.py:185-207): the base is relative to the overlay's directory,
    lists are replaced, dicts merge."""
    path = path.resolve()
    if path in seen:
        raise ValueError(f"sidecar config extends cycle: {' -> '.join(p.name for p in seen + (path,))}")
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    base = raw.pop("extends", None)
    return _deep_merge(_read(path.parent / base, seen + (path,)), raw) if base else raw


def cameras_for_role(cfg: dict, role: str | None) -> list[str]:
    """The sources one process holds, by `select_sources`'s rule: a tagged source matches
    only its roles, an untagged one matches every role (perception/engine/sources.py:104-119)."""
    sources = cfg.get("sources") or []
    if role:
        sources = [s for s in sources if not s.get("roles") or role in s["roles"]]
    return [str(s["id"]) for s in sources if s.get("id")]


def keys_for_role(cfg: dict, role: str | None) -> list[str]:
    """The stream key ids one process writes (perception/engine/sources.py stream_key_id): its cameras, or
    `<camera>@<role>` when its config has `role_scoped_keys`. Two processes collide only on the same key id, so a
    role-scoped sidecar beside the fall one on the same camera is no conflict."""
    cameras = cameras_for_role(cfg, role)
    if cfg.get("role_scoped_keys") and role:
        return [acl.key_id(camera, role) for camera in cameras]
    return cameras


def _docker(*args: str) -> str:
    try:
        return subprocess.run(["docker", *args], capture_output=True, text=True, check=True).stdout
    except FileNotFoundError:
        raise SystemExit("fall-up preflight: docker is not on PATH") from None
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"fall-up preflight: docker {' '.join(args)} failed: {exc.stderr.strip()}") from None


def live_sidecars() -> list[dict]:
    """Every running sidecar container, its config and the cameras its role selects.

    A container's command is the engine's command line, whose last `.yaml` argument is the
    sidecar config; its environment names the role the engine filters sources by.
    """
    names = [name for name in _docker("ps", "--format", "{{.Names}}").split() if "deepstream-sidecar" in name]
    live = []
    for name in names:
        info = json.loads(_docker("inspect", name))[0]
        command = info["Config"].get("Cmd") or []
        env = dict(item.split("=", 1) for item in info["Config"].get("Env") or [] if "=" in item)
        config = next((arg for arg in reversed(command) if arg.endswith((".yaml", ".yml"))), None)
        if config is None:
            raise SystemExit(f"fall-up preflight: the running sidecar {name} names no sidecar yaml")
        role = env.get("IRIS_SIDECAR_ROLE") or None
        live.append({"service": name, "cameras": keys_for_role(_read(REPO / config), role)})
    return live


def fall_cameras(path: Path | None = None) -> list[str]:
    """The stream key ids this product is about to take, from its own config: its cameras, or `<camera>@<key_role>`
    when it runs role-scoped (acl.key_id)."""
    path = Path(os.environ.get("FALL_DETECTION_CONFIG") or path or PRODUCT_CONFIG)
    raw = _read(path)
    return [acl.key_id(str(c["id"]), str(raw.get("key_role") or "")) for c in raw.get("cameras") or []]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="refuse to start the fall sidecar beside a live one on its cameras")
    parser.add_argument("--cameras", nargs="*", default=None,
                        help="override the camera ids (default: the product's fall_detection.yaml)")
    args = parser.parse_args(argv)
    wanted = args.cameras if args.cameras is not None else fall_cameras()
    try:
        check_cameras(wanted, live_sidecars())
    except PreflightConflict as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"fall-up preflight: {', '.join(wanted)} are free")
    return 0


if __name__ == "__main__":
    sys.exit(main())
