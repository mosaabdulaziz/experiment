"""The pipeline settings this app runs against -- one place to read them.

The app reports camera roles, mux size and which features are on, all from
the pipeline's own settings file, so the two must read the SAME file. Before
2026-10-01 (MR !25 review) that took three layers across two modules:
zones._load_sidecar_yaml wrapped two private helpers in stream_manager, which
wrapped the pipeline's loader. The middle layer also hid two silent failures:
without the loader it read the file WITHOUT following `extends:`, and any error
quietly returned {}.
"""
from __future__ import annotations

import copy
import logging
import os

from perception.engine.sources import load_sidecar_config, stream_key_id

logger = logging.getLogger(__name__)

DEFAULT_PATH = "perception/pipeline/sidecar.yaml"
_warned: set[str] = set()


def running_sidecar_path(app_config: dict) -> str:
    """Which settings file the pipeline is running.

    SIDECAR_CONFIG first: docker-compose hands the pipeline the same variable,
    so app and pipeline cannot disagree (2026-09-22: the dashboard showed no
    cameras during a run on an overlay for exactly this reason). Then
    config.yaml's zones.sidecar_config_path, then the shared base file.
    """
    configured = (app_config.get("zones") or {}).get("sidecar_config_path") or DEFAULT_PATH
    return os.environ.get("SIDECAR_CONFIG") or configured


_cache: dict = {}          # path -> ((mtime_ns, size), settings)


def _load(path: str) -> dict:
    """One file's settings, with `extends:` overlays resolved. A missing or unreadable file gives {}, so every
    feature that depends on it stays off -- and it is logged (once per path, since this is read on every request),
    instead of showing up only as an empty dashboard. Parsed again only when the file itself changes (the files it
    extends are the repo's, which change only with a restart); each caller gets its own copy."""
    try:
        st = os.stat(path)
        stamp = (st.st_mtime_ns, st.st_size)
        hit = _cache.get(path)
        if hit is None or hit[0] != stamp:
            hit = (stamp, load_sidecar_config(path) or {})
            _cache[path] = hit
        return copy.deepcopy(hit[1])
    except Exception as exc:
        if path not in _warned:
            _warned.add(path)
            logger.warning("pipeline settings %s could not be read (%s: %s); "
                           "features that depend on them stay off", path, type(exc).__name__, exc)
        return {}


def merge_role_configs(cfgs: list) -> dict:
    """Several role sidecars' settings as the one view the app reads (the safety product: one process per feature,
    several per camera). The first file's engine settings (they share the mux size); every role's modules; every
    role's sources, each renamed to the stream id its process publishes under (`<camera>@<role>` with role-scoped
    keys, perception/engine/sources.py stream_key_id), so they match the streams the app discovers."""
    if not cfgs:
        return {}
    merged = {k: v for k, v in cfgs[0].items() if k not in ("modules", "sources")}
    modules, sources = {}, []
    for cfg in cfgs:
        modules.update(cfg.get("modules") or {})
        role, scoped = cfg.get("sidecar_role"), bool(cfg.get("role_scoped_keys"))
        for source in cfg.get("sources") or []:
            if source.get("id"):
                sources.append({**source, "id": stream_key_id(str(source["id"]), role, scoped)})
    merged.update(modules=modules, sources=sources)
    return merged


def _running_paths(app_config: dict) -> list:
    return [p.strip() for p in running_sidecar_path(app_config).split(",") if p.strip()]


def running_sidecar_config(app_config: dict) -> dict:
    """The running pipeline's settings: one file (SIDECAR_CONFIG, else config.yaml's), or -- when SIDECAR_CONFIG lists
    several, comma-separated -- every role sidecar's, merged (merge_role_configs)."""
    paths = _running_paths(app_config)
    if len(paths) == 1:
        cfg = _load(paths[0])
        # a role-scoped process publishes <camera>@<role>: name its sources so, alone as among several
        return merge_role_configs([cfg]) if cfg.get("role_scoped_keys") else cfg
    return merge_role_configs([cfg for cfg in (_load(p) for p in paths) if cfg])


def module_enabled(sidecar_cfg: dict, name: str) -> bool:
    """Whether the pipeline runs the module `name`: a module is on exactly when its yaml lists it under `modules:`."""
    return name in (sidecar_cfg.get("modules") or {})


def module_setting(sidecar_cfg: dict, name: str, key: str, default):
    """One setting of the module `name` as the yaml states it, or `default` when it does not (the module's own
    default applies in the pipeline, so pass the same value the module's Settings uses)."""
    settings = (sidecar_cfg.get("modules") or {}).get(name)
    return settings.get(key, default) if isinstance(settings, dict) else default
