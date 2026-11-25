"""Which Redis streams the bridge reads: exactly the `iris:alert:<key id>` the shelf sidecar writes.

The engine names a camera's alert stream `iris:alert:<key id>` (perception/engine/modules.py EngineServices.alert),
where the key id is `stream_key_id(camera, role, role_scoped_keys)` (perception/engine/sources.py): the camera id, or
`<camera>@<role>` when the sidecar runs with `role_scoped_keys`. The bridge loads the SAME sidecar yaml
(SIDECAR_CONFIG, `extends:` followed) and resolves the role as perception/engine/main.py does -- IRIS_SIDECAR_ROLE
when set, else the yaml's `sidecar_role`; then that role's `role_overrides` -- so it can't drift from the sidecar.

The mapping file stays keyed by camera: a stream's `@<role>` suffix is stripped (split_stream_key_id) before the
pipelineId / cameraId lookup.
"""
import os
from dataclasses import dataclass
from typing import Optional

from perception.engine.sources import (apply_role_overrides, load_sidecar_config, role_scope_problem,
                                       split_stream_key_id, stream_key_id)

ALERT_PREFIX = "iris:alert:"        # perception/engine/modules.py EngineServices.alert


class KeyScopeError(ValueError):
    """The sidecar config can't tell which streams to read: the bridge refuses to start."""


@dataclass(frozen=True)
class AlertKeys:
    role: Optional[str]
    scoped: bool                    # the sidecar's role_scoped_keys
    streams: tuple                  # the iris:alert:<key id> streams, one per mapped camera, in camera order

    @property
    def mode(self) -> str:
        return "role-scoped" if self.scoped else "camera"


def alert_keys(cameras, sidecar_cfg: dict, env_role: Optional[str] = None) -> AlertKeys:
    """The alert streams of `cameras` for a sidecar with this (loaded) yaml. `env_role`: IRIS_SIDECAR_ROLE as the
    sidecar sees it (None = unset; set-but-empty means no role, as in the engine)."""
    role = (env_role if env_role is not None else sidecar_cfg.get("sidecar_role")) or None
    cfg = apply_role_overrides(sidecar_cfg, role)
    problem = role_scope_problem(cfg, role, [{"id": cam} for cam in cameras])
    if problem:
        raise KeyScopeError(problem)
    scoped = bool(cfg.get("role_scoped_keys", False))
    return AlertKeys(role=role, scoped=scoped,
                     streams=tuple(ALERT_PREFIX + stream_key_id(cam, role, scoped) for cam in cameras))


def load_alert_keys(cameras, sidecar_config: str, env_role: Optional[str] = None, root: str = "") -> AlertKeys:
    """alert_keys from the sidecar yaml at `sidecar_config` (relative paths from `root`, the repo checkout)."""
    if not sidecar_config:
        raise KeyScopeError("SIDECAR_CONFIG is required: the shelf sidecar's preset (the same value the sidecar runs "
                            "with, e.g. configs/presets/shelf/live-test.yaml), to read its role_scoped_keys and role")
    path = os.path.join(root, sidecar_config) if root else sidecar_config
    try:
        cfg = load_sidecar_config(path)
    except OSError as e:
        raise KeyScopeError(f"cannot read the sidecar config {path}: {e.strerror}") from e
    except ValueError as e:         # yaml errors and `extends:` cycles
        raise KeyScopeError(f"bad sidecar config {path}: {e}") from e
    return alert_keys(cameras, cfg, env_role)


def camera_of(stream: str) -> str:
    """The mapping's camera for an alert stream: `iris:alert:<camera>[@<role>]` -> `<camera>`."""
    return split_stream_key_id(stream[len(ALERT_PREFIX):])[0]
