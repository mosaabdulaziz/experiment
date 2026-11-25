"""Pure (GPU-free) config helpers for the sidecar. Safe to import anywhere."""
import logging
import os
import re
import time
from dataclasses import dataclass, field

logger = logging.getLogger("iris.sources")

# ${NAME} references in source URIs (e.g. RTSP credentials) are expanded from
# the environment at load time — never commit secrets into sidecar.yaml.
_ENV_REF = re.compile(r"\$\{(\w+)\}")


def _expand_env(value: str) -> str:
    """Expand ${NAME} refs from the environment; leave unset ones in place.

    RTSP URIs carry ${RTSP_USER}/${RTSP_PASS} placeholders (docker-compose
    passes the host .env values into the container). An unset var keeps the
    placeholder so the source fails visibly at connect time instead of
    producing a malformed URI with empty credentials.
    """
    def _sub(m):
        name = m.group(1)
        val = os.environ.get(name)
        if not val:
            logger.warning(
                "env var %s referenced in source config is unset; keeping "
                "placeholder (source will fail to authenticate)", name)
            return m.group(0)
        return val

    return _ENV_REF.sub(_sub, value)


def normalize_roles(value) -> list:
    """Coerce a raw yaml `roles:` value into a list of role strings.

    `roles: gate` (a bare scalar) is an easy yaml typo for `roles: [gate]`, and
    a plain `list(...)` over it silently yields ['f','i','r','e'] -- a role list
    that matches nothing, with no error anywhere. A string is therefore wrapped
    as a single-element list; anything else falsy becomes []; everything else is
    listed as-is.
    """
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def owns_recording(source: dict, role: "str | None", untagged_owner: "str | None" = None) -> bool:
    """Whether the process running `role` owns `source`'s recording state.

    This is deliberately NOT `select_sources`'s rule. Detection coverage is
    additive and safe to duplicate, so an untagged source is included in EVERY
    role's slice (nothing silently drops off a process just because nobody
    tagged it). Recording state is the opposite: `_flush_segment_index` and
    `_flush_svr_dir` DELETE what is already on disk / in Redis at boot, so an
    untagged source owned by both processes means whichever boots second wipes
    the other's in-flight recordings. Those operations must have exactly one
    owner, so an untagged source's recording ownership goes to the role the
    settings name (`untagged_recording_owner`), not to "every role". (The app's
    `_camera_roles` diverges from select_sources for the same kind of reason.)

    `role=None` (single-process deployment) owns everything -- today's behavior,
    byte-for-byte. An explicitly tagged source is owned by the role(s) it names,
    which is already exclusive by construction since select_sources partitions
    tagged sources cleanly. With no `untagged_owner` an untagged source is owned
    by no role: recording needs a named owner.
    """
    if role is None:
        return True
    return role in (normalize_roles(source.get("roles")) or ([untagged_owner] if untagged_owner else []))


def resolve_sources(cfg: dict) -> list:
    """Return [{'id': str, 'uri': str, 'roles': list}, ...] from config.

    Prefers cfg['sources'] (a list of {id, uri, roles?}); falls back to a single
    cfg['uri'] (+ optional cfg['source_id']) for backward compatibility with the
    single-stream config. Entries without a uri are skipped; missing ids default
    to camN by position. Each uri passes through _expand_env, so ${VAR} refs
    (e.g. RTSP credentials) are resolved from the container environment. Roles
    defaults to an empty list if not specified.
    """
    sources = cfg.get("sources")
    if sources:
        resolved = []
        for i, s in enumerate(sources):
            uri = s.get("uri")
            if not uri:
                continue
            resolved.append({"id": str(s.get("id") or f"cam{i}"),
                             "uri": _expand_env(uri),
                             "roles": normalize_roles(s.get("roles"))})
        return resolved
    uri = cfg.get("uri")
    if uri:
        return [{"id": str(cfg.get("source_id", "cam0")), "uri": _expand_env(uri), "roles": []}]
    return []


def select_sources(all_sources: list, shard_id: int, shard_count: int,
                   role: "str | None" = None) -> list:
    """Filter the full source list down to what one process should own.

    Role filtering happens first: a source with no `roles` tag matches every
    role (a safety default -- nothing silently drops off a process's slice
    just because nobody tagged it yet); a tagged source matches only when
    `role` is in its `roles` list. Index-modulo sharding then applies to
    whatever the role filter left, so the two mechanisms compose without
    special-casing. `role=None` (or falsy) skips role filtering entirely,
    reproducing today's shard-only behavior byte-for-byte.
    """
    if role:
        all_sources = [s for s in all_sources
                       if not s.get("roles") or role in s["roles"]]
    return [s for i, s in enumerate(all_sources) if i % shard_count == shard_id]


ROLE_KEY_SEPARATOR = "@"

def stream_key_id(source_id: str, role: "str | None", scoped: bool) -> str:
    """The id a process's per-camera Redis keys carry (iris:status/meta/jpeg/overlay/alert/... :<id>).

    Unscoped (the default, `role_scoped_keys` unset) it is the camera id, whatever the role -- every key name stays
    what it always was. Scoped it is `<camera>@<role>`, so several role processes can run on one camera (each with
    its own detector and modules) without overwriting each other's status
    (boot_id), mixing their meta, previews and overlays, or answering each other's clip requests."""
    if not scoped:
        return source_id
    if not role:
        raise ValueError("role_scoped_keys needs a role (IRIS_SIDECAR_ROLE or sidecar_role)")
    for name, value in (("camera id", source_id), ("role", role)):
        if ROLE_KEY_SEPARATOR in value:
            raise ValueError(f"role_scoped_keys: {name} {value!r} contains {ROLE_KEY_SEPARATOR!r}")
    return f"{source_id}{ROLE_KEY_SEPARATOR}{role}"


def split_stream_key_id(key_id: str) -> "tuple[str, str | None]":
    """(camera, role) from a key id; role is None for an unscoped one."""
    camera, sep, role = key_id.partition(ROLE_KEY_SEPARATOR)
    return (camera, role) if sep else (key_id, None)


def stale_status_keys(existing: list, current_ids: set, role: "str | None", scoped: bool) -> list:
    """The iris:status:* keys a booting process deletes as ghosts of an earlier camera list.

    Unscoped: every status key whose id is not a configured camera (unchanged behaviour). Scoped: only this role's own
    keys (`<camera>@<role>`) whose camera is not configured -- another role's keys belong to another process."""
    stale = []
    for key in existing:
        text = key.decode() if isinstance(key, bytes) else key
        key_id = text.split("iris:status:", 1)[-1]
        if not scoped:
            if key_id not in current_ids:
                stale.append(key)
            continue
        camera, key_role = split_stream_key_id(key_id)
        if key_role == role and camera not in current_ids:
            stale.append(key)
    return stale


def role_scope_problem(cfg: dict, role: "str | None", sources: list) -> "str | None":
    """Why this process cannot start with `role_scoped_keys`, or None (always None when the setting is off). Modules
    that write camera-keyed streams are refused once loaded (modules.camera_keyed_modules)."""
    if not cfg.get("role_scoped_keys"):
        return None
    if not role:
        return "role_scoped_keys needs a role: set IRIS_SIDECAR_ROLE (or sidecar_role)"
    try:
        for source in sources:
            stream_key_id(source["id"], role, scoped=True)
    except ValueError as exc:
        return str(exc)
    return None


@dataclass
class StreamState:
    id: str
    meta_key: str
    jpeg_key: str
    status_key: str
    # The id this stream's Redis keys carry: the camera id, or `<camera>@<role>` with role_scoped_keys
    # (stream_key_id). `id` stays the camera id: zones, settings and module state are per camera.
    key_id: str = ""
    _last_pts: "float | None" = None
    _last_stream: float = 0.0
    # Scratch space per module, keyed by module name (see perception/engine/modules.py). Replaces attributes
    # bolted onto the state from outside.
    module_state: dict = field(default_factory=dict)


def make_stream_state(source_id: str, key_id: "str | None" = None) -> StreamState:
    key = key_id or source_id
    return StreamState(
        id=source_id,
        meta_key=f"iris:meta:{key}",
        jpeg_key=f"iris:jpeg:{key}",
        status_key=f"iris:status:{key}",
        key_id=key,
    )


def apply_role_overrides(cfg: dict, role: "str | None") -> dict:
    """Return a copy of `cfg` changed for the role this process runs as, from the `role_overrides:` table in the
    settings (a role selects sources by tag; what else it changes is data, not code):

        role_overrides:
          gate:
            pgie_enabled: false                 # set an engine key
            drop_modules: [zones, face_detect]  # and/or leave modules out of this process

    `role=None`, or a role with no entry, is a total no-op, so an unset IRIS_SIDECAR_ROLE reproduces the plain
    deployment byte-for-byte. A role can only take modules away, never add one the yaml did not list.
    """
    cfg = dict(cfg)
    entry = dict((cfg.get("role_overrides") or {}).get(role) or {}) if role else {}
    drop = set(entry.pop("drop_modules", None) or [])
    cfg.update(entry)
    if drop:
        cfg["modules"] = {k: v for k, v in (cfg.get("modules") or {}).items() if k not in drop}
    return cfg


def known_roles(cfg: dict, sources: list) -> set:
    """The roles a process can run as: those the settings give an override entry and those some source is tagged with."""
    roles = set((cfg.get("role_overrides") or {}))
    for source in sources:
        roles.update(normalize_roles(source.get("roles")))
    return roles


def _deep_merge(base: dict, overlay: dict) -> dict:
    """overlay on top of base: nested dicts merge key by key; anything else
    (scalars, and lists such as `sources:`) is replaced outright."""
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_sidecar_config(path: str) -> dict:
    """Read a sidecar yaml. A file with `extends: <other.yaml>` (resolved
    relative to its own directory) is an overlay: it holds only what differs
    from its base, deep-merged on top -- so a per-deployment config such as
    a product's own overlay doesn't have to copy and then drift from every
    tuned knob in sidecar.yaml. Chains are allowed; a cycle is refused."""
    import yaml

    seen: list[str] = []

    def load(p: str) -> dict:
        p = os.path.abspath(p)
        if p in seen:
            raise ValueError(f"sidecar config extends cycle: {' -> '.join(seen + [p])}")
        seen.append(p)
        with open(p) as f:
            cfg = yaml.safe_load(f) or {}
        base = cfg.pop("extends", None)
        if base is None:
            return cfg
        return _deep_merge(load(os.path.join(os.path.dirname(p), base)), cfg)

    return load(path)


def preview_plan(need_meta, need_stream, pipeline_preview):
    """Decide this frame's preview work under the `pipeline_preview` mode.

    pipeline_preview=true (default): the classic path — every `need_stream`
    frame is surface-copied, box-drawn, JPEG-encoded and published to
    iris:jpeg:<id>.

    pipeline_preview=false (inference-only / raw preview): DeepStream stops
    producing annotated JPEGs entirely. The probe skips the GPU->CPU surface
    copy (the measured ~2 ms/frame `probe.surface_copy`) and the publisher skips
    draw + encode (the measured `publish.draw`). The frontend plays the source's
    raw RTSP (exposed as `preview_uri` in iris:status:<id>) and overlays boxes
    from iris:meta:<id>. `iris:meta` still advances at the analysis cadence, and
    modules that crop the frame (faces, bodies) still get a copy -- but ONLY when
    they have something to crop (their `wants_pixels` hook).

    Returns (do_preview, need_bgr):
      do_preview — publish an annotated JPEG for this frame.
      need_bgr   — the CPU BGR frame copy is required by the engine itself (the preview render); modules that crop
                   the frame ask for it separately.
    """
    do_preview = bool(need_stream and pipeline_preview)
    return do_preview, do_preview



class ViewerGate:
    """Cache whether a stream currently has a viewer.

    The web server refreshes `iris:watched:<id>` (short TTL) on every snapshot
    or MJPEG request for a stream. The sidecar's frame publisher consults this
    before spending GPU time encoding a preview JPEG: with viewer-gated preview
    enabled (sidecar.yaml `viewer_gated_preview: true`), streams nobody is
    watching skip the encode entirely, while analysis/meta/dets still run at
    full rate. The Redis check is cached (check_interval) so the publisher
    never pays a round-trip per frame; Redis errors fail OPEN (always encode)
    so a hiccup never starves previews.
    """

    KEY_PREFIX = "iris:watched:"

    def __init__(self, redis, check_interval: float = 2.0):
        self._r = redis
        self._interval = check_interval
        self._cache: dict = {}

    def is_watched(self, stream_id: str, now: "float | None" = None) -> bool:
        now = time.monotonic() if now is None else now
        hit = self._cache.get(stream_id)
        if hit is not None and now - hit[0] < self._interval:
            return hit[1]
        try:
            watched = bool(self._r.exists(f"{self.KEY_PREFIX}{stream_id}"))
        except Exception:
            watched = True  # fail open: never starve previews on a Redis hiccup
        self._cache[stream_id] = (now, watched)
        return watched
