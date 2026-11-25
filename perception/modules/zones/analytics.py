"""nvdsanalytics plumbing for the zones module: rendering this process's ROI config from the zones file and its own
streammux slot map, and the empty default that lets the pipeline boot when that render cannot be written.

Pure Python (no gi/pyds imports) so it stays unit-testable off-GPU.
"""
import json
import logging
import os

logger = logging.getLogger("iris.sidecar")


_HEADER = "[property]\nenable=1\nconfig-width={w}\nconfig-height={h}\nosd-mode=0\n"


def slot_map(sources: list) -> dict:
    """{streammux slot: camera id} for this process. Today a camera's slot is its position in the process's own
    source list (pads are requested in that order at build time); attach/detach (#173) will hand the module a live
    map instead, and everything downstream already takes the map, not the list."""
    return {k: s["id"] for k, s in enumerate(sources)}


def roi_lines(entry) -> list:
    """`roi-<id>=x;y;...` for each zone of one camera's stored entry that has a feature nvdsanalytics must flag.

    Reads the same shapes as app/zones.py:normalize_stream_entry and config.py: {"zones": [...]}, or a legacy
    single-zone entry; a zone without points is skipped but still takes its position for the z<i> fallback id. A
    zone with intrusion off and overcrowding off contributes nothing (the proximity / safety features don't use
    nvdsanalytics)."""
    if not isinstance(entry, dict):
        return []
    raw_zones = entry.get("zones")
    if raw_zones is None:
        raw_zones = [entry] if entry.get("points") else []
    lines = []
    for i, z in enumerate(raw_zones):
        if not isinstance(z, dict) or not z.get("points"):
            continue
        intr = z.get("intrusion")
        mode = (intr.get("mode", "restricted") if intr is not None
                else z.get("type", "off" if "proximity" in z else "restricted"))
        overcrowding = bool((z.get("overcrowding") or {}).get("enabled", False))
        if mode == "off" and not overcrowding:
            continue
        coords = ";".join(str(c) for point in z["points"] for c in point)
        lines.append(f"roi-{z.get('id') or f'z{i}'}={coords}")
    return lines


def render_analytics_config(zones: dict, slots: dict, mux_width: int, mux_height: int) -> str:
    """The nvdsanalytics config for one process: one [roi-filtering-stream-<slot>] per slot whose camera has active
    zones, in slot order. `zones` is the stored zones file ({camera id: entry}); `slots` is {slot: camera id}.

    nvdsanalytics keys roiStatus by streammux pad (slot), so each process renders from its OWN slot map -- there is
    no global numbering to translate. Groups are per stream and inverse-roi is group-level, so each stream is one
    NON-inverted group with a named ROI per zone; the module applies restricted/valid and overcrowding per zone in
    Python from roiStatus. class-id stays -1 (all classes); per-class filtering is done from object labels."""
    out = _HEADER.format(w=mux_width, h=mux_height)
    for slot, camera in sorted(slots.items()):
        roi = roi_lines(zones.get(camera))
        if roi:
            out += "\n" + "\n".join([f"[roi-filtering-stream-{slot}]", "enable=1", *roi,
                                     "inverse-roi=0", "class-id=-1"]) + "\n"
    return out


def read_zones(path: str) -> dict:
    """The stored zones file ({camera id: entry}); {} when it is missing or unreadable (no zones), as for _load."""
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        logger.warning("could not read zones file %s (%s); rendering no ROIs", path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def worker_config_path(analytics_path: str, role, shard_id: int = 0) -> str:
    """This process's own config file, next to the configured one. The role and shard in the name keep the containers
    that share the data/ mount from overwriting each other's copy: shard replicas of one config have the same role."""
    base, ext = os.path.splitext(analytics_path)
    return f"{base}_worker_{role or 'single'}{f'_shard{shard_id}' if shard_id else ''}{ext}"


def write_worker_config(analytics_path: str, text: str, role, mux_w: int, mux_h: int, shard_id: int = 0) -> str:
    """Write this process's rendered config and return its path. If it cannot be written, fall back to the configured
    file (created empty if missing, so the element still parses it and the pipeline boots)."""
    out_path = worker_config_path(analytics_path, role, shard_id)
    try:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        tmp = out_path + ".tmp"
        with open(tmp, "w") as f:
            f.write(text)
        os.replace(tmp, out_path)
    except OSError as exc:
        logger.warning("could not write analytics config %s (%s); falling back to %s -- zone ROIs may not match "
                       "this process's streams", out_path, exc, analytics_path)
        ensure_default_config(analytics_path, mux_w, mux_h)
        return analytics_path
    return out_path


def ensure_default_config(path: str, mux_w: int, mux_h: int) -> None:
    """Write the header-only default (no ROIs) to the fallback path, ALWAYS overwriting whatever is there.

    A missing config file is FATAL: nvdsanalytics posts GST_ELEMENT_ERROR "Failed to parse config file" on an
    unreadable path, which kills the whole pipeline at boot (a fresh clone has no data/nvdsanalytics_config.txt until
    the app writes zones). But an OLD file left at this path is just as wrong: when this process cannot write its own
    render, the shared file must not be served with another run's stale `[roi-filtering-stream-N]` sections (ROIs that
    may not even belong to this process's streams). So overwrite it header-only, atomically, rather than keep a stale
    file. Only the fallback of write_worker_config needs this: the process normally loads its own render."""
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            f.write(_HEADER.format(w=mux_w, h=mux_h))
        os.replace(tmp, path)
        logger.warning("nvdsanalytics config %s: wrote a header-only default "
                       "(no ROIs; zones will apply once saved in the app)", path)
    except OSError as exc:
        logger.warning("could not write default analytics config %s: %s", path, exc)
