"""Camera-pair graph: per-edge association gates.

Instead of global thresholds, each camera pair gets its own configuration:
timing mode (near-synchronous vs sequential), overlap tolerance, Re-ID
threshold, and whether to use the camera model (homography) for that pair.

Usage:
    graph = load_cam_graph("cam_graph.yaml", cameras=["c001","c002"])
    edge = graph.edge("c001", "c002")
    # edge.time_tolerance_s, edge.reid_sim, edge.use_camera_model, etc.

YAML schema (see ``mtmc/examples/cam_graph.yaml`` for a full example):

    defaults:
      time_tolerance_s: 1.5
      reid_sim: 0.55
      timing: near_synchronous   # near_synchronous | sequential | either
      use_camera_model: false

    cameras:                      # optional per-camera overrides
      c001:
        fps: 10.0

    edges:                        # optional per-edge overrides
      - [c001, c002]              # both orderings map to the same EdgeConfig
      - [c002, c003]
        timing: sequential
        max_transit_s: 30.0
      - [c001, c005]
        reid_sim: 0.50
        use_camera_model: true
        spatial_gate_px: 300.0
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Edge configuration
# ---------------------------------------------------------------------------

@dataclass
class EdgeConfig:
    """Association parameters for a specific camera pair."""

    # --- timing ---
    timing: str = "near_synchronous"  # near_synchronous | sequential | either
    time_tolerance_s: float = 1.5     # max gap between tracklets to allow merge
    max_transit_s: float | None = None  # sequential-only: upper bound on mid→mid travel

    # --- appearance ---
    reid_sim: float = 0.55            # cosine similarity threshold

    # --- camera model (homography) ---
    use_camera_model: bool = False
    spatial_gate_px: float = 400.0    # max projected distance before hard gate
    spatial_merge_px: float = 120.0   # below this = spatial proof (no sim needed)

    # --- transition windows ---
    use_transition_windows: bool = False

    # --- stitching (rarely per-edge, but available) ---
    stitch: bool = False
    stitch_sim: float = 0.65


@dataclass
class CamGraph:
    """A camera-pair graph with per-edge configuration."""

    cameras: list[str]
    defaults: EdgeConfig
    _edges: dict[tuple[str, str], EdgeConfig] = field(default_factory=dict)

    def edge(self, cam_a: str, cam_b: str) -> EdgeConfig:
        """Return the EdgeConfig for a camera pair (order-independent)."""
        key = _pair_key(cam_a, cam_b)
        return self._edges.get(key, self.defaults)

    def has_edge(self, cam_a: str, cam_b: str) -> bool:
        return _pair_key(cam_a, cam_b) in self._edges

    def all_edges(self) -> list[tuple[str, str, EdgeConfig]]:
        """Return all explicitly configured edges as (cam_a, cam_b, config)."""
        result = []
        seen = set()
        for (a, b), cfg in self._edges.items():
            pair = tuple(sorted((a, b)))
            if pair not in seen:
                seen.add(pair)
                result.append((a, b, cfg))
        return result

    def summary(self) -> str:
        lines = [f"cam-graph: {len(self.cameras)} cameras, "
                 f"{len(self._edges)} configured edges"]
        for a, b, cfg in self.all_edges():
            lines.append(f"  {a}↔{b}: timing={cfg.timing} sim={cfg.reid_sim:.2f} "
                         f"tol={cfg.time_tolerance_s:.1f}s"
                         f"{' cam-model' if cfg.use_camera_model else ''}")
        return "\n".join(lines)


def _pair_key(a: str, b: str) -> tuple[str, str]:
    """Canonical order-independent pair key."""
    return tuple(sorted((a, b)))  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_cam_graph(
    path: str | Path | None,
    cameras: list[str],
    cli_defaults: dict[str, Any] | None = None,
) -> CamGraph:
    """Load a camera graph from a YAML file.

    If *path* is None or the file doesn't exist, returns a CamGraph with
    the default EdgeConfig applied uniformly to every camera pair.

    Parameters
    ----------
    path : path to YAML (or None for defaults)
    cameras : camera ids in this run
    cli_defaults : optional dict of CLI overrides that apply to all edges
                   (e.g. ``{"reid_sim": 0.50}`` from ``--sim 0.50``)
    """
    defaults = EdgeConfig(**(cli_defaults or {}))

    if path is None or not Path(path).exists():
        # Uniform defaults for all pairs
        graph = CamGraph(cameras=cameras, defaults=defaults)
        return graph

    raw = yaml.safe_load(Path(path).read_text()) or {}

    # --- merge file defaults with CLI overrides ---
    file_defaults: dict[str, Any] = raw.get("defaults", {})
    merged = {**file_defaults, **(cli_defaults or {})}
    defaults = EdgeConfig(**merged)

    graph = CamGraph(cameras=cameras, defaults=defaults)

    # --- per-camera overrides (stored for future use, e.g. per-cam fps) ---
    for cam_id, cam_cfg in raw.get("cameras", {}).items():
        if cam_id in cameras:
            # Extend EdgeConfig with per-camera fields if needed in future
            pass

    # --- per-edge overrides ---
    for entry in raw.get("edges", []):
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            cam_a, cam_b = str(entry[0]), str(entry[1])
            overrides = entry[2] if len(entry) > 2 and isinstance(entry[2], dict) else {}
        elif isinstance(entry, dict):
            pair = entry.get("pair", [])
            if len(pair) < 2:
                continue
            cam_a, cam_b = str(pair[0]), str(pair[1])
            overrides = {k: v for k, v in entry.items() if k != "pair"}
        else:
            continue

        if cam_a not in cameras or cam_b not in cameras:
            continue

        edge = _merge_edge(defaults, overrides)
        key = _pair_key(cam_a, cam_b)
        graph._edges[key] = edge  # noqa: SLF001

    return graph


def _merge_edge(defaults: EdgeConfig, overrides: dict[str, Any]) -> EdgeConfig:
    """Create an EdgeConfig by overriding specific fields from defaults."""
    d = {
        "timing": defaults.timing,
        "time_tolerance_s": defaults.time_tolerance_s,
        "max_transit_s": defaults.max_transit_s,
        "reid_sim": defaults.reid_sim,
        "use_camera_model": defaults.use_camera_model,
        "spatial_gate_px": defaults.spatial_gate_px,
        "spatial_merge_px": defaults.spatial_merge_px,
        "use_transition_windows": defaults.use_transition_windows,
        "stitch": defaults.stitch,
        "stitch_sim": defaults.stitch_sim,
    }
    d.update(overrides)
    return EdgeConfig(**d)
