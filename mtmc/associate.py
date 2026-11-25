"""Cross-camera tracklet association.

Combines appearance (Re-ID cosine similarity) with a time gate: AICity22
cameras are near-synchronous, so the same vehicle appears in different cameras
at almost the same wall-clock time. Two tracklets can only belong to the same
global identity if their time intervals overlap within a tolerance, and each
global identity may hold at most one tracklet per camera.

Output: global_id -> list of (cam_id, local_track_id) memberships.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .cameramodel import CameraLinkModel, TrackletObs
from .camgraph import CamGraph
from .config import PipelineConfig


@dataclass
class TrackletWithEmbedding:
    cam_id: str
    track_id: int
    start_s: float   # wall-clock start (seconds)
    end_s: float     # wall-clock end
    embedding: np.ndarray
    frames: list[int] = field(default_factory=list)     # local frame numbers
    times_s: np.ndarray | None = None                   # global wall time per frame
    boxes: np.ndarray | None = None                     # Nx4 [x,y,w,h]


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _time_compatible(a: TrackletWithEmbedding, b: TrackletWithEmbedding, tol: float) -> bool:
    """True if the two tracklets could be the same vehicle timeline-wise."""
    overlap = min(a.end_s, b.end_s) - max(a.start_s, b.start_s)
    if overlap >= 0:
        return True  # intervals overlap in wall time
    return -overlap <= tol  # small gap is fine (missed detections)


def _to_obs(t: TrackletWithEmbedding) -> TrackletObs:
    return TrackletObs(
        cam_id=t.cam_id,
        track_id=t.track_id,
        frames=t.frames,
        boxes=t.boxes if t.boxes is not None else np.zeros((len(t.frames), 4)),
        start_s=t.start_s,
        end_s=t.end_s,
        times_s=t.times_s,
    )


def _camera_conflict(uf: UnionFind, n: int, tracklets: list, root: int, cam: str) -> bool:
    """True if the component at root already holds a tracklet from cam."""
    return any(
        uf.find(i) == root and tracklets[i].cam_id == cam for i in range(n)
    )


def associate(
    tracklets: list[TrackletWithEmbedding],
    cfg: PipelineConfig,
    camera_model: CameraLinkModel | None = None,
    cam_graph: CamGraph | None = None,
) -> dict[int, list[tuple[str, int]]]:
    """Merge tracklets into global identities.

    Returns {global_id: [(cam_id, local_track_id), ...]}.
    Unmatched tracklets get their own singleton global id.

    When *cam_graph* is provided, each camera pair gets its own association
    config (timing, sim threshold, camera model opt-in). Without a graph,
    global ``cfg`` fields are used for all pairs.

    Two-tier merging when a CameraLinkModel is available for a pair:

    Tier 1 — spatial proof. Two tracklets that overlap in wall time and
    project (via homography) to the same ground point merge unconditionally.

    Tier 2 — appearance. Remaining pairs merge by Re-ID cosine similarity
    above the per-edge threshold.
    """
    n = len(tracklets)
    if n == 0:
        return {}
    uf = UnionFind(n)
    obs: dict[int, TrackletObs] | None = None
    if camera_model is not None:
        obs = {i: _to_obs(t) for i, t in enumerate(tracklets)}

    # ---- per-pair candidate collection ------------------------------------
    spatial_pairs: list[tuple[float, int, int]] = []
    appearance_pairs: list[tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = tracklets[i], tracklets[j]
            if a.cam_id == b.cam_id:
                continue

            # Resolve per-edge config: graph edge > global cfg fallback
            if cam_graph is not None:
                edge = cam_graph.edge(a.cam_id, b.cam_id)
                tol = edge.time_tolerance_s
                sim_thr = edge.reid_sim
                use_cm = edge.use_camera_model
                sp_gate = edge.spatial_gate_px
                sp_merge = edge.spatial_merge_px
            else:
                tol = cfg.time_overlap_tolerance_s
                sim_thr = cfg.reid_sim_threshold
                use_cm = camera_model is not None
                sp_gate = cfg.spatial_gate_px
                sp_merge = cfg.spatial_merge_px

            # Use per-edge camera model only if the edge opts in
            # AND a camera model is actually available
            pair_has_cm = use_cm and camera_model is not None

            if pair_has_cm:
                oa, ob = obs[i], obs[j]
                if not camera_model.temporal_gate(oa, ob, tol):
                    continue
                d = camera_model.spatial_distance(oa, ob)
                if d is not None and d <= sp_merge:
                    spatial_pairs.append((d, i, j))
                    continue  # geometric proof: no appearance needed
                if d is not None and d > sp_gate:
                    continue  # spatially impossible: never compare
                if a.embedding is not None and b.embedding is not None:
                    sim = float(np.dot(a.embedding, b.embedding))
                    if sim >= sim_thr:
                        appearance_pairs.append((sim, i, j))
                else:
                    appearance_pairs.append((1.0, i, j))
            elif _time_compatible(a, b, tol):
                if a.embedding is not None and b.embedding is not None:
                    sim = float(np.dot(a.embedding, b.embedding))
                    if sim >= sim_thr:
                        appearance_pairs.append((sim, i, j))
                else:
                    appearance_pairs.append((1.0, i, j))

    # Spatial-proof merges first (strongest evidence), then appearance merges.
    for kind, key in ((spatial_pairs, 0), (appearance_pairs, 1)):
        order = sorted(kind, key=lambda t: t[key])
        for _, i, j in order:
            ri, rj = uf.find(i), uf.find(j)
            if ri == rj:
                continue
            if _camera_conflict(uf, n, tracklets, ri, tracklets[j].cam_id) or \
               _camera_conflict(uf, n, tracklets, rj, tracklets[i].cam_id):
                continue
            uf.union(i, j)

    # Group by root.
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(uf.find(i), []).append(i)

    result: dict[int, list[tuple[str, int]]] = {}
    for gid, members in enumerate(groups.values()):
        result[gid] = [(tracklets[i].cam_id, tracklets[i].track_id) for i in sorted(members)]
    return result
