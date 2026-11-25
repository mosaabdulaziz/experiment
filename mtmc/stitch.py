"""Same-camera tracklet stitching.

The IoU tracker fragments vehicles (occlusion, detection gaps): one physical
vehicle ends up as 2+ tracklets in the SAME camera. Each fragment emits its
own boxes (false positives) and — because a global identity may hold only one
tracklet per camera — fragments block correct cross-camera merges.

Stitching merges fragments of the same vehicle within a camera. Same-camera
fragments share viewpoint and lighting, so the Re-ID embedding is a reliable
signal here (unlike across cameras), and temporal/spatial continuity add
cheap, strong priors:

  * the fragments' time intervals must be near each other (gap or short
    overlap, never long co-existence);
  * at any overlapping frame their boxes must not belong to two different
    vehicles (small IoU conflict);
  * embedding similarity above a generous same-camera threshold.

Stitched tracklets replace the fragments before cross-camera association.
"""
from __future__ import annotations

import numpy as np

from .config import PipelineConfig
from .tracker import Tracklet, iou_matrix


def stitch_camera(
    cam_id: str,
    tracklets: list[Tracklet],
    embeddings: dict[int, np.ndarray],
    cfg: PipelineConfig,
) -> tuple[list[Tracklet], dict[int, int]]:
    """Stitch fragments of the same vehicle within one camera.

    Returns (stitched_tracklets, old_track_id -> new_track_id).
    """
    n = len(tracklets)
    if n < 2:
        return tracklets, {t.track_id: t.track_id for t in tracklets}

    # Candidate pairs: time-compatible + same camera + high embedding sim.
    pairs = []
    for i in range(n):
        a = tracklets[i]
        ea = embeddings.get(a.track_id)
        if ea is None or ea.shape[0] < 2:
            continue
        for j in range(i + 1, n):
            b = tracklets[j]
            eb = embeddings.get(b.track_id)
            if eb is None or eb.shape[0] < 2:
                continue
            if not _time_compatible(a, b, cfg):
                continue
            sim = float(np.dot(ea, eb))
            if sim >= cfg.stitch_sim_threshold:
                pairs.append((sim, i, j))
    pairs.sort(key=lambda t: -t[0])

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for sim, i, j in pairs:
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        if not _spatial_compatible(tracklets[i], tracklets[j], cfg):
            continue
        union(i, j)

    # Group fragments by root and merge.
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    stitched: list[Tracklet] = []
    mapping: dict[int, int] = {}
    for members in groups.values():
        if len(members) == 1:
            t = tracklets[members[0]]
            stitched.append(t)
            mapping[t.track_id] = t.track_id
            continue
        # merge: sort by frame, concatenate (fragments are time-ordered)
        ordered = sorted(members, key=lambda i: tracklets[i].start_frame)
        base = tracklets[ordered[0]]
        new = Tracklet(cam_id=cam_id, track_id=base.track_id)
        for idx in ordered:
            t = tracklets[idx]
            for f, box in zip(t.frames, t.boxes):
                new.add(f, box)
        stitched.append(new)
        for idx in ordered:
            mapping[tracklets[idx].track_id] = new.track_id
    return stitched, mapping


def _time_compatible(a: Tracklet, b: Tracklet, cfg: PipelineConfig) -> bool:
    """Fragments of one vehicle: near in time, not long co-existing."""
    gap = max(a.start_frame, b.start_frame) - min(a.end_frame, b.end_frame)
    if gap <= 0:
        # overlapping fragments: only merge if the overlap is tiny (a few
        # frames of double detection around an occlusion boundary)
        overlap = min(a.end_frame, b.end_frame) - max(a.start_frame, b.start_frame)
        return overlap <= cfg.stitch_max_overlap_frames
    return gap <= cfg.stitch_max_gap_frames


def _spatial_compatible(a: Tracklet, b: Tracklet, cfg: PipelineConfig) -> bool:
    """At overlapping frames, the boxes must not belong to different vehicles."""
    common = set(a.frames) & set(b.frames)
    if not common:
        return True
    fa = a.frames.index if hasattr(a.frames, "index") else None
    ious = []
    for f in common:
        i_a = a.frames.index(f)
        i_b = b.frames.index(f)
        ious.append(iou_matrix(a.boxes[i_a][None], b.boxes[i_b][None])[0, 0])
    # A large IoU at any shared frame means they are the same object, not two
    # different ones — that's compatible. Two distinct vehicles would show
    # small/zero IoU at the shared frame. So: reject only if they overlap in
    # time with a small IoU (two different vehicles side by side).
    return all(iou >= cfg.stitch_min_overlap_iou for iou in ious)
