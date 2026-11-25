"""Pure, dependency-free helpers for selecting chat frames across streams.

Kept out of stream_manager.py (which imports redis) so the local backend can
reuse them without pulling in Redis. stream_manager re-exports these for
backward compatibility.
"""


def _stride_pick(seq: list, k: int) -> list:
    """Evenly pick up to k items, preserving first and last. k<=0 -> []."""
    if k <= 0:
        return []
    n = len(seq)
    if n <= k:
        return list(seq)
    if k == 1:
        return [seq[-1]]
    step = (n - 1) / (k - 1)
    return [seq[round(i * step)] for i in range(k)]


def recent_strided(snapshot: list, size: int, step: int, with_offsets: bool = False) -> list:
    """From an oldest->newest ``(offset, jpeg)`` snapshot, return up to ``size``
    items taken from the recent end, spaced ~``step`` frames apart, oldest->newest.

    Looks back over the last ``size * step`` frames and stride-picks ``size`` of
    them (first+last of that window preserved). ``step`` below 1 is treated as 1.
    Returns JPEGs by default; with ``with_offsets`` returns the ``(offset, jpeg)``
    pairs so the caller can keep each frame's timestamp.
    """
    window_len = max(size * max(int(step), 1), 1)
    window = snapshot[-window_len:]
    if with_offsets:
        return _stride_pick(window, size)
    return _stride_pick([jpeg for _off, jpeg in window], size)


def frames_near_offset(snapshot: list, target: float, padding: float, count: int) -> list:
    """From an oldest->newest (offset, jpeg) snapshot, return up to `count` JPEGs whose
    offset falls within [target-padding, target+padding], evenly spread via _stride_pick."""
    window = [(o, j) for o, j in snapshot if target - padding <= o <= target + padding]
    return _stride_pick([j for _o, j in window], count)


def select_chat_frames(buffers: dict, max_images: int) -> tuple[dict, list]:
    """Cap total images across cameras for one chat turn.

    Drops empty cameras, splits a per-camera budget evenly, and stride-picks
    each camera's frames (first+last preserved) so the long sampling window is
    kept. Returns (per_camera_frames, included_ids) in input order.
    """
    non_empty = [(cid, f) for cid, f in buffers.items() if f]
    if not non_empty:
        return {}, []
    per_cam = max(1, int(max_images) // len(non_empty))
    out: dict = {}
    ids: list = []
    for cid, frames in non_empty:
        out[cid] = _stride_pick(frames, per_cam)
        ids.append(cid)
    return out, ids
