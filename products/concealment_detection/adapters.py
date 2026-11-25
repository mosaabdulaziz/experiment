"""Offline adapters for RetailS / PoseLift archives and event-level evaluation metrics."""
from __future__ import annotations

from collections import defaultdict
import math
import numpy as np

from concealment.types import COCO_KEYPOINTS, CandidateEvent, PoseFrame, TrackObservation


def load_retails_pose_json(
    raw_json: dict,
    camera: str = "cam-1",
    fps: float = 15.0,
    shape: tuple[int, int] = (1920, 1080),
    swap_yx: bool = True,
    frame_count: int | None = None,
) -> list[PoseFrame]:
    """Convert a RetailS / PoseLift JSON dict into a chronological list of `PoseFrame`s."""
    by_frame: dict[int, list[TrackObservation]] = defaultdict(list)
    min_f: int | None = None
    max_f: int | None = None

    for pid_str, frame_dict in (raw_json or {}).items():
        if not isinstance(frame_dict, dict):
            continue
        track_id = int(pid_str)
        for f_str, entry in frame_dict.items():
            f_idx = int(f_str)
            if min_f is None or f_idx < min_f:
                min_f = f_idx
            if max_f is None or f_idx > max_f:
                max_f = f_idx

            flat = entry.get("keypoints") if isinstance(entry, dict) else entry
            if not flat or len(flat) < 3 * COCO_KEYPOINTS:
                continue

            kps: list[tuple[float, float, float]] = []
            valid_xs: list[float] = []
            valid_ys: list[float] = []
            for j in range(COCO_KEYPOINTS):
                c0 = float(flat[3 * j])
                c1 = float(flat[3 * j + 1])
                conf = float(flat[3 * j + 2])
                if swap_yx:
                    x, y = c1, c0
                else:
                    x, y = c0, c1
                kps.append((x, y, conf))
                if conf >= 0.05:
                    valid_xs.append(x)
                    valid_ys.append(y)

            if valid_xs and valid_ys:
                x1, x2 = min(valid_xs), max(valid_xs)
                y1, y2 = min(valid_ys), max(valid_ys)
                pad_x = max(10.0, 0.12 * (x2 - x1))
                pad_y = max(12.0, 0.08 * (y2 - y1))
                box = (x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y)
            else:
                box = (0.0, 0.0, 0.0, 0.0)

            by_frame[f_idx].append(
                TrackObservation(
                    track_id=track_id,
                    box=box,
                    kps=tuple(kps),
                    has_bag=False,
                    bag_box=None,
                )
            )

    max_f = max(max_f if max_f is not None else -1, (frame_count or 0) - 1)
    fps_val = max(float(fps), 1e-6)
    frames: list[PoseFrame] = []
    for f_idx in range(0, max_f + 1):
        obs_list = sorted(by_frame.get(f_idx, []), key=lambda t: t.track_id)
        frames.append(
            PoseFrame(
                camera=camera,
                frame_num=f_idx,
                pts=f_idx / fps_val,
                shape=shape,
                tracks=tuple(obs_list),
                observed=True,
            )
        )
    return frames


def extract_mask_events(
    mask: np.ndarray,
    fps: float = 15.0,
) -> list[tuple[float, float]]:
    """Extract `(start_pts, end_pts)` positive event intervals in seconds from a binary frame mask."""
    arr = np.asarray(mask).astype(int).ravel()
    if arr.size == 0:
        return []
    fps_val = max(float(fps), 1e-6)
    intervals: list[tuple[float, float]] = []
    in_run = False
    start_idx = 0
    for idx, val in enumerate(arr):
        if val > 0 and not in_run:
            in_run = True
            start_idx = idx
        elif val <= 0 and in_run:
            in_run = False
            intervals.append((start_idx / fps_val, idx / fps_val))
    if in_run:
        intervals.append((start_idx / fps_val, arr.size / fps_val))
    return intervals


def evaluate_clip_predictions(
    candidates: list[CandidateEvent],
    gt_intervals: list[tuple[float, float]],
    duration_s: float,
    tolerance_s: float = 2.0,
) -> dict:
    """Match predicted `CandidateEvent`s against ground-truth intervals and compute recall and FA/h."""
    def sequence_start(cand: CandidateEvent) -> float:
        # Only the current REACH -> AT_BODY sequence extends the emission point.
        # DWELL can be arbitrarily old and is not concealment evidence.
        reach = None
        start = cand.pts
        previous = -math.inf
        for transition in cand.stage_timeline:
            t = transition.entered_pts
            if not math.isfinite(t) or t < previous or t > cand.pts:
                return cand.pts
            previous = t
            if transition.stage in ("IDLE", "DWELL", "COOLDOWN"):
                reach, start = None, cand.pts
            elif transition.stage == "REACH":
                reach, start = t, cand.pts
            elif transition.stage == "AT_BODY":
                start = reach if reach is not None else t
        return start

    order = sorted(range(len(candidates)), key=lambda i: (
        candidates[i].pts, candidates[i].track_id, repr(candidates[i])))
    ordered = [candidates[i] for i in order]
    intervals = sorted(gt_intervals)
    tolerance = max(0.0, tolerance_s)
    edges = [
        [idx for idx, (s, e) in enumerate(intervals)
         if sequence_start(c) <= e + tolerance and c.pts >= s - tolerance]
        for c in ordered
    ]
    # Deterministic augmenting paths give maximum cardinality one-to-one matching.
    owners: dict[int, int] = {}

    def augment(candidate_idx: int, visited: set[int]) -> bool:
        for gt_idx in edges[candidate_idx]:
            if gt_idx in visited:
                continue
            visited.add(gt_idx)
            if gt_idx not in owners or augment(owners[gt_idx], visited):
                owners[gt_idx] = candidate_idx
                return True
        return False

    for candidate_idx in range(len(ordered)):
        augment(candidate_idx, set())
    tp_events = len(owners)
    fp_candidates = len(candidates) - tp_events
    # `intervals` is sorted; report GT indices against the caller's order.
    sorted_to_input_gt = sorted(range(len(gt_intervals)), key=lambda i: gt_intervals[i])
    matched_gt: list[int | None] = [None] * len(candidates)
    for gt_idx, candidate_idx in owners.items():
        matched_gt[order[candidate_idx]] = sorted_to_input_gt[gt_idx]
    fn_events = max(0, len(gt_intervals) - tp_events)
    recall = tp_events / len(gt_intervals) if gt_intervals else 1.0
    hours = max(float(duration_s) / 3600.0, 1e-9)
    fa_per_hour = fp_candidates / hours

    return {
        "tp_events": tp_events,
        "fn_events": fn_events,
        "fp_candidates": fp_candidates,
        "recall": recall,
        "false_alarms_per_hour": fa_per_hour,
        "matched": [g is not None for g in matched_gt],
        "matched_gt": matched_gt,
    }


def false_alarm_upper_bound_95(
    false_alarms: int,
    camera_hours: float,
) -> float:
    """Return the 95% Poisson upper confidence bound on false alarms per camera-hour."""
    if camera_hours <= 0.0:
        raise ValueError("camera_hours must be > 0")
    k = max(0, int(false_alarms))
    if k == 0:
        lambda_95 = -math.log(0.05)
    else:
        # Wilson-Hilferty chi-square quantile approximation for 2*(k + 1) degrees of freedom at p=0.95
        v = 2.0 * (k + 1)
        z_95 = 1.6448536269514722
        term = 1.0 - 2.0 / (9.0 * v) + z_95 * math.sqrt(2.0 / (9.0 * v))
        lambda_95 = 0.5 * v * (term ** 3)
    return lambda_95 / float(camera_hours)
