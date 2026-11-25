"""Exact pure-NumPy MOT evaluator (HOTA, AssA, DetA, IDF1, MOTA, IDSW, Frag) + optional TrackEval."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from vema_tracker.baselines.base import TrackOutput
from vema_tracker.core.assignment import linear_assignment
from vema_tracker.core.costs import iou_matrix

HOTA_ALPHAS = np.round(np.arange(0.05, 0.96, 0.05), 2)


@dataclass
class SequenceMetrics:
    """Evaluation metrics for one sequence or an aggregated dataset."""

    name: str
    hota: float
    assa: float
    deta: float
    idf1: float
    mota: float
    idsw: int
    frag: int
    num_gt: int
    num_pred: int
    tp: int
    fp: int
    fn: int
    # Raw accumulators for exact dataset-level HOTA/IDF1 aggregation
    _alpha_tp: np.ndarray = field(default_factory=lambda: np.zeros(len(HOTA_ALPHAS), dtype=np.float64), repr=False)
    _alpha_fp: np.ndarray = field(default_factory=lambda: np.zeros(len(HOTA_ALPHAS), dtype=np.float64), repr=False)
    _alpha_fn: np.ndarray = field(default_factory=lambda: np.zeros(len(HOTA_ALPHAS), dtype=np.float64), repr=False)
    _alpha_ass_num: np.ndarray = field(default_factory=lambda: np.zeros(len(HOTA_ALPHAS), dtype=np.float64), repr=False)
    _idtp: int = field(default=0, repr=False)

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "name": self.name,
            "HOTA": round(self.hota, 3),
            "AssA": round(self.assa, 3),
            "DetA": round(self.deta, 3),
            "IDF1": round(self.idf1, 3),
            "MOTA": round(self.mota, 3),
            "IDSW": int(self.idsw),
            "Frag": int(self.frag),
            "NumGT": int(self.num_gt),
            "NumPred": int(self.num_pred),
        }


def trackeval_available() -> bool:
    try:
        import trackeval  # type: ignore  # noqa: F401
        return True
    except ImportError:
        return False


def evaluate_sequence(
    name: str,
    gt_by_frame: dict[int, list[tuple[int, np.ndarray]]],
    pred_by_frame: dict[int, list[TrackOutput]],
    num_frames: int | None = None,
    iou_thresh: float = 0.5,
    alphas: np.ndarray | None = None,
) -> SequenceMetrics:
    """Compute exact CLEAR MOT, IDF1, and HOTA metrics on a single sequence."""
    all_frames = sorted(set(gt_by_frame.keys()) | set(pred_by_frame.keys()))
    if num_frames is not None and num_frames > 0:
        max_f = max(max(all_frames, default=0), num_frames)
        frame_ids = list(range(1, max_f + 1))
    else:
        frame_ids = all_frames

    # Precompute per-frame IDs and IoU matrices once
    frame_cache: list[tuple[list[int], list[int], np.ndarray]] = []
    gt_counts: dict[int, int] = {}
    pred_counts: dict[int, int] = {}

    for fid in frame_ids:
        g_list = gt_by_frame.get(fid, [])
        p_list = pred_by_frame.get(fid, [])
        g_ids = [int(gid) for gid, _ in g_list]
        p_ids = [int(p.track_id) for p in p_list]
        for gid in g_ids:
            gt_counts[gid] = gt_counts.get(gid, 0) + 1
        for pid in p_ids:
            pred_counts[pid] = pred_counts.get(pid, 0) + 1

        if g_list and p_list:
            g_boxes = np.stack([box for _, box in g_list], axis=0)
            p_boxes = np.stack([p.bbox for p in p_list], axis=0)
            ious = iou_matrix(g_boxes, p_boxes)
        else:
            ious = np.zeros((len(g_ids), len(p_ids)), dtype=np.float64)
        frame_cache.append((g_ids, p_ids, ious))

    num_gt = sum(gt_counts.values())
    num_pred = sum(pred_counts.values())

    # ------------------------------------------------------------------
    # 1. CLEAR MOT (MOTA, IDSW, Frag) & IDF1 contingency at iou_thresh
    # ------------------------------------------------------------------
    total_tp = 0
    total_fp = 0
    total_fn = 0
    idsw = 0
    frag = 0
    prev_gt_to_pred: dict[int, int] = {}
    last_matched_gt_to_pred: dict[int, int] = {}
    gt_was_tracked: set[int] = set()
    gt_currently_lost: set[int] = set()

    gt_id_list = sorted(gt_counts.keys())
    pred_id_list = sorted(pred_counts.keys())
    gt_idx_map = {gid: i for i, gid in enumerate(gt_id_list)}
    pred_idx_map = {pid: i for i, pid in enumerate(pred_id_list)}
    contingency = np.zeros((len(gt_id_list), len(pred_id_list)), dtype=np.int64)

    for g_ids, p_ids, ious in frame_cache:
        ng, npred = len(g_ids), len(p_ids)
        if ng == 0:
            total_fp += npred
            prev_gt_to_pred.clear()
            continue
        if npred == 0:
            total_fn += ng
            for gid in g_ids:
                if gid in gt_was_tracked:
                    gt_currently_lost.add(gid)
            prev_gt_to_pred.clear()
            continue

        matched_g: set[int] = set()
        matched_p: set[int] = set()
        p_id_to_col = {pid: c for c, pid in enumerate(p_ids)}

        # CLEAR MOT identity continuity preference from previous frame
        pairs: list[tuple[int, int]] = []
        for gi, gid in enumerate(g_ids):
            prev_pid = prev_gt_to_pred.get(gid)
            if prev_pid is not None and prev_pid in p_id_to_col:
                pi = p_id_to_col[prev_pid]
                if ious[gi, pi] >= iou_thresh:
                    pairs.append((gi, pi))
                    matched_g.add(gi)
                    matched_p.add(pi)

        rem_g = [gi for gi in range(ng) if gi not in matched_g]
        rem_p = [pi for pi in range(npred) if pi not in matched_p]
        if rem_g and rem_p:
            sub_ious = ious[np.ix_(rem_g, rem_p)]
            cost = 1.0 - sub_ious
            m_sub, _, _ = linear_assignment(cost, 1.0 - iou_thresh)
            for lg, lp in m_sub:
                pairs.append((rem_g[lg], rem_p[lp]))
                matched_g.add(rem_g[lg])
                matched_p.add(rem_p[lp])

        tp_f = len(pairs)
        total_tp += tp_f
        total_fp += npred - tp_f
        total_fn += ng - tp_f

        new_prev: dict[int, int] = {}
        for gi, pi in pairs:
            gid = g_ids[gi]
            pid = p_ids[pi]
            new_prev[gid] = pid
            contingency[gt_idx_map[gid], pred_idx_map[pid]] += 1

            prev_known = last_matched_gt_to_pred.get(gid)
            if prev_known is not None and prev_known != pid:
                idsw += 1
            if gid in gt_currently_lost:
                frag += 1
                gt_currently_lost.discard(gid)
            gt_was_tracked.add(gid)
            last_matched_gt_to_pred[gid] = pid

        for gi, gid in enumerate(g_ids):
            if gi not in matched_g and gid in gt_was_tracked:
                gt_currently_lost.add(gid)

        prev_gt_to_pred = new_prev

    mota = 100.0 * (1.0 - (total_fn + total_fp + idsw) / max(num_gt, 1))

    # Global bipartite match for IDF1
    if contingency.size > 0 and np.any(contingency > 0):
        max_c = float(np.max(contingency))
        id_cost = np.where(contingency > 0, max_c - contingency, np.inf)
        id_matches, _, _ = linear_assignment(id_cost, max_c - 1.0)
        idtp = int(sum(contingency[r, c] for r, c in id_matches))
    else:
        idtp = 0
    idf1 = 100.0 * (2.0 * idtp) / max(num_gt + num_pred, 1)

    # ------------------------------------------------------------------
    # 2. Exact HOTA / DetA / AssA across alpha thresholds
    # ------------------------------------------------------------------
    alpha_grid = HOTA_ALPHAS if alphas is None else np.asarray(alphas, dtype=np.float64)
    n_alphas = len(alpha_grid)
    alpha_tp = np.zeros(n_alphas, dtype=np.float64)
    alpha_fp = np.zeros(n_alphas, dtype=np.float64)
    alpha_fn = np.zeros(n_alphas, dtype=np.float64)
    alpha_ass_num = np.zeros(n_alphas, dtype=np.float64)

    for a_idx, alpha in enumerate(alpha_grid):
        cost_limit = 1.0 - float(alpha)
        pair_counts: dict[tuple[int, int], int] = {}
        tp_a = 0
        fp_a = 0
        fn_a = 0

        for g_ids, p_ids, ious in frame_cache:
            ng, npred = len(g_ids), len(p_ids)
            if ng == 0 or npred == 0:
                fp_a += npred
                fn_a += ng
                continue
            cost = 1.0 - ious
            matches, _, _ = linear_assignment(cost, cost_limit)
            tp_f = len(matches)
            tp_a += tp_f
            fp_a += npred - tp_f
            fn_a += ng - tp_f
            for gi, pi in matches:
                key = (g_ids[gi], p_ids[pi])
                pair_counts[key] = pair_counts.get(key, 0) + 1

        alpha_tp[a_idx] = tp_a
        alpha_fp[a_idx] = fp_a
        alpha_fn[a_idx] = fn_a

        ass_sum = 0.0
        for (gid, pid), tpa in pair_counts.items():
            fna = gt_counts[gid] - tpa
            fpa = pred_counts[pid] - tpa
            denom = tpa + fna + fpa
            if denom > 0:
                ass_sum += tpa * (tpa / denom)
        alpha_ass_num[a_idx] = ass_sum

    det_a_arr = np.where(
        (alpha_tp + alpha_fp + alpha_fn) > 0,
        alpha_tp / np.maximum(alpha_tp + alpha_fp + alpha_fn, 1e-12),
        0.0,
    )
    ass_a_arr = np.where(
        alpha_tp > 0,
        alpha_ass_num / np.maximum(alpha_tp, 1e-12),
        0.0,
    )
    hota_arr = np.sqrt(det_a_arr * ass_a_arr)

    return SequenceMetrics(
        name=name,
        hota=float(100.0 * np.mean(hota_arr)),
        assa=float(100.0 * np.mean(ass_a_arr)),
        deta=float(100.0 * np.mean(det_a_arr)),
        idf1=float(idf1),
        mota=float(mota),
        idsw=int(idsw),
        frag=int(frag),
        num_gt=int(num_gt),
        num_pred=int(num_pred),
        tp=int(total_tp),
        fp=int(total_fp),
        fn=int(total_fn),
        _alpha_tp=alpha_tp,
        _alpha_fp=alpha_fp,
        _alpha_fn=alpha_fn,
        _alpha_ass_num=alpha_ass_num,
        _idtp=idtp,
    )


def aggregate_metrics(results: list[SequenceMetrics], name: str = "OVERALL") -> SequenceMetrics:
    """Dataset-level aggregation over multiple sequences (matches TrackEval summation)."""
    if not results:
        return SequenceMetrics(name, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0)

    alpha_tp = np.sum([r._alpha_tp for r in results], axis=0)
    alpha_fp = np.sum([r._alpha_fp for r in results], axis=0)
    alpha_fn = np.sum([r._alpha_fn for r in results], axis=0)
    alpha_ass_num = np.sum([r._alpha_ass_num for r in results], axis=0)

    det_a_arr = np.where(
        (alpha_tp + alpha_fp + alpha_fn) > 0,
        alpha_tp / np.maximum(alpha_tp + alpha_fp + alpha_fn, 1e-12),
        0.0,
    )
    ass_a_arr = np.where(
        alpha_tp > 0,
        alpha_ass_num / np.maximum(alpha_tp, 1e-12),
        0.0,
    )
    hota_arr = np.sqrt(det_a_arr * ass_a_arr)

    total_gt = sum(r.num_gt for r in results)
    total_pred = sum(r.num_pred for r in results)
    total_tp = sum(r.tp for r in results)
    total_fp = sum(r.fp for r in results)
    total_fn = sum(r.fn for r in results)
    total_idsw = sum(r.idsw for r in results)
    total_frag = sum(r.frag for r in results)
    total_idtp = sum(r._idtp for r in results)

    mota = 100.0 * (1.0 - (total_fn + total_fp + total_idsw) / max(total_gt, 1))
    idf1 = 100.0 * (2.0 * total_idtp) / max(total_gt + total_pred, 1)

    return SequenceMetrics(
        name=name,
        hota=float(100.0 * np.mean(hota_arr)),
        assa=float(100.0 * np.mean(ass_a_arr)),
        deta=float(100.0 * np.mean(det_a_arr)),
        idf1=float(idf1),
        mota=float(mota),
        idsw=int(total_idsw),
        frag=int(total_frag),
        num_gt=int(total_gt),
        num_pred=int(total_pred),
        tp=int(total_tp),
        fp=int(total_fp),
        fn=int(total_fn),
        _alpha_tp=alpha_tp,
        _alpha_fp=alpha_fp,
        _alpha_fn=alpha_fn,
        _alpha_ass_num=alpha_ass_num,
        _idtp=total_idtp,
    )
