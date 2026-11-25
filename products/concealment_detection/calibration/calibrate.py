"""Four-stage offline calibration with development-only selection and frozen evaluation."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import itertools
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import yaml

PRODUCT_DIR = Path(__file__).resolve().parents[1]
if str(PRODUCT_DIR) not in sys.path:
    sys.path.insert(0, str(PRODUCT_DIR))

from adapters import evaluate_clip_predictions, extract_mask_events, false_alarm_upper_bound_95
from calibration.dataset_utils import (
    build_provenance, canonical_hash, compute_empirical_quantiles, discover_retails_splits,
    load_sparse_pose_frames, parse_camera_id, partition_staged_pairs,
    propose_all_camera_zones, stratified_sample_files, zones_to_yaml_dict,
)
from concealment.state_machine import ConcealmentMonitor
from concealment.types import CameraZones, DetectorSettings, PoseFrame
from config import load_config

REPORTS_DIR = Path(__file__).resolve().parent / "reports"
DEFAULT_DATA_ROOT = Path(r"F:\IRIS\data\RetailS")
BALANCED_FA_WEIGHT = 0.035
WARNING_TIERS = ("high", "medium")
_GRID_POOL_STATE: tuple | None = None


def _init_grid_pool(camera_zones, staged_dev_pairs, tune_neg_files) -> None:
    global _GRID_POOL_STATE
    _GRID_POOL_STATE = (
        camera_zones,
        _preload_positive_split(staged_dev_pairs),
        _preload_negative_split(tune_neg_files),
    )


def _eval_grid_point(settings: DetectorSettings) -> dict:
    assert _GRID_POOL_STATE is not None
    camera_zones, staged_loaded, tune_neg_loaded = _GRID_POOL_STATE
    metrics = evaluate_settings_on_splits(settings, camera_zones, staged_loaded, tune_neg_loaded)
    return {"params": asdict(settings), "tuning_metrics": metrics}


def _stratified_sample(files: tuple[Path, ...], max_count: int) -> list[Path]:
    return stratified_sample_files(files, max_count)


def select_balanced_run(runs: list[dict]) -> dict:
    """Full-precision single-person recall - 0.035 * normal-train FA/hour.

    Canonical serialized settings break ties, independent of grid iteration order.
    Positive-split FP is reported separately, not substituted into this objective.
    """
    return min(runs, key=lambda r: (
        -(r["tuning_metrics"]["recall_single_person"]
          - BALANCED_FA_WEIGHT * r["tuning_metrics"]["fa_per_span_hour"]),
        json.dumps(r["params"], sort_keys=True, separators=(",", ":")),
    ))


def _preload_positive_split(pairs, gap_flush_s: float = 1.6):
    """Load all labeled clips, including normal clips, on the absolute mask timeline."""
    loaded = []
    for pose_path, gt_path in pairs:
        mask = np.load(gt_path).ravel()
        frames, span_s, _ = load_sparse_pose_frames(
            pose_path, gap_flush_s=gap_flush_s, frame_count=len(mask))
        gt_intervals = extract_mask_events(mask, fps=15.0)
        track_ids = {t.track_id for f in frames for t in f.tracks}
        loaded.append((pose_path.stem, parse_camera_id(pose_path.stem), frames,
                       gt_intervals, max(span_s, len(mask) / 15.0), len(track_ids)))
    return loaded


def _preload_negative_split(files, gap_flush_s: float = 1.6):
    loaded = []
    for pose_path in files:
        frames, span_s, active_s = load_sparse_pose_frames(pose_path, gap_flush_s=gap_flush_s)
        loaded.append((pose_path.stem, parse_camera_id(pose_path.stem), frames, span_s, active_s))
    return loaded


def evaluate_settings_on_splits(
    settings: DetectorSettings,
    camera_zones: dict[str, CameraZones],
    pos_clips: list[tuple[str, str, list[PoseFrame], list[tuple[float, float]], float, int]],
    neg_clips: list[tuple[str, str, list[PoseFrame], float, float]],
    tolerance_s: float = 2.5,
) -> dict:
    """One-to-one event recall, labeled-split FP, and separate normal-train FA rates.

    Also breaks hits and false alarms down by terminal stage and tier, and counts why
    sequences were discarded; warning recall re-matches without low-tier candidates.
    """
    tp_all = tot_all = tp_single = tot_single = pos_candidates_total = 0
    tp_single_warning = 0
    pos_fp = pos_normal_fp = pos_normal_clips = 0
    pos_hours = 0.0
    breakdown: dict[str, dict[str, dict[str, int]]] = {
        "by_terminal_stage": {}, "by_tier": {}, "by_stage_tier": {}}
    event_hits: list[list] = []
    reset_counts: Counter[str] = Counter()
    sequence_resets_in_gt: Counter[str] = Counter()

    def tally(cand, field: str) -> None:
        for group, key in (("by_terminal_stage", cand.terminal_stage), ("by_tier", cand.tier),
                           ("by_stage_tier", f"{cand.terminal_stage}/{cand.tier}")):
            row = breakdown[group].setdefault(key, {"tp": 0, "pos_fp": 0, "neg_fa": 0})
            row[field] += 1

    def collect_resets(mon) -> None:
        for (reason, stage), n in mon.reset_counts.items():
            reset_counts[f"{reason}/{stage}"] += n

    for stem, cam_id, frames, gt_intervals, dur_s, n_tracks in pos_clips:
        cz = camera_zones.get(cam_id, CameraZones(camera=cam_id, zone_free=True))
        mon = ConcealmentMonitor(cam_id, settings, cz)
        cands = [c for f in frames for c in mon.update(f).candidates]
        pos_candidates_total += len(cands)
        ev = evaluate_clip_predictions(cands, gt_intervals, dur_s, tolerance_s)
        warning = [c for c in cands if c.tier in WARNING_TIERS]
        ev_warning = evaluate_clip_predictions(warning, gt_intervals, dur_s, tolerance_s)
        for cand, gt_idx in zip(cands, ev["matched_gt"]):
            tally(cand, "pos_fp" if gt_idx is None else "tp")
            if gt_idx is not None:
                event_hits.append([stem, gt_idx])
        collect_resets(mon)
        for pts, _, reason, stage in mon.sequence_resets:
            if any(s - tolerance_s <= pts <= e + tolerance_s for s, e in gt_intervals):
                sequence_resets_in_gt[f"{reason}/{stage}"] += 1
        pos_fp += ev["fp_candidates"]
        pos_hours += dur_s / 3600.0
        if not gt_intervals:
            pos_normal_fp += ev["fp_candidates"]
            pos_normal_clips += 1
        tp_all += ev["tp_events"]
        tot_all += len(gt_intervals)
        if n_tracks == 1:
            tp_single += ev["tp_events"]
            tp_single_warning += ev_warning["tp_events"]
            tot_single += len(gt_intervals)

    neg_fa_all = neg_fa_warning = 0
    span_hours = active_hours = 0.0
    neg_fa_by_clip: dict[str, int] = {}
    for stem, cam_id, frames, span_s, active_s in neg_clips:
        cz = camera_zones.get(cam_id, CameraZones(camera=cam_id, zone_free=True))
        mon = ConcealmentMonitor(cam_id, settings, cz)
        cands = [c for f in frames for c in mon.update(f).candidates]
        for cand in cands:
            tally(cand, "neg_fa")
        collect_resets(mon)
        neg_fa_by_clip[stem] = len(cands)
        neg_fa_all += len(cands)
        neg_fa_warning += sum(c.tier in WARNING_TIERS for c in cands)
        span_hours += span_s / 3600.0
        active_hours += active_s / 3600.0
    for rows in breakdown.values():
        for row in rows.values():
            row["neg_fa_per_span_hour"] = row["neg_fa"] / max(span_hours, 1e-9)
    recall_single = tp_single / max(tot_single, 1)
    recall_single_warning = tp_single_warning / max(tot_single, 1)
    fa_rate = neg_fa_all / max(span_hours, 1e-9)
    warning_fa_rate = neg_fa_warning / max(span_hours, 1e-9)
    return {
        "tp_events": tp_all, "total_events": tot_all,
        "recall_all": tp_all / max(tot_all, 1),
        "tp_single_person": tp_single, "total_single_person": tot_single,
        "recall_single_person": recall_single,
        "tp_single_person_warning": tp_single_warning,
        "recall_single_person_warning": recall_single_warning,
        "objective": recall_single - BALANCED_FA_WEIGHT * fa_rate,
        "objective_warning": recall_single_warning - BALANCED_FA_WEIGHT * warning_fa_rate,
        "pos_candidates_emitted": pos_candidates_total,
        "pos_false_alarms": pos_fp, "pos_normal_clip_false_alarms": pos_normal_fp,
        "pos_normal_clips": pos_normal_clips, "pos_span_hours": pos_hours,
        "pos_fa_per_span_hour": pos_fp / max(pos_hours, 1e-9),
        "neg_false_alarms": neg_fa_all, "neg_warning_false_alarms": neg_fa_warning,
        "neg_span_hours": span_hours, "neg_active_hours": active_hours,
        "fa_per_span_hour": fa_rate,
        "warning_fa_per_span_hour": warning_fa_rate,
        "fa_per_active_hour": neg_fa_all / max(active_hours, 1e-9),
        "fa_upper_bound_95_per_span_hour": (
            false_alarm_upper_bound_95(neg_fa_all, span_hours) if span_hours > 0 else None),
        **breakdown,
        "event_hits": sorted(event_hits),
        "neg_fa_by_clip": neg_fa_by_clip,
        "reset_counts": dict(reset_counts),
        "sequence_resets_in_gt": dict(sequence_resets_in_gt),
    }


def _build_markdown_report(results: dict) -> str:
    provenance = results.get("provenance")
    lines = ["# Offline Calibration & Verification Report", ""]
    if not provenance:
        lines.append("UNVERIFIED legacy artifact: no settings/source/dataset provenance; "
                     "these results do not establish current-code performance.")
    else:
        lines.append(f"Run fingerprint: `{provenance['fingerprint']}`. Results apply only to "
                     "the serialized settings, zones, sources and dataset identities in this run; "
                     "older artifacts are not revalidated or rewritten.")
    lines.extend(["", "## Splits", json.dumps(results["splits"], indent=2), "",
                  "Staged development/test: frozen camera/class/person-count stratified 50/50. "
                  "Train negatives: camera/day-stratified sample. Real-world is final evaluation only.",
                  "", "## Zone mode"])
    for camera, zones in results["stage1_proposed_zones"]["cameras"].items():
        mode = "ZONE-FREE (not shelf-limited)" if zones["zone_free"] else "shelf-limited"
        lines.append(f"- {camera}: {mode}; {len(zones['zones'])} proposed polygons.")
    lines.extend(["", "## Empirical quantiles", "Eligible single-person positive GT only; "
                  "time-based production history and representative normal samples.",
                  "```json", json.dumps(results["stage2_empirical_quantiles"], indent=2), "```",
                  "", "## Development objective",
                  "single-person recall - 0.035 * normal-train FA/span-hour; canonical settings tie-break."])
    groups = [("Development", results["stage3_grid_search"]["selected_operating_points"], "tuning_metrics"),
              ("Frozen real-world / train holdout", results["stage4_locked_holdout"], "holdout_metrics"),
              ("Frozen staged test", results.get("stage4_staged_test", []), "holdout_metrics")]
    for title, rows, metric_key in groups:
        lines.extend(["", f"## {title}",
                      "| Point | Recall all | Recall single-person | Labeled-split FP | Normal labeled-clip FP | Train FA/span-hour | Train FA/active-hour |",
                      "|---|---|---|---|---|---|---|"])
        for row in rows:
            m = row[metric_key]
            lines.append(f"| {row['name']} | {m['tp_events']}/{m['total_events']} | "
                         f"{m['tp_single_person']}/{m['total_single_person']} | "
                         f"{m.get('pos_false_alarms', 'unreported')} | "
                         f"{m.get('pos_normal_clip_false_alarms', 'unreported')} | "
                         f"{m['fa_per_span_hour']:.3f} | {m['fa_per_active_hour']:.3f} |")
    selected = results["stage3_grid_search"]["selected_operating_points"]
    if len(selected) > 1:
        lines.extend(["", "## Exact recommended development-selected settings", "```yaml",
                      yaml.safe_dump({"detector": selected[1]["params"]}, sort_keys=True), "```"])
    return "\n".join(lines)


def run_calibration(
    data_root: Path = DEFAULT_DATA_ROOT,
    train_tune_sample_size: int = 36,
    train_holdout_sample_size: int = 36,
    zone_free: bool = False,
) -> dict:
    t0 = time.time()
    # Never overwrite historical artifacts. Each invocation has its own output directory.
    run_dir = REPORTS_DIR / f"run-{time.time_ns()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    splits = discover_retails_splits(data_root)
    staged_dev_pairs, staged_test_pairs = partition_staged_pairs(splits.staged_pairs)
    print(f"[Stage 0] Staged dev={len(staged_dev_pairs)}, test={len(staged_test_pairs)}")
    baseline_settings = load_config().detector

    print("[Stage 1] Proposing zones from development normal clips...")
    camera_zones = propose_all_camera_zones(
        splits.train_tune_files, max_files_per_camera=15, zone_free=zone_free)
    # Explicit mode also applies to cameras without a proposal; no hidden zone-free fallback.
    all_paths = list(splits.train_tune_files) + list(splits.train_holdout_files) + [
        p for p, _ in splits.staged_pairs + splits.realworld_pairs]
    for path in all_paths:
        camera = parse_camera_id(path.stem)
        camera_zones.setdefault(camera, CameraZones(camera, zone_free=zone_free))
    zones_yaml_dict = zones_to_yaml_dict(camera_zones)
    (run_dir / "proposed_zones.yaml").write_text(yaml.safe_dump(zones_yaml_dict), encoding="utf-8")

    print("[Stage 2] Development-only empirical quantiles...")
    quantiles = compute_empirical_quantiles(
        staged_dev_pairs, splits.train_tune_files, camera_zones,
        settings=baseline_settings, max_staged=250, max_train=24)
    staged_loaded = _preload_positive_split(staged_dev_pairs)
    tune_neg_files = _stratified_sample(splits.train_tune_files, train_tune_sample_size)
    holdout_neg_files = _stratified_sample(splits.train_holdout_files, train_holdout_sample_size)
    tune_neg_loaded = _preload_negative_split(tune_neg_files)
    baseline_tune = evaluate_settings_on_splits(baseline_settings, camera_zones, staged_loaded, tune_neg_loaded)

    grid_space = list(itertools.product(
        [.25, .32, .40], [130., 140., 150.], [.50, .60], [.25, .35, .50], [1.5, 2.5], [False, True]))
    print(f"[Stage 3] Evaluating {len(grid_space)} development configurations...")
    grid_settings = [
        replace(baseline_settings, min_track_s=.25, dwell_min_s=1.6,
                short_track_grace_s=grace, reach_excursion_torsos=reach,
                body_zone_radius_torsos=radius, max_conceal_elbow_deg=elbow,
                min_hold_s=hold, hidden_s=.40, cooldown_s=8., emit_low_tier=low)
        for reach, elbow, radius, hold, grace, low in grid_space
    ]
    main_file = getattr(sys.modules.get("__main__"), "__file__", None)
    use_pool = (
        len(grid_settings) > 1
        and evaluate_settings_on_splits.__module__.endswith("calibrate")
        and bool(main_file)
        and main_file != "<stdin>"
    )
    if use_pool:
        max_workers = min(max(1, (os.cpu_count() or 4) - 1), 8, len(grid_settings))
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_init_grid_pool,
            initargs=(camera_zones, staged_dev_pairs, tune_neg_files),
        ) as pool:
            all_grid_runs = list(pool.map(_eval_grid_point, grid_settings))
    else:
        all_grid_runs = [
            {"params": asdict(s), "tuning_metrics": evaluate_settings_on_splits(s, camera_zones, staged_loaded, tune_neg_loaded)}
            for s in grid_settings
        ]
    balanced = select_balanced_run(all_grid_runs)
    # Deterministic tie-breaking for the other frozen representatives too.
    ranked = sorted(all_grid_runs, key=lambda r: json.dumps(r["params"], sort_keys=True))
    high = max(ranked, key=lambda r: (r["tuning_metrics"]["recall_all"],
               r["tuning_metrics"]["recall_single_person"], -r["tuning_metrics"]["fa_per_span_hour"]))
    viable = [r for r in ranked if r["tuning_metrics"]["recall_single_person"] >= .45] or ranked
    strict = min(viable, key=lambda r: (r["tuning_metrics"]["fa_per_span_hour"],
                                       -r["tuning_metrics"]["recall_single_person"]))
    selected_points = [
        {"name": "Baseline (Current Config)", "params": asdict(baseline_settings), "tuning_metrics": baseline_tune},
        {"name": "Recommended Balanced (Development Selected)", **balanced},
        {"name": "High-Sensitivity (Max Recall)", **high},
        {"name": "Strict Low-FA (Conservative)", **strict},
    ]
    for point in selected_points:
        point["settings_sha256"] = canonical_hash(point["params"])
    # Persist the complete search and frozen choice BEFORE any holdout evaluation.
    search = {
        "objective": {"formula": "recall_single_person - weight * fa_per_span_hour",
                      "weight": BALANCED_FA_WEIGHT, "tie_break": "canonical serialized settings ascending"},
        "total_configs_evaluated": len(all_grid_runs), "all_runs": all_grid_runs,
        "selected_operating_points": selected_points,
    }
    (run_dir / "frozen_selection.json").write_text(json.dumps(search, indent=2), encoding="utf-8")

    print("[Stage 4] Final evaluation of frozen operating points; no reselection...")
    realworld_loaded = _preload_positive_split(splits.realworld_pairs)
    staged_test_loaded = _preload_positive_split(staged_test_pairs)
    holdout_neg_loaded = _preload_negative_split(holdout_neg_files)
    holdout_results, staged_test_results = [], []
    for point in selected_points:
        settings = DetectorSettings(**point["params"])
        holdout_results.append({**point, "holdout_metrics": evaluate_settings_on_splits(
            settings, camera_zones, realworld_loaded, holdout_neg_loaded)})
        staged_test_results.append({**point, "holdout_metrics": evaluate_settings_on_splits(
            settings, camera_zones, staged_test_loaded, [])})

    datasets = {"train_development": splits.train_tune_files,
                "train_tune_evaluated": tune_neg_files, "train_holdout_evaluated": holdout_neg_files,
                "staged_development": [p for pair in staged_dev_pairs for p in pair],
                "staged_test": [p for pair in staged_test_pairs for p in pair],
                "realworld_final": [p for pair in splits.realworld_pairs for p in pair]}
    provenance = build_provenance(DetectorSettings(**balanced["params"]), camera_zones, datasets)
    summary = {
        "data_root": str(data_root), "elapsed_s": round(time.time() - t0, 2),
        "provenance": provenance, "output_dir": str(run_dir),
        "splits": {
            "train_tune_days": list(splits.train_tune_days), "train_holdout_days": list(splits.train_holdout_days),
            "train_tune_files": len(splits.train_tune_files), "train_tune_sampled_files": len(tune_neg_files),
            "train_tune_span_hours": baseline_tune["neg_span_hours"],
            "train_holdout_files": len(splits.train_holdout_files),
            "train_holdout_sampled_files": len(holdout_neg_files),
            "train_holdout_span_hours": holdout_results[0]["holdout_metrics"]["neg_span_hours"],
            "staged_clips": len(staged_loaded), "staged_test_clips": len(staged_test_loaded),
            "staged_events": baseline_tune["total_events"],
            "staged_single_person_events": baseline_tune["total_single_person"],
            "realworld_clips": len(realworld_loaded),
            "realworld_events": holdout_results[0]["holdout_metrics"]["total_events"],
            "realworld_single_person_events": holdout_results[0]["holdout_metrics"]["total_single_person"],
        },
        "stage1_proposed_zones": zones_yaml_dict, "stage2_empirical_quantiles": quantiles,
        "stage3_grid_search": search, "stage4_locked_holdout": holdout_results,
        "stage4_staged_test": staged_test_results,
    }
    (run_dir / "calibration_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / "CALIBRATION_REPORT.md").write_text(_build_markdown_report(summary), encoding="utf-8")
    print(f"[Done] Reports written to {run_dir}")
    return summary


if __name__ == "__main__":
    run_calibration()
