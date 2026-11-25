"""Development-only ablation of FSM settings knobs against the deployed detector settings.

Each variant is `replace(load_config().detector, **overrides)` evaluated on the frozen staged
development split plus a camera/day-stratified sample of tuning-day normal footage, with
zones proposed from tuning days only (as `calibrate.py` does). Staged test, PoseLift
real-world pairs and the negative holdout are never loaded: they are scored once, by the
calibration runner, after the adopted settings are frozen.

The unmodified ("baseline") variant is fingerprinted. Re-running with the same settings,
zones, data and preload must reproduce its metrics exactly, otherwise `BaselineMismatch`.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, fields, replace
import json
from pathlib import Path
import sys

import numpy as np

CAL_DIR = Path(__file__).resolve().parent
PRODUCT_DIR = CAL_DIR.parent
for p in (str(PRODUCT_DIR), str(CAL_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from calibration.calibrate import (  # noqa: E402
    BALANCED_FA_WEIGHT,
    DEFAULT_DATA_ROOT,
    _preload_negative_split,
    _preload_positive_split,
    _stratified_sample,
    evaluate_settings_on_splits,
)
from calibration.dataset_utils import (  # noqa: E402
    canonical_hash,
    discover_retails_splits,
    parse_camera_id,
    parse_train_day,
    partition_staged_pairs,
    propose_all_camera_zones,
    zones_to_yaml_dict,
)
from concealment.types import CameraZones, DetectorSettings  # noqa: E402
from config import load_config  # noqa: E402

OUT_DIR = CAL_DIR / "reports" / "demo"
ABLATION_NAME = "fsm_upgrade_ablation.json"
BASELINE_NAME = "fsm_upgrade_baseline.json"
MIN_GAP_FLUSH_S = 1.6
TOLERANCE_S = 2.5
MIN_DELTA_OBJECTIVE = 0.010
MIN_NET_EVENTS = 2
MIN_BOOTSTRAP_P = 0.9
MAX_WARNING_FA_CHANGE = 0.03
MAX_HALF_FA_CHANGE = 0.05

LOW = {"emit_low_tier": True}
LONG_LOSS = {"lost_s": 2.5, "reid_max_gap_s": 2.5}
UPGRADE_SETTING_FIELDS = {
    "sequence_max_gap_s", "insufficient_grace_s", "zone_persist_across_shelves",
    "zone_exit_grace_s", "reach_expiry_s", "partial_min_near_body_s",
    "reach_seeds_dwell", "reach_seed_max_speed_torsos_s", "at_body_requires_flex",
    "hold_flicker_grace_s", "fast_dropin_min_hold_s", "spatial_hidden_s",
    "exit_min_hidden_frac", "hand_transfer", "hand_transfer_meet_torsos", "cooldown_scope",
}

VARIANTS: dict[str, dict] = {
    "baseline": {},
    "low_tier_on": LOW,
    "gap_1.0": {"sequence_max_gap_s": 1.0},
    "gap_1.4": {"sequence_max_gap_s": 1.4},
    "quality_grace_0.3": {"insufficient_grace_s": 0.3},
    "quality_grace_0.6": {"insufficient_grace_s": 0.6},
    "zone_persist": {"zone_persist_across_shelves": True},
    "zone_exit_0.2": {"zone_exit_grace_s": 0.2},
    "zone_exit_0.4": {"zone_exit_grace_s": 0.4},
    "reach_expiry_3.5": {"reach_expiry_s": 3.5},
    "reach_expiry_5.0": {"reach_expiry_s": 5.0},
    "partial_near_0.2_low": {"partial_min_near_body_s": 0.2, **LOW},
    "partial_near_0.4_low": {"partial_min_near_body_s": 0.4, **LOW},
    "reach_seed_0.3_low": {"reach_seeds_dwell": True, "reach_seed_max_speed_torsos_s": 0.3,
                           **LOW},
    "reach_seed_0.6_low": {"reach_seeds_dwell": True, **LOW},
    "no_flex": {"at_body_requires_flex": False},
    "flicker_0.1": {"hold_flicker_grace_s": 0.1},
    "flicker_0.2": {"hold_flicker_grace_s": 0.2},
    "flicker_0.3": {"hold_flicker_grace_s": 0.3},
    "fast_dropin_0.1_low": {"fast_dropin_min_hold_s": 0.1, **LOW},
    "fast_dropin_0.2_low": {"fast_dropin_min_hold_s": 0.2, **LOW},
    "spatial_hidden_0.7": {"spatial_hidden_s": 0.7},
    "spatial_hidden_1.0": {"spatial_hidden_s": 1.0},
    "spatial_hidden_1.5": {"spatial_hidden_s": 1.5},
    "exit_0.5": {"exit_min_hidden_frac": 0.5},
    "exit_1.0": {"exit_min_hidden_frac": 1.0},
    "hand_transfer": {"hand_transfer": True},
    "hand_transfer_meet_0.5": {"hand_transfer": True, "hand_transfer_meet_torsos": 0.5},
    "cooldown_track_zone": {"cooldown_scope": "track_zone"},
    "cooldown_6": {"cooldown_s": 6.0},
    "cooldown_4": {"cooldown_s": 4.0},
    # Longer track loss changes the shared preload flush; run these two on their own.
    "long_loss": LONG_LOSS,
    "gap_2.0_long_loss": {"sequence_max_gap_s": 2.0, **LONG_LOSS},
}


class BaselineMismatch(RuntimeError):
    """The unmodified detector no longer reproduces its recorded metrics."""


def legacy_settings_hash(settings: DetectorSettings) -> str:
    values = asdict(settings)
    defaults = asdict(DetectorSettings())
    identity = {k: v for k, v in values.items() if k not in UPGRADE_SETTING_FIELDS}
    identity.update({k: values[k] for k in UPGRADE_SETTING_FIELDS
                     if values[k] != defaults[k]})
    return canonical_hash(identity)


def variant_settings(base: DetectorSettings, overrides: dict) -> DetectorSettings:
    unknown = set(overrides) - {f.name for f in fields(base)}
    if unknown:
        raise ValueError(f"unknown detector settings: {sorted(unknown)}")
    return replace(base, **overrides)


def preload_gap_flush_s(settings) -> float:
    # A flush frame shorter than a variant's track-loss window would cut its tracks early.
    return max(MIN_GAP_FLUSH_S, max(s.lost_s for s in settings) + 0.1)


def compare(base: dict, variant: dict, single_person_stems: set[str]) -> dict:
    def single(hits):
        return {tuple(h) for h in hits if h[0] in single_person_stems}

    base_hits, var_hits = single(base["event_hits"]), single(variant["event_hits"])
    gained = sorted(var_hits - base_hits)
    lost = sorted(base_hits - var_hits)
    base_warning_fa = base["warning_fa_per_span_hour"]
    return {
        "delta_objective": variant["objective"] - base["objective"],
        "delta_objective_warning": variant["objective_warning"] - base["objective_warning"],
        "delta_recall_single_person": (variant["recall_single_person"]
                                       - base["recall_single_person"]),
        "delta_fa_per_span_hour": variant["fa_per_span_hour"] - base["fa_per_span_hour"],
        "warning_fa_change_frac": (
            (variant["warning_fa_per_span_hour"] - base_warning_fa) / base_warning_fa
            if base_warning_fa > 0 else None),
        "events_gained": [list(h) for h in gained],
        "events_lost": [list(h) for h in lost],
        "net_events": len(gained) - len(lost),
    }


def _objective(hits: np.ndarray, gt: np.ndarray, fa: np.ndarray, span_h: np.ndarray):
    return (hits.sum(-1) / np.maximum(gt.sum(-1), 1)
            - BALANCED_FA_WEIGHT * fa.sum(-1) / np.maximum(span_h.sum(-1), 1e-9))


def bootstrap_p_improvement(base: dict, variant: dict, single_gt: dict[str, int],
                            neg_span_h: dict[str, float], n_boot: int = 2000,
                            seed: int = 0) -> float:
    """P(objective delta > 0) under paired resampling of single-person and negative clips."""
    if not single_gt or not neg_span_h:
        return 0.0
    pos, neg = sorted(single_gt), sorted(neg_span_h)

    def hits_per_clip(m):
        counts = Counter(h[0] for h in m["event_hits"] if h[0] in single_gt)
        return np.array([counts[s] for s in pos], dtype=float)

    gt = np.array([single_gt[s] for s in pos], dtype=float)
    span = np.array([neg_span_h[s] for s in neg])
    rng = np.random.default_rng(seed)
    pi = rng.integers(0, len(pos), (n_boot, len(pos)))
    ni = rng.integers(0, len(neg), (n_boot, len(neg)))

    def objective(m):
        fa = np.array([m["neg_fa_by_clip"][s] for s in neg], dtype=float)
        return _objective(hits_per_clip(m)[pi], gt[pi], fa[ni], span[ni])

    return float(np.mean(objective(variant) - objective(base) > 1e-12))


def split_half_fa_change(base: dict, variant: dict, neg_span_h: dict[str, float]) -> list[float]:
    """Fractional FA/hour change on two halves alternating within each camera/day stratum."""
    strata = sorted(neg_span_h, key=lambda s: (parse_camera_id(s), parse_train_day(s), s))
    changes = []
    for half in (strata[0::2], strata[1::2]):
        b = sum(base["neg_fa_by_clip"][s] for s in half)
        v = sum(variant["neg_fa_by_clip"][s] for s in half)
        changes.append((v - b) / b if b else (0.0 if v == 0 else float("inf")))
    return changes


def adoption_verdict(vs_baseline: dict) -> dict:
    warning = vs_baseline["warning_fa_change_frac"]
    checks = {
        "delta_objective": vs_baseline["delta_objective"] >= MIN_DELTA_OBJECTIVE - 1e-12,
        "net_events": vs_baseline["net_events"] >= MIN_NET_EVENTS,
        "bootstrap": vs_baseline["bootstrap_p_improvement"] >= MIN_BOOTSTRAP_P,
        "warning_fa": warning is not None and warning <= MAX_WARNING_FA_CHANGE + 1e-12,
        "split_half_fa": all(c <= MAX_HALF_FA_CHANGE + 1e-12
                             for c in vs_baseline["split_half_fa_change"]),
    }
    return {"adopt": all(checks.values()), "checks": checks}


def check_baseline(identity: dict, metrics: dict, path: Path) -> None:
    records = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    key = canonical_hash(identity)
    digest = canonical_hash(metrics)
    data_context = {k: v for k, v in identity.items()
                    if k not in ("settings_sha256", "legacy_settings_sha256")}
    same_data = [record for record in records.values()
                 if {k: v for k, v in record["identity"].items()
                     if k not in ("settings_sha256", "legacy_settings_sha256")}
                 == data_context]

    def compatible(record: dict) -> bool:
        stored = record["identity"]
        current_full = identity["settings_sha256"]
        stored_full = stored["settings_sha256"]
        current_legacy = identity.get("legacy_settings_sha256")
        stored_legacy = stored.get("legacy_settings_sha256")
        if current_legacy is not None:
            return current_legacy == (stored_legacy if stored_legacy is not None else stored_full)
        if stored_legacy is not None:
            return current_full in (stored_full, stored_legacy)
        return current_full == stored_full

    references = [record for record in same_data if compatible(record)]
    if records and not references:
        if same_data:
            raise BaselineMismatch("legacy detector settings changed from the pinned baseline")
        raise BaselineMismatch("unrecognized baseline data/preload identity; initialize an empty reference")
    for stored in references:
        if stored["metrics_sha256"] != digest:
            raise BaselineMismatch(
                f"baseline {key[:12]} changed: recorded {stored['metrics_sha256'][:12]}, "
                f"got {digest[:12]} (objective {stored['objective']:.4f} -> "
                f"{metrics['objective']:.4f})")
    if references:
        return
    records[key] = {"identity": identity, "metrics_sha256": digest,
                    "objective": metrics["objective"]}
    path.write_text(json.dumps(records, indent=2, sort_keys=True), encoding="utf-8")


_POOL_STATE: tuple | None = None


def _init_eval_pool(staged_dev, neg_files, zones, gap_flush_s, tolerance_s):
    global _POOL_STATE
    pos = _preload_positive_split(staged_dev, gap_flush_s=gap_flush_s)
    neg = _preload_negative_split(neg_files, gap_flush_s=gap_flush_s)
    _POOL_STATE = (zones, pos, neg, tolerance_s)


def _eval_variant_worker(item):
    name, s = item
    zones, pos, neg, tolerance_s = _POOL_STATE
    return name, evaluate_settings_on_splits(s, zones, pos, neg, tolerance_s)


def run(
    variant_names: list[str] | None = None,
    data_root: Path = DEFAULT_DATA_ROOT,
    out_dir: Path = OUT_DIR,
    neg_sample_size: int = 36,
    baseline_reference: Path | None = None,
) -> dict:
    names = list(dict.fromkeys(["baseline", *(variant_names or VARIANTS)]))
    missing = [n for n in names if n not in VARIANTS]
    if missing:
        raise ValueError(f"unknown variants: {missing}")
    base = load_config().detector
    settings = {n: variant_settings(base, VARIANTS[n]) for n in names}
    gap_flush_s = preload_gap_flush_s(settings.values())

    splits = discover_retails_splits(data_root)
    staged_dev, _staged_test = partition_staged_pairs(splits.staged_pairs)
    neg_files = _stratified_sample(splits.train_tune_files, neg_sample_size)
    zones = propose_all_camera_zones(splits.train_tune_files, max_files_per_camera=15,
                                     zone_free=False)
    for path in [*neg_files, *(p for p, _ in staged_dev)]:
        camera = parse_camera_id(path.stem)
        zones.setdefault(camera, CameraZones(camera, zone_free=False))
    pos = _preload_positive_split(staged_dev, gap_flush_s=gap_flush_s)
    neg = _preload_negative_split(neg_files, gap_flush_s=gap_flush_s)
    single_person_stems = {clip[0] for clip in pos if clip[5] == 1}

    identity = {
        "settings_sha256": canonical_hash(asdict(settings["baseline"])),
        "legacy_settings_sha256": legacy_settings_hash(settings["baseline"]),
        "zones_sha256": canonical_hash(zones_to_yaml_dict(zones)),
        "positives": sorted(p.stem for p, _ in staged_dev),
        "negatives": sorted(p.stem for p in neg_files),
        "gap_flush_s": gap_flush_s,
        "tolerance_s": TOLERANCE_S,
    }
    results = {}
    main_file = getattr(sys.modules.get("__main__"), "__file__", None)
    use_pool = (
        evaluate_settings_on_splits.__module__.endswith("calibrate")
        and len(settings) > 1
        and (len(pos) + len(neg)) > 10
        and bool(main_file)
        and main_file != "<stdin>"
    )
    if use_pool:
        with ProcessPoolExecutor(
            max_workers=min(6, len(settings)),
            initializer=_init_eval_pool,
            initargs=(staged_dev, neg_files, zones, gap_flush_s, TOLERANCE_S),
        ) as executor:
            for name, m in executor.map(_eval_variant_worker, list(settings.items())):
                results[name] = m
    else:
        for name, s in settings.items():
            results[name] = evaluate_settings_on_splits(s, zones, pos, neg, TOLERANCE_S)
    for name in names:
        m = results[name]
        print(f"{name:>32}: objective {m['objective']:+.4f} "
              f"(warning {m['objective_warning']:+.4f}) | single recall "
              f"{m['recall_single_person']:.3f} | FA/h {m['fa_per_span_hour']:.2f} "
              f"(warning {m['warning_fa_per_span_hour']:.2f})")
    out_dir.mkdir(parents=True, exist_ok=True)
    reference = baseline_reference or out_dir / BASELINE_NAME
    check_baseline(identity, results["baseline"], reference)
    single_gt = {clip[0]: len(clip[3]) for clip in pos if clip[5] == 1 and clip[3]}
    neg_span_h = {clip[0]: clip[3] / 3600.0 for clip in neg}

    def versus_baseline(name: str) -> dict:
        base, var = results["baseline"], results[name]
        return {**compare(base, var, single_person_stems),
                "bootstrap_p_improvement": bootstrap_p_improvement(base, var, single_gt,
                                                                   neg_span_h),
                "split_half_fa_change": split_half_fa_change(base, var, neg_span_h)}

    comparisons = {name: versus_baseline(name) for name in names}
    report = {
        "identity": identity,
        "variants": {
            name: {
                "overrides": VARIANTS[name],
                "settings_sha256": canonical_hash(asdict(settings[name])),
                "metrics": results[name],
                "vs_baseline": comparisons[name],
                "adoption": adoption_verdict(comparisons[name]),
            }
            for name in names
        },
    }
    (out_dir / ABLATION_NAME).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", action="append", choices=sorted(VARIANTS),
                        help="variant to evaluate (repeatable); default: all")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--baseline-reference", type=Path)
    args = parser.parse_args()
    run(args.variant, data_root=args.data_root, out_dir=args.out_dir,
        baseline_reference=args.baseline_reference)


if __name__ == "__main__":
    main()
