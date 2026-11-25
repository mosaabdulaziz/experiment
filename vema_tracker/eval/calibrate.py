"""Leak-free threshold & hyperparameter calibration on disjoint train splits."""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from vema_tracker.baselines.vema_adapter import VEMABaselineAdapter
from vema_tracker.core.config import VEMAConfig
from vema_tracker.datasets import SequenceData, load_mot_dataset

from .metrics import SequenceMetrics, aggregate_metrics, evaluate_sequence

FAST_ALPHAS = np.array([0.25, 0.50, 0.75], dtype=np.float64)

_WORKER_SEQS: list[SequenceData] = []


def _init_worker(seqs: list[SequenceData]) -> None:
    global _WORKER_SEQS
    _WORKER_SEQS = seqs


def _eval_worker(cfg: VEMAConfig) -> tuple[float, float, float, int]:
    met = _eval_vema(_WORKER_SEQS, cfg, fast=True)
    return (met.hota, met.assa, met.idf1, met.idsw)


def _summarize_detector_stats(sequences: list[SequenceData]) -> dict[str, Any]:
    confs = [d.conf for seq in sequences for dets in seq.detections.values() for d in dets]
    if not confs:
        return {}
    arr = np.asarray(confs, dtype=np.float64)
    return {
        "num_dets": int(len(arr)),
        "min": float(np.min(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
    }


def _eval_vema(
    sequences: list[SequenceData],
    cfg: VEMAConfig,
    fast: bool = True,
) -> SequenceMetrics:
    alphas = FAST_ALPHAS if fast else None
    seq_mets: list[SequenceMetrics] = []
    for seq in sequences:
        tracker = VEMABaselineAdapter(config=cfg)
        tracker.reset(fps=seq.fps)
        preds = {}
        for fid in range(1, seq.num_frames + 1):
            dets = seq.detections.get(fid, [])
            dt = seq.frame_dts.get(fid) if seq.frame_dts else (1.0 / seq.fps if seq.fps > 0 else None)
            preds[fid] = tracker.update(dets, dt=dt)
        seq_mets.append(evaluate_sequence(seq.name, seq.gt, preds, num_frames=seq.num_frames, alphas=alphas))
    return aggregate_metrics(seq_mets, name="TRAIN")


def _objective_vals(hota: float, assa: float, idf1: float) -> float:
    return hota + 0.05 * assa + 0.02 * idf1


def calibrate_vema(
    train_dataset: str,
    base_config: VEMAConfig,
    data_dir: Path = Path("vema_tracker/data"),
    max_seqs: int | None = None,
    rounds: int = 2,
    workers: int | None = None,
    out_config: Path | None = None,
    out_log: Path | None = None,
) -> tuple[VEMAConfig, dict[str, Any]]:
    """Run leak-free parallel coordinate-descent calibration on a training split."""
    all_seqs = load_mot_dataset(train_dataset, data_root=data_dir)
    if max_seqs is not None and max_seqs < len(all_seqs):
        idx = np.linspace(0, len(all_seqs) - 1, max_seqs, dtype=int)
        cal_seqs = [all_seqs[i] for i in idx]
    else:
        cal_seqs = all_seqs

    det_stats = _summarize_detector_stats(cal_seqs)
    n_workers = workers or min(8, os.cpu_count() or 4)
    print(f"\n=== Calibrating VEMA on {train_dataset} ({len(cal_seqs)}/{len(all_seqs)} seqs, {n_workers} workers) ===", flush=True)
    print(f"Detector score quantiles: {det_stats}", flush=True)

    t0 = time.perf_counter()
    best_cfg = base_config
    init_exact = _eval_vema(all_seqs, base_config, fast=False)
    print(
        f"Baseline (all {len(all_seqs)} train seqs, exact): "
        f"HOTA={init_exact.hota:.2f} AssA={init_exact.assa:.2f} IDF1={init_exact.idf1:.2f} IDSW={init_exact.idsw}",
        flush=True,
    )

    base_fast = _eval_vema(cal_seqs, best_cfg, fast=True)
    best_score = _objective_vals(base_fast.hota, base_fast.assa, base_fast.idf1)

    param_grid: list[tuple[str, list[Any]]] = [
        ("tau_high", [0.45, 0.55, 0.60, 0.70, 0.80, 0.85]),
        ("tau_low", [0.05, 0.10, 0.15, 0.25, 0.35]),
        ("iou_thresh_high", [0.15, 0.20, 0.25, 0.30, 0.35]),
        ("iou_thresh_low", [0.10, 0.15, 0.20, 0.30]),
        ("probation_frames", [1, 2, 3]),
        ("alpha", [0.65, 0.72, 0.80, 0.85, 0.90]),
        ("gamma", [0.92, 0.95, 0.98, 1.00]),
        ("enable_scale_velocity", [True, False]),
        ("scale_alpha", [0.85, 0.90, 0.95]),
        ("enable_truncation_gate", [True, False]),
        ("area_ratio_min", [0.55, 0.65, 0.75, 0.85]),
        ("cos_min", [-0.30, -0.10, 0.0, 0.20]),
        ("biou_rate", [0.0, 0.03, 0.05, 0.08]),
        ("lambda_pos", [0.0, 0.15, 0.35, 0.55]),
        ("lambda_dir", [0.0, 0.15, 0.25, 0.40]),
        ("relocation_cost_thresh", [1.20, 1.45, 1.65, 1.90, 2.20]),
        ("max_lost_frames", [15, 30, 45, 60]),
    ]

    history: list[dict[str, Any]] = []
    n_trials = 0

    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker, initargs=(cal_seqs,)) as pool:
        for rnd in range(1, rounds + 1):
            improved_in_round = False
            for param, candidates in param_grid:
                if param in ("area_ratio_min", "cos_min") and not best_cfg.enable_truncation_gate:
                    continue
                if param == "scale_alpha" and not best_cfg.enable_scale_velocity:
                    continue
                cur_val = getattr(best_cfg, param)
                cand_cfgs: list[tuple[Any, VEMAConfig]] = []
                for val in candidates:
                    if val == cur_val:
                        continue
                    if param == "tau_low" and val >= best_cfg.tau_high - 0.05:
                        continue
                    if param == "tau_high" and val <= best_cfg.tau_low + 0.05:
                        continue
                    try:
                        cand_cfgs.append((val, best_cfg.with_overrides(**{param: val})))
                    except ValueError:
                        continue

                if not cand_cfgs:
                    continue
                n_trials += len(cand_cfgs)
                results = list(pool.map(_eval_worker, [c for _, c in cand_cfgs]))

                best_local = None
                for (val, ccfg), (hota, assa, idf1, idsw) in zip(cand_cfgs, results):
                    sc = _objective_vals(hota, assa, idf1)
                    if sc > best_score + 1e-4:
                        best_score = sc
                        best_local = (val, ccfg, hota, assa, idf1, idsw)

                if best_local is not None:
                    val, ccfg, hota, assa, idf1, idsw = best_local
                    best_cfg = ccfg
                    improved_in_round = True
                    history.append(
                        {
                            "round": rnd,
                            "param": param,
                            "value": val,
                            "fast_HOTA": round(hota, 3),
                            "fast_AssA": round(assa, 3),
                            "fast_IDF1": round(idf1, 3),
                            "IDSW": int(idsw),
                        }
                    )
                    print(
                        f"  [r{rnd} #{n_trials:03d}] {param}={val!r} -> "
                        f"fast HOTA={hota:.2f} AssA={assa:.2f} IDF1={idf1:.2f} IDSW={idsw}",
                        flush=True,
                    )
            if not improved_in_round:
                break

    final_exact = _eval_vema(all_seqs, best_cfg, fast=False)
    elapsed_s = time.perf_counter() - t0
    print(
        f"\nCalibrated (all {len(all_seqs)} train seqs, exact, {elapsed_s:.1f}s, {n_trials} trials): "
        f"HOTA={final_exact.hota:.2f} ({final_exact.hota - init_exact.hota:+.2f}) "
        f"AssA={final_exact.assa:.2f} ({final_exact.assa - init_exact.assa:+.2f}) "
        f"IDF1={final_exact.idf1:.2f} ({final_exact.idf1 - init_exact.idf1:+.2f}) "
        f"IDSW={final_exact.idsw} ({final_exact.idsw - init_exact.idsw:+d})",
        flush=True,
    )

    if out_config is not None:
        best_cfg.to_yaml(out_config)
        print(f"Saved calibrated config -> {out_config}", flush=True)

    report = {
        "train_dataset": train_dataset,
        "num_train_seqs_total": len(all_seqs),
        "num_cal_seqs": len(cal_seqs),
        "n_trials": n_trials,
        "elapsed_s": round(elapsed_s, 2),
        "detector_stats": det_stats,
        "before_train_exact": init_exact.to_dict(),
        "after_train_exact": final_exact.to_dict(),
        "calibrated_config": best_cfg.to_dict(),
        "updates": history,
    }
    if out_log is not None:
        out_log.parent.mkdir(parents=True, exist_ok=True)
        with open(out_log, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    return best_cfg, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate VEMA thresholds on disjoint train split.")
    parser.add_argument(
        "--train-dataset",
        required=True,
        choices=["mot17-train", "mot17-sdp-train", "dancetrack-train", "mot20-train"],
    )
    parser.add_argument("--base-config", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=Path("vema_tracker/data"))
    parser.add_argument("--max-seqs", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--out-config", type=Path, required=True)
    parser.add_argument("--out-log", type=Path, default=None)
    args = parser.parse_args()

    base_cfg = VEMAConfig.from_yaml(args.base_config) if args.base_config else VEMAConfig()
    calibrate_vema(
        train_dataset=args.train_dataset,
        base_config=base_cfg,
        data_dir=args.data_dir,
        max_seqs=args.max_seqs,
        rounds=args.rounds,
        workers=args.workers,
        out_config=args.out_config,
        out_log=args.out_log,
    )


if __name__ == "__main__":
    main()
