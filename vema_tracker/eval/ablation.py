"""Controlled ablation harness for VEMA components."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from vema_tracker.core.config import VEMAConfig

from .metrics import aggregate_metrics
from .runner import load_sequences_for_arg, run_tracker_on_sequence

ABLATIONS: list[tuple[str, dict[str, Any]]] = [
    ("full_vema", {}),
    ("-truncation_gate", {"enable_truncation_gate": False}),
    ("-size_lock", {"enable_size_lock": False}),
    ("-velocity_blending", {"enable_velocity_blending": False}),
    ("-anisotropic_cost", {"enable_anisotropic_cost": False}),
    ("-biou", {"enable_biou": False}),
    ("-dir_cost", {"enable_dir_cost": False}),
    ("-scale_velocity", {"enable_scale_velocity": False}),
]


def run_ablation(
    dataset: str,
    base_config: VEMAConfig | None = None,
    data_dir: Path = Path("vema_tracker/data"),
    det_dir: Path | None = None,
    out_path: Path | None = None,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    sequences = load_sequences_for_arg(dataset, data_dir, det_dir)
    base_cfg = base_config or VEMAConfig()

    rows: list[dict[str, Any]] = []
    baseline: dict[str, Any] | None = None

    for label, overrides in ABLATIONS:
        cfg = base_cfg.with_overrides(**overrides)
        seq_mets = []
        total_ms = 0.0
        total_frames = 0
        for seq in sequences:
            _, met, ms_pf = run_tracker_on_sequence("vema", seq, config=cfg)
            seq_mets.append(met)
            total_ms += ms_pf * seq.num_frames
            total_frames += seq.num_frames

        agg = aggregate_metrics(seq_mets, name=label)
        entry = agg.to_dict()
        entry["variant"] = label
        entry["ms_per_frame"] = round(total_ms / max(total_frames, 1), 4)
        if baseline is None:
            baseline = entry
        entry["dHOTA"] = round(entry["HOTA"] - baseline["HOTA"], 3)
        entry["dAssA"] = round(entry["AssA"] - baseline["AssA"], 3)
        entry["dIDF1"] = round(entry["IDF1"] - baseline["IDF1"], 3)
        entry["dIDSW"] = int(entry["IDSW"] - baseline["IDSW"])
        rows.append(entry)

    if verbose:
        print(f"\n=== VEMA Ablation Study: {dataset} ({len(sequences)} sequences) ===\n")
        headers = ["Variant", "HOTA", "dHOTA", "AssA", "dAssA", "IDF1", "dIDF1", "IDSW", "dIDSW", "ms/frame"]
        print("| " + " | ".join(headers) + " |")
        print("| " + " | ".join(["---"] + ["---:" for _ in headers[1:]]) + " |")
        for r in rows:
            print(
                f"| {r['variant']} | {r['HOTA']:.2f} | {r['dHOTA']:+.2f} | "
                f"{r['AssA']:.2f} | {r['dAssA']:+.2f} | {r['IDF1']:.2f} | {r['dIDF1']:+.2f} | "
                f"{r['IDSW']} | {r['dIDSW']:+d} | {r['ms_per_frame']:.3f} |"
            )

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"dataset": dataset, "ablation": rows}, f, indent=2)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VEMA component ablation study.")
    parser.add_argument(
        "--dataset",
        default="synthetic",
        choices=["synthetic", "mot17-val", "mot17-sdp-val", "dancetrack-val", "mot20-val"],
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=Path("vema_tracker/data"))
    parser.add_argument("--det-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cfg = VEMAConfig.from_yaml(args.config) if args.config else VEMAConfig()
    run_ablation(
        dataset=args.dataset,
        base_config=cfg,
        data_dir=args.data_dir,
        det_dir=args.det_dir,
        out_path=args.out,
        verbose=True,
    )


if __name__ == "__main__":
    main()
