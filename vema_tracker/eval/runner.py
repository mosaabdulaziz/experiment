"""Benchmark runner comparing VEMA against SOTA pure-motion trackers."""

from __future__ import annotations

import argparse
import inspect
import json
import time
from pathlib import Path
from typing import Any

from vema_tracker.baselines import TRACKER_NAMES, TrackOutput, get_tracker
from vema_tracker.core.config import VEMAConfig
from vema_tracker.datasets import SequenceData, generate_synthetic_suite, load_mot_dataset, write_mot_predictions

from .metrics import SequenceMetrics, aggregate_metrics, evaluate_sequence


def run_tracker_on_sequence(
    tracker_name: str,
    seq: SequenceData,
    config: VEMAConfig | None = None,
) -> tuple[dict[int, list[TrackOutput]], SequenceMetrics, float]:
    """Run one tracker over a sequence and return (predictions, metrics, ms_per_frame)."""
    tracker = get_tracker(tracker_name, config=config)
    reset_params = inspect.signature(tracker.reset).parameters
    accepts_img_size = "img_size" in reset_params or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in reset_params.values()
    )
    if accepts_img_size:
        tracker.reset(fps=seq.fps, img_size=(seq.width, seq.height))
    else:
        tracker.reset(fps=seq.fps)

    outputs: dict[int, list[TrackOutput]] = {}
    t0 = time.perf_counter()
    for fid in range(1, seq.num_frames + 1):
        dets = seq.detections.get(fid, [])
        dt = seq.frame_dts.get(fid) if seq.frame_dts else (1.0 / seq.fps if seq.fps > 0 else None)
        outputs[fid] = tracker.update(dets, dt=dt)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    ms_per_frame = elapsed_ms / max(seq.num_frames, 1)

    metrics = evaluate_sequence(seq.name, seq.gt, outputs, num_frames=seq.num_frames)
    return outputs, metrics, ms_per_frame


def load_sequences_for_arg(
    dataset: str,
    data_dir: Path,
    det_dir: Path | None,
) -> list[SequenceData]:
    if dataset == "synthetic":
        return generate_synthetic_suite()
    return load_mot_dataset(dataset, data_root=data_dir, det_dir=det_dir)


def format_markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = ["Tracker", "HOTA", "AssA", "DetA", "IDF1", "MOTA", "IDSW", "Frag", "ms/frame", "FPS"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:" for _ in headers[1:]]) + " |",
    ]
    for r in rows:
        ms = r["ms_per_frame"]
        fps = 1000.0 / ms if ms > 1e-6 else 0.0
        lines.append(
            f"| {r['tracker']} | {r['HOTA']:.2f} | {r['AssA']:.2f} | {r['DetA']:.2f} | "
            f"{r['IDF1']:.2f} | {r['MOTA']:.2f} | {r['IDSW']} | {r['Frag']} | "
            f"{ms:.3f} | {fps:.0f} |"
        )
    return "\n".join(lines)


def run_benchmark(
    dataset: str,
    trackers: list[str],
    config: VEMAConfig | None = None,
    data_dir: Path = Path("vema_tracker/data"),
    det_dir: Path | None = None,
    out_path: Path | None = None,
    save_mot_dir: Path | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    sequences = load_sequences_for_arg(dataset, data_dir, det_dir)
    cfg = config or VEMAConfig()

    overall_rows: list[dict[str, Any]] = []
    per_tracker_seqs: dict[str, list[dict[str, Any]]] = {}

    for tname in trackers:
        seq_metrics: list[SequenceMetrics] = []
        seq_rows: list[dict[str, Any]] = []
        total_ms = 0.0
        total_frames = 0

        for seq in sequences:
            preds, met, ms_pf = run_tracker_on_sequence(tname, seq, config=cfg)
            seq_metrics.append(met)
            row = met.to_dict()
            row["ms_per_frame"] = round(ms_pf, 4)
            seq_rows.append(row)
            total_ms += ms_pf * seq.num_frames
            total_frames += seq.num_frames

            if save_mot_dir is not None:
                write_mot_predictions(save_mot_dir / tname / f"{seq.name}.txt", preds)

        agg = aggregate_metrics(seq_metrics, name="OVERALL")
        avg_ms = total_ms / max(total_frames, 1)
        agg_dict = agg.to_dict()
        agg_dict["tracker"] = tname
        agg_dict["ms_per_frame"] = round(avg_ms, 4)
        overall_rows.append(agg_dict)
        per_tracker_seqs[tname] = seq_rows

    if verbose:
        print(f"\n=== Benchmark Summary: {dataset} ({len(sequences)} sequences) ===\n")
        print(format_markdown_table(overall_rows))
        print("\n--- Per-Sequence Breakdown ---")
        for seq_idx, seq in enumerate(sequences):
            print(f"\n[{seq.name}] ({seq.num_frames} frames @ {seq.fps:.0f} fps)")
            seq_table = []
            for tname in trackers:
                r = dict(per_tracker_seqs[tname][seq_idx])
                r["tracker"] = tname
                seq_table.append(r)
            print(format_markdown_table(seq_table))

    payload = {
        "dataset": dataset,
        "sequences": [s.name for s in sequences],
        "config": cfg.to_dict(),
        "overall": overall_rows,
        "per_sequence": per_tracker_seqs,
    }
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark VEMA vs SOTA pure-motion MOT trackers.")
    parser.add_argument(
        "--dataset",
        default="synthetic",
        choices=["synthetic", "mot17-val", "mot17-sdp-val", "dancetrack-val", "mot20-val"],
    )
    parser.add_argument("--trackers", nargs="+", default=list(TRACKER_NAMES))
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML config for VEMA.")
    parser.add_argument("--data-dir", type=Path, default=Path("vema_tracker/data"))
    parser.add_argument("--det-dir", type=Path, default=None, help="Optional directory of public YOLOX-X .txt files.")
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output file.")
    parser.add_argument("--save-mot-dir", type=Path, default=None, help="Optional directory for TrackEval MOT .txt dumps.")
    args = parser.parse_args()

    cfg = VEMAConfig.from_yaml(args.config) if args.config else VEMAConfig()
    run_benchmark(
        dataset=args.dataset,
        trackers=args.trackers,
        config=cfg,
        data_dir=args.data_dir,
        det_dir=args.det_dir,
        out_path=args.out,
        save_mot_dir=args.save_mot_dir,
        verbose=True,
    )


if __name__ == "__main__":
    main()
