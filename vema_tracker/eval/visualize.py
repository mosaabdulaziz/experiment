"""Diagnostic trajectory plotter for inspecting tracker behavior and ID switches."""

from __future__ import annotations

from pathlib import Path

from vema_tracker.baselines.base import TrackOutput
from vema_tracker.datasets.mot_format import SequenceData


def plot_trajectories(
    seq: SequenceData,
    predictions_by_tracker: dict[str, dict[int, list[TrackOutput]]],
    out_path: str | Path,
) -> Path:
    """Save 2D center-point trajectories per tracker alongside ground truth."""
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for trajectory plotting. Install with `pip install matplotlib`."
        ) from exc

    n_panels = 1 + len(predictions_by_tracker)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.5), sharex=True, sharey=True)
    if n_panels == 1:
        axes = [axes]

    def _collect_paths(frame_map):
        paths = {}
        for fid in sorted(frame_map.keys()):
            for item in frame_map[fid]:
                if isinstance(item, tuple):
                    tid, box = item[0], item[1]
                else:
                    tid, box = item.track_id, item.bbox
                cx = 0.5 * (box[0] + box[2])
                cy = 0.5 * (box[1] + box[3])
                paths.setdefault(tid, []).append((cx, cy))
        return paths

    # Ground truth panel
    ax0 = axes[0]
    for tid, pts in _collect_paths(seq.gt).items():
        xs, ys = zip(*pts)
        ax0.plot(xs, ys, linewidth=1.6, label=f"GT {tid}")
    ax0.set_title(f"{seq.name} — Ground Truth")
    ax0.invert_yaxis()

    for idx, (tname, preds) in enumerate(predictions_by_tracker.items(), start=1):
        ax = axes[idx]
        for tid, pts in _collect_paths(preds).items():
            xs, ys = zip(*pts)
            ax.plot(xs, ys, linewidth=1.5, alpha=0.85, label=f"ID {tid}")
        ax.set_title(f"{tname} ({len(_collect_paths(preds))} IDs)")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
