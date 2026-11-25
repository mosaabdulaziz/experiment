"""Ground-truth evaluation against the official AICity22 Track1 eval script.

Writes the pipeline's predictions in the eval's expected format
(space-separated: CameraId Id FrameId X Y Width Height -1 -1) and runs
eval/eval.py with all-white ROIs, so the dataset's road masks are never
applied. Returns IDF1 / IDP / IDR.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from .config import EVAL_DIR, ROI_DIR, PipelineConfig
from .data import CameraInfo


def _cam_number(cam_id: str) -> int:
    return int(re.search(r"\d+", cam_id).group())


def write_prediction_file(
    global_ids: dict[int, list[tuple[str, int]]],
    tracklets_by_cam: dict[str, dict[int, object]],
    cam_infos: dict[str, CameraInfo],
    cfg: PipelineConfig,
    out_path: Path,
) -> Path:
    """Serialize global identities into the eval's MOT-style prediction file.

    Each tracklet contributes all of its observations with its global id.
    """
    lines = []
    for gid, members in global_ids.items():
        for cam_id, local_tid in members:
            info = cam_infos[cam_id]
            t = tracklets_by_cam[cam_id][local_tid]
            cam_num = _cam_number(cam_id)
            for frame, box in zip(t.frames, t.boxes):
                x, y, w, h = (int(round(v)) for v in box)
                lines.append(f"{cam_num} {gid} {frame} {x} {y} {w} {h} -1 -1")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def prepare_scene_gt(scene: str, cam_ids: list[str], cfg: PipelineConfig, out_dir: Path) -> Path:
    """Extract the official GT rows for this scene's cameras into a small file."""
    gts = EVAL_DIR / "ground_truth_train.txt"
    if not gts.exists():
        raise FileNotFoundError(f"official GT missing: {gts}")
    keep = {_cam_number(c) for c in cam_ids}
    rows = [
        line for line in gts.read_text().splitlines()
        if line.strip() and int(line.split()[0]) in keep
    ]
    out = out_dir / f"ground_truth_{scene}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rows) + "\n")
    return out


def ensure_white_rois(cam_ids: list[str], cfg: PipelineConfig) -> Path:
    """Materialize all-white ROI images so the official eval's mask filter is a no-op."""
    from PIL import Image

    for cam in cam_ids:
        d = ROI_DIR / cfg.dstype / f"c{_cam_number(cam):03d}"
        d.mkdir(parents=True, exist_ok=True)
        img = d / "roi.jpg"
        if not img.exists():
            Image.fromarray(np.full((540, 960), 255, dtype=np.uint8)).save(img)
    return ROI_DIR


def run_eval(
    gt_path: Path,
    pred_path: Path,
    cfg: PipelineConfig,
    roidir: Path | None = None,
) -> dict[str, float]:
    """Run the official eval.py and return {'idf1': .., 'idp': .., 'idr': ..}."""
    roidir = roidir or ROI_DIR
    cmd = [
        sys.executable,
        str(EVAL_DIR / "eval.py"),
        str(gt_path.resolve()),
        str(pred_path.resolve()),
        "--dstype", cfg.dstype,
        "--roidir", str(roidir.resolve()),
        "--mread",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(EVAL_DIR))
    if proc.returncode != 0:
        raise RuntimeError(f"eval.py failed:\n{proc.stderr}\n{proc.stdout}")
    import json

    m = re.search(r"\{.*\}", proc.stdout, re.DOTALL)
    if not m:
        raise RuntimeError(f"could not parse eval output:\n{proc.stdout}")
    data = json.loads(m.group(0))
    row = data.get("results", data)  # --mread wraps the row as {"results": {...}}
    return {k: float(v) for k, v in row.items() if k in ("idf1", "idp", "idr")}
