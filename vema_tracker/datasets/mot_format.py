"""MOTChallenge / DanceTrack / MOT20 sequence reader and prediction writer."""

from __future__ import annotations

import configparser
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from vema_tracker.baselines.base import Detection, TrackOutput


@dataclass
class SequenceData:
    """In-memory representation of a single tracking sequence."""

    name: str
    fps: float
    width: int
    height: int
    num_frames: int
    detections: dict[int, list[Detection]] = field(default_factory=dict)
    gt: dict[int, list[tuple[int, np.ndarray]]] = field(default_factory=dict)
    frame_dts: dict[int, float] | None = None


def _parse_seqinfo(seq_dir: Path) -> tuple[float, int, int, int]:
    ini_path = seq_dir / "seqinfo.ini"
    fps, width, height, seq_len = 30.0, 1920, 1080, 0
    if ini_path.exists():
        cp = configparser.ConfigParser()
        cp.read(ini_path, encoding="utf-8")
        sec = cp["Sequence"] if "Sequence" in cp else {}
        fps = float(sec.get("frameRate", fps))
        width = int(sec.get("imWidth", width))
        height = int(sec.get("imHeight", height))
        seq_len = int(sec.get("seqLength", seq_len))
    return fps, width, height, seq_len


def _parse_gt_file(gt_path: Path) -> dict[int, list[tuple[int, np.ndarray]]]:
    gt_by_frame: dict[int, list[tuple[int, np.ndarray]]] = {}
    if not gt_path.exists():
        return gt_by_frame
    with open(gt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 6:
                continue
            frame_id = int(float(parts[0]))
            obj_id = int(float(parts[1]))
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            if len(parts) >= 8:
                mark = int(float(parts[6]))
                cls_label = int(float(parts[7]))
                if mark == 0 or cls_label != 1:
                    continue
            elif len(parts) == 7:
                mark = int(float(parts[6]))
                if mark == 0:
                    continue
            if w <= 0 or h <= 0:
                continue
            box = np.array([x, y, x + w, y + h], dtype=np.float64)
            gt_by_frame.setdefault(frame_id, []).append((obj_id, box))
    return gt_by_frame


def _parse_det_file(det_path: Path) -> dict[int, list[Detection]]:
    dets_by_frame: dict[int, list[Detection]] = {}
    if not det_path.exists():
        return dets_by_frame
    with open(det_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 6:
                continue
            frame_id = int(float(parts[0]))
            if len(parts) == 6:
                x, y, w, h, conf = map(float, parts[1:6])
                cls_id = 0
            else:
                x, y, w, h, conf = map(float, parts[2:7])
                cls_id = int(float(parts[7])) if len(parts) >= 8 and float(parts[7]) >= 0 else 0
            if w <= 0 or h <= 0:
                continue
            box = np.array([x, y, x + w, y + h], dtype=np.float64)
            dets_by_frame.setdefault(frame_id, []).append(Detection(bbox=box, conf=conf, cls_id=cls_id))
    return dets_by_frame


def load_mot_sequence(
    seq_dir: str | Path,
    det_file: str | Path | None = None,
    half_val: bool = False,
    half_train: bool = False,
) -> SequenceData:
    """Load one MOTChallenge sequence directory."""
    if half_val and half_train:
        raise ValueError("Cannot set both half_val=True and half_train=True")
    seq_path = Path(seq_dir)
    fps, width, height, seq_len = _parse_seqinfo(seq_path)
    gt_raw = _parse_gt_file(seq_path / "gt" / "gt.txt")
    det_path = Path(det_file) if det_file is not None else (seq_path / "det" / "det.txt")
    dets_raw = _parse_det_file(det_path)

    if seq_len <= 0:
        max_f = max(
            [max(gt_raw.keys(), default=0), max(dets_raw.keys(), default=0)]
        )
        seq_len = max_f

    if half_val and seq_len >= 2:
        start_frame = (seq_len // 2) + 1
        dets = {
            f - start_frame + 1: dets_raw.get(f, [])
            for f in range(start_frame, seq_len + 1)
            if f in dets_raw
        }
        gt = {
            f - start_frame + 1: gt_raw.get(f, [])
            for f in range(start_frame, seq_len + 1)
            if f in gt_raw
        }
        num_frames = seq_len - start_frame + 1
    elif half_train and seq_len >= 2:
        end_frame = seq_len // 2
        dets = {
            f: dets_raw.get(f, [])
            for f in range(1, end_frame + 1)
            if f in dets_raw
        }
        gt = {
            f: gt_raw.get(f, [])
            for f in range(1, end_frame + 1)
            if f in gt_raw
        }
        num_frames = end_frame
    else:
        dets = dets_raw
        gt = gt_raw
        num_frames = seq_len

    return SequenceData(
        name=seq_path.name,
        fps=fps,
        width=width,
        height=height,
        num_frames=num_frames,
        detections=dets,
        gt=gt,
    )


def load_mot_dataset(
    dataset_name: str,
    data_root: str | Path = "vema_tracker/data",
    det_dir: str | Path | None = None,
) -> list[SequenceData]:
    """Load a public MOT split ('mot17-val', 'mot17-train', 'mot17-sdp-val', 'mot17-sdp-train', 'dancetrack-val', 'dancetrack-train', 'mot20-val', 'mot20-train')."""
    root = Path(data_root)
    key = dataset_name.strip().lower()
    det_root = Path(det_dir) if det_dir is not None else None
    half_val = False
    half_train = False

    if key in ("mot17-val", "mot17-frcnn-val", "mot17-train", "mot17-frcnn-train"):
        base = root / "MOT17" / "train"
        if not base.exists():
            raise FileNotFoundError(
                f"MOT17 directory not found at {base}. Run `python -m vema_tracker.datasets.fetch_datasets --download`."
            )
        seq_dirs = sorted(p for p in base.iterdir() if p.is_dir())
        frcnn = [p for p in seq_dirs if p.name.endswith("-FRCNN")]
        if frcnn:
            seq_dirs = frcnn
        half_val = key.endswith("-val")
        half_train = key.endswith("-train")
    elif key in ("mot17-sdp-val", "mot17-sdp-train"):
        base = root / "MOT17" / "train"
        if not base.exists():
            raise FileNotFoundError(
                f"MOT17 directory not found at {base}. Run `python -m vema_tracker.datasets.fetch_datasets --download`."
            )
        seq_dirs = sorted(p for p in base.iterdir() if p.is_dir() and p.name.endswith("-SDP"))
        half_val = key.endswith("-val")
        half_train = key.endswith("-train")
    elif key in ("mot20-val", "mot20-train"):
        base = root / "MOT20" / "train"
        if not base.exists():
            raise FileNotFoundError(
                f"MOT20 directory not found at {base}. Run `python -m vema_tracker.datasets.fetch_datasets --download`."
            )
        seq_dirs = sorted(p for p in base.iterdir() if p.is_dir())
        half_val = key.endswith("-val")
        half_train = key.endswith("-train")
    elif key == "dancetrack-val":
        base = root / "DanceTrack" / "val"
        if not base.exists():
            raise FileNotFoundError(
                f"DanceTrack val directory not found at {base}. Run `python -m vema_tracker.datasets.fetch_datasets --download`."
            )
        seq_dirs = sorted(p for p in base.iterdir() if p.is_dir())
    elif key == "dancetrack-train":
        base = root / "DanceTrack" / "train"
        if not base.exists():
            raise FileNotFoundError(
                f"DanceTrack train directory not found at {base}. Run `python -m vema_tracker.datasets.fetch_datasets --download dancetrack-train`."
            )
        seq_dirs = sorted(p for p in base.iterdir() if p.is_dir())
    else:
        raise ValueError(f"Unsupported MOT dataset {dataset_name!r}")

    sequences: list[SequenceData] = []
    for sdir in seq_dirs:
        custom_det = None
        if det_root is not None:
            candidate = det_root / f"{sdir.name}.txt"
            if candidate.exists():
                custom_det = candidate
        sequences.append(
            load_mot_sequence(sdir, det_file=custom_det, half_val=half_val, half_train=half_train)
        )
    return sequences


def write_mot_predictions(
    out_path: str | Path,
    frame_outputs: dict[int, list[TrackOutput]],
) -> None:
    """Write tracker outputs in MOTChallenge format: frame,id,x,y,w,h,conf,-1,-1,-1."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for frame_id in sorted(frame_outputs.keys()):
            for trk in frame_outputs[frame_id]:
                x1, y1, x2, y2 = trk.bbox
                w = max(0.0, float(x2 - x1))
                h = max(0.0, float(y2 - y1))
                f.write(
                    f"{frame_id},{trk.track_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{trk.conf:.4f},-1,-1,-1\n"
                )
