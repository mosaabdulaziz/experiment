"""Dataset discovery, split partitioning, sparse frame loading, and quantile analysis for RetailS / PoseLift."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np

PRODUCT_DIR = Path(__file__).resolve().parents[1]
if str(PRODUCT_DIR) not in sys.path:
    sys.path.insert(0, str(PRODUCT_DIR))

from adapters import extract_mask_events, load_retails_pose_json  # noqa: E402
from concealment.features import extract_track_features  # noqa: E402
from concealment.state_machine import ConcealmentMonitor  # noqa: E402
from concealment.types import (  # noqa: E402
    COCO_KEYPOINTS,
    CameraZones,
    DetectorSettings,
    PoseFrame,
    TrackObservation,
)
from concealment.zones import propose_zones_from_dwell  # noqa: E402


@dataclass(frozen=True)
class DatasetSplits:
    train_tune_days: tuple[str, ...]
    train_holdout_days: tuple[str, ...]
    train_tune_files: tuple[Path, ...]
    train_holdout_files: tuple[Path, ...]
    staged_pairs: tuple[tuple[Path, Path], ...]
    realworld_pairs: tuple[tuple[Path, Path], ...]


def parse_camera_id(stem: str) -> str:
    """Normalize RetailS filename prefix (`01_0222`, `1_041812_1`) to `cam-<int>`."""
    prefix = stem.split("_")[0]
    try:
        return f"cam-{int(prefix)}"
    except ValueError:
        return "cam-1"


def parse_train_day(stem: str) -> str:
    """Extract 4-digit MMDD date code from a `RetailS_train` filename (`1_041812_1` -> `0418`)."""
    parts = stem.split("_")
    if len(parts) >= 2 and len(parts[1]) >= 4:
        return parts[1][:4]
    return "0000"


def discover_retails_splits(data_root: Path) -> DatasetSplits:
    """Locate RetailS splits under `data_root` and partition train into Days 1-8 (tune) and Days 9-11 (holdout)."""
    root = Path(data_root)
    inner = root / "RetailS" if (root / "RetailS").is_dir() else root

    train_dir = inner / "RetailS_train" / "pose" / "train"
    staged_pose_dir = inner / "RetailS_test_staged" / "pose" / "test"
    staged_gt_dir = inner / "RetailS_test_staged" / "gt" / "test_frame_mask"
    rw_pose_dir = inner / "RetailS_test_realworld" / "pose" / "test"
    rw_gt_dir = inner / "RetailS_test_realworld" / "gt" / "test_frame_mask"

    train_all = sorted(train_dir.glob("*.json"))
    unique_days = sorted({parse_train_day(p.stem) for p in train_all})
    tune_days = tuple(unique_days[:8])
    holdout_days = tuple(unique_days[8:])
    tune_set = set(tune_days)

    train_tune = tuple(p for p in train_all if parse_train_day(p.stem) in tune_set)
    train_holdout = tuple(p for p in train_all if parse_train_day(p.stem) not in tune_set)

    staged_gts = {p.stem: p for p in staged_gt_dir.glob("*.npy")}
    staged_pairs = tuple(
        (p, staged_gts[p.stem])
        for p in sorted(staged_pose_dir.glob("*.json"))
        if p.stem in staged_gts
    )

    rw_gts = {p.stem: p for p in rw_gt_dir.glob("*.npy")}
    rw_pairs = tuple(
        (p, rw_gts[p.stem])
        for p in sorted(rw_pose_dir.glob("*.json"))
        if p.stem in rw_gts
    )

    return DatasetSplits(
        train_tune_days=tune_days,
        train_holdout_days=holdout_days,
        train_tune_files=train_tune,
        train_holdout_files=train_holdout,
        staged_pairs=staged_pairs,
        realworld_pairs=rw_pairs,
    )


def representative_indices(count: int, limit: int) -> list[int]:
    """Evenly spaced indices including both ends, never a biased prefix."""
    if count <= 0 or limit <= 0:
        return []
    return np.linspace(0, count - 1, min(count, limit), dtype=int).tolist()


def stratified_sample_files(files: tuple[Path, ...] | list[Path], limit: int) -> list[Path]:
    """Deterministic equal-allocation camera/day sampling with capacity redistribution."""
    groups: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for path in sorted(files):
        groups[(parse_camera_id(path.stem), parse_train_day(path.stem))].append(path)
    quotas = dict.fromkeys(sorted(groups), 0)
    remaining = min(max(0, limit), len(files))
    while remaining:
        for key in quotas:
            if quotas[key] < len(groups[key]) and remaining:
                quotas[key] += 1
                remaining -= 1
    return [groups[key][i] for key, quota in quotas.items()
            for i in representative_indices(len(groups[key]), quota)]


def partition_staged_pairs(pairs):
    """Frozen 50/50 development/test split by camera, GT class, single/multi-person.

    Singleton/odd strata alternate the extra clip to keep overall sizes within one.
    File-content hashing randomizes within strata without process-dependent RNG state.
    """
    groups = defaultdict(list)
    for pose, gt in pairs:
        raw = json.loads(pose.read_text(encoding="utf-8"))
        tracks = sum(bool(frames) for frames in raw.values() if isinstance(frames, dict))
        key = (parse_camera_id(pose.stem), bool(np.any(np.load(gt) > 0)), tracks == 1)
        groups[key].append((pose, gt))
    dev, test = [], []
    extra_to_dev = True
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda p: (hashlib.sha256(p[0].name.encode()).hexdigest(), str(p[0])))
        n_dev = len(group) // 2
        if len(group) % 2:
            n_dev += int(extra_to_dev)
            extra_to_dev = not extra_to_dev
        dev.extend(group[:n_dev])
        test.extend(group[n_dev:])
    return tuple(sorted(dev)), tuple(sorted(test))


def file_identity(path: Path) -> dict:
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest(), "size": path.stat().st_size}


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def build_provenance(settings, camera_zones, datasets, source_paths=None) -> dict:
    """Bind exact settings, zones, dataset membership/content and current source code."""
    if source_paths is None:
        source_paths = sorted((PRODUCT_DIR / "concealment").glob("*.py")) + sorted(
            (PRODUCT_DIR / "calibration").glob("*.py")) + [
                PRODUCT_DIR / "adapters.py", PRODUCT_DIR / "config.py",
                PRODUCT_DIR / "concealment_detection.yaml"]
    payload = {
        "schema_version": 1,
        "settings": asdict(settings),
        "zones": {k: asdict(v) for k, v in sorted(camera_zones.items())},
        "datasets": {k: [file_identity(p) for p in sorted(paths)]
                     for k, paths in sorted(datasets.items())},
        "sources": [file_identity(p) for p in sorted(source_paths)],
        "runtime": {"python": sys.version, "numpy": np.__version__},
    }
    return {**payload, "fingerprint": canonical_hash(payload)}


def load_sparse_pose_frames(
    pose_path: Path,
    fps: float = 15.0,
    shape: tuple[int, int] = (1920, 1080),
    swap_yx: bool = True,
    gap_flush_s: float = 1.6,
    frame_count: int | None = None,
) -> tuple[list[PoseFrame], float, float]:
    """Load a RetailS JSON file while collapsing long empty-store gaps into a single flush frame.

    Preserves exact `pts = f_idx / fps` for all frames so temporal state machines and quality
    gates behave identically to dense frame iteration, at ~5x speed on multi-hour train files.
    Returns `(frames, span_duration_s, active_duration_s)`.
    """
    raw_json = json.loads(Path(pose_path).read_text(encoding="utf-8"))
    camera = parse_camera_id(Path(pose_path).stem)
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
                x, y = (c1, c0) if swap_yx else (c0, c1)
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
    if max_f < 0:
        return ([], 0.0, 0.0)

    fps_val = max(float(fps), 1e-6)
    span_duration_s = (max_f + 1) / fps_val
    active_duration_s = len(by_frame) / fps_val
    occupied_indices = sorted(set(by_frame) | {0, max_f})
    gap_flush_frames = max(2, int(math.ceil(gap_flush_s * fps_val)))

    frames: list[PoseFrame] = []
    prev_f: int | None = None
    for f_idx in occupied_indices:
        if prev_f is not None and (f_idx - prev_f) > 1:
            gap = f_idx - prev_f
            if gap <= gap_flush_frames:
                for empty_f in range(prev_f + 1, f_idx):
                    frames.append(
                        PoseFrame(
                            camera=camera,
                            frame_num=empty_f,
                            pts=empty_f / fps_val,
                            shape=shape,
                            tracks=(),
                            observed=True,
                        )
                    )
            else:
                flush_f = prev_f + gap_flush_frames
                frames.append(
                    PoseFrame(
                        camera=camera,
                        frame_num=flush_f,
                        pts=flush_f / fps_val,
                        shape=shape,
                        tracks=(),
                        observed=True,
                    )
                )
        obs_list = sorted(by_frame[f_idx], key=lambda t: t.track_id)
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
        prev_f = f_idx

    # Final flush frame at clip end so any open AT_BODY hold is flushed
    if prev_f is not None and frame_count is None:
        end_f = prev_f + gap_flush_frames
        frames.append(
            PoseFrame(
                camera=camera,
                frame_num=end_f,
                pts=end_f / fps_val,
                shape=shape,
                tracks=(),
                observed=True,
            )
        )

    return (frames, span_duration_s, active_duration_s)


def propose_all_camera_zones(
    train_files: tuple[Path, ...] | list[Path],
    grid_size: int = 10,
    min_dwell_s: float = 2.0,
    top_k: int = 2,
    max_files_per_camera: int = 25,
    zone_free: bool = False,
) -> dict[str, CameraZones]:
    """Propose shelf-front zones per camera (`cam-1 .. cam-6`) from stationary dwell on normal training clips."""
    by_cam: dict[str, list[Path]] = defaultdict(list)
    for p in train_files:
        by_cam[parse_camera_id(p.stem)].append(p)

    result: dict[str, CameraZones] = {}
    for cam_id, paths in sorted(by_cam.items()):
        sampled = stratified_sample_files(paths, max_files_per_camera)
        combined_frames: list[PoseFrame] = []
        time_offset = 0.0
        tid_offset = 0
        for p in sampled:
            frames, span_s, _ = load_sparse_pose_frames(p)
            max_tid = 0
            for f in frames:
                shifted_tracks = tuple(
                    TrackObservation(
                        track_id=t.track_id + tid_offset,
                        box=t.box,
                        kps=t.kps,
                        has_bag=t.has_bag,
                        bag_box=t.bag_box,
                    )
                    for t in f.tracks
                )
                for t in f.tracks:
                    if t.track_id > max_tid:
                        max_tid = t.track_id
                combined_frames.append(
                    PoseFrame(
                        camera=cam_id,
                        frame_num=f.frame_num,
                        pts=f.pts + time_offset,
                        shape=f.shape,
                        tracks=shifted_tracks,
                        observed=True,
                    )
                )
            time_offset += span_s + 10.0
            tid_offset += max_tid + 100

        proposed = propose_zones_from_dwell(
            combined_frames,
            grid_size=grid_size,
            min_dwell_s=min_dwell_s,
            top_k=top_k,
        )
        result[cam_id] = CameraZones(
            camera=cam_id,
            zone_free=zone_free,
            zones=proposed.zones,
        )
    return result


def zones_to_yaml_dict(camera_zones: dict[str, CameraZones]) -> dict:
    """Convert `{camera_id: CameraZones}` into a clean YAML-serializable dict."""
    cameras_out: dict[str, dict] = {}
    for cam_id, cz in sorted(camera_zones.items()):
        cameras_out[cam_id] = {
            "zone_free": cz.zone_free,
            "zones": [
                {
                    "id": z.id,
                    "kind": z.kind,
                    "facing": list(z.facing) if z.facing is not None else None,
                    "points": [[round(px, 4), round(py, 4)] for px, py in z.points],
                }
                for z in cz.zones
            ],
        }
    return {"cameras": cameras_out}


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "p10": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0}
    arr = np.asarray(values, dtype=float)
    pcts = np.percentile(arr, [10, 25, 50, 75, 90])
    return {
        "count": int(arr.size),
        "p10": round(float(pcts[0]), 4),
        "p25": round(float(pcts[1]), 4),
        "p50": round(float(pcts[2]), 4),
        "p75": round(float(pcts[3]), 4),
        "p90": round(float(pcts[4]), 4),
    }


def _eligible_samples(frames, camera, zones, settings, limit=None):
    # Use production quality, canonical IDs and time-based history, not a 45-frame cap.
    monitor = ConcealmentMonitor(camera, settings, zones)
    occupied = [i for i, f in enumerate(frames) if f.tracks]
    selected = set(occupied[i] for i in representative_indices(
        len(occupied), len(occupied) if limit is None else limit))
    for i, frame in enumerate(frames):
        monitor.update(frame)
        if i not in selected:
            continue
        for cid, hist in monitor._histories.items():
            status = monitor.quality_gate._tracks.get(cid)
            if (hist and hist[-1][0] == frame.pts and status is not None
                    and status.last_status is not None and status.last_status.state == "OK"):
                yield frame, hist[-1][1], list(hist)


def compute_empirical_quantiles(
    staged_pairs: tuple[tuple[Path, Path], ...],
    train_files: tuple[Path, ...],
    camera_zones: dict[str, CameraZones],
    settings: DetectorSettings | None = None,
    max_staged: int = 200,
    max_train: int = 30,
) -> dict:
    """Measure feature distributions inside positive GT intervals (`gt == 1`) vs normal shopping (`gt == 0`)."""
    cfg = settings or DetectorSettings()
    pos_excursion: list[float] = []
    pos_min_elbow: list[float] = []
    pos_body_dist: list[float] = []
    pos_wrist_conf: list[float] = []
    pos_event_dur_s: list[float] = []
    pos_pre_event_lead_s: list[float] = []

    neg_excursion: list[float] = []
    neg_min_elbow: list[float] = []
    neg_body_dist: list[float] = []
    neg_wrist_conf: list[float] = []

    selected_paths = set(stratified_sample_files([p for p, _ in staged_pairs], max_staged))
    for pose_path, gt_path in staged_pairs:
        if pose_path not in selected_paths:
            continue
        raw = json.loads(pose_path.read_text(encoding="utf-8"))
        mask = np.load(gt_path)
        cam_id = parse_camera_id(pose_path.stem)
        cz = camera_zones.get(cam_id, CameraZones(camera=cam_id, zone_free=True))
        frames = load_retails_pose_json(raw, camera=cam_id, frame_count=len(mask))
        # Frame masks do not identify a perpetrator in multi-person clips.
        if len({t.track_id for f in frames for t in f.tracks}) != 1:
            continue
        intervals = extract_mask_events(mask)
        for s, e in intervals:
            pos_event_dur_s.append(e - s)
            pos_pre_event_lead_s.append(s)

        for frame, obs, hist in _eligible_samples(frames, cam_id, cz, cfg):
            if frame.frame_num < len(mask) and mask[frame.frame_num] > 0:
                feat = extract_track_features(obs, hist, cz, frame.shape, cfg)
                pos_excursion.append(feat.wrist_excursion_torsos)
                pos_min_elbow.append(feat.min_elbow_deg)
                pos_body_dist.append(feat.wrist_to_body_torsos)
                pos_wrist_conf.append(feat.active_wrist_conf)

    for pose_path in stratified_sample_files(train_files, max_train):
        frames, _, _ = load_sparse_pose_frames(pose_path)
        cam_id = parse_camera_id(pose_path.stem)
        cz = camera_zones.get(cam_id, CameraZones(camera=cam_id, zone_free=True))
        for frame, obs, hist in _eligible_samples(frames, cam_id, cz, cfg, limit=600):
            feat = extract_track_features(obs, hist, cz, frame.shape, cfg)
            neg_excursion.append(feat.wrist_excursion_torsos)
            neg_min_elbow.append(feat.min_elbow_deg)
            neg_body_dist.append(feat.wrist_to_body_torsos)
            neg_wrist_conf.append(feat.active_wrist_conf)

    return {
        "positive_staged_gt": {
            "event_duration_s": _quantiles(pos_event_dur_s),
            "pre_event_lead_s": _quantiles(pos_pre_event_lead_s),
            "wrist_excursion_torsos": _quantiles(pos_excursion),
            "min_elbow_deg": _quantiles(pos_min_elbow),
            "wrist_to_body_torsos": _quantiles(pos_body_dist),
            "active_wrist_conf": _quantiles(pos_wrist_conf),
        },
        "negative_normal_train": {
            "wrist_excursion_torsos": _quantiles(neg_excursion),
            "min_elbow_deg": _quantiles(neg_min_elbow),
            "wrist_to_body_torsos": _quantiles(neg_body_dist),
            "active_wrist_conf": _quantiles(neg_wrist_conf),
        },
    }
