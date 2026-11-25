"""Cross-camera vehicle re-ID evaluation dataset.

Packs the per-camera vehicle crops (``mtmc/output/vehicle_crops/<cam>/``) plus
the hand-curated same-vehicle ground truth (``.../chosen crops/vehicle_N/``)
into an in-memory dataset of *instances*:

* a **labeled instance** is one (identity, camera, local tracklet) triple — all
  crops the curator picked for that vehicle/camera/tracklet;
* an **unknown instance** is every other (camera, tracklet) in the manifests —
  used as realistic gallery distractors / negative candidates (sampled to a few
  crops per tracklet so near-duplicate frames don't flood the gallery).

Ground-truth caveat (accepted at the time of writing): the curated set labels
only ~10 vehicles across a handful of cameras.  Unknown tracklets on cameras
where a curated vehicle was NOT labeled may still be that same physical car —
the benchmark therefore treats them as negatives, which makes every metric
*pessimistic* (real matches counted as false positives).  Report reads should
keep that in mind; FP montages are dumped so such cases can be eyeballed.

Crop identity: a crop is referenced as ``"<cam>/<filename>"`` and always
resolves to ``<vehicle_crops_root>/<cam>/crops/<filename>``.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

CHOSEN_CROPS_DIR_NAME = "chosen crops"


@dataclass
class Instance:
    """One (identity | None, camera, tracklet) group of crops."""

    uid: str            # "cam/tid" for unknowns, "identity/cam/tid" for labeled
    cam: str
    tid: int
    identity: str | None = None   # None => unknown vehicle (distractor/negative)
    crop_keys: list[str] = field(default_factory=list)   # "cam/filename"


@dataclass
class ReidDataset:
    root: Path
    cameras: list[str]
    instances: list[Instance]
    crop_root: dict[str, Path]          # "cam/filename" -> full path
    labeled_instances: list[Instance]   # identity is not None
    unknown_instances: list[Instance]   # identity is None
    identities: list[str]               # sorted identity ids
    n_dropped_ambiguous_crops: int = 0  # chosen files with no resolvable camera

    def instance_by_uid(self) -> dict[str, Instance]:
        return {inst.uid: inst for inst in self.instances}


def _scan_cameras(root: Path) -> list[str]:
    """Camera dirs = subdirs of root that contain a vehicles.csv + crops/."""
    cams = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if d.name == CHOSEN_CROPS_DIR_NAME:
            continue
        if (d / "vehicles.csv").exists() and (d / "crops").is_dir():
            cams.append(d.name)
    return cams


def _tracklets_per_camera(root: Path, cam: str) -> dict[int, list[dict]]:
    """Read vehicles.csv: track_id -> ordered rows (frame, crop_file, conf...)."""
    out: dict[int, list[dict]] = {}
    with open(root / cam / "vehicles.csv", newline="") as f:
        for row in csv.DictReader(f):
            out.setdefault(int(row["track_id"]), []).append(row)
    return out


def _sample_rows(rows: list[dict], k: int) -> list[dict]:
    """Evenly spread k rows across a tracklet's crop list (near-dup control)."""
    n = len(rows)
    if n <= k:
        return rows
    idx = {int(round(i * (n - 1) / (k - 1))) for i in range(k)}
    return [rows[i] for i in sorted(idx)]


def build_dataset(
    root: str | Path = "mtmc/output/vehicle_crops",
    unknown_crops_per_tracklet: int = 3,
    min_crops: int = 1,
) -> ReidDataset:
    """Load the crops + curated GT into a ReidDataset.

    Parameters
    ----------
    root : vehicle_crops directory (contains per-camera dirs + 'chosen crops/').
    unknown_crops_per_tracklet : max crops kept per unknown tracklet (sampled
        evenly across its frames; every crop kept for labeled instances).
    min_crops : instances with fewer usable crop files are dropped.
    """
    root = Path(root)
    cams = _scan_cameras(root)
    if not cams:
        raise FileNotFoundError(
            f"No camera directories (vehicles.csv + crops/) under {root}"
        )

    # full crop index cam -> filename -> exists
    crop_root: dict[str, Path] = {}
    for cam in cams:
        d = root / cam / "crops"
        for f in d.iterdir():
            if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                crop_root[f"{cam}/{f.name}"] = f

    tracklets = {cam: _tracklets_per_camera(root, cam) for cam in cams}

    # filename -> owning camera(s).  (tid, frame) restarts per camera, so a
    # filename can exist in several cameras; the curated crops are byte-copies
    # of the file in the source camera, which disambiguates them (see below).
    by_filename: dict[str, list[str]] = {}
    for key in crop_root:
        cam, _, name = key.partition("/")
        by_filename.setdefault(name, []).append(cam)

    def _resolve_cam(chosen_file: Path) -> str | None:
        """Find the camera whose crops dir holds the file this copy came from."""
        cands = by_filename.get(chosen_file.name, [])
        if not cands:
            return None
        if len(cands) == 1:
            return cands[0]
        # same name in several cameras — compare bytes with each candidate;
        # the source copy is byte-identical, coincidental names are not.
        want = chosen_file.read_bytes()
        hits = [c for c in cands
                if (root / c / "crops" / chosen_file.name).read_bytes() == want]
        return hits[0] if len(hits) == 1 else None

    # ---- curated ground truth -------------------------------------------
    chosen = root / CHOSEN_CROPS_DIR_NAME
    labeled: list[Instance] = []
    identities: set[str] = set()
    covered: set[tuple[str, int]] = set()  # (cam, tid) consumed by GT
    n_ambiguous = 0
    if chosen.is_dir():
        for vdir in sorted(p for p in chosen.iterdir() if p.is_dir()):
            identity = vdir.name
            # group chosen crops by (cam, tid)
            groups: dict[tuple[str, int], list[str]] = {}
            for f in sorted(vdir.iterdir()):
                if f.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                cam = _resolve_cam(f)
                if cam is None:
                    n_ambiguous += 1
                    continue  # not found / unresolved duplicate — drop from GT
                tid = int(f.name.split("_")[0][1:])
                groups.setdefault((cam, tid), []).append(f"{cam}/{f.name}")
            for (cam, tid), keys in sorted(groups.items()):
                covered.add((cam, tid))
                inst = Instance(
                    uid=f"{identity}/{cam}/t{tid:04d}",
                    cam=cam, tid=tid, identity=identity,
                    crop_keys=sorted(keys),
                )
                if len(keys) >= min_crops:
                    labeled.append(inst)
                    identities.add(identity)

    # ---- unknown instances (everything else, sampled) --------------------
    unknown: list[Instance] = []
    for cam in cams:
        for tid, rows in tracklets[cam].items():
            if (cam, tid) in covered:
                continue
            picked = _sample_rows(rows, unknown_crops_per_tracklet)
            keys = [f"{cam}/{r['crop_file']}" for r in picked
                    if f"{cam}/{r['crop_file']}" in crop_root]
            if len(keys) < min_crops:
                continue
            unknown.append(Instance(
                uid=f"{cam}/t{tid:04d}",
                cam=cam, tid=tid, identity=None,
                crop_keys=sorted(keys),
            ))

    labeled.sort(key=lambda i: (i.identity or "", i.cam, i.tid))
    unknown.sort(key=lambda i: (i.cam, i.tid))
    # Keep only the crops actually referenced by instances (labeled all, unknown
    # sampled) so embedding never touches the ~15k unreferenced files.
    used_keys = {k for inst in labeled + unknown for k in inst.crop_keys}
    crop_root = {k: p for k, p in crop_root.items() if k in used_keys}
    return ReidDataset(
        root=root,
        cameras=cams,
        instances=labeled + unknown,
        crop_root=crop_root,
        labeled_instances=labeled,
        unknown_instances=unknown,
        identities=sorted(identities),
        n_dropped_ambiguous_crops=n_ambiguous,
    )


def gt_summary(ds: ReidDataset) -> dict:
    """Counts that put the eval in context (queries/positives/negatives...)."""
    by_id: dict[str, set[str]] = {}
    for inst in ds.labeled_instances:
        by_id.setdefault(inst.identity, set()).add(inst.cam)
    multi = {k: v for k, v in by_id.items() if len(v) >= 2}
    # positive cross-cam labeled instance pairs
    n_pos_pairs = 0
    lab = ds.labeled_instances
    for i in range(len(lab)):
        for j in range(i + 1, len(lab)):
            a, b = lab[i], lab[j]
            if a.cam != b.cam and a.identity == b.identity:
                n_pos_pairs += 1
    return {
        "cameras": ds.cameras,
        "n_instances": len(ds.instances),
        "n_labeled_instances": len(ds.labeled_instances),
        "n_unknown_instances": len(ds.unknown_instances),
        "n_identities": len(ds.identities),
        "n_multi_camera_identities": len(multi),
        "multi_camera_identities": {k: sorted(v) for k, v in sorted(multi.items())},
        "n_positive_cross_cam_pairs": n_pos_pairs,
        "note": (
            "Unknown tracklets are treated as negatives although some may be "
            "unlabeled appearances of curated vehicles => metrics are pessimistic."
        ),
    }


if __name__ == "__main__":
    ds = build_dataset()
    import json
    print(json.dumps(gt_summary(ds), indent=2, default=str))
