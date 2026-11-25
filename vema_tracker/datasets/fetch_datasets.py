"""Download and verify lightweight MOT17, DanceTrack (YOLOX-X), and MOT20 benchmark files."""

from __future__ import annotations

import argparse
import io
import json
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MOT17_SEQS = ["MOT17-02", "MOT17-04", "MOT17-05", "MOT17-09", "MOT17-10", "MOT17-11", "MOT17-13"]
DANCETRACK_VAL_SEQS = [
    "dancetrack0004", "dancetrack0005", "dancetrack0007", "dancetrack0010", "dancetrack0014",
    "dancetrack0018", "dancetrack0019", "dancetrack0025", "dancetrack0026", "dancetrack0030",
    "dancetrack0034", "dancetrack0035", "dancetrack0041", "dancetrack0043", "dancetrack0047",
    "dancetrack0058", "dancetrack0063", "dancetrack0065", "dancetrack0073", "dancetrack0077",
    "dancetrack0079", "dancetrack0081", "dancetrack0090", "dancetrack0094", "dancetrack0097",
]


def _fetch_bytes(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def download_mot17(data_root: Path) -> None:
    print("Downloading MOT17 train annotations & public detections (FRCNN + SDP)...")
    tasks: list[tuple[str, Path]] = []
    for base_seq in MOT17_SEQS:
        frcnn_name = f"{base_seq}-FRCNN"
        frcnn_dir = data_root / "MOT17" / "train" / frcnn_name
        for rel in ("seqinfo.ini", "gt/gt.txt", "det/det.txt"):
            dst = frcnn_dir / rel
            if not dst.exists():
                url = f"https://huggingface.co/datasets/ling1016/MOT17/resolve/main/train/{frcnn_name}/{rel}"
                tasks.append((url, dst))

        sdp_name = f"{base_seq}-SDP"
        sdp_det = data_root / "MOT17" / "train" / sdp_name / "det" / "det.txt"
        if not sdp_det.exists():
            url = f"https://huggingface.co/datasets/ling1016/MOT17/resolve/main/train/{sdp_name}/det/det.txt"
            tasks.append((url, sdp_det))

    def _dl(item: tuple[str, Path]) -> None:
        url, dst = item
        data = _fetch_bytes(url)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)

    if tasks:
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(_dl, tasks))

    # Share seqinfo.ini & gt/gt.txt from FRCNN into SDP sequence directories
    for base_seq in MOT17_SEQS:
        frcnn_dir = data_root / "MOT17" / "train" / f"{base_seq}-FRCNN"
        sdp_dir = data_root / "MOT17" / "train" / f"{base_seq}-SDP"
        for rel in ("seqinfo.ini", "gt/gt.txt"):
            src = frcnn_dir / rel
            dst = sdp_dir / rel
            if src.exists() and not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(src.read_bytes())

    print(f"[OK] MOT17 ready ({len(MOT17_SEQS) * 2} detector sequences).")


def download_mot20(data_root: Path) -> None:
    target = data_root / "MOT20" / "train" / "MOT20-01" / "gt" / "gt.txt"
    if target.exists():
        print("[OK] MOT20 already present.")
        return
    print("Downloading MOT20Labels.zip (~19 MB)...")
    url = "https://huggingface.co/datasets/mrhwl123/MOT20/resolve/main/MOT20Labels.zip"
    zf = zipfile.ZipFile(io.BytesIO(_fetch_bytes(url, timeout=120)))
    for member in zf.namelist():
        if not member.startswith("MOT20Labels/train/") or member.endswith("/"):
            continue
        rel = member[len("MOT20Labels/"):]
        dst = data_root / "MOT20" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(zf.read(member))
    print("[OK] MOT20 ready (4 sequences).")


def download_dancetrack_val(data_root: Path) -> None:
    print("Downloading DanceTrack-val GT + seqinfo (25 sequences)...")
    tasks: list[tuple[str, Path]] = []
    for seq in DANCETRACK_VAL_SEQS:
        seq_dir = data_root / "DanceTrack" / "val" / seq
        for rel in ("seqinfo.ini", "gt/gt.txt"):
            dst = seq_dir / rel
            if not dst.exists():
                url = f"https://huggingface.co/datasets/wanda619/motipDataset/resolve/main/DanceTrack/val/{seq}/{rel}"
                tasks.append((url, dst))

    def _dl(item: tuple[str, Path]) -> None:
        url, dst = item
        data = _fetch_bytes(url)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)

    if tasks:
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(_dl, tasks))

    # Materialize YOLOX-X detections from MOTRv2 det_db_motrv2.zip (~21 MB)
    sample_det = data_root / "DanceTrack" / "val" / DANCETRACK_VAL_SEQS[0] / "det" / "det.txt"
    if not sample_det.exists():
        print("Downloading DanceTrack YOLOX-X detections (det_db_motrv2.zip, ~21 MB)...")
        g_url = "https://drive.usercontent.google.com/download?id=1cdhtztG4dbj7vzWSVSehLL6s0oPalEJo&export=download&confirm=t"
        zf = zipfile.ZipFile(io.BytesIO(_fetch_bytes(g_url, timeout=120)))
        db: dict[str, list[str]] = json.loads(zf.read(zf.namelist()[0]).decode("utf-8"))

        # Group by sequence name and frame id
        by_seq: dict[str, list[tuple[int, str]]] = {s: [] for s in DANCETRACK_VAL_SEQS}
        for key, det_lines in db.items():
            if not key.startswith("DanceTrack/val/"):
                continue
            parts = key.split("/")
            if len(parts) < 5:
                continue
            seq_name = parts[2]
            if seq_name not in by_seq:
                continue
            frame_id = int(Path(parts[4]).stem)
            for dl in det_lines:
                vals = dl.strip().split(",")
                if len(vals) >= 5:
                    x, y, w, h, score = vals[:5]
                    by_seq[seq_name].append(
                        (frame_id, f"{frame_id},-1,{x},{y},{w},{h},{score},-1,-1,-1\n")
                    )

        for seq_name, rows in by_seq.items():
            rows.sort(key=lambda t: t[0])
            det_out = data_root / "DanceTrack" / "val" / seq_name / "det" / "det.txt"
            det_out.parent.mkdir(parents=True, exist_ok=True)
            with open(det_out, "w", encoding="utf-8") as f:
                f.writelines(r[1] for r in rows)

    print(f"[OK] DanceTrack-val ready ({len(DANCETRACK_VAL_SEQS)} sequences with YOLOX-X detections).")


def download_dancetrack_train(data_root: Path) -> None:
    print("Downloading DanceTrack-train GT + seqinfo (40 sequences)...")
    seqmap_url = "https://huggingface.co/datasets/wanda619/motipDataset/resolve/main/DanceTrack/train_seqmap.txt"
    train_seqs = [
        line.strip()
        for line in _fetch_bytes(seqmap_url).decode("utf-8").splitlines()
        if line.strip() and line.strip() != "name"
    ]
    tasks: list[tuple[str, Path]] = []
    for seq in train_seqs:
        seq_dir = data_root / "DanceTrack" / "train" / seq
        for rel in ("seqinfo.ini", "gt/gt.txt"):
            dst = seq_dir / rel
            if not dst.exists():
                url = f"https://huggingface.co/datasets/wanda619/motipDataset/resolve/main/DanceTrack/train/{seq}/{rel}"
                tasks.append((url, dst))

    def _dl(item: tuple[str, Path]) -> None:
        url, dst = item
        data = _fetch_bytes(url)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)

    if tasks:
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(_dl, tasks))

    sample_det = data_root / "DanceTrack" / "train" / train_seqs[0] / "det" / "det.txt"
    if not sample_det.exists():
        print("Downloading DanceTrack YOLOX-X detections for train split...")
        g_url = "https://drive.usercontent.google.com/download?id=1cdhtztG4dbj7vzWSVSehLL6s0oPalEJo&export=download&confirm=t"
        zf = zipfile.ZipFile(io.BytesIO(_fetch_bytes(g_url, timeout=120)))
        db: dict[str, list[str]] = json.loads(zf.read(zf.namelist()[0]).decode("utf-8"))

        by_seq: dict[str, list[tuple[int, str]]] = {s: [] for s in train_seqs}
        for key, det_lines in db.items():
            if not key.startswith("DanceTrack/train/"):
                continue
            parts = key.split("/")
            if len(parts) < 5:
                continue
            seq_name = parts[2]
            if seq_name not in by_seq:
                continue
            frame_id = int(Path(parts[4]).stem)
            for dl in det_lines:
                vals = dl.strip().split(",")
                if len(vals) >= 5:
                    x, y, w, h, score = vals[:5]
                    by_seq[seq_name].append(
                        (frame_id, f"{frame_id},-1,{x},{y},{w},{h},{score},-1,-1,-1\n")
                    )

        for seq_name, rows in by_seq.items():
            rows.sort(key=lambda t: t[0])
            det_out = data_root / "DanceTrack" / "train" / seq_name / "det" / "det.txt"
            det_out.parent.mkdir(parents=True, exist_ok=True)
            with open(det_out, "w", encoding="utf-8") as f:
                f.writelines(r[1] for r in rows)

    print(f"[OK] DanceTrack-train ready ({len(train_seqs)} sequences with YOLOX-X detections).")


def check_datasets(data_root: Path) -> None:
    specs = [
        ("MOT17-val (half-split)", data_root / "MOT17" / "train", "https://motchallenge.net/data/MOT17/"),
        ("DanceTrack-val (YOLOX-X)", data_root / "DanceTrack" / "val", "https://github.com/DanceTrack/DanceTrack"),
        ("MOT20-val (half-split)", data_root / "MOT20" / "train", "https://motchallenge.net/data/MOT20/"),
    ]
    print(f"Checking dataset root: {data_root.resolve()}\n")
    for label, path, url in specs:
        if path.exists():
            seqs = [p.name for p in sorted(path.iterdir()) if p.is_dir()]
            print(f"[OK]      {label:26s} -> {len(seqs)} sequences found at {path}")
        else:
            print(f"[MISSING] {label:26s} -> expected at {path}")
            print(f"          Download via `--download` or from: {url}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download or check public MOT benchmark files.")
    parser.add_argument("--data-dir", type=Path, default=Path("vema_tracker/data"))
    parser.add_argument("--check", action="store_true", help="Check expected directories.")
    parser.add_argument(
        "--download",
        nargs="*",
        choices=["all", "mot17", "dancetrack", "dancetrack-train", "mot20"],
        help="Download lightweight GT + precomputed detections.",
    )
    args = parser.parse_args()

    if args.download is not None:
        targets = set(args.download or ["all"])
        if "all" in targets or "mot17" in targets:
            download_mot17(args.data_dir)
        if "all" in targets or "dancetrack" in targets:
            download_dancetrack_val(args.data_dir)
        if "all" in targets or "dancetrack-train" in targets:
            download_dancetrack_train(args.data_dir)
        if "all" in targets or "mot20" in targets:
            download_mot20(args.data_dir)

    check_datasets(args.data_dir)


if __name__ == "__main__":
    main()
