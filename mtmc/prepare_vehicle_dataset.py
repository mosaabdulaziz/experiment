#!/usr/bin/env python3
"""Consolidate the reviewed vehicle YOLO labels into one train/val dataset.

The labelled sessions under ``mtmc/output/vehicle_yolo_dataset/a_REVIEWED`` carry
a single top-level ``classes.txt`` and **every** label file indexes into that
global class list (the per-folder ``labels/classes.txt`` are stale leftovers from
the old annotation sessions and are ignored).  This script normalises the labels
into the requested target space and emits a portable YOLO layout::

    <out>/images/{train,val}/<SESSION>__<frame>.jpg
    <out>/labels/{train,val}/<SESSION>__<frame>.txt
    <out>/data.yaml

Layout details:

  * ``frameNNNNN.jpg`` names are NOT unique across sessions, so every file is
    namespaced ``<SESSION>__<frame>``.
  * Images that have no label file are kept as empty-label negatives.
  * The split is PER-SESSION, not per-image: consecutive reviewed frames of one
    session are near-duplicates, so a random image split would leak between
    train and val and flatter the val metrics.  Val sessions are chosen with a
    greedy pass that makes sure every class keeps validation examples.
  * ``data.yaml`` omits ``path:`` on purpose -- ultralytics then resolves train/
    val relative to the yaml's own directory, so the folder stays portable when
    copied to another machine (e.g. the PRO6000 training box).

Modes:

  * default: keep the full global class space (7 classes).
  * ``--one-class NAME``: collapse every kept class to a single ``NAME`` (id 0).
  * ``--exclude NAME [NAME ...]``: drop boxes of these classes entirely; combined
    with ``--one-class`` a class such as ``Jarar`` becomes background.

Usage:
    # 7-class dataset (global scheme)
    .venv/bin/python -m mtmc.prepare_vehicle_dataset

    # 1-class 'vehicle' dataset, Jarar dropped: the deployed recipe's dataset, which
    # mtmc.finetune_vehicle_detector reads by default (versioned with DVC at this path)
    .venv/bin/python -m mtmc.prepare_vehicle_dataset \\
        --out products/el-mohandes/datasets/vehicle_detection_1cls \\
        --one-class vehicle --exclude Jarar

    # shrink images (max side) if the ~3 GB native frame set is too bulky to
    # transfer; labels are normalized so they do not change
    .venv/bin/python -m mtmc.prepare_vehicle_dataset --max-side 2560
"""
from __future__ import annotations

import argparse
import collections
import os
import random
import shutil
import sys
from pathlib import Path

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

DEFAULT_SOURCE = Path("mtmc/output/vehicle_yolo_dataset/a_REVIEWED")
DEFAULT_OUT = Path("mtmc/datasets/vehicle_detection")
GLOBAL_CLASSES_FILE = "classes.txt"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def load_class_space(src: Path):
    """Return the global class list from the top-level classes.txt."""
    with open(src / GLOBAL_CLASSES_FILE, encoding="utf-8") as f:
        names = [ln.strip() for ln in f if ln.strip()]
    if not names:
        raise SystemExit(f"no classes in {src / GLOBAL_CLASSES_FILE}")
    return names


def collect_session(session_dir: Path, global_names: list[str], exclude: set[str],
                    one_class: bool):
    """Yield (frame_stem, [boxes]) with boxes mapped into the target space.

    Label ids index the GLOBAL classes.txt (the per-folder classes.txt is stale
    and deliberately ignored).  Iterates the images dir so images without a
    label file become empty-label negatives.
    """
    images_dir = session_dir / "images"
    labels_dir = session_dir / "labels"
    if not images_dir.is_dir():
        return []
    items = []
    for img in sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS):
        stem = img.stem
        label_file = labels_dir / f"{stem}.txt"
        boxes = []
        if label_file.exists():
            for line in label_file.read_text(encoding="utf-8").splitlines():
                line = line.split()
                if not line:
                    continue
                try:
                    local_idx = int(line[0])
                    vals = [float(x) for x in line[1:5]]
                except ValueError as exc:
                    raise SystemExit(f"bad label {label_file}: {line}") from exc
                if len(vals) != 4:
                    raise SystemExit(f"bad label {label_file}: expected 5 cols, got {len(line)}")
                if local_idx >= len(global_names):
                    raise SystemExit(f"{label_file}: class id {local_idx} >= "
                                     f"{len(global_names)} classes in {GLOBAL_CLASSES_FILE}")
                gname = global_names[local_idx]
                if gname in exclude:
                    continue
                target = 0 if one_class else local_idx
                boxes.append([target, *vals])
        items.append((stem, boxes))
    return items


def class_counts(items):
    cnt = collections.Counter()
    for _stem, boxes in items:
        for b in boxes:
            cnt[int(b[0])] += 1
    return cnt


def greedy_val_sessions(session_meta, totals, n_val, targets, rng):
    """Pick n_val sessions covering every class's val target with the LEAST
    leakage of training data: score = rare-class benefit / session box count,
    so the tanker/Jarar sessions that also contain hundreds of other boxes are
    kept for training and only a small carrier is surrendered to val."""
    session_counts = {s: meta["counts"] for s, meta in session_meta.items()}
    session_totals = {s: meta["boxes"] for s, meta in session_meta.items()}
    cands = list(session_counts)
    chosen = []
    val_cnt = collections.Counter()

    def score(s):
        benefit = penalty = 0.0
        for c, v in session_counts[s].items():
            total_c = totals.get(c, 0) or 1
            gap = targets[c] - val_cnt.get(c, 0)
            if v and gap > 0:
                benefit += min(v, gap) / total_c
            # Penalize removing a large SHARE of a class from training: a
            # session holding many rare boxes must not be sacrificed for val.
            penalty += (v / total_c) ** 2
        return benefit / (1.0 + session_totals[s]) - penalty + 1e-9 * rng.random()

    for _ in range(n_val):
        if not cands:
            break
        best = max(cands, key=score)
        chosen.append(best)
        cands.remove(best)
        for c, v in session_counts[best].items():
            val_cnt[c] += v

    # Correction pass: unless a class is tiny, it must appear in BOTH splits.
    def _best_carrier(sessions, c):
        return min(sessions, key=lambda s: (abs(session_counts[s].get(c, 0) - targets[c]),
                                            session_totals[s]))

    for c in totals:
        if totals[c] < 6:  # too few to guarantee both sides
            continue
        if val_cnt.get(c, 0) == 0:
            pick = _best_carrier([s for s in cands if session_counts[s].get(c, 0)], c)
            chosen.append(pick)
            cands.remove(pick)
            for k, v in session_counts[pick].items():
                val_cnt[k] += v
        in_val = sum(session_counts[s].get(c, 0) for s in chosen)
        if totals[c] - in_val == 0:
            pick = _best_carrier([s for s in chosen if session_counts[s].get(c, 0)], c)
            chosen.remove(pick)
            cands.append(pick)
            for k, v in session_counts[pick].items():
                val_cnt[k] -= v
    return chosen


def write_item(out_dir: Path, split: str, session: str, stem: str, boxes,
               image_src: Path, max_side: int):
    img_dir = out_dir / "images" / split
    lbl_dir = out_dir / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    name = f"{session}__{stem}"
    dst_img = img_dir / f"{name}.jpg"
    if max_side > 0 and cv2 is not None:
        im = cv2.imread(str(image_src))
        if im is None:
            raise SystemExit(f"can't read {image_src}")
        h, w = im.shape[:2]
        scale = max_side / max(h, w)
        if scale < 1.0:
            im = cv2.resize(im, (round(w * scale), round(h * scale)),
                            interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dst_img), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
    else:
        shutil.copyfile(image_src, dst_img)
    with open(lbl_dir / f"{name}.txt", "w", encoding="utf-8") as f:
        for b in boxes:
            f.write("{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(*b))


def write_data_yaml(out_dir: Path, names, train_n, val_n, session_count):
    rows = [f"# Vehicle detection dataset (reviewed, {len(names)} classes)",
            f"# Sessions: {session_count}, images: {train_n} train / {val_n} val.",
            "train: images/train",
            "val: images/val",
            f"nc: {len(names)}",
            "names:"]
    rows += [f"  {i}: {name}" for i, name in enumerate(names)]
    (out_dir / "data.yaml").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=str, default=str(DEFAULT_SOURCE))
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--val-frac", type=float, default=0.15,
                    help="fraction of SESSIONS held out for val (default 0.15)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-side", type=int, default=0,
                    help="resize images so the max side is <= this; 0 = keep native "
                         "(default 0; labels are normalized, so resizing is safe)")
    ap.add_argument("--one-class", type=str, default="",
                    help="collapse all kept classes into a single class NAME (id 0); "
                         "default keeps the full global class space")
    ap.add_argument("--exclude", type=str, nargs="*", default=[],
                    help="drop these classes entirely (boxes become background)")
    args = ap.parse_args()

    source = Path(args.source).resolve()
    out = Path(args.out).resolve()
    if not source.is_dir():
        raise SystemExit(f"source not found: {source}")
    if out.exists():
        raise SystemExit(f"output already exists (refusing to clobber): {out}")

    rng = random.Random(args.seed)
    global_names = load_class_space(source)
    exclude = set(args.exclude)
    bad = [e for e in exclude if e not in global_names]
    if bad:
        raise SystemExit(f"classes to exclude not in global list {global_names}: {bad}")
    if args.one_class and len(args.one_class.split()) != 1:
        raise SystemExit("--one-class takes a single class name")
    target_names = [args.one_class] if args.one_class else [g for g in global_names
                                                            if g not in exclude]
    one_class = bool(args.one_class)
    print(f"source classes ({len(global_names)}): {global_names}")
    print(f"target ({len(target_names)} class(es)): {target_names}"
          + (f"  [excluded: {sorted(exclude)}]" if exclude else ""))

    sessions = sorted(d for d in source.iterdir() if d.is_dir())
    if not sessions:
        raise SystemExit(f"no session folders under {source}")

    all_items = {}
    totals = collections.Counter()
    n_boxes = 0
    n_empty = 0
    for s in sessions:
        items = collect_session(s, global_names, exclude, one_class)
        if not items:
            print(f"  WARNING: {s.name} has no usable images; skipping")
            continue
        all_items[s.name] = items
        cnt = class_counts(items)
        totals.update(cnt)
        n_boxes += sum(cnt.values())
        n_empty += sum(1 for _st, bb in items if not bb)

    if not all_items:
        raise SystemExit("no usable sessions found")
    n_val = max(1, round(args.val_frac * len(all_items)))
    targets = {c: min(max(3, round(args.val_frac * t) // 2), t // 3)
               for c, t in totals.items()}
    session_meta = {s: {"counts": class_counts(it), "boxes": sum(len(bb) for _st, bb in it)}
                    for s, it in all_items.items()}
    val_sessions = set(greedy_val_sessions(session_meta, totals, n_val, targets, rng))
    train_sessions = [s for s in all_items if s not in val_sessions]
    val_sessions = sorted(val_sessions)
    if not train_sessions or not val_sessions:
        raise SystemExit("split produced an empty train or val set")

    print(f"sessions: {len(all_items)} total, {len(train_sessions)} train, "
          f"{len(val_sessions)} val, {n_boxes} boxes, {n_empty} empty-label images")
    print(f"val sessions: {val_sessions}")

    train_n = val_n = 0
    for s in train_sessions:
        for stem, boxes in all_items[s]:
            img = source / s / "images" / f"{stem}.jpg"
            if not img.exists():
                img = source / s / "images" / f"{stem}.png"
            write_item(out, "train", s, stem, boxes, img, args.max_side)
            train_n += 1
    for s in val_sessions:
        for stem, boxes in all_items[s]:
            img = source / s / "images" / f"{stem}.jpg"
            if not img.exists():
                img = source / s / "images" / f"{stem}.png"
            write_item(out, "val", s, stem, boxes, img, args.max_side)
            val_n += 1

    write_data_yaml(out, target_names, train_n, val_n, len(all_items))

    def split_counts(ss):
        c = collections.Counter()
        for s in ss:
            for stem, boxes in all_items[s]:
                for b in boxes:
                    c[int(b[0])] += 1
        return c

    tr, va = split_counts(train_sessions), split_counts(val_sessions)
    print(f"\n{'class':<14} {'train':>6} {'val':>6} {'total':>6}")
    for i, name in enumerate(target_names):
        print(f"{name:<14} {tr.get(i,0):>6} {va.get(i,0):>6} {totals.get(i,0):>6}")
    print(f"\ndataset ready at {out}")
    print(f"  images: {train_n} train / {val_n} val    data.yaml: {out / 'data.yaml'}")
    print("next: train with ultralytics "
          f"`yolo detect train data={out / 'data.yaml'} model=yolo11s.pt imgsz=1280`")


if __name__ == "__main__":
    main()