#!/usr/bin/env python3
"""Re-embed stored face galleries after upgrading the recognition model.

Replaces stale embeddings (e.g. buffalo_l) in Redis galleries (``iris:face:gallery``,
``attendance:face:gallery``) and/or from the Attendance SQLite roster using the active
commercial model pack (default: ``auraface``), reading the stored enrollment images on
disk with no manual re-registration required.

For each gallery identity it resolves stored images in order:
  1. ``entry["photo_path"]`` (supporting both host and container ``/app/...``, ``/photos/...`` paths)
  2. ``<photos-dir>/<face_id>.{jpg,jpeg,png}``
  3. Extra multi-angle photos in ``<photos-dir>/multi/<face_id>/*.{jpg,jpeg,png}``
  4. Assigned CCTV/upload captures in ``iris:face:captures`` (unless ``--no-captures``)

Entries are updated atomically per identity only when at least one stored image yields a
valid embedding; identities without readable images are left untouched and reported.

Usage:
    # Migrate default Redis galleries + Attendance DB on CPU (e.g. inside `iris` container):
    python3 scripts/migrate_face_gallery.py

    # Dry-run inside the GPU sidecar container:
    docker compose exec deepstream-sidecar python3 scripts/migrate_face_gallery.py --gpu --dry-run

    # Target one gallery key only:
    python3 scripts/migrate_face_gallery.py --gallery-key iris:face:gallery --attendance-db ""
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from perception import insightface_pack  # noqa: E402

logger = logging.getLogger("iris.migrate_face_gallery")

DEFAULT_REDIS_URL = "redis://localhost:6379/0"
DEFAULT_GALLERY_KEYS = ("iris:face:gallery", "attendance:face:gallery")
DEFAULT_CAPTURES_KEY = "iris:face:captures"
DEFAULT_CAPTURE_IMG_PREFIX = "iris:face:capture_img:"
IMAGE_EXTS = (".jpg", ".jpeg", ".png")
CONTAINER_PHOTO_PREFIXES = (
    "/app/data/face_photos/",
    "/photos/",
    "data/face_photos/",
)
CONTAINER_CAPTURE_PREFIXES = (
    "/app/data/face_captures/",
    "/workspace/iris/data/face_captures/",
    "data/face_captures/",
)

EmbedFn = Callable[[np.ndarray], np.ndarray | None]


@dataclass
class MigrationStats:
    total: int = 0
    migrated: int = 0
    already_migrated: int = 0
    would_migrate: int = 0
    skipped_no_images: list[str] = field(default_factory=list)
    failed_no_face: list[str] = field(default_factory=list)
    captures_updated: int = 0

    @property
    def unresolved(self) -> int:
        return len(self.skipped_no_images) + len(self.failed_no_face)


def _decode_str(raw: bytes | str) -> str:
    return raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)


def _normalize_vector(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    return arr / norm if norm > 1e-12 else arr


def _translate_container_path(raw_path: str, target_dir: Path, prefixes: Sequence[str]) -> Path | None:
    if not raw_path:
        return None
    norm = raw_path.replace("\\", "/")
    direct = Path(raw_path)
    if direct.is_file():
        return direct
    for prefix in prefixes:
        if norm.startswith(prefix):
            rel = norm[len(prefix) :].lstrip("/")
            candidate = target_dir / rel
            if candidate.is_file():
                return candidate
    rel_direct = REPO_ROOT / norm.lstrip("/")
    if rel_direct.is_file():
        return rel_direct
    return None


def resolve_photo_paths(face_id: str, entry: dict, photos_dir: Path) -> list[Path]:
    """Return deduplicated enrollment image paths on disk for *face_id*."""
    found: list[Path] = []
    seen: set[Path] = set()

    def _add(p: Path | None) -> None:
        if p is None or not p.is_file():
            return
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            found.append(p)

    _add(_translate_container_path(str(entry.get("photo_path", "")), photos_dir, CONTAINER_PHOTO_PREFIXES))
    for ext in IMAGE_EXTS:
        candidate = photos_dir / f"{face_id}{ext}"
        if candidate.is_file():
            _add(candidate)
            break

    multi_dir = photos_dir / "multi" / face_id
    if multi_dir.is_dir():
        for child in sorted(multi_dir.iterdir(), key=lambda p: p.name.lower()):
            if child.is_file() and child.suffix.lower() in IMAGE_EXTS:
                _add(child)

    return found


def _reembed_assigned_captures(
    r,
    captures_key: str,
    capture_img_prefix: str,
    captures_dir: Path,
    person_name: str,
    model_name: str,
    embed_fn: EmbedFn,
) -> tuple[list[list[float]], list[tuple[str, str]]]:
    """Re-embed captures assigned to *person_name*; returns (embeddings, redis updates)."""
    if not person_name:
        return [], []
    raw_caps = r.hgetall(captures_key)
    if not raw_caps:
        return [], []

    cap_embs: list[list[float]] = []
    cap_updates: list[tuple[str, str]] = []
    for cap_id_raw, cap_val_raw in sorted(raw_caps.items(), key=lambda kv: _decode_str(kv[0])):
        cap_id = _decode_str(cap_id_raw)
        try:
            cap_entry = json.loads(_decode_str(cap_val_raw))
        except json.JSONDecodeError:
            continue
        if cap_entry.get("assigned_to") != person_name:
            continue

        img: np.ndarray | None = None
        cached = r.get(f"{capture_img_prefix}{cap_id}")
        if cached is not None:
            buf = np.frombuffer(cached if isinstance(cached, bytes) else cached.encode("latin1"), np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            for ext in IMAGE_EXTS:
                disk_path = captures_dir / f"{cap_id}{ext}"
                if disk_path.is_file():
                    img = cv2.imread(str(disk_path))
                    break
        if img is None:
            continue

        vec = embed_fn(img)
        if vec is None:
            continue
        emb_list = _normalize_vector(vec).tolist()
        cap_embs.append(emb_list)
        cap_entry["embedding"] = emb_list
        cap_entry["model"] = model_name
        cap_updates.append((cap_id, json.dumps(cap_entry)))

    return cap_embs, cap_updates


def build_embedder(model_name: str = insightface_pack.DEFAULT_PACK, gpu: bool = False, det_thresh: float = 0.4) -> EmbedFn:
    """Initialize InsightFace ``FaceAnalysis`` on *model_name* and return a largest-face embedder."""
    from insightface.app import FaceAnalysis

    pack_root = insightface_pack.cli_root(model_name)
    app = FaceAnalysis(name=model_name, root=pack_root, allowed_modules=["detection", "recognition"])
    app.prepare(ctx_id=0 if gpu else -1, det_thresh=det_thresh, det_size=(640, 640))

    def _embed(img: np.ndarray) -> np.ndarray | None:
        faces = app.get(img)
        if not faces:
            return None
        faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
        face = faces[0]
        raw = getattr(face, "normed_embedding", None)
        if raw is None:
            raw = getattr(face, "embedding", None)
        return None if raw is None else _normalize_vector(raw)

    return _embed


def migrate_redis_gallery(
    r,
    gallery_key: str,
    photos_dir: Path,
    embed_fn: EmbedFn,
    *,
    model_name: str = insightface_pack.DEFAULT_PACK,
    captures_key: str = DEFAULT_CAPTURES_KEY,
    capture_img_prefix: str = DEFAULT_CAPTURE_IMG_PREFIX,
    captures_dir: Path | None = None,
    include_captures: bool = True,
    force: bool = False,
    dry_run: bool = False,
) -> MigrationStats:
    """Re-embed every identity in Redis *gallery_key* from stored enrollment images."""
    stats = MigrationStats()
    raw_entries = r.hgetall(gallery_key)
    if not raw_entries:
        return stats

    captures_root = captures_dir or (photos_dir.parent / "face_captures")
    for raw_fid, raw_val in sorted(raw_entries.items(), key=lambda kv: _decode_str(kv[0])):
        face_id = _decode_str(raw_fid)
        stats.total += 1
        try:
            entry = json.loads(_decode_str(raw_val))
        except json.JSONDecodeError:
            logger.warning("[%s] %s: invalid JSON entry, skipping", gallery_key, face_id)
            stats.skipped_no_images.append(face_id)
            continue

        if not force and entry.get("model") == model_name:
            stats.already_migrated += 1
            continue

        name = str(entry.get("name", face_id))
        photo_paths = resolve_photo_paths(face_id, entry, photos_dir)

        cap_embs: list[list[float]] = []
        cap_updates: list[tuple[str, str]] = []
        if include_captures:
            cap_embs, cap_updates = _reembed_assigned_captures(
                r, captures_key, capture_img_prefix, captures_root, name, model_name, embed_fn
            )

        if not photo_paths and not cap_embs:
            logger.warning("[%s] SKIP %s (%s): no stored enrollment images found", gallery_key, face_id, name)
            stats.skipped_no_images.append(face_id)
            continue

        new_embeddings: list[list[float]] = []
        for path in photo_paths:
            img = cv2.imread(str(path))
            if img is None:
                continue
            vec = embed_fn(img)
            if vec is not None:
                new_embeddings.append(_normalize_vector(vec).tolist())
        new_embeddings.extend(cap_embs)

        if not new_embeddings:
            logger.warning("[%s] SKIP %s (%s): no face detected in %d stored image(s)", gallery_key, face_id, name, len(photo_paths))
            stats.failed_no_face.append(face_id)
            continue

        if dry_run:
            stats.would_migrate += 1
            logger.info(
                "[%s] WOULD MIGRATE %s (%s): %d photo(s) -> %d embedding(s)",
                gallery_key, face_id, name, len(photo_paths), len(new_embeddings),
            )
            continue

        updated = dict(entry)
        updated.pop("embedding", None)
        updated["embeddings"] = new_embeddings
        updated["model"] = model_name
        updated["migrated_at"] = time.time()
        r.hset(gallery_key, face_id, json.dumps(updated))
        for cap_id, cap_json in cap_updates:
            r.hset(captures_key, cap_id, cap_json)
        stats.migrated += 1
        stats.captures_updated += len(cap_updates)
        logger.info("[%s] MIGRATED %s (%s): %d embedding(s)", gallery_key, face_id, name, len(new_embeddings))

    return stats


def migrate_attendance_db(
    r,
    db_path: Path,
    photos_dir: Path,
    embed_fn: EmbedFn,
    *,
    gallery_key: str = "attendance:face:gallery",
    model_name: str = insightface_pack.DEFAULT_PACK,
    force: bool = False,
    dry_run: bool = False,
) -> MigrationStats:
    """Ensure active employees in *db_path* are embedded in *gallery_key* with *model_name*."""
    stats = MigrationStats()
    if not db_path.is_file():
        return stats

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        employees = conn.execute("SELECT id, name, gallery_rec_id FROM employee WHERE active = 1").fetchall()
    finally:
        conn.close()

    for emp in employees:
        face_id = str(emp["gallery_rec_id"])
        name = str(emp["name"])
        stats.total += 1

        existing_raw = r.hget(gallery_key, face_id)
        existing_entry: dict = {}
        if existing_raw is not None:
            try:
                existing_entry = json.loads(_decode_str(existing_raw))
            except json.JSONDecodeError:
                existing_entry = {}
            if not force and existing_entry.get("model") == model_name:
                stats.already_migrated += 1
                continue

        photo_paths = resolve_photo_paths(face_id, existing_entry, photos_dir)
        if not photo_paths:
            stats.skipped_no_images.append(face_id)
            continue

        new_embeddings: list[list[float]] = []
        for path in photo_paths:
            img = cv2.imread(str(path))
            if img is None:
                continue
            vec = embed_fn(img)
            if vec is not None:
                new_embeddings.append(_normalize_vector(vec).tolist())

        if not new_embeddings:
            stats.failed_no_face.append(face_id)
            continue

        if dry_run:
            stats.would_migrate += 1
            continue

        updated = {
            "name": name,
            "embeddings": new_embeddings,
            "photo_path": str(photo_paths[0]),
            "created_at": existing_entry.get("created_at", time.time()),
            "model": model_name,
            "migrated_at": time.time(),
        }
        r.hset(gallery_key, face_id, json.dumps(updated))
        stats.migrated += 1

    return stats


def _print_summary(label: str, stats: MigrationStats, dry_run: bool) -> None:
    action = f"would_migrate={stats.would_migrate}" if dry_run else f"migrated={stats.migrated}"
    print(
        f"[{label}] total={stats.total} {action} already_migrated={stats.already_migrated} "
        f"skipped_no_images={len(stats.skipped_no_images)} failed_no_face={len(stats.failed_no_face)}"
        + (f" captures_updated={stats.captures_updated}" if stats.captures_updated else "")
    )
    if stats.skipped_no_images:
        print(f"  skipped (no images): {', '.join(stats.skipped_no_images)}")
    if stats.failed_no_face:
        print(f"  failed (no face detected): {', '.join(stats.failed_no_face)}")


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--redis-url", default=os.environ.get("REDIS_URL", DEFAULT_REDIS_URL))
    ap.add_argument(
        "--gallery-key",
        action="append",
        dest="gallery_keys",
        default=None,
        help="Redis gallery hash key to migrate (repeatable; default: iris:face:gallery and attendance:face:gallery)",
    )
    ap.add_argument("--photos-dir", type=Path, default=REPO_ROOT / "data" / "face_photos")
    ap.add_argument("--captures-dir", type=Path, default=REPO_ROOT / "data" / "face_captures")
    ap.add_argument(
        "--attendance-db",
        default=str(REPO_ROOT / "products" / "attendance" / "data" / "attendance.db"),
        help="Optional Attendance SQLite DB to ensure active employees are embedded (empty string disables)",
    )
    ap.add_argument("--model", default=insightface_pack.DEFAULT_PACK, help="InsightFace pack name (default: %(default)s)")
    ap.add_argument("--det-thresh", type=float, default=0.4)
    ap.add_argument("--gpu", action="store_true", help="Run InsightFace on GPU (ctx_id=0)")
    ap.add_argument("--force", action="store_true", help="Re-embed even entries already tagged with --model")
    ap.add_argument("--dry-run", action="store_true", help="Inspect galleries without writing to Redis")
    ap.add_argument("--no-captures", action="store_true", help="Do not re-embed assigned captures in iris:face:captures")
    ap.add_argument("--strict", action="store_true", help="Exit 2 if any identity has missing images or no detected face")
    args = ap.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")

    import redis

    r = redis.Redis.from_url(args.redis_url)
    r.ping()

    embed_fn = build_embedder(args.model, gpu=args.gpu, det_thresh=args.det_thresh)
    gallery_keys = tuple(args.gallery_keys) if args.gallery_keys else DEFAULT_GALLERY_KEYS
    unresolved_total = 0

    for gkey in gallery_keys:
        stats = migrate_redis_gallery(
            r,
            gkey,
            args.photos_dir,
            embed_fn,
            model_name=args.model,
            captures_dir=args.captures_dir,
            include_captures=(not args.no_captures and gkey == "iris:face:gallery"),
            force=args.force,
            dry_run=args.dry_run,
        )
        _print_summary(f"redis:{gkey}", stats, args.dry_run)
        unresolved_total += stats.unresolved

    if args.attendance_db:
        db_path = Path(args.attendance_db)
        if db_path.is_file():
            db_stats = migrate_attendance_db(
                r,
                db_path,
                args.photos_dir,
                embed_fn,
                gallery_key="attendance:face:gallery",
                model_name=args.model,
                force=args.force,
                dry_run=args.dry_run,
            )
            _print_summary("sqlite:attendance.db", db_stats, args.dry_run)
            unresolved_total += db_stats.unresolved

    return 2 if (args.strict and unresolved_total > 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
