"""Unit tests for scripts/migrate_face_gallery.py."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts import migrate_face_gallery as mfg


class _FakeRedis:
    def __init__(self) -> None:
        self.hashes: dict[str, dict[bytes, bytes]] = {}
        self.kv: dict[bytes, bytes] = {}

    def _key(self, key: str | bytes) -> str:
        return key.decode() if isinstance(key, bytes) else key

    def _b(self, val: str | bytes) -> bytes:
        return val if isinstance(val, bytes) else val.encode("utf-8")

    def hgetall(self, key: str) -> dict[bytes, bytes]:
        return dict(self.hashes.get(self._key(key), {}))

    def hget(self, key: str, field: str) -> bytes | None:
        return self.hashes.get(self._key(key), {}).get(self._b(field))

    def hset(self, key: str, field: str, value: str | bytes) -> int:
        bucket = self.hashes.setdefault(self._key(key), {})
        b_field = self._b(field)
        existed = b_field in bucket
        bucket[b_field] = self._b(value)
        return 0 if existed else 1

    def get(self, key: str) -> bytes | None:
        return self.kv.get(self._b(key))

    def set(self, key: str, value: str | bytes) -> bool:
        self.kv[self._b(key)] = self._b(value)
        return True


def _write_img(path: Path, seed: int = 1) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((64, 64, 3), seed % 255, dtype=np.uint8)
    ok = cv2.imwrite(str(path), img)
    assert ok
    return path


def _det_embed(img: np.ndarray) -> np.ndarray:
    val = float(img[0, 0, 0]) + 1.0
    vec = np.zeros(512, dtype=np.float32)
    vec[0] = val
    vec[1] = 2.0
    return vec


def test_resolve_photo_paths_handles_container_prefix_and_multi(tmp_path):
    photos = tmp_path / "face_photos"
    p_main = _write_img(photos / "person-1.jpg", seed=10)
    p_a = _write_img(photos / "multi" / "person-1" / "angle1.jpg", seed=20)
    p_b = _write_img(photos / "multi" / "person-1" / "angle2.png", seed=30)

    resolved = mfg.resolve_photo_paths(
        "person-1",
        {"photo_path": "/app/data/face_photos/person-1.jpg"},
        photos,
    )
    assert [p.resolve() for p in resolved] == [p_main.resolve(), p_a.resolve(), p_b.resolve()]


def test_migrate_redis_gallery_reembeds_legacy_and_captures_idempotently(tmp_path):
    r = _FakeRedis()
    photos = tmp_path / "face_photos"
    captures_dir = tmp_path / "face_captures"
    _write_img(photos / "emp-1.jpg", seed=10)
    _write_img(photos / "multi" / "emp-1" / "side.jpg", seed=25)
    cap_img = _write_img(captures_dir / "cap-1.jpg", seed=40)

    old_vec = [0.5] * 512
    r.hset(
        "iris:face:gallery",
        "emp-1",
        json.dumps(
            {
                "name": "Alice",
                "embedding": old_vec,
                "photo_path": "/app/data/face_photos/emp-1.jpg",
                "created_at": 1234.5,
            }
        ),
    )
    r.hset(
        "iris:face:captures",
        "cap-1",
        json.dumps({"assigned_to": "Alice", "embedding": old_vec, "pts": 10.0}),
    )
    assert cap_img.is_file()

    # Dry-run leaves Redis untouched
    dry = mfg.migrate_redis_gallery(
        r, "iris:face:gallery", photos, _det_embed, captures_dir=captures_dir, dry_run=True
    )
    assert dry.would_migrate == 1 and dry.migrated == 0
    assert json.loads(r.hget("iris:face:gallery", "emp-1"))["embedding"] == old_vec

    # Real migration updates primary + multi + assigned capture and tags model
    stats = mfg.migrate_redis_gallery(r, "iris:face:gallery", photos, _det_embed, captures_dir=captures_dir)
    assert stats.total == 1
    assert stats.migrated == 1
    assert stats.captures_updated == 1
    assert stats.unresolved == 0

    updated = json.loads(r.hget("iris:face:gallery", "emp-1"))
    assert "embedding" not in updated
    assert updated["name"] == "Alice"
    assert updated["created_at"] == 1234.5
    assert updated["model"] == "auraface"
    assert len(updated["embeddings"]) == 3
    for vec in updated["embeddings"]:
        assert pytest.approx(float(np.linalg.norm(vec)), rel=1e-5) == 1.0
        assert vec != old_vec

    updated_cap = json.loads(r.hget("iris:face:captures", "cap-1"))
    assert updated_cap["model"] == "auraface"
    assert updated_cap["embedding"] == updated["embeddings"][2]

    # Second run is idempotent without --force
    second = mfg.migrate_redis_gallery(r, "iris:face:gallery", photos, _det_embed, captures_dir=captures_dir)
    assert second.already_migrated == 1 and second.migrated == 0

    # With --force it re-embeds again
    forced = mfg.migrate_redis_gallery(
        r, "iris:face:gallery", photos, _det_embed, captures_dir=captures_dir, force=True
    )
    assert forced.migrated == 1


def test_migrate_leaves_entries_intact_when_images_missing_or_unreadable(tmp_path):
    r = _FakeRedis()
    photos = tmp_path / "face_photos"
    _write_img(photos / "no-face.jpg", seed=5)

    r.hset(
        "iris:face:gallery",
        "missing-id",
        json.dumps({"name": "Ghost", "embeddings": [[0.1] * 512], "photo_path": "/missing/ghost.jpg"}),
    )
    r.hset(
        "iris:face:gallery",
        "no-face",
        json.dumps({"name": "Blank", "embeddings": [[0.2] * 512], "photo_path": "/app/data/face_photos/no-face.jpg"}),
    )

    stats = mfg.migrate_redis_gallery(
        r,
        "iris:face:gallery",
        photos,
        lambda _img: None,
        include_captures=False,
    )
    assert stats.migrated == 0
    assert stats.skipped_no_images == ["missing-id"]
    assert stats.failed_no_face == ["no-face"]
    # Existing entries were preserved intact
    assert json.loads(r.hget("iris:face:gallery", "missing-id"))["embeddings"] == [[0.1] * 512]
    assert json.loads(r.hget("iris:face:gallery", "no-face"))["embeddings"] == [[0.2] * 512]


def test_migrate_attendance_db_embeds_active_employees(tmp_path):
    r = _FakeRedis()
    photos = tmp_path / "face_photos"
    _write_img(photos / "person-1.jpg", seed=12)
    _write_img(photos / "multi" / "person-1" / "m1.jpg", seed=22)

    db_path = tmp_path / "attendance.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE employee (id TEXT PRIMARY KEY, name TEXT, gallery_rec_id TEXT, active INTEGER)")
    conn.execute("INSERT INTO employee VALUES ('emp-1', 'Youssef', 'person-1', 1)")
    conn.execute("INSERT INTO employee VALUES ('emp-2', 'Inactive', 'person-2', 0)")
    conn.execute("INSERT INTO employee VALUES ('emp-3', 'NoPhoto', 'person-3', 1)")
    conn.commit()
    conn.close()

    stats = mfg.migrate_attendance_db(r, db_path, photos, _det_embed)
    assert stats.total == 2
    assert stats.migrated == 1
    assert stats.skipped_no_images == ["person-3"]

    stored = json.loads(r.hget("attendance:face:gallery", "person-1"))
    assert stored["name"] == "Youssef"
    assert stored["model"] == "auraface"
    assert len(stored["embeddings"]) == 2
    assert r.hget("attendance:face:gallery", "person-2") is None
