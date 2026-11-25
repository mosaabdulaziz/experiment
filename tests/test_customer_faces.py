"""perception/modules/customer_faces/collector.py: gates, quality, crop files, the collector."""
import json
import os
import stat
import sys

import numpy as np

from perception.lib.imaging import decode_embedding
from perception.modules.customer_faces import collector as cf

FRONTAL = [[40, 50], [72, 50], [56, 66], [44, 82], [68, 82]]
TURNED = [[40, 50], [72, 50], [70, 66], [44, 82], [68, 82]]   # the nose near the right eye


def checker(n=112):
    y, x = np.mgrid[0:n, 0:n]
    return (((x // 4 + y // 4) % 2) * 255).astype(np.uint8)[:, :, None].repeat(3, axis=2)


class FakeRedis:
    def __init__(self):
        self.added = []

    def xadd(self, key, fields, maxlen=None, approximate=True):
        self.added.append((key, fields))


def collector(r, tmp_path, aligned=None):
    img = checker() if aligned is None else aligned
    return cf.CustomerFaceCollector(r, str(tmp_path), align=lambda frame, lmk: img,
                                    embed=lambda a: np.arange(1, 513, dtype=np.float32))


def face(tid=7, size=40, lmk=FRONTAL):
    return {"left": 100.0, "top": 100.0, "width": float(size), "height": float(size), "parent_id": tid,
            "landmarks": [[100 + x * size / 112, 100 + y * size / 112] for x, y in lmk]}


FRAME = np.full((720, 1280, 3), 120, np.uint8)


def test_yaw_is_zero_when_frontal_and_large_when_turned():
    assert cf.face_yaw(FRONTAL) == 0.0
    assert cf.face_yaw(TURNED) > 0.35


def test_sharpness_separates_a_flat_patch_from_texture():
    assert cf.sharpness(np.full((112, 112, 3), 90, np.uint8)) == 0.0
    assert cf.sharpness(checker()) > 60


def test_quality_grows_with_size_sharpness_and_frontalness():
    q = lambda s, sh, y: cf.face_quality(s, sh, y, 60.0, 0.35)
    assert q(112, 200, 0.0) == 1.0
    assert q(40, 200, 0.0) < q(80, 200, 0.0)
    assert q(80, 70, 0.0) < q(80, 200, 0.0)
    assert q(80, 200, 0.3) < q(80, 200, 0.1)


def test_save_jpeg_writes_host_deletable_files(tmp_path):
    rel = cf.save_jpeg(str(tmp_path), "faces", checker(), now=1790300000.0)
    path = tmp_path / rel
    assert rel.startswith("faces/") and rel.endswith(".jpg") and path.is_file()
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o666
    assert stat.S_IMODE(os.stat(path.parent).st_mode) == 0o777
    assert stat.S_IMODE(os.stat(tmp_path / "faces").st_mode) == 0o777


def test_collector_publishes_an_accepted_face(tmp_path):
    r = FakeRedis()
    item = collector(r, tmp_path).offer("barber-c", FRAME, face(), now=100.0)
    key, fields = r.added[0]
    assert key == "iris:customer_face:barber-c" and json.loads(fields["item"]) == item
    assert item["track_id"] == 7 and (tmp_path / item["crop_path"]).is_file()
    assert decode_embedding(item["embedding"]).shape == (512,) and 0 < item["quality"] <= 1


def test_collector_rejects_small_turned_blurry_and_landmarkless_faces(tmp_path):
    r = FakeRedis()
    c = collector(r, tmp_path)
    assert c.offer("barber-c", FRAME, face(size=30), now=1.0) is None
    assert c.offer("barber-c", FRAME, face(tid=8, lmk=TURNED), now=1.0) is None
    no_lmk = dict(face(tid=9)); no_lmk.pop("landmarks")
    assert c.offer("barber-c", FRAME, no_lmk, now=1.0) is None
    blurry = collector(r, tmp_path, aligned=np.full((112, 112, 3), 90, np.uint8))
    assert blurry.offer("barber-c", FRAME, face(tid=10), now=1.0) is None
    assert r.added == []
    assert c.rejected["size"] == 1 and c.rejected["yaw"] == 1 and c.rejected["landmarks"] == 1
    assert blurry.rejected["sharpness"] == 1


def test_collector_samples_each_track_at_most_once_per_interval(tmp_path):
    c = collector(FakeRedis(), tmp_path)
    assert c.offer("barber-c", FRAME, face(), now=10.0) is not None
    assert c.offer("barber-c", FRAME, face(), now=10.5) is None
    assert c.offer("barber-c", FRAME, face(tid=8), now=10.5) is not None
    assert c.offer("barber-c", FRAME, face(), now=11.0) is not None
