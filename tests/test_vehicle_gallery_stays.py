"""One best crop per stay: a vehicle back on a camera after a long absence gets its own picture."""
import json
import pathlib
import sys
import time

import fakeredis
import numpy as np

from perception.modules.mtmc.vehicle_gallery import VehicleGallery, _REDIS_KEY


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def _img(v):
    return np.full((20, 30, 3), v, np.uint8)


def _gallery(tmp_path, r, clock):
    return VehicleGallery(enabled=True, storage_backend="local", local_dir=str(tmp_path), redis_client=r,
                          stay_gap_s=90.0, clock=clock)


def _settle(g):
    end = time.time() + 3
    while g._queue.qsize() and time.time() < end:      # the worker thread persists asynchronously
        time.sleep(0.02)
    time.sleep(0.2)


def _submit(g, cam, gid, v, score):
    g.maybe_submit(cam, gid, _img(v), score, score, (0, 0, 30, 20), 0.0)


def test_a_return_after_a_long_absence_gets_its_own_file_field_and_stay_number(tmp_path):
    r, clock = fakeredis.FakeRedis(), Clock()
    g = _gallery(tmp_path, r, clock)
    _submit(g, "cam-01", 0, 10, 0.80)                 # first stay
    clock.t += 5
    _submit(g, "cam-01", 0, 20, 0.90)                 # same stay, a better shot replaces it
    clock.t += 900                                    # 15 minutes later
    _submit(g, "cam-01", 0, 30, 0.60)                 # a new stay: a worse score is fine, it is a new picture
    _settle(g)
    g.close()
    assert (tmp_path / "cam-01" / "0.jpg").exists() and (tmp_path / "cam-01" / "0-1.jpg").exists()
    first = json.loads(r.hget(_REDIS_KEY, "cam-01:0"))
    second = json.loads(r.hget(_REDIS_KEY, "cam-01:0:1"))
    assert (first["stay"], second["stay"]) == (0, 1) and first["score"] == 0.9 and second["score"] == 0.6
    assert second["updated_at"] - first["updated_at"] > 800


def test_within_a_stay_only_a_strictly_better_shot_replaces_the_crop(tmp_path):
    r, clock = fakeredis.FakeRedis(), Clock()
    g = _gallery(tmp_path, r, clock)
    for score in (0.7, 0.7, 0.6, 0.9, 0.85):
        clock.t += 10                                  # always inside the 90 s gap
        _submit(g, "cam-02", 5, int(score * 100), score)
    _settle(g)
    g.close()
    assert json.loads(r.hget(_REDIS_KEY, "cam-02:5"))["score"] == 0.9
    assert r.hget(_REDIS_KEY, "cam-02:5:1") is None


def test_each_camera_and_vehicle_counts_its_stays_separately(tmp_path):
    r, clock = fakeredis.FakeRedis(), Clock()
    g = _gallery(tmp_path, r, clock)
    _submit(g, "cam-01", 0, 1, 0.5)
    _submit(g, "cam-02", 0, 2, 0.5)
    clock.t += 500
    _submit(g, "cam-01", 0, 3, 0.5)                    # cam-01 returns; cam-02 never did
    _settle(g)
    g.close()
    assert r.hget(_REDIS_KEY, "cam-01:0:1") is not None and r.hget(_REDIS_KEY, "cam-02:0:1") is None


def test_after_a_restart_numbering_and_best_scores_continue_from_the_index(tmp_path):
    r, clock = fakeredis.FakeRedis(), Clock()
    g = _gallery(tmp_path, r, clock)
    _submit(g, "cam-01", 0, 10, 0.9)
    clock.t += 500
    _submit(g, "cam-01", 0, 20, 0.8)                   # stay 1
    _settle(g)
    g.close()
    clock.t += 10                                      # restart shortly after: the vehicle is still in stay 1
    g2 = _gallery(tmp_path, r, clock)
    _submit(g2, "cam-01", 0, 99, 0.5)                  # worse than stay 1's best: must not overwrite it
    clock.t += 500
    _submit(g2, "cam-01", 0, 77, 0.4)                  # back later again: stay 2, not an overwrite of stay 0/1
    _settle(g2)
    g2.close()
    assert json.loads(r.hget(_REDIS_KEY, "cam-01:0:1"))["score"] == 0.8
    assert json.loads(r.hget(_REDIS_KEY, "cam-01:0:2"))["stay"] == 2
    assert json.loads(r.hget(_REDIS_KEY, "cam-01:0"))["score"] == 0.9
