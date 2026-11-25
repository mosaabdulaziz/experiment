"""An exit completed by a long absence is stamped in the past (its last sighting), but it is still news
when the sidecar has just published it."""
import time

from app.tracking import TrackingService


class _Manager:
    async def broadcast(self, *_a, **_k):
        pass


class _Feed:
    def __init__(self):
        self.calls = []

    def note_exit(self, gid, cam, at, reopens=False):
        self.calls.append((gid, cam, at, reopens))


def _svc():
    svc = TrackingService({"mtmc": {"cameras": ["cam-01", "cam-02"]}, "truck_feed": {}}, _Manager())
    svc._truck_feed = _Feed()
    svc._start_time = time.time() - 120          # the service started two minutes ago
    return svc


def test_an_absence_exit_with_an_old_time_but_just_published_is_reported():
    svc = _svc()
    now = time.time()
    svc._apply_exited([{"global_id": 0, "cam_id": "cam-02", "at": now - 900, "reopens": True, "emitted": now}])
    assert svc._truck_feed.calls == [(0, "cam-02", now - 900, True)]
    assert svc._global_tracks.get(0) is None or not svc._global_tracks[0].exited      # the vehicle itself stays


def test_a_replayed_old_event_is_not_reported_again():
    svc = _svc()
    now = time.time()
    svc._apply_exited([{"global_id": 0, "cam_id": "cam-02", "at": now - 900, "reopens": True, "emitted": now - 600}])
    assert svc._truck_feed.calls == []


def test_a_real_exit_is_still_permanent():
    svc = _svc()
    now = time.time()
    svc._apply_exited([{"global_id": 3, "cam_id": "cam-01", "at": now - 5, "reopens": False, "emitted": now - 5}])
    assert svc._global_tracks[3].exited and svc._truck_feed.calls == [(3, "cam-01", now - 5, False)]


def test_direction_verdicts_are_kept_per_vehicle_and_shown_with_its_track():
    import fakeredis
    svc = _svc()
    svc._r = r = fakeredis.FakeRedis()
    r.xadd("iris:mtmc:direction", {"global_id": 4, "cam_id": "cam-02", "local_id": 9, "direction": "exiting",
                                   "orientation": "front", "confidence": 0.93, "at": 1000.5})
    r.xadd("iris:mtmc:direction", {"global_id": 4, "cam_id": "cam-02", "local_id": 12, "direction": "entering",
                                   "orientation": "rear", "confidence": "None", "at": 2000.0})
    svc._poll_direction()
    got = svc._directions[4]
    assert [(g["direction"], g["cam"]) for g in got] == [("exiting", "cam-02"), ("entering", "cam-02")]
    assert got[0]["confidence"] == 0.93 and got[1]["confidence"] is None
    svc._poll_direction()                                    # nothing new: nothing repeated
    assert len(svc._directions[4]) == 2


def test_the_gate_lines_can_be_switched_off():
    on = TrackingService({"mtmc": {"cameras": ["cam-01"]}, "truck_feed": {}}, _Manager())
    off = TrackingService({"mtmc": {"cameras": ["cam-01"]}, "truck_feed": {}, "line_count": {"enabled": False}}, _Manager())
    assert on._lines_enabled is True and off._lines_enabled is False
