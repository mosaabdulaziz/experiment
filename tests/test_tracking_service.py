"""Unit tests for the multi-camera tracking service (app/tracking.py)."""

import fakeredis

from app.tracking import GlobalTrack, TrackingService


def _service(cameras=("cam-01", "cam-02"), topology=None):
    config = {
        "mtmc": {
            "cameras": list(cameras),
            "camera_names": {c: c for c in cameras},
        },
        "perception": {"redis_url": "redis://localhost:6379/0"},
    }
    if topology is not None:
        config["mtmc"]["topology"] = topology
    svc = TrackingService(config, manager=None)
    svc._r = fakeredis.FakeStrictRedis()
    return svc


def test_poll_redis_keeps_entries_with_identical_ids_across_cameras():
    """Regression: Redis stream IDs collide across cameras (same <ms>-<seq> when
    the sidecar's per-stream publisher threads XADD in the same millisecond), and
    the dedup set must key on (cam, msg_id) so cam-02's entries are not silently
    dropped as duplicates of cam-01's."""
    svc = _service()
    # Force the exact same stream ID on both cameras — what production produces
    # when both publishers land in the same millisecond.
    svc._r.xadd("iris:meta:cam-01", {"dets": "[]", "global_ids": '{"1": 3}', "pts": "10.0"},
                id="1000-0")
    svc._r.xadd("iris:meta:cam-02", {"dets": "[]", "global_ids": '{"2": 3}', "pts": "12.0"},
                id="1000-0")

    results, skipped_by_cam = svc._poll_redis()

    cams = sorted(cam for cam, _mid, _d in results)
    assert cams == ["cam-01", "cam-02"], \
        f"cam-02 was dropped by cross-stream dedup: {cams}"
    assert skipped_by_cam == {}

    # A second poll must not re-read the same entries.
    results2, skipped2 = svc._poll_redis()
    assert results2 == []
    assert skipped2 == {}


def test_invalid_topology_jump_logs_warning(caplog):
    svc = _service(topology={"cam-01": ["cam-02"], "cam-02": ["cam-03"], "cam-03": []})
    gt = GlobalTrack(global_id=3, cameras_seen=["cam-01"],
                     entry_times={"cam-01": 10.0}, exit_times={"cam-01": 11.0},
                     observation_counts={"cam-01": 5}, first_seen_pts=10.0, last_seen_pts=11.0)

    with caplog.at_level("WARNING", logger="iris.tracking"):
        prev = svc._record_camera_appearance(gt, "cam-03", 15.0)

    assert prev == "cam-01"
    assert gt.cameras_seen == ["cam-01", "cam-03"]
    assert any("without a valid topology transition" in r.getMessage()
               for r in caplog.records)


def test_valid_topology_jump_logs_no_warning(caplog):
    svc = _service(topology={"cam-01": ["cam-02"], "cam-02": ["cam-03"], "cam-03": []})
    gt = GlobalTrack(global_id=3, cameras_seen=["cam-01"],
                     entry_times={"cam-01": 10.0}, exit_times={"cam-01": 11.0},
                     observation_counts={"cam-01": 5}, first_seen_pts=10.0, last_seen_pts=11.0)

    with caplog.at_level("WARNING", logger="iris.tracking"):
        prev = svc._record_camera_appearance(gt, "cam-02", 15.0)

    assert prev == "cam-01"
    assert not any("without a valid topology transition" in r.getMessage()
                   for r in caplog.records)

def test_assignment_event_adds_camera_and_transition_to_existing_track():
    """A vehicle whose global id was decided AFTER it left the frame (slow VLM
    verdict) is never in a per-frame entry; the sidecar announces it on
    iris:mtmc:assigned and the track must gain that camera + a transition."""
    svc = _service(topology={"cam-01": ["cam-02"]})
    svc._global_tracks[19] = GlobalTrack(
        global_id=19, cameras_seen=["cam-01"], entry_times={"cam-01": 43.0},
        exit_times={"cam-01": 45.2}, observation_counts={"cam-01": 5},
        label="vehicle", first_seen_pts=43.0, last_seen_pts=45.2)
    svc._r.xadd("iris:mtmc:assigned", {"cam_id": "cam-02", "local_id": "41", "global_id": "19",
                                   "pts_first": "49.0", "pts_last": "49.66"})

    svc._apply_assigned(svc._poll_assigned())

    gt = svc._global_tracks[19]
    assert gt.cameras_seen == ["cam-01", "cam-02"]
    assert gt.exit_times["cam-02"] == 49.66
    assert [t.to_cam for t in svc._all_transitions] == ["cam-02"]
    assert svc._poll_assigned() == []  # consumed once


def test_assignment_event_creates_track_when_no_per_frame_entry_was_published():
    svc = _service()
    svc._apply_assigned([{"cam_id": "cam-02", "global_id": 30, "pts_first": 50.0, "pts_last": 50.5}])
    assert svc._global_tracks[30].cameras_seen == ["cam-02"]


def test_origin_camera_sighting_missed_by_per_frame_entries_is_registered_first():
    """Regression (Vehicle #2 started at Cam-16): a very brief vehicle got its id on the
    origin camera but no per-frame entry ever carried it, so the track's first camera was
    the NEXT one. The assignment events restore the origin camera first, in order."""
    svc = _service(topology={"cam-01": ["cam-02"]})
    svc._r.xadd("iris:mtmc:assigned", {"cam_id": "cam-01", "local_id": "69", "global_id": "2",
                                       "pts_first": "137.5", "pts_last": "138.25"})
    svc._r.xadd("iris:mtmc:assigned", {"cam_id": "cam-02", "local_id": "191", "global_id": "2",
                                       "pts_first": "334.9", "pts_last": "335.4"})

    svc._apply_assigned(svc._poll_assigned())

    gt = svc._global_tracks[2]
    assert gt.cameras_seen == ["cam-01", "cam-02"]
    assert [(t.from_cam, t.to_cam) for t in svc._all_transitions] == [("cam-01", "cam-02")]
