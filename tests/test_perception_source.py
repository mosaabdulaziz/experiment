import asyncio
import json
import fakeredis

from app.perception_source import PerceptionSource


def _seed(r, source_id, n):
    """Push n entries (frame 0..n-1) with one car detection each, jpeg = b'JPG<i>'."""
    key = f"iris:meta:{source_id}"
    for i in range(n):
        r.xadd(key, {
            "frame_num": i,
            "pts": f"{i * 0.5:.3f}",
            "dets": json.dumps([["car", 0.9, i]]),  # track id == frame for assertion
            "jpeg": f"JPG{i}".encode(),
        })


def _make_source(r, batch_size=4):
    cfg = {
        "sampling": {"frames_per_batch": batch_size},
        "perception": {"backend": "deepstream", "source_id": "cam0",
                       "redis_url": "redis://unused"},
    }
    src = PerceptionSource(cfg, redis_client=r)
    return src


def test_grab_annotated_batch_returns_last_n_oldest_first():
    r = fakeredis.FakeStrictRedis()
    _seed(r, "cam0", 10)
    src = _make_source(r, batch_size=4)

    frames, dets_list, offsets = src.grab_annotated_batch()

    assert len(frames) == 4
    # last 4 entries (frames 6,7,8,9) in oldest->newest order
    assert frames == [b"JPG6", b"JPG7", b"JPG8", b"JPG9"]
    assert offsets == [3.0, 3.5, 4.0, 4.5]
    # tuples are (class, conf, track_id, bbox); track_id from seed is frame number, bbox None (not in seed data)
    assert dets_list[0] == [("car", 0.9, 6, None)]


def test_grab_annotated_batch_empty_returns_none():
    r = fakeredis.FakeStrictRedis()
    src = _make_source(r)
    assert src.grab_annotated_batch() is None


def test_grab_frame_batch_drops_detections():
    r = fakeredis.FakeStrictRedis()
    _seed(r, "cam0", 3)
    src = _make_source(r, batch_size=4)
    frames, offsets = src.grab_frame_batch()
    assert frames == [b"JPG0", b"JPG1", b"JPG2"]   # fewer than batch is fine
    assert offsets == [0.0, 0.5, 1.0]


def test_grab_display_frame_reads_latest_jpeg_key():
    r = fakeredis.FakeStrictRedis()
    r.set("iris:jpeg:cam0", b"LATEST")
    src = _make_source(r)
    assert src.grab_display_frame() == b"LATEST"


def test_mark_watched_sets_ttl_key():
    r = fakeredis.FakeStrictRedis()
    src = _make_source(r)
    src.mark_watched(ttl=10)
    assert r.get("iris:watched:cam0") == b"1"
    assert r.ttl("iris:watched:cam0") == 10


def test_mark_watched_expires_and_never_raises():
    r = fakeredis.FakeStrictRedis()
    src = _make_source(r)
    src.mark_watched(ttl=1)
    assert r.exists("iris:watched:cam0")
    r.delete("iris:watched:cam0")
    src.mark_watched(ttl=1)  # refreshes
    assert r.exists("iris:watched:cam0")


def test_set_source_publishes_switch_source():
    r = fakeredis.FakeStrictRedis()
    src = _make_source(r)
    src.set_source("file:///vids/a.mp4")
    items = r.lrange("iris:ctrl:cam0", 0, -1)
    assert [json.loads(x) for x in items] == [{"cmd": "switch_source", "uri": "file:///vids/a.mp4"}]


def test_start_and_stop_publish_commands():
    r = fakeredis.FakeStrictRedis()
    src = _make_source(r)
    asyncio.run(src.start())
    asyncio.run(src.stop())
    items = [json.loads(x) for x in r.lrange("iris:ctrl:cam0", 0, -1)]
    assert items == [{"cmd": "start"}, {"cmd": "stop"}]


def test_status_properties_read_status_key():
    r = fakeredis.FakeStrictRedis()
    r.set("iris:status:cam0", json.dumps({"running": True, "is_file": True, "source": "a.mp4"}))
    src = _make_source(r)
    assert src.is_running is True
    assert src.is_file is True
    assert src.source_label == "a.mp4"


def test_status_missing_reports_unavailable():
    r = fakeredis.FakeStrictRedis()
    src = _make_source(r)
    assert src.is_running is False
    assert src.is_file is False
    assert src.source_label == "sidecar unavailable"


def test_upload_uri_uses_prefix():
    r = fakeredis.FakeStrictRedis()
    src = _make_source(r)
    assert src.upload_uri("clip.mp4") == \
        "file:///workspace/iris/data/uploads/clip.mp4"


def test_explicit_source_id_overrides_config():
    r = fakeredis.FakeStrictRedis()
    cfg = {"sampling": {"frames_per_batch": 4},
           "perception": {"source_id": "cam0", "redis_url": "redis://unused"}}
    src = PerceptionSource(cfg, redis_client=r, source_id="cam7")
    r.set("iris:jpeg:cam7", b"X")
    assert src.grab_display_frame() == b"X"          # uses cam7 keys, not cam0
    src.set_source("file://v")
    assert r.lrange("iris:ctrl:cam7", 0, -1)          # control key is cam7


def test_status_read_logs_at_debug(caplog):
    import logging
    cfg = {"perception": {"redis_url": "redis://x", "source_id": "cam0"}}
    src = PerceptionSource(cfg, redis_client=fakeredis.FakeRedis(), source_id="cam0")
    with caplog.at_level(logging.DEBUG, logger="iris.perception"):
        _ = src.is_running   # triggers a status read
    assert any(r.name == "iris.perception" for r in caplog.records)


from app.perception_source import parse_dets


def test_parse_dets_reads_label_conf_track_id_and_bbox():
    raw = [["person", 0.91, 17, [10, 20, 30, 40]]]
    assert parse_dets(raw) == [("person", 0.91, 17, (10.0, 20.0, 30.0, 40.0))]


def test_parse_dets_reads_flat_7_element_format():
    raw = [["person", 0.91, 17, 10.0, 20.0, 30.0, 40.0]]
    assert parse_dets(raw) == [("person", 0.91, 17, (10.0, 20.0, 30.0, 40.0))]


def test_parse_dets_tolerates_sidecar_without_bbox():
    """An older sidecar publishes [label, conf, object_id] and nothing more."""
    assert parse_dets([["truck", 0.5, 3]]) == [("truck", 0.5, 3, None)]


def test_parse_dets_tolerates_sidecar_without_track_id():
    """Older still: [label, conf]."""
    assert parse_dets([["car", 0.6]]) == [("car", 0.6, None, None)]


def test_parse_dets_treats_empty_bbox_as_missing():
    assert parse_dets([["car", 0.6, 2, []]]) == [("car", 0.6, 2, None)]


def test_parse_dets_handles_empty_input():
    assert parse_dets([]) == []


def test_parse_dets_normalises_untracked_sentinel_to_none():
    """DeepStream's UNTRACKED_OBJECT_ID (0xFFFFFFFFFFFFFFFF) means "no track id
    assigned" and must not be passed through as a real track_id."""
    raw = [["person", 0.91, 0xFFFFFFFFFFFFFFFF, [10, 20, 30, 40]]]
    assert parse_dets(raw) == [("person", 0.91, None, (10.0, 20.0, 30.0, 40.0))]


def test_parse_dets_preserves_track_id_zero():
    """track_id 0 is a real (falsy but valid) id and must not be coerced to None."""
    assert parse_dets([["car", 0.6, 0]]) == [("car", 0.6, 0, None)]


def test_parse_dets_treats_wrong_length_bbox_as_missing():
    """A bbox that isn't exactly [l, t, w, h] is malformed input, not a partial box."""
    assert parse_dets([["bus", 0.4, 7, [1, 2, 3]]]) == [("bus", 0.4, 7, None)]


class _ExplodingJpegFields(dict):
    """iris:meta fields whose jpeg is a tripwire: reading it fails the test.

    grab_detections runs on every sampler tick, so pulling a whole frame it never
    uses would be pure Redis bandwidth.
    """

    def get(self, key, default=None):
        if key == b"jpeg":
            raise AssertionError("grab_detections must not read the JPEG field")
        return super().get(key, default)


class _FakeRedis:
    """Minimal xrevrange stub — enough for grab_detections, nothing else."""

    def __init__(self, entries=None, error=None):
        self._entries = entries or []
        self._error = error

    def xrevrange(self, key, count=None):
        if self._error is not None:
            raise self._error
        return self._entries[:count] if count else self._entries


def test_grab_detections_returns_newest_entry_id_and_parsed_dets():
    r = fakeredis.FakeStrictRedis()
    key = "iris:meta:cam0"
    for i in range(3):
        r.xadd(key, {"pts": f"{i:.3f}",
                     "dets": json.dumps([["person", 0.9, 17 + i, [10, 20, 30, 40]]]),
                     "jpeg": b"JPG"})
    src = _make_source(r)

    entry_id, dets = src.grab_detections()

    newest = r.xrevrange(key, count=1)[0][0].decode()
    assert entry_id == newest                       # the id the caller dedupes on
    assert dets == [("person", 0.9, 19, (10.0, 20.0, 30.0, 40.0))]   # newest entry, parse_dets shape


def test_grab_detections_returns_none_on_empty_stream():
    src = _make_source(fakeredis.FakeStrictRedis())
    assert src.grab_detections() is None


def test_grab_detections_returns_none_on_redis_error():
    import redis as _redis
    src = _make_source(_FakeRedis(error=_redis.RedisError("down")))
    assert src.grab_detections() is None


def test_grab_detections_returns_none_on_malformed_dets():
    src = _make_source(_FakeRedis(entries=[(b"5-0", {b"dets": b"{not json"})]))
    assert src.grab_detections() is None


def test_grab_detections_never_reads_the_jpeg():
    fields = _ExplodingJpegFields({b"pts": b"1.5",
                                   b"dets": json.dumps([["car", 0.6, 3]]).encode(),
                                   b"jpeg": b"HUGE"})
    src = _make_source(_FakeRedis(entries=[(b"7-0", fields)]))
    assert src.grab_detections() == ("7-0", [("car", 0.6, 3, None)])
