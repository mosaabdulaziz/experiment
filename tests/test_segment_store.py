import fakeredis
import pytest

from app.segment_store import SegmentStore, SegmentRef


class _FakeMinio:
    """Minimal in-memory stand-in for the subset of minio.Minio's API SegmentStore
    calls. No real MinIO server needed for these tests."""
    def __init__(self):
        self.objects = {}  # (bucket, key) -> bytes

    def fput_object(self, bucket, key, file_path):
        with open(file_path, "rb") as fh:
            self.objects[(bucket, key)] = fh.read()

    def fget_object(self, bucket, key, dest_path):
        with open(dest_path, "wb") as fh:
            fh.write(self.objects[(bucket, key)])

    def remove_object(self, bucket, key):
        self.objects.pop((bucket, key), None)


class _RaisingFakeRedis:
    """Wrapper around FakeRedis that raises on xadd call, for testing error paths."""
    def __init__(self):
        self._redis = fakeredis.FakeRedis()

    def xadd(self, *args, **kwargs):
        raise RuntimeError("xadd failed")

    def xrange(self, *args, **kwargs):
        return self._redis.xrange(*args, **kwargs)

    def keys(self, *args, **kwargs):
        return self._redis.keys(*args, **kwargs)

    def xdel(self, *args, **kwargs):
        return self._redis.xdel(*args, **kwargs)


def _store():
    return SegmentStore(fakeredis.FakeRedis(), minio_client=_FakeMinio(), bucket="iris-segments")


def test_put_then_list_range_returns_overlapping_segment(tmp_path):
    store = _store()
    seg_path = tmp_path / "seg.ts"
    seg_path.write_bytes(b"tsdata")
    store.put("cam0", start_pts=10.0, end_pts=18.0, local_path=str(seg_path), clock=lambda: 1000.0)

    refs = store.list_range("cam0", lo=12.0, hi=14.0)

    assert len(refs) == 1
    assert refs[0].start_pts == 10.0
    assert refs[0].end_pts == 18.0
    assert refs[0].recorded_at == 1000.0


def test_list_range_excludes_non_overlapping_segments(tmp_path):
    store = _store()
    p = tmp_path / "seg.ts"
    p.write_bytes(b"x")
    store.put("cam0", 0.0, 8.0, str(p))
    store.put("cam0", 8.0, 16.0, str(p))

    refs = store.list_range("cam0", lo=20.0, hi=24.0)

    assert refs == []


def test_list_range_sorted_by_start_pts(tmp_path):
    store = _store()
    p = tmp_path / "seg.ts"
    p.write_bytes(b"x")
    store.put("cam0", 8.0, 16.0, str(p))
    store.put("cam0", 0.0, 8.0, str(p))

    refs = store.list_range("cam0", lo=0.0, hi=16.0)

    assert [r.start_pts for r in refs] == [0.0, 8.0]


def test_fetch_downloads_to_dest_dir(tmp_path):
    store = _store()
    src = tmp_path / "seg.ts"
    src.write_bytes(b"payload")
    store.put("cam0", 0.0, 8.0, str(src))
    ref = store.list_range("cam0", 0.0, 8.0)[0]
    dest_dir = tmp_path / "fetched"
    dest_dir.mkdir()

    local_path = store.fetch(ref, str(dest_dir))

    assert open(local_path, "rb").read() == b"payload"


def test_expire_removes_old_segments_keeps_fresh(tmp_path):
    store = _store()
    p = tmp_path / "seg.ts"
    p.write_bytes(b"x")
    store.put("cam0", 0.0, 8.0, str(p), clock=lambda: 0.0)       # old
    store.put("cam0", 8.0, 16.0, str(p), clock=lambda: 1000.0)   # fresh

    removed = store.expire(retention_seconds=100.0, clock=lambda: 1010.0)

    assert removed == 1
    remaining = store.list_range("cam0", 0.0, 20.0)
    assert [r.start_pts for r in remaining] == [8.0]


def test_put_cleans_up_orphaned_object_on_xadd_failure(tmp_path):
    """If xadd fails after fput_object succeeds, the MinIO object should be removed."""
    minio = _FakeMinio()
    redis = _RaisingFakeRedis()
    store = SegmentStore(redis, minio_client=minio, bucket="iris-segments")

    seg_path = tmp_path / "seg.ts"
    seg_path.write_bytes(b"tsdata")

    # put() should raise because xadd raises
    with pytest.raises(RuntimeError, match="xadd failed"):
        store.put("cam0", start_pts=10.0, end_pts=18.0, local_path=str(seg_path))

    # Verify the MinIO object was cleaned up (not present)
    assert len(minio.objects) == 0


def test_list_range_wall_clock_time(tmp_path):
    store = _store()
    p = tmp_path / "seg.ts"
    p.write_bytes(b"tsdata")

    # Segment 1: recorded at wall time 1000.0, duration 6.0s (span 994.0 -> 1000.0)
    store.put("cam0", 0.0, 6.0, str(p), clock=lambda: 1000.0)
    # Segment 2: recorded at wall time 1006.0, duration 6.0s (span 1000.0 -> 1006.0)
    store.put("cam0", 6.0, 12.0, str(p), clock=lambda: 1006.0)
    # Segment 3: recorded at wall time 1012.0, duration 6.0s (span 1006.0 -> 1012.0)
    store.put("cam0", 12.0, 18.0, str(p), clock=lambda: 1012.0)

    # Query with wall time window 998.0 -> 1008.0 in wall mode
    refs = store.list_range("cam0", lo=998.0, hi=1008.0, time_mode="wall")
    assert len(refs) == 3
    assert refs[0].start_pts == pytest.approx(994.0)
    assert refs[0].end_pts == pytest.approx(1000.0)
    assert refs[1].start_pts == pytest.approx(1000.0)
    assert refs[1].end_pts == pytest.approx(1006.0)
    assert refs[2].start_pts == pytest.approx(1006.0)
    assert refs[2].end_pts == pytest.approx(1012.0)

    # Query non-overlapping window
    refs_none = store.list_range("cam0", lo=1050.0, hi=1060.0, time_mode="wall")
    assert len(refs_none) == 0

    # Auto mode with timestamps > 1e8 automatically uses wall-clock mode
    refs_auto = store.list_range("cam0", lo=1789224910.0, hi=1789224920.0, time_mode="auto")
    assert len(refs_auto) == 0  # No segments at 1.78e9

