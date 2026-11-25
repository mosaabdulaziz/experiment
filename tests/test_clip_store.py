from app.clip_store import ClipStore


def _mp4(tmp_path, name: str, content: bytes) -> str:
    path = str(tmp_path / name)
    with open(path, "wb") as fh:
        fh.write(content)
    return path


def test_add_and_get_roundtrip(tmp_path):
    cs = ClipStore(ttl_seconds=1000, root_dir=str(tmp_path / "clips"))
    cid = cs.add({"cam0": _mp4(tmp_path, "a.mp4", b"mp4A"),
                  "cam1": _mp4(tmp_path, "b.mp4", b"mp4B")}, {"range": [10, 20]})

    meta = cs.get(cid)

    assert meta["cameras"] == ["cam0", "cam1"]
    assert meta["meta"]["range"] == [10, 20]
    assert open(cs.get_clip_path(cid, "cam0"), "rb").read() == b"mp4A"
    assert open(cs.get_clip_path(cid, "cam1"), "rb").read() == b"mp4B"


def test_missing_id_returns_none(tmp_path):
    cs = ClipStore(root_dir=str(tmp_path / "clips"))
    assert cs.get("nope") is None
    assert cs.get_clip_path("nope", "cam0") is None


def test_unknown_camera_returns_none(tmp_path):
    cs = ClipStore(root_dir=str(tmp_path / "clips"))
    cid = cs.add({"cam0": _mp4(tmp_path, "a.mp4", b"x")}, {})
    assert cs.get_clip_path(cid, "camX") is None


def test_ttl_prune_lazy_on_access(tmp_path):
    t = {"v": 0.0}
    clk = lambda: t["v"]
    cs = ClipStore(ttl_seconds=100, root_dir=str(tmp_path / "clips"))
    cid = cs.add({"cam0": _mp4(tmp_path, "a.mp4", b"x")}, {}, clock=clk)
    t["v"] = 50
    assert cs.get(cid, clock=clk) is not None       # within ttl
    t["v"] = 201
    assert cs.get(cid, clock=clk) is None            # expired -> lazy pruned
    assert len(cs) == 0


def test_prune_expired_removes_old_keeps_fresh(tmp_path):
    t = {"v": 0.0}
    clk = lambda: t["v"]
    cs = ClipStore(ttl_seconds=100, root_dir=str(tmp_path / "clips"))
    old = cs.add({"cam0": _mp4(tmp_path, "a.mp4", b"a")}, {}, clock=clk)
    t["v"] = 80
    fresh = cs.add({"cam0": _mp4(tmp_path, "b.mp4", b"b")}, {}, clock=clk)
    t["v"] = 150  # old age 150 > 100 (expired); fresh age 70 (alive)

    assert cs.prune_expired(clock=clk) == 1

    assert cs.get(old, clock=clk) is None
    assert cs.get(fresh, clock=clk) is not None


def test_corrupt_metadata_returns_none(tmp_path):
    cs = ClipStore(ttl_seconds=1000, root_dir=str(tmp_path / "clips"))
    cid = cs.add({"cam0": _mp4(tmp_path, "a.mp4", b"mp4A")}, {"range": [10, 20]})

    # Corrupt the meta.json by overwriting with invalid JSON
    meta_path = str(tmp_path / "clips" / cid / "meta.json")
    with open(meta_path, "w") as fh:
        fh.write("not valid json")

    # get() should return None, not raise
    assert cs.get(cid) is None
    # get_clip_path() should also return None, not raise
    assert cs.get_clip_path(cid, "cam0") is None
    # The corrupt clip directory should have been cleaned up
    assert len(cs) == 0


def test_clear_removes_every_clip_and_keeps_the_store_usable(tmp_path):
    from app.clip_store import ClipStore

    store = ClipStore(ttl_seconds=600.0, root_dir=str(tmp_path / "clips"))
    src = tmp_path / "a.mp4"
    src.write_bytes(b"x")
    clip_id = store.add({"cam0": str(src)})
    assert store.get(clip_id) is not None

    removed = store.clear()

    assert removed == 1
    assert store.get(clip_id) is None
    assert len(store) == 0

    # Still writable afterwards -- the root directory must survive.
    src2 = tmp_path / "b.mp4"
    src2.write_bytes(b"y")
    assert store.get(store.add({"cam0": str(src2)})) is not None
