import yaml
from fastapi.testclient import TestClient
from app.server import create_app


def _cfg(tmp_path):
    return {
        "perception": {"backend": "local"},
        "api": {"backend": "cloud", "api_key": "x", "model": "m"},
        # vlm_enabled defaults to False in create_app, and /api/chat 503s when it is
        # off -- chat tests must opt in explicitly.
        "monitoring": {"vlm_enabled": True},
        "sampling": {}, "events": {"db_path": str(tmp_path / "e.db")},
        "chat": {"tools_enabled": False, "preview": {"dir": str(tmp_path / "clips")}},
        "cv": {"enabled": False}, "camera": {"source": 0},
    }


def test_chat_sse_streams_done(tmp_path, monkeypatch):
    app = create_app(_cfg(tmp_path))

    from app import analyzer as az_mod
    from app import local_chat_context as lcc_mod

    async def fake_chat_multi(self, frames, message, **kw):
        yield {"type": "token", "delta": "FOUND\nhel"}
        yield {"type": "token", "delta": "lo"}
    monkeypatch.setattr(az_mod.Analyzer, "chat_multi", fake_chat_multi)
    monkeypatch.setattr(lcc_mod.LocalChatContext, "gather_chat_context",
                        lambda self: ({"camera": [b"a"]}, {"camera": [0.0]}, ["camera"]))

    with TestClient(app) as client:
        r = client.post("/api/chat", json={"message": "hi", "mode": "search"})
        assert r.status_code == 200
        assert "text/event-stream" in r.headers["content-type"]
        body = r.text
        assert "event: token" in body
        assert "event: done" in body
        # reconstruct streamed tokens
        deltas = []
        for line in body.splitlines():
            if line.startswith("data: ") and '"delta"' in line:
                import json
                deltas.append(json.loads(line[6:])["delta"])
        assert "".join(deltas) == "hello"


def test_chat_sse_search_clamps_out_of_range_timestamp(tmp_path, monkeypatch):
    app = create_app(_cfg(tmp_path))
    from app import analyzer as az_mod
    from app import local_chat_context as lcc_mod

    async def fake_chat_multi(self, frames, message, **kw):
        # Model leaks a wall-clock time well outside the shown 0..270s window.
        for ch in "FOUND\nA red car ran the light at 12:45:36.":
            yield {"type": "token", "delta": ch}
    monkeypatch.setattr(az_mod.Analyzer, "chat_multi", fake_chat_multi)
    monkeypatch.setattr(lcc_mod.LocalChatContext, "gather_chat_context",
                        lambda self: ({"camera": [b"a", b"b"]}, {"camera": [0.0, 270.0]}, ["camera"]))

    with TestClient(app) as client:
        body = client.post("/api/chat", json={"message": "did a car run the light?", "mode": "search"}).text
        deltas = []
        for line in body.splitlines():
            if line.startswith("data: ") and '"delta"' in line:
                import json
                deltas.append(json.loads(line[6:])["delta"])
        answer = "".join(deltas)
        assert "12:45:36" not in answer                 # out-of-range time scrubbed
        assert "00:00:00–00:04:30" in answer            # replaced with the shown footage range


def _jpeg(val):
    import cv2, numpy as np
    img = np.full((16, 16, 3), val, np.uint8)
    return cv2.imencode(".jpg", img)[1].tobytes()


def _done_data(body):
    import json
    for line in body.splitlines():
        if line.startswith("data: ") and '"session_id"' in line:
            return json.loads(line[6:])
    return None


def test_local_backend_search_found_falls_back_to_a_frames_clip(tmp_path, monkeypatch):
    """Local backend has no segment recorder, so there is no footage to cut. It
    used to surface no clip at all; it now falls back to the frames the model was
    shown, tagged sources={"camera": "frames"} so the UI can say what it is."""
    app = create_app(_cfg(tmp_path))
    from app import analyzer as az_mod
    from app import local_chat_context as lcc_mod

    async def fake_chat_multi(self, frames, message, **kw):
        for ch in "FOUND\nA red car at 00:00:01.":
            yield {"type": "token", "delta": ch}
    monkeypatch.setattr(az_mod.Analyzer, "chat_multi", fake_chat_multi)
    monkeypatch.setattr(lcc_mod.LocalChatContext, "gather_chat_context",
                        lambda self: ({"camera": [_jpeg(40), _jpeg(120), _jpeg(200)]},
                                      {"camera": [1.0, 2.0, 3.0]}, ["camera"]))

    with TestClient(app) as client:
        body = client.post("/api/chat", json={"message": "find the red car", "mode": "search"}).text
        clip = _done_data(body)["clip"]
        assert clip is not None
        assert clip["cameras"] == ["camera"]
        assert clip["sources"] == {"camera": "frames"}
        assert clip["range"] == [1.0, 3.0]
        assert client.get(f"/api/chat/clip/{clip['id']}/camera").status_code == 200
        assert client.get("/api/chat/clip/nope/camera").status_code == 404


def _cfg_deepstream(tmp_path):
    sidecar_yaml = tmp_path / "sidecar.yaml"
    sidecar_yaml.write_text(yaml.safe_dump({
        "analytics_config_path": str(tmp_path / "nvdsanalytics_config.txt"),
    }))
    return {
        "perception": {"backend": "deepstream", "redis_url": "redis://localhost:6379/0"},
        "segments": {"enabled": True},
        "api": {"backend": "cloud", "api_key": "x", "model": "m"},
        # vlm_enabled defaults to False in create_app, and /api/chat 503s when it is
        # off -- chat tests must opt in explicitly.
        "monitoring": {"vlm_enabled": True},
        "sampling": {}, "events": {"db_path": str(tmp_path / "e.db")},
        "chat": {"tools_enabled": False, "preview": {"dir": str(tmp_path / "clips")}},
        "cv": {"enabled": False}, "camera": {"source": 0},
        "zones": {
            "file_path": str(tmp_path / "zones_config.txt"),
            "sidecar_config_path": str(sidecar_yaml),
        },
    }


def test_deepstream_search_found_builds_preview_clip_from_segments(tmp_path, monkeypatch):
    import subprocess
    import fakeredis

    monkeypatch.setattr("redis.Redis.from_url", lambda url: fakeredis.FakeRedis())

    class _FakeMinioClient:
        def __init__(self, *a, **kw):
            self.objects = {}

        def bucket_exists(self, bucket):
            return True

        def make_bucket(self, bucket):
            pass

        def fput_object(self, bucket, key, file_path):
            with open(file_path, "rb") as fh:
                self.objects[(bucket, key)] = fh.read()

        def fget_object(self, bucket, key, dest_path):
            with open(dest_path, "wb") as fh:
                fh.write(self.objects[(bucket, key)])

        def remove_object(self, bucket, key):
            self.objects.pop((bucket, key), None)

    monkeypatch.setattr("minio.Minio", _FakeMinioClient)

    app = create_app(_cfg_deepstream(tmp_path))

    # Seed one real H.264 segment covering stream offsets [0, 4) into the app's
    # own SegmentStore (exposed via app.state for exactly this purpose).
    seg_path = tmp_path / "seg0.ts"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=4:size=64x64:rate=10",
         "-c:v", "libx264", "-g", "1", "-pix_fmt", "yuv420p", "-f", "mpegts", str(seg_path)],
        check=True, capture_output=True,
    )
    app.state.segment_store.put("cam0", start_pts=0.0, end_pts=4.0, local_path=str(seg_path))

    from app import analyzer as az_mod
    from app.usecases import chat as chat_mod

    async def fake_chat_multi(self, frames, message, **kw):
        for ch in "FOUND\nA red car on cam0 at 00:00:02.":
            yield {"type": "token", "delta": ch}
    monkeypatch.setattr(az_mod.Analyzer, "chat_multi", fake_chat_multi)
    monkeypatch.setattr(chat_mod.ChatHistory, "gather_chat_context",
                        lambda self: ({"cam0": [_jpeg(40), _jpeg(120)]},
                                      {"cam0": [1.0, 3.0]}, ["cam0"]))

    with TestClient(app) as client:
        body = client.post("/api/chat", json={"message": "where is the red car?", "mode": "search"}).text
        clip = _done_data(body)["clip"]
        assert clip is not None
        assert clip["cameras"] == ["cam0"]
        # clip_trim_enabled defaults to False, so build_clip never trims --
        # range is the full fetched segment's own span (the single seeded
        # segment [0.0, 4.0)), not the original [lo,hi]=[1.0,3.0] evidence window.
        assert clip["range"] == [0.0, 4.0]
        r = client.get(f"/api/chat/clip/{clip['id']}/cam0")
        assert r.status_code == 200
        assert r.headers["content-type"] == "video/mp4"
        assert len(r.content) > 0


def test_deepstream_clip_build_failure_still_completes_turn(tmp_path, monkeypatch):
    """build_preview_clip can raise (ffmpeg failure, or a segment evicted by the
    retention-prune loop between listing and fetch). That must degrade to
    clip=None, not strand the SSE stream before its 'done' event."""
    import fakeredis

    monkeypatch.setattr("redis.Redis.from_url", lambda url: fakeredis.FakeRedis())

    class _FakeMinioClient:
        def __init__(self, *a, **kw):
            self.objects = {}

        def bucket_exists(self, bucket):
            return True

        def make_bucket(self, bucket):
            pass

    monkeypatch.setattr("minio.Minio", _FakeMinioClient)

    app = create_app(_cfg_deepstream(tmp_path))

    from app import analyzer as az_mod
    from app.usecases import chat as chat_mod
    from app import server as srv_mod

    def _boom(*a, **kw):
        raise RuntimeError("segment missing (evicted by retention prune)")
    monkeypatch.setattr(srv_mod, "build_preview_clip", _boom)

    async def fake_chat_multi(self, frames, message, **kw):
        for ch in "FOUND\nA red car on cam0 at 00:00:02.":
            yield {"type": "token", "delta": ch}
    monkeypatch.setattr(az_mod.Analyzer, "chat_multi", fake_chat_multi)
    monkeypatch.setattr(chat_mod.ChatHistory, "gather_chat_context",
                        lambda self: ({"cam0": [_jpeg(40), _jpeg(120)]},
                                      {"cam0": [1.0, 3.0]}, ["cam0"]))

    with TestClient(app) as client:
        body = client.post("/api/chat", json={"message": "where is the red car?", "mode": "search"}).text
        assert "event: done" in body   # turn completes despite the clip-build failure
        done = _done_data(body)
        assert done is not None
        assert done["clip"] is None


def test_deepstream_found_with_no_segments_falls_back_to_a_frames_clip(tmp_path, monkeypatch):
    """No recorded segment covers the FOUND window (an RTSP source under
    smart-record, or a gap in the index). The turn must still return a playable
    clip, built from the frames the model was actually shown, and say so via
    clip["sources"]."""
    import fakeredis

    monkeypatch.setattr("redis.Redis.from_url", lambda url: fakeredis.FakeRedis())

    class _FakeMinioClient:
        def __init__(self, *a, **kw):
            self.objects = {}

        def bucket_exists(self, bucket):
            return True

        def make_bucket(self, bucket):
            pass

    monkeypatch.setattr("minio.Minio", _FakeMinioClient)

    app = create_app(_cfg_deepstream(tmp_path))
    # Deliberately seed NO segments, unlike the test above.

    from app import analyzer as az_mod
    from app.usecases import chat as chat_mod

    async def fake_chat_multi(self, frames, message, **kw):
        for ch in "FOUND\nA red car on cam0 at 00:00:02.":
            yield {"type": "token", "delta": ch}
    monkeypatch.setattr(az_mod.Analyzer, "chat_multi", fake_chat_multi)
    monkeypatch.setattr(chat_mod.ChatHistory, "gather_chat_context",
                        lambda self: ({"cam0": [_jpeg(40), _jpeg(120), _jpeg(200)]},
                                      {"cam0": [1.0, 2.0, 3.0]}, ["cam0"]))

    with TestClient(app) as client:
        body = client.post("/api/chat",
                           json={"message": "where is the red car?", "mode": "search"}).text
        clip = _done_data(body)["clip"]
        assert clip is not None, "a FOUND with no segments must still return a clip"
        assert clip["cameras"] == ["cam0"]
        assert clip["sources"] == {"cam0": "frames"}
        # Range comes from the frames' own offsets, so it agrees with the
        # timestamps clamp_answer_timestamps already put in the answer.
        assert clip["range"] == [1.0, 3.0]

        r = client.get(f"/api/chat/clip/{clip['id']}/cam0")
        assert r.status_code == 200
        assert r.headers["content-type"] == "video/mp4"
        assert len(r.content) > 0


def test_search_not_found_has_null_clip(tmp_path, monkeypatch):
    app = create_app(_cfg_search(tmp_path))
    from app import analyzer as az_mod
    from app import local_chat_context as lcc_mod

    async def fake_chat_multi(self, frames, message, **kw):
        yield {"type": "token", "delta": "NOT_FOUND"}
    monkeypatch.setattr(az_mod.Analyzer, "chat_multi", fake_chat_multi)
    monkeypatch.setattr(lcc_mod.LocalChatContext, "gather_chat_context",
                        lambda self: ({"camera": [_jpeg(40)]}, {"camera": [1.0]}, ["camera"]))
    monkeypatch.setattr(lcc_mod.LocalChatContext, "sweep_history_snapshots",
                        lambda self: {"camera": []})
    monkeypatch.setattr(lcc_mod.LocalChatContext, "sweep_live_window",
                        lambda self, since: ({}, {}, since))

    with TestClient(app) as client:
        body = client.post("/api/chat", json={"message": "find a unicorn", "mode": "search"}).text
        done = _done_data(body)
        assert done is not None and done.get("clip") is None


def test_chat_sse_503_when_no_frames(tmp_path, monkeypatch):
    app = create_app(_cfg(tmp_path))
    from app import local_chat_context as lcc_mod
    monkeypatch.setattr(lcc_mod.LocalChatContext, "gather_chat_context",
                        lambda self: ({}, {}, []))
    with TestClient(app) as client:
        r = client.post("/api/chat", json={"message": "hi"})
        assert r.status_code == 503


def _cfg_search(tmp_path):
    c = _cfg(tmp_path)
    c["chat"].update({"history": {"window_seconds": 10, "fps": 2},
                      "search": {"enabled": True, "round_seconds": 4,
                                 "window_frames": 2, "max_calls": 12, "max_seconds": 30}})
    return c


def test_low_confidence_first_pass_found_falls_through_to_sweep(tmp_path, monkeypatch):
    # A FOUND whose self-reported confidence is below found_min_confidence must be
    # downgraded to NOT_FOUND: no answer emitted, the history sweep runs, null clip.
    cfg = _cfg_search(tmp_path)
    cfg["chat"]["search"]["found_min_confidence"] = 0.7
    app = create_app(cfg)
    from app import analyzer as az_mod
    from app import local_chat_context as lcc_mod

    async def fake_chat_multi(self, frames, message, **kw):
        text = "FOUND 0.3\nmaybe a car" if kw.get("use_tools") else "NOT_FOUND"
        for ch in text:
            yield {"type": "token", "delta": ch}

    monkeypatch.setattr(az_mod.Analyzer, "chat_multi", fake_chat_multi)
    monkeypatch.setattr(lcc_mod.LocalChatContext, "gather_chat_context",
                        lambda self: ({"camera": [_jpeg(1)]}, {"camera": [0.0]}, ["camera"]))
    monkeypatch.setattr(lcc_mod.LocalChatContext, "sweep_history_snapshots",
                        lambda self: {"camera": [(float(i), _jpeg(i)) for i in range(6)]})
    monkeypatch.setattr(lcc_mod.LocalChatContext, "sweep_live_window",
                        lambda self, since: ({}, {}, since))

    with TestClient(app) as client:
        body = client.post("/api/chat", json={"message": "any red car?", "mode": "search"}).text
        assert "searching_history" in body          # low-conf FOUND did NOT short-circuit
        assert "maybe a car" not in body            # the low-conf answer was suppressed
        assert _done_data(body)["clip"] is None     # no preview clip from a rejected match


def test_chat_sse_runs_history_sweep_on_not_found(tmp_path, monkeypatch):
    app = create_app(_cfg_search(tmp_path))
    from app import analyzer as az_mod
    from app import local_chat_context as lcc_mod

    state = {"round": 0}

    async def fake_chat_multi(self, frames, message, **kw):
        # First pass + sweep rounds all stream a marker. Find it on the 3rd probe.
        state["round"] += 1
        text = "FOUND\nthere it is" if state["round"] == 3 else "NOT_FOUND"
        for ch in text:
            yield {"type": "token", "delta": ch}

    monkeypatch.setattr(az_mod.Analyzer, "chat_multi", fake_chat_multi)
    # Provide history snapshots + a live window so the sweep has frames to probe.
    monkeypatch.setattr(lcc_mod.LocalChatContext, "gather_chat_context",
                        lambda self: ({"camera": [b"a"]}, {"camera": [0.0]}, ["camera"]))
    monkeypatch.setattr(lcc_mod.LocalChatContext, "sweep_history_snapshots",
                        lambda self: {"camera": [(float(i), bytes([i])) for i in range(20)]})
    monkeypatch.setattr(lcc_mod.LocalChatContext, "sweep_live_window",
                        lambda self, since: ({}, {}, since))

    with TestClient(app) as client:
        body = client.post("/api/chat", json={"message": "did a car run the light?", "mode": "search"}).text
        assert "event: status" in body
        assert "searching_history" in body
        # final answer streamed
        deltas = []
        for line in body.splitlines():
            if line.startswith("data: ") and '"delta"' in line:
                import json
                deltas.append(json.loads(line[6:])["delta"])
        assert "there it is" in "".join(deltas)
        assert "event: done" in body


def test_found_first_pass_clears_stale_resume_cursor(tmp_path, monkeypatch):
    app = create_app(_cfg_search(tmp_path))
    from app import analyzer as az_mod
    from app import local_chat_context as lcc_mod
    from app import server as srv_mod

    # First-pass marker per request: NOT_FOUND, then FOUND, then NOT_FOUND.
    first_pass = iter(["NOT_FOUND", "FOUND\nthere it is", "NOT_FOUND"])

    async def fake_chat_multi(self, frames, message, **kw):
        # Sweep probes call with use_tools=False and always miss; first passes pull from the iterator.
        text = "NOT_FOUND" if kw.get("use_tools") is False else next(first_pass)
        for ch in text:
            yield {"type": "token", "delta": ch}

    monkeypatch.setattr(az_mod.Analyzer, "chat_multi", fake_chat_multi)
    monkeypatch.setattr(lcc_mod.LocalChatContext, "gather_chat_context",
                        lambda self: ({"camera": [b"a"]}, {"camera": [0.0]}, ["camera"]))
    # 1000 frames so the first sweep hits the call cap (not history-exhausted) and SETS a cursor.
    monkeypatch.setattr(lcc_mod.LocalChatContext, "sweep_history_snapshots",
                        lambda self: {"camera": [(float(i), bytes([i % 256])) for i in range(1000)]})
    monkeypatch.setattr(lcc_mod.LocalChatContext, "sweep_live_window",
                        lambda self, since: ({}, {}, since))

    seen_start = []
    real_run_sweep = srv_mod.run_sweep

    async def spy_run_sweep(**kw):
        seen_start.append(kw.get("start_index"))
        async for it in real_run_sweep(**kw):
            yield it

    monkeypatch.setattr(srv_mod, "run_sweep", spy_run_sweep)

    import json as _json

    def _sid(body):
        for line in body.splitlines():
            if line.startswith("data: ") and '"session_id"' in line:
                return _json.loads(line[6:])["session_id"]
        return None

    with TestClient(app) as client:
        b1 = client.post("/api/chat", json={"message": "did a car run the light?", "mode": "search"}).text
        sid = _sid(b1)
        assert sid is not None
        client.post("/api/chat", json={"message": "what's there now?", "session_id": sid, "mode": "search"})   # FOUND -> clears
        client.post("/api/chat", json={"message": "and earlier?", "session_id": sid, "mode": "search"})         # NOT_FOUND -> fresh

    # run_sweep ran for req1 and req3 (req2 was FOUND). Both must start fresh (cursor cleared by req2).
    assert seen_start == [None, None]


def test_chat_sse_conversation_does_not_sweep(tmp_path, monkeypatch):
    app = create_app(_cfg_search(tmp_path))
    from app import analyzer as az_mod
    from app import local_chat_context as lcc_mod

    async def fake_chat_multi(self, frames, message, **kw):
        for ch in "CONVERSATION\nHello, I watch the cameras.":
            yield {"type": "token", "delta": ch}

    monkeypatch.setattr(az_mod.Analyzer, "chat_multi", fake_chat_multi)
    monkeypatch.setattr(lcc_mod.LocalChatContext, "gather_chat_context",
                        lambda self: ({"camera": [b"a"]}, {"camera": [0.0]}, ["camera"]))

    with TestClient(app) as client:
        body = client.post("/api/chat", json={"message": "hi there", "mode": "search"}).text
        assert "searching_history" not in body   # no sweep on a conversational turn
        deltas = []
        for line in body.splitlines():
            if line.startswith("data: ") and '"delta"' in line:
                import json
                deltas.append(json.loads(line[6:])["delta"])
        assert "Hello, I watch the cameras." in "".join(deltas)
        assert "event: done" in body


def test_chat_normal_mode_streams_plain_answer(tmp_path, monkeypatch):
    app = create_app(_cfg(tmp_path))
    from app import analyzer as az_mod
    from app import local_chat_context as lcc_mod

    seen_sweep = []

    async def fake_chat_multi(self, frames, message, **kw):
        seen_sweep.append(kw.get("sweep"))
        yield {"type": "token", "delta": "Two cars are visible."}

    monkeypatch.setattr(az_mod.Analyzer, "chat_multi", fake_chat_multi)
    monkeypatch.setattr(lcc_mod.LocalChatContext, "gather_chat_context",
                        lambda self: ({"camera": [b"a"]}, {"camera": [0.0]}, ["camera"]))

    with TestClient(app) as client:
        body = client.post("/api/chat",
                           json={"message": "what do you see?", "mode": "normal"}).text
        assert "searching_history" not in body
        assert "event: done" in body
        deltas = []
        for line in body.splitlines():
            if line.startswith("data: ") and '"delta"' in line:
                import json
                deltas.append(json.loads(line[6:])["delta"])
        assert "".join(deltas) == "Two cars are visible."
        # Normal mode must call the analyzer with sweep=False (marker-free prompt).
        assert seen_sweep == [False]


def test_chat_normal_mode_never_sweeps_on_not_found(tmp_path, monkeypatch):
    # Even with search config present and the model emitting NOT_FOUND, Normal
    # mode must never launch the history sweep — the mode gates it, not the marker.
    app = create_app(_cfg_search(tmp_path))
    from app import analyzer as az_mod
    from app import local_chat_context as lcc_mod

    async def fake_chat_multi(self, frames, message, **kw):
        yield {"type": "token", "delta": "NOT_FOUND"}

    monkeypatch.setattr(az_mod.Analyzer, "chat_multi", fake_chat_multi)
    monkeypatch.setattr(lcc_mod.LocalChatContext, "gather_chat_context",
                        lambda self: ({"camera": [b"a"]}, {"camera": [0.0]}, ["camera"]))
    monkeypatch.setattr(lcc_mod.LocalChatContext, "sweep_history_snapshots",
                        lambda self: {"camera": [(float(i), bytes([i])) for i in range(20)]})

    with TestClient(app) as client:
        body = client.post("/api/chat",
                           json={"message": "did a car run the light?"}).text  # no mode -> default normal
        assert "searching_history" not in body
        assert "event: done" in body


def test_chat_turn_is_logged(tmp_path, monkeypatch, caplog):
    import logging
    app = create_app(_cfg(tmp_path))
    from app import analyzer as az_mod
    from app import local_chat_context as lcc_mod

    async def fake_chat_multi(self, frames, message, **kw):
        yield {"type": "token", "delta": "FOUND\nyes"}
    monkeypatch.setattr(az_mod.Analyzer, "chat_multi", fake_chat_multi)
    monkeypatch.setattr(lcc_mod.LocalChatContext, "gather_chat_context",
                        lambda self: ({"camera": [b"a", b"b"]}, {"camera": [0.0, 1.0]}, ["camera"]))

    with caplog.at_level(logging.INFO, logger="iris.chat"):
        with TestClient(app) as client:
            client.post("/api/chat", json={"message": "hello there"})
    text = "\n".join(r.getMessage() for r in caplog.records if r.name == "iris.chat")
    assert "[chat] turn" in text
    assert "camera:2" in text   # per-stream frame count
