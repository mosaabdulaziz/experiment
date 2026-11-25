"""The frame publisher sheds previews, never detection data (perception/engine/publisher.py can only be imported
inside the DeepStream image, so the class is lifted out of the source and run with small stand-ins)."""
import ast
import collections
import logging
import pathlib
import queue
import threading
import time
import types

SRC = (pathlib.Path(__file__).resolve().parents[1] / "perception" / "engine" / "publisher.py").read_text()


def _load_publisher():
    tree = ast.parse(SRC)
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_FramePublisher")
    ns = {
        "collections": collections, "queue": queue, "threading": threading, "time": time,
        "logger": logging.getLogger("t"), "_DISCARD_QUEUE": object(),
        "telemetry": types.SimpleNamespace(timeit=lambda _n: (lambda f: f),
                                           stage=lambda _n: types.SimpleNamespace(
                                               __enter__=lambda s: s, __exit__=lambda *a: False)),
        "GpuJpegEncoder": None, "ViewerGate": None, "redis": types.SimpleNamespace(RedisError=Exception),
    }
    exec(compile(ast.Module([node], []), "publisher.py", "exec"), ns)
    return ns["_FramePublisher"]


class _Cfg:
    gpu_jpeg = False
    pipeline_preview = False
    viewer_gated_preview = False
    jpeg_encoders = 3


def _job(cam, n, need_meta=True):
    st = types.SimpleNamespace(id=cam)
    return {"st": st, "need_meta": need_meta, "need_stream": True, "n": n}


def _publisher(published, gate=None):
    P = _load_publisher()
    pub = P(_Cfg(), max_pending=2)
    gate = gate or threading.Event()

    def fake_publish(job):                     # stands in for the draw + redis work
        if job["need_stream"]:
            gate.wait(5)                       # a slow preview: the worker is stuck here
        published.append((job["st"].id, job["n"], job["need_stream"]))
    pub._publish = fake_publish
    return pub, gate


def test_a_full_queue_sheds_the_preview_but_still_publishes_the_detections():
    published = []
    pub, gate = _publisher(published)
    for n in range(8):                         # the worker is blocked on job 0; the queue holds 2
        pub.submit(_job("cam-01", n))
    time.sleep(0.3)
    shed_meta = sorted(n for _c, n, stream in published if not stream)
    assert shed_meta and pub.shed_count("cam-01") == len(shed_meta)
    assert pub._drops == 0                     # nothing was lost
    gate.set()
    time.sleep(0.5)
    seen = sorted(n for _c, n, _s in published)
    assert seen == list(range(8))              # every frame's detections went out exactly once
    pub.stop()


def test_a_preview_only_job_is_shed_without_a_meta_entry():
    published = []
    pub, gate = _publisher(published)
    for n in range(6):
        pub.submit(_job("cam-02", n, need_meta=False))
    time.sleep(0.2)
    assert pub.shed_count("cam-02") > 0 and not [p for p in published if not p[2]]
    gate.set()
    pub.stop()


def test_detections_are_only_lost_when_the_safety_queue_is_full_too():
    published = []
    pub, gate = _publisher(published)
    pub._meta_q = queue.Queue(maxsize=1)
    pub._publish = lambda job: time.sleep(5) if not job["need_stream"] else gate.wait(5)   # nothing drains
    time.sleep(0.1)
    for n in range(10):
        pub.submit(_job("cam-03", n))
    time.sleep(0.2)
    assert pub._drops > 0 and pub._drops_by["cam-03"] == pub._drops
    gate.set()


def test_cameras_are_spread_evenly_over_the_encoders_and_keep_theirs():
    pub, gate = _publisher([])
    pub._encoders = [("e0", threading.Lock()), ("e1", threading.Lock()), ("e2", threading.Lock())]
    for i in range(1, 10):
        pub._worker_for(f"cam-{i:02d}")
    used = collections.Counter(pub.encoder_for(f"cam-{i:02d}")[0] for i in range(1, 10))
    assert used == {"e0": 3, "e1": 3, "e2": 3}
    assert pub.encoder_for("cam-04")[0] == pub.encoder_for("cam-04")[0]
    gate.set()


def test_no_encoder_means_the_cv2_fallback():
    pub, gate = _publisher([])
    assert pub.encoder_for("cam-01") == (None, None)
    gate.set()
