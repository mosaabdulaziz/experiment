import asyncio
import fakeredis
from app.stream_manager import StreamManager


def _mgr():
    cfg = {"perception": {"redis_url": "redis://x", "queue_depth": 3},
           "sampling": {"interval_seconds": 1}, "monitoring": {}, "chat": {}}
    return StreamManager(cfg, analyzer=None, store=None, conn_manager=None,
                         redis_client=fakeredis.FakeRedis())


def test_queue_depths_reflects_pending_items():
    m = _mgr()
    m.analysis.queues = {"cam0": asyncio.Queue(maxsize=3), "cam1": asyncio.Queue(maxsize=3)}
    m.analysis.queues["cam0"].put_nowait(("cam0", 0, [], [], []))
    m.analysis.queues["cam0"].put_nowait(("cam0", 0, [], [], []))
    depths = m.queue_depths()
    assert depths == {"cam0": 2, "cam1": 0}


def test_worker_state_none_when_idle():
    m = _mgr()
    assert m.worker_state() is None
