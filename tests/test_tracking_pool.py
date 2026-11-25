"""The tracking loop must not wait behind the app's other blocking work (see TrackingService._in_pool)."""
import asyncio
import threading
import time

from app.tracking import TrackingService


class _Manager:
    async def broadcast(self, *_a, **_k):
        pass


def test_a_saturated_default_executor_does_not_delay_the_tracking_loop():
    svc = TrackingService({"mtmc": {"cameras": ["cam-01"]}, "truck_feed": {}}, _Manager())
    release = threading.Event()

    async def scenario():
        loop = asyncio.get_running_loop()
        # what the stream manager does: park a default-executor thread per camera in a blocking read
        hogs = [loop.run_in_executor(None, release.wait, 5) for _ in range(40)]
        await asyncio.sleep(0.05)
        t0 = time.monotonic()
        out = await svc._in_pool(lambda: "polled")
        took = time.monotonic() - t0
        release.set()
        await asyncio.gather(*hogs)
        return out, took

    out, took = asyncio.run(scenario())
    assert out == "polled" and took < 0.5
    svc._pool.shutdown(wait=False)
