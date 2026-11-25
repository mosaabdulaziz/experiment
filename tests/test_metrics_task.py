import asyncio
import logging
from app.metrics import MetricsRegistry
from app.server import _metrics_loop


def test_metrics_loop_logs_table_once(caplog):
    reg = MetricsRegistry()
    reg.record_analyze("cam0", 2.0, now=100.0)

    async def go():
        task = asyncio.create_task(_metrics_loop(
            reg, lambda: {"cam0": 1}, lambda: "cam0", lambda: 0, interval=0.01))
        await asyncio.sleep(0.03)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    with caplog.at_level(logging.INFO, logger="iris.metrics"):
        asyncio.run(go())
    assert any("analysis status" in r.getMessage() for r in caplog.records)
