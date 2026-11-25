import asyncio
from app.scheduler import Scheduler


class _Cam:
    is_running = True
    is_paused = False


def _make(active_mode):
    cfg = {
        "sampling": {"interval_seconds": 0.01},
        "monitoring": {"active_mode": active_mode, "chat_in_flight": 0},
    }
    return Scheduler(cfg, _Cam(), None, None, None, None, None)


def _run_loop_briefly(sched):
    calls = []

    async def fake_cycle():
        calls.append(1)

    sched._cycle = fake_cycle

    async def run():
        task = asyncio.create_task(sched.run())
        await asyncio.sleep(0.06)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(run())
    return calls


def test_scheduler_skips_cycle_in_chat_mode():
    assert _run_loop_briefly(_make("chat")) == []


def test_scheduler_runs_cycle_in_proactive_mode():
    assert len(_run_loop_briefly(_make("proactive"))) >= 1
