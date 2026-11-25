"""The live websocket's fan-out (2026-10-15): a client that stopped reading (a CLOSE-WAIT socket holding 2.6 MB
unsent) blocked every broadcast, so the dashboards behind it got the first snapshot and then frozen boxes."""
import asyncio

from server import ConnectionManager


class FakeWS:
    def __init__(self, stuck=False):
        self.sent, self.stuck, self.closed = [], stuck, False

    async def accept(self):
        pass

    async def send_json(self, m):
        if self.stuck:
            await asyncio.Event().wait()            # never drains
        self.sent.append(m)

    async def close(self, code=1000):
        self.closed = True


def test_a_client_that_stopped_reading_does_not_hold_up_the_others():
    async def run():
        m = ConnectionManager()
        stuck, ok = FakeWS(stuck=True), FakeWS()
        await m.connect(stuck, first={"type": "live_snapshot"})
        await m.connect(ok, first={"type": "live_snapshot"})
        for i in range(40):
            await m._broadcast({"type": "scene", "n": i})
            await asyncio.sleep(0.001)                                  # 5 /s in the service; a healthy client keeps up
        await asyncio.sleep(0.05)
        return m, stuck, ok
    m, stuck, ok = asyncio.run(run())
    assert [x["type"] for x in ok.sent[:2]] == ["live_snapshot", "scene"] and ok.sent[-1] == {"type": "scene", "n": 39}
    assert ok in m._clients
    assert stuck not in m._clients and stuck.closed                    # dropped once its queue overflowed


def test_a_disconnected_client_is_forgotten():
    async def run():
        m = ConnectionManager()
        ws = FakeWS()
        await m.connect(ws, first={"type": "live_snapshot"})
        m.disconnect(ws)
        await m._broadcast({"type": "scene"})
        await asyncio.sleep(0.01)
        return m, ws
    m, ws = asyncio.run(run())
    assert ws not in m._clients and not any(x["type"] == "scene" for x in ws.sent)


def test_a_cancelled_sender_is_kept_until_it_has_finished():
    """22:02 live log: 'Task was destroyed but it is pending!' -- disconnect() cancelled the sender and dropped the
    last strong reference to it (the loop only holds tasks weakly), so it could be collected mid-cancel."""
    async def run():
        m = ConnectionManager()
        ws = FakeWS()
        await m.connect(ws, first={"type": "live_snapshot"})
        await asyncio.sleep(0.01)
        task = m._senders[ws]
        m.disconnect(ws)
        held = task in m._ending
        await asyncio.sleep(0.01)
        return held, task.done(), task in m._ending
    assert asyncio.run(run()) == (True, True, False)
