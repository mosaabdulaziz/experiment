from app.scheduler import Scheduler


def test_pending_depth_accessor():
    s = Scheduler.__new__(Scheduler)  # bare instance; only test the pure accessor
    s._pending = 2
    assert s.pending_depth() == 2
