"""shelf_void core/situations.py, vendored from shelf-demo: its own tests, ported from shelf-demo (commit 1953907,
tests/test_service.py, SituationTest) with only the imports changed."""
import unittest
from datetime import datetime, timedelta, timezone

from perception.modules.shelf_void.core.situations import Situation, Spot

T0 = datetime(2026, 11, 12, 10, 0, 0, tzinfo=timezone.utc)


def spot(stab=0.9, score=0.11):
    return Spot([0.69, 0.66, 0.72, 0.76], score, stab)


class SituationTest(unittest.TestCase):
    """Synthetic sequences at 5 fps (0.2 s samples)."""

    def run_seq(self, sit, seq, dt=0.2, start=0):
        """seq: list of (n_spots, active), sample i at t = (start + i) * dt. Returns [(t, state, reason, n)]."""
        out = []
        for i, (n, active) in enumerate(seq, start):
            t = i * dt
            for it in sit.update([spot()] * n, active, t, T0 + timedelta(seconds=t)):
                out.append((round(t, 1), it.state, it.reason, len(it.spots)))
        return out

    def test_raise_once_then_quiet(self):
        sit = Situation(clear_seconds=10, reraise_seconds=300)
        ev = self.run_seq(sit, [(0, True)] * 5 + [(1, True)] * 100)
        self.assertEqual(ev, [(1.0, "raised", "new", 1)])

    def test_more_spots_reraise_fewer_do_not(self):
        sit = Situation(10, 300)
        ev = self.run_seq(sit, [(1, True)] * 5 + [(2, True)] * 5 + [(1, True)] * 5 + [(2, True)] * 5 + [(3, True)])
        self.assertEqual([e[1:] for e in ev], [("raised", "new", 1), ("raised", "more", 2), ("raised", "more", 3)])

    def test_clears_after_clear_seconds_of_zero(self):
        sit = Situation(clear_seconds=10, reraise_seconds=300)
        ev = self.run_seq(sit, [(1, True)] * 10 + [(0, True)] * 60)
        # zero from t=2.0; 10 s of zero-to-zero samples -> cleared at t=12.0
        self.assertEqual(ev, [(0.0, "raised", "new", 1), (12.0, "cleared", "gone", 0)])

    def test_spot_returning_resets_clear_timer(self):
        sit = Situation(10, 300)
        seq = [(1, True)] + [(0, True)] * 40 + [(1, True)] + [(0, True)] * 40  # 8 s gaps < 10 s
        self.assertEqual([e[1] for e in self.run_seq(sit, seq)], ["raised"])

    def test_person_time_does_not_count_towards_clear(self):
        sit = Situation(10, 300)
        # 6 s clear, 20 s person, 3.8 s clear: total person-free zero time < 10 s -> still open
        seq = [(1, True)] + [(0, True)] * 31 + [(0, False)] * 100 + [(0, True)] * 20
        self.assertEqual([e[1] for e in self.run_seq(sit, seq)], ["raised"])
        ev = self.run_seq(sit, [(0, True)] * 30, start=len(seq))  # continue: crosses 10 s of person-free zero time
        self.assertEqual([e[1:3] for e in ev], [("cleared", "gone")])

    def test_periodic_reraise_every_5_minutes(self):
        sit = Situation(30, 300)
        ev = self.run_seq(sit, [(1, True)] * int(11 * 60 / 0.2), dt=0.2)
        self.assertEqual([(t, r) for t, _, r, _ in ev], [(0.0, "new"), (300.0, "periodic"), (600.0, "periodic")])

    def test_raise_after_clear_opens_new(self):
        sit = Situation(5, 300)
        ev = self.run_seq(sit, [(1, True)] * 3 + [(0, True)] * 40 + [(1, True)] * 3)
        self.assertEqual([e[1:3] for e in ev], [("raised", "new"), ("cleared", "gone"), ("raised", "new")])

    def test_close_only_when_open(self):
        sit = Situation(30, 300)
        self.assertEqual(sit.close(T0, "paused"), [])
        sit.update([spot()], True, 0.0, T0)
        [it] = sit.close(T0, "camera_removed")
        self.assertEqual((it.state, it.reason), ("cleared", "camera_removed"))
        self.assertEqual(sit.close(T0, "paused"), [])

    def test_naive_wall_rejected(self):
        with self.assertRaises(ValueError):
            Situation(30, 300).update([spot()], True, 0.0, datetime(2026, 11, 12))
