"""shelf_void core/tracker.py and core/person_gate.py, vendored from shelf-demo: its own tests, ported from shelf-demo
(commit 1953907, tests/test_tracker.py) with only the imports changed."""
import random
import unittest

import numpy as np

from perception.modules.shelf_void.core.person_gate import PersonGate, pad_boxes, region_masks
from perception.modules.shelf_void.core.tracker import StableTracker, iou


# ---- Reference: the frame-count implementation from void_roi.py before the move (commit 6db96d4) ----
class FrameTracker:
    def __init__(self, confirm, max_miss, min_iou=0.3, min_hit_ratio=0.5):
        self.confirm, self.max_miss = confirm, max_miss
        self.min_iou, self.min_hit_ratio = min_iou, min_hit_ratio
        self.tracks = []

    def update(self, boxes, frame_idx):
        for x1, y1, x2, y2, c in boxes:
            box = (x1, y1, x2, y2)
            best = max(self.tracks, key=lambda t: iou(t["box"], box), default=None)
            if best is not None and iou(best["box"], box) >= self.min_iou and best["last"] != frame_idx:
                best.update(box=box, conf=c, last=frame_idx, hits=best["hits"] + 1)
            else:
                self.tracks.append(dict(box=box, conf=c, first=frame_idx, last=frame_idx, hits=1))
        self.tracks = [t for t in self.tracks if frame_idx - t["last"] <= self.max_miss]

    def is_confirmed(self, t, frame_idx):
        age = frame_idx - t["first"] + 1
        return age >= self.confirm and t["hits"] / age >= self.min_hit_ratio

    def freeze(self, frame_idx):
        kept = []
        for t in self.tracks:
            if t.get("confirmed") or self.is_confirmed(t, frame_idx):
                t["confirmed"] = True
                t["last"] = frame_idx
                kept.append(t)
        self.tracks = kept

    def split(self, frame_idx):
        conf, pend = [], []
        for t in self.tracks:
            if t.get("confirmed") or self.is_confirmed(t, frame_idx):
                t["confirmed"] = True
                conf.append(t)
            else:
                pend.append(t)
        return conf, pend


GAP = (100, 100, 150, 200, 0.2)  # a gap that never moves


def run_time_tracker(fps, frames, confirm_s=2.0, miss_s=1.0, seen=lambda i: True):
    """Feed GAP at `fps`; return the first sample index at which it is confirmed (or None)."""
    dt = 1.0 / fps
    tr = StableTracker(confirm_s, miss_s, sample_interval=dt)
    for i in range(frames):
        now = i * dt
        tr.update([GAP] if seen(i) else [], now)
        if tr.split(now)[0]:
            return i
    return None


class StableTrackerTest(unittest.TestCase):
    def test_confirms_on_50th_frame_at_25fps(self):
        # baseline: first seen frame 1128, confirmed frame 1177 (= 1128 + 49)
        self.assertEqual(run_time_tracker(25, 200), 49)

    def test_confirms_on_10th_sample_at_5fps(self):
        self.assertEqual(run_time_tracker(5, 50), 9)  # 10 samples x 0.2 s = 2.0 s

    def test_confirm_seconds_scale(self):
        self.assertEqual(run_time_tracker(5, 100, confirm_s=5.0), 24)

    def test_needs_half_the_samples(self):
        # seen every 3rd sample only: hit ratio 1/3 < 0.5 -> never confirmed (and never forgotten: miss 1 s)
        self.assertIsNone(run_time_tracker(5, 100, seen=lambda i: i % 3 == 0))
        # seen every other sample: ratio >= 0.5 -> confirmed
        self.assertIsNotNone(run_time_tracker(5, 100, seen=lambda i: i % 2 == 0))

    def test_moving_box_never_confirmed(self):
        dt = 0.2
        tr = StableTracker(2.0, 1.0, sample_interval=dt)
        for i in range(50):
            x = 40 * i  # moves a whole box width each sample: IoU 0
            tr.update([(x, 0, x + 40, 40, 0.5)], i * dt)
            self.assertEqual(tr.split(i * dt)[0], [])

    def test_forgets_after_max_miss(self):
        dt = 0.2
        tr = StableTracker(2.0, 1.0, sample_interval=dt)
        tr.update([GAP], 0.0)
        for i in range(1, 6):  # 1.0 s without it: still kept
            tr.update([], i * dt)
        self.assertEqual(len(tr.tracks), 1)
        tr.update([], 6 * dt)  # 1.2 s: gone
        self.assertEqual(tr.tracks, [])

    def test_freeze_keeps_confirmed_and_drops_pending(self):
        dt = 0.2
        tr = StableTracker(2.0, 1.0, sample_interval=dt)
        for i in range(10):
            tr.update([GAP], i * dt)
        tr.update([(400, 400, 450, 450, 0.3)], 10 * dt)  # new candidate, pending
        self.assertEqual(len(tr.split(10 * dt)[1]), 1)
        for i in range(11, 100):  # person blocks the shelf for ~18 s
            tr.freeze(i * dt)
        conf, pend = tr.split(99 * dt)
        self.assertEqual(len(conf), 1)
        self.assertEqual(pend, [])

    def test_matches_frame_implementation_on_random_sequences(self):
        """Exact equivalence with the original frame-count tracker at 25 fps and 5 fps."""
        rng = random.Random(1234)
        for fps in (25, 5):
            dt = 1.0 / fps
            confirm_n, miss_n = int(round(2.0 * fps)), int(round(1.0 * fps))
            for _ in range(200):
                ref = FrameTracker(confirm_n, miss_n)
                new = StableTracker(confirm_n * dt, miss_n * dt, sample_interval=dt)
                anchors = [(rng.randint(0, 800), rng.randint(0, 400)) for _ in range(3)]
                for i in range(300):
                    now = i * dt
                    if rng.random() < 0.15:  # person in region
                        ref.freeze(i)
                        new.freeze(now)
                    else:
                        boxes = []
                        for ax, ay in anchors:
                            if rng.random() < 0.6:
                                jx, jy = rng.randint(-8, 8), rng.randint(-8, 8)
                                boxes.append((ax + jx, ay + jy, ax + jx + 60, ay + jy + 90, rng.random()))
                        ref.update(boxes, i)
                        new.update(boxes, now)
                    rc, rp = ref.split(i)
                    nc, np_ = new.split(now)
                    self.assertEqual([t["box"] for t in rc], [t["box"] for t in nc], f"fps={fps} frame={i}")
                    self.assertEqual([t["box"] for t in rp], [t["box"] for t in np_], f"fps={fps} frame={i}")
                    self.assertEqual([t["hits"] for t in ref.tracks], [t["hits"] for t in new.tracks])

    def test_rejects_bad_interval(self):
        with self.assertRaises(ValueError):
            StableTracker(2.0, 1.0, sample_interval=0)


class PersonGateTest(unittest.TestCase):
    W, H = 200, 100
    LEFT = [[0, 0], [99, 0], [99, 99], [0, 99]]
    RIGHT = [[120, 0], [199, 0], [199, 99], [120, 99]]

    def gate(self, resume_s):
        return PersonGate(region_masks([self.LEFT, self.RIGHT], self.W, self.H), resume_s)

    def test_starts_active(self):
        self.assertEqual(self.gate(0.4).active(0.0), [True, True])

    def test_only_touched_region_pauses_and_resumes_after_0_4s_at_5fps(self):
        g = self.gate(0.4)
        person = [(10, 10, 30, 60)]  # inside LEFT only
        self.assertEqual(g.update(person, 0.0), [False, True])
        self.assertEqual(g.update([], 0.2), [False, True])  # 1 person-free sample
        self.assertEqual(g.update([], 0.4), [True, True])   # 2 samples = 0.4 s -> resumes

    def test_matches_frame_streak_at_25fps(self):
        """Same active flags as the original clear_streak / resume_after=10 logic."""
        rng = random.Random(7)
        resume_after, dt = 10, 1 / 25
        g = self.gate(resume_after * dt)
        streak = [resume_after] * 2
        for i in range(2000):
            persons = [(rng.randint(0, 180), 0, rng.randint(0, 180) + 20, 50)] if rng.random() < 0.1 else []
            persons = [(min(a, c), b, max(a, c), d) for a, b, c, d in persons]
            touched = g.touched(persons)
            streak = [0 if t else s + 1 for t, s in zip(touched, streak)]
            self.assertEqual(g.update(persons, i * dt), [s >= resume_after for s in streak], f"frame {i}")

    def test_pad_boxes_grows_and_clips(self):
        self.assertEqual(pad_boxes([(10, 10, 110, 60)], 0.15, 200, 100), [(0, 2, 125, 67)])
        self.assertEqual(pad_boxes([(150, 50, 199, 99)], 0.5, 200, 100), [(125, 25, 200, 100)])

    def test_padding_makes_near_person_touch_region(self):
        g = self.gate(0.4)
        near = (102, 10, 118, 60)  # in the gap between LEFT (x<=99) and RIGHT (x>=120)
        self.assertEqual(g.touched([near]), [False, False])
        # 30% of 16 px = 4.8 px per side -> x 97..122 (slice end exclusive, as in the original)
        self.assertEqual(g.touched(pad_boxes([near], 0.3, self.W, self.H)), [True, True])
        # 15% -> x 99..120: reaches LEFT's last column but stops just short of RIGHT
        self.assertEqual(g.touched(pad_boxes([near], 0.15, self.W, self.H)), [True, False])


if __name__ == "__main__":
    unittest.main()
