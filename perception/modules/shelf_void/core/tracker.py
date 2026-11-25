# Vendored from shelf-demo (~/sarah/shelf-demo), branch feat/iris-integration, commit 195390728433235b0daced2c0f4dc8614a2d212e,
# service/tracker.py. Copied unchanged below this header: fix it there first, then copy it again.
"""Stability check for empty-shelf detections, measured in seconds.

A real gap on a shelf does not move; a head or a hand does. A detection only becomes
"confirmed" once it has stayed in roughly the same place for `confirm_s` seconds.

Times are stream seconds (for a video file: frame_index / fps). Every sample stands for
`sample_interval` seconds, so the maths matches the original frame-count version exactly
when samples are evenly spaced: at 25 fps, confirm_s=2.0 confirms on the 50th frame.
"""

EPS = 1e-6  # float slack so that e.g. 49/25 + 1/25 >= 2.0 holds


def iou(a, b):
    """Intersection-over-Union of two boxes (x1, y1, x2, y2). 1 = identical, 0 = no overlap."""
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class StableTracker:
    """Remembers empty boxes over time in ONE region and only confirms boxes that stay put.

    confirm_s        a box must be tracked this long (first sample to now, inclusive) to be confirmed
    max_miss_s       forget an unconfirmed box not seen for longer than this
    sample_interval  seconds between samples (1/fps for a video, 0.2 for 5 fps sampling)
    min_iou          overlap needed to count a new box as the same gap
    min_hit_ratio    fraction of samples since first seen in which the box must have been detected
    """

    def __init__(self, confirm_s, max_miss_s, sample_interval, min_iou=0.3, min_hit_ratio=0.5):
        if sample_interval <= 0:
            raise ValueError("sample_interval must be > 0")
        self.confirm_s, self.max_miss_s, self.interval = confirm_s, max_miss_s, sample_interval
        self.min_iou, self.min_hit_ratio = min_iou, min_hit_ratio
        self.tracks = []  # each: dict(box, conf, first, last, hits[, confirmed]); first/last in seconds

    def _age(self, t, now):
        """Seconds covered by track t up to `now`, counting the current sample."""
        return now - t["first"] + self.interval

    def update(self, boxes, now):
        """boxes = [(x1, y1, x2, y2, conf), ...] detected at time `now` (seconds)."""
        for x1, y1, x2, y2, c in boxes:
            box = (x1, y1, x2, y2)
            best = max(self.tracks, key=lambda t: iou(t["box"], box), default=None)
            if best is not None and iou(best["box"], box) >= self.min_iou and best["last"] != now:
                best.update(box=box, conf=c, last=now, hits=best["hits"] + 1)  # same gap seen again
            else:
                self.tracks.append(dict(box=box, conf=c, first=now, last=now, hits=1))  # new candidate
        # forget boxes we have not seen for a while
        self.tracks = [t for t in self.tracks if now - t["last"] <= self.max_miss_s + EPS]

    def is_confirmed(self, t, now):
        age = self._age(t, now)
        samples = age / self.interval
        return age >= self.confirm_s - EPS and t["hits"] / samples >= self.min_hit_ratio - EPS

    def stability(self, t, now):
        """0..1 fraction of samples since first seen in which this gap was detected (hit ratio).
        Samples while frozen (person blocking) count as misses, as in is_confirmed."""
        return min(1.0, t["hits"] / (self._age(t, now) / self.interval))

    def progress(self, t, now):
        """0..1 how far an unconfirmed track is towards confirm_s (for drawing)."""
        return min(1.0, self._age(t, now) / self.confirm_s) if self.confirm_s > 0 else 1.0

    def freeze(self, now):
        """Called while a person blocks the region: keep confirmed gaps alive, drop unconfirmed
        candidates (they may have been created by the person)."""
        kept = []
        for t in self.tracks:
            if t.get("confirmed") or self.is_confirmed(t, now):
                t["confirmed"] = True
                t["last"] = now  # don't let it expire while we can't see the shelf
                kept.append(t)
        self.tracks = kept

    def split(self, now):
        """Return (confirmed, pending) lists of tracks."""
        conf, pend = [], []
        for t in self.tracks:
            if t.get("confirmed") or self.is_confirmed(t, now):
                t["confirmed"] = True
                conf.append(t)
            else:
                pend.append(t)
        return conf, pend
