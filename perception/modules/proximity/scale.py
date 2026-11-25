"""The person-scale map: how tall a standing person is, in mux pixels, at each foot row of one camera.

h(v) = a * v + b, learned online from the camera's own clear, standing, untruncated people. Gaps and the stationary
test are measured in these person heights, so one threshold works across views (RESEARCH 4.2-4.3). The fit is robust
to people partly hidden (they are short, never tall): four times, the 20 % of samples furthest below the line are
dropped. The camera gives each person at most one sample a second (camera.py), so someone who lingers counts like a
passer-by; still, a few people on a narrow band of rows could supply the first samples: the map is ready only once
`min_samples` came from at least MIN_PEOPLE different people. Until then the rule uses each person's own box height.
The map remembers the latest `max_samples`: about half an hour on the busiest in-scope camera, two hours on a quiet
one. `drift` compares the most recent RECENT samples with the whole map at every refit: a camera that was moved or
zoomed shows there long before the map has re-learned it.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

MIN_HEIGHT_PX = 5.0
EDGE_MARGIN_PX = 5.0
MAX_SLOPE = 0.95            # h = (a v_head + b) / (1 - a) needs a < 1; steeper is not a camera
MIN_PEOPLE = 5              # distinct tracks a fit needs
RECENT = 200                # samples the drift check fits on their own


def _robust_fit(vs: np.ndarray, hs: np.ndarray) -> tuple:
    """h = a v + b, four times dropping the 20 % of samples furthest below the line (partly hidden people)."""
    keep = np.ones(len(vs), bool)
    for _ in range(4):
        a, b = np.polyfit(vs[keep], hs[keep], 1)
        res = hs - (a * vs + b)
        keep = res >= np.percentile(res, 20)
    return a, b


def is_clear_standing(conf: float, box, frame_w: float, frame_h: float) -> bool:
    """A sample worth learning from: confident, upright (taller than 2.2 widths) and clear of every frame edge."""
    left, top, w, h = box
    return (conf >= 0.6 and h > 2.2 * w and left > EDGE_MARGIN_PX and top > EDGE_MARGIN_PX
            and left + w < frame_w - EDGE_MARGIN_PX and top + h < frame_h - EDGE_MARGIN_PX)


class PersonScale:
    def __init__(self, min_samples: int = 50, refit_every: int = 100, max_samples: int = 5000) -> None:
        self.min_samples, self.refit_every = min_samples, refit_every
        self._v: deque = deque(maxlen=max_samples)
        self._h: deque = deque(maxlen=max_samples)
        self._who: deque = deque(maxlen=max_samples)
        self.a, self.b, self.n = 0.0, 0.0, 0
        self.drift = 0.0            # recent people's height / the map's, at their median row, at the last refit
        self._since_fit = 0
        self._hold = 0              # a restored map stands until this many new samples are in

    @property
    def ready(self) -> bool:
        return self.n >= self.min_samples

    def add(self, foot_v: float, height: float, person_id) -> bool:
        """Learn from one clear standing person (`person_id`: their track id); True when this sample triggered a
        refit."""
        self._v.append(float(foot_v))
        self._h.append(float(height))
        self._who.append(person_id)
        self._since_fit += 1
        if len(self._v) >= self.min_samples and (not self.ready or self._since_fit >= self.refit_every):
            return self.fit()
        return False

    def fit(self) -> bool:
        vs, hs = np.asarray(self._v), np.asarray(self._h)
        if len(vs) < self.min_samples or len(set(self._who)) < MIN_PEOPLE or np.ptp(vs) < 1.0:
            return False
        a, b = _robust_fit(vs, hs)
        if not (math.isfinite(a) and math.isfinite(b) and a < MAX_SLOPE):
            return False
        if self.ready and len(vs) < self._hold:
            # a restored map (a seed, an earlier run) stands until as many new samples are in as it was fitted on;
            # meanwhile the new people only say how far they are from it
            row = float(np.median(vs))
            self.drift = max(MIN_HEIGHT_PX, a * row + b) / self.height(row) - 1.0
            self._since_fit = 0
            return True
        self.a, self.b, self.n, self._since_fit, self._hold = float(a), float(b), len(vs), 0, 0
        self.drift = 0.0
        recent_v, recent_h = vs[-RECENT:], hs[-RECENT:]
        if len(vs) > RECENT and np.ptp(recent_v) >= 1.0:
            ra, rb = _robust_fit(recent_v, recent_h)
            row = float(np.median(recent_v))
            self.drift = max(MIN_HEIGHT_PX, ra * row + rb) / self.height(row) - 1.0
        return True

    def height(self, foot_v: float) -> float:
        """Expected height (mux px) of a person standing with their feet on row `foot_v`."""
        return max(MIN_HEIGHT_PX, self.a * foot_v + self.b)

    def standing_height_from_head(self, head_v: float) -> float:
        """The full height of a person whose head is on row `head_v`: v_foot = v_head + h and h = a v_foot + b, so
        h = (a v_head + b) / (1 - a)."""
        return max(MIN_HEIGHT_PX, (self.a * head_v + self.b) / (1.0 - self.a))

    def to_dict(self) -> dict:
        return {"a": self.a, "b": self.b, "n": self.n}

    def restore(self, d: dict) -> None:
        """Take a fit saved by an earlier run (to_dict). New samples refit it after `refit_every` more."""
        a, b, n = float(d["a"]), float(d["b"]), int(d["n"])
        if not (math.isfinite(a) and math.isfinite(b) and a < MAX_SLOPE):
            raise ValueError(f"not a person-scale fit: a={a} b={b}")
        self.a, self.b, self.n, self._since_fit = a, b, n, 0
        self._hold = min(n, self._v.maxlen)
