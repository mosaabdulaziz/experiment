"""The live preview's boxes, steadied -- display only (Ticket 5): the rule, the alerts and the published detections keep
the detector's boxes.

On warehouse camera C26 a box jumped by up to 7 % of its height (p90) on each frame the detector ran on (one in three),
against 1 % on the frames in between where the tracker predicts it, and half of the times a person's box vanished it
was back within 0.5 s. Each box here follows its detections as an exponential average with time constant `tau_s`
(at 25 fps and 0.08 s the jumps halve; a moving box lags about speed x tau_s), and a box whose object vanished stays
where it was for up to `hold_s`."""
from __future__ import annotations

import math

REWIND_S = 1.0          # the clock went back by more than this: a looping file started over, every box starts afresh


class SteadyBoxes:
    def __init__(self, tau_s: float, hold_s: float) -> None:
        self.tau_s, self.hold_s = float(tau_s), float(hold_s)
        self._boxes: dict = {}      # key -> (time last seen, steady box (left, top, width, height))
        self._t = None

    def update(self, t: float, boxes: dict) -> tuple:
        """This frame's boxes, key -> (left, top, width, height): (key -> steady box for each of them, key -> held box
        for each one that vanished less than hold_s ago)."""
        if self._t is not None and t < self._t - REWIND_S:
            self._boxes.clear()
        self._t = t
        steady = {}
        for key, box in boxes.items():
            seen = self._boxes.get(key)
            if seen is None or self.tau_s <= 0 or t <= seen[0]:
                new = tuple(float(v) for v in box) if seen is None or t > seen[0] else seen[1]
            else:
                a = 1.0 - math.exp(-(t - seen[0]) / self.tau_s)
                new = tuple(s + a * (float(b) - s) for s, b in zip(seen[1], box))
            self._boxes[key] = (t, new)
            steady[key] = new
        held = {}
        for key, (seen_t, box) in list(self._boxes.items()):
            if key in boxes:
                continue
            if t - seen_t <= self.hold_s:
                held[key] = box
            else:
                del self._boxes[key]
        return steady, held
