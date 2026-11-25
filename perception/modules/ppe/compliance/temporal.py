# Vendored from ppe-compliance (/home/iris2/IRIS/ppe-compliance), commit b07c335, ppe_compliance/temporal.py.
# Rule changes must be made in BOTH places, here and in ppe-compliance (whose evaluation the numbers in
# docs/EVALUATION.md come from); see perception/modules/ppe/README.md.
# IRIS change, not in ppe-compliance yet: an open episode closes only after
# cfg.close_after_present_s of unbroken smoothed PRESENT (_TrackState.present_since). With 0 it is unchanged.
"""Temporal smoothing of per-frame results and event de-duplication.

A single noisy frame should not create an event. For each person track we
keep a sliding window of per-item statuses and only treat an item as ABSENT
once enough frames in the window agree. An event is emitted when an item
first becomes ABSENT for a track; that "episode" stays open (no repeat events)
until the item is observed PRESENT again and the cooldown has elapsed.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .config import TemporalConfig
from .types import ComplianceStatus, ItemStatus, PersonResult


@dataclass
class _TrackState:
    history: dict[str, deque] = field(default_factory=dict)
    open_episodes: set[str] = field(default_factory=set)
    last_event_time: dict[str, float] = field(default_factory=dict)
    last_seen_s: float = 0.0
    # item -> time its current unbroken run of smoothed PRESENT started
    present_since: dict[str, float] = field(default_factory=dict)


@dataclass
class SmoothedPerson:
    result: PersonResult
    item_status: dict[str, ItemStatus]
    status: ComplianceStatus
    # Items that just started a new violation episode on this frame.
    new_violations: list[str]
    window_stats: dict[str, dict]


class TemporalFilter:
    def __init__(self, cfg: TemporalConfig, required_items: list[str]):
        self.cfg = cfg
        self.required = required_items
        self._tracks: dict[int, _TrackState] = {}

    def _smooth(self, hist: deque) -> ItemStatus:
        absent = sum(1 for s, _ in hist if s is ItemStatus.ABSENT)
        present = sum(1 for s, _ in hist if s is ItemStatus.PRESENT)
        if absent >= self.cfg.min_absent_frames and absent > present:
            return ItemStatus.ABSENT
        if present >= self.cfg.min_present_frames and present > absent:
            return ItemStatus.PRESENT
        return ItemStatus.UNKNOWN

    def update(self, persons: list[PersonResult], timestamp_s: float) -> list[SmoothedPerson]:
        out = []
        for person in persons:
            tid = person.track_id
            if tid is None:
                raise ValueError("TemporalFilter requires tracked persons (track_id set)")
            state = self._tracks.setdefault(tid, _TrackState())
            state.last_seen_s = timestamp_s

            item_status, stats, new = {}, {}, []
            for name in self.required:
                hist = state.history.setdefault(name, deque(maxlen=self.cfg.window))
                r = person.items[name]
                hist.append((r.status, r))
                smoothed = self._smooth(hist)
                item_status[name] = smoothed
                stats[name] = {
                    "window_frames": len(hist),
                    "absent_frames": sum(1 for s, _ in hist if s is ItemStatus.ABSENT),
                    "present_frames": sum(1 for s, _ in hist if s is ItemStatus.PRESENT),
                    "unknown_frames": sum(1 for s, _ in hist if s is ItemStatus.UNKNOWN),
                    "max_negative_conf": round(max(x.negative_conf for _, x in hist), 4),
                    "max_positive_conf": round(max(x.positive_conf for _, x in hist), 4),
                }
                if smoothed is ItemStatus.PRESENT:
                    since = state.present_since.setdefault(name, timestamp_s)
                    if timestamp_s - since >= self.cfg.close_after_present_s:
                        state.open_episodes.discard(name)
                    continue
                state.present_since.pop(name, None)
                if smoothed is ItemStatus.ABSENT and name not in state.open_episodes:
                    last = state.last_event_time.get(name)
                    if last is None or timestamp_s - last >= self.cfg.cooldown_s:
                        state.open_episodes.add(name)
                        state.last_event_time[name] = timestamp_s
                        new.append(name)

            if any(s is ItemStatus.ABSENT for s in item_status.values()):
                status = ComplianceStatus.NON_COMPLIANT
            elif all(s is ItemStatus.PRESENT for s in item_status.values()):
                status = ComplianceStatus.COMPLIANT
            else:
                status = ComplianceStatus.UNKNOWN
            out.append(SmoothedPerson(person, item_status, status, new, stats))

        stale = [t for t, s in self._tracks.items() if timestamp_s - s.last_seen_s > self.cfg.forget_after_s]
        for t in stale:
            del self._tracks[t]
        return out
