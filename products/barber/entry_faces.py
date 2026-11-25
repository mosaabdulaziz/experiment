"""Faces for the face register at the door (2026-10-05; spec docs/superpowers/specs/2026-10-05-barber-face-register-
design.md). The sidecar detects faces only inside the centre camera's door area; this keeps those face candidates per
track for fr_keep_s, and when a track makes a non-staff `in` crossing (entry), it hands the track's faces from
fr_before_s before to fr_after_s after the crossing to the register as one Registration, fr_after_s after the
crossing. Every face that ends in no registration is returned for deletion (drain_discards).
2026-10-07 (review finding 4: 6 of 18 entries without faces had them on a NEW track id 0.6-20 s after the crossing):
an entry follows its entrant onto a successor track -- one first seen within SUCC_S after the entry's last track
was last seen, its first foot point within SUCC_DIST of that track's last one, not staff -- until it is due."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from domain import FaceCandidate, Frame, Registration, dt

SUCC_S = 3.0            # a successor is first seen at most this long after the entry's last track was last seen
SUCC_GONE_S = 0.5       # ... which has been unseen at least this long (not two people side by side)
SUCC_DIST = 0.10        # ... its first foot point at most this far (normalised frame units) from that track's last one
TRACK_TTL_S = 60.0


@dataclass
class _Entry:
    id: str
    track_id: int
    at: float
    tracks: list[int] = field(default_factory=list)     # the entrant's tracks, the crossing one first


@dataclass
class _Track:
    first: float
    first_pos: tuple[float, float]
    last: float
    pos: tuple[float, float]


class EntryFaces:
    def __init__(self, camera: str | None):
        self.camera = camera
        self.clear()

    def clear(self) -> list[str]:
        paths = [f.crop_path for f in getattr(self, "_faces", [])] + getattr(self, "_discards", [])
        self._faces: list[FaceCandidate] = []
        self._entries: list[_Entry] = []
        self._discards: list[str] = []
        self._tracks: dict[int, _Track] = {}
        return paths

    def set_camera(self, camera: str | None) -> None:
        """The door camera moved: faces kept for the old one go to the discards."""
        if camera == self.camera:
            return
        self._discards += [f.crop_path for f in self._faces]
        self._faces = []
        self.camera = camera

    def add(self, face: FaceCandidate) -> None:
        if face.camera != self.camera:
            self._discards.append(face.crop_path)
        else:
            self._faces.append(face)

    def entry(self, entry_id: str, track_id: int, at: float) -> None:
        self._entries.append(_Entry(entry_id, track_id, at, [track_id]))

    def observe(self, frame: Frame, foot, is_staff) -> None:
        """A door-camera frame: keep each track's first / last foot point, and hand a pending entry whose last
        track is gone to its successor. foot(box) -> normalised (x, y); is_staff(track_id) -> bool."""
        if frame.camera != self.camera:
            return
        ts = frame.ts
        for t in frame.tracks:
            p = foot(t.box)
            s = self._tracks.get(t.track_id)
            if s is None:
                self._tracks[t.track_id] = _Track(ts, p, ts, p)
            else:
                s.last, s.pos = ts, p
        taken = {tid for e in self._entries for tid in e.tracks}
        for e in self._entries:
            last = self._tracks.get(e.tracks[-1])
            if last is None or ts - last.last < SUCC_GONE_S:
                continue
            best = None
            for t in frame.tracks:
                n = self._tracks[t.track_id]
                if t.track_id in taken or is_staff(t.track_id) or not (last.last - SUCC_GONE_S <= n.first <= last.last + SUCC_S):
                    continue
                d = math.dist(n.first_pos, last.pos)
                if d <= SUCC_DIST and (best is None or d < best[0]):
                    best = (d, t.track_id)
            if best is not None:
                e.tracks.append(best[1])
                taken.add(best[1])
        for tid in [k for k, s in self._tracks.items() if ts - s.last > TRACK_TTL_S]:
            del self._tracks[tid]

    def due(self, now: float, policy: dict) -> list[Registration]:
        """Registrations of the entries whose settle time has passed; stale faces go to the discards."""
        before, after = policy["fr_before_s"], policy["fr_after_s"]
        regs, waiting = [], []
        for e in self._entries:
            if now < e.at + after:
                waiting.append(e)
                continue
            lo, hi = e.at - before, e.at + after
            ids = set(e.tracks)
            mine = [f for f in self._faces if f.camera == self.camera and f.track_id in ids and lo <= f.ts <= hi]
            self._discards += [f.crop_path for f in self._faces if f.track_id in ids and f.ts < lo]
            # the tracks' faces after the window stay: a later entry by the same track may use them
            self._faces = [f for f in self._faces if f.track_id not in ids or f.ts > hi]
            regs.append(Registration(e.id, dt(e.at), tuple(mine), ()))
        self._entries = waiting
        pending = {tid for e in self._entries for tid in e.tracks}
        stale = [f for f in self._faces if f.ts < now - policy["fr_keep_s"] and f.track_id not in pending]
        self._discards += [f.crop_path for f in stale]
        gone = {id(f) for f in stale}
        self._faces = [f for f in self._faces if id(f) not in gone]
        return regs

    def drain_discards(self) -> list[str]:
        out, self._discards = self._discards, []
        return out
