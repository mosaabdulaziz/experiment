"""Per-visit pools of customer face and body candidates (project C): only
the best (by quality) are kept while a visit runs. Everything else's crop
file is reported back so the caller can delete it."""
from __future__ import annotations

from domain import BodyCandidate, FaceCandidate


def _keep_best(items: list, new, cap: int) -> list[str]:
    items.append(new)
    items.sort(key=lambda c: c.quality, reverse=True)
    dropped = items[cap:]
    del items[cap:]
    return [c.crop_path for c in dropped]


class VisitPool:
    """`sent`: crop paths already handed to a registration (register_at 1) -- a customer's rows own
    those files now, so the pool never reports them for deletion."""

    def __init__(self, max_faces: int, max_bodies: int):
        self.max_faces, self.max_bodies = max_faces, max_bodies
        self.faces: list[FaceCandidate] = []
        self.bodies: list[BodyCandidate] = []
        self.sent: set[str] = set()

    def _free(self, paths: list[str]) -> list[str]:
        return [p for p in paths if p not in self.sent]

    def add_face(self, c: FaceCandidate) -> list[str]:
        return self._free(_keep_best(self.faces, c, self.max_faces))

    def add_body(self, c: BodyCandidate) -> list[str]:
        """The best bodies by quality -- and always the earliest one (2026-09-29, user: the customer
        card shows his first body crop)."""
        items = self.bodies + [c]
        first = min(items, key=lambda b: b.ts)
        rest = sorted((b for b in items if b is not first), key=lambda b: b.quality, reverse=True)
        kept = [first] + rest[:max(0, self.max_bodies - 1)]
        dropped = [b.crop_path for b in items if b not in kept]
        self.bodies = sorted(kept, key=lambda b: b.quality, reverse=True)
        return self._free(dropped)

    def take_unsent(self) -> tuple[tuple[FaceCandidate, ...], tuple[BodyCandidate, ...]]:
        """The candidates not handed to a registration yet; they count as sent from now on."""
        faces = tuple(c for c in self.faces if c.crop_path not in self.sent)
        bodies = tuple(c for c in self.bodies if c.crop_path not in self.sent)
        self.sent |= {c.crop_path for c in faces + bodies}
        return faces, bodies

    def drop_track(self, camera: str, track_id: int, since: float | None = None) -> list[str]:
        """Forget every candidate that one track gave (it turned out to be
        staff); return their crop paths. since: that track's first frame --
        an earlier body with the same id (the tracker's ids restart with the
        sidecar) keeps its candidates."""
        def of_it(c) -> bool:
            return (c.camera, c.track_id) == (camera, track_id) and (since is None or c.ts >= since)
        gone = [c for c in self.faces + self.bodies if of_it(c)]
        self.faces = [c for c in self.faces if not of_it(c)]
        self.bodies = [c for c in self.bodies if not of_it(c)]
        return self._free([c.crop_path for c in gone])

    def paths(self) -> list[str]:
        return self._free([c.crop_path for c in self.faces] + [c.crop_path for c in self.bodies])
