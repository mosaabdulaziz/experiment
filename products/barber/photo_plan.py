"""Haircut photos, the engine's side (2026-10-22, spec 2026-10-22-barber-haircut-photos-design.md): when to ask for a
frame of a haircut's chair and which part of it. A start photo once a sitting opens a new haircut (a resume keeps its
haircut's, a haircut reopened from `recent` too), then an end photo every hp_end_every_s while its customer is in the
chair -- each replaces the last, so the newest is from at most that long before the customer stood up -- none while
the customer is away.
A haircut that closes unconfirmed drops its photos. The frames are fetched by haircut_photos.HaircutPhotos, never on
the engine's thread. A haircut first seen while hp_photos is 0 never gets a start photo (it would be from its middle).
2026-10-22 (user: no photo of an empty chair): a photo is asked for only while the chair camera's newest frame has a
non-staff person on the seat (the engine keeps a chair's sitter box only then; BOX_FRESH_S covers a silent camera)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from domain import EngineUpdate, PhotoRequest, dt
from zones import CHAIR, area_of

START, END = "start", "end"
BOX_FRESH_S = 5.0          # a sitter box older than this (engine time) is not taken: wait for a newer one


def crop_rect(outline: tuple, box: tuple | None, margin: float, head: float) -> tuple[float, float, float, float]:
    """The close-up: the chair outline's box joined with the sitter's box (x0, y0, x1, y1 fractions of the frame),
    widened by `margin` of its width on each side and raised by `head` of its height, clamped to the frame."""
    x0, y0, x1, y1 = outline
    if box is not None:
        x0, y0, x1, y1 = min(x0, box[0]), min(y0, box[1]), max(x1, box[2]), max(y1, box[3])
    w, h = x1 - x0, y1 - y0
    return (max(0.0, x0 - margin * w), max(0.0, y0 - head * h), min(1.0, x1 + margin * w), min(1.0, y1))


def outline_boxes(zones_by_cam: dict[str, list[dict]], camera_of: dict[str, str]) -> dict[str, tuple]:
    """chair id -> the bounding box of its chair zone on the camera that owns it."""
    out = {}
    for cam, zones in zones_by_cam.items():
        for z in zones:
            if z.get("type") != CHAIR or camera_of.get(area_of(z)) != cam:
                continue
            p = np.asarray(z.get("points") or [], dtype=float).reshape(-1, 2)
            if len(p) >= 3:
                out[area_of(z)] = (float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max()))
    return out


@dataclass
class _Seen:
    haircut: object                 # haircuts.Haircut: its confirmed flag is read when it leaves
    started: bool                   # its start photo was asked for (or it was first seen while off)
    last_at: float | None = None    # engine ts of its newest request


class PhotoPlanner:
    def __init__(self):
        self.seen: dict[str, _Seen] = {}

    def step(self, now_ts: float, open_haircuts: dict, recent: dict, camera_of: dict[str, str],
             outline_of: dict[str, tuple], sitter_box: dict[str, tuple[float, tuple]], policy: dict,
             upd: EngineUpdate, pending=()) -> None:
        """pending: the chairs whose re-ID check is running (HaircutBook.pending)."""
        on = int(policy["hp_photos"]) == 1
        live = {h.id for h in open_haircuts.values()}
        reopenable = {h.id for h in recent.values()}
        for hid in [k for k in self.seen if k not in live]:
            s = self.seen[hid]
            if not s.haircut.confirmed:
                upd.dropped_photos.add(hid)
                del self.seen[hid]
            elif hid not in reopenable:
                del self.seen[hid]
        for chair, h in open_haircuts.items():
            s = self.seen.get(h.id)
            if s is None:
                s = self.seen[h.id] = _Seen(h, started=not on)
            s.haircut = h
            # off, away / ended, or re-ID deciding who took the seat (a reseat split must not give the old customer
            # the new sitter's end photo; final review): no photo
            if not on or h.last_end is not None or chair in pending:
                continue
            cam, outline, sb = camera_of.get(chair), outline_of.get(chair), sitter_box.get(chair)
            if cam is None or outline is None or sb is None or now_ts - sb[0] > BOX_FRESH_S:
                continue
            if s.started and s.last_at is None:          # first seen while off: the interval starts now
                s.last_at = now_ts
                continue
            if s.started and now_ts - s.last_at < policy["hp_end_every_s"]:
                continue
            kind = END if s.started else START
            crop = crop_rect(outline, sb[1], policy["hp_crop_margin"], policy["hp_crop_head"])
            upd.photos.append(PhotoRequest(h.id, chair, cam, kind, dt(now_ts), crop))
            s.started, s.last_at = True, now_ts
