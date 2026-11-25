"""Chair-first seat rule, one frame of one side camera (project 3, spec 2a).

Each person's mask outline is rasterised; its bottom ce_lower_frac (hips, not head) is measured on
every seat of this camera, and the person goes to the one chair with the highest share. That chair
is occupied when the share >= ce_seat_min, the detection conf >= ce_conf_min and the person's latest
staff_p < staff_threshold (unknown = not staff). A staff person never occupies; one whose whole mask
covers >= ce_carry_outline of a chair outline or >= ce_carry_seat of its seat is that chair's
staff cover (barber carry). With ce_carry_lower_frac < 1 (2026-10-15, user, Palm Hills hand-overs) only the bottom
ce_carry_lower_frac of a staff mask is measured on the seat and the outline test is off: a barber standing at the
empty chair has his torso over the seat in the image, not his hips. With ce_seat_inside_min > 0, the person also needs at least that share of the
WHOLE mask inside the chosen chair's outline (2026-10-05: staff standing by a seat's aisle corner have their
lower mask on the seat but most of the body outside the chair). Seats live in the owner (side) camera's chairs; a seat
drawn in a non-owner pair member (2026-10-15, Palm Hills L7 moved to its better view) claims the people whose lower mask is
mostly on it in that camera -- they can't occupy a neighbouring chair there -- and signals nothing (other_view).
Per-chair overrides (policy "<key>@<chair>", 2026-10-05): ce_seat_min, ce_conf_min and ce_seat_inside_min are read per
chair. The person still goes to the chair with the highest share; that chair's own thresholds only decide whether
it counts. A track below every chair's conf floor (of this camera) is low_conf; a staff track covers only chairs whose floor it meets."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Iterable

from domain import TrackObs
from masks import Raster, raster
from policy import policy_for
from zones import CHAIR, SEAT, area_of, owner_cameras

logger = logging.getLogger("barber.seat")


@dataclass(frozen=True)
class ChairGeom:
    chair: str
    camera: str
    seat: Raster
    outline: Raster
    owner: bool = True     # False: another camera owns this pair -- the seat claims people here, signals nothing
    kind: str = "haircut"  # the chair zone's kind (zones.PEDICURE: a pedicure chair, 2026-11-10)


@dataclass(frozen=True)
class ChairSignal:
    occupied: bool
    cover: bool
    track_id: int | None


def chair_geoms(zones_by_cam: dict[str, list[dict]], log=logger.warning) -> dict[str, list[ChairGeom]]:
    out: dict[str, list[ChairGeom]] = {}
    owners = owner_cameras(zones_by_cam)
    for cam, zones in zones_by_cam.items():
        chairs = {z["id"]: z for z in zones if z.get("type") == CHAIR}
        seats = {z.get("seat_of"): z for z in zones if z.get("type") == SEAT}
        for cid, cz in chairs.items():
            sz = seats.get(cid)
            if sz is None:
                if not cz.get("pair") or cz.get("owner") or any(z.get("type") == SEAT for z in zones):
                    log(f"chair {area_of(cz)} in {cam} has no seat: skipped")
                continue
            s, o = raster(sz["points"]), raster(cz["points"])
            if s is None or o is None or s.area == 0 or o.area == 0:
                log(f"chair {area_of(cz)} in {cam}: seat or outline has no area: skipped")
                continue
            owner = owners.get(area_of(cz), cam) == cam
            out.setdefault(cam, []).append(ChairGeom(area_of(cz), cam, s, o, owner, cz.get("kind") or "haircut"))
    return out


@dataclass(frozen=True)
class Verdict:
    """Why one track did what it did this frame (shown on the dashboard overlay)."""
    chair: str | None
    share: float | None
    why: str     # occupied | staff_cover | staff | low_conf | no_mask | below_min | outside_chair | other_chair
    #              | no_chair | other_view (on a chair another camera owns)


def evaluate(tracks: Iterable[TrackObs], geoms: list[ChairGeom], staff_of: Callable[[int], float | None],
             policy: dict) -> dict[str, ChairSignal]:
    return evaluate_detail(tracks, geoms, staff_of, policy)[0]


def evaluate_detail(tracks: Iterable[TrackObs], geoms: list[ChairGeom], staff_of: Callable[[int], float | None],
                    policy: dict) -> tuple[dict[str, ChairSignal], dict[int, Verdict]]:
    best: dict[str, tuple[float, int]] = {}
    cover: set[str] = set()
    verdicts: dict[int, Verdict] = {}
    pc = {g.chair: policy_for(policy, g.chair) for g in geoms}       # each chair's knobs
    conf_floor = min((p["ce_conf_min"] for p in pc.values()), default=policy["ce_conf_min"])
    for t in tracks:
        if t.outline is None:
            verdicts[t.track_id] = Verdict(None, None, "no_mask")
            continue
        if t.conf is None or t.conf < conf_floor:
            verdicts[t.track_id] = Verdict(None, None, "low_conf")
            continue
        m = raster(t.outline)
        if m is None or m.area == 0:
            verdicts[t.track_id] = Verdict(None, None, "no_mask")
            continue
        sp = staff_of(t.track_id)
        if sp is not None and sp >= policy["staff_threshold"]:
            covered = None
            lowers: dict[float, object] = {}          # the mask's bottom part per carry_low (chairs may differ)
            for g in geoms:
                pg = pc[g.chair]                       # the cover knobs per chair too (2026-10-18)
                if not g.owner or t.conf < pg["ce_conf_min"]:
                    continue
                carry_low = pg["ce_carry_lower_frac"]
                if carry_low < 1.0 and carry_low not in lowers:
                    lowers[carry_low] = m.lower(carry_low)
                ml = lowers[carry_low] if carry_low < 1.0 else m
                if ((carry_low >= 1.0 and m.overlap(g.outline) / g.outline.area >= pg["ce_carry_outline"])
                        or ml.overlap(g.seat) / g.seat.area >= pg["ce_carry_seat"]):
                    cover.add(g.chair)
                    covered = covered or g.chair
            verdicts[t.track_id] = Verdict(covered, None, "staff_cover" if covered else "staff")
            continue
        low = m.lower(policy["ce_lower_frac"])
        shares = [(low.overlap(g.seat) / g.seat.area, g.chair, i) for i, g in enumerate(geoms)]
        if not shares:
            verdicts[t.track_id] = Verdict(None, None, "no_chair")
            continue
        v, chair, gi = max(shares)
        cp = pc[chair]
        inside_min = cp["ce_seat_inside_min"]
        if v <= 0:
            verdicts[t.track_id] = Verdict(None, None, "no_chair")
        elif t.conf < cp["ce_conf_min"]:
            verdicts[t.track_id] = Verdict(chair, round(v, 3), "low_conf")
        elif v < cp["ce_seat_min"]:
            verdicts[t.track_id] = Verdict(chair, round(v, 3), "below_min")
        elif inside_min > 0 and m.overlap(geoms[gi].outline) / m.area < inside_min:
            verdicts[t.track_id] = Verdict(chair, round(v, 3), "outside_chair")
        elif not geoms[gi].owner:
            verdicts[t.track_id] = Verdict(chair, round(v, 3), "other_view")
        elif v > best.get(chair, (0.0, -1))[0]:
            if chair in best:
                old = best[chair][1]
                verdicts[old] = Verdict(chair, verdicts[old].share, "other_chair")
            best[chair] = (v, t.track_id)
            verdicts[t.track_id] = Verdict(chair, round(v, 3), "occupied")
        else:
            verdicts[t.track_id] = Verdict(chair, round(v, 3), "other_chair")
    signals = {g.chair: ChairSignal(g.chair in best, g.chair in cover, best[g.chair][1] if g.chair in best else None)
               for g in geoms if g.owner}
    return signals, verdicts
