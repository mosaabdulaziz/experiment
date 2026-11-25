"""The customer register (project C): recognise a returning customer by face
when his visit ends.

Matching: drop a visit's stray faces (unlike the visit's medoid face), score
every known customer by the MEDIAN over the visit's faces of each face's best
similarity to that customer's stored faces, and link only when the top score
clears match_threshold and leads the runner-up by match_margin. Anything
else creates a new customer. Two different people are almost never merged;
a returning customer is sometimes counted twice (the user's choice)."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

import crops
from storage import load_body_gallery, load_face_gallery, register_visit, visit_customer

logger = logging.getLogger("barber.register")


@dataclass(frozen=True)
class Decision:
    customer_id: str | None          # None = a new customer
    score: float | None
    runner_up: float | None
    clean: tuple[int, ...]           # indices of the visit's faces kept
    usable: bool                     # False = too few clean faces (a face_count-0 customer)


def clean_faces(embs: np.ndarray, min_sim: float) -> np.ndarray:
    n = len(embs)
    if n <= 1:
        return np.arange(n)
    sims = embs @ embs.T
    mean_other = (sims.sum(axis=1) - 1.0) / (n - 1)
    medoid = int(np.argmax(mean_other))
    return np.flatnonzero(sims[medoid] >= min_sim)


def clean_bodies(bodies, min_sim: float) -> tuple[int, ...]:
    """Indices of the body crops to keep: per camera (body re-ID doesn't compare across cameras), those
    at least min_sim like that camera's medoid body; a camera's lone body is kept; one without an
    embedding can't be vouched for and is dropped (crop_check, 2026-09-29)."""
    keep = []
    by_cam: dict[str, list[int]] = {}
    for i, b in enumerate(bodies):
        if b.embedding is not None:
            by_cam.setdefault(b.camera, []).append(i)
    for idx in by_cam.values():
        embs = np.asarray([bodies[i].embedding for i in idx], np.float32)
        keep += [idx[j] for j in clean_faces(embs, min_sim)]
    return tuple(sorted(keep))


def _like(embs: np.ndarray, gallery: np.ndarray | None, min_sim: float) -> np.ndarray:
    """Which of embs are at least min_sim like their best match in gallery (all, with no gallery)."""
    if gallery is None or not len(gallery) or not len(embs):
        return np.ones(len(embs), bool)
    return (embs @ gallery.T).max(axis=1) >= min_sim


def checked(reg, decision: Decision, cid: str | None, topup: bool, conn, policy: dict):
    """crop_check 1: (the Decision with only the faces kept, the bodies kept). Faces joining a known
    customer must look like his stored faces (a top-up is checked against them alone: its few faces
    make a weak medoid); bodies pass clean_bodies, then must look like his stored bodies of the same
    camera. 0: today's (every clean face, every body)."""
    if int(policy.get("crop_check", 0)) < 1:
        return decision, list(reg.bodies)
    embs = (np.asarray([f.embedding for f in reg.faces], np.float32) if reg.faces else np.zeros((0, 512), np.float32))
    idx = list(range(len(reg.faces))) if topup else list(decision.clean)
    if cid is not None and idx:
        ok = _like(embs[idx], load_face_gallery(conn).get(cid), policy["customer_face_min_sim"])
        idx = [i for i, k in zip(idx, ok) if k]
    decision = replace(decision, clean=tuple(idx), usable=decision.usable or topup)
    bodies = [reg.bodies[i] for i in clean_bodies(reg.bodies, policy["body_min_sim"])]
    if cid is not None and bodies:
        stored = load_body_gallery(conn, cid)
        bodies = [b for b in bodies
                  if _like(np.asarray([b.embedding], np.float32), stored.get(b.camera), policy["body_min_sim"])[0]]
    return decision, bodies


def returnable(conn, visit_id: str, gallery: dict, policy: dict, entered_at: datetime | None = None) -> dict:
    """min_return_gap_h (2026-09-29): a customer last seen less than that many hours before this visit
    entered can't be this visit -- nobody comes back for another haircut minutes later (top-down faces
    of different men scored above match_threshold: 5 visits in 25 min went to one customer)."""
    if int(policy.get("link_returns", 1)) < 1:
        return {}                   # 2026-09-29 (user): the page is the day's customers -- one visit each
    gap = policy.get("min_return_gap_h", 0.0)
    row = conn.execute("SELECT entered_at FROM visit WHERE id = ?", (visit_id,)).fetchone()
    entered = row["entered_at"] if row is not None else None
    if entered is None:             # a door entry (face register) has no visit row: its crossing time is the entry time
        door = conn.execute("SELECT at FROM door_crossing WHERE entry_id = ?", (visit_id,)).fetchone()
        entered = door["at"] if door is not None else None
    if entered is None and entered_at is not None:     # 2026-10-07 review: no row (a lost write) -- the
        entered = entered_at.isoformat(timespec="seconds")   # registration's own entry time, never "no limit"
    if not gap or entered is None:
        return gallery
    cutoff = (datetime.fromisoformat(entered) - timedelta(hours=gap)).isoformat(timespec="seconds")
    recent = {r["id"] for r in conn.execute("SELECT id FROM customer WHERE last_seen > ?", (cutoff,))}
    return {cid: g for cid, g in gallery.items() if cid not in recent}


def score_customers(embs: np.ndarray, gallery: dict[str, np.ndarray]) -> dict[str, float]:
    return {cid: float(np.median((embs @ g.T).max(axis=1))) for cid, g in gallery.items() if len(g)}


MOMENT_S = 1.0


def _moments(ts: list[float]) -> int:
    """How many of the times are at least MOMENT_S after the previous one counted."""
    n, last = 0, None
    for t in sorted(ts):
        if last is None or t - last >= MOMENT_S:
            n, last = n + 1, t
    return n


def decide(embs: np.ndarray, gallery: dict[str, np.ndarray], policy: dict, times=None) -> Decision:
    """times: each face's ts. 2026-10-07 (review finding 4): to LINK to a known customer the clean faces must
    hold min_link_faces moments at least MOMENT_S apart (customer_face_interval_s 0.25 gave "two faces" from
    one glimpse); fewer -> a new customer: a duplicate card is the safer error than a merge of two people.
    2026-10-12: min_visit_faces (a card at all) and min_link_faces (a link) are separate knobs."""
    idx = clean_faces(embs, policy["visit_face_min_sim"]) if len(embs) else np.arange(0)
    clean = tuple(int(i) for i in idx)
    if len(idx) < int(policy["min_visit_faces"]):
        return Decision(None, None, None, clean, False)
    ranked = sorted(score_customers(embs[idx], gallery).items(), key=lambda kv: kv[1], reverse=True)
    if not ranked:
        return Decision(None, None, None, clean, True)
    top_id, top = ranked[0]
    runner = ranked[1][1] if len(ranked) > 1 else None
    moments = _moments([times[i] for i in idx]) if times is not None else len(idx)
    if (top >= policy["match_threshold"] and (runner is None or top - runner >= policy["match_margin"])
            and moments >= int(policy["min_link_faces"])):
        return Decision(top_id, top, runner, clean, True)
    return Decision(None, top, runner, clean, True)


def apply(conn, update, crops_root: Path, policy: dict, new_id) -> list[str]:
    """Register every finished visit in `update`, and delete every crop file
    the engine or the register let go. Returns the customer ids registered.

    Each registration gets its own try/except: a bad one (a new_id
    collision, a foreign-key failure because the matched customer was
    purged, ...) is logged and skipped, so it never costs the others in
    the same update."""
    ids = []
    for reg in update.registrations:
        try:
            embs = (np.asarray([f.embedding for f in reg.faces], np.float32)
                    if reg.faces else np.zeros((0, 512), np.float32))
            known = visit_customer(conn, reg.visit_id)     # register_at 1: registered at the haircut's end
            if known is not None:
                clean = tuple(int(i) for i in clean_faces(embs, policy["visit_face_min_sim"])) if len(embs) else ()
                decision = Decision(known, None, None, clean, True)
            else:
                decision = decide(embs, returnable(conn, reg.visit_id, load_face_gallery(conn), policy, reg.ended_at),
                                  policy, times=[f.ts for f in reg.faces])
            decision, bodies = checked(reg, decision, known or decision.customer_id, known is not None, conn, policy)
            cid, dropped = register_visit(conn, reg, decision, policy, new_id, bodies)
            if cid is None:
                logger.info("visit %s not registered: no usable crop", reg.visit_id)
                for rel in dropped:
                    crops.delete(crops_root, rel)
                continue
            logger.info("visit %s -> customer %s (%s): faces kept %d/%d, bodies kept %d/%d", reg.visit_id, cid,
                        "top-up" if known else ("linked" if decision.customer_id else "new"),
                        len(decision.clean) if decision.usable else 0, len(reg.faces), len(bodies), len(reg.bodies))
        except Exception:
            logger.exception("registration failed for visit %s -- skipping it", reg.visit_id)
            continue
        ids.append(cid)
        for rel in dropped:
            crops.delete(crops_root, rel)
    for rel in update.discard_crops:
        crops.delete(crops_root, rel)
    return ids
