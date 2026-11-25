"""What the plate reader got for a vehicle, from the files the sidecar writes (data/plates/<gid>.json).

Each camera keeps its own best plate of a vehicle, read separately; here they are put together:
`best_read` is the reading the vehicle is shown under for now (the surest one with enough digits).
How to mix the cameras' readings is open until there are results to look at.
"""
from __future__ import annotations

import json
import os

MIN_DIGITS = 3        # the platform's plate parser needs a 3-4 digit run
MIN_CONF = 0.4      # below this a reading is too unsure to show (the page still lists it under its crop)

_cache: dict[str, tuple[float, list]] = {}


def readings(plates_dir: str, gid: int, name: str = "") -> list[dict]:
    """Every camera's reading of this vehicle, surest first ([] when none); `name` "_reads" gives
    every reading made so far instead of each camera's latest."""
    path = os.path.join(plates_dir, f"{int(gid)}{name}.json")
    try:
        mtime = os.stat(path).st_mtime
    except OSError:
        return []
    hit = _cache.get(path)
    if hit and hit[0] == mtime:
        return hit[1]
    try:
        with open(path, encoding="utf-8") as fh:
            rows = json.load(fh)
    except (OSError, ValueError):
        return hit[1] if hit else []
    _cache[path] = (mtime, rows)
    return rows


def digits_of(text: str) -> str:
    return "".join(c for c in text if c.isdigit())


def _letters(text: str) -> str:
    return "".join(c for c in text if "\u0621" <= c <= "\u064a")


LEN3_SHARE = 0.2      # the plate has three letters when reads holding this share of the confidence show three or more
LEN2_SHARE = 0.4      # (else two letters when reads with at least two hold this share, else one)
_FILL = 0.05          # small credit per aligned letter, so a letter read wrongly still goes to its place rather than away


def _letter_list(text: str) -> list[str]:
    return [c for c in text if "\u0621" <= c <= "\u064a"]


def _align(letters: list[str], profile: list[dict[str, float]]) -> list[tuple[int, str]]:
    """Which slot of the plate each letter of one reading stands for: the letters keep their order, a letter
    may be left out (invented) and a slot may stay empty (a letter the reader dropped). The best alignment is
    the one that agrees most with what the other readings put in each slot; a letter read wrongly counts as a
    substitution (it still takes its slot), so one dropped letter does not shift all the rest."""
    from itertools import combinations
    n = len(profile)
    best, best_score = [], -1.0
    for k in range(min(len(letters), n), 0, -1):
        for li in combinations(range(len(letters)), k):
            for si in combinations(range(n), k):
                score = 0.0
                for i, j in zip(li, si):
                    tot = sum(profile[j].values())
                    score += (profile[j].get(letters[i], 0.0) / tot if tot else 0.0) + _FILL
                if score > best_score + 1e-9:
                    best, best_score = [(j, letters[i]) for i, j in zip(li, si)], score
    return best


def consensus(reads: list[dict]) -> dict | None:
    """One reading from everything read of a vehicle, over all its cameras and moments.

    A single read of a plate is unsteady (a letter dropped here, one misread there, one invented). The digits
    are the digit string with the most confidence behind it. The letters are decided in three steps:
      1. How many: three when reads holding LEN3_SHARE of the confidence show three or more (a dropped letter is
         common, an invented one rare, and truck plates mostly have three), else two, else one.
      2. Where each reading's letters stand: every reading is aligned to that many slots (see _align), longest
         and surest reading first, so a reading that lost a letter still puts the others in the right slots.
      3. Each slot takes the letter with the most confidence behind it.
    Two readings that differ in one letter ("رس", "وس") are one two-letter plate with a doubtful letter, not a
    three-letter plate holding both.
    """
    ok = [r for r in reads if float(r.get("ocr_conf", 0)) >= MIN_CONF]
    votes: dict[str, float] = {}
    for r in ok:
        d = digits_of(r.get("text", ""))
        if len(d) >= MIN_DIGITS:
            votes[d] = votes.get(d, 0.0) + float(r["ocr_conf"])
    if not votes:
        return None
    digits = max(votes, key=votes.get)
    same = [r for r in ok if digits_of(r.get("text", "")) == digits]
    total = sum(float(r["ocr_conf"]) for r in same)
    items = [(_letter_list(r["text"]), float(r["ocr_conf"])) for r in same]

    def share(n: int) -> float:
        return sum(c for ls, c in items if len(ls) >= n) / total

    n_letters = 3 if share(3) >= LEN3_SHARE else 2 if share(2) >= LEN2_SHARE else 1 if share(1) >= LEN2_SHARE else 0
    letters = ""
    if n_letters:
        order = sorted((it for it in items if it[0]), key=lambda it: (-min(len(it[0]), n_letters), -it[1]))
        profile: list[dict[str, float]] = [{} for _ in range(n_letters)]
        for ls, conf in order:                    # build the profile reading by reading
            for slot, ch in _align(ls, profile):
                profile[slot][ch] = profile[slot].get(ch, 0.0) + conf
        final: list[dict[str, float]] = [{} for _ in range(n_letters)]
        for ls, conf in order:                    # then place everyone against the whole profile
            for slot, ch in _align(ls, profile):
                final[slot][ch] = final[slot].get(ch, 0.0) + conf
        profile = final
        letters = "".join(max(slot, key=slot.get) for slot in profile if slot)
    return {"text": digits + letters, "ocr_conf": max(float(r["ocr_conf"]) for r in same), "reads": len(same)}


def best_read(rows: list[dict]) -> dict | None:
    """The surest reading that has a usable number, or None."""
    ok = [r for r in rows if len(digits_of(r.get("text", ""))) >= MIN_DIGITS
          and float(r.get("ocr_conf", 0)) >= MIN_CONF]
    return max(ok, key=lambda r: float(r["ocr_conf"])) if ok else None


def logical(text: str) -> str:
    """The reading in logical order, as a plate is spoken: the reader gives characters as they stand
    on the plate, left to right (digits first, letters after), but Arabic letters read right to
    left, so they are reversed: '5356روس' -> 'س و ر 5356'."""
    digits = digits_of(text)
    letters = [c for c in text if "\u0621" <= c <= "\u064a"]
    return (" ".join(reversed(letters)) + " " + digits).strip()
