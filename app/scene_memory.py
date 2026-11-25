"""Per-stream memory of what is *going on*, as interpreted by the model.

`app/track_observer.py` counts what the app can measure; this module holds what only
the model can name - a truck being unloaded, an aisle filling up, a wet floor - and
lends it the clock the model does not have. Each analysis response carries a small
delta; merging it stamps `first_seen` once and `last_confirmed` on every mention, so
the block rendered into the next prompt reports an accumulating duration nobody had
to estimate. That accumulation is the whole point: it is what makes "a blocked route
that remains unresolved" expressible at all.

Self-healing by construction. The model never reads back its own prose, only this
module's summary of state it has repeatedly confirmed, so anything it stops
confirming ages out and a hallucination decays instead of compounding.

Expiry is counted in *cycles*, not seconds: one `merge()` is one cycle. The analysis
cadence varies with load (10-30s on the current rig), so a seconds-based TTL would
silently mean a different number of missed confirmations at different loads.

Bounds are the app's job, not the model's discipline - this block shares a context
window with the images, so it needs a ceiling that no amount of model verbosity can
lift.

Pure state and text - no Redis, no config, no IO - the same shape as
`app/track_observer.py` and `app/severity.py`.
"""
from dataclasses import dataclass

from .track_observer import duration_seconds, format_duration

_HEADING = "## SCENE MEMORY (durations measured by the system, not estimated)"
_DURATION_WIDTH = 9  # "[11m03s] " - the longest stamp plus its separating space


@dataclass
class Situation:
    key: str            # model-chosen slug
    description: str
    first_seen: float
    last_confirmed: float
    unconfirmed_cycles: int = 0


@dataclass
class Area:
    zone: str           # must match a configured zone name
    condition: str
    first_seen: float
    last_confirmed: float
    unconfirmed_cycles: int = 0


def validate_delta(raw) -> dict:
    """Coerce anything into {entities, areas, situations, scene_notes}. Never raises.

    Every slot is validated on its own, because a memory failure must never cost an
    event (spec 5.8): a `situations` value the model mangled must not discard the
    `scene_notes` it got right in the same response. `scene_notes` is `None` when
    absent or unusable, which `merge` reads as "leave what we already have".
    """
    delta = {"entities": [], "areas": [], "situations": [], "scene_notes": None}
    if not isinstance(raw, dict):
        return delta
    for slot, clean in (("entities", _clean_entities),
                        ("areas", _clean_areas),
                        ("situations", _clean_situations)):
        try:
            delta[slot] = clean(raw.get(slot))
        except Exception:
            pass  # this slot contributes nothing; the others are untouched
    try:
        notes = raw.get("scene_notes")
        delta["scene_notes"] = _text(notes) if isinstance(notes, str) else None
    except Exception:
        pass
    return delta


class SceneMemory:
    """One per stream. Folds the model's deltas into bounded state; renders the block."""

    def __init__(self, *, max_situations: int = 8, ttl_cycles: int = 3,
                 notes_max_chars: int = 400):
        self._max_situations = int(max_situations)
        self._ttl_cycles = int(ttl_cycles)
        self._notes_max_chars = int(notes_max_chars)
        self._situations: dict[str, Situation] = {}
        self._areas: dict[str, Area] = {}
        # track id -> note. No clock of its own: an entity belongs to the measured half
        # and dies when its track does, which is what `live_track_ids` decides below.
        self._entities: dict = {}
        self._notes = ""

    def merge(self, delta: dict, now: float, *,
              live_track_ids: set | None = None,
              known_zones: set | None = None) -> None:
        """Fold one response's delta in. One call is one cycle for expiry purposes.

        `live_track_ids` / `known_zones` are the caller's live sets; either being
        `None` skips that filter (unit tests, and the local backend, which has no
        tracks).
        """
        d = validate_delta(delta)

        confirmed_situations = set()
        for entry in d["situations"]:
            key = entry["key"]
            if entry["state"] == "resolved":
                # Closed situations are already in the events DB. Retaining them here
                # is exactly how the memory grows unbounded (spec 5.2).
                self._situations.pop(key, None)
                continue
            rec = self._situations.get(key)
            if rec is None:
                if not entry["description"]:
                    continue  # nothing to render, and no existing record to confirm
                self._situations[key] = Situation(key=key, description=entry["description"],
                                                  first_seen=now, last_confirmed=now)
            else:
                # first_seen is NEVER restamped - the accumulating duration is the point.
                rec.description = entry["description"] or rec.description
                rec.last_confirmed = now
                rec.unconfirmed_cycles = 0
            confirmed_situations.add(key)

        confirmed_areas = set()
        for entry in d["areas"]:
            zone = entry["zone"]
            if known_zones is not None and zone not in known_zones:
                continue
            rec = self._areas.get(zone)
            if rec is None:
                if not entry["condition"]:
                    continue
                self._areas[zone] = Area(zone=zone, condition=entry["condition"],
                                         first_seen=now, last_confirmed=now)
            else:
                rec.condition = entry["condition"] or rec.condition
                rec.last_confirmed = now
                rec.unconfirmed_cycles = 0
            confirmed_areas.add(zone)

        _age(self._situations, confirmed_situations, self._ttl_cycles)
        _age(self._areas, confirmed_areas, self._ttl_cycles)

        for entry in d["entities"]:
            note = entry["note"] or self._entities.get(entry["id"], "")
            if note:
                self._entities[entry["id"]] = note
        if live_track_ids is not None:
            self._entities = {oid: note for oid, note in self._entities.items()
                              if oid in live_track_ids}

        if d["scene_notes"] is not None:
            self._notes = d["scene_notes"][:self._notes_max_chars]

        _cap(self._situations, self._max_situations, lambda k: self._situations[k].last_confirmed)
        _cap(self._areas, self._max_situations, lambda k: self._areas[k].last_confirmed)
        _cap(self._entities, self._max_situations, None)

    def render(self, now: float) -> str:
        """The prompt block. `""` when the memory is empty, newline-wrapped otherwise.

        Same contract as `TrackObserver.render`, for the same reason: the analyzer's
        `{scene}` slot must collapse to today's exact bytes when there is nothing to
        say. An empty section is omitted entirely rather than left as a bare heading.
        """
        sections = []
        if self._entities:
            rows = [f"    #{oid}: {note}" for oid, note
                    in sorted(self._entities.items(), key=lambda kv: str(kv[0]))]
            sections.append("  ENTITIES\n" + "\n".join(rows))
        if self._situations:
            rows = []
            for rec in sorted(self._situations.values(), key=lambda s: s.first_seen):
                stamp = f"[{format_duration(now - rec.first_seen)}]".ljust(_DURATION_WIDTH)
                rows.append(f"    {stamp}{rec.description}")
            sections.append("  ONGOING\n" + "\n".join(rows))
        if self._areas:
            rows = [f"    {rec.zone}: {rec.condition} ({format_duration(now - rec.first_seen)})"
                    for rec in sorted(self._areas.values(), key=lambda a: a.first_seen)]
            sections.append("  AREAS\n" + "\n".join(rows))
        if self._notes:
            sections.append("  NOTES\n    " + self._notes)
        if not sections:
            return ""
        return "\n" + _HEADING + "\n" + "\n".join(sections) + "\n"

    def snapshot(self, now: float) -> dict:
        """`render`'s structured twin for the HTTP route: same order, no formatting.

        Read-only, and durations rather than `first_seen` for the same reason
        `TrackObserver.snapshot` gives - the stamp is `time.monotonic()`, which no
        client can turn into a wall clock. `scene_notes` is `None` rather than `""`
        so "nothing was said" is one value, not two.
        """
        return {
            "situations": [{"key": rec.key, "description": rec.description,
                            "duration_s": duration_seconds(now - rec.first_seen)}
                           for rec in sorted(self._situations.values(),
                                             key=lambda s: s.first_seen)],
            "areas": [{"zone": rec.zone, "condition": rec.condition,
                       "duration_s": duration_seconds(now - rec.first_seen)}
                      for rec in sorted(self._areas.values(), key=lambda a: a.first_seen)],
            "entities": [{"id": oid, "note": note} for oid, note
                         in sorted(self._entities.items(), key=lambda kv: str(kv[0]))],
            "scene_notes": self._notes or None,
        }


def _age(store: dict, confirmed: set, ttl_cycles: int) -> None:
    """One cycle passes: confirmed entries reset, the rest count up and then expire."""
    for key in list(store):
        if key in confirmed:
            continue
        rec = store[key]
        rec.unconfirmed_cycles += 1
        if rec.unconfirmed_cycles >= ttl_cycles:
            del store[key]


def _cap(store: dict, limit: int, key) -> None:
    """Trim to `limit`, dropping least-recently-confirmed first (`key=None` = oldest)."""
    excess = len(store) - limit
    if excess <= 0:
        return
    for name in (sorted(store, key=key) if key else list(store))[:excess]:
        del store[name]


def _text(raw) -> str:
    """Whitespace-collapsed str; anything else -> "".

    Collapsing rather than merely stripping: a newline inside a description or a note
    would break the rendered block's indentation.
    """
    return " ".join(raw.split()) if isinstance(raw, str) else ""


def _entity_id(raw):
    """The track id as the tracker spells it (an int), tolerating "17" and "#17"."""
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    text = _text(raw).lstrip("#")
    if not text:
        return None
    return int(text) if text.isdigit() else text


def _entries(raw):
    return [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []


def _clean_entities(raw) -> list:
    out = []
    for item in _entries(raw):
        oid = _entity_id(item.get("id"))
        if oid is not None:
            out.append({"id": oid, "note": _text(item.get("note"))})
    return out


def _clean_areas(raw) -> list:
    out = []
    for item in _entries(raw):
        zone = _text(item.get("zone"))
        if zone:
            out.append({"zone": zone, "condition": _text(item.get("condition"))})
    return out


def _clean_situations(raw) -> list:
    out = []
    for item in _entries(raw):
        key = _text(item.get("key"))
        if key:
            out.append({"key": key,
                        "description": _text(item.get("description")),
                        "state": _text(item.get("state")).lower()})
    return out
