"""Anti-Corruption Layer: translates surveillance-camera-sys's raw recognition
stream into attendance domain facts (Sighting). Nothing here imports or
depends on surveillance-camera-sys's own code — only the Redis wire shape,
which is intentionally re-declared below (RawRecognitionEntry) rather than
imported, so a change on that side breaks loudly here instead of silently
propagating.

Contract agreed via /mattpocock-skills:grilling on 2026-08-26 — see
~/.claude/mentor/decisions/attendance-system.md for the full reasoning behind
every decision encoded here.
"""

import json
import logging
import math
import re
from collections import deque
import sqlite3
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta

import redis

from domain import Employee, Sighting
from storage import (get_employee_by_name, get_employee_by_rec_id,
                     move_sighting_first_seen, upsert_sighting)

logger = logging.getLogger("attendance.acl")

RECOGNITION_STREAM_PREFIX = "iris:face_rec:"
META_STREAM_PREFIX = "iris:meta:"

# camera_<n>_<YYYY-MM-DD-HH:MM:SS>.avi — CHIRLA's own filename convention. The clips in this repo are stored with
# dashes in the time (…-HH-MM-SS), because ':' is not a legal character in a Windows file name; both are accepted.
_CHIRLA_FILENAME_RE = re.compile(
    r"camera_\d+_(\d{4}-\d{2}-\d{2}-\d{2}[:-]\d{2}[:-]\d{2})\.avi$"
)


@dataclass(frozen=True)
class RawRecognitionEntry:
    """Mirrors perception/modules/face_recognition/recognizer.py's actual published shape exactly
    (verified against _publish_results and FaceRecognitionResult) — do not
    add fields it doesn't have (no bbox, no wall-clock timestamp, no
    source_id inside the entry itself)."""

    pts: float
    parent_id: int | None
    name: str
    confidence: float
    rec_id: str | None
    # Issue 4 (2026-09-12): True only when the sidecar's vote lock has
    # confirmed this name (face_recognition_min_votes agreeing matches) --
    # the same decision the live label uses. False = a single raw match not
    # yet confirmed; the table ignores those. Defaults to True for entries
    # from a sidecar predating the field, so a version mismatch can't empty
    # the table.
    confirmed: bool = True


class RecognitionStreamReader:
    """Direct Redis XREAD against iris:face_rec:<source_id>, mirroring the
    existing internal pattern in app/stream_manager.py's _read_stream().

    Known, accepted limitation: the sidecar publishes with maxlen=64
    (approximate trim). If this reader falls behind or is disconnected long
    enough for more than ~64 new entries to land, the gap is lost — there is
    no replay/backfill. Acceptable for the spike; a real fix (consumer
    groups with acknowledgment) is deferred, not solved here.
    """

    def __init__(self, redis_client: redis.Redis):
        self.r = redis_client
        self._last_id: dict[str, str] = {}

    def read_new_entries(self, source_id: str, block_ms: int = 1000) -> list[RawRecognitionEntry]:
        key = f"{RECOGNITION_STREAM_PREFIX}{source_id}"
        last_id = self._last_id.get(source_id, "$")  # "$" = only entries from now on, first call
        response = self.r.xread({key: last_id}, block=block_ms, count=10)
        if not response:
            return []

        entries: list[RawRecognitionEntry] = []
        for _stream_key, records in response:
            for record_id, fields in records:
                self._last_id[source_id] = record_id
                pts = float(fields[b"pts"])
                for face in json.loads(fields[b"face_rec"]):
                    entries.append(RawRecognitionEntry(
                        pts=pts,
                        parent_id=face.get("parent_id"),
                        name=face["name"],
                        confidence=face["confidence"],
                        rec_id=face.get("rec_id"),
                        confirmed=face.get("confirmed", True),
                    ))
        return entries


# ── Camera clock (issue 1 Phase 2, 2026-09-12) ─────────────────────────────
# The sidecar attaches each frame's camera-side capture time (from the
# camera's RTCP Sender Reports) to iris:meta as `ntp_ts`, in nanoseconds --
# 0 when the camera never sent one (see sidecar.py's attach-sys-ts note).
# Seconds from the NTP epoch (1900-01-01) to the Unix epoch (1970-01-01).
_NTP_TO_UNIX_SECONDS = 2208988800
# A seconds value above this can only be NTP-epoch-based (it's ~2065 in Unix
# time, ~1995 in NTP time). DeepStream reports Unix-epoch ns; accepting both
# costs one comparison.
_NTP_EPOCH_THRESHOLD = 3.0e9
# A camera whose clock was never set reports e.g. 1970 or 2000. Anything
# before this counts as "no usable camera clock" rather than dating
# sightings decades back (which would also drop them out of today's table).
_MIN_PLAUSIBLE_CAMERA_TIME = datetime(2020, 1, 1).timestamp()


def camera_clock_offset_seconds(ntp_ts_ns: int, record_id: bytes | str) -> float | None:
    """(camera clock - server clock) in seconds for one iris:meta entry, or
    None if it carries no usable camera time. The server-side reference is
    the entry's own Redis stream id, whose first half is the millisecond time
    Redis stored it -- this machine's clock at publish time, a fraction of a
    second after the frame was captured."""
    if ntp_ts_ns <= 0:
        return None
    camera_s = ntp_ts_ns / 1e9
    if camera_s > _NTP_EPOCH_THRESHOLD:
        camera_s -= _NTP_TO_UNIX_SECONDS
    if camera_s < _MIN_PLAUSIBLE_CAMERA_TIME:
        return None
    rid = record_id.decode() if isinstance(record_id, bytes) else record_id
    return camera_s - int(rid.split("-", 1)[0]) / 1000.0


class CameraClock:
    """Latest camera-vs-server clock offset per camera, shared by the consumer
    thread (stamps sightings with it) and the API (/api/recordings, which
    feeds the camera tile's clock) -- both run in this one process. A single
    dict assignment/read is atomic under the GIL, so no lock."""

    # No fresh reading for this long = the camera stopped providing its clock
    # (or its stream dropped): fall back to server time.
    MAX_AGE_SECONDS = 10.0

    def __init__(self) -> None:
        self._latest: dict[str, tuple[float, float]] = {}  # cam -> (offset_s, monotonic learned)

    def note(self, source_id: str, offset_seconds: float) -> None:
        self._latest[source_id] = (offset_seconds, time.monotonic())

    def offset(self, source_id: str) -> float | None:
        """Seconds to add to this server's clock to get the camera's, or None
        when there's no fresh camera clock (the caller uses server time)."""
        entry = self._latest.get(source_id)
        if entry is None or time.monotonic() - entry[1] > self.MAX_AGE_SECONDS:
            return None
        return entry[0]


camera_clock = CameraClock()


class LiveCameraWatch:
    """Which LIVE cameras are actually delivering video right now, and since
    when (2026-09-13). data/stream_starts.json is written only by the test
    streamer, for the recordings it plays -- so without this a live camera
    never counted as streaming: the consumer skipped it and nobody seen on it
    ever reached the attendance table (or got a tile clock).

    "Delivering video" = the sidecar published a frame record for it
    (iris:meta:<cam>) within FRESH_SECONDS. The sidecar's own status flag says
    running even for a camera with no video (checked 2026-09-13), so it can't
    be used. Refreshed by the consumer thread every cycle; read by the API
    (same process) for the camera tile's clock. refresh() publishes a new dict
    by plain assignment, so snapshot() never sees one being modified."""

    FRESH_SECONDS = 10.0

    def __init__(self) -> None:
        self._since: dict[str, datetime] = {}

    def refresh(self, r: redis.Redis, cameras: list[str]) -> dict[str, datetime]:
        now_ms = time.time() * 1000
        old, new = self._since, {}
        for cam in cameras:
            try:
                tip = r.xrevrange(f"{META_STREAM_PREFIX}{cam}", count=1)
            except redis.RedisError:
                if cam in old:  # Redis hiccup: keep the last known state
                    new[cam] = old[cam]
                continue
            if tip:
                rid = tip[0][0]
                rid = rid.decode() if isinstance(rid, bytes) else rid
                if now_ms - int(rid.split("-", 1)[0]) <= self.FRESH_SECONDS * 1000:
                    new[cam] = old.get(cam, datetime.now())
        self._since = new
        return dict(new)

    def snapshot(self) -> dict[str, datetime]:
        return dict(self._since)


live_watch = LiveCameraWatch()


class MetaStreamReader:
    """Direct Redis XREAD against iris:meta:<source_id> -- the sidecar's
    general per-frame stream, published every frame regardless of face
    visibility (unlike iris:face_rec, which only has an entry when a face
    was actually detected). Used only to read `dets` (every currently-
    tracked object this frame: [label, confidence, object_id]) so
    PersonPresenceTracker can know a person is still on screen even with
    no face showing -- e.g. their back turned. Nothing else on this stream
    (jpeg bytes, non-person detections) is read here.

    Non-blocking by design (no `block=` passed to XREAD): iris:meta is high
    volume (every frame, every camera) compared to iris:face_rec, so it
    almost always already has something waiting; the per-camera face_rec
    read's own block_ms already paces the consumer loop, and blocking
    again here would just add latency on top for no benefit. Same
    maxlen-trim limitation as RecognitionStreamReader.
    """

    def __init__(self, redis_client: redis.Redis):
        self.r = redis_client
        self._last_id: dict[str, str] = {}

    def read_present_person_tracks(self, source_id: str) -> tuple[set[int], dict[int, tuple[float, float, float, float]]] | None:
        """(person object_ids, {object_id: (left, top, w, h)}) seen in the
        newest meta entries read this call, or None if nothing new has
        arrived since the last call -- callers must treat None as "no
        information this cycle", NOT as "nobody present" (that would wrongly
        look like everyone just left). Boxes come from the sidecar's
        additive `person_boxes` meta field (2026-09-01) and may be missing
        for a given entry (older sidecar, no boxes that frame) -- the ids
        set is the authoritative presence signal, boxes are best-effort
        extra geometry for the collision guard."""
        key = f"{META_STREAM_PREFIX}{source_id}"
        last_id = self._last_id.get(source_id)
        if last_id is None:
            # Prime the cursor to the stream's current tip. "$" only means
            # "new entries" for BLOCKING reads -- a non-blocking XREAD from
            # "$" returns nothing, forever, so the cursor never advances.
            # Found live 2026-09-01: this reader had never consumed a single
            # entry since it was written (the presence bridge appeared to
            # work only because frequent face matches masked it).
            tip = self.r.xrevrange(key, count=1)
            self._last_id[source_id] = (tip[0][0].decode() if tip else "0-0")
            return None  # nothing reportable this cycle; cursor is primed now
        # count=64 = the stream's own maxlen: reads the entire retained window
        # in one call, so the cursor can never fall behind the trim horizon
        # (at ~12 meta entries/s a 10-entry read lost ground every cycle).
        response = self.r.xread({key: last_id}, count=64)
        if not response:
            return None

        present: set[int] = set()
        boxes: dict[int, tuple[float, float, float, float]] = {}
        for _stream_key, records in response:
            for record_id, fields in records:
                self._last_id[source_id] = record_id
                # Camera clock (Phase 2): present only when the camera sent an
                # RTCP Sender Report; absent = no camera clock for this frame.
                ntp_raw = fields.get(b"ntp_ts")
                if ntp_raw:
                    offset = camera_clock_offset_seconds(int(ntp_raw), record_id)
                    if offset is not None:
                        camera_clock.note(source_id, offset)
                dets = json.loads(fields[b"dets"])
                for label, _conf, object_id in dets:
                    if label == "person":
                        present.add(object_id)
                raw_boxes = fields.get(b"person_boxes")
                if raw_boxes:
                    for left, top, w, h, object_id in json.loads(raw_boxes):
                        boxes[object_id] = (left, top, w, h)  # newest entry wins
        return present, boxes


class PersonPresenceTracker:
    """Bridges a person's recognized identity across frames where their
    face isn't visible (e.g. their back is turned) -- Q3/point-5 (2026-08-31).

    The sidecar's face stream simply has no entry at all for a tracked
    person on a frame with no detected face, so without this, a
    back-turned stretch longer than the coalescer's stale-bucket window
    looks identical to "the person left" -- their sighting fragments into
    disconnected pieces instead of one continuous one.

    Built entirely from data already published -- iris:meta's `dets`
    (every tracked person, every frame, face or no face) cross-referenced
    against iris:face_rec's own `parent_id` -- no surveillance-camera-sys
    code touched, per the "everything on our side" decision.

    Once a person-track has been matched to an employee (a real face_rec
    hit), the mapping is kept alive as long as that SAME track_id keeps
    appearing in `dets`. Once it disappears from `dets` for
    `stale_after_seconds`, the mapping is dropped -- protects against
    DeepStream recycling a track_id onto a different physical person and
    that new person silently inheriting the old one's identity. This is
    the same hazard the sidecar's own "teleport guard" defends against for
    its live overlay (see face_recognizer.py's _stabilize_names) -- solved
    independently here using presence continuity instead of face position,
    since face position isn't available on the frames we most need to
    survive (no face detected at all).
    """

    # Two person boxes overlapping by at least this share of the SMALLER box
    # count as a collision (intersection/min-area, not IoU: a near person
    # occluding a far one produces a big overlap of the far box while IoU
    # stays small -- exactly the case where the tracker swaps ids).
    COLLIDE_OVERLAP = 0.35
    # Motion (2026-09-01, user-reported issues: seated back-turned person
    # losing credit; passer-by gaining it): a track is "moving" when its box
    # center travelled more than MOTION_FRACTION of its box height (floored
    # at MOTION_MIN_PX) within the last MOTION_WINDOW seconds. Stationary
    # bodies keep their identity through collisions (a tracker only swaps
    # ids between movers) and only stationary bodies count as at-desk.
    MOTION_WINDOW = 3.5      # seconds of history compared (sized to ~0.3-1s consumer cycles)
    MOTION_MIN_SPAN = 0.5    # need at least this much history to judge, else "moving"
    MOTION_FRACTION = 0.2
    MOTION_MIN_PX = 12.0

    def __init__(self, stale_after_seconds: float = 5.0):
        self._stale_after = stale_after_seconds
        # (source_id, track_id) -> (employee_id, real time last confirmed
        # present, tainted). tainted=True means the track went through a
        # collision since its identity was last face-confirmed -- the
        # track->employee binding can no longer be trusted (the tracker may
        # have swapped ids between the two colliding people), so presence
        # bridging is SUSPENDED until a fresh face match re-confirms who this
        # body actually is (user-reported case 2026-09-01: P1 recognized,
        # collides with P2, leaves -- P2 must not keep logging time as P1).
        self._tracks: dict[tuple[str, int], tuple[str, float, bool]] = {}
        # (source_id, track_id) -> confidence of the face match that won the
        # binding (identity uniqueness, 2026-09-01: per camera one name <-> one
        # body; a contested name goes to the higher-confidence claim).
        self._claim_conf: dict[tuple[str, int], float] = {}
        # (source_id, track_id) -> (video wall-clock first seen in dets, real
        # time last confirmed present). Kept for EVERY person track, matched
        # or not -- this is what lets a face match that finally happens
        # backdate the person's first_seen to when their BODY first appeared
        # (the back-turned-since-the-start case, 2026-09-01: e.g. seq_020
        # cam-4 opens with people facing away for minutes). The same
        # stale_after continuity guard applies: a track absent from dets
        # longer than that is dropped, so a recycled track_id starts a fresh
        # first-seen instead of inheriting an ancient one.
        self._track_first_seen: dict[tuple[str, int], tuple[datetime, float]] = {}
        # (source_id, track_id) -> recent (monotonic, cx, cy, box_h) samples
        self._motion: dict[tuple[str, int], deque] = {}

    def claim(self, source_id: str, track_id: int | None, employee_id: str,
              confidence: float = 1.0) -> bool:
        """A face match says track_id is employee_id with this confidence.
        Enforces GLOBAL identity uniqueness (user requirement 2026-09-01):
        a name can be held by ONE body across all cameras. If another
        live track already holds the name, the higher-confidence claim wins
        and the loser is unbound (becomes unnamed); an unnamed body can only
        take a name nobody holds. Returns True if track_id owns the name
        after this call. A winning fresh face match also clears any
        collision taint on the binding."""
        if track_id is None:
            return False
        key = (source_id, track_id)
        now = time.monotonic()
        holder = None
        for other_key, (emp, _last, _tainted) in self._tracks.items():
            # GLOBAL: the name may be held by one body across all cameras
            # (each camera covers a different room -- user, 2026-09-01).
            if other_key != key and emp == employee_id:
                holder = other_key
                break
        if holder is not None:
            if confidence > self._claim_conf.get(holder, 0.0):
                logger.info("identity %s on %s: track %s (conf %.2f) outbids track %s (conf %.2f) -- "
                            "track %s is now unnamed", employee_id, source_id, track_id, confidence,
                            holder[1], self._claim_conf.get(holder, 0.0), holder[1])
                del self._tracks[holder]
                self._claim_conf.pop(holder, None)
            else:
                return False  # name is taken by a stronger claim; this body stays unnamed
        self._tracks[key] = (employee_id, now, False)
        self._claim_conf[key] = max(confidence, self._claim_conf.get(key, 0.0) if self._tracks.get(key, (None,))[0] == employee_id else 0.0)
        return True

    def note_match(self, source_id: str, track_id: int | None, employee_id: str) -> None:
        """Back-compat wrapper: an unconditional claim (confidence 1.0)."""
        self.claim(source_id, track_id, employee_id, 1.0)

    def note_present_tracks(self, source_id: str, present_track_ids: set[int],
                            wall_clock_at: datetime,
                            boxes: dict[int, tuple[float, float, float, float]] | None = None) -> None:
        """Record every person track present this cycle (identity known or
        not), remembering each track's earliest continuously-connected
        sighting time. Call once per cycle per camera, BEFORE ingesting the
        cycle's face matches, so a first-ever face match can already see how
        long its track has been around.

        When boxes are available, also detect COLLISIONS (two person boxes
        overlapping >= COLLIDE_OVERLAP of the smaller): the tracker may swap
        ids during the occlusion, so both involved tracks' identity mappings
        get tainted (presence bridging suspended until a face re-confirms)
        and both tracks' first-seen anchors reset to now (a later backdate
        must not credit time from before the swap point)."""
        now = time.monotonic()
        for track_id in present_track_ids:
            key = (source_id, track_id)
            existing = self._track_first_seen.get(key)
            if existing is None:
                self._track_first_seen[key] = (wall_clock_at, now)
            else:
                self._track_first_seen[key] = (existing[0], now)
        # Motion samples
        for track_id, box in (boxes or {}).items():
            if track_id not in present_track_ids:
                continue
            key = (source_id, track_id)
            hist = self._motion.setdefault(key, deque(maxlen=64))
            left, top, w, h = box
            hist.append((now, left + w / 2.0, top + h / 2.0, h))
        # Continuity prune (same hazard as _tracks: recycled ids)
        stale = [k for k, (_, last) in self._track_first_seen.items()
                 if k[0] == source_id and now - last > self._stale_after]
        for k in stale:
            del self._track_first_seen[k]
            self._motion.pop(k, None)

        # ── Collision guard ────────────────────────────────────────────
        if not boxes or len(boxes) < 2:
            return
        ids = [t for t in boxes if t in present_track_ids]
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                if self._overlap_ratio(boxes[a], boxes[b]) < self.COLLIDE_OVERLAP:
                    continue
                # An id swap needs TWO movers: trackers essentially never
                # swap a moving target with a stationary one, so a walker
                # passing a seated person taints nobody (found live
                # 2026-09-01: tainting the walker alone still produced ~90
                # suspensions per 4 min in a small office). A track without
                # motion history yet counts as still here -- it has no
                # identity binding worth protecting, and after every reset
                # everybody briefly lacks history.
                if not (self.is_moving(source_id, a, unknown=False)
                        and self.is_moving(source_id, b, unknown=False)):
                    continue
                for track_id, other in ((a, b), (b, a)):
                    key = (source_id, track_id)
                    self._track_first_seen[key] = (wall_clock_at, now)
                    mapped = self._tracks.get(key)
                    if mapped is not None and not mapped[2]:
                        logger.info(
                            "collision on %s: moving track %s (mapped to %s) overlaps track %s -- "
                            "presence bridge suspended until a face re-confirms",
                            source_id, track_id, mapped[0], other)
                        self._tracks[key] = (mapped[0], mapped[1], True)

    def is_moving(self, source_id: str, track_id: int, unknown: bool = True) -> bool:
        """True if the track's box center travelled more than
        MOTION_FRACTION of its height (min MOTION_MIN_PX) over the last
        MOTION_WINDOW seconds. `unknown` is the answer for a track without
        enough history yet (< MOTION_MIN_SPAN of samples): True for desk
        credit (conservative -- the debounce absorbs the moment's delay),
        False for the collision guard (see note_present_tracks)."""
        hist = self._motion.get((source_id, track_id))
        if not hist:
            return unknown
        now = time.monotonic()
        recent = [s for s in hist if now - s[0] <= self.MOTION_WINDOW]
        if len(recent) < 4 or recent[-1][0] - recent[0][0] < self.MOTION_MIN_SPAN:
            return unknown
        # Robust to detector jitter (found live 2026-09-01: a seated person
        # near the camera, partly hidden by the desk, gets a box that
        # flickers between full and partial size -- the raw center jumps
        # 50-100px while nobody moves). Compare the MEDIAN position of the
        # older half of the window with the median of the newer half: jitter
        # averages out, real walking is a consistent shift.
        half = len(recent) // 2
        older, newer = recent[:half], recent[half:]
        med = lambda vals: sorted(vals)[len(vals) // 2]
        dx = med([s[1] for s in newer]) - med([s[1] for s in older])
        dy = med([s[2] for s in newer]) - med([s[2] for s in older])
        h = med([s[3] for s in recent])
        threshold = max(self.MOTION_MIN_PX, self.MOTION_FRACTION * h)
        return math.hypot(dx, dy) > threshold

    @staticmethod
    def _overlap_ratio(a: tuple[float, float, float, float],
                       b: tuple[float, float, float, float]) -> float:
        """Intersection area over the SMALLER box's area (0 when disjoint)."""
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
        iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
        inter = ix * iy
        smaller = min(aw * ah, bw * bh)
        return inter / smaller if smaller > 0 else 0.0

    def binding_sources(self) -> dict[str, str]:
        """{employee_id: source_id} of every live (untainted) binding."""
        return {emp: key[0] for key, (emp, _last, tainted) in self._tracks.items() if not tainted}

    def employee_for_track(self, source_id: str, track_id: int) -> str | None:
        """Employee currently bound to this track, or None if unmapped OR the
        binding is collision-tainted (then nobody may be credited for what
        this body does -- productivity monitoring, 2026-09-01)."""
        mapped = self._tracks.get((source_id, track_id))
        if mapped is None or mapped[2]:
            return None
        return mapped[0]

    def first_seen_of_track(self, source_id: str, track_id: int | None) -> datetime | None:
        """Video wall-clock when this track first appeared in dets, if it has
        been continuously present since -- None for unknown/expired tracks."""
        if track_id is None:
            return None
        entry = self._track_first_seen.get((source_id, track_id))
        return entry[0] if entry else None

    def clear(self) -> None:
        """Forget everything -- called on a loop-wrap reset so no mapping or
        first-seen time from the previous lap can leak into the new one."""
        self._tracks.clear()
        self._claim_conf.clear()
        self._track_first_seen.clear()
        self._motion.clear()

    def still_present_employees(self, source_id: str, present_track_ids: set[int]) -> list[str]:
        """Given this cycle's currently-present person track_ids for a
        camera (from MetaStreamReader), return the employee_id(s) whose
        last-matched track is among them -- these should have their
        sighting extended even though no fresh face match happened this
        cycle. Also prunes any mapping that's aged out of `dets` (see
        class docstring)."""
        now = time.monotonic()
        present: list[str] = []
        stale_keys: list[tuple[str, int]] = []
        for key, (employee_id, last_seen, tainted) in list(self._tracks.items()):
            src, track_id = key
            if src != source_id:
                continue
            if track_id in present_track_ids:
                # A tainted mapping stays alive (waiting for a face to
                # re-confirm or re-assign it) but does NOT extend presence:
                # after a collision this body might be the OTHER person.
                if not tainted:
                    present.append(employee_id)
                self._tracks[key] = (employee_id, now, tainted)  # refresh
            elif now - last_seen > self._stale_after:
                stale_keys.append(key)
        for key in stale_keys:
            del self._tracks[key]
            self._claim_conf.pop(key, None)
        return present


def wall_clock_for_spike(source_filename: str, pts: float, duration_seconds: float) -> datetime:
    """SUPERSEDED — see wall_clock_from_stream_time below. Kept only for the
    historical record (still used by run_spike.py's old backfill mode).

    CHIRLA-specific (see Q6, 2026-08-26): the filename embeds the recording's
    own wall-clock start time; pts is seconds elapsed since then, mod'd by
    the file's duration (2026-08-30 fix — the source file loops forever, pts
    doesn't reset per loop).

    Turned out insufficient: the sidecar's own `pts` doesn't reset on an
    *upstream RTSP reconnect* either (the sidecar container itself keeps
    running; each source's pts apparently tracks time since that source's own
    connection/muxer reference, not the file's actual content position) —
    caught live when two cameras that should have agreed reported pts values
    ~5000s apart after routine reconnects. Do not use `pts` from the
    recognition stream for wall-clock purposes; use wall_clock_from_stream_time."""
    match = _CHIRLA_FILENAME_RE.search(source_filename)
    if not match:
        raise ValueError(f"filename does not match CHIRLA's camera_N_<timestamp>.avi convention: {source_filename}")
    start = datetime.strptime(match.group(1).replace(":", "-"), "%Y-%m-%d-%H-%M-%S")
    return start + timedelta(seconds=pts % duration_seconds)


def wall_clock_from_stream_time(
    recording_start: datetime,
    stream_started_at: datetime,
    duration_seconds: float,
    at: datetime | None = None,
) -> datetime:
    """Position in the source file derived from REAL elapsed wall-clock time
    since ffmpeg actually started streaming it, not from the sidecar's
    self-reported pts (see wall_clock_for_spike's docstring for why that
    doesn't work). ffmpeg reads with -re (real-time pacing) and loops
    forever, so "how far into the file are we" is authoritatively
    `(now - stream_started_at) % duration_seconds`, regardless of anything
    happening on the perception/pipeline/RTSP side.

    `at` defaults to the real now (datetime.now()) — pass it explicitly only
    for testing.
    """
    now = at if at is not None else datetime.now()
    elapsed = (now - stream_started_at).total_seconds()
    return recording_start + timedelta(seconds=elapsed % duration_seconds)


class SightingCoalescer:
    """Groups raw recognition entries into one Sighting row per
    (employee, camera, minute) — see Q10/Q12. Writes to the database on
    EVERY ingest() call, immediately -- upsert_sighting's natural-key lookup
    already makes repeated touches of the same bucket a cheap no-op-ish
    UPDATE, so there's no real correctness reason to delay it. This used to
    buffer in memory and only write once a bucket went quiet for
    stale_after_seconds (8s) -- found live 2026-08-31: with someone
    continuously on screen resetting that 8s idle clock on every touch, plus
    up to ~4s until the next flush-check cycle even ran, the table could lag
    up to ~40s behind what the live camera was already showing. `_open` is
    kept only to know which bucket (i.e. which first_seen_at) a new touch
    belongs to -- it's bookkeeping, not a write buffer, now.
    """

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
        # (employee_id, camera, minute_start) -> (Sighting, real-time last touched).
        # minute_start is *video* wall-clock (from wall_clock_for_spike, now bounded/
        # wrapping once pts exceeds a file's duration) -- correct for grouping repeat
        # sightings of the same video moment across loops, but useless for deciding
        # when a bucket is done: video time isn't monotonic across a loop wrap. So
        # flush timing (below) is driven by real processing time instead, entirely
        # decoupled from the video clock.
        self._open: dict[tuple[str, str, datetime], tuple[Sighting, float]] = {}

    def ingest(self, source_id: str, entry: RawRecognitionEntry, wall_clock_at: datetime) -> Employee | None:
        # Unknowns are filtered out entirely (Q9) — attendance only tracks
        # known employees; unrecognized-face review is surveillance's job.
        if entry.name == "unknown":
            return None

        # rec_id is present on a fresh, this-frame match. It's deliberately
        # None on a *stabilized* match (face_recognizer.py's anti-flicker
        # vote-lock) whenever the locked name differs from the raw frame's
        # own candidate -- which is the common case for any frame where the
        # face angle doesn't independently clear the recognition threshold
        # but the already-locked identity is still correct. `name` stays
        # correct in that case even though rec_id doesn't -- fall back to a
        # name lookup rather than dropping the sighting. Found live
        # 2026-08-31: a genuine "Person-6" entry sitting in the stream with
        # rec_id: null, silently discarded before this fallback existed --
        # very likely the main cause of people missing from the table.
        employee = self.resolve(entry)
        if employee is None:
            return None
        self.record(source_id, employee, wall_clock_at)
        return employee

    def resolve(self, entry: RawRecognitionEntry) -> Employee | None:
        """Which enrolled employee this entry names, or None (unknown /
        not enrolled). Split from ingest() so the consumer can run the
        identity-uniqueness claim BEFORE anything is written."""
        if entry.name == "unknown":
            return None
        employee = (get_employee_by_rec_id(self._conn, entry.rec_id) if entry.rec_id is not None
                    else get_employee_by_name(self._conn, entry.name))
        # None here = the sidecar's gallery matched *someone* never enrolled
        # as an employee in our forked gallery -- not our concern (Q3).
        return employee

    def record(self, source_id: str, employee: Employee, wall_clock_at: datetime) -> None:
        """Write the sighting for an ACCEPTED claim. No confidence gate
        (Q11): the sidecar's stabilization already resolved per-frame noise."""
        self._touch_bucket(employee.id, source_id, wall_clock_at)

    def touch_presence(self, employee_id: str, source_id: str, wall_clock_at: datetime) -> None:
        """Extend an employee's current sighting from person-track presence
        alone -- no fresh face match this cycle. See PersonPresenceTracker:
        handles the back-turned case, where a track keeps appearing in
        iris:meta's `dets` but has no face for face_rec to publish."""
        self._touch_bucket(employee_id, source_id, wall_clock_at)

    def backdate_first_seen(self, employee_id: str, source_id: str,
                            track_first_seen: datetime) -> bool:
        """Pull an employee's earliest open sighting back to when their BODY
        track first appeared -- the mirror image of touch_presence
        (2026-09-01): touch_presence extends a sighting FORWARD through a
        back-turned stretch after a face match; this extends it BACKWARD
        through a back-turned stretch that happened BEFORE the first face
        match (someone facing away since entering frame is present that
        whole time, they just couldn't be named until they turned around).
        The caller passes PersonPresenceTracker.first_seen_of_track, whose
        continuity guard is what makes this safe. Returns True if a row
        was actually moved."""
        earliest_key, earliest = None, None
        for key, (candidate, _touched) in self._open.items():
            if key[0] != employee_id or key[1] != source_id:
                continue
            # `earliest` holds the Sighting itself (NOT the (sighting, touched)
            # tuple stored in _open) -- mixing the two aborted the consumer
            # cycle with an AttributeError whenever an employee had 2+ open
            # buckets (found live 2026-09-01: 28 lost cycles in an hour).
            if earliest is None or candidate.first_seen_at < earliest.first_seen_at:
                earliest_key, earliest = key, candidate
        if earliest is None or earliest_key is None:
            return False
        sighting = earliest
        touched_at = self._open[earliest_key][1]
        if track_first_seen >= sighting.first_seen_at:
            return False  # nothing to gain -- already starts earlier
        moved = Sighting(
            id=sighting.id,
            employee_id=sighting.employee_id,
            camera=sighting.camera,
            first_seen_at=track_first_seen,
            last_seen_at=sighting.last_seen_at,
        )
        self._open[earliest_key] = (moved, touched_at)
        # NOT upsert_sighting: its natural key includes first_seen_at, which
        # is exactly what's changing -- see move_sighting_first_seen.
        move_sighting_first_seen(self._conn, moved.id, track_first_seen)
        return True

    def _touch_bucket(self, employee_id: str, source_id: str, wall_clock_at: datetime) -> None:
        minute_start = wall_clock_at.replace(second=0, microsecond=0)
        bucket_key = (employee_id, source_id, minute_start)
        touched_at = time.monotonic()

        existing = self._open.get(bucket_key)
        if existing is None:
            sighting = Sighting(
                id=str(uuid.uuid4()),
                employee_id=employee_id,
                camera=source_id,
                first_seen_at=wall_clock_at,
                last_seen_at=wall_clock_at,
            )
        else:
            existing_sighting, _ = existing
            sighting = Sighting(
                id=existing_sighting.id,
                employee_id=existing_sighting.employee_id,
                camera=existing_sighting.camera,
                first_seen_at=existing_sighting.first_seen_at,
                last_seen_at=wall_clock_at,
            )
        self._open[bucket_key] = (sighting, touched_at)
        upsert_sighting(self._conn, sighting)  # write immediately -- see class docstring

    def clear(self) -> None:
        """Discard every open (not-yet-flushed) bucket without writing it --
        used on a loop-wrap reset, so a bucket still in flight from the lap
        that just ended can't flush afterwards and repopulate a row the
        reset was meant to clear."""
        self._open.clear()

    def flush_due_buckets(self, stale_after_seconds: float = 8.0, force: bool = False) -> list[Sighting]:
        """Evict every open bucket that hasn't been touched recently (in
        real processing time) -- or every open bucket regardless, if
        force=True. NOT the write path anymore (ingest() already wrote each
        of these the moment they were last touched) -- this is memory
        bookkeeping (a bucket that's gone quiet doesn't need to keep costing
        a dict entry) plus the trigger consumer.py uses to know when it's
        worth pushing a fresh websocket snapshot. Call periodically, not
        per-entry.

        Deliberately NOT based on video wall-clock: that clock now wraps
        every time the source file loops (see wall_clock_for_spike), so it
        can't tell "this bucket is done" from "the video looped back to an
        earlier minute" -- real elapsed time can."""
        now = time.monotonic()
        due_keys = [
            key for key, (_, touched_at) in self._open.items()
            if force or (now - touched_at) > stale_after_seconds
        ]

        flushed: list[Sighting] = []
        for key in due_keys:
            sighting, _ = self._open.pop(key)
            # No upsert_sighting call here -- ingest() already wrote this
            # sighting's latest state the moment it was last touched.
            flushed.append(sighting)
        return flushed
