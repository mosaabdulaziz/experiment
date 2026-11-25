"""Persistent consumer loop: the piece that was missing before -- continuously
reads the recognition stream (rather than one-shot backfill like run_spike.py)
and calls a callback whenever new sightings are flushed, so a live view (the
websocket in server.py) has something real to push.

Runs in a background thread (started from server.py's startup hook), since
RecognitionStreamReader.read_new_entries blocks on Redis XREAD.
"""
import logging
import threading
import time
from datetime import datetime
from typing import Callable

import redis

from acl import (
    MetaStreamReader,
    PersonPresenceTracker,
    RecognitionStreamReader,
    SightingCoalescer,
    camera_clock,
    live_watch,
)
from cameras import CAM_DURATIONS, CAM_FILENAMES, load_stream_starts
from domain import Sighting
from pathlib import Path
from productivity import DeskStateEngine, SeatOccupancy, ZoneBody, policy_store
from sessions import attendance_policy_store
from storage import clear_sightings_for_cameras, connect
from violations import ROLE_NOT_ALLOWED, UNIDENTIFIED, ZoneEvidence, ZoneViolationEngine
from zones import FRAME_H, FRAME_W, effective_restricted_zones, role_allowed, seat_zones, zone_hit, zone_store

logger = logging.getLogger("attendance.consumer")

FLUSH_EVERY_N_CYCLES = 20  # cycles ~= 20 * block_ms; block_ms=200 -> broadcast + in-memory
                            # eviction check ~every 4s. Sightings are written
                            # to the database immediately in ingest() now --
                            # this only paces the websocket push and memory
                            # cleanup, no longer the write itself.

class ConsumerThread(threading.Thread):
    """Continuously ingests every configured camera's recognition stream into
    Sighting rows, invoking on_flush(sightings) whenever buckets close.
    Daemon thread -- dies with the process, no graceful-shutdown handling
    needed for this spike-stage tool."""

    def __init__(self, db_path: str, redis_url: str, on_flush: Callable[[list[Sighting]], None]):
        super().__init__(daemon=True)
        self._db_path = db_path
        self._redis_url = redis_url
        self._on_flush = on_flush
        # Productivity monitoring (2026-09-01): hot-reloading policy/zones
        # stores, the debounced desk-state engine, and the latest per-camera
        # scene (who is where, at-desk or not) for the monitoring page's
        # overlays. Read by server.py; written only by this thread.
        data_dir = Path(db_path).parent
        self.policy = policy_store(data_dir)
        self.zones = zone_store(data_dir, Path(__file__).parent / "zones.seed.json")
        # Attendance v2 (2026-09-02): its OWN store, separate file --
        # deliberately not merged into self.policy above, see sessions.py's
        # module docstring for why.
        self.attendance_policy = attendance_policy_store(data_dir)
        self.engine: DeskStateEngine | None = None
        self.violations: ZoneViolationEngine | None = None
        self.scene: dict[str, list[dict]] = {}
        self.seats = SeatOccupancy()

    def run(self) -> None:
        conn = connect(self._db_path)
        r = redis.Redis.from_url(self._redis_url)
        reader = RecognitionStreamReader(r)
        meta_reader = MetaStreamReader(r)
        presence_tracker = PersonPresenceTracker()
        coalescer = SightingCoalescer(conn)
        self.engine = DeskStateEngine(conn, self.policy)
        self.violations = ZoneViolationEngine(
            conn, lambda: float(self.attendance_policy.load().get("violation_debounce_seconds", 3.0)))

        # elapsed-time position (seconds into the file, wrapped) last seen
        # for each *actively streaming* camera -- used only to detect a loop
        # wrap (position suddenly drops), not for timestamping.
        last_position: dict[str, float] = {}

        # Backoff + rate-limited logging for Redis read failures (e.g. Redis
        # down). Without this, a failed XREAD raises immediately (doesn't
        # block for block_ms), so the loop retries as fast as the connection
        # keeps refusing -- found live 2026-08-31: 7.6M tracebacks, 12+ GB of
        # log, ~81% CPU sustained, in about 20 minutes. Note this only stops
        # the *wasted spinning* -- it does NOT recover any data: while Redis
        # is unreachable, the sidecar can't publish either (its own xadd is
        # try/excepted the same way), so events from that window are simply
        # never generated anywhere, not merely delayed. Same reasoning as
        # RecognitionStreamReader's documented maxlen=64 trim limitation --
        # accepted for this spike, not solved here.
        consecutive_failures = 0
        last_failure_logged_at = 0.0

        cycles = 0

        logger.info("consumer started, watching %d cameras", len(CAM_FILENAMES))
        while True:
            # The entire cycle body below is wrapped in this try/except.
            # Previously it wasn't -- any unhandled exception ANYWHERE in it
            # (not just the specifically try/excepted spots) silently killed
            # this whole thread: a Python thread that raises doesn't crash
            # the process or get retried, it just stops, with nothing more
            # than a stderr printout from the default excepthook (easy to
            # miss, and lost entirely across a log truncation). Found live
            # 2026-08-31: the thread had been dead for several minutes with
            # zero trace in the log -- the table froze at whatever it last
            # had while the live cam kept going, which from the outside
            # looks exactly like "table shows a different time than the live
            # cam." A background thread meant to run forever must not be
            # able to die from any single bad cycle.
            try:
                # Test recordings (the test streamer's file, re-read each cycle to
                # pick up reconnects) plus LIVE cameras actually delivering video
                # (2026-09-13) -- without the latter a live camera never counted
                # as streaming, so nothing seen on it reached the table. Live =
                # attendance cameras with no fixed recording length
                # (cameras.py), checked against the sidecar's frame records
                # every cycle; see acl.LiveCameraWatch.
                live_cams = [c for c in CAM_FILENAMES if c not in CAM_DURATIONS]
                stream_starts = {**load_stream_starts(), **live_watch.refresh(r, live_cams)}

                # Reset the whole table together when ANY currently-streaming
                # ("chosen") camera's video loops back to its start -- not
                # scoped to just that one camera. All the CHIRLA seq_004
                # cameras are the same synchronized recording watched from
                # different angles, so one of them completing a lap means
                # the whole scene did; treating them as one unit avoids a
                # confusing state where some cameras' history has reset and
                # others' hasn't. Without this, compute_daily_attendance's
                # MIN/MAX aggregation can't tell one lap of this dev-loop
                # streamer from the next -- every lap just keeps merging
                # into the same running total forever.
                now = datetime.now()
                wrapped_cam = None
                for cam_id, started_at in stream_starts.items():
                    duration = CAM_DURATIONS.get(cam_id)
                    if duration is None:
                        continue
                    position = (now - started_at).total_seconds() % duration
                    if cam_id in last_position and position < last_position[cam_id]:
                        wrapped_cam = cam_id
                    last_position[cam_id] = position

                if wrapped_cam is not None:
                    coalescer.clear()
                    presence_tracker.clear()  # no track first-seen/identity may leak across laps
                    # 2026-09-13: only the TEST recordings' rows (cameras with a
                    # fixed length, CAM_DURATIONS). Live cameras' attendance must
                    # persist day to day -- this used to DELETE every row of every
                    # day, live cameras included, on every lap of any test clip.
                    removed = clear_sightings_for_cameras(conn, CAM_DURATIONS)
                    logger.info("%s looped back to the start of its recording -- removed %d "
                                "test-recording sighting row(s) for a fresh lap (live cameras' "
                                "rows kept)", wrapped_cam, removed)
                    self.engine.clear()
                    self.violations.clear()
                    self.seats.clear()
                    self.scene = {}
                    try:
                        self._on_flush([])
                    except Exception:
                        logger.exception("on_flush callback failed (reset)")

                cycle_had_failure = False
                cycle_changed = False  # any sighting written/extended this cycle
                # Desk evidence for this cycle, unioned across cameras:
                # employee -> True if in a seat zone on ANY camera (Q2/union).
                desk_evidence: dict[str, bool] = {}
                # Restricted-zone violation evidence, unioned across cameras
                # the same way (2026-09-02): debounce-key -> ZoneEvidence.
                violation_evidence: dict[str, ZoneEvidence] = {}
                # role lookup, once per cycle -- small table, cheap to requery
                # fresh each time rather than caching (an admin's role edit
                # takes effect on the very next cycle, no invalidation to get
                # wrong).
                roles = {r["id"]: r["role"] for r in conn.execute("SELECT id, role FROM employee").fetchall()}
                evidence_clock: datetime | None = None
                # Share one ~200 ms wait across the cameras actually streaming
                # (2026-09-13): each camera's read blocks up to block_ms when it
                # has nothing new, and with live cameras now counted, a fixed
                # 200 ms each would stretch every cycle -- and with it the
                # table's live push (FLUSH_EVERY_N_CYCLES) -- by 200 ms per camera.
                active_cams = sum(1 for c in CAM_FILENAMES if c in stream_starts)
                block_ms = max(20, 200 // max(1, active_cams))
                for cam_id in CAM_FILENAMES:
                    # Skip cameras that aren't streaming BEFORE the blocking
                    # read: each idle camera otherwise costs a full block_ms
                    # per cycle (7 idle cams = ~1.4s), which throttled the
                    # whole loop -- websocket pushes, debounce granularity and
                    # the motion metric (found live 2026-09-01: one motion
                    # sample per ~2s cycle made every track read as moving).
                    if cam_id not in stream_starts:
                        continue
                    try:
                        raw_entries = reader.read_new_entries(cam_id, block_ms=block_ms)
                    except Exception:
                        cycle_had_failure = True
                        now_mono = time.monotonic()
                        # At most one full traceback per 30s, however many
                        # cameras/cycles fail in between -- still visible in
                        # the log that something's wrong, without flooding it.
                        if now_mono - last_failure_logged_at > 30.0:
                            last_failure_logged_at = now_mono
                            logger.exception("read_new_entries failed for %s "
                                            "(further failures logged at most once/30s while this persists)", cam_id)
                        continue

                    started_at = stream_starts.get(cam_id)
                    if started_at is None:
                        continue  # not streaming yet -- nothing to timestamp

                    # 2026-09-12 (issue 1): every sighting is timestamped with
                    # real time -- the camera's own clock when it sends one
                    # over RTCP (Phase 2, below), else this server's (Phase
                    # 1). The whole SPIKE_DAY/RECORDING_START simulation -- treat
                    # elapsed-time-since-stream-start as "position inside a
                    # looping recording", then offset that onto one shared
                    # fake date -- only ever made sense because every CHIRLA
                    # camera shared that one deliberately-fake day by design.
                    # It breaks for anything with a real date (2026-09-11's
                    # cam-20/21 saga) and has no meaning at all for a live
                    # camera with no loop or duration (issue 6, the actual
                    # point of this change). server.py's day-default queries
                    # were switched to date.today() in the same pass, so the
                    # table's own filtering stays consistent with this.
                    # Real time when THIS batch was received is close enough
                    # to "when it happened" -- the whole pipeline runs at
                    # ~10fps, so processing lag here is sub-second,
                    # negligible at our minute-bucket granularity. Computed
                    # even with no fresh face_rec entries this cycle -- the
                    # presence-touch pass below needs it too.
                    # Phase 2: camera_clock holds (camera - server) seconds,
                    # learned from the RTCP time the sidecar attaches to each
                    # frame (see acl.CameraClock); None = this camera sent no
                    # clock recently -> server time, the agreed fallback.
                    cam_offset = camera_clock.offset(cam_id)
                    wall_clock = (datetime.fromtimestamp(time.time() + cam_offset)
                                  if cam_offset is not None else datetime.now())

                    # Read this cycle's present person tracks FIRST and log
                    # every track's earliest continuous appearance -- so by
                    # the time a face match lands below, the tracker already
                    # knows how long that body has been in frame (back-
                    # turned-since-entry case, 2026-09-01). Boxes feed the
                    # collision guard (id-swap case, same day).
                    meta_frame = meta_reader.read_present_person_tracks(cam_id)
                    present_track_ids = None
                    if meta_frame is not None:
                        present_track_ids, person_boxes = meta_frame
                        presence_tracker.note_present_tracks(cam_id, present_track_ids, wall_clock, person_boxes)

                    face_confirmed_tracks: set[int] = set()
                    for entry in raw_entries:
                        # Issue 4 (2026-09-12): only names the sidecar's vote
                        # lock has confirmed reach the table -- the same rule
                        # as the live label, so the two can't disagree and one
                        # stray frame can't log a sighting (precision over
                        # recall, as decided).
                        if not entry.confirmed:
                            continue
                        employee = coalescer.resolve(entry)
                        if employee is None:
                            continue
                        # Identity uniqueness (2026-09-01): the name goes to the
                        # strongest claim across ALL cameras; a losing claim is NOT
                        # recorded as a sighting either -- otherwise the table
                        # would credit a body the binding just refused.
                        if not presence_tracker.claim(cam_id, entry.parent_id, employee.id, entry.confidence):
                            continue
                        coalescer.record(cam_id, employee, wall_clock)
                        if entry.parent_id is not None:
                            face_confirmed_tracks.add(entry.parent_id)
                        cycle_changed = True
                        # Retroactive back-turned credit: if this track's
                        # body has been continuously in frame since before
                        # this (possibly first-ever) face match, pull the
                        # sighting's start back to the body's arrival.
                        track_first = presence_tracker.first_seen_of_track(cam_id, entry.parent_id)
                        if track_first is not None and coalescer.backdate_first_seen(
                                employee.id, cam_id, track_first):
                            logger.info("backdated %s on %s to %s (body tracked before first face match)",
                                        employee.id, cam_id, track_first.isoformat())

                    # Back-turned bridge (Q3/point-5, 2026-08-31): extend any
                    # already-matched employee's sighting purely from their
                    # person-track still being present in iris:meta's `dets`,
                    # even on a cycle with zero face_rec entries at all --
                    # see PersonPresenceTracker for why this is safe (tracks
                    # a track's own presence continuity, not a fixed timer).
                    # (present_track_ids was read above, before the ingest loop.)
                    if present_track_ids is not None:
                        for employee_id in presence_tracker.still_present_employees(cam_id, present_track_ids):
                            coalescer.touch_presence(employee_id, cam_id, wall_clock)
                            cycle_changed = True

                        # ── Productivity: seat-zone evidence + overlay scene ──
                        cam_zones = self.zones.load().get(cam_id, [])
                        # Zones now mix two types (2026-09-02: seat and
                        # restricted) in the same list -- each side must only
                        # ever see its own kind, or a restricted zone would
                        # wrongly credit desk time, or a seat zone would
                        # wrongly gate a violation.
                        cam_seat_zones = seat_zones(cam_zones)
                        # the zones drawn on this camera, or the whole stream if it has an access check and none
                        cam_restricted = {z["id"]: z for z in effective_restricted_zones(
                            cam_id, cam_zones, self.attendance_policy.load().get("access_control"))}
                        seated_max_aspect = float(self.policy.load().get("seated_max_aspect", 2.1))
                        scene: list[dict] = []
                        bodies: list[ZoneBody] = []
                        for track_id in present_track_ids:
                            box = person_boxes.get(track_id)
                            employee_id = presence_tracker.employee_for_track(cam_id, track_id)
                            zone_id = zone_hit(box, cam_seat_zones) if box else None
                            moving = presence_tracker.is_moving(cam_id, track_id)
                            # seated-shaped: squat box (see policy seated_max_aspect)
                            seated_shape = bool(box) and box[2] > 0 and (box[3] / box[2]) <= seated_max_aspect
                            bodies.append(ZoneBody(track_id=track_id, zone_id=zone_id, employee_id=employee_id,
                                                   stationary=not moving,
                                                   face_confirmed=track_id in face_confirmed_tracks,
                                                   seated=seated_shape))
                            scene.append({
                                "track_id": track_id,
                                "employee_id": employee_id,
                                "zone_id": zone_id,
                                "moving": moving,
                                "seated": seated_shape,
                                "box": ([box[0] / FRAME_W, box[1] / FRAME_H, box[2] / FRAME_W, box[3] / FRAME_H]
                                        if box else None),
                            })

                            # ── Restricted zones: role/unidentified violations ──
                            restricted_id = zone_hit(box, list(cam_restricted.values())) if box else None
                            if restricted_id is not None:
                                zone = cam_restricted[restricted_id]
                                if employee_id is None:
                                    # No stable identity -- key and debounce
                                    # per track (Q4, round 2: the more
                                    # important alert, must not require a
                                    # known identity to fire).
                                    key = f"track:{cam_id}:{track_id}"
                                    violation_evidence[key] = ZoneEvidence(
                                        key=key, camera=cam_id, zone_id=restricted_id,
                                        employee_id=None, violation_type=UNIDENTIFIED)
                                elif not role_allowed(roles.get(employee_id), zone):
                                    key = f"role:{employee_id}"
                                    violation_evidence[key] = ZoneEvidence(
                                        key=key, camera=cam_id, zone_id=restricted_id,
                                        employee_id=employee_id, violation_type=ROLE_NOT_ALLOWED)
                        # Seat occupancy decides who is credited for each seat
                        # (identity anchored to the seat, survives occlusion,
                        # back-turned sitting and track-id churn -- see
                        # productivity.SeatOccupancy).
                        credited = self.seats.occupant_credits(cam_id, bodies, time.monotonic())
                        for employee_id in credited:
                            desk_evidence[employee_id] = True
                        occupant_by_zone = {z: e for e, z in credited.items()}
                        for entry in scene:
                            z = entry["zone_id"]
                            entry["credited_as"] = (occupant_by_zone.get(z)
                                                    if z is not None and not entry["moving"] and entry["seated"] else None)
                        self.scene[cam_id] = scene
                        evidence_clock = wall_clock

                # Identity is global (one name <-> one body across ALL
                # cameras, user requirement 2026-09-01): a seat held only by
                # continuity on camera A is released once its occupant's live
                # binding is seen on camera B -- they can't be in both rooms.
                released = self.seats.release_where_elsewhere(presence_tracker.binding_sources())
                for employee_id in released:
                    desk_evidence.pop(employee_id, None)

                # One debounced state evaluation per cycle over the union of
                # every camera's evidence (skipped when no camera had fresh
                # meta this cycle -- no information is not "everyone left").
                if evidence_clock is not None and self.engine.observe(desk_evidence, evidence_clock):
                    cycle_changed = True
                if evidence_clock is not None and self.violations.observe(violation_evidence, evidence_clock):
                    cycle_changed = True

                # Push the fresh snapshot the moment anything landed in the
                # table, not when the bucket eventually closes. Found live
                # 2026-08-31: the websocket only fired from flush_due_buckets,
                # whose stale window (8s) only starts counting once the person
                # LEAVES the frame -- someone visible for 10s reached the
                # table ~20s after first appearing. The DB write was always
                # immediate; only this notification lagged. Snapshot is a
                # handful of rows, so one broadcast per changed cycle (~1-2s)
                # is nothing.
                if cycle_changed:
                    try:
                        self._on_flush([])
                    except Exception:
                        logger.exception("on_flush callback failed (live update)")

                if cycle_had_failure:
                    consecutive_failures += 1
                    # 2s, 4s, 6s, ... capped at 10s -- long enough to stop
                    # hammering a downed Redis, short enough to notice it come
                    # back quickly once it does.
                    time.sleep(min(2.0 * consecutive_failures, 10.0))
                else:
                    consecutive_failures = 0

                cycles += 1
                if cycles >= FLUSH_EVERY_N_CYCLES:
                    cycles = 0
                    flushed = coalescer.flush_due_buckets()
                    if flushed:
                        logger.info("flushed %d sighting(s)", len(flushed))
                        try:
                            self._on_flush(flushed)
                        except Exception:
                            logger.exception("on_flush callback failed")
            except Exception:
                logger.exception("consumer cycle failed unexpectedly -- continuing (thread must not die)")
                time.sleep(2.0)
