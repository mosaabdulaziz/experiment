"""Preview clips for fired alerts.

An alert fires at pts T, but the footage covering T does not exist as a fetchable
object yet -- the segment recorder rotates every `record_seconds`, so the segment
containing T is not closed, stored and indexed until up to that long afterwards.
Clips are therefore built by a deferred worker, never synchronously at fire time.

The pure helpers below carry the decisions worth testing on their own; the service
that drives them lives in the same module because they have no other consumer.
"""
import asyncio
import logging
import os
import tempfile
import threading
import time

import cv2

from .clip_builder import (apply_overlay, apply_overlays, build_preview_clip, extract_frame, probe_dimensions,
                           probe_duration,
                           remux_to_mp4)
from .frame_extract import zone_overlay_png, zones_overlay_png
from .overlay_track import (CLIP_HOLD_S, HOLD_S, REACH_S, clip_style, draw, merge_pairs, nearest, pick_clip,
                            read_entries, select, timeline, write_sequence)

logger = logging.getLogger("iris.alertclip")
# Smart-record clips drawn on at once (each a full libx264 re-encode in a worker thread): more wait their turn.
_SVR_ENCODES = threading.BoundedSemaphore(2)

SEGMENT_LOOKBACK_S = 120.0     # a segment indexed this recently, holding frame times this near, is this run's


def _clip_start_pts(refs: list, start: float) -> "float | None":
    """The frame time of a clip's first frame: from the segment holding its start (its wall label and its own frame
    time), or None when the index gave no frame times."""
    ref = next((r for r in refs if r.start_pts - 0.5 <= start < r.end_pts), refs[0] if refs else None)
    if ref is None or getattr(ref, "pts_start", None) is None:
        return None
    return ref.pts_start + (start - ref.start_pts)


def find_coalescible(jobs: list, sid: str, pts: float, coalesce_seconds: float,
                     zone_id: "str | None" = None, draw_zone: bool = True):
    """The queued job on `sid` and `zone_id` (drawn or not, as `draw_zone`) whose anchor is
    nearest `pts` and within `coalesce_seconds`, or None.

    Ten people crossing a zone inside one segment length would otherwise mean ten
    near-identical ffmpeg builds of the same footage; instead they attach to one
    job and share one clip id. Only *building* jobs are excluded -- their footage is
    being cut, so a late arrival must get its own clip. "awaiting-svr" is a waiting
    state like "queued", not a building one: matching on "queued" alone meant SVR
    jobs never coalesced at all, so every alert of a burst opened its own job for the
    one smart-record file they all shared.

    Coalescing is scoped to one zone because the shared clip draws the zone that
    fired: alerts from two zones on one clip would leave some of them pointing at an
    outline they did not fire on. Two zoneless alerts (proactive VLM alerts) still
    coalesce with each other, as they always did. A kind whose clips leave the zone out
    (proximity) never shares a clip with one that draws it.
    """
    best, best_distance = None, None
    for job in jobs:
        if job.get("append"):
            continue                # a follow-up clip belongs to one event only
        if job["sid"] != sid or job["state"] not in ("queued", "awaiting-svr"):
            continue
        if (job.get("zone") or {}).get("id") != zone_id or job.get("draw_zone", True) != draw_zone:
            continue
        distance = abs(pts - job["anchor"])
        if distance > coalesce_seconds:
            continue
        if best_distance is None or distance < best_distance:
            best, best_distance = job, distance
    return best


def segments_cover(refs: list, lo: float, hi: float, epsilon: float = 0.5) -> bool:
    """True only when `refs` span [lo, hi] with no gap.

    A hole from a sidecar restart or a dropped upload returns False, so the worker
    keeps waiting rather than concatenating across a discontinuity. Small jitter
    below `epsilon` (e.g. keyframe snapping or sub-second rotation seam) is bridged.
    """
    reach = None
    for ref in sorted(refs, key=lambda s: s.start_pts):
        if reach is None:
            if ref.start_pts > lo + epsilon:
                return False
            reach = ref.end_pts
        elif ref.start_pts > reach + epsilon:
            return False
        else:
            reach = max(reach, ref.end_pts)
        if reach >= hi - epsilon:
            return True
    return False


def contiguous_run_containing(refs: list, t: float, epsilon: float = 0.5) -> list:
    """The maximal gap-free run of `refs` whose span contains `t`, or [].

    Used when the wait deadline expires with only partial footage: a short clip
    beats none, but only if it actually holds the alert moment and has no
    discontinuity inside it. Overlapping and nested refs are folded into the same
    run -- the run's right edge is the furthest end seen so far, not the last
    element's own end.
    """
    run: list = []
    reach = None
    for ref in sorted(refs, key=lambda s: s.start_pts):
        if run and ref.start_pts > reach + epsilon:
            if run[0].start_pts <= t <= reach:
                return run
            run, reach = [], None
        run.append(ref)
        reach = ref.end_pts if reach is None else max(reach, ref.end_pts)
    if run and run[0].start_pts <= t <= reach:
        return run
    return []



class AlertClipService:
    """Builds a preview clip for each fired alert and attaches it to the event.

    Runs its own worker rather than sharing the VLM worker: clip building is
    ffmpeg plus disk/network IO, and the VLM worker is a single serialised
    inference slot, so parking a multi-second concat in it would stall inference
    behind IO for no benefit.
    """

    def __init__(self, config, segment_store, clip_store, events_store, conn_manager,
                 build=build_preview_clip, redis_client=None):
        cfg = (config.get("alerts", {}) or {}).get("clip", {}) or {}
        self._segments = segment_store
        self._clips = clip_store
        self._events = events_store
        self._conn = conn_manager
        self._build_fn = build
        self._redis = redis_client      # reads iris:overlay:<camera> for the lines drawn on clips; None = zone only
        self.snapshots: list = []       # incidents waiting for their clips before their snapshot is taken
        self.enabled = bool(cfg.get("enabled", False)) and segment_store is not None
        self._pad = float(cfg.get("pad_seconds", 5.0))
        self._coalesce = float(cfg.get("coalesce_seconds", 2.5))
        self._max_wait = float(cfg.get("max_wait_seconds", 20.0))
        self._depth = max(int(cfg.get("queue_depth", 8)), 1)
        self._draw_zones = bool(cfg.get("draw_zones", True))
        # alert kinds whose clips leave the zone out (a proximity clip shows the pair's masks and line instead)
        self._zone_free = frozenset(cfg.get("zone_free_kinds", ("proximity",)) or ())
        # Drawing on smart-record clips (zones, the alerts' objects): the safety product's, which switches it on
        # (app/server.py); every other product's smart-record clips are stored as recorded.
        self.svr_overlays = bool(cfg.get("svr_overlays", False))
        # An overlay entry is written once its frame's inference is done: the boxes of a file's last frames come up to
        # that long after it ends, so the drawing waits for them.
        self._overlay_settle = float(cfg.get("overlay_settle_seconds", 1.5))
        if self.enabled:
            if self._pad <= 0:
                raise ValueError(
                    f"alerts.clip.pad_seconds must be positive, got {self._pad!r} -- a "
                    "non-positive pad makes every clip window empty and silently "
                    "resolves every alert 'unavailable'")
            if self._coalesce < 0:
                raise ValueError(
                    f"alerts.clip.coalesce_seconds must not be negative, got {self._coalesce!r}")
            if self._max_wait < 0:
                raise ValueError(
                    f"alerts.clip.max_wait_seconds must not be negative, got {self._max_wait!r}")
        self.jobs: list = []
        self._running = False
        # Sources whose clips arrive via the sidecar's sr-done announcement rather
        # than the segment index. Registered by StreamManager once it knows each
        # stream's recorder kind; smart-record writes nothing continuously, so these
        # jobs would otherwise wait out max_wait_seconds and resolve 'unavailable'.
        self._svr_sources: set = set()
        # Strong refs to in-flight _resolve tasks. asyncio only holds a weak
        # reference to a task, so one created and dropped can be garbage-collected
        # mid-await -- the event would silently stay 'pending'.
        self._resolve_tasks: set = set()

    def mark_svr_source(self, source_id: str) -> None:
        """Record that this source's clips come from smart-record, not segments."""
        self._svr_sources.add(source_id)

    async def submit(self, event_id: int, source_id: str, pts, zone=None, wall_time=None,
                     append: bool = False, pair=None, kind: "str | None" = None) -> bool:
        """Attach `event_id` to a coalescible queued job, or open a new one.

        Resolves the event to 'unavailable' immediately when clips are off, the
        anchor is unknown, or the queue is saturated -- no event is ever left on
        'pending'. Returns True when the event is now owned by a job.

        `anchor` is derived from `wall_time` if available (real-world time),
        falling back to `pts`.

        `append`: a follow-up clip of an open incident (its escalation), added to
        the event's clips; never shared, never its first clip, and nothing on the
        event changes when it fails.

        `pair`: a proximity incident's (person ids, vehicle ids): the clip draws
        that pair's masks, boxes and line for its whole length (overlay_track.select).

        `kind`: the alert's kind; a kind in alerts.clip.zone_free_kinds gets no zone outline.
        """
        anchor = wall_time if wall_time is not None else pts
        if wall_time is not None and pts is not None and source_id not in self._svr_sources:
            on_segments = await asyncio.to_thread(self._segment_anchor, source_id, pts)
            if on_segments is not None:
                anchor = on_segments
        if append and (not self.enabled or anchor is None or source_id in self._svr_sources):
            # smart-record keeps one recording per session; a follow-up never marks the event unavailable
            logger.info("[%s] no follow-up clip for event %s", source_id, event_id)
            return False
        if not self.enabled or anchor is None:
            logger.info("[%s] no clip for event %s: %s", source_id, event_id,
                        "alert clips disabled" if not self.enabled else "alert has no time anchor")
            await self._resolve([event_id], "unavailable", None)
            return False

        zone_id, draw_zone = (zone or {}).get("id"), kind not in self._zone_free
        job = None if append else find_coalescible(self.jobs, source_id, anchor, self._coalesce, zone_id,
                                                   draw_zone)
        if job is not None:
            job["event_ids"].append(event_id)
            job["pair"] = merge_pairs(job.get("pair"), pair)
            if pair is None:
                job["everyone"] = True          # an alert with no subject shares it (read by the smart-record path)
            logger.info("[%s] event %s coalesced onto the clip job anchored at %.3f",
                        source_id, event_id, job["anchor"])
            return True

        if len(self.jobs) >= self._depth:
            # Reject the newcomer rather than dropping the oldest: queued jobs are
            # closer to having their footage and sit on segments that expire
            # sooner, and the earliest alert of a burst is usually the one that
            # matters.
            logger.warning("alert clip queue full (%d) — no clip for event %s",
                           self._depth, event_id)
            if not append:
                await self._resolve([event_id], "unavailable", None)
            return False

        self.jobs.append({
            "sid": source_id,
            "anchor": anchor,
            "pts": pts,
            "wall_time": wall_time,
            "zone": zone,
            "draw_zone": draw_zone,
            "lo": anchor - self._pad,
            "hi": anchor + self._pad,
            "event_ids": [event_id],
            "pair": pair,
            "append": append,
            "state": "awaiting-svr" if source_id in self._svr_sources else "queued",
            "deadline": time.monotonic() + self._max_wait,
        })
        logger.info("[%s] clip job queued for event %s: window %.3f-%.3f, waiting up to %gs",
                    source_id, event_id, anchor - self._pad, anchor + self._pad, self._max_wait)
        return True


    def attach_svr_file(self, source_id: str, path: str, recorded_at: "float | None" = None) -> bool:
        """Resolve every outstanding job on this source with one smart-record file.

        Correlation needs no session id: the sidecar's smart-rec-status guard permits
        only one recording per source at a time, so an sr-done on a given source can
        only be the session its alert started.

        That guard is also why ALL of the source's jobs resolve from this one file and
        SHARE one clip. Alerts landing inside a running session are deliberately given
        no recording of their own -- SvrTrigger returns coalesce or skip -- so this
        file is the only footage they will ever have, and it already covers them.

        The clip is stored exactly ONCE. ClipStore.add MOVES the file, so storing it
        per job found the path gone on the second job; since the job was dequeued
        first, nothing was left to resolve its events or expire them at the deadline,
        and they sat on 'pending' for the life of the process. Nothing here may leave
        an event pending: a store failure resolves 'unavailable' instead.

        Called from the Redis consumer task, so it schedules the resolve rather than
        awaiting it. Dequeuing stays SYNCHRONOUS while the rewrap and store do not:
        that ordering is what keeps "stored exactly once" true, because a second
        sr-done arriving mid-rewrap finds no jobs left to claim.

        `recorded_at` (the sidecar's sr-done time: the file's end) places the overlay
        stream's boxes on the file -- every job's object (_maybe_overlay_svr).
        """
        jobs = [j for j in self.jobs if j["sid"] == source_id]
        if not jobs:
            logger.warning("[%s] smart-record file %s arrived with no queued alert job",
                           source_id, os.path.basename(path))
            return False
        # jobs[0] is the earliest submitted, i.e. the alert that actually started this
        # session -- so its window is the one the recording was cut around.
        anchor, lo, hi = jobs[0]["anchor"], jobs[0]["lo"], jobs[0]["hi"]
        event_ids = [eid for job in jobs for eid in job["event_ids"]]
        pair = None
        for job in jobs:
            pair = merge_pairs(pair, job.get("pair"))
            self.jobs.remove(job)
        if any(job.get("pair") is None or job.get("everyone") for job in jobs):
            pair = None                 # an alert about everyone (overcrowding, no subject) shares the file: all drawn
        # the zones the clip shows: every job's (the sidecar's session is per camera, so alerts in different zones
        # can share one file -- and the one clip each of them gets)
        zone_jobs = list({self._zone_key(job): job for job in jobs if self._draws_zone(job)}.values())
        task = asyncio.create_task(
            self._store_and_resolve(source_id, path, anchor, lo, hi, event_ids, recorded_at, pair, zone_jobs))
        self._resolve_tasks.add(task)
        task.add_done_callback(self._on_resolve_done)
        return True

    async def store_standalone(self, source_id: str, path: str,
                               recorded_at: "float | None" = None) -> "tuple[str, str | None]":
        """Rewrap + store a smart-record file with no queued alert job waiting on it.

        Happens for a fire/smoke incident's follow-up clips: the first clip already
        went through submit()/attach_svr_file() and resolved its own event, but the
        incident is still open (see sidecar.py's has_open_incident re-arm), so later
        sr-done files arrive with nothing queued for this source. The caller is
        responsible for attaching the returned clip_id to whichever incident is
        still open -- this method only knows how to turn a file into a stored clip.
        Returns `(status, clip_id)`, same contract as the queued path; never raises.
        """
        return await asyncio.to_thread(
            self._store_svr_recording, source_id, path, None, None, None, [], recorded_at)

    async def _store_and_resolve(self, source_id, path, anchor, lo, hi, event_ids, recorded_at=None, pair=None,
                                 zone_jobs=()):
        """Rewrap + store off the event loop, then resolve the events.

        The rewrap is an ffmpeg subprocess (~90ms for one recording) and the store
        is a file move, both blocking. They run on the loop that serves SSE chat
        tokens and WS broadcasts, so they go to a thread -- the same reason clip
        building has its own worker rather than running inline.
        """
        try:
            status, clip_id = await asyncio.to_thread(
                self._store_svr_recording, source_id, path, anchor, lo, hi, event_ids, recorded_at, pair, zone_jobs)
        except Exception:
            # _store_svr_recording is written not to raise, but it is now behind a
            # task boundary where an escape would be swallowed and the events would
            # sit on 'pending' for the life of the process. Resolve them anyway.
            logger.error("[%s] storing the smart-record file %s raised — resolving "
                         "events %s unavailable", source_id, path, event_ids,
                         exc_info=True)
            status, clip_id = "unavailable", None
        await self._resolve(event_ids, status, clip_id)

    def _store_svr_recording(self, source_id, path, anchor, lo, hi, event_ids, recorded_at=None, pair=None,
                             zone_jobs=()):
        """Rewrap the recording and hand it to the clip store. Blocking; returns
        `(clip_status, clip_id)` and never raises -- an event left 'pending' never
        clears, so every failure path here must still name a terminal status."""
        status, clip_id = "unavailable", None
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Smart Record writes Matroska -- see clip_builder.remux_to_mp4 for why
            # MP4 is not available on the sidecar side. Rewrap (stream copy) before
            # storing: ClipStore names every clip `<camera>.mp4` and the browser
            # plays it straight from disk, so Matroska bytes under an .mp4 name
            # would store a clip no <video> element can play. The suffix check
            # keeps an app newer than its sidecar working, the same version-skew
            # tolerance this path already practises elsewhere.
            playable = remux_to_mp4(path, tmp_dir) if path.endswith(".mkv") else path
            if playable is None:
                logger.error("[%s] smart-record file %s could not be rewrapped — no "
                             "clip for events %s", source_id, path, event_ids)
            else:
                playable = self._maybe_overlay_svr(playable, source_id, recorded_at, pair, zone_jobs, tmp_dir)
                try:
                    clip_id = self._clips.add(
                        {source_id: playable},
                        meta={"camera": source_id, "anchor": anchor,
                              "range": [lo, hi], "source": "svr"})
                except Exception:
                    logger.error("[%s] could not store smart-record file %s — no clip "
                                 "for events %s", source_id, path, event_ids,
                                 exc_info=True)
                    clip_id = None
                else:
                    status = "ready"
                    logger.info("[%s] alert clip %s attached from smart-record for "
                                "events %s", source_id, clip_id, event_ids)
                    # ClipStore.add MOVES what it is given, so on the pre-remux path
                    # the recording left svr_dir by itself. It no longer does: the
                    # move takes the temp copy and the original would stay forever,
                    # which is exactly how a 30-minute retention once accumulated
                    # three weeks of segments. Only the startup flush would clear it.
                    if playable != path:
                        try:
                            os.remove(path)
                        except OSError as exc:
                            logger.warning("[%s] could not delete the recording %s "
                                           "after rewrapping: %s", source_id, path, exc)
        return status, clip_id

    def _maybe_overlay_svr(self, path: str, sid: str, recorded_at: "float | None", pair, zone_jobs,
                           tmp_dir: str) -> str:
        """A smart-record file with its alerts' zones (`zone_jobs`', as the zone editor draws them) and the overlay stream's
        boxes burned in: the alerts' objects (`pair`, merged over the jobs sharing the file; all candidates without
        one), fire and smoke, proximity's lines. The boxes are placed by wall time -- the file ends at `recorded_at`
        (sr-done) and lasts its duration, each entry at its frame's arrival -- as a smart-record file carries no frame
        times of the pipeline. The safety product's only (svr_overlays). Total: any failure keeps the plain file."""
        if not self.svr_overlays:
            return path
        try:
            dims = probe_dimensions(path)
            if dims is None:
                return path
            png = None
            if zone_jobs:
                png = zones_overlay_png(
                    [(job["zone"]["points"], (job["zone"].get("intrusion") or {}).get("mode", "restricted"))
                     for job in zone_jobs], dims[0], dims[1], os.path.join(tmp_dir, "zone.png"))
            timed, drawn = None, 0
            duration = probe_duration(path) if self._redis is not None and recorded_at is not None else None
            if duration:
                start = recorded_at - duration
                late = recorded_at + self._overlay_settle - time.time()
                if late > 0:                                # the last frames' boxes are still being written
                    time.sleep(min(late, self._overlay_settle))
                # written up to the inference delay after their frames arrived: read past the file's end, place by
                # arrival
                entries = read_entries(self._redis, sid, None, None, start - CLIP_HOLD_S, recorded_at + REACH_S)
                placed = [{**e, "pts": e["wall"]} for e in (select(e, pair) for e in entries) if e is not None]
                if placed:
                    segments = timeline(placed, start, duration, CLIP_HOLD_S)
                    timed = write_sequence(segments, dims[0], dims[1], tmp_dir, style=clip_style(self._redis))
                    drawn = sum(1 for s in segments if s[2] is not None)
            if png is None and timed is None:
                return path
            with _SVR_ENCODES:                              # a full re-encode each: a burst queues, not piles up
                out = apply_overlays(path, png, timed, tmp_dir)
            if out is None:
                return path
            logger.info("[%s] drew %s%d overlay frame(s) on the smart-record clip", sid,
                        "its zone and " if png else "", drawn)
            return out
        except Exception:
            logger.warning("[%s] overlay on the smart-record clip failed — keeping it plain", sid, exc_info=True)
            return path

    def _on_resolve_done(self, task) -> None:
        """Drop the strong ref, and never let a resolve failure pass silently -- a
        raised _resolve leaves its events on 'pending' with no second chance."""
        self._resolve_tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logger.error("alert clip resolve failed", exc_info=task.exception())

    async def _resolve(self, event_ids: list, status: str, clip_id):
        for event_id in event_ids:
            record = await self._events.update_clip(event_id, status, clip_id)
            if record is None:
                continue        # the event was cleared while its clip was building
            # the whole clip list too: an incident's escalation clip may have been appended before this one resolved
            await self._conn.broadcast({"type": "event_update", "id": event_id,
                                        "clip_status": status, "clip_id": clip_id, "clips": record["clips"]})

    def snapshot(self, event_id: int, source_id: str, at_pts: float, person_ids: list, vehicle_ids: list) -> None:
        """Take the incident's snapshot at `at_pts` (frame time) once its clips are built; never raises."""
        if not self.enabled or at_pts is None:
            return
        self.snapshots.append({"event_id": event_id, "sid": source_id, "at_pts": at_pts,
                               "pair": (set(person_ids), set(vehicle_ids)),
                               "deadline": time.monotonic() + self._max_wait + 30.0})

    async def _take_snapshots(self) -> None:
        """Each queued snapshot whose event no longer waits for a clip (or that waited long enough)."""
        for snap in list(self.snapshots):
            try:
                await self._take_snapshot(snap)
            except Exception:
                # one snapshot's failure (a database hiccup, a bad frame) never holds up the others
                logger.warning("[%s] snapshot failed for event %s", snap["sid"], snap["event_id"], exc_info=True)
                if snap in self.snapshots:
                    self.snapshots.remove(snap)

    async def _take_snapshot(self, snap: dict) -> None:
        event = await self._events.get_event(snap["event_id"])
        waiting = event is not None and (event.get("clip_status") == "pending"
                                         or any(snap["event_id"] in j["event_ids"] for j in self.jobs))
        if waiting and time.monotonic() < snap["deadline"]:
            return
        self.snapshots.remove(snap)
        if event is None:
            return
        taken = await asyncio.to_thread(self._make_snapshot, snap, event)
        if taken is None:
            return
        record = await self._events.set_snapshot(snap["event_id"], taken[0], taken[1])
        if record is not None:
            await self._conn.broadcast({"type": "event_update", "id": snap["event_id"], "event": record})

    def _make_snapshot(self, snap: dict, event: dict):
        """(clip id, exact) of the stored snapshot, or None (no usable clip, or the frame could not be read)."""
        ids = [c["clip_id"] for c in event.get("clips") or []] or (
            [event["clip_id"]] if event.get("clip_id") else [])
        metas = [(cid, rec["meta"]) for cid in ids if (rec := self._clips.get(cid)) is not None]
        picked = pick_clip(metas, snap["at_pts"])
        if picked is None:
            return None
        clip_id, offset, exact = picked
        path = self._clips.get_clip_path(clip_id, snap["sid"])
        if path is None:
            return None
        with tempfile.TemporaryDirectory() as tmp:
            jpg = os.path.join(tmp, "snapshot.jpg")
            if not extract_frame(path, offset, jpg):
                return None
            entry = None
            if self._redis is not None:
                lo, hi = (float(v) for v in self._clips.get(clip_id)["meta"]["range"])
                wall = (lo - REACH_S, hi + REACH_S) if lo > 1e8 else (None, None)     # a wall-clock clip
                entry = nearest(read_entries(self._redis, snap["sid"], snap["at_pts"], HOLD_S, *wall), snap["at_pts"])
            entry = None if entry is None else select(entry, snap["pair"])
            if entry is not None:
                img = cv2.imread(jpg)
                if img is not None:
                    cv2.imwrite(jpg, draw(img, entry, clip_style(self._redis)))
            if not self._clips.add_snapshot(clip_id, snap["sid"], jpg, snap["event_id"]):
                return None
        return clip_id, exact

    async def _resolve_job(self, job, status: str, clip_id):
        """Resolve a job's events: their clip, or for a follow-up job an extra clip."""
        if job.get("append"):
            await self._append(job["event_ids"], status, clip_id)
        else:
            await self._resolve(job["event_ids"], status, clip_id)

    async def _append(self, event_ids: list, status: str, clip_id):
        """A follow-up clip (an incident's escalation) joins the event's clips; a failure changes nothing."""
        if status != "ready" or clip_id is None:
            return
        for event_id in event_ids:
            record = await self._events.append_clip(event_id, clip_id)
            if record is not None:
                await self._conn.broadcast({"type": "event_update", "id": event_id, "clips": record["clips"]})

    async def run(self):
        """Worker loop: build each job once its footage exists, or once it times out.

        Scans queued jobs rather than blocking on the head of the list, so one
        stream whose recorder is lagging cannot hold up another's clip. Builds are
        serialised -- one ffmpeg at a time is intended. Nothing is allowed to
        escape this loop: a dead worker would strand every later alert on
        'pending' with no restart path.
        """
        if not self.enabled:
            logger.info("alert clip worker not started (alerts.clip.enabled is off, or "
                        "there is no segment store) — alerts will resolve 'unavailable'")
            return
        logger.info("alert clip worker started: pad=%gs coalesce=%gs max_wait=%gs depth=%d",
                    self._pad, self._coalesce, self._max_wait, self._depth)
        self._running = True
        while self._running:
            job = None
            try:
                await self._take_snapshots()
                job = await self._next_ready()
                if job is None:
                    await asyncio.sleep(0.5)
                    continue
                # Close coalescing BEFORE the first await: the build suspends in an
                # executor, other tasks run during that suspension, and an alert must
                # never attach to footage that is already being cut.
                job["state"] = "building"
                await self._build(job)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.error("alert clip worker iteration failed%s",
                             f" for events {job['event_ids']}" if job else "",
                             exc_info=True)
                if job is not None:
                    try:
                        await self._resolve_job(job, "unavailable", None)
                    except Exception:
                        logger.error("alert clip resolve failed for events %s",
                                     job["event_ids"], exc_info=True)
                else:
                    # The failure was in job selection (e.g. Redis down). Back off
                    # so a persistent outage does not spin the loop.
                    await asyncio.sleep(1.0)
            finally:
                if job is not None and job in self.jobs:
                    self.jobs.remove(job)

    def stop(self):
        self._running = False

    async def _next_ready(self):
        now = time.monotonic()
        for job in list(self.jobs):
            if job["state"] == "awaiting-svr":
                # Never handed to the builder: smart-record writes no segments, so
                # segments_cover would be false forever and the deadline branch below
                # would cut a clip from footage that does not exist. The sr-done
                # announcement resolves these via attach_svr_file instead.
                if now >= job["deadline"]:
                    # It never arrived (sidecar restart, dropped Redis entry). Resolve
                    # rather than leave the event on 'pending' for the life of the
                    # process -- gate the pending on the resolver, always.
                    self.jobs.remove(job)
                    logger.warning("[%s] no smart-record file arrived within %gs for "
                                   "events %s", job["sid"], self._max_wait,
                                   job["event_ids"])
                    await self._resolve_job(job, "unavailable", None)
                continue
            if job["state"] != "queued":
                continue
            refs = await asyncio.to_thread(self._segments.list_range, job["sid"],
                                           job["lo"], job["hi"])
            if segments_cover(refs, job["lo"], job["hi"]) or now >= job["deadline"]:
                return job
        return None

    @staticmethod
    def _zone_key(job: dict):
        zone = job["zone"]
        return zone.get("id") or tuple(tuple(p) for p in zone.get("points") or ())

    def _draws_zone(self, job: dict) -> bool:
        zone = job.get("zone")
        return self._draw_zones and job.get("draw_zone", True) and bool(zone and zone.get("points"))

    def _maybe_overlay_zone(self, path: str, job: dict, tmp_dir: str) -> str:
        """Burn the fired zone onto `path`, or return `path` untouched.

        Deliberately total: every failure mode -- feature off, no zone, unreadable
        dimensions, degenerate polygon, ffmpeg failure, or an unexpected exception --
        returns the clip that was already built. The overlay is cosmetic and must never
        cost an alert its clip, so nothing is allowed to raise out of here.
        """
        zone = job.get("zone")
        if not self._draws_zone(job):
            return path
        try:
            dims = probe_dimensions(path)
            if dims is None:
                logger.warning("[%s] could not read clip dimensions — no zone overlay",
                               job["sid"])
                return path
            png = zone_overlay_png(zone["points"], dims[0], dims[1],
                                   os.path.join(tmp_dir, "zone.png"),
                                   (zone.get("intrusion") or {}).get("mode", "restricted"))
            if png is None:
                logger.warning("[%s] zone %s has no drawable polygon — no overlay",
                               job["sid"], zone.get("id"))
                return path
            overlaid = apply_overlay(path, png, tmp_dir)
            if overlaid is None:
                return path                 # apply_overlay already logged the reason
            logger.info("[%s] drew zone %s on the clip for events %s",
                        job["sid"], zone.get("id"), job["event_ids"])
            return overlaid
        except Exception:
            logger.warning("[%s] zone overlay failed — keeping the un-overlaid clip",
                           job["sid"], exc_info=True)
            return path

    def _segment_anchor(self, source_id: str, pts: float) -> "float | None":
        """The alert's moment on the clock the camera's segments are cut by (wall labels: indexed time minus duration):
        its frame time plus the label-to-frame-time offset of the camera's latest segment -- exact, where the app's
        receipt time runs seconds off, by a different amount per camera. None when no recent segment of this run holds
        a usable offset (the receipt time is used then)."""
        try:
            now = time.time()
            refs = self._segments.list_range(source_id, now - SEGMENT_LOOKBACK_S, now + 60.0)
        except Exception:
            logger.warning("[%s] segment index unreadable -- the clip is anchored on receipt time", source_id,
                           exc_info=True)
            return None
        recent = [r for r in refs if getattr(r, "pts_start", None) is not None
                  and now - (r.recorded_at or 0.0) <= SEGMENT_LOOKBACK_S]
        if not recent:
            return None
        ref = max(recent, key=lambda r: r.recorded_at)
        if abs(pts - ref.pts_start) > SEGMENT_LOOKBACK_S:      # frame times of another run (the clock restarted)
            return None
        return pts + (ref.start_pts - ref.pts_start)

    def _overlay_entries(self, job) -> list:
        """The overlay stream's entries around this job's alert, or [] (no Redis, no pts, or a read failure)."""
        if self._redis is None or job.get("pts") is None:
            return []
        # entries are written as their frames are analysed: a clip on the wall clock only takes entries written
        # around its own window (an earlier run's frame times can repeat this one's)
        wall = (job["lo"] - REACH_S, job["hi"] + REACH_S) if job.get("wall_time") is not None else (None, None)
        try:
            return read_entries(self._redis, job["sid"], job["pts"], self._pad + REACH_S, *wall)
        except Exception:
            logger.warning("[%s] could not read the overlay stream — zone only", job["sid"], exc_info=True)
            return []

    def _maybe_overlay(self, path: str, job: dict, tmp_dir: str, start: float, end: float,
                       start_pts: "float | None" = None) -> str:
        """The zone outline and the overlay stream's lines in one pass; just the zone when there are no lines.
        Total, like _maybe_overlay_zone: any failure keeps the zone-only (or plain) clip."""
        entries = [e for e in (select(e, job.get("pair")) for e in self._overlay_entries(job)) if e is not None]
        if not entries:
            return self._maybe_overlay_zone(path, job, tmp_dir)
        try:
            dims = probe_dimensions(path)
            if dims is None:
                return self._maybe_overlay_zone(path, job, tmp_dir)
            zone, png = job.get("zone"), None
            if self._draws_zone(job):
                png = zone_overlay_png(zone["points"], dims[0], dims[1], os.path.join(tmp_dir, "zone.png"),
                                       (zone.get("intrusion") or {}).get("mode", "restricted"))
            # the clip's first frame time: from its segment (exact), else from its alert (older behaviour)
            clip_pts = start_pts if start_pts is not None else job["pts"] - (job["anchor"] - start)
            segments = timeline(entries, clip_pts, end - start, CLIP_HOLD_S)
            timed = write_sequence(segments, dims[0], dims[1], tmp_dir, style=clip_style(self._redis))
            if timed is None:
                return self._maybe_overlay_zone(path, job, tmp_dir)
            out = apply_overlays(path, png, timed, tmp_dir)
            if out is None:
                return self._maybe_overlay_zone(path, job, tmp_dir)
            logger.info("[%s] drew %d overlay frame(s) on the clip for events %s",
                        job["sid"], sum(1 for s in segments if s[2] is not None), job["event_ids"])
            return out
        except Exception:
            logger.warning("[%s] overlay lines failed — keeping the zone-only clip", job["sid"], exc_info=True)
            return self._maybe_overlay_zone(path, job, tmp_dir)

    async def _build(self, job):
        loop = asyncio.get_running_loop()
        refs = await asyncio.to_thread(self._segments.list_range, job["sid"],
                                       job["lo"], job["hi"])
        if not segments_cover(refs, job["lo"], job["hi"]):
            # Deadline path: settle for the gap-free run that actually holds the
            # fired moment. A clip without its own alert in it is worthless.
            logger.warning("[%s] clip window %.3f-%.3f still not covered at the deadline "
                           "(%d segment(s) indexed) — falling back to the gap-free run "
                           "around the alert, so this clip will be short",
                           job["sid"], job["lo"], job["hi"], len(refs))
            refs = contiguous_run_containing(refs, job["anchor"])
            if not refs:
                logger.warning("[%s] no clip for events %s: no gap-free footage holds the "
                               "fired moment %.3f", job["sid"], job["event_ids"], job["anchor"])
                await self._resolve_job(job, "unavailable", None)
                return
        lo = max(job["lo"], min(r.start_pts for r in refs))
        hi = min(job["hi"], max(r.end_pts for r in refs))

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = await loop.run_in_executor(
                None, self._build_fn, self._segments, job["sid"], lo, hi,
                0.0, False, 0.0, tmp_dir, (lo, hi))
            if result is None:
                logger.warning("[%s] no clip for events %s: the builder found nothing "
                               "overlapping %.3f-%.3f", job["sid"], job["event_ids"], lo, hi)
                await self._resolve_job(job, "unavailable", None)
                return
            path, actual_start, actual_end = result
            # actual_start/end are deliberately not recomputed: the overlay pass performs
            # no cut, so the duration carries over (pinned by test_apply_overlay_*).
            # the lines (a Redis read, PNGs, an ffmpeg pass) and the store run off the event loop
            start_pts = _clip_start_pts(refs, actual_start)
            path = await asyncio.to_thread(self._maybe_overlay, path, job, tmp_dir, actual_start, actual_end,
                                           start_pts)
            clip_id = await asyncio.to_thread(
                self._clips.add, {job["sid"]: path},
                meta={"camera": job["sid"], "anchor": job["anchor"], "pts": job["pts"],
                      "range": [actual_start, actual_end], "start_pts": start_pts})

        # actual_* are measured off the built file, so a clip that came out short or
        # long shows up here as a span that disagrees with the requested window.
        logger.info("[%s] alert clip %s built %.3f-%.3f (%.2fs, wanted %.2fs) for events %s",
                    job["sid"], clip_id, actual_start, actual_end,
                    actual_end - actual_start, hi - lo, job["event_ids"])
        # Per-clip self-check: the clip's content span should match the wall window
        # the alert anchored on (recording is wall-locked 1:1). A divergence > 1s means
        # the footage covers a different stretch than the fired moment -- a lost or
        # duplicated segment, or a recorder that fell out of step with the wall clock.
        if abs((actual_end - actual_start) - (hi - lo)) > 1.0:
            logger.warning(
                "[%s] alert clip %s content span %.2fs differs from the alert window "
                "%.2fs by >1s (events %s) -- footage covers a different stretch than "
                "the fired moment; check the recorder clock for drift",
                job["sid"], clip_id, actual_end - actual_start, hi - lo, job["event_ids"])
        await self._resolve_job(job, "ready", clip_id)
