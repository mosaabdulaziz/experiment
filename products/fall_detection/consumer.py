"""The consumer: the Redis reader thread, the per-camera queues and the per-camera workers.

One worker per camera, because the monitor's tracks are keyed by the track id alone: two
cameras sharing a monitor would cross-talk (design §5.1).

    Redis consumer thread --XREAD iris:meta:<cam>--> one entry per frame
            |  parse the `pose` field -> the people
            v
    the camera's bounded queue (the oldest frame is dropped when it is full)
            |
            v
    the camera's worker: the monitor's update -> the candidates -> the policy -> the routes
"""
from __future__ import annotations

import json
import logging
import threading
import time
from queue import Empty, Full, Queue

import numpy as np
import redis

import acl
import scoring

logger = logging.getLogger("fall_detection.consumer")

# Frames a camera may fall behind by. The monitor throttles its own work (eval_every_s), so a
# stale frame is worth nothing: the queue drops it rather than growing (design §5.1).
QUEUE_MAXSIZE = 64

# A malformed frame repeats at the camera's rate. The warning is for the operator, so each kind
# is said once per interval instead of once per frame -- a per-frame line buries the log it is
# written to and says nothing the first line did not.
WARN_INTERVAL_S = 60.0


class FrameQueue:
    """A bounded queue of frame entries that drops the oldest when it is full."""

    def __init__(self, maxsize: int = QUEUE_MAXSIZE):
        self._queue: Queue = Queue(maxsize=maxsize)

    def put(self, entry) -> None:
        try:
            self._queue.put_nowait(entry)
        except Full:
            self._queue.get_nowait()          # the oldest frame: stale, and worth nothing
            self._queue.put_nowait(entry)

    def get(self, timeout: float | None = None):
        return self._queue.get() if timeout is None else self._queue.get(timeout=timeout)

    def get_nowait(self):
        return self._queue.get_nowait()

    def qsize(self) -> int:
        return self._queue.qsize()


class CameraWorker:
    """One camera: its monitor, its queue, its routes and its aftermath bookkeeping."""

    def __init__(self, camera: str, monitor, redis, storage, policy: dict, vendor_cfg: dict,
                 queue: FrameQueue | None = None, key_id: str | None = None, zones=None,
                 zones_mux: tuple | None = None):
        self.camera = camera
        # The id this camera's streams carry (acl.key_id): its meta is read, its alert and clip request written there.
        self.key_id = key_id or camera
        # acl.ZonesFile (or None = no zones) and the mux size its points are in: a candidate whose person stands
        # outside every fall zone of this camera is not this product's to raise.
        self.zones = zones
        self.zones_mux = zones_mux
        self._last_boxes: dict = {}            # track -> its box (capture pixels) on the last frame that carried it
        self._last_shape = None
        self.monitor = monitor
        self.redis = redis
        self.storage = storage
        self.policy = policy
        self.vendor_cfg = vendor_cfg
        self.queue = queue if queue is not None else FrameQueue()
        # track -> the time of the last fall that was alerted on it (the aftermath's suppressor)
        self._alerted_falls: dict = {}
        self._warned: dict[str, float] = {}
        self._thread: threading.Thread | None = None
        self._stopped = False

    def _warn(self, kind: str, message: str, *args) -> None:
        """The warning for a repeated malformation, at most once per interval."""
        now = time.monotonic()
        if now - self._warned.get(kind, -WARN_INTERVAL_S) >= WARN_INTERVAL_S:
            self._warned[kind] = now
            logger.warning(message, *args)

    # ── the frames ──────────────────────────────────────────────────────────────────────────────────────

    def submit(self, entry) -> None:
        """Hand a frame to this camera's queue."""
        self.queue.put(entry)

    def handle_entry(self, fields: dict) -> list[dict]:
        """One frame's fields -> the monitor's update -> the candidates' routes.

        `fields` is the frame entry's fields: the `pts` (the sidecar's frame clock) and the
        `pose` (the people, as JSON).
        """
        pts = acl.text(fields.get("pts"))
        try:
            t = float(pts)
        except ValueError:
            # the sidecar's frame clock is the product's only timeline; a frame without one
            # cannot be placed, so it is dropped rather than guessed at
            self._warn("clock", "the frame of %s has no clock (%r is not a number) -- dropped",
                       self.camera, pts)
            return []
        shape, items, observed = self._pose(fields)
        if not observed:
            # No pose field: the frame says nothing about the people, which is NOT the same as
            # saying there are none. Feeding the monitor empty ids would drop every track it holds
            # (the vendor's monitor moves a track it cannot see into the lost ones), so the frame
            # is skipped and the tracks live on until a frame does carry the field.
            return []
        set_shape = getattr(self.monitor, "set_shape", None)
        if set_shape is not None and shape:
            set_shape(shape)
        ids = [item[0] for item in items]
        boxes = [item[1] for item in items]
        kps = [item[2] for item in items]
        if self.zones is not None:
            self._last_shape = shape or self._last_shape
            self._last_boxes.update(zip(ids, boxes))
            if len(self._last_boxes) > 512:     # tracks come and go; keep the ones seen on this frame
                self._last_boxes = dict(zip(ids, boxes))
        candidates = self.monitor.update(t, ids, boxes, kps) or []
        return self.handle_candidates(candidates, t=t)

    def _pose(self, fields: dict):
        """The frame's shape, its people, and whether the frame carried a pose at all.

        A frame without the field carries nobody AND says nothing: the caller must not tell the
        monitor that the people are gone. A frame WITH the field and no people in it says the
        opposite -- the people are gone -- and the monitor has to hear it. A field of the wrong
        shape is treated as no field: the worst thing a malformed frame can do is make the
        consumer believe a scene it never described."""
        raw = acl.text(fields.get("pose"))
        if not raw:
            return None, [], False
        try:
            pose = json.loads(raw)
        except ValueError:
            self._warn("pose", "the pose of %s is not JSON -- the frame carries nobody", self.camera)
            return None, [], False
        if not isinstance(pose, dict):
            self._warn("pose", "the pose of %s is not an object -- the frame carries nobody", self.camera)
            return None, [], False
        shape = pose.get("shape")
        if not (isinstance(shape, (list, tuple)) and len(shape) == 2
                and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in shape)):
            shape = None                       # a shape the monitor cannot read is no shape
        raw_items = pose.get("items")
        items = []
        for item in raw_items if isinstance(raw_items, list) else []:
            if not isinstance(item, dict):
                continue
            try:
                items.append((int(item["id"]),
                              np.asarray(item["box"], dtype=float),
                              np.asarray(item["kps"], dtype=float)))
            except (KeyError, TypeError, ValueError):
                continue                       # a malformed person is dropped, not the frame
        return shape, items, True

    # ── the candidates ─────────────────────────────────────────────────────────────────────────────────

    def handle_candidates(self, candidates: list[dict], t: float | None = None) -> list[dict]:
        """The candidates -> the policy -> the routes -> the platform's alert and the review queue.

        One candidate failing only costs that candidate: the exception is logged and the rest of
        the frame still routes, so a locked sqlite or a bad row cannot mute the camera.
        """
        results = []
        for rec in candidates:
            try:
                routed = self._route_candidate(rec, t)
            except Exception:
                logger.exception("a candidate of %s could not be routed", self.camera)
                continue
            if routed is not None:
                results.append(routed)
        return results

    def _route_candidate(self, rec: dict, t: float | None) -> dict | None:
        """One candidate, scored and routed: the alert, the clip, the review row, the history row."""
        kind = rec.get("type", "fall")
        if kind == "fall_recovered":
            return None                        # the product ignores it: no alert, no row, no state
        scored = scoring.score_candidate(rec, self.policy, self.vendor_cfg)
        now = float(rec.get("t_alert", t if t is not None else 0.0))
        track = rec.get("track")
        suppressed = self._is_aftermath(track, now)
        covered, zone = self._zone_of(track)
        result = {"evaluation": scored["evaluation"], "combined": scored["combined"],
                  "scores": scored["scores"], "severity": scored["severity"],
                  "suppressed": suppressed, "alert": None, "review_id": None, "out_of_zone": not covered}
        if not covered:
            return result                      # outside every fall zone: no alert, no review, no history row
        if not suppressed and scored["severity"] is not None and self._above_floor(scored["combined"]):
            detailed = bool(self.policy.get("alert_details"))
            message = (scoring.alert_statement(scored["evaluation"], scored["combined"], scored["review"])
                       if detailed else f"{scored['evaluation']} {scored['combined']:.2f}")
            result["alert"] = {"severity": scored["severity"], "message": message, "pts": now, "zone": zone}
            written = acl.write_alert(self.redis, self.key_id, severity=scored["severity"],
                                     message=message, pts=now, zone=zone,
                                     **(self._details(track, scored) if detailed else {}))
            acl.request_clip(self.redis, self.key_id, now)
            if kind == "fall" and written:
                self._remember_fall(track, now)
        if not suppressed and scored["review"]:
            result["review_id"] = self.storage.add_review(self._review_row(rec, scored, now))
        if not suppressed:
            self.storage.add_event(self._event_row(rec, scored, now, result["review_id"]))
        return result

    @staticmethod
    def _details(track, scored: dict) -> dict:
        """The alert's fields beyond the platform's, with `alert_details` (the safety stack): the person's track (the
        clip draws that person) and what the message says, as data a console words in its own language."""
        return {"person_ids": "" if track is None else str(track),
                "extra": {"evaluation": scored["evaluation"], "confidence": f"{scored['combined']:.2f}",
                          "review": "1" if scored["review"] else "0"}}

    def _above_floor(self, combined: float) -> bool:
        """Whether a candidate is sure enough to be the platform's alert: its confidence, as the alert states it (two
        decimals), above the policy's optional `alert_above` (the safety stack: 0.0, so nothing that reads 0% is an
        incident). Below it the candidate still goes to the review queue and the history."""
        floor = self.policy["scoring"].get("alert_above")
        return floor is None or round(float(combined), 2) > float(floor)

    def _zone_of(self, track) -> tuple[bool, str]:
        """Whether the track's person is in scope, by where they are (box bottom centre, scaled from capture to the
        zones' mux pixels), and the fall zone they are in. No zones, or no box seen for the track: in scope."""
        if self.zones is None:
            return True, ""
        box = self._last_boxes.get(track)
        if box is None:
            return True, ""
        x, y = (float(box[0]) + float(box[2])) / 2.0, float(box[3])
        if self.zones_mux and self._last_shape:
            x *= self.zones_mux[0] / float(self._last_shape[0])
            y *= self.zones_mux[1] / float(self._last_shape[1])
        return self.zones.covering(self.camera, x, y)

    def _remember_fall(self, track, now: float) -> None:
        """Record an alerted fall and forget the ones whose aftermath has passed.

        The window is the only thing that reads this map, so an entry older than it is dead
        weight; without the sweep the map grows by a track per fall for as long as the service
        runs (design §13.3).
        """
        window = float(self.policy["suppress_after_fall_s"])
        self._alerted_falls = {t: at for t, at in self._alerted_falls.items() if now - at <= window}
        self._alerted_falls[track] = now

    def _is_aftermath(self, track, now: float) -> bool:
        """A candidate that follows an alerted fall on the same track is the fall's aftermath.

        The suppression is per track and per fall event, not a mute on the kind: the candidate
        still scores, it just does not become a second event (design §13.3).
        """
        last = self._alerted_falls.get(track)
        if last is None:
            return False
        return (now - last) <= float(self.policy["suppress_after_fall_s"])

    def _review_row(self, rec: dict, scored: dict, pts: float) -> dict:
        """The candidate's own evidence, beside the operator's decision."""
        reading = rec.get("reading") or {}
        evidence = rec.get("evidence") or {}
        return {"camera": self.camera, "pts": pts, "track": rec.get("track"),
                "event_id": rec.get("event_id"), "kind": rec.get("type"),
                "evaluation": scored["evaluation"], "combined": scored["combined"],
                "path_scores": scored["scores"], "top_class": reading.get("top"),
                "clip_window": evidence.get("clip_window"), "probs": reading.get("probs")}

    def _event_row(self, rec: dict, scored: dict, pts: float, review_id: int | None) -> dict:
        """The scored candidate as the product's own history row.

        The review row is the operator's work list; this row is the record. A confirmed fall
        has no review row and is still kept here -- that is the whole reason the table exists.
        """
        evidence = rec.get("evidence") or {}
        return {"camera": self.camera, "pts": pts, "track": rec.get("track"),
                "event_id": rec.get("event_id"), "kind": rec.get("type"),
                "evaluation": scored["evaluation"], "combined": scored["combined"],
                "severity": scored["severity"], "path_scores": scored["scores"],
                "clip_window": evidence.get("clip_window"), "review_id": review_id}

    # ── the loop ──────────────────────────────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Drain the queue until the worker is stopped."""
        while not self._stopped:
            try:
                entry = self.queue.get(timeout=0.2)
            except Empty:
                continue
            try:
                self.handle_entry(entry)
            except Exception:
                logger.exception("a frame of %s could not be handled", self.camera)

    def start(self) -> None:
        """Run the worker on its own thread."""
        self._thread = threading.Thread(target=self.run, name=f"fall-worker-{self.camera}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stopped = True

    def join(self, timeout: float | None = None) -> None:
        """Wait for the worker's loop to end."""
        if self._thread is not None:
            self._thread.join(timeout)

    def drain(self) -> list[dict]:
        """Process what the queue holds, oldest first."""
        results = []
        while True:
            try:
                entry = self.queue.get_nowait()
            except Empty:
                return results
            results.extend(self.handle_entry(entry))

    @property
    def alive(self) -> bool:
        """The worker is usable: its loop has not been stopped."""
        if self._thread is not None:
            return self._thread.is_alive()
        return not self._stopped


def build_workers(cameras, monitor_factory, redis, storage, policy: dict,
                  vendor_cfg: dict, key_ids: dict | None = None, zones=None,
                  zones_mux: tuple | None = None) -> dict[str, CameraWorker]:
    """One worker per camera, each with its own monitor. `key_ids`: camera -> its streams' key id (default the
    camera id); `zones`/`zones_mux`: the fall zones, shared by every worker (see CameraWorker)."""
    key_ids = key_ids or {}
    return {camera: CameraWorker(camera=camera, monitor=monitor_factory(camera),
                                 redis=redis, storage=storage,
                                 policy=policy, vendor_cfg=vendor_cfg, key_id=key_ids.get(camera),
                                 zones=zones, zones_mux=zones_mux)
            for camera in cameras}


class Consumer:
    """The Redis reader thread: one XREAD over every camera's meta stream.

    Each camera's frames go into that camera's queue; the camera's worker drains it.
    """

    def __init__(self, redis, workers: dict[str, CameraWorker], batch: int = 64,
                 interval: float = 0.05, retry_interval: float = 1.0):
        self.workers = workers
        self.batch = batch
        self.interval = interval
        self.retry_interval = retry_interval
        self._reader = acl.MetaStreamReader(redis)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._failing = False

    def start(self) -> None:
        self._thread = threading.Thread(target=self.run, name="fall-consumer", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: float | None = None) -> None:
        """Wait for the reader's loop to end."""
        if self._thread is not None:
            self._thread.join(timeout)

    @property
    def alive(self) -> bool:
        """The reader thread is running (a reader that was never started is not)."""
        return self._thread is not None and self._thread.is_alive()

    def _read_once(self) -> None:
        """One pass over every camera's stream; each frame goes to its camera's queue."""
        for worker in self.workers.values():
            for entry in self._reader.read(worker.key_id, self.batch):
                worker.submit(entry["fields"])

    def run(self) -> None:
        """Read until stopped, saying an outage once instead of once per attempt.

        A Redis that goes away would otherwise print a traceback every `interval` seconds for
        every camera until it came back; the first failure is said once, the reads are retried
        slowly while it lasts, and the recovery is said once.
        """
        while not self._stop.is_set():
            try:
                self._read_once()
            except redis.RedisError as exc:
                if not self._failing:
                    self._failing = True
                    logger.warning("the meta streams are unreachable -- retrying every %.1f s: %s",
                                   self.retry_interval, exc)
                self._stop.wait(self.retry_interval)
                continue
            except Exception:
                logger.exception("a frame could not be queued")
            else:
                if self._failing:
                    self._failing = False
                    logger.info("the meta streams are readable again")
            self._stop.wait(self.interval)
