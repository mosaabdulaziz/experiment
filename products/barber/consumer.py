"""The consumer loop: reads every camera's frames (iris:meta) and sampled
appearances (iris:appearance) from Redis, runs the one ShopEngine over all
of them, persists what changed and asks the server to push a live snapshot.

A daemon thread that must never die from one bad cycle; backs off while
Redis is down; handles test-clip loops (all three clips loop together, so
any wrap resets the whole engine). One cycle is step(), so it can be tested
against a fake Redis. The reads never block: the loop sleeps briefly when
no frame arrived."""
from __future__ import annotations

import json
import logging
import os
import statistics
import threading
import time
import uuid
from datetime import datetime
from typing import Callable

import redis

import crops
import register
from acl import AppearanceStreamReader, FaceCandidateReader, MetaStreamReader, live_watch
from config import ServiceConfig, load_stream_starts
from chair_engine import ChairEngine
from domain import EngineUpdate
from engine import ShopEngine
from face_regions import REDIS_KEY, face_regions_payload
from haircut_photos import HaircutPhotos
from kids_car import BUSY, KIDS, CarLocator, KidsCar, OnnxCarDetector, merge
from policy import DEFAULT_POLICY, MAXIMUMS, effective, policy_store
from shopday import TicketCounter, shop_day_range
from snapshots import Snapshotter, _http_get
from storage import (apply_update, clear_all, close_interrupted, connect, get_state, max_ticket, purge_inactive,
                     purge_haircut_photos, purge_snapshots, referenced_crops, roster_at, set_state, set_visit_baseline, visit_baseline)
from zones import zone_store

logger = logging.getLogger("barber.consumer")

HEARTBEAT_S = 5.0  # push a live snapshot at least this often, changed or not


REGIONS_REPUBLISH_S = 30.0
# The visit baseline is taken this long after the service starts (or a clips-loop clear), from everything seen until
# then: the most haircuts + the most open waits + the median wash-area count (2026-10-11: a snapshot at 120 s read 1-7
# on one replay -- haircuts confirm late, clustered sofa waits start late, the wash count flickers; real-engine
# replays of 4 recordings: total error vs the people inside at the start 4 -> 1, plateau 120-240 s).
BASELINE_SETTLE_S = 180.0


class ConsumerThread(threading.Thread):
    def __init__(self, cfg: ServiceConfig, on_change: Callable[[], None] = lambda: None,
                 on_scene: Callable[[dict], None] = lambda scene: None):
        super().__init__(daemon=True, name="barber-consumer")
        self.cfg = cfg
        self.on_change = on_change
        self.on_scene = on_scene
        self.base_policy = {**DEFAULT_POLICY, **cfg.policy}
        self.policy = policy_store(cfg.data_dir)
        self.zones = zone_store(cfg.data_dir)
        self._zones = self._zones_by_cam(self.zones.load())
        self.tickets = TicketCounter(int(self.effective_policy()["day_start_hour"]))
        engine_cls = ChairEngine if cfg.engine == "chair" else ShopEngine
        kw = {"wait_camera": cfg.wait_camera} if engine_cls is ChairEngine else {}
        self.engine = engine_cls(self._zones, cfg.frame, barber_for=self._barber_for, tickets=self.tickets, **kw)
        self.lock = threading.Lock()   # the engine is read by the API thread (live())
        # a run on recordings (2026-10-18): called when its clips play to their end, before the lap's rows are cleared
        # -- its report (server.py); not on a restart of the clips mid-lap
        self.on_lap_end: Callable[[], None] | None = None
        self.scene: dict[str, list[dict]] = {}
        self.chairs: dict[str, dict] = {}
        self.kids = KidsCar(cfg.data_dir / "kids_car.json", time.time())     # the kids car's place (sub-project 4)
        self.locator: CarLocator | None = None
        self._conn = None
        self._redis = None
        self.snapshots: Snapshotter | None = None
        self.photos: HaircutPhotos | None = None
        self._last_position: dict[str, float] = {}
        self._failures = 0
        self._last_failure_log = 0.0
        self._last_push = 0.0
        self._next_scene_due = 0.0
        self._last_scene_sent: dict | None = None
        self._scene_version = 0
        self._checked_version = -1
        self._last_retention_day = None
        self._regions_ok = False
        self._regions_src = 1
        self._regions_at = 0.0
        self._regions_log = -60.0
        self.clock: Callable[[], datetime] = datetime.now   # wall clock (a test injects its own)
        self._settle_from: datetime | None = None
        self._first_ts: float | None = None
        self._armed = False
        self._baseline_day: str | None = None
        self._baseline_log = -60.0

    # ── the kids car (sub-project 4) ────────────────────────────────────────

    def _start_locator(self) -> None:
        if not int(self.effective_policy()["kc_enabled"]):
            return
        if self.cfg.kids_car_model is None:
            logger.warning("kids car: kc_enabled but no model file (kids_car_model): the stored Kids zones stay")
            return
        base = self.cfg.go2rtc_url.rstrip("/")
        self.locator = CarLocator([c.id for c in self.cfg.cameras],
                                  fetch=lambda cam: _http_get(f"{base}/api/frame.jpeg?src={cam}"),
                                  detect=OnnxCarDetector(str(self.cfg.kids_car_model)).detect,
                                  every_s=lambda: self.effective_policy()["kc_every_s"])
        self.locator.start()
        logger.info("kids car: locator on (%s, every %.0f s)", self.cfg.kids_car_model.name,
                    self.effective_policy()["kc_every_s"])

    def _with_car(self, zones: dict, policy: dict) -> dict:
        """The stored zones with the Kids chair where the car is (kc_enabled), else unchanged."""
        if not int(policy["kc_enabled"]):
            return zones
        if self.locator is not None:
            busy = self.chairs.get(KIDS, {}).get("state") in BUSY
            self.kids.observe(self.locator.latest(), time.time(), busy, policy)
        return merge(zones, self.kids.place)

    def shown_zones(self) -> dict:
        """What the dashboard draws: the stored zones, the Kids chair where the car is."""
        raw = self.zones.load()
        return merge(raw, self.kids.place) if int(self.effective_policy()["kc_enabled"]) else raw

    def _zones_by_cam(self, raw: dict) -> dict[str, list[dict]]:
        return {c.id: raw.get(c.id, []) for c in self.cfg.cameras}

    def _barber_for(self, chair: str, at: datetime) -> str | None:
        return roster_at(self._conn, chair, at) if self._conn is not None else None

    # ── lifecycle ───────────────────────────────────────────────────────────

    def effective_policy(self) -> dict[str, float]:
        """DEFAULT_POLICY <- the site's policy block <- data/policy.json."""
        return effective(self.policy.load(), base=self.base_policy)

    def run(self) -> None:
        self.attach(connect(str(self.cfg.db_path)), redis.Redis.from_url(self.cfg.redis_url))
        logger.info("consumer started, cameras: %s", ", ".join(c.id for c in self.cfg.cameras))
        while True:
            try:
                self.step()
            except Exception:
                logger.exception("consumer cycle failed unexpectedly -- continuing (thread must not die)")
                time.sleep(2.0)

    def attach(self, conn, r) -> None:
        self._conn, self._redis = conn, r
        self._run = self._clip_run(load_stream_starts(self.cfg.stream_starts_file))
        stored = get_state(conn, "clip_run")
        if os.environ.get("BARBER_CLEAR_ON_START") == "1":      # `just barber-restart hard` (2026-10-11)
            self._clear_rows("hard restart")
        elif self._run and any(c.clear_on_loop for c in self.cfg.cameras) and (stored is None or json.loads(stored) != self._run):
            # 2026-10-11 (user): the rows belong to another run of the test clips -- the streamer restarted them,
            # the footage changed or they looped while the service was down. Start clean, as a loop would.
            self._clear_rows("new run of the test clips since the last start")
        else:
            closed = close_interrupted(conn)
            if any(closed.values()):
                logger.info("closed rows left open by the previous run: %s", closed)
        if self._run:
            set_state(conn, "clip_run", json.dumps(self._run))
        start, end = shop_day_range(datetime.now(), self.tickets.hour)
        self.tickets.resume(start.date(), max_ticket(conn, start, end))
        if self.cfg.engine == "chair":
            self.snapshots = Snapshotter(self.cfg.go2rtc_url, self.cfg.crops_dir, str(self.cfg.db_path))
            self.photos = HaircutPhotos(self.cfg.go2rtc_url, self.cfg.crops_dir, str(self.cfg.db_path))
            self._start_locator()
        self._sweep_on_start()
        self._meta = MetaStreamReader(r)
        self._apps = AppearanceStreamReader(r)
        self._faces = FaceCandidateReader(r)
        self._publish_face_regions(self._zones, int(self.effective_policy()["register_source"]))
        self._arm_baseline()

    def _sweep_on_start(self) -> None:
        """Crops of visits a restart cut short are orphans: sweep once now
        rather than waiting for 03:00. The stream cursors start at the tip,
        so nothing written before this start is ever pooled; five minutes
        covers crops written just before their stream entry."""
        try:
            swept = crops.sweep_orphans(self.cfg.crops_dir, referenced_crops(self._conn), time.time(), 300)
        except Exception:
            logger.exception("startup crop sweep failed (the nightly one retries)")
            return
        if swept:
            logger.info("startup: %d orphan crop(s) swept", swept)

    def _publish_face_regions(self, zones: dict, register_source: int) -> None:
        """Set the key; on failure remember it so step() retries (logged at most once a minute)."""
        self._regions_src = register_source
        try:
            self._redis.set(REDIS_KEY, face_regions_payload(zones, register_source))
        except redis.RedisError:
            self._regions_ok = False
            now = time.monotonic()
            if now - self._regions_log >= 60.0:
                self._regions_log = now
                logger.exception("could not publish the face regions (retrying each cycle)")
            return
        self._regions_ok = True
        self._regions_at = time.monotonic()

    def live(self) -> dict:
        with self.lock:
            return self.engine.live()

    @property
    def clip_run(self) -> dict:
        """The test clips' run as last seen -- set at the start and at each loop: {camera: [start_ts, clip_duration_s,
        lap]}; empty with live cameras or before attach. A run on recordings' footage clock (footage_clock.py)."""
        return dict(getattr(self, "_run", None) or {})

    # ── one cycle ───────────────────────────────────────────────────────────

    def step(self) -> None:
        policy = self.effective_policy()
        if self.cfg.engine == "chair":
            # register_source 0 (chair crops, 2026-10-05, temporary): every haircut its own customer; 1 (door faces)
            # and 2 (seated faces, 2026-10-14) link returns
            policy["link_returns"] = 1.0 if int(policy["register_source"]) in (1, 2) else 0.0
        zones = self._with_car(self._zones_by_cam(self.zones.load()), policy)
        live_cams = [c.id for c in self.cfg.cameras if c.clip_duration_s is None]
        starts = {**load_stream_starts(self.cfg.stream_starts_file), **live_watch.refresh(self._redis, live_cams)}
        changed = self._handle_loop_wraps(starts)
        frames, apps, faces, failed = [], [], [], False
        for cam in [c for c in self.cfg.cameras if c.id in starts]:
            try:
                frames += self._meta.read_frames(cam.id)
                apps += self._apps.read(cam.id)
                faces += self._faces.read(cam.id)
            except redis.RedisError:
                failed = True
                if time.monotonic() - self._last_failure_log > 30.0:
                    self._last_failure_log = time.monotonic()
                    logger.exception("redis read failed for %s (logged at most once/30s)", cam.id)
        with self.lock:
            source = int(policy["register_source"])
            if zones != self._zones or source != self._regions_src:
                if zones != self._zones:
                    self.engine.set_zones(zones)
                    self._zones = zones
                self._publish_face_regions(zones, source)
            elif not self._regions_ok or time.monotonic() - self._regions_at > REGIONS_REPUBLISH_S:
                self._publish_face_regions(zones, source)   # retry after a failure; re-set in case Redis lost the key
            update = self.engine.observe(frames, apps, policy, faces=faces)
            if frames:
                self._scene_version += 1
                self.scene = self.engine.scene(policy)
                self.chairs = self.engine.live()["chairs"]
        if update:
            apply_update(self._conn, update, datetime.now())
            if self.snapshots is not None:
                for a in update.alerts:
                    if a.open and a.subject_cam:
                        self.snapshots.request(a.id, a.subject_cam)
            self._photos_out(update, policy)
            changed = True
        if update.registrations or update.discard_crops:
            try:
                ids = register.apply(self._conn, update, self.cfg.crops_dir, policy, lambda: uuid.uuid4().hex[:12])
                if ids:
                    logger.info("registered %d visit(s) to customers %s", len(ids), ", ".join(ids))
            except Exception:
                logger.exception("customer register update failed (visits keep customer_id NULL)")
        self._retention(policy)
        self._maybe_baseline(policy, frames)
        if failed:
            self._failures += 1
            time.sleep(min(2.0 * self._failures, 10.0))
        else:
            self._failures = 0
            if not frames:
                time.sleep(0.05)
        now_mono = time.monotonic()
        if changed or now_mono - self._last_push >= HEARTBEAT_S:
            self._last_push = now_mono
            try:
                self.on_change()
            except Exception:
                logger.exception("on_change callback failed")
        scene_fps = min(max(policy["scene_fps"], 0.0), MAXIMUMS["scene_fps"])
        self.maybe_push_scene(now_mono, scene_fps)

    def _arm_baseline(self) -> None:
        """Start (or restart) the settle: BASELINE_SETTLE_S of wall time AND of frame time from the first frame."""
        self._settle_from, self._first_ts, self._armed = self.clock(), None, True
        self._seen: list[tuple[int, int, int]] = []        # (haircuts, open waits, wash-area people) per cycle

    def _maybe_baseline(self, policy: dict, frames: list) -> None:
        """Once per shop day, BASELINE_SETTLE_S after the service started (wall time and engine/frame time, so a
        late stream never gives a cold baseline), store who was already inside, from every cycle of the settle: the
        most running haircuts + the most waiting customers + the median of the wash-area people beyond the away
        haircuts. A new shop day re-arms it. Never raises."""
        try:
            if not hasattr(self.engine, "occupancy"):
                return
            now = self.clock()
            start, _ = shop_day_range(now, int(policy["day_start_hour"]))
            day = start.date().isoformat()
            if self._baseline_day is not None and day != self._baseline_day:
                self._arm_baseline()                         # the shop day rolled over
            self._baseline_day = day
            if not self._armed:
                return
            if visit_baseline(self._conn, day) is not None:
                self._armed = False
                return
            if self._first_ts is None and frames:
                self._first_ts = min(f.ts for f in frames)   # frame time of the first frame since arming
            with self.lock:
                ts = self.engine.now_ts
                if ts is None or self._first_ts is None:
                    return                                   # no frame yet: the shop is not seen
                occ = self.engine.occupancy()
            self._seen.append((occ["haircuts"], occ["waiting"], max(0, occ["washing_people"] - occ["away"])))
            if (now - self._settle_from).total_seconds() < BASELINE_SETTLE_S or ts - self._first_ts < BASELINE_SETTLE_S:
                return
            haircuts, waiting = max(h for h, _, _ in self._seen), max(w for _, w, _ in self._seen)
            washing = statistics.median_low(x for _, _, x in self._seen)
            set_visit_baseline(self._conn, day, now, haircuts, waiting, washing)
            self._armed = False
            logger.info("visit baseline for %s: %d = %d haircuts + %d waiting + %d washing (most / most / median over "
                        "%d cycles of the first %.0f s)", day, haircuts + waiting + washing, haircuts, waiting, washing,
                        len(self._seen), BASELINE_SETTLE_S)
        except Exception:
            if time.monotonic() - self._baseline_log >= 60.0:
                self._baseline_log = time.monotonic()
                logger.exception("visit baseline failed (retried next cycle, logged at most once a minute)")

    def maybe_push_scene(self, now_mono: float, fps: float) -> bool:
        """Push the scene (boxes + chair states) at most `fps` times a second, and only
        when it moved. The full snapshot (on_change) only goes out when a
        visit, haircut or alert changes, or every HEARTBEAT_S -- far too
        rarely for boxes to follow the video."""
        if fps <= 0:
            return False
        interval = 1.0 / fps
        # Slots open on a fixed schedule (every `interval`), with 10% slack so
        # a lap finishing a hair early still counts.
        if now_mono < self._next_scene_due - 0.1 * interval:
            return False
        with self.lock:
            version = self._scene_version
            payload = {"scene": {cam: list(entries) for cam, entries in self.scene.items()},
                       "chairs": {cam: dict(states) for cam, states in self.chairs.items()}}
        if version == self._checked_version and now_mono < self._next_scene_due + interval:
            # No new frame since the last look (frames arrive unevenly): keep
            # the slot open and look again next lap -- for at most one extra
            # interval, so a stalled camera can't spin the loop.
            return False
        # Spend the slot. The next one is due a full interval after THIS one
        # was due, not after it was used -- so waiting for a frame doesn't push
        # every later slot back (that would drift the rate below fps) -- but
        # never sooner than half an interval from now (no catch-up bursts).
        # A look at a new frame spends the slot even when nothing moved, or a
        # still shop would keep the slot open and spin the loop.
        self._next_scene_due = max(self._next_scene_due + interval, now_mono + 0.5 * interval)
        self._checked_version = version
        if payload == self._last_scene_sent:
            return False
        self._last_scene_sent = payload
        try:
            self.on_scene(payload)
        except Exception:
            logger.exception("on_scene callback failed")
        return True

    def _retention(self, policy: dict) -> None:
        """Once a day at 03:00 local: delete customers inactive for
        customer_retention_days, and orphaned crop files older than a day."""
        now = datetime.now()
        if now.hour != 3 or self._last_retention_day == now.date():
            return
        self._last_retention_day = now.date()
        try:
            crop_paths = purge_inactive(self._conn, now, policy["customer_retention_days"])
            snapshot_paths = purge_snapshots(self._conn, now, policy["customer_retention_days"])
            photo_paths = purge_haircut_photos(self._conn, now, policy["customer_retention_days"])
            paths = crop_paths + snapshot_paths + photo_paths
            for rel in paths:
                crops.delete(self.cfg.crops_dir, rel)
            swept = crops.sweep_orphans(self.cfg.crops_dir, referenced_crops(self._conn), time.time(), 86400)
        except Exception:
            logger.exception("customer retention failed (retried tomorrow)")
            return
        logger.info("retention: %d customer crop(s) deleted with inactive customers, %d alert snapshot(s) and %d "
                    "haircut photo(s) purged, %d orphan crop(s) swept", len(crop_paths), len(snapshot_paths),
                    len(photo_paths), swept)

    def _photos_out(self, update: EngineUpdate, policy: dict) -> None:
        """The haircut photos the engine asked for, and the drops (2026-10-22): on the worker's thread, never here."""
        if self.photos is None:
            return
        for r in update.photos:
            self.photos.request(r, int(policy["hp_photo_px"]))
        for hid in sorted(update.dropped_photos):
            self.photos.drop(hid)

    def _clip_run(self, starts: dict[str, datetime], now: datetime | None = None) -> dict:
        """The test clips' run: {camera: [start_ts, clip_duration_s, lap]} (live cameras: not in it)."""
        now = now or datetime.now()
        return {c.id: [round(starts[c.id].timestamp(), 3), c.clip_duration_s,
                       int((now - starts[c.id]).total_seconds() // c.clip_duration_s)]
                for c in self.cfg.cameras if c.clip_duration_s is not None and c.id in starts}

    def _clear_rows(self, why: str) -> None:
        paths = clear_all(self._conn)
        for rel in paths:
            crops.delete(self.cfg.crops_dir, rel)
        self.tickets.clear()
        logger.info("%s: rows cleared (%d crop path(s) deleted), tickets from #1", why, len(paths))

    def _handle_loop_wraps(self, starts: dict[str, datetime]) -> bool:
        """A looping test clip restarted. The three clips loop together, so
        any wrap resets the whole engine (the same people walk in again).
        A new start time (the streamer restarted the clips) is a wrap too, wherever the position landed."""
        now, wrapped, played_out = datetime.now(), False, False
        run = self._clip_run(starts, now)
        for cam, (start, dur, _) in run.items():
            before = getattr(self, "_run", {}).get(cam)
            if before is not None and (before[0], before[1]) != (start, dur):
                wrapped = True
        for cam in self.cfg.cameras:
            if cam.clip_duration_s is None or cam.id not in starts:
                continue
            position = (now - starts[cam.id]).total_seconds() % cam.clip_duration_s
            previous = self._last_position.get(cam.id)
            self._last_position[cam.id] = position
            if previous is not None and position < previous:
                wrapped = played_out = True
        if not wrapped:
            return False
        if played_out and self.on_lap_end is not None:
            try:
                self.on_lap_end()               # the lap's rows are all still there
            except Exception:
                logger.exception("end-of-lap hook failed")
        with self.lock:
            for rel in self.engine.clear():
                crops.delete(self.cfg.crops_dir, rel)
        self._arm_baseline()
        if any(c.clear_on_loop for c in self.cfg.cameras):
            self._clear_rows("clips looped: engine reset")
        else:
            logger.info("clips looped: engine reset")
        self._run = run
        if run:
            set_state(self._conn, "clip_run", json.dumps(run))
        return True
