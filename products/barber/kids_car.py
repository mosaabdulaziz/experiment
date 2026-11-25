"""The kids car chair (sub-project 4; spec docs/superpowers/specs/2026-10-15-barber-kids-car-design.md).

The shop's kids chair is a toy car staff move anywhere on the floor. A one-class detector looks at one go2rtc frame
per camera every kc_every_s (CarLocator, below); KidsCar decides where the car is: among fresh detections
(<= 2 intervals) at kc_conf_min or more whose box keeps 2 % from the frame edge, the camera with the highest
confidence x box area; the camera holding the car keeps it while it still sees it in its place; a new place counts
after kc_confirm_n snapshots in a row there and only while the Kids chair is empty (a running haircut is never split). The chair: one unpaired chair `Kids` (outline = the box x 1.15) and
its seat (the middle half of the box's width, 15-55 % of its height from the top), merged over the stored zones in
place of the stored `Kids` chair and seat. Unseen
kc_lost_s: no Kids chair. A manual pin (the dashboard's "the car is here") holds until cleared or until the car is
seen elsewhere for kc_release_s. The place is saved (data/<site>/kids_car.json) so a restart keeps it."""
from __future__ import annotations

import io
import json
import logging
import math
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

log = logging.getLogger("barber.kids_car")

KIDS = "Kids"
EDGE = 0.02
GROW = 1.15
# 2026-10-18 (user; the first real kids haircut, ph-1): the seat was the box's central 60 %, a guess (spec) -- the
# box holds the car's nose, wheels and the base under it, the child sits higher, and his hips got 0.09 of it (min
# 0.10, never occupied). Here 0.15 on all 4 of his detections, his arm beside the car 0; x kept symmetric (the car
# can face either way). Fractions of the box: x from-to, y from-to (from the top).
SEAT_X, SEAT_Y = (0.25, 0.75), (0.15, 0.55)
BUSY = ("starting", "sitting", "away")

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Detection:
    camera: str
    box: Box          # normalised x0, y0, x1, y1
    conf: float
    ts: float


@dataclass(frozen=True)
class Place:
    camera: str
    box: Box
    source: str       # auto | manual
    conf: float | None
    since: float


def _clip(v: float) -> float:
    return min(1.0, max(0.0, v))


def scaled(box: Box, k: float) -> Box:
    x0, y0, x1, y1 = box
    cx, cy, hw, hh = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2 * k, (y1 - y0) / 2 * k
    return (_clip(cx - hw), _clip(cy - hh), _clip(cx + hw), _clip(cy + hh))


def part(box: Box, fx: tuple[float, float], fy: tuple[float, float]) -> Box:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    return (x0 + fx[0] * w, y0 + fy[0] * h, x0 + fx[1] * w, y0 + fy[1] * h)


def _rect(b: Box) -> list[list[float]]:
    x0, y0, x1, y1 = (round(v, 4) for v in b)
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def zones_for(place: Place) -> list[dict]:
    return [{"id": KIDS, "name": KIDS, "type": "chair", "points": _rect(scaled(place.box, GROW))},
            {"id": f"{KIDS}-seat", "name": KIDS, "type": "seat", "points": _rect(part(place.box, SEAT_X, SEAT_Y)),
             "seat_of": KIDS}]


def _stored_kids(z: dict) -> bool:
    return z.get("type") in ("chair", "seat") and z.get("name") == KIDS


def merge(zones_by_cam: dict, place: Place | None) -> dict:
    out = {cam: [z for z in zs if not _stored_kids(z)] for cam, zs in zones_by_cam.items()}
    if place is not None and place.camera in out:
        out[place.camera] = out[place.camera] + zones_for(place)
    return out


def _area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inside(b: Box) -> bool:
    return b[0] >= EDGE and b[1] >= EDGE and b[2] <= 1 - EDGE and b[3] <= 1 - EDGE


def _same(cam_a: str, a: Box, cam_b: str, b: Box) -> bool:
    """The same place: the same camera, the centre moved at most half the box width."""
    if cam_a != cam_b:
        return False
    return math.dist(((a[0] + a[2]) / 2, (a[1] + a[3]) / 2), ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)) <= (a[2] - a[0]) / 2


class KidsCar:
    def __init__(self, path: Path | None, now: float):
        self.path = path
        self.place: Place | None = None
        self._cand: tuple[str, Box, int, float] | None = None    # camera, box, snapshots in a row, the last one's ts
        self._last_seen = now                       # a loaded place lives kc_lost_s from the start unless seen
        if path is not None and path.is_file():
            try:
                d = json.loads(path.read_text())
                self.place = Place(d["camera"], tuple(d["box"]), d["source"], d.get("conf"), float(d["since"]))
            except (ValueError, KeyError, TypeError):
                log.warning("kids car: %s unreadable, starting without a place", path)

    def _save(self) -> None:
        if self.path is None:
            return
        try:
            if self.place is None:
                self.path.unlink(missing_ok=True)
            else:
                self.path.write_text(json.dumps(asdict(self.place)))
        except OSError:
            log.warning("kids car: could not save %s", self.path)

    def _set(self, place: Place | None) -> None:
        if place != self.place:
            old = self.place
            self.place = place
            self._save()
            log.info("kids car: %s -> %s", "none" if old is None else f"{old.camera} ({old.source})",
                     "none" if place is None else f"{place.camera} ({place.source}, conf {place.conf})")

    def observe(self, dets: dict, now: float, busy: bool, policy: dict) -> Place | None:
        every = policy["kc_every_s"]
        fresh = [d for d in dets.values() if d is not None and now - d.ts <= 2 * every
                 and d.conf >= policy["kc_conf_min"] and _inside(d.box)]
        manual = self.place is not None and self.place.source == "manual"
        if not fresh:
            if not manual and self.place is not None and now - self._last_seen > policy["kc_lost_s"]:
                self._set(None)
            return self.place
        self._last_seen = now
        # the owner keeps the car while it still sees it in its place, whatever the others score (10 Oct GT hour:
        # the parked car seen by ph-1 and ph-2 flipped the chair six times when ph-1 missed twice in a row)
        if self.place is not None and any(_same(self.place.camera, self.place.box, d.camera, d.box) for d in fresh):
            self._cand = None
            return self.place
        cur = self.place.camera if self.place else None
        best = max(fresh, key=lambda d: (d.conf * _area(d.box), d.camera == cur))
        # a snapshot counts once: observe runs every consumer cycle with the locator's latest (2026-10-16 bug: one
        # detection reached kc_confirm_n within a second)
        if self._cand and _same(self._cand[0], self._cand[1], best.camera, best.box):
            n = self._cand[2] + (best.ts > self._cand[3])
        else:
            n = 1
        self._cand = (best.camera, best.box, n, best.ts)
        need = math.ceil(policy["kc_release_s"] / every) if manual else int(policy["kc_confirm_n"])
        if n >= need and not busy:
            self._cand = None
            self._set(Place(best.camera, best.box, "auto", round(best.conf, 3), now))
        return self.place

    def pin(self, camera: str, x: float, y: float, size: tuple[float, float], now: float) -> Place:
        w, h = size
        self._cand = None
        self._set(Place(camera, (_clip(x - w / 2), _clip(y - h / 2), _clip(x + w / 2), _clip(y + h / 2)), "manual",
                        None, now))
        return self.place

    def clear(self) -> None:
        self._cand = None
        self._set(None)


# ── the detector and the snapshot thread ─────────────────────────────────────

def decode(out: np.ndarray, scale: float, pad: tuple[float, float], w: int, h: int) -> tuple[Box, float]:
    """A one-class YOLO head (1, 5, N): cx, cy, bw, bh, score in input pixels -> the best box, normalised."""
    a = out[0]
    i = int(np.argmax(a[4]))
    cx, cy, bw, bh, s = (float(v) for v in a[:, i])
    x0, y0 = (cx - bw / 2 - pad[0]) / scale, (cy - bh / 2 - pad[1]) / scale
    x1, y1 = (cx + bw / 2 - pad[0]) / scale, (cy + bh / 2 - pad[1]) / scale
    return (_clip(x0 / w), _clip(y0 / h), _clip(x1 / w), _clip(y1 / h)), s


class OnnxCarDetector:
    """The one-class model on CPU with one thread (user, 2026-10-15: as few resources as possible)."""

    def __init__(self, path: str, size: int = 416):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.name, self.size = self.sess.get_inputs()[0].name, size
        self.out = self.sess.get_outputs()[0].name

    def detect(self, jpeg: bytes) -> tuple[Box, float] | None:
        from PIL import Image
        img = Image.open(io.BytesIO(jpeg)).convert("RGB")
        w, h = img.size
        scale = self.size / max(w, h)
        nw, nh = round(w * scale), round(h * scale)
        px, py = (self.size - nw) // 2, (self.size - nh) // 2
        canvas = Image.new("RGB", (self.size, self.size), (114, 114, 114))
        canvas.paste(img.resize((nw, nh)), (px, py))
        x = np.asarray(canvas, np.float32).transpose(2, 0, 1)[None] / 255.0
        box, conf = decode(self.sess.run([self.out], {self.name: x})[0], scale, (px, py), w, h)
        return (box, conf) if conf >= 0.01 else None


class CarLocator:
    """One frame per camera every every_s() seconds, spread over the interval; the latest result per camera
    (None: nothing found, or the frame failed -- go2rtc, JPEG or model trouble is logged at most once a minute)."""

    def __init__(self, cameras: list[str], fetch: Callable[[str], bytes], detect: Callable[[bytes], tuple | None],
                 every_s: Callable[[], float], clock: Callable[[], float] = time.time):
        self.cameras, self.fetch, self.detect, self.every_s, self.clock = list(cameras), fetch, detect, every_s, clock
        self._latest: dict[str, Detection | None] = {c: None for c in self.cameras}
        self._lock, self._stop = threading.Lock(), threading.Event()
        self._thread: threading.Thread | None = None
        self._warned = 0.0

    def step(self, cam: str) -> None:
        try:
            got = self.detect(self.fetch(cam))
        except Exception as exc:                                       # never stop the thread
            if time.monotonic() - self._warned > 60:
                self._warned = time.monotonic()
                log.warning("kids car: %s skipped (%s; logged at most once a minute)", cam, exc)
            got = None
        with self._lock:
            self._latest[cam] = None if got is None else Detection(cam, got[0], got[1], self.clock())

    def latest(self) -> dict[str, Detection | None]:
        with self._lock:
            return dict(self._latest)

    def _run(self) -> None:
        while not self._stop.is_set():
            for cam in self.cameras:
                if self._stop.wait(self.every_s() / max(1, len(self.cameras))):
                    return
                self.step(cam)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="kids-car", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
