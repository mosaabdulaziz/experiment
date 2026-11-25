"""Haircut photos, the worker's side (2026-10-22, spec 2026-10-22-barber-haircut-photos-design.md): each PhotoRequest
of the engine is one go2rtc frame of the chair's camera (3 s timeout), cut to its close-up, scaled to at most
hp_photo_px wide and saved as <crops_dir>/haircuts/<haircut id>_<kind>.jpg -- under a temporary name, then renamed,
so an end photo is replaced, never half written -- and linked in haircut_photo. A drop goes through the same queue:
it follows its haircut's requests and removes what they saved. go2rtc down or slow: no photo, a warning per camera
at most once a minute; nothing else changes. Pattern: snapshots.Snapshotter."""
from __future__ import annotations

import io
import logging
import os
import queue
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Callable

from PIL import Image

import crops
from domain import PhotoRequest
from snapshots import _http_get
from storage import connect, drop_haircut_photos, set_haircut_photo

logger = logging.getLogger("barber.haircut_photos")
QUEUE_MAX = 200
QUALITY = 85
WARN_EVERY_S = 60.0


def cropped(jpeg: bytes, crop: tuple, px: int) -> bytes:
    """The close-up: crop (x0, y0, x1, y1 fractions) of the frame's own size, at most px wide (never enlarged)."""
    im = Image.open(io.BytesIO(jpeg)).convert("RGB")
    w, h = im.size
    x0, y0 = int(crop[0] * w), int(crop[1] * h)
    im = im.crop((x0, y0, max(x0 + 1, round(crop[2] * w)), max(y0 + 1, round(crop[3] * h))))
    if im.width > px:
        im = im.resize((px, max(1, round(im.height * px / im.width))), Image.LANCZOS)
    out = io.BytesIO()
    im.save(out, "JPEG", quality=QUALITY)
    return out.getvalue()


class HaircutPhotos:
    def __init__(self, go2rtc_url: str, crops_dir: Path, db_path: str, fetch: Callable[[str], bytes] | None = None):
        self.base = go2rtc_url.rstrip("/")
        self.crops_dir = Path(crops_dir)
        self.db_path = db_path
        self.fetch = fetch or _http_get
        self._q: queue.Queue = queue.Queue(maxsize=QUEUE_MAX)
        self._warned: dict[str, float] = {}            # camera -> monotonic time of its last warning
        self._thread = threading.Thread(target=self._run, name="haircut-photos", daemon=True)
        self._thread.start()

    def request(self, r: PhotoRequest, px: int) -> None:
        """Queue one photo; never blocks."""
        self._put(("photo", r, int(px)), r.haircut_id)

    def drop(self, haircut_id: str) -> None:
        """A haircut that closed unconfirmed: its photos go, after any of its requests still queued."""
        self._put(("drop", haircut_id, 0), haircut_id)

    def _put(self, item: tuple, haircut_id: str) -> None:
        try:
            self._q.put_nowait(item)
        except queue.Full:
            logger.warning("haircut %s: photo queue full, %s dropped", haircut_id, item[0])

    def stop(self) -> None:
        self._q.put(None)
        self._thread.join(timeout=10)

    def _run(self) -> None:
        try:
            conn = connect(self.db_path)
        except Exception:
            logger.exception("haircut photo worker could not open the database -- no haircut photos")
            return
        while True:
            item = self._q.get()
            if item is None:
                return
            what, arg, px = item
            try:
                if what == "drop":
                    for rel in drop_haircut_photos(conn, arg):
                        crops.delete(self.crops_dir, rel)
                else:
                    self._take(conn, arg, px)
            except Exception:
                logger.exception("haircut photo %s for %s failed", what, arg if what == "drop" else arg.haircut_id)

    def _take(self, conn, r: PhotoRequest, px: int) -> None:
        url = f"{self.base}/api/frame.jpeg?src={urllib.parse.quote(r.camera)}"
        try:
            data = self.fetch(url)
            if not data:
                raise ValueError("empty frame")
            jpg = cropped(data, r.crop, px)
        except Exception as exc:
            now, last = time.monotonic(), self._warned.get(r.camera)
            if last is None or now - last >= WARN_EVERY_S:
                self._warned[r.camera] = now
                logger.warning("haircut %s: no %s photo from %s (%s) (logged at most once a minute per camera)",
                               r.haircut_id, r.kind, r.camera, exc)
            return
        rel = f"haircuts/{r.haircut_id}_{r.kind}.jpg"
        path = self.crops_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(jpg)
        os.replace(tmp, path)
        set_haircut_photo(conn, r.haircut_id, r.kind, rel, r.at)
