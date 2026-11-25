"""Alert snapshots (dashboard spec 2026-10-05): when an alert opens, one frame of its camera is fetched from
go2rtc by a background thread (never the engine's), saved under <crops_dir>/alerts/<alert id>.jpg and linked to the
alert row. The close-up is drawn by the dashboard from the full frame and the alert's subject box. go2rtc down or
slow (3 s timeout) leaves the alert without a picture and logs a warning."""
from __future__ import annotations

import logging
import queue
import threading
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable

from storage import connect, set_alert_snapshot

logger = logging.getLogger("barber.snapshots")
TIMEOUT_S = 3.0


def _http_get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=TIMEOUT_S) as r:
        return r.read()


class Snapshotter:
    def __init__(self, go2rtc_url: str, crops_dir: Path, db_path: str, fetch: Callable[[str], bytes] | None = None):
        self.base = go2rtc_url.rstrip("/")
        self.crops_dir = Path(crops_dir)
        self.db_path = db_path
        self.fetch = fetch or _http_get
        self._q: queue.Queue = queue.Queue(maxsize=100)
        self._seen: set[str] = set()
        self._thread = threading.Thread(target=self._run, name="alert-snapshots", daemon=True)
        self._thread.start()

    def request(self, alert_id: str, camera: str) -> None:
        """Queue one snapshot (once per alert); never blocks."""
        if alert_id in self._seen:
            return
        self._seen.add(alert_id)
        try:
            self._q.put_nowait((alert_id, camera))
        except queue.Full:
            logger.warning("alert %s: snapshot queue full, no picture", alert_id)

    def stop(self) -> None:
        self._q.put(None)
        self._thread.join(timeout=TIMEOUT_S + 2)

    def _run(self) -> None:
        try:
            conn = connect(self.db_path)
        except Exception:
            logger.exception("snapshot worker could not open the database -- no alert pictures")
            return
        while True:
            item = self._q.get()
            if item is None:
                return
            alert_id, camera = item
            url = f"{self.base}/api/frame.jpeg?src={urllib.parse.quote(camera)}"
            try:
                data = self.fetch(url)
                if not data:
                    raise ValueError("empty frame")
                rel = f"alerts/{alert_id}.jpg"
                path = self.crops_dir / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                set_alert_snapshot(conn, alert_id, rel)
            except Exception as exc:
                logger.warning("alert %s: no snapshot from %s (%s)", alert_id, camera, exc)
