"""The boxes a module draws on its alerts' clips: the objects an alert can be about -- PPE's workers, the pose module's
people (the fall service's subject), fire and smoke, people in a zone -- written to the camera's overlay stream
(EngineServices.overlay, `iris:overlay:<stream>`), which the app's clip builder draws (app/overlay_track.py: the
alert's own object in the alert colour, fire and smoke always).

A box shape is {"type": "box", "rect": [left, top, width, height] (mux px), "style", "id"?, "label"?}: style "alert"
for an object an alert can name by its tracker id (drawn when it is the alert's subject, or when the alert names
none), "fire" / "smoke" for the hazard itself. At most one entry per `interval` per stream, so about three minutes stay
in the stream (OVERLAY_MAXLEN) -- a clip is built twenty to thirty seconds after its alert, with ten seconds either
side. Only a pipeline whose config says `overlay_boxes: true` writes them (the safety stack's roles,
products/safety/safety.yaml): every other product's Redis and clips stay as they were."""
from __future__ import annotations

OVERLAY_INTERVAL_S = 0.1


def box(rect, style: str, oid=None, label: "str | None" = None) -> dict:
    """A box from (left, top, width, height)."""
    out = {"type": "box", "rect": [round(float(v), 1) for v in rect], "style": style}
    if oid is not None:
        out["id"] = int(oid)
    if label:
        out["label"] = label
    return out


def corners_box(corners, style: str, oid=None, label: "str | None" = None) -> dict:
    """A box from (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = (float(v) for v in corners)
    return box((x1, y1, x2 - x1, y2 - y1), style, oid, label)


class OverlayWriter:
    """One module's writes to its cameras' overlay streams, at most one per `interval` of frame time per stream."""

    def __init__(self, services, module: str, interval: float = OVERLAY_INTERVAL_S):
        self._services, self._module, self._interval = services, module, interval
        self._last: dict = {}
        self.enabled = bool((getattr(services, "config", None) or {}).get("overlay_boxes", False))

    def write(self, stream_id: str, pts: float, shapes: list, at: float = 0.0) -> bool:
        """Write `shapes` at frame time `pts` (True), or not (False) when there is nothing to draw or the last write
        is too recent -- a frame time earlier than the last one (a looping file, a restarted source) writes again.
        `at` is the frame's arrival (FrameView.arrived): it places the entry on a smart-record clip."""
        last = self._last.get(stream_id)
        if not self.enabled or not shapes or (last is not None and 0.0 <= pts - last < self._interval - 1e-6):
            return False
        self._last[stream_id] = pts
        self._services.overlay(stream_id, self._module, pts, self._services.mux_size, shapes, at)
        return True
