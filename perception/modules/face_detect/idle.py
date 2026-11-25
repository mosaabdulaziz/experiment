"""Auto-toggle the face SGIE (sgie_face) on gallery registration state.

Pure stdlib — no gi / GStreamer imports, so this module is unit-testable on
the host (the sidecar's GPU code lives in sidecar.py; this is the policy).

Why `infer-on-class-ids` is the toggle (verified in the DeepStream 9.0 source):
  The nvinfer `interval` property only gates the FULL-FRAME path
  (gst_nvinfer_process_full_frame, gstnvinfer.cpp:1715 — "Process batch only
  when interval_counter is 0"). Secondary GIEs (process-mode=2, which the face
  SGIE uses) run the OBJECT path (gst_nvinfer_process_objects, gstnvinfer.cpp
  :1922), which has NO interval check — a huge `interval` does NOT stop a
  secondary GIE's inference. The object path's gate is `infer-on-class-ids`
  (GST_PARAM_MUTABLE_READY, settable while PLAYING): should_infer_object()
  returns FALSE for every object whose class_id is not in the list, so
  pointing it at a class that never appears (999) makes the SGIE queue zero
  inferences — zero GPU — while buffers still flow through with the
  pgie/tracker metadata. Restoring the original class list re-enables on the
  very next batch.

Rule (what the sidecar's monitor applies):
  the SGIE runs when the gallery has registered faces OR face capture is
  enabled (capture *consumes* SGIE face detections to build the gallery — the
  gallery can never fill if the SGIE is off while capture is on). Disabled
  only when both are off.
"""

import logging

logger = logging.getLogger("iris.sidecar")

# infer-on-class-ids value that matches no object class (the yolo pgie model
# has ~80 classes, so 999 never appears). With no matching object, the SGIE
# queues zero object-level inferences and does zero GPU work.
SGIE_DISABLED_CLASS = "999"

FACE_GALLERY_KEY = "iris:face:gallery"


def sgie_should_run(gallery_count: int, face_capture_enabled: bool) -> bool:
    """Whether the face SGIE should infer, given gallery + capture state."""
    return gallery_count > 0 or face_capture_enabled


class SgieGalleryMonitor:
    """Poll the face gallery and flip the SGIE's `infer-on-class-ids`.

    Wired by sidecar.py's main() with a GLib timeout (runs on the main loop
    thread, same thread deepstream-app uses for its event-based updates).
    Callables keep this importable off-GPU:

      hlen()         -> current gallery size (Redis HLEN)
      set_enabled()  -> apply the enabled/disabled state to the sgie_face
                        element (the sidecar maps this onto
                        `infer-on-class-ids`, restoring the element's original
                        class list when enabled)
    """

    def __init__(self, hlen, set_enabled, face_capture_enabled,
                 poll_seconds: float = 10.0):
        self._hlen = hlen
        self._set_enabled = set_enabled
        self._face_capture_enabled = bool(face_capture_enabled)
        self._poll_seconds = poll_seconds
        self._last = None      # last applied enabled state (None = not applied)
        self._last_count = 0   # gallery size seen at last apply, for logging

    @property
    def current_enabled(self):
        return self._last

    def poll(self):
        """One check: apply a transition if the state changed.

        Returns the enabled state just applied (True/False), or None if
        nothing changed.
        """
        try:
            count = int(self._hlen() or 0)
        except Exception as exc:
            logger.warning("sgie gallery poll failed (non-fatal): %s", exc)
            return None
        want = sgie_should_run(count, self._face_capture_enabled)
        if want == self._last:
            return None
        self._set_enabled(want)
        prev = self._last
        self._last = want
        self._last_count = count
        state = "enabled" if want else "disabled"
        if prev is None:
            logger.info("sgie_face auto-%s at boot (gallery=%d, capture=%s)",
                        state, count, self._face_capture_enabled)
        else:
            logger.info("sgie_face auto-%s (gallery=%d, capture=%s)",
                        state, count, self._face_capture_enabled)
        return want
