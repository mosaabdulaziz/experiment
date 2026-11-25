"""Per-stream frame publisher: draws the preview, encodes JPEG, writes iris:meta / iris:jpeg."""

import collections
import queue
import sys
import threading
import time
import json
import redis
import cv2
from .sources import ViewerGate
from .jpeg_encoder import GpuJpegEncoder
from . import telemetry

from .gst_util import _DISCARD_QUEUE, logger




META_VERSION = "1"              # the shape of an iris:meta entry; see the `v` field where entries are written
BOX_DEFAULT = (0, 255, 0)       # BGR green: what a box is drawn in unless a module says otherwise


@telemetry.timeit("publish.draw")
def _draw_and_encode(img, boxes, gcfg, encoder=None, lock=None, annotations=None, draws=None):
    """Draw THIS frame's own boxes on a CPU BGR image, then JPEG-encode it.

    Runs on the frame-publisher worker thread, never the streaming thread.
    `boxes` are this frame_meta's detections only, so streams cannot cross-
    contaminate — this replaces nvdsosd, whose batched rendering was mixing
    boxes across streams. A box is green unless a module annotated it with a
    colour of its own (a zone violation, a hazard).

    A module can replace a box's label (a recognised name, "" to hide it) with an annotation, and paint its own
    shapes after the boxes with a draw function (FrameOutput.draw).

    When `encoder` (a GpuJpegEncoder) is given, the JPEG is encoded on the GPU
    (nvjpegenc); if the GPU path returns None the CPU cv2.imencode path is used
    as fallback.
    """
    annotations = annotations or {}
    try:
        sx = gcfg.capture_w / gcfg.mux_w   # detection coords are in mux res; scale to capture res
        sy = gcfg.capture_h / gcfg.mux_h

        # `boxes` arrive in draw order (the probe sorts them by layer, so face boxes -- where a name is written --
        # come before the person boxes)
        for box in boxes:
            left, top, w, h, label, conf, oid = box.left, box.top, box.width, box.height, box.label, box.shown_conf, box.id
            note = annotations.get(box)
            if note is not None and note.hidden:
                continue
            if note is not None and note.box is not None:
                left, top, w, h = note.box          # a module's steadier box for this object
            x1, y1 = int(left * sx), int(top * sy)
            x2, y2 = int((left + w) * sx), int((top + h) * sy)
            color = note.color if note is not None and note.color is not None else BOX_DEFAULT
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            if oid == -1 or oid < 0:
                text = f"{label} {conf:.2f}"
            else:
                text = f"{label} {conf:.2f} #{oid}"

            if note is not None and note.text is not None:
                text = note.text            # a module's own label ("" hides the default one)

            if text:
                scale = getattr(gcfg, "label_scale", 0.5)
                org = (x1, max(18, y1 - 8)) if scale > 0.5 else (x1, max(12, y1 - 5))
                cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                            scale, color, getattr(gcfg, "label_thickness", 1), cv2.LINE_AA)

            if note is not None:
                for extra_text, extra_color, extra_scale, extra_thick in note.extra:
                    cv2.putText(img, extra_text, (x1, y2 + 22), cv2.FONT_HERSHEY_SIMPLEX,
                                extra_scale, extra_color, extra_thick, cv2.LINE_AA)
        for draw in draws or ():
            draw(img, sx, sy)
        if encoder is not None:
            # The GPU encoder mini-pipeline is SHARED across the per-stream
            # publisher workers. Serialize the encode call: GStreamer pads are
            # thread-safe, but concurrent try-pull-sample on the shared appsink
            # (drop=True, max-buffers=2) makes a transient timeout likely during
            # a 14-source batch burst — and 3 consecutive timeouts permanently
            # poison the shared encoder for every stream (see _note_failure).
            # With the lock, encodes are ~6.5ms each and never contend.
            if lock is not None:
                with lock:
                    jpeg = encoder.encode(img)
            else:
                jpeg = encoder.encode(img)
            if jpeg is not None:
                return jpeg
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, gcfg.jpeg_quality])
        return buf.tobytes() if ok else None
    except Exception as exc:
        sys.stderr.write(f"render failed: {exc}\n")
        return None



class _FramePublisher:
    """Render + encode + publish annotated JPEGs off the GStreamer streaming thread.

    The probe still maps the NVMM surface and copies it to CPU (buffer lifetime
    ties that to the streaming thread), but box drawing, JPEG encode, and the
    Redis writes move to the workers. Without that, cv2.imencode at stream_fps and
    the Redis set/xadd ran on the streaming thread and throttled the whole
    pipeline: at 1280x720 imencode alone is ~5-15ms per frame per stream, plus a
    network round-trip per publish.

    One worker thread + bounded queue PER STREAM (lazily created on first
    submit). This keeps each source's delivery cadence independent: the mux
    delivers all sources' frames in one batch every ~1/analysis_fps s, and a
    single shared queue with one worker turned that batch into burst-then-drop
    (the newest frame won, so other streams' frames were discarded and the site
    saw ~1 fps with big jumps). Per-stream queues mean a 14-source batch is
    handled by 14 workers, each publishing its own frame at the throttle cadence.

    Each per-stream queue is bounded; on overflow the OLDEST pending job is
    dropped (the newest frame wins — latency stays flat instead of stalling the
    pipeline behind the encoder). The GPU JPEG encoder is shared across workers
    (GStreamer elements are thread-safe; appsrc/appsink serialize naturally).
    """

    def __init__(self, gcfg, max_pending=8):
        self.gcfg = gcfg
        self._max_pending = max_pending
        # stream_id -> (queue, thread)
        self._streams: dict = {}
        self._lock = threading.Lock()
        # Serializes the SHARED GPU encoder across per-stream workers (see
        # _draw_and_encode). cv2.imencode fallback path is lock-free.
        self._encoder_lock = threading.Lock()
        self._stopped = False
        self._drops = 0
        self._last_drop_log = 0.0
        # per stream: frames dropped when its queue was full, and how long its last jobs waited in
        # the queue before a worker took them (a growing wait = the worker cannot keep up)
        self._drops_by: dict = collections.defaultdict(int)
        self._waits: dict = {}
        # GPU JPEG encode (nvjpegenc) when enabled and available; cv2.imencode
        # otherwise. Building the mini-pipeline here (main thread) means a
        # missing element disables it once at startup, with the workers never
        # touching a half-built pipeline.
        # In inference-only mode (pipeline_preview=false) no JPEG is ever
        # encoded, so skip building the encoder and the viewer gate entirely.
        self._gpu_jpeg = None
        self._encoders: list = []         # [(GpuJpegEncoder, its lock)]; a stream always uses the same one
        self._enc_of: dict = {}
        if getattr(gcfg, "gpu_jpeg", True) and getattr(gcfg, "pipeline_preview", True):
            for i in range(max(1, int(getattr(gcfg, "jpeg_encoders", 1)))):
                try:
                    enc = GpuJpegEncoder(gcfg.capture_w, gcfg.capture_h, gcfg.jpeg_quality,
                                         stage="gpu.jpeg" if i == 0 and getattr(gcfg, "jpeg_encoders", 1) == 1
                                         else f"gpu.jpeg.{i}")
                    self._encoders.append((enc, threading.Lock()))
                except Exception as exc:
                    logger.warning("GPU JPEG encoder %d unavailable (%s)", i, exc)
                    break
            if self._encoders:
                self._gpu_jpeg = self._encoders[0][0]
                logger.info("frame publisher: %d GPU JPEG encoder(s) active (nvjpegenc, quality=%d)",
                            len(self._encoders), gcfg.jpeg_quality)
            else:
                logger.warning("no GPU JPEG encoder available; using cv2.imencode")
        # A preview job the queue had no room for is not thrown away: its detection metadata is still
        # published, only the drawing/encoding is skipped ("shed"). It goes through this small queue
        # and worker. Only if that queue is also full is anything really lost (self._drops).
        self._shed_by: dict = collections.defaultdict(int)
        self._meta_q: queue.Queue = queue.Queue(maxsize=2000)
        self._meta_thread = threading.Thread(target=self._run_meta_only, daemon=True,
                                             name="frame-publisher-meta-only")
        self._meta_thread.start()
        # When viewer-gated preview is on, skip the encode for streams nobody is
        # watching (cached iris:watched:<id> check). None = always encode.
        self._viewer_gate = (ViewerGate(gcfg.r)
                             if getattr(gcfg, "viewer_gated_preview", False)
                             and getattr(gcfg, "pipeline_preview", True) else None)

    def _worker_for(self, stream_id):
        """Return this stream's (queue, thread), creating both on first use.

        After stop() the publisher returns a throwaway sink so late probe
        submissions are dropped instead of spawning an unjoinable worker.
        """
        with self._lock:
            if self._stopped:
                return _DISCARD_QUEUE, None
            pair = self._streams.get(stream_id)
            if pair is not None:
                return pair
            q = queue.Queue(maxsize=self._max_pending)
            self._waits[stream_id] = collections.deque(maxlen=30)
            if self._encoders:
                self._enc_of[stream_id] = len(self._enc_of) % len(self._encoders)
            t = threading.Thread(target=self._run, args=(q, stream_id), daemon=True,
                                 name=f"frame-publisher-{stream_id}")
            t.start()
            self._streams[stream_id] = (q, t)
            return q, t

    def submit(self, job):
        stream_id = job["st"].id
        q, _t = self._worker_for(stream_id)
        if q is _DISCARD_QUEUE:
            return  # publisher is stopping; drop rather than spawn a worker
        job["_enq"] = time.monotonic()
        try:
            q.put_nowait(job)
        except queue.Full:
            # The worker cannot keep up. Make room by SHEDDING the oldest pending job: its preview
            # (draw + JPEG, the expensive part) is skipped but its detection metadata is still
            # published, so the tracking app loses nothing. The newest frame still wins the queue.
            try:
                old = q.get_nowait()
            except queue.Empty:
                old = None
            try:
                q.put_nowait(job)
            except queue.Full:
                old = job          # lost the race for the slot: shed the new one instead
            if old is not None:
                self._shed(stream_id, old)

    def _shed(self, stream_id, job) -> None:
        """Skip the preview of ``job`` but keep its metadata (see submit)."""
        with self._lock:
            self._shed_by[stream_id] += 1
        if job.get("need_meta"):
            job["need_stream"] = False
            try:
                self._meta_q.put_nowait(job)
                return
            except queue.Full:
                pass
            with self._lock:           # the safety net is full too: this one really is lost
                self._drops += 1
                self._drops_by[stream_id] += 1
                drops = self._drops
            now = time.monotonic()
            if now - self._last_drop_log > 5.0:
                self._last_drop_log = now
                logger.warning("frame publisher LOST detection data (%s): even the meta-only queue is full "
                               "-- %d jobs lost so far", stream_id, drops)

    def _run_meta_only(self):
        while True:
            job = self._meta_q.get()
            if job is None:
                return
            try:
                self._publish(job)
            except Exception as exc:
                logger.error("meta-only publish failed (continuing): %s", exc)

    def shed_count(self, stream_id) -> int:
        with self._lock:
            return self._shed_by.get(stream_id, 0)

    def encoder_for(self, stream_id):
        """(encoder, lock) this stream encodes with, or (None, None) for the cv2 fallback."""
        if not self._encoders:
            return None, None
        enc, lock = self._encoders[self._enc_of.get(stream_id, 0) % len(self._encoders)]
        return enc, lock

    def stop(self):
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
            streams = list(self._streams.items())
        for _sid, (q, t) in streams:
            q.put(None)
        for _sid, (q, t) in streams:
            t.join(timeout=10.0)
        self._meta_q.put(None)
        self._meta_thread.join(timeout=5.0)
        for enc, _lock in self._encoders:
            enc.close()

    def pending(self) -> int:
        """Total queued jobs across all per-stream workers (telemetry gauge).

        Snapshot under the lock so the telemetry flush can't race _worker_for's
        insert while iterating _streams."""
        with self._lock:
            return sum(q.qsize() for q, _t in self._streams.values())

    def stream_stats(self, stream_id) -> tuple[int, int, float]:
        """(queue depth, frames dropped so far, worst recent queue wait in ms) of one stream."""
        with self._lock:
            pair = self._streams.get(stream_id)
            waits = self._waits.get(stream_id)
            return (pair[0].qsize() if pair else 0, self._drops_by.get(stream_id, 0),
                    max(waits) * 1000.0 if waits else 0.0)

    def _run(self, q, stream_id=None):
        while True:
            job = q.get()
            if job is None:
                return
            if stream_id is not None and "_enq" in job:
                self._waits[stream_id].append(time.monotonic() - job["_enq"])
            try:
                self._publish(job)
            except Exception as exc:
                logger.error("frame publish failed (continuing): %s", exc)
            finally:
                q.task_done()

    @telemetry.timeit("publish")
    def _publish(self, job):
        g = self.gcfg
        st = job["st"]
        # Viewer-gated preview: skip the (expensive, lock-serialized) JPEG encode
        # for streams nobody is watching. The job's need_meta path still runs
        # (dets/recognition feed analysis) but embeds no jpeg for gated streams.
        encode = (job["need_stream"]
                  and (self._viewer_gate is None
                       or self._viewer_gate.is_watched(st.key_id)))
        jpeg_bytes = None
        # A module that crops faces or bodies out of the frame must see the CLEAN frame (2026-09-12): _draw_and_encode
        # draws boxes and labels onto frame_bgr IN PLACE, and the face recognizer used to get that same array after
        # it -- so every recognized face had the green face box (sometimes a name label too) painted on it.
        # Copy first, only for frames that are both drawn and handed to a module.
        hooks = job.get("publish_hooks")
        clean = job.get("frame_bgr")
        if encode and hooks and clean is not None:
            clean = clean.copy()
        if encode:
            jpeg_bytes = _draw_and_encode(
                job["frame_bgr"], job["boxes"], g,
                encoder=self.encoder_for(st.id)[0], lock=self.encoder_for(st.id)[1],
                annotations=job.get("annotations"), draws=job.get("draws"))
        with telemetry.stage("publish.redis"):
            if job["need_meta"]:
                # "v" names the SHAPE of this entry (the legacy one: `dets` of 3 fields, or 7 when a module asks for the
                # boxes, plus whatever fields modules add). A consumer that pins the shape can refuse a newer one.
                fields = {"v": META_VERSION, "frame_num": job["frame_num"], "pts": f"{job['pts']:.3f}",
                          "dets": json.dumps(job["dets"])}
                # Additive (2026-09-12, issue 1 Phase 2): the camera's own
                # capture time from RTCP, ns since the Unix epoch. Omitted when
                # the camera sent none -- a consumer then uses server
                # time for this camera (acl.CameraClock).
                if job.get("ntp_ts"):
                    fields["ntp_ts"] = str(job["ntp_ts"])
                for key, value in (job.get("fields") or {}).items():
                    fields[key] = json.dumps(value)        # whatever the modules published
                if jpeg_bytes is not None:
                    fields["jpeg"] = jpeg_bytes
                try:
                    g.r.xadd(st.meta_key, fields, maxlen=g.maxlen, approximate=True)
                except redis.RedisError as exc:
                    logger.error("[%s] redis xadd failed: %s", st.id, exc)

                # Modules that took the frame's pixels (face recognition, picture capture) get the clean frame to hand
                # to workers of their own. In inference-only mode frame_bgr is only copied when a module asked for it.
                if hooks and clean is not None:
                    for hook in hooks:
                        try:
                            hook(clean)
                        except Exception as exc:  # noqa: BLE001 -- a module's hook must not stop the publish
                            logger.error("[%s] publish hook failed (continuing): %s", st.id, exc)

            if job["need_stream"] and jpeg_bytes is not None:
                try:
                    g.r.set(st.jpeg_key, jpeg_bytes)
                except redis.RedisError as exc:
                    logger.error("[%s] redis set failed: %s", st.id, exc)
