"""Rolling-segment recording: encode branch, NVENC scheduler, segment uploader."""

import os
import queue
import select
import shutil
import subprocess
import sys
import threading
import time
import uuid
from .segment_index import (recorder_clock, SegmentBoundaryTracker, StreamClockOffsets,  # noqa: E402
                           has_parameter_sets)
from . import telemetry

import gi

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

from .gst_util import _make, logger




def _sample_pts(sample):
    """Seconds of the first buffer in a GstSample, or None when unavailable.

    splitmuxsink hands `format-location-full` the sample that opens the new
    fragment; its buffer pts is the rotation boundary.
    """
    try:
        buf = sample.get_buffer()
        if buf is None:
            return None
        pts = buf.pts
        if pts == Gst.CLOCK_TIME_NONE:
            return None
        return pts / 1e9
    except Exception:
        return None



class _SegmentUploader:
    """Upload + index finalized recording fragments off the GStreamer thread.

    `format-location-full` fires on the recording branch's streaming thread. Doing
    the MinIO upload, redis index and file delete there (as the original code did)
    blocks that thread on every rotation -- and a slow object store backs up the
    encoder queue, which via the tee pressures the whole source. All of that work
    is pushed onto this worker's queue instead; the callback only does cheap
    bookkeeping and returns the next path immediately.
    """

    def __init__(self, gcfg):
        self.gcfg = gcfg
        self._q = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="segment-uploader")
        self._last_backlog_log = 0.0
        self._thread.start()

    def submit(self, source_id, prev_path, span, offset):
        self._q.put((source_id, prev_path, span, offset))
        # Segments are the source of truth for alert clips, so we never drop
        # them (unlike _FramePublisher); but a backing-up queue means the
        # object store is behind and footage is aging on disk — surface it.
        now = time.monotonic()
        if self._q.qsize() > 50 and now - self._last_backlog_log > 30.0:
            self._last_backlog_log = now
            logger.warning("segment uploader backlog: %d segments pending "
                           "(object store slow?); segments still rotate on "
                           "schedule", self._q.qsize())

    def stop(self):
        """Drain remaining jobs, then stop the worker. Called on shutdown; the
        pipeline has already been set to NULL, so no new submissions arrive."""
        self._q.put(None)
        self._thread.join(timeout=10.0)

    def _run(self):
        while True:
            job = self._q.get()
            if job is None:
                return
            try:
                self._upload(*job)
            except Exception as exc:
                sys.stderr.write(f"[{job[0]}] segment upload failed: {exc}\n")
            finally:
                self._q.task_done()

    @telemetry.timeit("upload")
    def _upload(self, source_id, prev_path, span, offset):
        if span is None:
            raise RuntimeError("boundary tracker desynced from fragment state")
        if offset is None:
            # Probe has not produced a frame for this source yet, so the two
            # clocks cannot be related. Index uncorrected rather than drop the
            # footage; this is the same accuracy the pre-fix code always had.
            logger.warning("[%s] pts clock offset unavailable; indexing this "
                           "segment in recorder pts", source_id)
        if not has_parameter_sets(prev_path):
            # ADVISORY ONLY -- index it anyway. Refusing to index these was
            # tried and was strictly worse: on a pipeline where most fragments
            # lack an SPS, no clip window is ever covered and EVERY alert
            # resolves 'unavailable'. A fragment without parameter sets is
            # still usable behind one that has them; only leading a concat is
            # fatal, and app/clip_builder.py drops it there with an
            # authoritative ffprobe check. This line stays because it is the
            # only signal that the encoder is not emitting SPS per IDR.
            logger.warning(
                "[%s] fragment %s carries no SPS — it cannot lead a clip "
                "(indexed anyway; the builder will skip it if it would)",
                source_id, os.path.basename(prev_path))
        start_pts, end_pts = span
        if offset is not None:
            start_pts, end_pts = start_pts + offset, end_pts + offset
        # Same scheme as SegmentStore.put: the uuid keeps the key unique.
        # Without it every run regenerates the SAME low-pts key space (pts
        # restarts near zero on each sidecar start), so one run's segments
        # silently overwrite another's -- and a stale index entry pointing
        # at a reused name resolves to footage from a different run. That
        # window is exactly the one the first alert of a run reads from.
        object_key = (f"{source_id}/{start_pts:.3f}_{end_pts:.3f}"
                      f"_{uuid.uuid4().hex[:8]}.ts")
        try:
            if self.gcfg.storage_backend == "minio":
                self.gcfg.minio.fput_object(self.gcfg.record_bucket, object_key, prev_path)
            else:
                dest = os.path.join(self.gcfg.local_segments_dir, object_key)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copy2(prev_path, dest)
            try:
                self.gcfg.r.xadd(f"iris:segments:{self.gcfg.key_id(source_id)}", {
                    "start_pts": f"{start_pts:.3f}", "end_pts": f"{end_pts:.3f}",
                    "object_key": object_key, "recorded_at": f"{time.time():.3f}",
                })
                if self.gcfg.segment_debug_logging:
                    logger.info("[%s] segment recorded %.3f-%.3f -> %s (%s)",
                               source_id, start_pts, end_pts, object_key,
                               self.gcfg.storage_backend)
            except Exception:
                # If xadd fails after the bytes are already stored, remove the
                # orphan. Swallow any secondary exception from cleanup so the
                # original error propagates.
                try:
                    if self.gcfg.storage_backend == "minio":
                        self.gcfg.minio.remove_object(self.gcfg.record_bucket, object_key)
                    else:
                        os.remove(os.path.join(self.gcfg.local_segments_dir, object_key))
                except Exception:
                    pass
                raise
        finally:
            try:
                os.remove(prev_path)
            except OSError:
                pass



def _make_rotation_callback(gcfg, source_id, uploader, clock_sample=None):
    """`splitmuxsink`'s `format-location-full` signal fires just before it opens
    each NEW fragment, so by the time it fires for fragment N+1 the PREVIOUS
    fragment (N) is already finalized on disk. Hand that previous fragment to the
    uploader worker (which uploads+indexes it on a background thread), then hand
    back this fragment's own path.

    Segment spans come from the real buffer pts carried by the signal's sample
    (see SegmentBoundaryTracker), corrected by a per-stream offset measured at
    the first rotation (see StreamClockOffsets) so they are written directly in
    the probe's clock -- directly comparable with the post-mux `frame_meta.buf_pts`
    the probe stamps on alerts. This is what lets an alert clip be cut tightly
    around its own fired moment.
    """
    tracker = SegmentBoundaryTracker(gcfg.record_seconds)
    state = {"prev_path": None, "seg_index": 0, "offset_logged": False}

    def _on_format_location_full(_sink, _fragment_id, sample, _udata=None):
        with telemetry.stage("rotation"):
            boundary = _sample_pts(sample)
            if gcfg.pts_parity_logging:
                # Compare this against the probe's parity line: both print pts minus
                # wall clock, and the two constants must agree if the pre-mux record
                # branch and the post-mux probe share a time base.
                if boundary is not None:
                    logger.info("[%s] parity rotate pts-wall=%.3f", source_id,
                                boundary - time.time())

            # the latest encoded frame's own (pts, wall), not the boundary with now: see recorder_clock
            offset = gcfg.clock_offsets.offset(source_id, *recorder_clock(clock_sample or {}, boundary, time.time()))
            prev_offset = state.get("last_logged_offset")
            if offset is not None and (prev_offset is None or abs(offset - prev_offset) > 0.5):
                state["last_logged_offset"] = offset
                if prev_offset is not None:
                    state["steady_offset"] = offset
                logger.info("[%s] pts clock offset %s at %+.3fs (recorder -> probe)",
                            source_id, "initialized" if prev_offset is None else "adjusted", offset)

            # Cheap per-segment self-check (recorded content vs wall clock): the
            # recorder->probe offset is the conversion constant between the recorder's
            # content clock and the wall-derived probe clock that alert anchors are
            # stamped in. Measured behaviour: initialized at +3.0s, adjusted once to
            # +5.7s at boot, then flat for the whole run. If it ever moves again,
            # recorded frames are falling behind (or ahead of) the wall clock the
            # segment index and clip windows are keyed on -- a clip cut around an
            # alert's wall anchor then shows footage captured at a different moment,
            # the "clip shows footage from before the incident" symptom. Warn when it
            # drifts > 1s for several rotations, throttled so a persistently-off
            # recorder yells without flooding the log.
            steady = state.get("steady_offset")
            if offset is not None and steady is not None:
                drift = offset - steady
                if abs(drift) > 1.0:
                    state["drift_count"] = state.get("drift_count", 0) + 1
                    last_warn = state.get("drift_last_warn", 0.0)
                    if state["drift_count"] >= 3 and time.monotonic() - last_warn > 60.0:
                        state["drift_last_warn"] = time.monotonic()
                        logger.warning(
                            "[%s] segment clock drift ~%+.1fs: recorder->probe offset "
                            "%+.3fs vs steady %+.3fs over %d rotations -- recorded content "
                            "is misaligned with the wall clock alert clips are anchored "
                            "on; alert clips may show footage from the wrong moment",
                            source_id, drift, offset, steady, state["drift_count"])
                else:
                    state["drift_count"] = 0

            span = tracker.rotate(boundary)

            prev_path = state["prev_path"]
            if prev_path is not None:
                uploader.submit(source_id, prev_path, span, offset)
            path = f"{gcfg.record_tmp_dir}/{source_id}/{state['seg_index']:05d}.ts"
            state["seg_index"] += 1
            state["prev_path"] = path
            return path
    return _on_format_location_full



def _flush_svr_dir(gcfg, sources):
    """Delete smart-record output from previous runs.

    Same lesson as _flush_segment_index: files nothing points at are unreachable by
    any expiry path and accumulate for the life of the deployment. A clip is claimed
    by an app-side Redis announcement and those do not survive a restart, so every
    file present at startup is already orphaned. (In steady state the directory stays
    near-empty anyway -- ClipStore.add MOVES the file out when an alert claims it.)
    """
    removed = 0
    for s in sources:
        prefix = f"{s['id']}_"
        try:
            for name in os.listdir(gcfg.svr_dir):
                if not name.startswith(prefix):
                    continue
                try:
                    os.remove(os.path.join(gcfg.svr_dir, name))
                    removed += 1
                except OSError:
                    pass
        except FileNotFoundError:
            return          # nothing recorded yet
        except Exception as exc:
            logger.warning("smart-record dir purge failed: %s", exc)
            return
    if removed:
        logger.info("discarded %d smart-record file(s) from a previous run", removed)



def _flush_segment_index(gcfg, sources):
    """Drop this run's sources from the segment index, and every byte stored for them.

    Segments indexed under a previous process's pts origin are not comparable with
    this run's, so a lookup spanning the restart would return the wrong footage.
    Segments only live `retention_seconds` anyway, and a restart invalidates their
    clock, so discarding them is strictly better than serving them.

    The whole prefix is purged, not just what the index still names. Deleting only
    indexed objects leaves an orphan behind whenever an index entry is lost (a Redis
    restart without persistence, or a flush that raced the write) -- and an orphan is
    unreachable by `SegmentStore.expire`, which can only delete files an entry points
    at. Those accumulate for as long as the deployment lives: this was found with
    three weeks of segments, ~1 GB, under a 30-minute retention.
    """
    for s in sources:
        sid = s["id"]
        try:
            gcfg.r.delete(f"iris:segments:{gcfg.key_id(sid)}")
        except Exception as exc:
            logger.warning("[%s] segment index flush failed: %s", sid, exc)
        removed = 0
        try:
            if gcfg.storage_backend == "minio":
                for obj in gcfg.minio.list_objects(gcfg.record_bucket,
                                                   prefix=f"{sid}/", recursive=True):
                    try:
                        gcfg.minio.remove_object(gcfg.record_bucket, obj.object_name)
                        removed += 1
                    except Exception:
                        pass
            else:
                src_dir = os.path.join(gcfg.local_segments_dir, sid)
                for name in os.listdir(src_dir):
                    path = os.path.join(src_dir, name)
                    try:
                        if os.path.isfile(path):
                            os.remove(path)
                            removed += 1
                    except OSError:
                        pass
        except FileNotFoundError:
            pass        # nothing recorded for this source yet
        except Exception as exc:
            logger.warning("[%s] segment byte purge failed: %s", sid, exc)
        if removed:
            logger.info("[%s] discarded %d segment file(s) from a previous run", sid, removed)



def _build_record_branch(pipeline, gcfg, source_id, idx, uploader):
    """queue -> nvv4l2h264enc -> h264parse -> splitmuxsink, rotating a new
    segment every gcfg.record_seconds. Entirely independent of nvinfer/nvtracker/
    the batched mux — recording never competes with inference for GPU throughput
    and keeps running even if analysis is paused. `uploader` receives each
    finalized fragment and uploads+indexes it off the streaming thread."""
    queue = _make("queue", f"rec_queue_{idx}")
    enc = _make("nvv4l2h264enc", f"rec_enc_{idx}")
    enc.set_property("bitrate", gcfg.record_bitrate_kbps * 1000)
    # Each rotated segment is later concatenated standalone by ffmpeg
    # (app/clip_builder.py), so it must carry its own SPS/PPS — without this,
    # a segment cut mid-GOP (or one whose encoder only sent parameter sets
    # once, out-of-band, at stream start) fails to decode on its own
    # ("non-existing PPS referenced", "decode_slice_header error").
    try:
        enc.set_property("insert-sps-pps", True)
        # Read back rather than trusting the set: on some nvv4l2h264enc builds the
        # property exists and accepts a value that the encoder then ignores, which
        # is indistinguishable from success unless it is logged.
        logger.info("[%s] recorder insert-sps-pps=%s", source_id,
                    enc.get_property("insert-sps-pps"))
    except Exception as exc:
        logger.warning("[%s] insert-sps-pps not settable (non-fatal): %s", source_id, exc)
    try:
        enc.set_property("iframeinterval", 30)  # keyframe at least once/sec typical fps
    except Exception as exc:
        logger.warning("[%s] iframeinterval not settable (non-fatal): %s", source_id, exc)
    try:
        # insert-sps-pps attaches parameter sets to IDR frames, and IDR cadence is
        # NOT iframeinterval -- that governs I-frames, which need not be IDRs. Left
        # unset, IDRs are rare, so most fragments open on a plain I-frame with no
        # SPS/PPS and cannot lead an ffmpeg concat. Observed directly: fragments
        # 00001-00004 of a run had no SPS while 00000 and 00005+ did, because only
        # those happened to contain one of the sparse IDRs.
        enc.set_property("idrinterval", 30)
        logger.info("[%s] recorder idrinterval=%s", source_id,
                    enc.get_property("idrinterval"))
    except Exception as exc:
        logger.warning("[%s] idrinterval not settable (non-fatal): %s -- fragments may "
                       "lack SPS and clips will be shorter", source_id, exc)
    parse = _make("h264parse", f"rec_parse_{idx}")
    # Belt-and-suspenders: h264parse's own config-interval=-1 re-inserts
    # SPS/PPS before every IDR at the parser level, regardless of what the
    # encoder emits — this is the standard, version-independent fix for
    # splitmuxsink segments that must each be independently decodable.
    parse.set_property("config-interval", -1)

    # file-loop restarts the source by seeking to 0, so its PTS jumps backwards
    # (~ -clip_duration) on every loop. splitmuxsink derives GOP duration from PTS
    # and posts a fatal ERROR ("Queued GOP time is negative") on that jump, which
    # kills this branch, backs up the tee, and freezes the WHOLE pipeline (the
    # analysis/JPEG branch survives loops only because nvstreammux re-times frames).
    # Rewrite PTS/DTS on the ENCODED stream (parse src -> splitmuxsink) to be
    # monotonically increasing across loops so the muxer never sees a reset. We tap
    # here, not before the tee, because pre-tee buffers are shared with the analysis
    # branch and must not be mutated. No-op for RTSP (PTS already monotonic).
    # Max value of a guint64 (same numeric value as Gst.CLOCK_TIME_NONE, the "no
    # timestamp" sentinel).  Any PTS/DTS written back to a GstBuffer must stay within
    # 0.._MAX_TS or the property setter raises OverflowError and kills the pipeline.
    _MAX_TS = (1 << 64) - 1
    _rec_ts = {"offset": 0, "last_ref": None, "last_out_ref": 0, "first_segment_seen": False}
    clock_sample: dict = {}     # the latest encoded frame's (pts s, wall s), for the clock correction (recorder_clock)

    def _monotonic_ts(_pad, info, _u=None):
        if info.type & Gst.PadProbeType.EVENT_DOWNSTREAM:
            event = info.get_event()
            if event and event.type == Gst.EventType.SEGMENT:
                if not _rec_ts["first_segment_seen"]:
                    _rec_ts["first_segment_seen"] = True
                    return Gst.PadProbeReturn.OK
                else:
                    return Gst.PadProbeReturn.DROP
            return Gst.PadProbeReturn.OK

        if info.type & Gst.PadProbeType.BUFFER:
            buf = info.get_buffer()
            if buf is None:
                return Gst.PadProbeReturn.OK

            ref_ts = buf.dts if buf.dts != Gst.CLOCK_TIME_NONE else buf.pts
            if ref_ts == Gst.CLOCK_TIME_NONE:
                return Gst.PadProbeReturn.OK

            prev_ref = _rec_ts["last_ref"]
            # A backwards jump of >1s in DTS indicates the source file looped.
            # The offset must be "previous output end - new input start", NOT just
            # "previous output end": the old code assumed the file rewinds to DTS 0
            # on restart, but DVR/AVI files often loop back to a large timestamp
            # (e.g. wall-clock DTS).  Each loop then inflated the output timeline by
            # that start timestamp on top of the file duration, ballooning the offset
            # past 2^64 and overflowing the guint64 property setter below.
            if prev_ref is not None and ref_ts + Gst.SECOND < prev_ref:
                _rec_ts["offset"] = _rec_ts["last_out_ref"] + int(Gst.SECOND / 30) - ref_ts

            _rec_ts["last_ref"] = ref_ts
            out_ref = ref_ts + _rec_ts["offset"]
            _rec_ts["last_out_ref"] = max(_rec_ts["last_out_ref"], out_ref)

            # Rewrite PTS/DTS on the buffer.  Guard the uint64 range: the GObject
            # property setter raises OverflowError (killing the pipeline) for any value
            # outside 0.._MAX_TS, so if the input timestamps are already pathological
            # (near the max), leave the buffer untouched rather than crash.
            for attr in ("pts", "dts"):
                val = getattr(buf, attr)
                if val != Gst.CLOCK_TIME_NONE:
                    new_val = val + _rec_ts["offset"]
                    if 0 <= new_val < _MAX_TS:
                        setattr(buf, attr, new_val)
                    else:
                        logger.debug("[%s] ts %d out of uint64 range - leaving buffer untouched",
                                     source_id, new_val)
            if buf.pts != Gst.CLOCK_TIME_NONE:
                clock_sample["last"] = (buf.pts / 1e9, time.time())

            return Gst.PadProbeReturn.OK

        return Gst.PadProbeReturn.OK

    parse.get_static_pad("src").add_probe(
        Gst.PadProbeType.BUFFER | Gst.PadProbeType.EVENT_DOWNSTREAM, _monotonic_ts)

    sink = _make("splitmuxsink", f"rec_sink_{idx}")
    sink.set_property("max-size-time", int(gcfg.record_seconds * 1e9))
    sink.set_property("muxer-factory", "mpegtsmux")
    os.makedirs(f"{gcfg.record_tmp_dir}/{source_id}", exist_ok=True)
    sink.connect("format-location-full", _make_rotation_callback(gcfg, source_id, uploader, clock_sample))

    for el in (queue, enc, parse, sink):
        pipeline.add(el)
    queue.link(enc)
    enc.link(parse)
    parse.link(sink)
    # The second return is the branch's elements in source->sink order; the
    # _RecordBranchScheduler uses it to bring each element to PLAYING (the
    # encoder's transition blocks until its NVENC session is open).
    return queue, (queue, enc, parse, sink)



class _RecordBranchScheduler:
    """Create record branches one at a time after the pipeline is PLAYING.

    Builds and links each source's record branch (queue -> nvv4l2h264enc ->
    h264parse -> splitmuxsink) in a staggered, serialized fashion instead of at
    build time. See SidecarConfig.record_stagger_ms for the NVENC
    concurrent-open deadlock this avoids: with a static build, every encoder
    opens its NVENC session at once during the single set_state(PLAYING), and
    on some GPUs 6+ simultaneous opens wedge the hardware encoder (sessions
    open in BLOCKING mode but never encode) which freezes the whole pipeline.

    One branch is created per tick while the pipeline is PLAYING — a paused
    pipeline holds the remaining branches until it resumes — and each branch is
    brought to PLAYING synchronously, so the blocking NVENC session open
    completes before the next branch is started. There is no configured session
    cap: the GPU's concurrent-session ceiling is MEASURED at startup by a
    throwaway subprocess probe (perception/engine/nvenc_probe.py — a wedge during the
    probe cannot touch the real pipeline), and branch creation stops at the
    measured number. note_failure() is the backstop: a branch the driver
    rejects anyway (S_FMT failure) is torn down and creation halts. Streams
    past the ceiling keep analyzing/previewing but don't record.
    """

    def __init__(self, gcfg, pipeline, uploader, targets):
        self.gcfg = gcfg
        self.pipeline = pipeline
        self.uploader = uploader
        # [(source_id, mux pad index, tee element)] in creation order
        self._pending = list(targets)
        self._done = 0
        self._total = len(targets)
        self._stopped = False
        self._failed = False
        self._is_playing = lambda: True
        # [None|int]: None while the NVENC capacity probe runs, the measured
        # concurrent-session ceiling once known (0 = probe failed, recording
        # disabled this boot). The ceiling varies by GPU AND by boot (the same
        # RTX 5050 cleanly rejected the 13th session on one boot and accepted
        # 14 on another before wedging), so it is measured at runtime rather
        # than configured.
        self._capacity = [None]
        self._disabled_logged = False
        self._proc = None
        # idx -> (tee, requested src pad, branch elements) for every created
        # branch; lets note_failure() tear down exactly the branch that failed
        # (releasing its NVENC session) without disturbing its source chain.
        self._branches: dict = {}
        self._last_idx = None

    def start(self, is_playing):
        self._is_playing = is_playing
        if self._pending and not self._stopped:
            if self._capacity[0] is None:
                threading.Thread(target=self._probe_capacity, daemon=True,
                                 name="nvenc-capacity-probe").start()
            GLib.timeout_add(int(self.gcfg.record_stagger_ms), self._tick)

    def _probe_capacity(self):
        """Measure the GPU's max concurrent NVENC sessions on a THROWAWAY
        subprocess pipeline (perception/engine/nvenc_probe.py), so a driver wedge during
        probing cannot touch the real pipeline. Stores the result (0 on
        failure) in self._capacity. Retried once: a wedged probe is killed
        (its sessions released) and the driver may recover.

        The read is select()-based so the deadline still fires if the probe
        hangs mid-open (a wedged NVENC open blocks forever and prints
        nothing) — a plain blocking read would otherwise stall this thread
        past the deadline and recording would never start."""
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "nvenc_probe.py")
        best = None
        for attempt in range(2):
            best = None
            diag = ""
            proc = None
            try:
                proc = subprocess.Popen(
                    [sys.executable, script, str(self._total)],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                self._proc = proc
                deadline = time.monotonic() + 90.0
                while time.monotonic() < deadline:
                    rlist, _, _ = select.select([proc.stdout], [], [],
                                                deadline - time.monotonic())
                    if not rlist:
                        break  # probe hung (e.g. wedged NVENC open); killed below
                    line = proc.stdout.readline()
                    if not line:
                        break
                    line = line.strip()
                    if line.startswith("OK "):
                        best = int(line.split()[1])
                    elif line.startswith("LIMIT"):
                        break
            except Exception as exc:
                diag = str(exc)
            finally:
                if proc is not None:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    try:
                        diag = (diag + " " if diag else "") + \
                               (proc.stderr.read() or "")[:300]
                    except Exception:
                        pass
                    self._proc = None
            if best is not None:
                break
            logger.warning("NVENC capacity probe attempt %d failed (%s)%s",
                           attempt + 1, diag.strip() or "no result",
                           " — retrying once" if attempt == 0 else "")
        self._capacity[0] = best if best is not None else 0
        if best:
            logger.info("NVENC capacity: %d concurrent sessions on this GPU — "
                        "record branches capped at %d", best, best)

    def stop(self):
        self._stopped = True
        self._pending = []
        if self._proc is not None:
            try:
                self._proc.kill()
            except Exception:
                pass
            self._proc = None

    def note_failure(self, idx=None, reason=""):
        """Called when a record-branch encoder posts a bus error (e.g. the NVENC
        driver rejects another session). Tears down the failed branch (unlinks
        its tee pad, releases the NVENC session) and halts further creation so
        the rest of the pipeline keeps streaming."""
        if self._failed:
            return
        target = idx if idx is not None else self._last_idx
        self._teardown_branch(target)
        self._halt(reason or "encoder bus error")

    def _halt(self, reason):
        if self._failed:
            return
        self._failed = True
        self._pending = []
        logger.warning("record branch creation halted (%s) — %d/%d online; "
                       "remaining streams keep analyzing/previewing but will "
                       "not record", reason, self._done, self._total)

    def _teardown_branch(self, idx):
        """Unlink the branch from its tee and drop the branch elements to NULL,
        releasing its NVENC session. Idempotent; missing branch = no-op."""
        branch = self._branches.get(idx) if idx is not None else None
        if branch is None:
            return
        tee, pad, els = branch
        try:
            peer = pad.get_peer()
            if peer is not None:
                pad.unlink(peer)
            tee.release_request_pad(pad)
        except Exception as exc:
            logger.debug("record branch pad cleanup failed (non-fatal): %s", exc)
        for el in els:
            try:
                el.set_state(Gst.State.NULL)
                self.pipeline.remove(el)
            except Exception:
                pass
        self._branches.pop(idx, None)
        logger.warning("record branch %d torn down (tee pad unlinked, "
                       "NVENC session released)", idx)

    def _wait_tick(self):
        """Fast (1s) cadence while we cannot create branches yet — the capacity
        probe is still measuring, or the pipeline is paused. Hands back to the
        normal record_stagger_ms cadence the moment creation can proceed, so
        the stagger interval is never shortened by the wait."""
        if self._stopped or self._failed or not self._pending:
            return False
        if self._capacity[0] is not None and self._is_playing():
            GLib.timeout_add(int(self.gcfg.record_stagger_ms), self._tick)
            return False
        return True  # keep waiting at 1s

    def _tick(self):
        if self._stopped or self._failed or not self._pending:
            return False
        if self._capacity[0] is None or not self._is_playing():
            # Not ready to create a branch yet; wait on the 1s cadence (this
            # also avoids a 0-millisecond stagger busy-spinning the main loop).
            GLib.timeout_add(1000, self._wait_tick)
            return False
        cap = self._capacity[0]
        if cap <= 0:
            if not self._disabled_logged:
                self._disabled_logged = True
                logger.warning("recording disabled this boot — NVENC capacity "
                               "probe failed; record branches were not created "
                               "(analysis/preview unaffected)")
            self._pending = []
            return False
        if self._done >= cap:
            skipped = [sid for sid, _i, _t in self._pending]
            logger.warning("NVENC capacity %d reached (%d record branches) — "
                           "no recording for: %s (analysis/preview unaffected)",
                           cap, self._done, ", ".join(skipped))
            self._pending = []
            return False
        source_id, idx, tee = self._pending.pop(0)
        try:
            self._create_one(source_id, idx, tee)
            self._done += 1
            logger.info("[%s] record branch online (%d/%d, capacity %d) — "
                        "NVENC session opened", source_id, self._done,
                        self._total, cap)
            if not self._pending:
                logger.info("all %d record branches online (staggered startup "
                            "complete)", self._total)
        except Exception as exc:
            # _create_one already tore down the partially-built branch; just
            # halt further creation.
            self._halt(f"{source_id} branch startup failed: {exc}")
        return bool(self._pending)

    def _create_one(self, source_id, idx, tee):
        rec_branch, rec_els = _build_record_branch(
            self.pipeline, self.gcfg, source_id, idx, self.uploader)
        # Optional record-branch rate cap (videorate + capsfilter, mirror of the
        # static path in build_pipeline; see SidecarConfig.record_fps).
        chain = list(rec_els)
        if self.gcfg.record_fps > 0:
            rvrate = _make("videorate", f"rec_vrate_{idx}")
            rrcaps = _make("capsfilter", f"rec_rate_caps_{idx}")
            rrcaps.set_property("caps", Gst.Caps.from_string(
                f"video/x-raw(memory:NVMM), framerate={int(self.gcfg.record_fps)}/1"))
            self.pipeline.add(rvrate)
            self.pipeline.add(rrcaps)
            rvrate.link(rrcaps)
            chain = [rvrate, rrcaps] + chain
        rec_src = None
        try:
            # Bring the new branch to PLAYING NOW, one element at a time. The
            # encoder's PLAYING transition blocks until its NVENC session is
            # open, and this runs on the main loop — the serialization point.
            for el in chain:
                ret = el.sync_state_with_parent()
                if ret == Gst.StateChangeReturn.FAILURE:
                    raise RuntimeError(f"{el.get_name()} failed to reach PLAYING")
            rec_src = tee.request_pad_simple("src_%u")
            if rec_src is None:
                raise RuntimeError("tee refused a record src pad")
            if self.gcfg.record_fps > 0:
                rec_src.link(rvrate.get_static_pad("sink"))
                rrcaps.get_static_pad("src").link(rec_branch.get_static_pad("sink"))
            else:
                rec_src.link(rec_branch.get_static_pad("sink"))
        except Exception:
            # Tear down the partially-built branch so a failed encoder does not
            # hold an NVENC session or leave orphaned elements in the pipeline.
            for el in chain:
                try:
                    el.set_state(Gst.State.NULL)
                    self.pipeline.remove(el)
                except Exception:
                    pass
            raise
        self._branches[idx] = (tee, rec_src, chain)
        self._last_idx = idx
