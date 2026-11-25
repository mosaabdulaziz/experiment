"""Smart Video Record: per-source clip recording and alert-triggered clips."""

import os
import threading
import time
import redis
from .svr_trigger import SvrTrigger, svr_window_for_age
from .svr_session import SvrFileTracker

from .gst_util import logger




def _configure_svr_source(gcfg, srcbin, source_id):
    """Turn one nvurisrcbin into a Smart Video Record source.

    Every property here is flagged "changeable only in NULL or READY state" by
    gst-inspect, so this must run at construction; unlike zones, none of it
    hot-reloads.
    """
    os.makedirs(gcfg.svr_dir, exist_ok=True)
    # GstNvSourceType: (0) auto, (1) uri, (2) rtsp. The nicks are bare -- "rtsp",
    # NOT "source-type-rtsp" as the property's own description text implies.
    srcbin.set_property("type", 2)
    srcbin.set_property("smart-record", 2)          # cloud + local event triggers
    # DeepStream defaults smart-rec-cache to 0 -- no pre-roll at all -- which would
    # silently produce clips that start at the trigger instant. Always set.
    srcbin.set_property("smart-rec-cache", gcfg.svr_cache_seconds)
    # MKV, not MP4, and deliberately not configurable. qtmux refuses ANY caps
    # change once a track is configured, and an RTSP source whose SDP parameter
    # sets disagree with the ones it sends in-band changes caps ~0.65s after
    # connecting (go2rtc does this on H.265; real cameras do it at a day/night
    # switch, hours in). The refusal returns NOT_NEGOTIATED into nvurisrcbin's
    # INTERNAL pre-decode tee, which has no queue to absorb it, so it stops
    # rtspsrc and the camera is lost for the life of the process -- verified
    # with perception/pipeline/svr_probe.py, which also verified matroskamux accepts the
    # change and produces a playable file. A knob here would only be a way to
    # reintroduce that; use the probe's SVR_CONTAINER to experiment instead.
    # The app rewraps to MP4 on arrival (app/clip_builder.remux_to_mp4).
    srcbin.set_property("smart-rec-container", 1)   # MKV
    srcbin.set_property("smart-rec-mode", 1)        # video only
    srcbin.set_property("smart-rec-dir-path", gcfg.svr_dir)
    # A unique prefix per source is required for unique filenames, and
    # _newest_svr_file matches on it.
    srcbin.set_property("smart-rec-file-prefix", f"{source_id}_")
    srcbin.set_property("smart-rec-default-duration", gcfg.svr_duration_seconds)
    srcbin.connect("sr-done", _make_sr_done_handler(gcfg, source_id))
    gcfg.svr_triggers[source_id] = SvrTrigger(gcfg.svr_coalesce_seconds,
                                              gcfg.svr_rearm_cooldown_seconds)
    gcfg.svr_file_trackers[source_id] = SvrFileTracker()
    # Read the properties BACK rather than trusting the sets. A property that
    # exists, accepts a value and then ignores it is indistinguishable from
    # success unless it is logged -- the same trap `insert-sps-pps` documents in
    # _build_record_branch, and the reason the first SVR attempt cost a round trip.
    readback = []
    for prop in ("type", "smart-record", "smart-rec-cache", "smart-rec-container",
                 "smart-rec-mode", "smart-rec-default-duration", "smart-rec-dir-path",
                 "smart-rec-file-prefix"):
        try:
            readback.append(f"{prop}={srcbin.get_property(prop)}")
        except Exception as exc:
            readback.append(f"{prop}=<unreadable: {exc}>")
    logger.info("[%s] recorder=svr %s", source_id, " ".join(readback))



def _svr_names(gcfg, source_id):
    """Every SVR filename for one source currently on disk (prefix-matched)."""
    prefix = f"{source_id}_"
    try:
        return [n for n in os.listdir(gcfg.svr_dir) if n.startswith(prefix)]
    except OSError:
        return []



def _newest_svr_file(gcfg, source_id):
    """The most recently modified smart-record output for one source.

    DIAGNOSED FALLBACK ONLY -- do not use this as the primary way to resolve an
    sr-done. It used to be: the smart-rec-status guard allows only one session per
    source at a time, so (the reasoning went) an sr-done on a given nvurisrcbin can
    only refer to the session we started on it, making an mtime scan unambiguous.
    That guard IS unambiguous for the decision to *emit* start-sr (_SVR_EMIT_LOCK)
    -- but NOT for the on-disk write race: an alert incident that stays open
    re-arms recording the instant the previous session's smart-rec-status flips
    false, so a brand-new session's file can start existing at essentially the
    same instant the just-finished session's file gets its last write. An mtime
    comparison can then match the completion event to the WRONG (just-started,
    ~1s-of-content) file -- traced as the cause of alert clips that are
    anomalously ~1 second long. SvrFileTracker (svr_session.py) now does this
    correlation properly, at session-start time rather than guessed at
    completion time; _on_sr_done falls back to this function only when a
    session's tracker entry is unexpectedly missing, and logs loudly when it
    does. The sessionId out-param is still not usable to correlate instead --
    this pyds build exposes no Smart Record bindings at all (dir(pyds) has no
    NvDsSRRecordingInfo), so the signal's arguments cannot be read from Python.
    """
    names = _svr_names(gcfg, source_id)
    if not names:
        return None
    return max(names, key=lambda n: os.path.getmtime(os.path.join(gcfg.svr_dir, n)))



def _make_sr_done_handler(gcfg, source_id):
    """Announce a finished smart-record file so the app can attach it to its alert.

    The app reads data/ through its own mount, so the published path is in the APP's
    namespace (svr_dir_app), not the sidecar's (svr_dir). The signal's arguments are
    deliberately ignored -- see _newest_svr_file's docstring."""
    def _on_sr_done(_bin, *_args):
        trigger = gcfg.svr_triggers.get(source_id)
        if trigger is not None:
            trigger.closed(time.monotonic())     # the gap before the next session counts from now (SvrTrigger.closed)
        tracker = gcfg.svr_file_trackers.get(source_id)
        name = tracker.session_done() if tracker is not None else None
        if name is None:
            logger.warning("[%s] sr-done with no tracked session -- falling back to a "
                           "directory scan; investigate a tracker miss", source_id)
            name = _newest_svr_file(gcfg, source_id)
        if name is None:
            logger.warning("[%s] sr-done fired but no smart-record file found under %s",
                           source_id, gcfg.svr_dir)
            return
        app_path = f"{gcfg.svr_dir_app}/{name}"
        try:
            gcfg.r.xadd(f"iris:svr_done:{gcfg.key_id(source_id)}",
                        {"path": app_path, "recorded_at": f"{time.time():.3f}"},
                        maxlen=64, approximate=True)
            logger.info("[%s] smart-record complete -> %s", source_id, app_path)
        except redis.RedisError as exc:
            logger.error("[%s] redis xadd (svr_done) failed: %s", source_id, exc)
    return _on_sr_done



def _emit_start_sr(srcbin, source_id, pad, duration) -> bool:
    """Emit `start-sr` and VERIFY that a recording session actually opened.

    The first argument is an OUT parameter (`NvDsSRSessionId *`) and getting it
    across the Python boundary is not obvious -- two independent failures hide
    behind it, neither visible from the call site:

      * `None` marshals to NULL and DeepStream refuses it:
        `NvDsSRStart: assertion 'sessionId != NULL' failed`
      * a raw address is rejected before it ever reaches C:
        `TypeError: could not convert type int to gpointer`

    A PyCapsule wrapping a ctypes buffer satisfies both; verified on DS 9.0 with
    perception/pipeline/svr_probe.py, which is the tool to reach for if a future DeepStream or
    PyGObject changes this. pyds on this build exposes no Smart Record bindings
    (dir(pyds) has none), so there is no sanctioned allocator to use instead.

    `start-sr` is an action signal returning void, so emit() cannot report
    failure: a silent no-op is indistinguishable from success. The readable
    `smart-rec-status` property is the only evidence, and checking it is what
    turned "start-sr logged, nothing recorded" from a mystery into a fact -- so
    the check stays in production, it is not diagnostic scaffolding.
    """
    import ctypes

    # `session` must outlive the emit: DeepStream writes the session id through
    # the pointer during the (synchronous) signal call.
    session = ctypes.c_uint32(0)
    try:
        PyCapsule_New = ctypes.pythonapi.PyCapsule_New
        PyCapsule_New.restype = ctypes.py_object
        PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
        session_arg = PyCapsule_New(ctypes.addressof(session), None, None)
    except Exception as exc:
        logger.error("[%s] cannot build the sessionId capsule (%s) -- smart-record "
                     "cannot be triggered from Python on this build", source_id, exc)
        return False

    try:
        srcbin.emit("start-sr", session_arg, pad, duration, None)
    except Exception as exc:
        logger.error("[%s] start-sr emit failed: %s", source_id, exc)
        return False

    try:
        if bool(srcbin.get_property("smart-rec-status")):
            logger.info("[%s] smart-record start-sr (-%ds, %ds) — session open",
                        source_id, pad, duration)
            return True
    except Exception as exc:
        logger.warning("[%s] smart-rec-status unreadable after start-sr: %s",
                       source_id, exc)
        return False

    logger.error("[%s] start-sr emitted but smart-rec-status is still false -- no "
                 "recording session opened, so no clip will arrive for this alert. "
                 "Reproduce in isolation with perception/pipeline/svr_probe.py.", source_id)
    return False



_SVR_EMIT_LOCK = threading.Lock()



def _maybe_start_svr(ctx, st, now, age=0.0):
    """Start a smart-record clip for a just-fired alert, if this source uses SVR.

    `start-sr` returns void, so a start issued while a session is already running
    cannot be detected here -- the readable `smart-rec-status` property is the only
    way to know, and NVIDIA's own reference gates on the equivalent flag
    (deepstream_source_bin.c checks ctx->recordOn before calling NvDsSRStart).
    SvrTrigger turns that flag into start/coalesce/skip.

    `age` is how many seconds old the moment being captured is. Module alerts come from
    the buffer probe on the triggering frame itself and leave it at 0; a proactive VLM
    alert arrives through _poll_svr_requests only after the model answered, so it
    passes its real age and svr_window_for_age reaches further back to cover it.

    Called from two threads -- the streaming thread (zone alerts) and the GLib main
    loop (app requests) -- so the read-decide-emit sequence is serialised: SvrTrigger
    holds mutable session state, and interleaving it with a concurrent emit would let
    both threads read smart-rec-status false and start twice.

    A no-op for encode sources: svr_triggers only has entries for SVR ones.
    """
    trigger = ctx.g.svr_triggers.get(st.id)
    if trigger is None:
        return
    srcbin = getattr(st, "srcbin", None)
    if srcbin is None:
        return
    pad = max(ctx.g.svr_duration_seconds // 2, 1)
    window = svr_window_for_age(age, pad, ctx.g.svr_duration_seconds,
                                ctx.g.svr_cache_seconds)
    if window is None:
        # Checked before observe() so a request too old to record does not register a
        # phantom session start on the trigger.
        logger.info("[%s] no smart-record clip: the moment is %.1fs old but the cache "
                    "holds only %ss", st.id, age, ctx.g.svr_cache_seconds)
        return
    with _SVR_EMIT_LOCK:
        try:
            active = bool(srcbin.get_property("smart-rec-status"))
        except Exception as exc:
            logger.warning("[%s] smart-rec-status unreadable (%s); skipping the SVR start "
                           "rather than risking one during a live session", st.id, exc)
            return
        decision = trigger.observe(now, active)
        started, before_names = False, None
        if decision == "start":
            start_time, duration = window
            # Snapshot before the emit, while still holding the lock -- cheap (one
            # listdir), and it must be taken before ANY session for this source
            # can start, including one a concurrent thread might race to open the
            # instant this one finishes. See _track_new_svr_session.
            before_names = _svr_names(ctx.g, st.id)
            started = _emit_start_sr(srcbin, st.id, int(round(start_time)), int(round(duration)))
        elif decision == "coalesce":
            logger.info("[%s] alert coalesced onto the running smart-record session", st.id)
        elif decision == "cooldown":
            logger.info("[%s] no smart-record clip: within svr_rearm_cooldown_seconds "
                        "of the previous session's close", st.id)
        else:
            logger.info("[%s] no smart-record clip: a session is already running and past "
                        "its coalesce window", st.id)
    if started:
        # Outside the lock: _SVR_EMIT_LOCK is global across every source, and a
        # slow disk must not block other cameras' concurrent SVR starts while
        # this one waits for its file to appear.
        _track_new_svr_session(ctx.g, st.id, before_names)



_SVR_NAME_POLL_INTERVAL_SECONDS = 0.1

_SVR_NAME_POLL_TIMEOUT_SECONDS = 2.0



def _track_new_svr_session(gcfg, source_id, before_names):
    """Record the file a just-opened SVR session got, for SvrFileTracker to hand
    back on its sr-done -- see _newest_svr_file's docstring for why this replaced
    a completion-time mtime guess. `before_names` is the directory listing taken
    immediately before start-sr was emitted; poll (bounded, well under
    svr_duration_seconds) until the new file appears, since DeepStream may not
    have created it on disk the instant start-sr returns."""
    tracker = gcfg.svr_file_trackers.get(source_id)
    if tracker is None:
        return
    deadline = time.monotonic() + _SVR_NAME_POLL_TIMEOUT_SECONDS
    while True:
        after_names = _svr_names(gcfg, source_id)
        new_names = tracker.session_started(before_names, after_names)
        if new_names:
            logger.info("[%s] tracking new smart-record session -> %s", source_id, new_names)
            return
        if time.monotonic() >= deadline:
            logger.error("[%s] no new smart-record file appeared within %.1fs of start-sr -- "
                         "this session's sr-done will fall back to a directory scan. "
                         "Reproduce in isolation with perception/pipeline/svr_probe.py.",
                         source_id, _SVR_NAME_POLL_TIMEOUT_SECONDS)
            return
        time.sleep(_SVR_NAME_POLL_INTERVAL_SECONDS)
