"""Pure helpers + driver for history-aware chat search. No redis/openai imports
here beyond what the driver needs as injected callables — keeps it unit-testable."""

import re

# Some models leak a short reasoning preamble before the required marker line
# (e.g. a bare "thought" / "Thought:" line, or "thought FOUND ..."). If that
# slips through, the first line isn't the marker, so the gate falls back to PLAIN
# and the literal FOUND/NOT_FOUND token reaches the user. We peel this preamble
# before resolving the marker so it is detected and never shown.
_NOISE_WORDS = r"(?:thoughts?|thinking|thought\s+process|reasoning|analysis)"
# A whole leading line that is ONLY a reasoning word (+ optional punctuation/markdown).
_NOISE_LINE_RE = re.compile(rf"^[\s>*_`\"'#\-]*{_NOISE_WORDS}[\s.:;,\-–—]*$", re.IGNORECASE)
# A leading reasoning word + separators on the marker line itself ("thought FOUND").
_INLINE_NOISE_RE = re.compile(rf"^[\s>*_`\"'#\-]*{_NOISE_WORDS}\b[\s:.\-]*", re.IGNORECASE)


def _drop_lead_noise_lines(text: str) -> str:
    """Strip leading blank lines and reasoning-only lines (each terminated by a
    newline) so the marker line surfaces. A trailing partial line (no newline) is
    left untouched, so this is safe to call on a still-streaming buffer."""
    while True:
        nl = text.find("\n")
        if nl == -1:
            break
        line = text[:nl]
        if line.strip() == "" or _NOISE_LINE_RE.match(line):
            text = text[nl + 1:]
            continue
        break
    return text


def _strip_inline_noise(head: str) -> str:
    """Remove a leading reasoning word + separators from a single line so an
    inline-prefixed marker ("thought FOUND ...") is still recognised."""
    return _INLINE_NOISE_RE.sub("", head, count=1)


def _fmt_hms(sec: float) -> str:
    """Stream position as `HH:MM:SS` (mirror of analyzer._fmt_clock; kept local so
    this module stays free of the openai-importing analyzer)."""
    total = max(int(round(float(sec))), 0)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def select_result_streams(answer: str, camera_ids) -> list:
    """Stream ids whose name appears as a whole word in the model's answer
    (case-insensitive). Used to scope a FOUND preview clip to the stream the
    model actually attributed the result to. Whole-word match so `cam0` does not
    match `cam01`. Returns ids in `camera_ids` order; empty if none are named."""
    if not answer or not camera_ids:
        return []
    out = []
    for cid in camera_ids:
        if not cid:
            continue
        pat = rf"(?<![0-9A-Za-z_]){re.escape(str(cid))}(?![0-9A-Za-z_])"
        if re.search(pat, answer, re.IGNORECASE):
            out.append(cid)
    return out


def snapshots_newest_offset(snapshots) -> float:
    """Newest (max) stream offset across all per-camera `(offset, jpeg)` history
    snapshots, or `0.0` when empty. Used to seed the sweep's *live* cursor so the
    first live round only probes frames that arrive AFTER the sweep starts — without
    it the cursor begins at 0 and the first live round dumps the WHOLE history buffer
    (stride-picked), giving sparse frames spanning the entire window with a bogus
    wide `@ HH:MM:SS to HH:MM:SS` header."""
    vals = [o for items in (snapshots or {}).values() for (o, _j) in (items or [])]
    return max(vals) if vals else 0.0


def offsets_bounds(offsets_by_cam) -> tuple | None:
    """(min, max) stream offset across all cameras shown in a chat/search call, or
    None when no offsets are available (nothing to validate against)."""
    if not offsets_by_cam:
        return None
    vals = [o for offs in offsets_by_cam.values() for o in (offs or [])]
    if not vals:
        return None
    return (min(vals), max(vals))


_TS_RE = re.compile(r"\b(\d{1,2}):([0-5]\d):([0-5]\d)\b")


def clamp_answer_timestamps(answer: str, bounds: tuple | None, pad: float = 3.0) -> str:
    """Rewrite any `HH:MM:SS` in `answer` that falls outside the footage window
    actually shown to the model. `bounds` is (min, max) seconds of the frames in
    that call. In-range times are kept; out-of-range ones (e.g. a wall-clock
    `12:45:36` when the clip only spans `00:00:00`–`00:05:00`) are replaced with the
    canonical shown range. `bounds` None/empty -> unchanged."""
    if not answer or not bounds:
        return answer
    lo, hi = bounds
    if hi < lo:
        lo, hi = hi, lo
    lo_b, hi_b = lo - pad, hi + pad
    canon = _fmt_hms(lo) if lo == hi else f"{_fmt_hms(lo)}–{_fmt_hms(hi)}"

    def _repl(m):
        secs = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
        return m.group(0) if lo_b <= secs <= hi_b else canon

    return _TS_RE.sub(_repl, answer)


# A self-reported confidence right after the FOUND token: a decimal in [0,1]
# (`0.92`, `.85`, `1`, `1.0`) or an integer percentage (`92%`). Deliberately does
# NOT match a bare integer >1 without `%` (so "FOUND 2 cars" is not misread as a
# confidence) — answer digits stay in the answer.
_CONF_RE = re.compile(r"^[\s:=(]*(?:(?P<pct>\d{1,3})\s*%|(?P<dec>[01](?:\.\d+)?|\.\d+))\)?")


def _parse_confidence(after_found: str) -> tuple[float | None, str]:
    """Pull a leading confidence value from the text right after `FOUND`.

    Returns `(confidence in [0,1] or None, remaining_answer_text)`. When no
    confidence is present the value is `None` and the remainder is the original
    text with a leading `: ` separator stripped (back-compat with bare `FOUND` /
    `FOUND: answer`)."""
    m = _CONF_RE.match(after_found)
    if not m:
        return None, after_found.lstrip(": ").strip()
    if m.group("pct") is not None:
        val = float(m.group("pct")) / 100.0
    else:
        val = float(m.group("dec"))
    val = max(0.0, min(val, 1.0))
    return val, after_found[m.end():].lstrip(": ").strip()


def parse_sweep_marker(text: str, min_confidence: float = 0.0) -> tuple[bool, str]:
    """(found, answer_without_marker) from a joined sweep reply.

    Leading reasoning-preamble noise is peeled first; then the first remaining
    line decides: FOUND[ <conf>][: ...] -> (True, remainder); NOT_FOUND -> (False, "");
    anything else -> (False, original text). A FOUND whose self-reported confidence
    is below `min_confidence` is downgraded to NOT_FOUND -> (False, ""); a FOUND with
    no parsable confidence is accepted (absence can't be thresholded).
    """
    stripped = _drop_lead_noise_lines(text.lstrip()).lstrip()
    if not stripped:
        return False, ""
    first, _, rest = stripped.partition("\n")
    head = _strip_inline_noise(first.strip()).strip()
    upper = head.upper()
    if upper.startswith("NOT_FOUND"):
        return False, ""
    if upper.startswith("FOUND"):
        conf, inline = _parse_confidence(head[len("FOUND"):])
        if conf is not None and conf < min_confidence:
            return False, ""
        return True, (inline or rest.strip())
    return False, text


class SweepMarkerGate:
    """Incremental gate over a streamed first-pass reply. Resolves the first line
    to a decision and controls what reaches the user:
      FOUND / CONVERSATION -> stream the answer (text after the marker)
      NOT_FOUND            -> suppress everything (caller runs the history sweep)
      PLAIN                -> no recognised marker; treat the whole reply as the answer
    """

    def __init__(self, min_confidence: float = 0.0):
        self._buf = ""
        self._min_conf = min_confidence
        self.decision: str | None = None

    def feed(self, delta: str) -> str:
        if self.decision in ("FOUND", "CONVERSATION", "PLAIN"):
            return delta
        if self.decision == "NOT_FOUND":
            return ""
        self._buf += delta
        # Peel any leading reasoning-preamble lines as they complete, so the
        # marker line surfaces instead of resolving the preamble as PLAIN.
        self._buf = _drop_lead_noise_lines(self._buf)
        if "\n" not in self._buf:
            return ""
        first, _, rest = self._buf.partition("\n")
        return self._resolve(first, rest, has_newline=True)

    def _resolve(self, first: str, rest: str, has_newline: bool = False) -> str:
        head = first.strip()
        core = _strip_inline_noise(head).strip()  # tolerate "thought FOUND ..."
        upper = core.upper()
        if upper.startswith("NOT_FOUND"):
            self.decision = "NOT_FOUND"
            return ""
        if upper.startswith("FOUND"):
            conf, inline = _parse_confidence(core[len("FOUND"):])
            # A FOUND the model itself is not confident enough about is treated as
            # NOT_FOUND (suppress + let the caller keep sweeping). Missing confidence
            # passes — absence can't be thresholded.
            if conf is not None and conf < self._min_conf:
                self.decision = "NOT_FOUND"
                return ""
            self.decision = "FOUND"
            return inline if inline else rest
        if upper.startswith("CONVERSATION"):
            self.decision = "CONVERSATION"
            inline = core[len("CONVERSATION"):].lstrip(": ").strip()
            return inline if inline else rest
        # No recognised marker -> treat the entire buffered reply as a plain answer.
        # Emit the original line (not the noise-stripped core) so real content is
        # never dropped from a genuine plain reply.
        self.decision = "PLAIN"
        if has_newline:
            return first + "\n" + rest
        return first + ("\n" + rest if rest else "")

    def flush(self) -> str:
        if self.decision is None:
            self._buf = _drop_lead_noise_lines(self._buf)
            if self._buf:
                first, _, rest = self._buf.partition("\n")
                has_newline = "\n" in self._buf
                out = self._resolve(first, rest, has_newline=has_newline)
                self._buf = ""
                return out
        return ""


def history_chunks(total: int, per_round: int, start: int | None = None) -> list[tuple[int, int]]:
    """Newest→oldest `(lo, hi)` index pairs over a snapshot of length `total`.

    Each chunk spans `hi - lo ≤ per_round` frames. `start` (default `total`) is the
    resume cursor (exclusive upper bound of the deepest already-searched region).
    """
    per_round = max(int(per_round), 1)
    hi = total if start is None else max(0, min(int(start), total))
    out: list[tuple[int, int]] = []
    while hi > 0:
        lo = max(0, hi - per_round)
        out.append((lo, hi))
        hi = lo
    return out


def secs_ago(hi: int, total: int, fps: float) -> float:
    """Seconds before 'now' that index `hi` represents: `(total - hi) / fps`."""
    fps = max(float(fps), 0.1)
    return (total - hi) / fps


import time

from .chat_select import _stride_pick


def _fmt_back(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"T-{m}:{s:02d}"


def _history_window(snapshots: dict, lo: int, hi: int, window_frames: int) -> tuple[dict, dict]:
    """Per-camera stride-pick of ``(offset, jpeg)`` in snapshot index range
    [lo, hi). Returns ``(frames, offsets)`` so the probe can show each frame's
    timestamp (stream position) to the model."""
    frames: dict = {}
    offsets: dict = {}
    for cid, items in snapshots.items():
        picked = _stride_pick(items[lo:hi], window_frames)
        if picked:
            frames[cid] = [j for _o, j in picked]
            offsets[cid] = [o for o, _j in picked]
    return frames, offsets


async def run_sweep(*, snapshots, live_window, probe, fps, round_seconds,
                    window_frames, max_calls, max_seconds, start_index=None,
                    start_since=0.0, monotonic=time.monotonic, log=None):
    import logging as _logging
    emit = log if log is not None else _logging.getLogger("iris.sweep").info

    total = max((len(v) for v in snapshots.values()), default=0)
    per_round = max(int(round_seconds * fps), 1)
    chunks = history_chunks(total, per_round, start=start_index)
    started = monotonic()
    calls = 0
    since = start_since
    ci = 0  # index into chunks
    history_index = chunks[0][1] if chunks else 0

    resume = _fmt_back(secs_ago(start_index, total, fps)) if start_index is not None else "none"
    emit(f"[chat] sweep start streams={len(snapshots)} hist_frames={total} "
         f"round={int(round_seconds)}s cap={max_calls}/{int(max_seconds)}s resume={resume}")

    def capped() -> bool:
        return calls >= max_calls or (monotonic() - started) >= max_seconds

    def done(reason: str) -> None:
        emit(f"[chat] sweep done reason={reason} calls={calls} "
             f"reached={_fmt_back(secs_ago(history_index, total, fps))} "
             f"elapsed={monotonic() - started:.1f}s")

    while True:
        # ── LIVE round ──
        frames, frame_offsets, since = live_window(since)
        if frames:
            new_k = sum(len(v) for v in frames.values())
            emit(f"[chat] live round new={new_k} since={since:.1f} -> probing")
            yield {"type": "status", "stage": "searching_history", "detail": "checking live"}
            found, answer = await probe(frames, frame_offsets)
            calls += 1
            emit(f"[chat] live round result -> {'FOUND' if found else 'NOT_FOUND'}")
            if found:
                done("found")
                yield {"type": "token", "delta": answer}
                yield {"type": "result", "found": True, "history_index": history_index,
                       "since": since, "exhausted": False}
                return
            if capped():
                break

        # ── HISTORY round ──
        if ci >= len(chunks):
            done("exhausted")
            yield {"type": "token",
                   "delta": "No match found in the recorded history."}
            yield {"type": "result", "found": False, "history_index": 0,
                   "since": since, "exhausted": True}
            return
        lo, hi = chunks[ci]
        ci += 1
        history_index = lo
        window, window_offsets = _history_window(snapshots, lo, hi, window_frames)
        detail = f"back to {_fmt_back(secs_ago(hi, total, fps))}"
        emit(f"[chat] hist chunk=[{lo},{hi}] back={_fmt_back(secs_ago(hi, total, fps))} "
             f"picked={sum(len(v) for v in window.values())} -> probing")
        yield {"type": "status", "stage": "searching_history", "detail": detail}
        if window:
            found, answer = await probe(window, window_offsets)
            calls += 1
            emit(f"[chat] hist chunk=[{lo},{hi}] result -> {'FOUND' if found else 'NOT_FOUND'}")
            if found:
                done("found")
                yield {"type": "token", "delta": answer}
                yield {"type": "result", "found": True, "history_index": lo,
                       "since": since, "exhausted": False}
                return
        if ci >= len(chunks):
            done("exhausted")
            yield {"type": "token", "delta": "No match found in the recorded history."}
            yield {"type": "result", "found": False, "history_index": 0,
                   "since": since, "exhausted": True}
            return
        if capped():
            break

    done("cap")
    back = _fmt_back(secs_ago(history_index, total, fps))
    yield {"type": "token", "delta": f"Searched back to {back} — nothing found yet. Ask me to keep going to search further."}
    yield {"type": "result", "found": False, "history_index": history_index,
           "since": since, "exhausted": False}


def segments_for_found(store, source_id: str, lo: float, hi: float, round_seconds: float) -> list:
    """Segments (via `store.list_range`) covering the FOUND window `[lo, hi]`,
    padded by one `round_seconds` on each side. The chat sweep's chunk boundaries
    are computed retroactively (relative to when the search ran); the segment
    recorder rotates on its own independent, continuous schedule. The two clocks
    share a period but are not phase-aligned — padding by one segment length
    guarantees the true event span is never cropped by that drift, and it treats
    a live-window FOUND and a history-chunk FOUND identically (no branching)."""
    pad = max(float(round_seconds), 0.0)
    return store.list_range(source_id, lo - pad, hi + pad)
