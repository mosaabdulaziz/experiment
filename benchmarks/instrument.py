#!/usr/bin/env python3
"""benchmarks/instrument.py — inject [bench] timing wrappers into sidecar code.

Idempotent source patcher. Wraps the three hot-path functions that this repo's
"Deepstream enhancements" changed, so a benchmark can measure the SAME
instrumentation on both the before (32aaf69) and after (HEAD) code versions:

  1. `probe`                     — GStreamer pad probe; measures how long the
                                   streaming thread spends per buffer (busy %).
  2. `_on_format_location_full`  — splitmuxsink rotation callback; measures how
                                   long the recording streaming thread blocks on
                                   each segment rotation.
  3. `FaceRecognizer._process_frame` — measures per-frame face recognition work
                                   on the recognition thread.

Usage:  python3 instrument.py <sidecar.py> [face_recognizer.py]

Each file is patched in place. Running twice is a no-op (guarded by a marker).
The wrapper code is deliberately tiny so it measures the same thing regardless
of which version of the underlying function is present.
"""

import sys
from pathlib import Path

MARKER = "iris-bench-wrapper"  # idempotence guard

# ── transformation 1: probe ──────────────────────────────────────────────────
# Before:
#   def probe(pad, info, ctx):
# After:
#   _iris_bm_probe = {...}          (module-level accumulator)
#   def probe(pad, info, ctx):      (wrapper — the name the pipeline attaches)
#       ...
#   def _probe_impl(pad, info, ctx):  (renamed original)
PROBE_ANCHOR = "def probe(pad, info, ctx):"
PROBE_HEADER = (
    "_iris_bm_probe = {\"t0\": time.monotonic(), \"frames\": 0, \"ms\": 0.0}  # " + MARKER + "\n"
)
PROBE_WRAPPER = (
    "def probe(pad, info, ctx):\n"
    "    _s = time.monotonic()\n"
    "    _r = _probe_impl(pad, info, ctx)\n"
    "    _st = _iris_bm_probe\n"
    "    _st[\"ms\"] += (time.monotonic() - _s) * 1000.0\n"
    "    _st[\"frames\"] += 1\n"
    "    _now = time.monotonic()\n"
    "    if _now - _st[\"t0\"] >= 5.0:\n"
    "        _dur = _now - _st[\"t0\"]\n"
    "        logger.info(\"[bench] probe busy=%.1f%% frames=%d avg=%.2fms\",\n"
    "                    _st[\"ms\"] / (_dur * 10.0), _st[\"frames\"],\n"
    "                    _st[\"ms\"] / max(1, _st[\"frames\"]))\n"
    "        _st[\"t0\"] = _now\n"
    "        _st[\"ms\"] = 0.0\n"
    "        _st[\"frames\"] = 0\n"
    "    return _r\n"
)
PROBE_RENAME = "def _probe_impl(pad, info, ctx):"

# ── transformation 2: splitmuxsink rotation callback ─────────────────────────
ROT_ANCHOR = "    def _on_format_location_full(_sink, _fragment_id, sample, _udata=None):"
ROT_WRAPPER = (
    "    def _on_format_location_full(_sink, _fragment_id, sample, _udata=None):  # " + MARKER + "\n"
    "        _s = time.monotonic()\n"
    "        _r = _on_format_location_full_impl(_sink, _fragment_id, sample, _udata)\n"
    "        logger.info(\"[bench] [%s] rotation_ms=%.3f\", source_id,\n"
    "                    (time.monotonic() - _s) * 1000.0)\n"
    "        return _r\n"
)
ROT_RENAME = "    def _on_format_location_full_impl(_sink, _fragment_id, sample, _udata=None):"

# ── transformation 3: FaceRecognizer._process_frame ──────────────────────────
REC_ANCHOR = "    def _process_frame(self, frame: np.ndarray, faces: List[Dict[str, Any]],"
REC_WRAPPER = (
    "    def _process_frame(self, frame: np.ndarray, faces: List[Dict[str, Any]],\n"
    "                       source_id: str, pts: float):  # " + MARKER + "\n"
    "        _s = time.monotonic()\n"
    "        _r = self._process_frame_impl(frame, faces, source_id, pts)\n"
    "        logger.info(\"[bench] [%s] recognizer_ms=%.3f faces=%d\", source_id,\n"
    "                    (time.monotonic() - _s) * 1000.0, len(faces))\n"
    "        return _r\n"
)
REC_RENAME = ("    def _process_frame_impl(self, frame: np.ndarray, faces: List[Dict[str, Any]],")


def _apply(text: str, anchor: str, prefix: str, wrapper: str, rename: str) -> str:
    if anchor not in text:
        raise SystemExit(f"anchor not found:\n{anchor}\n--- file unchanged ---")
    # Insert prefix (module-level) before the anchor, then the wrapper, then the
    # renamed original def (whose continuation lines + body follow untouched).
    # Idempotence is handled by `instrument()` (MARKER check) — the first
    # wrapper inserts the marker, so later transformations must still run.
    return text.replace(anchor, prefix + wrapper + rename, 1)


def instrument(path: str, anchor_spec) -> bool:
    p = Path(path)
    text = p.read_text()
    if MARKER in text:
        print(f"[instrument] {path}: already instrumented (skip)")
        return False
    for anchor, prefix, wrapper, rename in anchor_spec:
        text = _apply(text, anchor, prefix, wrapper, rename)
    p.write_text(text)
    print(f"[instrument] {path}: patched")
    return True


def main(argv) -> int:
    if len(argv) < 2:
        print("usage: instrument.py <sidecar.py> [face_recognizer.py]")
        return 2
    for path in argv[1:]:
        text = Path(path).read_text()
        spec = []
        if PROBE_ANCHOR in text:
            # probe + splitmuxsink rotation callback live in sidecar.py
            spec.append((PROBE_ANCHOR, PROBE_HEADER, PROBE_WRAPPER, PROBE_RENAME))
            spec.append((ROT_ANCHOR, "", ROT_WRAPPER, ROT_RENAME))
        if REC_ANCHOR in text:
            # _process_frame lives in face_recognizer.py
            spec.append((REC_ANCHOR, "", REC_WRAPPER, REC_RENAME))
        if not spec:
            print(f"[instrument] {path}: no known anchors, skipping")
            continue
        instrument(path, spec)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
