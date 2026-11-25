"""VLM-based cross-camera vehicle re-id arbiter.

For the ambiguous re-id zone (gate <= cosine similarity < match threshold,
see perception/modules/mtmc/core/_associate.py), the ResNet50 embedding alone can't decide
confidently and the existing code just gives up ("don't gamble on a merge"),
creating a duplicate global track. This module asks a vision-language model
to look at both vehicles' crops directly and adjudicate same/different/
uncertain -- the exact prompt, JSON schema, shrink-retry ladder and stacked-
image fallback below are copied byte-for-byte from mtmc/compare_vehicles.py,
which was benchmarked (see mtmc/vehicle_comparisons.json) against qwen3-vl:4b
served over Ollama.

Runs entirely on background worker threads -- a VLM call can take several
seconds (shrink-retry ladder + 2-image flakiness fallback), far too slow for
the per-frame pipeline thread. GlobalVehicleAssociation submits a job and
polls for the result across subsequent frames; the tracklet simply doesn't
get a global_id (see get_or_create_global_id) until the verdict resolves or
max_wait_s elapses, at which point it falls back to creating a new track --
same "don't gamble on a merge" philosophy as the plain-embedding path, just
given a chance to actually resolve the ambiguity first.
"""
from __future__ import annotations

import base64
import collections
import json
import logging
import os
import queue
import re
import threading
import time
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("iris.vlm_arbiter")

# Base prompt + schema validated in mtmc/compare_vehicles.py's benchmark run
# (see mtmc/vehicle_comparisons.json). Two additions on top of that baseline
# (2026-09-28, after live deployment showed some pairs taking 60-300+s): the
# CONCISE block below and the maxLength/maxItems schema caps. Root cause
# (traced on the actual Ollama host): under temperature=0.0 greedy decoding,
# the model sometimes emits a correct verdict in the first few tokens, then
# falls into a VERBATIM REPETITION LOOP restating one sentence over and over
# until it hits num_ctx -- burning 60-300s and often truncating the JSON
# mid-string (unparseable). num_predict below is the hard stop for that; the
# schema caps + concise instruction make the model much less likely to start
# the loop in the first place (verified: mean 4.1s / max 4.3s over 10 pairs,
# vs. 67s+ before). See _parse_verdict's prefix-regex fallback for the case
# where a cap still truncates output.
PROMPT = (
    "You are a vehicle matching system for a multi-camera surveillance setup. "
    "Image 1 and Image 2 are crops of vehicles captured by (possibly different) cameras "
    "looking at the same scene from DIFFERENT VIEW ANGLES and DISTANCES. "
    "Determine whether they show the SAME physical vehicle.\n"
    "Compare visible traits: color(s), body type/silhouette, brand/logo, grille, lights, "
    "window/decal layout, cargo, trailer, roof rack, visible marks, damage, stickers.\n"
    "MULTI-CAMERA RULES (missing trait is NOT a contradiction):\n"
    "- A brand/logo/text visible in one crop but NOT in the other is simply out of frame "
    "  or too small at that angle — never treat 'brand not visible' as evidence of a "
    "  different vehicle. Only a clearly DIFFERENT brand in the same visible location counts.\n"
    "- A vehicle appearing smaller in one crop just means that camera was farther away — "
    "  never call it a 'smaller vehicle' because of zoom/distance.\n"
    "- A crop may show only PART of the vehicle (e.g. just the cab, or a corner). Compare "
    "  the parts that ARE visible; if the visible parts are compatible, they can still match.\n"
    "DECIDE:\n"
    "- 'same': visible attributes are compatible and no genuinely incompatible trait is seen.\n"
    "- 'different': a GENUINELY incompatible trait is visible (clearly different color, "
    "  different body type/silhouette, clearly different brand in the same location, "
    "  incompatible grille/lights/window layout).\n"
    "- 'uncertain': you cannot decide from these crops — too little detail, too occluded, "
    "  or only generic portions visible (e.g. both crops show only a plain white box wall); "
    "  another crop/view would be needed to decide. Prefer 'uncertain' over forcing a wrong "
    "  answer when the evidence is insufficient.\n"
    "CONCISE: keep 'explanation' under 25 words. At most 3 items per array field. "
    "State each point once -- never repeat a sentence or phrase."
)

COMPARE_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["same", "different", "uncertain"]},
        "confidence": {"type": "number", "description": "0.0 to 1.0"},
        "matched_features": {"type": "array", "items": {"type": "string", "maxLength": 40}, "maxItems": 3},
        "conflicting_features": {"type": "array", "items": {"type": "string", "maxLength": 40}, "maxItems": 3},
        "note_missing_visibility": {"type": "array", "items": {"type": "string", "maxLength": 40}, "maxItems": 3},
        "explanation": {"type": "string", "description": "brief verdict reasoning, under 25 words", "maxLength": 220},
    },
    "required": ["decision", "confidence", "matched_features", "conflicting_features", "explanation"],
}

# Hosted OpenAI-compatible models have no Ollama-style format=schema grammar, so
# without the key names they pick their own (observed: "verdict" instead of
# "decision"). Spell the keys out for that provider.
_JSON_KEYS_HINT = (" Reply with ONLY a JSON object with keys: decision (same|different|uncertain), "
                   "confidence, matched_features, conflicting_features, explanation.")

_WAITING = object()  # sentinel: job submitted, not yet resolved

# Direction task: which end of a vehicle faces the camera. One call per tracked vehicle at a gate camera
# that looks at vehicles from one side (e.g. D16 faces the cab of vehicles LEAVING).
DIRECTION_PROMPT = (
    "A surveillance camera watches the gate of a site. The attached image(s) should show ONE vehicle (a truck "
    "or pickup) captured at different moments, oldest first. Decide which end of the vehicle faces the camera.\n"
    "- FRONT: the camera sees the vehicle's front: windshield, headlights, grille, hood, front bumper; "
    "the cab is the part nearest the camera.\n"
    "- REAR: the camera sees the vehicle's back: tailgate, cargo bed or load, rear lights, rear bumper; "
    "on a pickup the bed is the part nearest the camera.\n"
    "- SIDE: it is seen from the side or too obliquely to tell.\n"
    "- UNCLEAR: there is a vehicle but you cannot tell.\n"
    "- NONE: there is no vehicle in the picture at all (empty ground, a person, an object, a shadow). Not for "
    "a vehicle seen from an odd angle or only in part: that is UNCLEAR.\n"
    "Judge the vehicle itself, not people or things around it. If the views differ, answer for the majority "
    "of the clear views. Reply with ONLY a JSON object with keys: orientation (front|rear|side|unclear|none), "
    "confidence (0 to 1), evidence (a list of up to 3 short strings naming what you saw)."
)


# Side-view variant (D41 sees vehicles from the side, not head-on): which way does the front point in the picture.
DIRECTION_PROMPT_SIDE = (
    "A surveillance camera watches the gate of a site from the side. The attached image(s) should show ONE "
    "vehicle (a truck or pickup) captured at different moments, oldest first. Decide which way the FRONT of the vehicle "
    "points in the picture (the front is the cab/windshield/grille end; the rear is the cargo bed/tailgate end).\n"
    "- LEFT: the front points to the left side of the picture (the vehicle drives right to left).\n"
    "- RIGHT: the front points to the right side of the picture (the vehicle drives left to right).\n"
    "- TOWARD: the camera sees the front head-on or nearly so.\n"
    "- AWAY: the camera sees the rear head-on or nearly so.\n"
    "- UNCLEAR: there is a vehicle but you cannot tell.\n"
    "- NONE: there is no vehicle in the picture at all (empty ground, a person, an object, a shadow). Not for "
    "a vehicle seen from an odd angle or only in part: that is UNCLEAR.\n"
    "Judge the vehicle itself, not people or things around it. If the views differ, answer for the majority "
    "of the clear views. Reply with ONLY a JSON object with keys: orientation (left|right|toward|away|unclear|none), "
    "confidence (0 to 1), evidence (a list of up to 3 short strings naming what you saw)."
)
SIDE_MODE = "side_right_to_left_enters"     # direction_cameras mode of a side-view camera


def _parse_direction(text: str, side: bool = False):
    """The direction model's JSON, as {"decision": front|rear|side|unclear|none, "confidence", "evidence"}; for the
    side-view question (side=True) the decision is left|right|toward|away|unclear|none. none = no vehicle in the picture."""
    words = "left|right|toward|away|unclear|none" if side else "front|rear|side|unclear|none"
    allowed = tuple(words.split("|"))
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:].lstrip()
    data = None
    for cand in (s, s[s.find("{"):s.rfind("}") + 1] if "{" in s else ""):
        try:
            data = json.loads(cand)
            break
        except (json.JSONDecodeError, ValueError):
            continue
    if not isinstance(data, dict):
        m = re.search(r'"?orientation"?\s*[:=]\s*"?(' + words + ')', s, re.I)
        if not m:
            return None
        conf = re.search(r'"?confidence"?\s*[:=]\s*([0-9.]+)', s)
        data = {"orientation": m.group(1), "confidence": float(conf.group(1)) if conf else None}
    o = str(data.get("orientation", data.get("decision", ""))).strip().lower()
    if o not in allowed:
        return None
    try:
        conf = None if data.get("confidence") is None else float(data["confidence"])
    except (TypeError, ValueError):
        conf = None
    return {"decision": o, "confidence": conf, "evidence": list(data.get("evidence") or [])[:3]}


def _shrink(crop_bgr: np.ndarray, max_side: float) -> np.ndarray:
    h, w = crop_bgr.shape[:2]
    if max(h, w) <= max_side:
        return crop_bgr
    scale = max_side / float(max(h, w))
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return cv2.resize(crop_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _to_b64(crop_bgr: np.ndarray, max_side: int) -> str:
    ok, buf = cv2.imencode(".jpg", _shrink(crop_bgr, max_side), [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise ValueError("JPEG encode failed")
    return base64.b64encode(buf.tobytes()).decode()


def _stacked_b64(crop_a: np.ndarray, crop_b: np.ndarray, max_side: int) -> str:
    """Combine two crops side-by-side into ONE image (workaround for the flaky 2-image path)."""
    a, b = _shrink(crop_a, max_side / 2), _shrink(crop_b, max_side / 2)
    h = max(a.shape[0], b.shape[0])
    canvas = np.zeros((h, a.shape[1] + b.shape[1], 3), dtype=np.uint8)
    ya, yb = (h - a.shape[0]) // 2, (h - b.shape[0]) // 2
    canvas[ya:ya + a.shape[0], :a.shape[1]] = a
    canvas[yb:yb + b.shape[0], a.shape[1]:a.shape[1] + b.shape[1]] = b
    ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise ValueError("JPEG encode failed")
    return base64.b64encode(buf.tobytes()).decode()


def _as_list(x) -> list:
    return list(x) if isinstance(x, (list, tuple)) else [x]


def _montage(views: list) -> np.ndarray:
    """Views of one vehicle side by side in one image (equal height)."""
    if len(views) == 1:
        return views[0]
    h = max(v.shape[0] for v in views)
    parts = [cv2.resize(v, (max(1, round(v.shape[1] * h / v.shape[0])), h), interpolation=cv2.INTER_AREA)
             for v in views]
    return np.hstack(parts)


def _multi_prompt(k: int, m: int, captions: list | None = None) -> str:
    """PROMPT reworded for k views of vehicle A (images 1..k) and m of vehicle B.

    ``captions`` (one per image, A's then B's) say which camera/moment each image is."""
    head = ("You are a vehicle matching system for a multi-camera surveillance setup. "
            f"Vehicle A is a NEW sighting we must identify, shown in images 1-{k}: ONE single vehicle "
            f"captured at different moments (different distances/angles). Vehicle B is a KNOWN vehicle "
            f"we have already been tracking, shown in images {k + 1}-{k + m}: also ONE single vehicle, "
            "its images taken on the cameras it passed through (different angles). They come from (possibly different) "
            "cameras looking at the same scene from DIFFERENT VIEW ANGLES and DISTANCES. "
            "Use ALL views of each vehicle together: a view that shows a distinguishing trait "
            "outweighs views where that trait is hidden. "
            "Determine whether A and B are the SAME physical vehicle.\n")
    if captions:
        head += "Image captions:\n" + "".join(f"- Image {i}: {c}\n" for i, c in enumerate(captions, 1))
    return head + PROMPT.split("Determine whether they show the SAME physical vehicle.\n", 1)[1]


def _is_garbage(text: str) -> bool:
    s = text.strip()
    return len(s) > 0 and set(s) <= {"?", "�", " "}


def _parse_verdict(text: str):
    """Parse the model's JSON output, tolerating markdown fences, stray prose or a bare verdict word.

    A model that names the key "verdict" instead of "decision" is accepted too.
    """
    verdict = _parse_verdict_raw(text)
    if isinstance(verdict, dict) and "decision" not in verdict and verdict.get("verdict") in (
            "same", "different", "uncertain"):
        verdict["decision"] = verdict["verdict"]
    return verdict


def _parse_verdict_raw(text: str):
    if not text:
        return None
    s = text.strip()
    bare = s.strip(" .\n\t\"'`")
    if bare.lower() in ("same", "different", "uncertain"):
        return {"decision": bare.lower(), "confidence": None, "explanation": "bare verdict (no JSON)"}
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:].lstrip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(s[start:end + 1])
        except json.JSONDecodeError:
            pass
    # Truncated output (e.g. num_predict cut it off, or a repetition loop ate
    # the budget before the closing brace): 'decision' is always the first
    # field the model emits, so a prefix regex can still recover a usable
    # verdict even when the JSON never closes.
    m = re.search(r'"decision"\s*:\s*"(same|different|uncertain)"', s)
    if m:
        conf_m = re.search(r'"confidence"\s*:\s*([0-9.]+)', s)
        confidence = float(conf_m.group(1)) if conf_m else None
        return {"decision": m.group(1), "confidence": confidence,
                "explanation": "recovered from truncated output"}
    return None


class VLMArbiter:
    """Background VLM-arbitration service for ambiguous-zone re-id candidates.

    ``submit()`` enqueues a comparison and returns a job_id immediately (never
    blocks the caller); ``take_result()`` returns the verdict once ready (None
    while still pending) and CONSUMES it -- call it at most once per job_id.
    """

    def __init__(self, enabled: bool = True, host: str = "http://127.0.0.1:11434",
                 model: str = "qwen3-vl:4b", num_ctx: int = 8192, temperature: float = 0.0,
                 keep_alive: float = -1, shrink_steps=(1024, 896, 768, 640),
                 use_format: bool = False, min_confidence: float = 0.5,
                 max_wait_s: float = 12.0, worker_threads: int = 2,
                 log_dir: str | None = None, num_predict: int = 400,
                 request_timeout_s: float = 30.0, provider: str = "ollama",
                 base_url: str = "", api_key_env: str = "QWEN_API_KEY"):
        self.enabled = bool(enabled)
        # "ollama": native Ollama client at ``host``. "openai": any
        # OpenAI-compatible hosted API (e.g. DashScope Qwen) at ``base_url``;
        # the key is read from the ``api_key_env`` environment variable and is
        # never stored in config.
        self._provider = provider
        self._openai_thinking_off = True
        self.min_confidence = float(min_confidence)
        self.max_wait_s = float(max_wait_s)
        self._model = model
        self._num_ctx = int(num_ctx)
        self._temperature = float(temperature)
        self._keep_alive = keep_alive
        self._shrink_steps = list(shrink_steps)
        self._use_format = bool(use_format)
        # Hard cap on generated tokens -- the real fix for the repetition-loop
        # pathology (see the PROMPT comment above): under greedy decoding the
        # model occasionally restates one sentence until it hits num_ctx,
        # burning 60-300s+ and truncating the JSON. This bounds the blast
        # radius to a few seconds regardless; _parse_verdict's prefix-regex
        # fallback recovers a verdict even if this cuts the output off.
        self._num_predict = int(num_predict)
        # num_predict bounds *tokens*, not wall-clock time: under real host
        # contention (colleagues' GPU jobs) decode speed craters and even a
        # 400-token cap can take minutes (observed: 318s, 814s against the
        # ~4-8s benchmark). With worker_threads=1 that single call blocks the
        # ENTIRE arbitration queue for its whole duration. This is a hard
        # client-side abort so one contended call can't stall every other
        # pending comparison -- the abandoned request may keep running
        # server-side, but this worker moves on and treats it as a failure
        # (caught by _compare's try/except -> falls through to the next
        # shrink step / stacked fallback / "uncertain").
        self._request_timeout_s = float(request_timeout_s)

        # Diagnostic capture: every resolved job (match, reject, or timeout)
        # gets its two crops + full verdict/context written under log_dir, so
        # "why didn't this merge?" can be answered by looking at the actual
        # images instead of re-running the pipeline. Off by default (None).
        self._log_dir: Path | None = Path(log_dir) if log_dir else None
        self._log_lock = threading.Lock()
        if self._log_dir is not None:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            # Each run starts a fresh log: the live review page (app /vlm) shows
            # one run at a time. Only our own artifacts are removed.
            # (SCALE_KEEP_RUN_FILES=1, set for a full-day run: the sidecar is restarted by the healer, global ids
            # continue from Redis, and the day's records must survive each restart)
            keep = os.environ.get("SCALE_KEEP_RUN_FILES") == "1"
            for f in [] if keep else list(self._log_dir.glob("*.jpg")) + [self._log_dir / "arbitration_log.jsonl",
                                                                           self._log_dir / "run.json"]:
                try:
                    f.unlink()
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    logger.warning("[vlm_arbiter] could not clear %s: %s", f, exc)

        self._client = None
        if self.enabled:
            try:
                if self._provider == "openai":
                    from openai import OpenAI
                    api_key = os.environ.get(api_key_env, "")
                    if not api_key or not base_url:
                        raise ValueError(f"openai provider needs base_url and ${api_key_env}")
                    self._client = OpenAI(api_key=api_key, base_url=base_url,
                                          timeout=self._request_timeout_s, max_retries=0)
                else:
                    from ollama import Client
                    self._client = Client(host=host, timeout=self._request_timeout_s)
            except Exception as exc:
                logger.warning("[vlm_arbiter] ollama client init failed (disabling): %s", exc)
                self.enabled = False

        self._queue: "queue.Queue" = queue.Queue()
        # health numbers for the monitoring gauges (see health())
        self._stats_lock = threading.Lock()
        self._n = {"submitted": 0, "answered": 0, "errors": 0, "same": 0, "different": 0,
                   "uncertain": 0, "direction": 0}
        self._in_flight = 0
        self._last_answer_ts = 0.0
        self._recent: "collections.deque" = collections.deque(maxlen=20)   # (compare_s, wait_s)
        self._results: dict[int, dict] = {}
        self._results_lock = threading.Lock()
        self._next_job_id = 0
        self._job_id_lock = threading.Lock()
        self._workers: list[threading.Thread] = []
        if self.enabled:
            for i in range(max(1, worker_threads)):
                t = threading.Thread(target=self._loop, name=f"vlm-arbiter-{i}", daemon=True)
                t.start()
                self._workers.append(t)
            self._write_run_info(model, base_url if self._provider == "openai" else host)
            logger.info("[vlm_arbiter] started (%d worker(s), host=%s, model=%s, log_dir=%s, "
                        "num_predict=%d, request_timeout_s=%.0f)",
                        len(self._workers), base_url if self._provider == "openai" else host,
                        model, self._log_dir,
                        self._num_predict, self._request_timeout_s)

    def _prompt_for(self, views_a: list, views_b: list, context: dict | None = None) -> str:
        """The exact text sent with the images for a k-vs-m comparison."""
        multi = len(views_a) > 1 or len(views_b) > 1
        ctx = context or {}
        captions = (ctx.get("captions_a") or []) + (ctx.get("captions_b") or [])
        if len(captions) != len(views_a) + len(views_b):
            captions = []
        prompt = _multi_prompt(len(views_a), len(views_b), captions) if multi else PROMPT
        suffix = ("" if multi else "\nImage 1 (attached first) is the NEW sighting A; Image 2 (attached second) is the KNOWN vehicle B.") + (
            _JSON_KEYS_HINT if self._provider == "openai" else
            "" if self._use_format else
            " Return ONLY valid JSON matching the required schema.")
        return prompt + suffix

    def _write_run_info(self, model: str, endpoint: str) -> None:
        if self._log_dir is None:
            return
        try:
            info = {
                "started_at": time.time(), "provider": self._provider, "model": model,
                "endpoint": endpoint, "temperature": self._temperature,
                "min_confidence": self.min_confidence, "max_tokens": self._num_predict,
                "prompt_single_image_each": self._prompt_for([0], [0]),
                "prompt_three_views_each": self._prompt_for([0] * 3, [0] * 3),
            }
            (self._log_dir / "run.json").write_text(json.dumps(info, indent=2))
        except Exception as exc:
            logger.warning("[vlm_arbiter] run.json write failed: %s", exc)

    def close(self) -> None:
        for _ in self._workers:
            self._queue.put(None)
        for t in self._workers:
            try:
                t.join(timeout=5.0)
            except Exception:
                pass
        self._workers = []

    def submit(self, crop_a, crop_b, context: dict | None = None) -> int:
        """Enqueue a comparison job; returns a job_id to poll with take_result().

        ``context`` is opaque caller metadata (cam_id, local_id, candidate
        global_id, mode, similarity, pts, ...) -- not used for the comparison
        itself, only echoed into the diagnostic log entry (see log_dir) so a
        saved verdict can be traced back to which tracklets it was about.
        """
        with self._job_id_lock:
            job_id = self._next_job_id
            self._next_job_id += 1
        # crop_a / crop_b: one image, or a list of views of that vehicle (time-ordered).
        views_a = [c.copy() for c in _as_list(crop_a)]
        views_b = [c.copy() for c in _as_list(crop_b)]
        with self._stats_lock:
            self._n["submitted"] += 1
        self._queue.put((job_id, views_a, views_b, context or {}, time.time()))
        # Opportunistic sweep: drop unconsumed results older than 5 minutes so a
        # caller bug (never calling take_result) can't leak memory forever.
        cutoff = time.time() - 300.0
        with self._results_lock:
            stale = [jid for jid, r in self._results.items() if r.get("_ts", 0.0) < cutoff]
            for jid in stale:
                del self._results[jid]
        return job_id

    def submit_direction(self, views, context: dict | None = None) -> int:
        """Enqueue a direction job (which end of the vehicle faces the camera); poll with take_result().
        The verdict's ``decision`` is front | rear | side | unclear."""
        return self.submit(views, [], {**(context or {}), "task": "direction"})

    def take_result(self, job_id: int) -> dict | None:
        """Return and consume the verdict for job_id, or None if not ready yet."""
        with self._results_lock:
            return self._results.pop(job_id, None)

    def _loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            job_id, views_a, views_b, context, submitted_ts = item
            t0 = time.time()
            with self._stats_lock:
                self._in_flight += 1
            is_direction = context.get("task") == "direction"
            try:
                verdict = (self._direction(views_a, context) if is_direction
                           else self._compare(views_a, views_b, context))
            except Exception as exc:
                logger.warning("[vlm_arbiter] %s failed: %s", "direction" if is_direction else "compare", exc)
                verdict = {"error": str(exc), "decision": "unclear" if is_direction else "uncertain"}
            elapsed = time.time() - t0
            verdict["_ts"] = time.time()
            self._note_done(verdict, elapsed, t0 - submitted_ts, is_direction)
            if self._log_dir is not None:
                self._persist(job_id, views_a, views_b, context, verdict,
                               submitted_ts, elapsed)
            with self._results_lock:
                self._results[job_id] = verdict

    def _note_done(self, verdict: dict, compare_s: float, wait_s: float, direction: bool = False) -> None:
        d = str(verdict.get("decision", "uncertain"))
        with self._stats_lock:
            self._in_flight = max(0, self._in_flight - 1)
            self._n["answered"] += 1
            if verdict.get("error"):
                self._n["errors"] += 1
            if direction:
                self._n["direction"] += 1
            else:
                self._n[d if d in ("same", "different") else "uncertain"] += 1
            self._last_answer_ts = time.time()
            self._recent.append((compare_s, wait_s))

    def health(self) -> dict[str, float]:
        """Numbers for the monitoring gauges: backlog, in-flight calls, totals by outcome and
        the compare / queue-wait times of the last 20 jobs."""
        with self._stats_lock:
            recent = list(self._recent)
            out = {f"{k}_total": float(v) for k, v in self._n.items()}
            out["in_flight"] = float(self._in_flight)
            last = self._last_answer_ts
        out["queue_depth"] = float(self._queue.qsize())
        out["last_answer_age_s"] = time.time() - last if last else -1.0
        out["compare_last_s"] = recent[-1][0] if recent else 0.0
        out["compare_avg_s"] = sum(r[0] for r in recent) / len(recent) if recent else 0.0
        out["compare_max_s"] = max((r[0] for r in recent), default=0.0)
        out["wait_max_s"] = max((r[1] for r in recent), default=0.0)
        return out

    def _persist(self, job_id: int, views_a: list, views_b: list,
                 context: dict, verdict: dict, submitted_ts: float, elapsed: float) -> None:
        """Write this resolved job's images + prompt + full verdict/context to log_dir."""
        try:
            ts_tag = time.strftime("%Y%m%d_%H%M%S", time.localtime(submitted_ts))
            stem = f"{ts_tag}_job{job_id:06d}"
            names = {"a": [], "b": []}
            for side, views in (("a", views_a), ("b", views_b)):
                for i, v in enumerate(views):
                    name = f"{stem}_{side}{i}.jpg"
                    cv2.imwrite(str(self._log_dir / name), v)
                    names[side].append(name)
            record = {
                "job_id": job_id,
                "submitted_at": submitted_ts,
                "resolved_at": time.time(),
                "compare_elapsed_s": round(elapsed, 2),
                "images_a": names["a"],
                "images_b": names["b"],
                "prompt": ((DIRECTION_PROMPT_SIDE if context.get("mode") == SIDE_MODE else DIRECTION_PROMPT)
                           if context.get("task") == "direction"
                           else self._prompt_for(views_a, views_b, context)),
                "context": context,
                "verdict": {k: v for k, v in verdict.items() if k != "_ts"},
            }
            line = json.dumps(record, default=str) + "\n"
            with self._log_lock:
                with open(self._log_dir / "arbitration_log.jsonl", "a") as f:
                    f.write(line)
        except Exception as exc:
            logger.warning("[vlm_arbiter] diagnostic log write failed: %s", exc)

    def _unload(self) -> None:
        if self._provider == "openai":
            return  # hosted API: nothing to unload
        try:
            self._client.stop(self._model)
        except Exception:
            pass

    def _call(self, prompt: str, images: list[str]) -> str | None:
        if self._provider == "openai":
            content = [{"type": "text", "text": prompt}] + [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}}
                for b in images
            ]
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": content}],
                temperature=self._temperature,
                max_tokens=self._num_predict,
                extra_body={"enable_thinking": False},
            )
            return resp.choices[0].message.content
        response = self._client.chat(
            model=self._model,
            messages=[{"role": "user", "content": prompt, "images": images}],
            options={"temperature": self._temperature, "num_ctx": self._num_ctx,
                     "num_predict": self._num_predict},
            format=COMPARE_SCHEMA if self._use_format else None,
            keep_alive=self._keep_alive,
        )
        return response.message.content

    def _direction(self, views: list, context: dict | None = None) -> dict:
        """Which end of the vehicle faces the camera, from its time-spread views."""
        side = self._shrink_steps[0] if self._shrink_steps else 768
        imgs = [_to_b64(v, side) for v in views]
        side_view = (context or {}).get("mode") == SIDE_MODE
        text = self._call(DIRECTION_PROMPT_SIDE if side_view else DIRECTION_PROMPT, imgs)
        verdict = _parse_direction(text, side=side_view)
        if verdict is None:
            return {"decision": "unclear", "confidence": None, "explanation": f"unparseable answer: {str(text)[:120]}"}
        verdict["explanation"] = "; ".join(verdict.pop("evidence", []))
        return verdict

    def _compare(self, views_a: list, views_b: list, context: dict | None = None) -> dict:
        prompt = self._prompt_for(views_a, views_b, context)
        last_err = None
        for side in self._shrink_steps:
            imgs = [_to_b64(v, side) for v in views_a + views_b]
            # 1 attempt per size, not 3: under temperature=0.0 greedy decoding
            # a same-size retry after garbage/parse-failure reproduces the
            # IDENTICAL output deterministically (no benefit), and after a
            # request_timeout_s abort it just multiplies a slow/contended
            # call by 3x for no better odds. A genuinely different image size
            # is a real retry; the same size again is not. Cuts the worst
            # case from 4 sizes x 3 attempts x 2 (2-image + stacked fallback)
            # = 24 possible request_timeout_s hits down to 4 x 2 = 8.
            for _ in range(1):
                try:
                    text = self._call(prompt, imgs)
                    if text is None or _is_garbage(text):
                        self._unload()
                        continue
                    verdict = _parse_verdict(text)
                    if verdict is not None:
                        return verdict
                    self._unload()
                except Exception as exc:
                    last_err = str(exc)
                    self._unload()
        # Flaky 2-image path failed at every shrink step; fall back to a single stacked image.
        return self._compare_stacked(_montage(views_a), _montage(views_b), last_err)

    def _compare_stacked(self, crop_a: np.ndarray, crop_b: np.ndarray, last_err) -> dict:
        suffix = ("\nThe two crops are shown side by side in ONE image: LEFT half is Image 1, "
                  "RIGHT half is Image 2."
                  + (_JSON_KEYS_HINT if self._provider == "openai" else
                     "" if self._use_format else
                     " Return ONLY valid JSON matching the required schema."))
        for side in self._shrink_steps:
            stacked = _stacked_b64(crop_a, crop_b, side)
            for _ in range(1):  # see _compare()'s comment on why 1, not 3
                try:
                    text = self._call(PROMPT + suffix, [stacked])
                    if text is None or _is_garbage(text):
                        self._unload()
                        continue
                    verdict = _parse_verdict(text)
                    if verdict is not None:
                        verdict.setdefault("fallback", "stacked-single-image")
                        return verdict
                    self._unload()
                except Exception as exc:
                    last_err = str(exc)
                    self._unload()
        return {"error": f"failed in stacked fallback too ({last_err})", "decision": "uncertain"}
