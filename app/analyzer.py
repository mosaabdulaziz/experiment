import base64
import json
import logging
import os
import re
import shutil
import time
from datetime import datetime, timezone

from openai import AsyncOpenAI

logger = logging.getLogger("iris.analyzer")

from .personas import DEFAULT_PERSONA, PERSONAS
from .event_tools import QUERY_EVENTS_TOOL, dispatch_tool_call
from .scene_memory import validate_delta
from .severity import SEVERITIES, derive_event_type, normalise_category, normalise_severity

# The "{{"/"}}" in the example JSON below are literal braces, escaped because this
# string is concatenated into _SYSTEM_TEMPLATE BEFORE .format() is called on the
# whole template. If a future phase ever calls .format() on ANALYZE_SCHEMA alone, or
# appends it AFTER formatting, these must be un-escaped to single braces.
#
# The "{description_*}"/"{reasoning_placement}"/"{memory_*}"/"{example_memory_*}"/
# "{return_note}" slots are filled by build_analyze_schema with str.replace, NOT
# str.format -- formatting here would consume the escaped braces above a step too
# early.
_ANALYZE_SCHEMA_TEMPLATE = """\
## RESPONSE RULES
- Return ONLY valid JSON — no prose, no markdown fences.
- Never include "ignore" entries — simply omit events that are not reportable.
- Be concise: one clear sentence in "message" is enough.

### Fields
- "severity"  (required): one of "critical", "warning", "info".
    "critical" — a person is in immediate danger, or an incident is actively unfolding —
                 specifically: fire, a collision, a person who has fallen and is not
                 getting up, or a person directly in the path of a moving vehicle whose
                 driver appears not to have seen them.
    "warning"  — a real risk, violation, or operational problem that needs attention,
                 with nobody in immediate danger. Specifically includes: a moving
                 vehicle (forklift, truck, loader) operating in an area where people
                 are walking or working, a person standing or stepping into the path
                 of a vehicle that is moving, and any vehicle–person interaction in
                 motion — these are AT LEAST "warning" even when no collision has
                 begun and even if the vehicle appears to be moving slowly.
    "info"     — worth logging; no action needed.
  When you are torn between two levels, choose the LOWER one — but never below
  "warning" for a moving vehicle near people (see the anchors above).
  If the monitoring instructions use ALERT/COMMENT wording, they take precedence over
  the anchors above: treat ALERT as at least "warning", and COMMENT as "info" even when
  an anchor above would read higher.
- "category"  (required): one of "safety", "security", "storage", "operations".
- "message"   (required): the one-sentence headline.
- "tags"      (required): a list of short lowercase keywords.
- "location"  (optional): a short phrase saying where IN THE FRAME this is —
  "by the bay door", "centre of the aisle". Use null when you are unsure or it
  does not apply. Do not guess.
- "recommended_action" (optional): a short action. Use null unless the right
  action is obvious — do not pad.
{description_field}
### How to reason about an event
Work through this order before you write the fields:
1. Observation — what is actually visible.
2. Context — what it means given the rest of the scene.
3. Risk — what could go wrong if nobody acts.
4. Action — what someone should do, if anything obvious.
{reasoning_placement}
above. Never speculate beyond what you can see.
{memory_delta}
## Example OUTPUT
{example_memory_open}[
      {{
        "severity": "critical",
        "category": "safety",
        "message": "A forklift is reversing toward a worker who is walking behind it.",
        "tags": ["forklift","near-miss","person"],
        "location": "centre of the loading aisle",
        "recommended_action": "Radio the forklift driver to stop."{example_description_1}
      }},
      {{
        "severity": "warning",
        "category": "safety",
        "message": "A forklift is driving through the aisle where two workers are walking.",
        "tags": ["forklift","pedestrians"],
        "location": "main aisle",
        "recommended_action": null{example_description_2}
      }},
      {{
        "severity": "info",
        "category": "operations",
        "message": "A pallet has been left in the walkway margin, outside the marked staging area.",
        "tags": ["pallets","staging"],
        "location": "by the bay door",
        "recommended_action": null{example_description_2}
      }}
    ]{example_memory_close}
{return_note}
"""

_DESCRIPTION_FIELD = """\
- "description" (optional): 1–2 sentences giving the context and why it matters.
  No speculation, no hedging, no filler, no restating the message.
"""
_PLACEMENT_WITH_DESCRIPTION = (
    'Then: put 1 in "message", 2 and 3 in "description", and 4 in "recommended_action"\n'
    '(null when no action is obvious), and let step 3 decide "severity" using the rubric'
)
# Steps 2 and 3 still have to be *reasoned*, or severity is guesswork -- they simply have
# nowhere to be written down, so the model is told to keep them internal rather than
# being left to invent a field for them.
_PLACEMENT_WITHOUT_DESCRIPTION = (
    'Then: put 1 in "message" and 4 in "recommended_action" (null when no action is\n'
    'obvious). Keep 2 and 3 to yourself -- do NOT add any field for them -- and let\n'
    'step 3 decide "severity" using the rubric'
)


# ── the interpreted-memory delta (monitoring.memory.scene_delta) ──────────────
#
# Off by default and byte-free when off: this section costs OUTPUT tokens on every
# single analysis call and changes the response from a bare array to an object, so a
# deployment on a small local model must be able to run without it.
#
# The wording is deliberately about keeping state CURRENT, never about deduplication
# (scope decision 2026-09-01) -- telling the model it is de-duplicating invites it to
# suppress genuinely new events, which is the opposite of what this is for.
_MEMORY_DELTA_SECTION = """
## SCENE MEMORY DELTA
Your reply is a JSON OBJECT with exactly two keys: "events" (the array described above)
and "memory" (a small delta describing the state of the scene right now). The memory is
state, not a second event log: it is how the system knows what is still going on, and it
is what lets the system measure how long each thing has been going on for.

### The "memory" object
- "entities"    (optional): [{{"id": <track id>, "note": "<what this object is doing>"}}]
  for objects listed under TRACKED OBJECTS that are worth remembering. Use the id
  exactly as it is shown there.
- "areas"       (optional): [{{"zone": "<zone name>", "condition": "<its current state>"}}]
  Use only zone names that appear elsewhere in this prompt.
- "situations"  (optional): [{{"key": "<short_stable_slug>", "description": "<one short
  phrase>", "state": "ongoing"}}] — something that persists across cycles.
- "scene_notes" (optional): one or two short sentences of standing context (weather,
  lighting, staffing) that a later cycle would want to know.

### Rules for the memory
- Re-list every situation that is STILL happening, on every cycle, under the SAME "key"
  you used for it before. That is how its duration accumulates. Anything you stop
  listing is forgotten after a few cycles.
- Use "state": "resolved" for a situation that is over; it is then dropped.
- When a condition is already shown to you in the SCENE MEMORY block above as ongoing,
  confirm it here instead of emitting an event for it.
- LEAVING "scene_notes" OUT means "no change" — whatever is already stored is kept.
  Send "" for it only when the stored notes should be cleared.
- Send "memory": {{}} when there is nothing to add or confirm.
"""

_EXAMPLE_MEMORY_OPEN = '{{\n  "events": '
_EXAMPLE_MEMORY_CLOSE = """,
  "memory": {{
    "entities": [{{"id": 17, "note": "forklift reversing in the loading aisle"}}],
    "areas": [{{"zone": "Loading Bay", "condition": "unloading in progress"}}],
    "situations": [
      {{"key": "bay2_unload", "description": "truck at Bay 2 being unloaded", "state": "ongoing"}}
    ],
    "scene_notes": "Overcast; wet floor by the bay door."
  }}
}}"""

_RETURN_EVENTS_ONLY = (
    "Return [] (empty array) when nothing relevant is happening or all events "
    "were already reported."
)
_RETURN_WITH_MEMORY = (
    'Return an empty "events" array when nothing relevant is happening or all events '
    'were already reported — the "memory" object is still expected, so the scene state '
    "stays current."
)


def build_analyze_schema(*, descriptions: bool = True, scene_delta: bool = False) -> str:
    """The response-rules block, with or without the optional "description" field
    and with or without the scene-memory delta.

    One source for every variant: a second hand-maintained copy of this text would
    drift from the first the next time the severity rubric is tuned. Both flags are
    off-by-omission in the sense that matters -- `scene_delta=False` reproduces the
    text byte-for-byte, which is what keeps ANALYZE_SCHEMA a stable constant.
    """
    return (
        _ANALYZE_SCHEMA_TEMPLATE
        .replace("{description_field}", _DESCRIPTION_FIELD if descriptions else "")
        .replace("{reasoning_placement}",
                 _PLACEMENT_WITH_DESCRIPTION if descriptions else _PLACEMENT_WITHOUT_DESCRIPTION)
        .replace("{example_description_1}",
                 ',\n        "description": "The forklift is backing up while a worker '
                 'crosses behind it, outside the driver\'s view."' if descriptions else "")
        .replace("{example_description_2}",
                 ',\n        "description": null' if descriptions else "")
        .replace("{memory_delta}", _MEMORY_DELTA_SECTION if scene_delta else "")
        .replace("{example_memory_open}", _EXAMPLE_MEMORY_OPEN if scene_delta else "")
        .replace("{example_memory_close}", _EXAMPLE_MEMORY_CLOSE if scene_delta else "")
        .replace("{return_note}", _RETURN_WITH_MEMORY if scene_delta else _RETURN_EVENTS_ONLY)
    )


ANALYZE_SCHEMA = build_analyze_schema()

# {tracked}{scene} deliberately occupy one line of their own between the
# instructions and the history: TrackObserver.render() and SceneMemory.render() both
# return a block wrapped in leading and trailing newlines, so a non-empty value
# spaces itself like every other section, while "" (memory off) collapses the slot
# line away and leaves the prompt byte-identical to the pre-memory one. They SHARE
# that line rather than taking one each -- a second slot line would add a newline the
# pre-memory prompt never had. Pinned in test_analyzer_prompt_schema.py.
_SYSTEM_TEMPLATE_HEAD = """\
{role}

## YOUR TASK
Analyze the provided security camera clip and produce a JSON array of observations — one entry
per distinct event you observe. A single clip may contain multiple noteworthy events; report each
separately. Return an empty array [] when nothing relevant is happening.
Use your native video understanding to reason about motion, temporal events, and changes
across the clip — not just what's visible in any single moment.

## MONITORING INSTRUCTIONS
{instructions}
{tracked}{scene}
## RECENT OBSERVATION HISTORY
{history}

## BEFORE YOU RESPOND — MANDATORY HISTORY CHECK
The list above is everything already reported for this camera — by you on earlier cycles,
and by the system's automatic zone alerts — grouped by severity. For each event you are
about to output, ask:
  "Is this the same real-world condition as any entry above?"
If YES → omit it. A condition that is still present has already been reported; it is not
   news simply because it is still there. Report it again ONLY if it has visibly worsened
   since that entry — something is now happening that was not happening then — and in
   that case say plainly what changed.
If NO  → include it.
A single real-world condition must produce exactly one event, however many analysis cycles
it remains visible for.

"""


def build_system_template(*, descriptions: bool = True, scene_delta: bool = False) -> str:
    """The analysis system prompt, with or without the optional "description" field
    and with or without the scene-memory delta.

    When descriptions are off the field is absent from the schema entirely rather than
    being asked for and discarded: every description is 1-2 sentences of generation per
    event, which is latency the operator does not want and tokens they do not pay for.
    The same reasoning governs `scene_delta`, which costs a delta on every call.
    """
    return _SYSTEM_TEMPLATE_HEAD + build_analyze_schema(descriptions=descriptions,
                                                        scene_delta=scene_delta)


_SYSTEM_TEMPLATE = build_system_template()


def _strip_think(text: str) -> str:
    """Remove <think>...</think> blocks emitted by qwen3 reasoning mode."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


class _ThinkStripper:
    """Incremental <think>...</think> suppressor for streamed deltas. Buffers a
    trailing partial-tag candidate so a tag split across chunks is still caught."""

    _OPEN = "<think>"
    _CLOSE = "</think>"

    def __init__(self):
        self._in_think = False
        self._buf = ""  # holds a possible partial tag at the tail

    def feed(self, delta: str) -> str:
        self._buf += delta
        out = []
        while self._buf:
            if not self._in_think:
                i = self._buf.find(self._OPEN)
                if i != -1:
                    out.append(self._buf[:i])
                    self._buf = self._buf[i + len(self._OPEN):]
                    self._in_think = True
                    continue
                # no full open tag; emit all but a possible partial-tag tail
                keep = self._partial_tail_len(self._buf, self._OPEN)
                out.append(self._buf[:len(self._buf) - keep])
                self._buf = self._buf[len(self._buf) - keep:]
                break
            else:
                j = self._buf.find(self._CLOSE)
                if j != -1:
                    self._buf = self._buf[j + len(self._CLOSE):]
                    self._in_think = False
                    continue
                # stay in think; drop all but a possible partial close-tag tail
                keep = self._partial_tail_len(self._buf, self._CLOSE)
                self._buf = self._buf[len(self._buf) - keep:]
                break
        return "".join(out)

    def flush(self) -> str:
        # Any leftover buffer outside a think block is real text; inside, drop it.
        if self._in_think:
            return ""
        out, self._buf = self._buf, ""
        return out

    @staticmethod
    def _partial_tail_len(s: str, tag: str) -> int:
        """Length of the longest suffix of s that is a prefix of tag (a possible
        split tag), so we hold it back rather than emit/scan prematurely."""
        max_k = min(len(s), len(tag) - 1)
        for k in range(max_k, 0, -1):
            if s[-k:] == tag[:k]:
                return k
        return 0


_CHAT_SYSTEM_TEMPLATE = """\
{role}

## YOUR TASK
You are answering an operator's question about the surveillance feeds in a live
chat. The operator's question is the final text in their latest message. Answer
it directly and concisely.

## WHAT YOU CAN SEE
- **Live frames** — attached to the operator's latest message. Each camera's
  frames are preceded by a `=== Camera <id> @ HH:MM:SS to HH:MM:SS ===` header
  giving the source camera and the time range they span; attribute what you
  describe to the right camera and time by that header. A single unlabeled image
  is the only active camera.
- **STREAM ANALYTICS** (below) — live aggregate counts over the whole event
  database (totals, per-camera alerts, today / last 24h, most-recent alert, top
  tags). Use this for "how many", "which camera", and trend questions. Any time in
  this block (e.g. "recorded ... UTC") is a wall-clock date-time — NOT a footage
  position; never report it as a frame timestamp.
- **Conversation memory** — the prior turns of this chat.

## ANSWERING RULES
- Prefer what the **live frames** show for "what is happening now" / "locate the object X" / "point out this specific event"; use the
  **analytics** for counts and trends.
- For historical questions about events, use the event-query
  tool if one is available to you; otherwise a  nswer from the analytics summary
  and say so if the detail isn't there. Do NOT assume what happened earlier from
  the frames alone — the frames only show the time window labeled on them.
- Attribute observations and findings to specific cameras by their id.
- Be concise, descriptive, and direct. If something is unclear or not visible
  in the frames and not present in the data, say so plainly — do not invent it.

## STREAM ANALYTICS (live aggregate)
{analytics}
"""

_SWEEP_SUFFIX = """\

## SEARCH MODE — STRICT OUTPUT FORMAT
You are answering the operator's question using the attached cameras frames.
Each group of frames is labeled with WHERE and WHEN it comes from: a header
`=== Camera <id> @ HH:MM:SS to HH:MM:SS ===` tells you which stream the
following frames belong to and the time range they span (positions in that
camera's footage). Use these labels to localise what you see in space and time.

The ONLY valid source of a time is the `@ HH:MM:SS to HH:MM:SS` part of a header.
Any time you report MUST fall within a range shown in a header above. Never output
a clock/wall-clock time or any time that is not present in a header — when unsure,
quote the header range verbatim.

## STRICT MATCH RULE — read before deciding
Break the question into EVERY attribute it asks for and confirm each one is
clearly, unambiguously visible in these frames before you say FOUND:
  - object/colour: e.g. "red car" requires BOTH a car AND that it is red.
  - action/event: e.g. "collision" requires actual contact or damage between
    vehicles — two cars merely close together, passing, or parked is NOT a collision.
  - person/attribute: every described detail (clothing, action) must be visible.
If ANY attribute is uncertain, partial, "similar", "possible", or only implied,
the answer is NOT_FOUND. Do NOT stretch to make a near-match fit. When in doubt,
choose NOT_FOUND — a missed match is recoverable (the search keeps looking), a
false match is not.

Your reply MUST begin with a single first line that is EXACTLY one of these tokens:
  FOUND <confidence>  - ALL attributes the question asks about ARE clearly visible in these frames. `<confidence>` is a decimal 0.00–1.00 = your certainty that every attribute is genuinely present (e.g. `FOUND 0.93`). On the following lines, answer the question AND state where it happened (the camera id) and when (a HH:MM:SS time taken from the header range it falls in, or that HH:MM:SS–HH:MM:SS range quoted verbatim).
  NOT_FOUND    - the question asks to locate a specific object/action/person in the footage, and at least one asked-for attribute is NOT clearly visible in these frames. Output only this token.
  CONVERSATION - the question is not a request to locate a specific object/action/person in the footage (a greeting, small talk, or a general/meta question). Answer it normally on the following lines.

Do NOT include any other text, commentary, or JSON in your reply. Do NOT invent or assume any information. Only use what you can see in the frames and what is present in the labels.
"""

_DESCRIPTION_BREVITY = """\

## LENGTH
Answer in 3-4 sentences at most, using plain, precise language. State only what is
clearly visible — no speculation, no hedging, no filler or restating the question."""

_INTRUSION_SYSTEM = """\
A zone-intrusion alert has already been confirmed by the surveillance system for camera
{source_id}. The attached frames span the moments around when the intrusion was detected.

A red polygon outline, when visible in the frames, marks the boundary of the restricted
zone that was entered.

Describe specifically the person or object that entered the restricted zone: their
appearance, and actions. Do not report on unrelated activity
elsewhere in the frame that is not part of this intrusion. Be concise, brief, and factual.
""" + _DESCRIPTION_BREVITY

_OVERCROWDING_SYSTEM = """\
An overcrowding alert has already been confirmed by the surveillance system for camera
{source_id}{count_clause}. The attached frames span the moments around when the alert fired.

A red polygon outline, when visible in the frames, marks the boundary of the zone.

Describe the crowd: how people are distributed in the zone, and any notable activity.
Do not report on unrelated activity elsewhere in the frame. Be concise, brief, and factual.
""" + _DESCRIPTION_BREVITY

_EXIT_SYSTEM = """\
A zone-exit alert has already been confirmed by the surveillance system for camera
{source_id}. This zone is a permitted area -- people are expected to stay within it,
and this alert means someone was detected outside its boundary. The attached frames
span the moments around when the exit was detected.

A green polygon outline, when visible in the frames, marks the boundary of the valid
zone that was left.

Describe specifically the person or object that left the valid zone: their appearance,
last known position relative to the zone, and actions. Do not report on unrelated
activity elsewhere in the frame that is not part of this exit. Be concise, brief, and
factual.
""" + _DESCRIPTION_BREVITY

_FIRE_SMOKE_SYSTEM = """You are reviewing frames from surveillance camera {source_id}.
An automated detector has flagged possible fire or smoke, and it has been present
continuously for several seconds.

Your job is to ADJUDICATE, not to embellish. Detectors of this kind are routinely
fooled by steam, water vapour, engine or chimney exhaust, airborne dust, welding
arcs and sparks, sunlight glare, and vehicle headlights.

State plainly what is visible. If it is genuinely fire or smoke, say so and describe
the location, apparent size, and whether it is spreading. If it is NOT -- if it is
steam, vapour, exhaust, dust, welding, glare, a headlight, or anything else -- say
that clearly and directly. A confident "this is not fire" is more useful to an
operator than a hedge."""

_ALERT_SYSTEM_TEMPLATES = {
    "intrusion": _INTRUSION_SYSTEM,
    "exit": _EXIT_SYSTEM,
    "overcrowding": _OVERCROWDING_SYSTEM,
    "fire_smoke": _FIRE_SMOKE_SYSTEM,
}


def _fmt_clock(sec: float) -> str:
    """Stream position as `HH:MM:SS`."""
    total = max(int(round(float(sec))), 0)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _safe_name(name: str) -> str:
    """Filesystem-safe slug for a mode/camera id used in debug paths."""
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(name)) or "x"


def _elide_image_payloads(messages: list) -> list:
    """Copy a chat `messages` list for JSON logging, replacing each base64 image
    block with a short placeholder so the saved prompt stays readable (the actual
    JPEGs are written alongside as files)."""
    out = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            blocks = []
            img_n = 0
            for block in content:
                if isinstance(block, dict) and block.get("type") == "image_url":
                    blocks.append({"type": "image_url", "image_url": f"<frame {img_n}>"})
                    img_n += 1
                else:
                    blocks.append(block)
            new_msg = {k: v for k, v in msg.items() if k != "content"}
            new_msg["content"] = blocks
            out.append(new_msg)
        else:
            out.append(msg)
    return out


_HISTORY_TIERS = ("critical", "warning", "info")


def select_history_by_severity(events: list[dict], per_severity: int) -> dict[str, list[dict]]:
    """Bucket recent events by severity, keeping at most `per_severity` of each.

    `events` is expected in the order `EventStore.get_recent` returns — oldest-first
    (chronological). The model needs to see what it MOST RECENTLY reported, not what
    it reported longest ago, so this walks the input newest-first (`reversed`) and
    stops adding to a tier once that tier holds `per_severity` entries — keeping the
    newest `per_severity` of each tier and discarding the oldest. Each tier's kept
    rows are then reversed back to chronological order before returning, so rendering
    downstream is unaffected by the newest-first selection pass.

    Capping per tier rather than overall matters: a persona that emits mostly warnings
    would otherwise crowd critical and info entirely out of the model's view, and the
    model can only avoid repeating what it can see.

    Rows written before the severity column existed carry `severity: None`; they fall
    back to their event_type (alert -> warning, comment -> info). An unrecognisable
    severity is dropped rather than raising — history is an aid, not the product.
    """
    out: dict[str, list[dict]] = {tier: [] for tier in _HISTORY_TIERS}
    for ev in reversed(events):
        sev = ev.get("severity")
        if sev not in _HISTORY_TIERS:
            if sev is None:
                sev = "info" if ev.get("event_type") == "comment" else "warning"
            else:
                continue
        if len(out[sev]) < per_severity:
            out[sev].append(ev)
    for tier in _HISTORY_TIERS:
        out[tier].reverse()
    return out


class Analyzer:
    def __init__(self, config: dict, metrics=None):
        self._metrics = metrics
        api_cfg = config.get("api", {})
        self._monitoring = config.get("monitoring", {})
        # Off => the schema never asks for "description", and _parse nulls one out if the
        # model volunteers it anyway. Events then carry just the one-sentence message.
        #
        # Defaults to FALSE: a deployment whose config.yaml predates this key gets the
        # short form, which is the wanted behaviour. Note this is deliberately the
        # opposite of build_analyze_schema/_validate_event's own `descriptions=True`
        # defaults -- those describe the full schema (what ANALYZE_SCHEMA pins), while
        # this one is the product decision about what operators actually want emitted.
        self._event_descriptions = bool(self._monitoring.get("event_descriptions", False))
        # The interpreted half of monitoring.memory. `enabled` is the master switch for
        # both halves, so `scene_delta: true` under `enabled: false` is inert -- the
        # measured half is free at inference time, this one is not, and an operator on a
        # small local model must be able to run measured-only.
        _mem_cfg = self._monitoring.get("memory", {}) or {}
        self._scene_delta = bool(_mem_cfg.get("enabled", False)) and \
            bool(_mem_cfg.get("scene_delta", False))
        self._system_template = build_system_template(descriptions=self._event_descriptions,
                                                      scene_delta=self._scene_delta)
        # Set by the chat-context provider (StreamManager) when scene memory is on: a
        # zero-arg callable returning the rendered SCENE MEMORY block for the live
        # streams. A callable and not a stored string because the chat prompt is
        # assembled here, where there is no view of the streams, and the block must be
        # current at the moment of the turn rather than whenever a stream last ran a
        # cycle. Cross-stream by nature (a chat turn shows every camera), so unlike the
        # per-stream delta in `analyze` there is nothing here to leak between streams.
        self.chat_scene_provider = None
        self._backend = api_cfg.get("backend", "cloud")  # "cloud" (DashScope) | "local"

        if self._backend == "local":
            self._temperature = float(api_cfg.get("local_temperature", 0.2))
            self._client = AsyncOpenAI(
                api_key=api_cfg.get("local_api_key", "not-needed"),
                base_url=api_cfg.get("local_base_url", "http://localhost:8001/v1"),
                timeout=api_cfg.get("local_timeout_seconds", 60),
            )
            self._model = api_cfg.get("local_model", "local-vlm")
            self._max_tokens = api_cfg.get("local_max_tokens", 8192)
            # Reasoning models (Qwen3 family) emit a long `reasoning` block before any
            # `content`, so a small max_tokens gets entirely consumed by chain-of-thought
            # and the visible answer is empty.
            #
            # On the OpenAI-compat /v1/chat/completions endpoint the correct param is
            # `reasoning_effort` (Ollama docs), NOT `think: false` which is only for
            # Ollama's native /api/chat. Measured on qwen3-vl:4b (Ollama 0.32.3): neither
            # "none" nor omitting it stops the reasoning — busy scenes can emit 4k+ reasoning
            # tokens — but "low" keeps it bounded. max_tokens must leave room for reasoning
            # PLUS the JSON answer; 8192 does with ~6.5k-token prompts inside a 32k ctx.
            _effort = str(api_cfg.get("local_reasoning_effort", "low"))
            self._extra_body = {"reasoning_effort": _effort} if _effort else {}
            # "image_sequence" = N image_url blocks | "video_url" = frames encoded to mp4
            self._local_visual_format = api_cfg.get("local_visual_format", "image_sequence")
        else:
            self._temperature = float(api_cfg.get("temperature", 0.1))
            self._client = AsyncOpenAI(
                api_key=api_cfg.get("api_key"),
                base_url=api_cfg.get("base_url", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
                timeout=api_cfg.get("timeout_seconds", 20),
            )
            self._model = api_cfg.get("model", "qwen3-vl-32b-instruct")
            self._max_tokens = api_cfg.get("max_tokens", 300)
            # None = don't send the parameter (safe for qwen2-vl); False = disable thinking for qwen3
            _et = api_cfg.get("enable_thinking")
            self._extra_body = {"enable_thinking": _et} if _et is not None else {}
            self._local_visual_format = None

        sampling = config.get("sampling", {})
        interval = max(float(sampling.get("interval_seconds", 6)), 0.1)
        batch_size = max(int(sampling.get("frames_per_batch", 4)), 1)
        self._fps = round(batch_size / interval, 3)

        chat_cfg = config.get("chat", {})
        _te = chat_cfg.get("tools_enabled")
        self._tools_enabled = (self._backend == "cloud") if _te is None else bool(_te)
        self._max_tool_calls = int(chat_cfg.get("max_tool_calls", 3))

        self._prompt_log_path: str = api_cfg.get("prompt_log_path", "data/prompt_log.json")
        self._prompt_log_max: int = int(api_cfg.get("prompt_log_max_entries", 50))
        self._response_log_path: str = api_cfg.get("response_log_path", "data/response_log.json")
        self._response_log_max: int = int(api_cfg.get("response_log_max_entries", 200))

        self._calls_dir: str = api_cfg.get("debug_calls_path", "data/analysis_calls")
        self._calls_keep: int = int(api_cfg.get("debug_calls_keep", 3))
        self._call_slots: dict[str, int] = {}  # per-stream rotating slot counter
        self._last_mp4: bytes | None = None  # set by _frames_to_mp4, written by _save_call_debug

        self._chat_debug_dir: str = api_cfg.get("chat_debug_path", "data/chat_calls")
        self._chat_debug_keep: int = int(api_cfg.get("chat_debug_keep", 3))
        self._chat_debug_slots: dict[str, int] = {}  # per-mode rotating slot counter

        self._desc_debug_dir: str = api_cfg.get("desc_debug_path", "data/description_calls")
        self._desc_debug_keep: int = int(api_cfg.get("desc_debug_keep", 3))
        self._desc_debug_slot: int = 0  # single rotating counter across intrusion + overcrowding

    def _build_visual_content(
        self, frames: list[bytes], b64_frames: list[str],
        offset_start: float | None = None, offset_end: float | None = None,
    ) -> tuple[list[dict], str, str]:
        """Return (visual_content, prompt_text, visual_type) for the current backend.

        When `offset_start` is given (analyze() only — the model otherwise has no
        clock to compare the history block's `[HH:MM:SS-HH:MM:SS]` rows against when
        judging whether a condition is "still present"), a `Current clip: [...]` line
        is prepended to `prompt_text`, formatted the same way history rows are."""
        duration = round(len(frames) / self._fps)
        interval = round(1 / self._fps, 1)

        def _prefix(text: str) -> str:
            if offset_start is None:
                return text
            end_str = self._fmt_offset(offset_end) if offset_end is not None else self._fmt_offset(offset_start)
            return f"Current clip: [{self._fmt_offset(offset_start)}-{end_str}].\n{text}"

        if self._backend == "local":
            if self._local_visual_format == "video_url" and len(frames) > 1:
                # vLLM's OpenAI-compatible video_url block expects a single string
                # (mp4 data URI). vLLM decodes it server-side and samples frames
                # for the model's video processor.
                mp4_b64 = base64.b64encode(self._frames_to_mp4(frames)).decode()
                content = [{"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{mp4_b64}"}}]
                text = f"Analyze this {duration}s security camera clip ({len(frames)} frames at {self._fps} fps)."
                return content, _prefix(text), "video_url"
            elif len(frames) > 1:
                # Multiple image_url blocks — universal for vLLM, Ollama, LMStudio.
                content = [{"type": "image_url", "image_url": {"url": f}} for f in b64_frames]
                text = (
                    f"Analyze this sequence of {len(frames)} security camera frames "
                    f"in order (oldest to newest), captured ~{interval}s apart."
                )
                return content, _prefix(text), "image_url_sequence"
            else:
                content = [{"type": "image_url", "image_url": {"url": b64_frames[0]}}]
                return content, _prefix("Analyze this security camera frame."), "image_url"

        # Cloud / DashScope path
        if len(frames) >= 4:
            content = [{"type": "video", "video": b64_frames, "fps": self._fps}]
            text = f"Analyze this {duration}s security camera clip ({len(frames)} frames at {self._fps} fps)."
            return content, _prefix(text), "video_block"
        elif len(frames) > 1:
            content = [{"type": "image_url", "image_url": {"url": f}} for f in b64_frames]
            text = f"Analyze this sequence of {len(frames)} security camera frames, in order oldest to newest."
            return content, _prefix(text), "image_url_sequence"
        else:
            content = [{"type": "image_url", "image_url": {"url": b64_frames[0]}}]
            return content, _prefix("Analyze this security camera frame."), "image_url"

    def _frames_to_mp4(self, frames: list[bytes], fps: float | None = None,
                       store_last: bool = True) -> bytes:
        """Encode JPEG frames to an in-memory mp4 via cv2. vLLM decodes the mp4
        server-side and samples frames into the model's video processor. `fps`
        overrides the default analysis fps (used for preview clips); `store_last`
        caches the bytes in `_last_mp4` for the call-debug dump (off for previews)."""
        import tempfile
        import cv2
        import numpy as np

        t0 = time.monotonic()
        imgs = [cv2.imdecode(np.frombuffer(f, np.uint8), cv2.IMREAD_COLOR) for f in frames]
        imgs = [im for im in imgs if im is not None]  # skip undecodable frames
        t_decode = time.monotonic() - t0
        if not imgs:
            return b""
        h, w = imgs[0].shape[:2]
        fps = max(float(fps if fps is not None else self._fps), 1.0)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            t1 = time.monotonic()
            writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            for img in imgs:
                writer.write(img)
            writer.release()
            with open(tmp_path, "rb") as fh:
                mp4_bytes = fh.read()
            logger.info(
                "mp4 pack: %d frames %dx%d @ %.1ffps -> %.1fKB (jpeg_decode=%.2fs encode+read=%.2fs total=%.2fs)",
                len(frames), w, h, fps, len(mp4_bytes) / 1024,
                t_decode, time.monotonic() - t1, time.monotonic() - t0,
            )
            if store_last:
                self._last_mp4 = mp4_bytes
            return mp4_bytes
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def _language_directives(self) -> tuple[str, str]:
        """Return (analyze_directive, chat_directive) for the active output language."""
        lang = self._monitoring.get("output_language", "english").lower()
        if lang == "arabic":
            analyze = (
                '\n\n## OUTPUT LANGUAGE\n'
                'Write the "message", "location", "recommended_action" and '
                '"description" field values in Arabic (العربية) only. '
                'Keep "severity", "category" and "tags" in English.'
            )
            chat = "\n\nRespond in Arabic (العربية) only."
            return analyze, chat
        # Small local models drift off-script into other scripts, which can then
        # detokenize into rows of '?' — make English explicit rather than assumed.
        analyze = (
            '\n\n## OUTPUT LANGUAGE\n'
            'Write every field value in English. Never use another script.'
        )
        chat = "\n\nRespond in English."
        return analyze, chat

    def _active_persona(self) -> dict:
        persona_id = self._monitoring.get("active_persona", DEFAULT_PERSONA)
        return PERSONAS.get(persona_id, PERSONAS[DEFAULT_PERSONA])

    async def analyze(
        self,
        frames: list[bytes],
        history: list[dict],
        offset_start: float | None = None,
        offset_end: float | None = None,
        source_id: str | None = None,
        tracked: str = "",
        scene: str = "",
        scene_memory=None,
        live_track_ids=None,
        known_zones=None,
    ) -> list[dict]:
        """One analysis call. Returns the parsed events.

        `tracked` and `scene` are pre-rendered prompt blocks (`TrackObserver.render`
        and `SceneMemory.render`); "" for either costs zero prompt bytes.

        `scene_memory` is the CALLER's `SceneMemory` and is MUTATED IN PLACE: when the
        response carries a memory delta it is merged here, stamped with
        `time.monotonic()` — so the caller must render that same memory with
        `time.monotonic()` too. It is merged here rather than returned because one
        `Analyzer` is shared by every stream, and a delta stashed on `self` would leak
        one stream's scene state into another's.

        `live_track_ids` and `known_zones` are that same caller's live sets, forwarded
        verbatim to the merge: they are what stops an entity note outliving its track
        and an invented zone name rendering forever. `None` (the local backend, which
        has no tracker) skips the filter rather than dropping everything.
        """
        if not frames:
            return []

        self._last_mp4 = None
        persona = self._active_persona()
        instructions = self._monitoring.get("instructions", persona["instructions"])
        history_text = self._format_history(
            history, per_severity=int(self._monitoring.get("history_per_severity", 5)))
        system_prompt = self._system_template.format(
            role=persona["role"],
            instructions=instructions.strip(),
            tracked=tracked,
            scene=scene,
            history=history_text,
        )
        analyze_dir, _ = self._language_directives()
        system_prompt += analyze_dir
        b64_frames = [f"data:image/jpeg;base64,{base64.b64encode(f).decode()}" for f in frames]
        visual_content, prompt_text, visual_type = self._build_visual_content(
            frames, b64_frames, offset_start=offset_start, offset_end=offset_end)

        self._log_prompt(
            system_prompt=system_prompt,
            user_text=prompt_text,
            frame_count=len(frames),
            visual_type=visual_type,
        )

        src = source_id or "local"
        logger.info("[%s] analyze start frames=%d", src, len(frames))
        t0 = time.monotonic()
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                max_tokens=self._max_tokens,
                extra_body=self._extra_body or None,
                temperature=self._temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [*visual_content, {"type": "text", "text": prompt_text}],
                    },
                ],
            )
        except Exception as exc:
            logger.error("[%s] analyze FAILED %.2fs: %s", src, time.monotonic() - t0, exc)
            raise
        dur = time.monotonic() - t0

        raw = response.choices[0].message.content.strip()
        results, delta = self._parse(raw, descriptions=self._event_descriptions)
        if scene_memory is not None and delta is not None:
            # Never on the events' critical path: a memory failure must cost one stale
            # cycle, never a real observation (spec 5.8).
            try:
                scene_memory.merge(delta, time.monotonic(),
                                   live_track_ids=live_track_ids, known_zones=known_zones)
            except Exception as exc:
                logger.warning("[%s] scene memory merge skipped: %s", src, exc)
        types = ",".join(sorted({r.get("type", "") for r in results})) or "none"
        logger.info("[%s] analyze done %.2fs -> %d event(s) (%s)", src, dur, len(results), types)
        if self._metrics is not None:
            self._metrics.record_analyze(src, dur)
        self._log_response(len(frames), results, offset_start, offset_end)
        self._save_call_debug(frames, system_prompt, prompt_text, raw, results, offset_start, offset_end, source_id)
        return results

    @staticmethod
    def _short_text(raw) -> "str | None":
        """A stripped string, or None — never an empty string, so the UI can test for absence."""
        if not isinstance(raw, str):
            return None
        text = raw.strip()
        return text or None

    @staticmethod
    def _validate_event(item: dict, *, descriptions: bool = True) -> dict | None:
        """Validate and normalise a single event dict. Returns None if it should be dropped.

        Severity is primary and `type` is derived from it. An entry carrying only the
        legacy `type` still validates (older prompts, and the traffic/indoor personas
        until Phase 3 rewrites them). A missing or unparseable severity is NEVER a
        reason to drop the entry -- it clamps to "info" (or "warning" when the legacy
        `type` reads "alert"); losing a real observation is worse than logging it a
        notch too quiet. The entry is dropped ONLY for non-dict input or an empty
        message. Severity always clamps downward -- nothing reaches "critical" by
        accident.
        """
        if not isinstance(item, dict):
            return None
        message = str(item.get("message", "")).strip()
        if not message:
            return None
        if message.count("?") / len(message) > 0.6:
            # Small models sometimes drift into another script that detokenizes
            # to literal '?' runs (llama.cpp mapper). A message that is mostly
            # '?' carries no information — drop it rather than store garbage.
            logger.warning("dropping event with uninformative message: %r", message[:80])
            return None
        raw_type = item.get("type", "")
        raw_severity = item.get("severity")
        alert_hint = raw_type == "alert"
        if raw_severity is not None and not (
            isinstance(raw_severity, str) and raw_severity.strip().lower() in SEVERITIES
        ):
            logger.warning(
                "model returned unparseable severity %r (message=%r); clamping to %s",
                raw_severity, message[:80], "warning" if alert_hint else "info",
            )
        severity = normalise_severity(raw_severity, alert_hint=alert_hint)
        tags = item.get("tags", [])
        return {
            "type": derive_event_type(severity),
            "severity": severity,
            "category": normalise_category(item.get("category")),
            "message": message,
            "tags": tags if isinstance(tags, list) else [],
            "location": Analyzer._short_text(item.get("location")),
            "recommended_action": Analyzer._short_text(item.get("recommended_action")),
            # Belt and braces: the schema did not ask for one, but a model that
            # volunteers it anyway must not sneak a description block into the event.
            "description": (Analyzer._short_text(item.get("description"))
                            if descriptions else None),
        }

    async def chat(
        self,
        frame_jpeg: bytes,
        message: str,
        history: list[dict] | None = None,
        prior_turns: list[dict] | None = None,
    ) -> str:
        persona = self._active_persona()
        history_text = self._format_history(history or [])
        system_prompt = _CHAT_SYSTEM_TEMPLATE.format(role=persona["role"], history=history_text)
        _, chat_dir = self._language_directives()
        system_prompt += chat_dir

        b64 = base64.b64encode(frame_jpeg).decode()
        messages: list[dict] = [{"role": "system", "content": system_prompt}]
        if prior_turns:
            messages.extend(prior_turns)
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                    {"type": "text", "text": message},
                ],
            }
        )

        t0 = time.monotonic()
        response = await self._client.chat.completions.create(
            model=self._model,
            max_tokens=600,
            extra_body=self._extra_body or None,
            messages=messages,
        )
        logger.info("chat: %.2fs", time.monotonic() - t0)
        return _strip_think(response.choices[0].message.content.strip())

    async def describe_alert(self, prompt_kind: str, frames: list[bytes],
                             source_id: str, count: "int | None" = None) -> str:
        """Describe a fired alert. `prompt_kind` selects the system prompt; the count
        clause is only rendered for templates that declare it (overcrowding today)."""
        if not frames:
            return ""
        template = _ALERT_SYSTEM_TEMPLATES.get(prompt_kind)
        if template is None:
            raise ValueError(
                f"unknown prompt_kind {prompt_kind!r}; expected one of "
                f"{sorted(_ALERT_SYSTEM_TEMPLATES)}")
        fields = {"source_id": source_id}
        if "{count_clause}" in template:
            fields["count_clause"] = (
                f" — approximately {count} people were detected in the restricted zone, "
                "exceeding the configured threshold" if count is not None else "")
        system_prompt = template.format(**fields)
        _, chat_dir = self._language_directives()
        system_prompt += chat_dir
        return await self._run_description(prompt_kind, system_prompt, frames,
                                           source_id, count)

    async def _run_description(self, debug_kind: str, system_prompt: str,
                               frames: list[bytes], source_id: str,
                               count: "int | None") -> str:
        """The model call shared by every alert description. Lifted verbatim from the
        three describe_* methods this replaced -- they differed only in their prompt and
        their debug label."""
        b64_frames = [f"data:image/jpeg;base64,{base64.b64encode(f).decode()}" for f in frames]
        visual_content, prompt_text, _visual_type = self._build_visual_content(frames, b64_frames)

        response = await self._client.chat.completions.create(
            model=self._model,
            max_tokens=180,
            extra_body=self._extra_body or None,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [*visual_content, {"type": "text", "text": prompt_text}]},
            ],
        )
        answer = _strip_think(response.choices[0].message.content.strip())
        self._save_description_debug(debug_kind, source_id, frames, system_prompt,
                                     prompt_text, answer, count=count)
        return answer

    def _chat_scene(self) -> str:
        """The chat-context provider's SCENE MEMORY block, or "". Never raises.

        Same rule as the analysis path: a memory failure costs one stale block, never
        the turn (spec 5.8).
        """
        provider = self.chat_scene_provider
        if provider is None:
            return ""
        try:
            return provider() or ""
        except Exception as exc:
            logger.warning("chat scene memory skipped: %s", exc)
            return ""

    def _build_chat_messages(self, per_camera_frames, message, history, analytics,
                             prior_turns, max_images, per_camera_offsets=None,
                             scene: str = "") -> list[dict]:
        """Assemble the chat messages list (system + prior turns + labeled image
        user content). Shared by the tool loop and the streaming call. Each
        camera's frames get a `=== Camera <id> @ HH:MM:SS to HH:MM:SS ===` header
        spanning the first and last frame's stream position (when
        ``per_camera_offsets`` is given), so the model can ground its answer in
        time. ``history`` (recent event log) is intentionally NOT injected — see
        the chat endpoint; historical questions go through the query_events tool."""
        persona = self._active_persona()
        analytics_text = self._format_analytics(analytics)
        system_prompt = _CHAT_SYSTEM_TEMPLATE.format(
            role=persona["role"], analytics=analytics_text) + scene
        _, chat_dir = self._language_directives()
        system_prompt += chat_dir

        offsets_map = per_camera_offsets or {}
        # Final guard: enforce the global image cap (defense in depth with gather).
        cameras = [(cid, f, list(offsets_map.get(cid) or []))
                   for cid, f in per_camera_frames.items() if f]
        total = sum(len(f) for _cid, f, _o in cameras)
        if total > max_images and cameras:
            per_cam = max(1, max_images // len(cameras))
            cameras = [(cid, f[-per_cam:], o[-per_cam:]) for cid, f, o in cameras]

        label_cameras = len(cameras) > 1
        content: list[dict] = []
        for cid, frames, offsets in cameras:
            if offsets:
                lo, hi = _fmt_clock(offsets[0]), _fmt_clock(offsets[-1])
                span = lo if lo == hi else f"{lo} to {hi}"
                content.append({"type": "text", "text": f"=== Camera {cid} @ {span} ==="})
            elif label_cameras:
                content.append({"type": "text", "text": f"=== Camera {cid} ==="})
            for jpeg in frames:
                b64 = base64.b64encode(jpeg).decode()
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })
        content.append({"type": "text", "text": message})

        messages: list[dict] = [{"role": "system", "content": system_prompt}]
        if prior_turns:
            messages.extend(prior_turns)
        messages.append({"role": "user", "content": content})
        return messages

    async def chat_multi(self, per_camera_frames, message, store=None, history=None,
                         analytics=None, prior_turns=None, max_images=12,
                         use_tools=True, sweep=False, per_camera_offsets=None,
                         debug_record=None):
        """Cross-stream chat as an async generator. Yields dicts:
        {"type":"status","stage":...} and {"type":"token","delta":...}.

        Runs the query_events tool loop first when tools are enabled and a store
        is provided, then streams the final answer (think-blocks stripped).
        Always uses labeled image_url blocks (NOT a video block) so camera
        attribution survives; single-camera input omits the camera separator.

        ``debug_record``: when a list is given, the final prompt (messages, after
        any tool rounds) plus the frames for this model call are appended to it,
        so the chat endpoint can dump the request for debugging.
        """
        # In search mode (first pass + every sweep probe) drop the analytics block:
        # its "most recent alert" line carries a wall-clock UTC time the model would
        # otherwise confuse with the footage HH:MM:SS positions it must cite.
        chat_analytics = None if sweep else analytics
        # SCENE MEMORY is dropped in search mode for exactly the same reason: its
        # durations are a second clock, and a search answer must cite only the footage
        # HH:MM:SS positions written on the frame headers.
        scene = "" if sweep else self._chat_scene()
        messages = self._build_chat_messages(per_camera_frames, message, history,
                                              chat_analytics, prior_turns, max_images,
                                              per_camera_offsets=per_camera_offsets,
                                              scene=scene)

        if sweep and messages and messages[0]["role"] == "system":
            messages[0]["content"] += _SWEEP_SUFFIX

        run_tools = use_tools and self._tools_enabled and store is not None
        if run_tools:
            for _round in range(self._max_tool_calls):
                resp = await self._client.chat.completions.create(
                    model=self._model, max_tokens=600, extra_body=self._extra_body or None,
                    messages=messages, tools=[QUERY_EVENTS_TOOL], tool_choice="auto",
                )
                msg = resp.choices[0].message
                tool_calls = getattr(msg, "tool_calls", None)
                if not tool_calls:
                    break
                yield {"type": "status", "stage": "querying_events"}
                messages.append({"role": "assistant", "content": msg.content or "",
                                 "tool_calls": [
                                     {"id": tc.id, "type": "function",
                                      "function": {"name": tc.function.name,
                                                   "arguments": tc.function.arguments}}
                                     for tc in tool_calls]})
                for tc in tool_calls:
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    result = await dispatch_tool_call(store, tc.function.name, args)
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        if debug_record is not None:
            debug_record.append({
                "messages": messages,
                "frames": per_camera_frames,
                "offsets": per_camera_offsets,
                "sweep": sweep,
                "use_tools": run_tools,
            })

        # Stream the final answer.
        stripper = _ThinkStripper()
        stream = await self._client.chat.completions.create(
            model=self._model, max_tokens=600, extra_body=self._extra_body or None,
            messages=messages, stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if not delta:
                continue
            clean = stripper.feed(delta)
            if clean:
                yield {"type": "token", "delta": clean}
        tail = stripper.flush()
        if tail:
            yield {"type": "token", "delta": tail}

    def _log_prompt(self, system_prompt: str, user_text: str, frame_count: int, visual_type: str) -> None:
        entry = {
            "t": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "model": self._model,
            "frames": frame_count,
            "visual_type": visual_type,
            "fps": self._fps,
            "system_prompt": system_prompt,
            "user_text": user_text,
        }
        try:
            try:
                with open(self._prompt_log_path, "r", encoding="utf-8") as f:
                    entries: list = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                entries = []
            entries.append(entry)
            if len(entries) > self._prompt_log_max:
                entries = entries[-self._prompt_log_max:]
            with open(self._prompt_log_path, "w", encoding="utf-8") as f:
                json.dump(entries, f, indent=2, ensure_ascii=False)
        except OSError as exc:
            logger.warning("Prompt log write failed: %s", exc)

    def clear_prompt_log(self) -> None:
        try:
            with open(self._prompt_log_path, "w", encoding="utf-8") as f:
                json.dump([], f)
        except OSError as exc:
            logger.warning("Prompt log clear failed: %s", exc)

    @staticmethod
    def _fmt_offset(s: float) -> str:
        hh, rem = divmod(int(s), 3600)
        mm, ss = divmod(rem, 60)
        return f"{hh:02d}:{mm:02d}:{ss:02d}"

    def _log_response(
        self,
        frame_count: int,
        results: list[dict],
        offset_start: float | None,
        offset_end: float | None,
    ) -> None:
        if offset_start is not None:
            end_str = self._fmt_offset(offset_end) if offset_end is not None else self._fmt_offset(offset_start)
            ts = f"[{self._fmt_offset(offset_start)}-{end_str}]"
        else:
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        entry = {
            "t": ts,
            "frames": frame_count,
            "results": results,
        }
        try:
            try:
                with open(self._response_log_path, "r", encoding="utf-8") as f:
                    entries: list = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                entries = []
            entries.append(entry)
            if len(entries) > self._response_log_max:
                entries = entries[-self._response_log_max:]
            with open(self._response_log_path, "w", encoding="utf-8") as f:
                json.dump(entries, f, indent=2, ensure_ascii=False)
        except OSError as exc:
            logger.warning("Response log write failed: %s", exc)

    def clear_response_log(self) -> None:
        try:
            with open(self._response_log_path, "w", encoding="utf-8") as f:
                json.dump([], f)
        except OSError as exc:
            logger.warning("Response log clear failed: %s", exc)

    def _save_call_debug(
        self,
        frames: list[bytes],
        system_prompt: str,
        prompt_text: str,
        raw: str,
        results: list[dict],
        offset_start: float | None,
        offset_end: float | None,
        source_id: str | None = None,
    ) -> None:
        if not self._calls_dir:
            return
        label = source_id or "default"  # one subfolder per stream; "default" for the local/single path
        slot = self._call_slots.get(label, 0) % self._calls_keep
        self._call_slots[label] = self._call_slots.get(label, 0) + 1
        slot_dir = os.path.join(self._calls_dir, label, str(slot))
        try:
            os.makedirs(slot_dir, exist_ok=True)
            for fname in os.listdir(slot_dir):
                if fname.endswith((".jpg", ".mp4")) or fname == "meta.json":
                    os.remove(os.path.join(slot_dir, fname))
            for i, frame in enumerate(frames):
                with open(os.path.join(slot_dir, f"frame_{i:02d}.jpg"), "wb") as fh:
                    fh.write(frame)
            if self._last_mp4 is not None:
                with open(os.path.join(slot_dir, "clip.mp4"), "wb") as fh:
                    fh.write(self._last_mp4)
            if offset_start is not None:
                end_str = self._fmt_offset(offset_end) if offset_end is not None else self._fmt_offset(offset_start)
                ts = f"[{self._fmt_offset(offset_start)}-{end_str}]"
            else:
                ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            meta = {
                "t": ts,
                "source_id": label,
                "wall_time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "backend": self._backend,
                "model": self._model,
                "frame_count": len(frames),
                "fps": self._fps,
                "system_prompt": system_prompt,
                "prompt_text": prompt_text,
                "raw_response": raw,
                "results": results,
            }
            with open(os.path.join(slot_dir, "meta.json"), "w", encoding="utf-8") as fh:
                json.dump(meta, fh, indent=2, ensure_ascii=False)
        except OSError as exc:
            logger.warning("Call debug save failed: %s", exc)

    def clear_call_debug(self) -> None:
        if not self._calls_dir:
            return
        try:
            if os.path.isdir(self._calls_dir):
                shutil.rmtree(self._calls_dir)
            self._call_slots = {}
        except OSError as exc:
            logger.warning("Call debug clear failed: %s", exc)

    def _save_description_debug(
        self,
        prompt_kind: str,
        source_id: str,
        frames: list[bytes],
        system_prompt: str,
        prompt_text: str,
        answer: str,
        count: "int | None" = None,
    ) -> None:
        """Dump one describe_alert call's frames + prompt to a rotating slot. Keeps the
        most recent `desc_debug_keep` calls total (all prompt kinds share the same
        rotation, most-recent-first)."""
        if not self._desc_debug_dir:
            return
        keep = max(self._desc_debug_keep, 1)
        slot = self._desc_debug_slot % keep
        self._desc_debug_slot += 1
        slot_dir = os.path.join(self._desc_debug_dir, str(slot))
        try:
            if os.path.isdir(slot_dir):
                shutil.rmtree(slot_dir)  # wipe the previous occupant of this slot
            os.makedirs(slot_dir, exist_ok=True)
            for i, frame in enumerate(frames):
                with open(os.path.join(slot_dir, f"frame_{i:02d}.jpg"), "wb") as fh:
                    fh.write(frame)
            meta = {
                "wall_time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "prompt_kind": prompt_kind,
                "source_id": source_id,
                "count": count,
                "backend": self._backend,
                "model": self._model,
                "frame_count": len(frames),
                "system_prompt": system_prompt,
                "prompt_text": prompt_text,
                "answer": answer,
            }
            with open(os.path.join(slot_dir, "meta.json"), "w", encoding="utf-8") as fh:
                json.dump(meta, fh, indent=2, ensure_ascii=False)
        except OSError as exc:
            logger.warning("Description debug save failed: %s", exc)

    def clear_description_debug(self) -> None:
        if not self._desc_debug_dir:
            return
        try:
            if os.path.isdir(self._desc_debug_dir):
                shutil.rmtree(self._desc_debug_dir)
            self._desc_debug_slot = 0
        except OSError as exc:
            logger.warning("Description debug clear failed: %s", exc)

    def save_chat_debug(self, mode: str, message: str, calls: list,
                        answer: str, session_id: str | None = None) -> None:
        """Dump one chat request's full prompt(s) + frames to a rotating slot.

        Keeps the most recent `chat_debug_keep` requests per mode under
        `<chat_debug_dir>/<mode>/<slot>/`. A request may make several model calls
        (a search turn does a first pass plus sweep rounds); each call's frames
        are written as JPEGs and its full message list (image payloads elided) is
        recorded in meta.json so the exact prompt is recoverable."""
        if not self._chat_debug_dir or not calls:
            return
        label = _safe_name(mode or "normal")
        keep = max(self._chat_debug_keep, 1)
        slot = self._chat_debug_slots.get(label, 0) % keep
        self._chat_debug_slots[label] = self._chat_debug_slots.get(label, 0) + 1
        slot_dir = os.path.join(self._chat_debug_dir, label, str(slot))
        try:
            if os.path.isdir(slot_dir):
                shutil.rmtree(slot_dir)  # wipe the previous occupant of this slot
            os.makedirs(slot_dir, exist_ok=True)
            call_metas = []
            for ci, call in enumerate(calls):
                frames = call.get("frames") or {}
                offsets = call.get("offsets") or {}
                frame_files = []
                for cid, jpegs in frames.items():
                    for fi, jpeg in enumerate(jpegs):
                        fname = f"call{ci:02d}_{_safe_name(cid)}_{fi:02d}.jpg"
                        with open(os.path.join(slot_dir, fname), "wb") as fh:
                            fh.write(jpeg)
                        frame_files.append(fname)
                call_metas.append({
                    "index": ci,
                    "sweep": call.get("sweep"),
                    "tool_loop_ran": call.get("use_tools"),
                    "frames_per_camera": {cid: len(j) for cid, j in frames.items()},
                    "offsets": {cid: list(o) for cid, o in offsets.items()},
                    "frame_files": frame_files,
                    "messages": _elide_image_payloads(call.get("messages") or []),
                })
            meta = {
                "wall_time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "mode": label,
                "session_id": session_id,
                "message": message,
                "backend": self._backend,
                "model": self._model,
                "model_calls": len(calls),
                "answer": answer,
                "calls": call_metas,
            }
            with open(os.path.join(slot_dir, "meta.json"), "w", encoding="utf-8") as fh:
                json.dump(meta, fh, indent=2, ensure_ascii=False)
        except OSError as exc:
            logger.warning("Chat debug save failed: %s", exc)

    def clear_chat_debug(self) -> None:
        if not self._chat_debug_dir:
            return
        try:
            if os.path.isdir(self._chat_debug_dir):
                shutil.rmtree(self._chat_debug_dir)
            self._chat_debug_slots = {}
        except OSError as exc:
            logger.warning("Chat debug clear failed: %s", exc)

    @staticmethod
    def _format_history(history: list[dict], include_camera: bool = False,
                        per_severity: "int | None" = None) -> str:
        if not history:
            return "No previous observations."

        def _fmt_offset(s: float) -> str:
            hh, rem = divmod(int(s), 3600)
            mm, ss = divmod(rem, 60)
            return f"{hh:02d}:{mm:02d}:{ss:02d}"

        def _line(h: dict, label: "str | None") -> str:
            start = h.get("stream_offset_start")
            end = h.get("stream_offset_end")
            if start is not None:
                end_str = _fmt_offset(end) if end is not None else _fmt_offset(start)
                time_str = f"[{_fmt_offset(start)}-{end_str}]"
            else:
                time_str = f"[{h.get('timestamp', '?')}]"
            cam = h.get("camera")
            cam_str = f" [{cam}]" if (include_camera and cam) else ""
            prefix = f"{label}: " if label else ""
            return f"{time_str}{cam_str} {prefix}{h.get('message', '')}"

        if per_severity is None:
            # Flat legacy rendering — exists for Analyzer.chat() and its direct tests.
            # chat() itself has no callers anywhere in app/ today; production chat goes
            # through chat_multi() instead.
            return "\n".join(_line(h, (h.get("event_type") or "?").upper()) for h in history)

        grouped = select_history_by_severity(history, per_severity)
        blocks = []
        for tier in _HISTORY_TIERS:
            rows = grouped[tier]
            if not rows:
                continue  # an empty heading costs tokens and implies nothing was seen
            blocks.append(f"{tier.upper()}:\n" + "\n".join("  " + _line(h, None) for h in rows))
        return "\n".join(blocks) if blocks else "No previous observations."

    @staticmethod
    def _format_analytics(summary: dict | None) -> str:
        """Compact aggregate block for the chat system prompt. Both backends."""
        if not summary or (summary.get("total_alerts", 0) == 0
                           and summary.get("total_comments", 0) == 0):
            return "No events recorded yet."
        lines = [
            f"Totals: {summary['total_alerts']} alerts, {summary['total_comments']} comments"
            f" | Today: {summary['alerts_today']} alerts | Last 24h: {summary['alerts_24h']} alerts",
        ]
        if summary["per_camera"]:
            pc = ", ".join(f"{p['camera']}={p['alerts']}" for p in summary["per_camera"])
            lines.append(f"Per camera: {pc}")
        mr = summary.get("most_recent_alert")
        if mr:
            ts = mr.get("timestamp") or ""
            # Full dated UTC wall-clock stamp so it can never be mistaken for a
            # footage position (HH:MM:SS) shown in the frame headers.
            when = f"recorded {ts[:10]} {ts[11:19]} UTC" if len(ts) >= 19 else "recorded ?"
            lines.append(f"Most recent alert [{mr['camera']}, {when}]: {mr['message']}")
        if summary["top_tags"]:
            tt = ", ".join(f"{t['tag']} ({t['count']})" for t in summary["top_tags"])
            lines.append(f"Top alert tags: {tt}")
        return "\n".join(lines)

    @classmethod
    def _parse(cls, raw: str, *, descriptions: bool = True) -> tuple[list[dict], dict | None]:
        """`(events, scene delta)` for all three response shapes.

        `{"events": [...], "memory": {...}}` is the scene_delta shape; a bare array is
        today's; a bare object is the legacy single-event one. The delta is `None` for
        both legacy shapes and whenever the `memory` half is absent or unusable —
        version-skew tolerance is the house pattern, and a memory failure must never
        cost an event (spec 5.8), so nothing about the memory half can reach the
        events.
        """
        # Strip <think>...</think> blocks (qwen3 thinking mode) then markdown fences
        cleaned = _strip_think(raw)
        cleaned = re.sub(r"```(?:json)?\s*(.*?)\s*```", r"\1", cleaned, flags=re.DOTALL).strip()
        try:
            result = json.loads(cleaned)
        except json.JSONDecodeError:
            # Fall back: treat raw text as a single comment
            event = cls._validate_event({"type": "comment", "message": raw[:300], "tags": []},
                                        descriptions=descriptions)
            return ([event] if event else []), None

        if isinstance(result, list):
            return [e for item in result
                    if (e := cls._validate_event(item, descriptions=descriptions))], None
        if isinstance(result, dict):
            if isinstance(result.get("events"), list):
                events = [e for item in result["events"]
                          if (e := cls._validate_event(item, descriptions=descriptions))]
                return events, cls._parse_delta(result.get("memory"))
            # Model used the old single-object format — wrap and validate
            event = cls._validate_event(result, descriptions=descriptions)
            return ([event] if event else []), None
        return [], None

    @staticmethod
    def _parse_delta(raw) -> dict | None:
        """The `memory` half, or None when it is absent or not an object.

        `{}` is NOT None: the prompt asks for it on a quiet cycle, and merging an empty
        delta is what ages unconfirmed records out. `validate_delta` never raises by
        contract; the guard is here anyway because this sits on the path that returns
        the events.
        """
        if not isinstance(raw, dict):
            return None
        try:
            return validate_delta(raw)
        except Exception:  # pragma: no cover - validate_delta is contractually total
            return None
