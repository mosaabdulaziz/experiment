"""The monitoring prompt must describe every field the validator accepts.

A field the prompt never mentions is a field the model never emits, so this pins
the schema half of the prompt against the validator's field list. Phase 3 adds the
severity RUBRIC (when to choose critical over warning) — see
test_analyzer_prompt_schema.py's later tests for that.
"""
import re

from app.analyzer import ANALYZE_SCHEMA


def _flat(s: str) -> str:
    """Collapse whitespace/newlines so a wording check survives line wrapping."""
    return re.sub(r"\s+", " ", s)


def test_prompt_maps_the_persona_alert_comment_vocabulary_onto_severity():
    """The shipped personas (app/personas.py) still say ALERT/COMMENT; the schema
    must translate that vocabulary onto severity so the model isn't left to invent
    the mapping itself (this is a translation, not a severity rubric)."""
    flat = _flat(ANALYZE_SCHEMA)
    assert 'treat ALERT as at least "warning"' in flat
    assert 'COMMENT as "info"' in flat


def test_prompt_documents_every_validated_field():
    for field in ("severity", "category", "location", "recommended_action", "description"):
        assert field in ANALYZE_SCHEMA, f"prompt never mentions {field!r}"


def test_prompt_lists_the_closed_severity_set():
    for value in ("critical", "warning", "info"):
        assert value in ANALYZE_SCHEMA


def test_prompt_lists_the_closed_category_set():
    for value in ("safety", "security", "storage", "operations"):
        assert value in ANALYZE_SCHEMA


def test_prompt_says_nullable_fields_may_be_omitted():
    assert "null" in ANALYZE_SCHEMA.lower()


def test_prompt_defines_all_three_severity_anchors():
    """Without concrete anchors the model inflates severity."""
    for anchor in ("immediate danger", "needs attention", "no action needed"):
        assert anchor in ANALYZE_SCHEMA.lower(), f"missing severity anchor: {anchor!r}"


def test_prompt_tells_the_model_to_choose_the_lower_severity_when_torn():
    """Pairs with normalise_severity's downward clamp — the bias lives in both places."""
    lowered = ANALYZE_SCHEMA.lower()
    assert "choose the lower" in lowered
    assert "torn between two levels" in lowered


def test_prompt_still_bridges_the_legacy_alert_comment_vocabulary():
    """traffic and indoor_security are untouched and still say ALERT/COMMENT."""
    assert "ALERT" in ANALYZE_SCHEMA
    assert "COMMENT" in ANALYZE_SCHEMA


def test_prompt_no_longer_carries_the_phase2_placeholder_clause():
    """Phase 2's stop-gap said 'critical when a person is in immediate danger' inside the
    vocabulary bridge. The rubric replaces it; the bridge must not still carry it."""
    assert 'or "critical" when a person is in immediate danger' not in ANALYZE_SCHEMA


def test_prompt_states_the_reasoning_flow_and_maps_it_to_fields():
    lowered = ANALYZE_SCHEMA.lower()
    for step in ("observation", "context", "risk", "action"):
        assert step in lowered, f"reasoning flow missing: {step}"
    assert 'put 1 in "message"' in ANALYZE_SCHEMA


from app.analyzer import _SYSTEM_TEMPLATE


def test_history_check_states_the_standing_condition_rule():
    """A condition that is still present is not news — the old 'same incident?' wording
    let a model justify re-reporting it in fresh words every cycle."""
    lowered = _SYSTEM_TEMPLATE.lower()
    assert "still present" in lowered
    assert "already reported" in lowered


def test_history_check_allows_re_reporting_only_on_visible_worsening():
    """The exception keys on observable change, not a severity re-derivation — the
    schema's "when torn, choose the LOWER severity" bias would otherwise fight the
    only condition that permits re-reporting a worsening situation."""
    lowered = _SYSTEM_TEMPLATE.lower()
    assert "visibly worsened" in lowered
    assert "what changed" in lowered


def test_history_check_states_the_one_event_per_condition_rule():
    lowered = _SYSTEM_TEMPLATE.lower()
    assert "exactly one event" in lowered


# ── measured memory: the TRACKED OBJECTS block in the system prompt ────────────

import asyncio
from types import SimpleNamespace

from app.analyzer import Analyzer
from app.track_observer import TrackObserver


def _render(tracked="", scene=""):
    return _SYSTEM_TEMPLATE.format(role="ROLE", instructions="INSTR",
                                   history="HIST", tracked=tracked, scene=scene)


def _tracked_block():
    obs = TrackObserver(frame_width=1920, frame_height=1080)
    obs.observe(0.0, [("truck", 0.9, 23, (100.0, 100.0, 40.0, 40.0))])
    return obs.render(60.0)


def test_prompt_is_byte_identical_when_no_tracked_block_is_supplied():
    """monitoring.memory.enabled: false must cost ZERO prompt bytes, not merely
    produce an equivalent prompt — the {tracked} slot sits alone on its own line so
    an empty value collapses back to exactly the pre-memory spacing."""
    assert "## MONITORING INSTRUCTIONS\nINSTR\n\n## RECENT OBSERVATION HISTORY" in _render()
    # ...and structurally: the slot is alone on its line, so an empty value leaves that
    # line as the blank separator the sections already had. Appending it to the
    # instructions line instead would swallow a newline; giving it padding of its own
    # would leave a stray blank one.
    assert _SYSTEM_TEMPLATE.count("{tracked}") == 1
    # {scene} shares the line rather than taking one of its own: two slot lines would
    # add a newline the pre-memory prompt never had. Both render newline-wrapped
    # blocks, so sharing spaces them correctly and collapses to nothing when empty.
    assert "\n{tracked}{scene}\n" in _SYSTEM_TEMPLATE
    assert "\n\n\n" not in _render()


def test_tracked_block_sits_between_the_instructions_and_the_history():
    rendered = _render(_tracked_block())
    assert (rendered.index("## MONITORING INSTRUCTIONS")
            < rendered.index("## TRACKED OBJECTS")
            < rendered.index("## RECENT OBSERVATION HISTORY"))


def test_tracked_block_keeps_the_blank_lines_around_it():
    """render()'s leading/trailing newlines are what make the slot collapse cleanly
    when empty; when it is present they must still produce the section spacing the
    rest of the prompt uses."""
    rendered = _render(_tracked_block())
    assert "INSTR\n\n## TRACKED OBJECTS" in rendered
    assert "\n\n## RECENT OBSERVATION HISTORY" in rendered


def test_tracked_block_tells_the_model_the_durations_are_exact():
    """Prompt text doing real work: without it the model re-estimates durations it
    could simply have read, which is the whole defect this block exists to fix."""
    assert "exact, not estimates" in _tracked_block()


def test_analyze_threads_the_tracked_block_into_the_system_prompt(tmp_path):
    """The slot is useless if analyze() never fills it."""
    az = Analyzer({"api": {"backend": "cloud", "api_key": "x", "model": "m",
                           "prompt_log_path": str(tmp_path / "p.json"),
                           "response_log_path": str(tmp_path / "r.json"),
                           "debug_calls_path": str(tmp_path / "calls")},
                   "monitoring": {}, "sampling": {}, "chat": {}})
    calls = {}

    async def create(*a, **k):
        calls["kwargs"] = k
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="[]"))])

    az._client.chat.completions.create = create
    asyncio.run(az.analyze([b"a"], history=[], tracked=_tracked_block()))
    assert "## TRACKED OBJECTS" in calls["kwargs"]["messages"][0]["content"]


def test_analyze_without_tracked_leaves_the_prompt_unchanged(tmp_path):
    az = Analyzer({"api": {"backend": "cloud", "api_key": "x", "model": "m",
                           "prompt_log_path": str(tmp_path / "p.json"),
                           "response_log_path": str(tmp_path / "r.json"),
                           "debug_calls_path": str(tmp_path / "calls")},
                   "monitoring": {}, "sampling": {}, "chat": {}})
    calls = {}

    async def create(*a, **k):
        calls["kwargs"] = k
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="[]"))])

    az._client.chat.completions.create = create
    asyncio.run(az.analyze([b"a"], history=[]))
    system = calls["kwargs"]["messages"][0]["content"]
    assert "TRACKED OBJECTS" not in system
    assert "## RECENT OBSERVATION HISTORY" in system


# ── optional event descriptions (monitoring.event_descriptions) ────────────────

import json as _json

from app.analyzer import build_analyze_schema, build_system_template


def _example_json(schema: str):
    """The example OUTPUT, parsed. Guards the hardest failure mode of stripping a
    field out of a JSON example: a dangling comma the model would then imitate.

    Splits on the closing "Return ..." sentence rather than on "Return []" so it works
    for the scene-delta variant too, whose example is an object and whose closing line
    talks about an empty "events" array instead.
    """
    body = schema.split("## Example OUTPUT", 1)[1].split("\nReturn ", 1)[0].strip()
    return _json.loads(body.replace("{{", "{").replace("}}", "}"))


def _analyzer(tmp_path, **monitoring):
    return Analyzer({"api": {"backend": "cloud", "api_key": "x", "model": "m",
                             "prompt_log_path": str(tmp_path / "p.json"),
                             "response_log_path": str(tmp_path / "r.json"),
                             "debug_calls_path": str(tmp_path / "calls")},
                     "monitoring": monitoring, "sampling": {}, "chat": {}})


def test_descriptions_on_is_the_default_and_matches_the_module_constant():
    """Omitting the config key entirely must leave the prompt exactly as it shipped."""
    assert build_analyze_schema() == ANALYZE_SCHEMA
    assert build_analyze_schema(descriptions=True) == ANALYZE_SCHEMA


def test_descriptions_off_removes_every_mention_of_the_field():
    off = build_analyze_schema(descriptions=False)
    assert "description" not in off
    assert "description" in ANALYZE_SCHEMA  # the ON variant still has it


def test_descriptions_off_still_emits_a_valid_json_example():
    """Stripping the last field of an object leaves a trailing comma unless the comma
    travels with the field. The model copies this example, so it has to parse."""
    events = _example_json(build_analyze_schema(descriptions=False))
    assert len(events) == 3
    assert all("description" not in e for e in events)
    assert events[0]["severity"] == "critical"
    assert any(e["severity"] == "warning" for e in events)


def test_descriptions_on_example_still_parses_and_keeps_the_field():
    events = _example_json(ANALYZE_SCHEMA)
    assert events[0]["description"]
    assert events[1]["description"] is None


def test_descriptions_off_still_demands_the_risk_reasoning():
    """Steps 2 and 3 have nowhere to be written, but severity is derived from step 3 —
    dropping the field must not quietly drop the reasoning that sets severity."""
    off = build_analyze_schema(descriptions=False)
    assert 'put 1 in "message"' in off
    assert 'step 3 decide "severity"' in off


def test_system_template_off_keeps_the_analysis_sections():
    off = build_system_template(descriptions=False).format(
        role="R", instructions="I", history="H", tracked="", scene="")
    assert "## MONITORING INSTRUCTIONS" in off
    assert "## RECENT OBSERVATION HISTORY" in off
    assert "description" not in off


def test_validate_event_drops_a_description_the_model_volunteered():
    """The schema stopped asking, but a model may emit one anyway — it must not reach
    the event, or turning the setting off would not actually shorten what is stored."""
    item = {"severity": "warning", "category": "safety", "message": "m", "tags": [],
            "description": "a long block of context nobody asked for"}
    assert Analyzer._validate_event(item, descriptions=False)["description"] is None
    assert Analyzer._validate_event(item)["description"] is not None


def test_parse_drops_volunteered_descriptions_end_to_end():
    raw = '[{"severity":"warning","category":"safety","message":"m","tags":[],' \
          '"description":"unrequested"}]'
    assert Analyzer._parse(raw, descriptions=False)[0][0]["description"] is None
    assert Analyzer._parse(raw)[0][0]["description"] == "unrequested"


def test_analyzer_defaults_to_no_descriptions(tmp_path):
    """A config.yaml predating the key must get the short form, not the long one."""
    assert _analyzer(tmp_path)._event_descriptions is False         # absent key
    assert _analyzer(tmp_path, event_descriptions=True)._event_descriptions is True
    assert _analyzer(tmp_path, event_descriptions=False)._event_descriptions is False


def test_the_default_analyzer_sends_a_schema_with_no_description_field(tmp_path):
    """The config default is what a fresh deployment gets, so pin it end to end rather
    than only at the flag: build_analyze_schema's own default is the opposite."""
    assert '"description"' not in _analyzer(tmp_path)._system_template


def test_analyze_sends_the_stripped_schema_when_descriptions_are_off(tmp_path):
    az = _analyzer(tmp_path, event_descriptions=False)
    calls = {}

    async def create(*a, **k):
        calls["kwargs"] = k
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content='[{"severity":"warning","category":"safety","message":"m","tags":[],'
                    '"description":"sneaked in"}]'))])

    az._client.chat.completions.create = create
    results = asyncio.run(az.analyze([b"a"], history=[]))
    # The JSON field name, not the bare word: a persona's own prose may say
    # "description" for unrelated reasons (the traffic persona does).
    assert '"description"' not in calls["kwargs"]["messages"][0]["content"]
    assert results[0]["description"] is None


# ── interpreted memory: the scene-delta half of the schema (monitoring.memory) ──

from app.scene_memory import SceneMemory, validate_delta

_MEMORY_SLOTS = ("entities", "areas", "situations", "scene_notes")


def test_scene_delta_off_is_the_default_and_matches_the_module_constant():
    """ANALYZE_SCHEMA is pinned by every assertion above it; adding a second variant
    must not move it, exactly as adding the descriptions variant did not."""
    assert build_analyze_schema() == ANALYZE_SCHEMA
    assert build_analyze_schema(scene_delta=False) == ANALYZE_SCHEMA
    assert build_system_template(scene_delta=False) == _SYSTEM_TEMPLATE


def test_scene_delta_off_mentions_none_of_the_memory_slots():
    """Off must cost ZERO prompt tokens — the delta is charged on every single call."""
    off = build_analyze_schema(scene_delta=False)
    assert '"memory"' not in off
    for slot in _MEMORY_SLOTS:
        assert f'"{slot}"' not in off, f"scene_delta=False still mentions {slot!r}"


def test_scene_delta_on_asks_for_the_memory_object_and_names_every_slot():
    on = build_analyze_schema(scene_delta=True)
    assert '"memory"' in on
    assert '"events"' in on
    for slot in _MEMORY_SLOTS:
        assert f'"{slot}"' in on, f"scene_delta=True never names {slot!r}"


def test_scene_delta_on_example_output_still_parses_as_json():
    """The model copies this example verbatim. Wrapping the events array in an object
    is exactly where a dangling comma or an unbalanced brace would hide."""
    example = _example_json(build_analyze_schema(scene_delta=True))
    assert isinstance(example, dict)
    assert len(example["events"]) == 3
    assert example["events"][0]["severity"] == "critical"
    assert any(e["severity"] == "warning" for e in example["events"])
    memory = example["memory"]
    for slot in _MEMORY_SLOTS:
        assert slot in memory, f"the example never demonstrates {slot!r}"
    assert memory["situations"][0]["key"]
    assert memory["situations"][0]["state"] == "ongoing"


def test_scene_delta_on_example_survives_descriptions_being_off():
    """Two independent switches multiply into four prompts; the example has to parse
    in all of them, and the descriptions variant is the one that edits field commas."""
    example = _example_json(build_analyze_schema(descriptions=False, scene_delta=True))
    assert len(example["events"]) == 3
    assert all("description" not in e for e in example["events"])
    assert example["memory"]["scene_notes"]


def test_scene_delta_on_example_matches_what_validate_delta_accepts():
    """The worked example is the model's template; if SceneMemory drops it, the prompt
    is teaching a shape the app throws away."""
    memory = _example_json(build_analyze_schema(scene_delta=True))["memory"]
    delta = validate_delta(memory)
    assert delta["situations"] and delta["areas"] and delta["entities"]
    assert delta["scene_notes"]


def test_scene_delta_says_an_omitted_scene_notes_means_no_change():
    """Settled 2026-09-01: absent != clear. Without this sentence the model drops the
    field on a quiet cycle and is surprised the stored notes survive."""
    flat = _flat(build_analyze_schema(scene_delta=True)).lower()
    assert "no change" in flat
    assert "cleared" in flat


def test_scene_delta_asks_the_model_to_reuse_the_same_key_across_cycles():
    """The accumulating duration is the entire point of the phase, and it is keyed on
    a slug the model has to keep stable itself."""
    flat = _flat(build_analyze_schema(scene_delta=True))
    assert "same" in flat.lower() and '"key"' in flat


def test_scene_delta_asks_for_confirmation_instead_of_a_repeat_event():
    flat = _flat(build_analyze_schema(scene_delta=True)).lower()
    assert "confirm it" in flat
    assert "instead of emitting an event" in flat


def test_scene_delta_is_not_framed_as_deduplication():
    """Scope decision 2026-09-01: this phase is state, not a dedup mechanism. Telling
    the model it is deduplicating invites it to suppress genuinely new events."""
    lowered = build_analyze_schema(scene_delta=True).lower()
    for word in ("duplicate", "deduplicat", "de-dup"):
        assert word not in lowered, f"the delta is framed as dedup ({word!r})"


def test_scene_slot_costs_zero_bytes_when_empty():
    """The strong form of the byte-identity requirement: rendering with scene="" must
    equal rendering a template that has no {scene} slot at all."""
    no_slot = _SYSTEM_TEMPLATE.replace("{scene}", "")
    for tracked in ("", _tracked_block()):
        rendered = _SYSTEM_TEMPLATE.format(role="ROLE", instructions="INSTR",
                                           history="HIST", tracked=tracked, scene="")
        legacy = no_slot.format(role="ROLE", instructions="INSTR",
                                history="HIST", tracked=tracked)
        assert rendered == legacy


def _scene_block():
    mem = SceneMemory()
    mem.merge({"situations": [{"key": "bay2", "description": "truck at Bay 2 being unloaded",
                               "state": "ongoing"}]}, 0.0)
    return mem.render(60.0)


def test_scene_block_sits_between_the_tracked_block_and_the_history():
    rendered = _render(tracked=_tracked_block(), scene=_scene_block())
    assert (rendered.index("## MONITORING INSTRUCTIONS")
            < rendered.index("## TRACKED OBJECTS")
            < rendered.index("## SCENE MEMORY")
            < rendered.index("## RECENT OBSERVATION HISTORY"))
    assert "\n\n\n" not in rendered


def test_scene_block_alone_still_gets_the_section_spacing():
    """Measured memory off, interpreted memory on is a real configuration."""
    rendered = _render(scene=_scene_block())
    assert "INSTR\n\n## SCENE MEMORY" in rendered
    assert "\n\n## RECENT OBSERVATION HISTORY" in rendered


# ── _parse: three response shapes, and a memory half that must never cost events ──

_EVENT = '{"severity":"warning","category":"safety","message":"m","tags":[]}'


def test_parse_reads_the_new_object_shape():
    raw = ('{"events":[' + _EVENT + '],'
           '"memory":{"situations":[{"key":"k","description":"d","state":"ongoing"}],'
           '"scene_notes":"quiet"}}')
    events, delta = Analyzer._parse(raw)
    assert [e["message"] for e in events] == ["m"]
    assert delta["situations"][0]["key"] == "k"
    assert delta["scene_notes"] == "quiet"


def test_parse_returns_no_delta_for_the_bare_array_shape():
    events, delta = Analyzer._parse("[" + _EVENT + "]")
    assert [e["message"] for e in events] == ["m"]
    assert delta is None


def test_parse_returns_no_delta_for_the_legacy_single_object_shape():
    events, delta = Analyzer._parse(_EVENT)
    assert [e["message"] for e in events] == ["m"]
    assert delta is None


def test_parse_returns_no_delta_for_unparseable_text():
    events, delta = Analyzer._parse("I could not comply.")
    assert len(events) == 1
    assert delta is None


def test_parse_keeps_the_events_when_the_memory_key_is_absent():
    """An older model, or a truncated reply, simply omits it. Reading it with [] would
    raise here and lose a whole cycle's events (spec 5.8)."""
    events, delta = Analyzer._parse('{"events":[' + _EVENT + ']}')
    assert [e["message"] for e in events] == ["m"]
    assert delta is None


def test_parse_keeps_the_events_when_the_memory_half_is_garbage():
    """A memory failure must never cost an event. Every one of these is a shape a
    fumbled generation actually produces."""
    for junk in ('"a string"', '["a", "list"]', "null", "17", "true"):
        raw = '{"events":[' + _EVENT + '],"memory":' + junk + "}"
        events, delta = Analyzer._parse(raw)
        assert [e["message"] for e in events] == ["m"], f"events lost for memory={junk}"
        assert delta is None, f"garbage memory={junk} produced a delta"


def test_parse_salvages_the_usable_half_of_a_partly_garbled_memory():
    raw = ('{"events":[' + _EVENT + '],'
           '"memory":{"situations":"not-a-list","scene_notes":"still readable"}}')
    events, delta = Analyzer._parse(raw)
    assert [e["message"] for e in events] == ["m"]
    assert delta["situations"] == []
    assert delta["scene_notes"] == "still readable"


def test_parse_treats_an_empty_memory_object_as_a_real_cycle():
    """An empty "memory" is what the prompt asks for on a quiet cycle. It must merge
    (which ages unconfirmed records) rather than read as "no delta at all"."""
    events, delta = Analyzer._parse('{"events":[],"memory":{}}')
    assert events == []
    assert delta is not None
    assert delta["situations"] == [] and delta["scene_notes"] is None


def test_parse_still_drops_volunteered_descriptions_in_the_object_shape():
    raw = ('{"events":[{"severity":"warning","category":"safety","message":"m",'
           '"tags":[],"description":"unrequested"}],"memory":{}}')
    assert Analyzer._parse(raw, descriptions=False)[0][0]["description"] is None
    assert Analyzer._parse(raw)[0][0]["description"] == "unrequested"


# ── the analyzer's own wiring: config gate, prompt slot, and the merge ────────────


def test_scene_delta_needs_both_switches(tmp_path):
    """monitoring.memory.enabled is the master; scene_delta alone is inert."""
    assert _analyzer(tmp_path)._scene_delta is False
    assert _analyzer(tmp_path, memory={"enabled": True, "scene_delta": True})._scene_delta is True
    assert _analyzer(tmp_path, memory={"enabled": False, "scene_delta": True})._scene_delta is False
    assert _analyzer(tmp_path, memory={"enabled": True, "scene_delta": False})._scene_delta is False
    assert _analyzer(tmp_path, memory={"enabled": True})._scene_delta is False


def test_the_default_analyzer_never_asks_for_a_memory_delta(tmp_path):
    assert '"memory"' not in _analyzer(tmp_path)._system_template


def test_the_delta_request_reaches_the_system_prompt_when_enabled(tmp_path):
    az = _analyzer(tmp_path, memory={"enabled": True, "scene_delta": True})
    assert '"memory"' in az._system_template
    assert '"scene_notes"' in az._system_template


def _capture(az, content):
    calls = {}

    async def create(*a, **k):
        calls["kwargs"] = k
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    az._client.chat.completions.create = create
    return calls


def test_analyze_threads_the_scene_block_into_the_system_prompt(tmp_path):
    az = _analyzer(tmp_path)
    calls = _capture(az, "[]")
    asyncio.run(az.analyze([b"a"], history=[], scene=_scene_block()))
    assert "## SCENE MEMORY" in calls["kwargs"]["messages"][0]["content"]


def test_analyze_without_a_scene_block_leaves_the_prompt_unchanged(tmp_path):
    az = _analyzer(tmp_path)
    calls = _capture(az, "[]")
    asyncio.run(az.analyze([b"a"], history=[]))
    system = calls["kwargs"]["messages"][0]["content"]
    assert "SCENE MEMORY" not in system
    assert "## RECENT OBSERVATION HISTORY" in system


def test_analyze_merges_the_returned_delta_into_the_callers_scene_memory(tmp_path):
    """analyze mutates the caller's object: one Analyzer serves every stream, so the
    delta can never be stashed on self."""
    az = _analyzer(tmp_path, memory={"enabled": True, "scene_delta": True})
    _capture(az, '{"events":[' + _EVENT + '],'
                 '"memory":{"situations":[{"key":"bay2","description":"truck at Bay 2",'
                 '"state":"ongoing"}]}}')
    mem = SceneMemory()
    results = asyncio.run(az.analyze([b"a"], history=[], scene_memory=mem))
    assert [e["message"] for e in results] == ["m"]
    assert "truck at Bay 2" in mem.render(0.0)


def test_analyze_keeps_the_events_when_the_memory_half_is_garbage(tmp_path):
    az = _analyzer(tmp_path, memory={"enabled": True, "scene_delta": True})
    _capture(az, '{"events":[' + _EVENT + '],"memory":"nonsense"}')
    mem = SceneMemory()
    results = asyncio.run(az.analyze([b"a"], history=[], scene_memory=mem))
    assert [e["message"] for e in results] == ["m"]
    assert mem.render(0.0) == ""


def test_analyze_ignores_a_delta_when_no_scene_memory_was_passed(tmp_path):
    """The local backend and every pre-Phase-5 caller pass nothing; a delta arriving
    anyway must not raise."""
    az = _analyzer(tmp_path, memory={"enabled": True, "scene_delta": True})
    _capture(az, '{"events":[' + _EVENT + '],"memory":{"scene_notes":"n"}}')
    assert [e["message"] for e in asyncio.run(az.analyze([b"a"], history=[]))] == ["m"]


def test_analyze_keeps_the_events_when_the_merge_itself_raises(tmp_path):
    """The last line of the same defence: even a SceneMemory that blows up must cost
    one stale cycle, never the cycle's real observations."""
    class _Exploding:
        def merge(self, *a, **k):
            raise RuntimeError("boom")

    az = _analyzer(tmp_path, memory={"enabled": True, "scene_delta": True})
    _capture(az, '{"events":[' + _EVENT + '],"memory":{"scene_notes":"n"}}')
    results = asyncio.run(az.analyze([b"a"], history=[], scene_memory=_Exploding()))
    assert [e["message"] for e in results] == ["m"]


# ── the two filters: what stops the memory outliving the thing it describes ──────


class _RecordingMemory:
    def __init__(self):
        self.kwargs = None

    def merge(self, delta, now, **kw):
        self.kwargs = kw


def _merge_kwargs(tmp_path, **analyze_kw):
    az = _analyzer(tmp_path, memory={"enabled": True, "scene_delta": True})
    _capture(az, '{"events":[' + _EVENT + '],"memory":{"scene_notes":"n"}}')
    mem = _RecordingMemory()
    asyncio.run(az.analyze([b"a"], history=[], scene_memory=mem, **analyze_kw))
    return mem.kwargs


def test_analyze_forwards_the_live_track_and_zone_filters_to_the_merge(tmp_path):
    """SceneMemory.merge takes both filters, but they are dead code unless somebody
    supplies them — and then a note about a track that left renders forever."""
    assert _merge_kwargs(tmp_path, live_track_ids={17}, known_zones={"Bay 2"}) == {
        "live_track_ids": {17}, "known_zones": {"Bay 2"}}


def test_analyze_defaults_both_filters_to_none_rather_than_an_empty_set(tmp_path):
    """None means "skip the filter" and empty set means "drop everything". The local
    backend has no tracker, so defaulting to a set would silently erase its memory."""
    assert _merge_kwargs(tmp_path) == {"live_track_ids": None, "known_zones": None}


def test_analyze_drops_an_entity_note_whose_track_is_gone(tmp_path):
    az = _analyzer(tmp_path, memory={"enabled": True, "scene_delta": True})
    _capture(az, '{"events":[' + _EVENT + '],"memory":{"entities":['
                 '{"id":17,"note":"carrying a long bag"},'
                 '{"id":99,"note":"left the yard ten minutes ago"}]}}')
    mem = SceneMemory()
    asyncio.run(az.analyze([b"a"], history=[], scene_memory=mem, live_track_ids={17}))
    block = mem.render(0.0)
    assert "carrying a long bag" in block
    assert "left the yard ten minutes ago" not in block


def test_analyze_drops_an_area_for_a_zone_that_does_not_exist(tmp_path):
    az = _analyzer(tmp_path, memory={"enabled": True, "scene_delta": True})
    _capture(az, '{"events":[' + _EVENT + '],"memory":{"areas":['
                 '{"zone":"Loading Bay","condition":"unloading in progress"},'
                 '{"zone":"Invented Wing","condition":"quiet"}]}}')
    mem = SceneMemory()
    asyncio.run(az.analyze([b"a"], history=[], scene_memory=mem,
                           known_zones={"Loading Bay"}))
    block = mem.render(0.0)
    assert "unloading in progress" in block
    assert "Invented Wing" not in block


# ── chat: the block rides normal turns and is dropped in search mode ─────────────


def _chat_analyzer(tmp_path, scene_provider=None):
    az = _analyzer(tmp_path, memory={"enabled": True, "scene_delta": True})
    az._tools_enabled = False          # one streaming call, no tool rounds
    az.chat_scene_provider = scene_provider
    return az


def _capture_stream(az, text="ok"):
    """Fake create() recording the streaming call's kwargs; returns the record."""
    calls = {}

    async def create(*a, **k):
        calls["kwargs"] = k

        async def gen():
            for ch in text:
                yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=ch))])
        return gen()

    az._client.chat.completions.create = create
    return calls


def _chat_system_prompt(az, calls, **kw):
    async def go():
        async for _ in az.chat_multi({"cam0": [b"a"]}, "what's happening?",
                                     history=[], analytics=None, prior_turns=[], **kw):
            pass
    asyncio.run(go())
    return calls["kwargs"]["messages"][0]["content"]


def test_a_normal_chat_turn_carries_the_scene_memory_block(tmp_path):
    az = _chat_analyzer(tmp_path, scene_provider=lambda: _scene_block())
    calls = _capture_stream(az)
    system = _chat_system_prompt(az, calls, sweep=False)
    assert "## SCENE MEMORY" in system
    assert "truck at Bay 2 being unloaded" in system


def test_a_search_turn_never_carries_the_scene_memory_block(tmp_path):
    """Required behaviour, not an optimisation: search mode already drops the
    analytics block because its wall-clock stamps were read as footage positions, and
    the memory block's durations are the same second clock."""
    az = _chat_analyzer(tmp_path, scene_provider=lambda: _scene_block())
    calls = _capture_stream(az)
    system = _chat_system_prompt(az, calls, sweep=True)
    assert "SCENE MEMORY" not in system
    assert "truck at Bay 2" not in system


def test_a_chat_turn_with_no_provider_is_byte_identical_to_the_pre_memory_prompt(tmp_path):
    az_off = _chat_analyzer(tmp_path, scene_provider=None)
    off = _capture_stream(az_off)
    az_on = _chat_analyzer(tmp_path, scene_provider=lambda: "")
    on = _capture_stream(az_on)
    assert (_chat_system_prompt(az_off, off, sweep=False)
            == _chat_system_prompt(az_on, on, sweep=False))


def test_a_provider_that_raises_costs_the_block_and_not_the_chat_turn(tmp_path):
    def boom():
        raise RuntimeError("no memory today")

    az = _chat_analyzer(tmp_path, scene_provider=boom)
    calls = _capture_stream(az)
    system = _chat_system_prompt(az, calls, sweep=False)
    assert "SCENE MEMORY" not in system
    assert "## STREAM ANALYTICS" in system
