"""SceneMemory - the interpreted, self-healing half of the scene memory.

Two properties carry the whole phase and each has a dedicated regression test
below. `first_seen` is stamped once and never restamped, because the accumulating
duration is the only thing the model cannot work out for itself. And expiry counts
merges, not seconds, because the analysis cadence varies with load - a seconds-based
TTL would silently mean a different number of missed confirmations at 10s than at 30s.

The third is negative: `validate_delta` must never raise and must isolate its slots,
so a response the model mangled in one place still contributes the parts it got
right - a memory failure must never cost an event (spec 5.8).
"""
import pytest

from app.scene_memory import Area, SceneMemory, Situation, validate_delta


def _sit(key, description="something happening", state="ongoing"):
    return {"key": key, "description": description, "state": state}


def _delta(situations=None, areas=None, entities=None, notes=None):
    d = {}
    if situations is not None:
        d["situations"] = situations
    if areas is not None:
        d["areas"] = areas
    if entities is not None:
        d["entities"] = entities
    if notes is not None:
        d["scene_notes"] = notes
    return d


# --- accumulating duration (design point 2) -------------------------------

def test_a_situation_confirmed_across_three_merges_keeps_its_original_first_seen():
    """The rendered duration grows across confirmations - it never restarts."""
    m = SceneMemory()
    m.merge(_delta([_sit("aisle", "pallets accumulating in the north aisle")]), 0.0)
    m.merge(_delta([_sit("aisle", "pallets accumulating in the north aisle")]), 252.0)
    m.merge(_delta([_sit("aisle", "pallets accumulating in the north aisle")]), 663.0)
    assert "[11m03s]" in m.render(663.0)


def test_re_confirming_a_situation_updates_its_description_without_moving_first_seen():
    m = SceneMemory()
    m.merge(_delta([_sit("bay", "truck arriving at Bay 2")]), 0.0)
    m.merge(_delta([_sit("bay", "truck being unloaded at Bay 2")]), 663.0)
    block = m.render(663.0)
    assert "truck being unloaded at Bay 2" in block
    assert "[11m03s]" in block


def test_a_confirmation_carrying_no_description_keeps_the_one_already_stored():
    m = SceneMemory()
    m.merge(_delta([_sit("bay", "truck being unloaded at Bay 2")]), 0.0)
    m.merge(_delta([{"key": "bay", "state": "ongoing"}]), 60.0)
    assert "truck being unloaded at Bay 2" in m.render(60.0)


def test_a_brand_new_situation_with_no_description_is_not_created():
    m = SceneMemory()
    m.merge(_delta([{"key": "bay", "state": "ongoing"}]), 0.0)
    assert m.render(0.0) == ""


# --- expiry counted in cycles (design point 3) ----------------------------

def test_a_situation_unmentioned_for_ttl_cycles_merges_is_dropped():
    m = SceneMemory(ttl_cycles=3)
    m.merge(_delta([_sit("aisle")]), 0.0)
    for t in (10.0, 20.0, 30.0):
        m.merge({}, t)
    assert m.render(30.0) == ""


def test_a_situation_mentioned_again_before_the_ttl_survives_and_resets_its_counter():
    m = SceneMemory(ttl_cycles=3)
    m.merge(_delta([_sit("aisle")]), 0.0)
    m.merge({}, 10.0)
    m.merge({}, 20.0)
    m.merge(_delta([_sit("aisle")]), 30.0)
    m.merge({}, 40.0)
    m.merge({}, 50.0)
    assert "something happening" in m.render(50.0)
    assert m._situations["aisle"].unconfirmed_cycles == 2


def test_expiry_counts_merges_not_elapsed_seconds():
    """Regression for design point 3: three merges at the same instant still expire a
    situation, and no amount of elapsed time expires one the model keeps confirming."""
    fast = SceneMemory(ttl_cycles=3)
    fast.merge(_delta([_sit("aisle")]), 0.0)
    for _ in range(3):
        fast.merge({}, 0.0)          # zero seconds elapsed, three cycles gone
    assert fast.render(0.0) == ""

    slow = SceneMemory(ttl_cycles=3)
    for t in (0.0, 100_000.0):       # a day apart, still confirmed every cycle
        slow.merge(_delta([_sit("aisle")]), t)
    assert "something happening" in slow.render(100_000.0)


def test_areas_expire_on_the_same_cycle_count_as_situations():
    m = SceneMemory(ttl_cycles=2)
    m.merge(_delta(areas=[{"zone": "Loading Bay", "condition": "unloading in progress"}]), 0.0)
    m.merge({}, 10.0)
    m.merge({}, 20.0)
    assert m.render(20.0) == ""


# --- resolution (design point 4) ------------------------------------------

def test_state_resolved_removes_the_situation_on_that_merge():
    m = SceneMemory()
    m.merge(_delta([_sit("aisle")]), 0.0)
    m.merge(_delta([_sit("aisle", state="resolved")]), 10.0)
    assert m.render(10.0) == ""


def test_state_resolved_for_a_situation_that_was_never_open_creates_nothing():
    m = SceneMemory()
    m.merge(_delta([_sit("aisle", state="resolved")]), 0.0)
    assert m.render(0.0) == ""


# --- bounds (design point 5) ----------------------------------------------

def test_over_max_situations_the_least_recently_confirmed_is_dropped():
    m = SceneMemory(max_situations=3, ttl_cycles=99)
    m.merge(_delta([_sit("a", "situation a"), _sit("b", "situation b")]), 0.0)
    m.merge(_delta([_sit("b", "situation b"), _sit("c", "situation c")]), 10.0)
    m.merge(_delta([_sit("b", "situation b"), _sit("c", "situation c"),
                    _sit("d", "situation d")]), 20.0)
    block = m.render(20.0)
    assert "situation a" not in block          # confirmed longest ago
    assert "situation d" in block              # the newest survives
    assert len(m._situations) == 3


def test_scene_notes_longer_than_the_cap_is_truncated():
    m = SceneMemory(notes_max_chars=20)
    m.merge(_delta(notes="x" * 100), 0.0)
    assert "x" * 20 in m.render(0.0)
    assert "x" * 21 not in m.render(0.0)


def test_scene_notes_are_replaced_wholesale_by_a_later_delta():
    m = SceneMemory()
    m.merge(_delta(notes="wet floor by the bay door"), 0.0)
    m.merge(_delta(notes="floor dry again"), 10.0)
    block = m.render(10.0)
    assert "floor dry again" in block
    assert "wet floor" not in block


def test_a_delta_without_scene_notes_leaves_the_stored_notes_alone():
    m = SceneMemory()
    m.merge(_delta(notes="wet floor by the bay door"), 0.0)
    m.merge({}, 10.0)
    assert "wet floor by the bay door" in m.render(10.0)


# --- entity and zone filtering (design point 6) ---------------------------

def test_an_entity_whose_id_is_not_live_is_dropped_and_a_live_one_survives():
    m = SceneMemory()
    m.merge(_delta(entities=[{"id": 17, "note": "unloading pallets"},
                             {"id": 99, "note": "ghost"}]),
            0.0, live_track_ids={17})
    block = m.render(0.0)
    assert "unloading pallets" in block
    assert "ghost" not in block


def test_an_entity_already_stored_is_dropped_once_its_track_is_gone():
    m = SceneMemory()
    m.merge(_delta(entities=[{"id": 17, "note": "unloading pallets"}]), 0.0,
            live_track_ids={17})
    m.merge({}, 10.0, live_track_ids=set())
    assert m.render(10.0) == ""


def test_entities_are_not_filtered_when_live_track_ids_is_none():
    m = SceneMemory()
    m.merge(_delta(entities=[{"id": 17, "note": "unloading pallets"}]), 0.0)
    assert "unloading pallets" in m.render(0.0)


def test_an_area_whose_zone_is_not_configured_is_dropped():
    m = SceneMemory()
    m.merge(_delta(areas=[{"zone": "Loading Bay", "condition": "unloading in progress"},
                          {"zone": "Atlantis", "condition": "flooded"}]),
            0.0, known_zones={"Loading Bay"})
    block = m.render(0.0)
    assert "Loading Bay" in block
    assert "Atlantis" not in block


def test_areas_are_not_filtered_when_known_zones_is_none():
    m = SceneMemory()
    m.merge(_delta(areas=[{"zone": "Atlantis", "condition": "flooded"}]), 0.0)
    assert "Atlantis" in m.render(0.0)


# --- validate_delta (design point 1) --------------------------------------

@pytest.mark.parametrize("raw", [None, [], "text", 7, 0.5, True, object()])
def test_validate_delta_never_raises_and_returns_an_empty_delta_for_non_dicts(raw):
    assert validate_delta(raw) == {"entities": [], "areas": [], "situations": [],
                                   "scene_notes": None}


def test_validate_delta_drops_a_slot_that_is_not_a_list():
    d = validate_delta({"situations": "not-a-list", "areas": {"zone": "x"}})
    assert d["situations"] == []
    assert d["areas"] == []


def test_validate_delta_drops_entries_missing_their_key_field():
    d = validate_delta({
        "situations": [{"description": "no key"}, _sit("ok", "kept")],
        "areas": [{"condition": "no zone"}, {"zone": "Bay", "condition": "kept"}],
        "entities": [{"note": "no id"}, {"id": 17, "note": "kept"}],
    })
    assert [s["key"] for s in d["situations"]] == ["ok"]
    assert [a["zone"] for a in d["areas"]] == ["Bay"]
    assert [e["id"] for e in d["entities"]] == [17]


def test_validate_delta_drops_non_dict_entries_inside_a_valid_list():
    d = validate_delta({"situations": ["oops", None, _sit("ok", "kept")]})
    assert [s["key"] for s in d["situations"]] == ["ok"]


def test_a_malformed_situations_value_does_not_discard_a_valid_scene_notes():
    """Design point 1: every slot is validated on its own."""
    d = validate_delta({"situations": "not-a-list", "scene_notes": "wet floor"})
    assert d["situations"] == []
    assert d["scene_notes"] == "wet floor"


def test_a_malformed_scene_notes_does_not_discard_valid_situations():
    d = validate_delta({"situations": [_sit("ok", "kept")],
                        "scene_notes": ["not", "a", "string"]})
    assert [s["key"] for s in d["situations"]] == ["ok"]
    assert d["scene_notes"] is None


def test_merging_a_malformed_delta_leaves_the_memory_intact():
    m = SceneMemory(ttl_cycles=99)
    m.merge(_delta([_sit("aisle", "pallets accumulating")]), 0.0)
    m.merge("not a delta at all", 10.0)
    assert "pallets accumulating" in m.render(10.0)


def test_an_entity_id_written_as_a_string_still_matches_a_live_integer_track():
    m = SceneMemory()
    m.merge(_delta(entities=[{"id": "#17", "note": "unloading pallets"}]), 0.0,
            live_track_ids={17})
    assert "unloading pallets" in m.render(0.0)


# --- rendering (design point 7) -------------------------------------------

def test_render_is_empty_when_the_memory_is_empty():
    assert SceneMemory().render(0.0) == ""


def test_render_is_wrapped_in_newlines_so_the_prompt_slot_spaces_correctly():
    m = SceneMemory()
    m.merge(_delta([_sit("aisle")]), 0.0)
    block = m.render(0.0)
    assert block.startswith("\n") and block.endswith("\n")


def test_render_omits_a_section_that_has_nothing_under_it():
    m = SceneMemory()
    m.merge(_delta([_sit("aisle", "pallets accumulating")]), 0.0)
    block = m.render(0.0)
    assert "ONGOING" in block
    assert "AREAS" not in block
    assert "NOTES" not in block
    assert "ENTITIES" not in block


def test_render_orders_the_longest_running_situation_first():
    m = SceneMemory()
    m.merge(_delta([_sit("old", "the older one")]), 0.0)
    m.merge(_delta([_sit("old", "the older one"), _sit("new", "the newer one")]), 100.0)
    rows = [r.strip() for r in m.render(200.0).strip().splitlines()[2:]]
    assert rows[0].endswith("the older one")
    assert rows[1].endswith("the newer one")


def test_render_matches_the_documented_block_format():
    """Golden render, byte for byte, including the load-bearing parenthetical."""
    m = SceneMemory()
    m.merge(_delta([_sit("bay", "truck parked at Bay 2, being unloaded by two workers")],
                   areas=[{"zone": "Loading Bay", "condition": "unloading in progress"}]),
            0.0)
    m.merge(_delta([_sit("bay", "truck parked at Bay 2, being unloaded by two workers"),
                    _sit("aisle", "pallets accumulating in the north aisle")],
                   areas=[{"zone": "Loading Bay", "condition": "unloading in progress"}],
                   notes="Overcast, wet floor by the bay door. ~6 staff on shift."),
            411.0)
    assert m.render(663.0) == (
        "\n## SCENE MEMORY (durations measured by the system, not estimated)\n"
        "  ONGOING\n"
        "    [11m03s] truck parked at Bay 2, being unloaded by two workers\n"
        "    [4m12s]  pallets accumulating in the north aisle\n"
        "  AREAS\n"
        "    Loading Bay: unloading in progress (11m03s)\n"
        "  NOTES\n"
        "    Overcast, wet floor by the bay door. ~6 staff on shift.\n"
    )


def test_render_puts_entity_notes_in_their_own_section():
    m = SceneMemory()
    m.merge(_delta(entities=[{"id": 17, "note": "unloading pallets from truck #23"}]), 0.0)
    assert m.render(0.0) == (
        "\n## SCENE MEMORY (durations measured by the system, not estimated)\n"
        "  ENTITIES\n"
        "    #17: unloading pallets from truck #23\n"
    )


def test_render_collapses_newlines_so_a_multiline_note_cannot_break_the_block():
    m = SceneMemory()
    m.merge(_delta(notes="wet floor\nby the bay door"), 0.0)
    assert m.render(0.0).strip().splitlines()[-1] == "    wet floor by the bay door"


def test_the_dataclasses_carry_both_clocks_and_an_unconfirmed_counter():
    assert Situation("k", "d", 0.0, 0.0).unconfirmed_cycles == 0
    assert Area("z", "c", 0.0, 0.0).unconfirmed_cycles == 0
