from app.chat_search import (
    parse_sweep_marker, SweepMarkerGate, history_chunks, secs_ago,
    clamp_answer_timestamps, offsets_bounds, select_result_streams,
    snapshots_newest_offset, segments_for_found,
)


def test_snapshots_newest_offset_picks_max_across_cameras():
    snaps = {"cam0": [(10.0, b"a"), (30.0, b"b")], "cam1": [(5.0, b"c"), (42.0, b"d")]}
    assert snapshots_newest_offset(snaps) == 42.0


def test_snapshots_newest_offset_empty_is_zero():
    assert snapshots_newest_offset({}) == 0.0
    assert snapshots_newest_offset({"cam0": []}) == 0.0
    assert snapshots_newest_offset(None) == 0.0


def test_select_result_streams_picks_named_stream():
    assert select_result_streams("A red car ran the light on cam1 at 00:04:10.",
                                  ["cam0", "cam1", "cam2"]) == ["cam1"]


def test_select_result_streams_multiple_named_in_order():
    assert select_result_streams("Activity on cam2 and cam0.",
                                  ["cam0", "cam1", "cam2"]) == ["cam0", "cam2"]


def test_select_result_streams_none_named():
    assert select_result_streams("A red car ran the light.", ["cam0", "cam1"]) == []


def test_select_result_streams_whole_word_only():
    # "cam0" must not match inside "cam01"
    assert select_result_streams("Seen on cam01.", ["cam0", "cam01"]) == ["cam01"]


def test_select_result_streams_case_insensitive():
    assert select_result_streams("Found on CAM0.", ["cam0"]) == ["cam0"]


def test_select_result_streams_empty_inputs():
    assert select_result_streams("", ["cam0"]) == []
    assert select_result_streams("cam0", []) == []


def test_offsets_bounds_flattens_all_cameras():
    assert offsets_bounds({"cam0": [10.0, 30.0], "cam1": [5.0, 12.0]}) == (5.0, 30.0)


def test_offsets_bounds_empty_is_none():
    assert offsets_bounds({}) is None
    assert offsets_bounds({"cam0": []}) is None
    assert offsets_bounds(None) is None


def test_clamp_keeps_in_range_timestamp():
    # window 00:04:10–00:04:30 (250–270s); 00:04:20 is inside -> kept.
    out = clamp_answer_timestamps("The car runs the light at 00:04:20.", (250.0, 270.0))
    assert out == "The car runs the light at 00:04:20."


def test_clamp_replaces_out_of_range_wall_clock_with_shown_range():
    # window 00:00:00–00:05:00 (0–300s); 12:45:36 (45936s) is far out -> replaced.
    out = clamp_answer_timestamps("A red car ran the light at 12:45:36.", (0.0, 300.0))
    assert "12:45:36" not in out
    assert "00:00:00–00:05:00" in out


def test_clamp_single_value_window_uses_single_time():
    out = clamp_answer_timestamps("Seen at 09:99 ... 01:00:00 here.", (310.0, 310.0))
    # 01:00:00 = 3600s, out of [310,310] -> replaced with single 00:05:10
    assert "01:00:00" not in out
    assert "00:05:10" in out


def test_clamp_handles_multiple_timestamps_independently():
    out = clamp_answer_timestamps(
        "First at 00:04:15, then a stale 23:59:59 reference.", (250.0, 270.0))
    assert "00:04:15" in out          # in range, kept
    assert "23:59:59" not in out      # out of range, replaced
    assert "00:04:10–00:04:30" in out


def test_clamp_none_bounds_passes_through():
    s = "Something at 12:45:36 happened."
    assert clamp_answer_timestamps(s, None) == s


def test_clamp_respects_pad_tolerance():
    # 00:04:32 (272s) is just past max 270 but within pad=3 -> kept.
    out = clamp_answer_timestamps("at 00:04:32", (250.0, 270.0), pad=3.0)
    assert out == "at 00:04:32"


def test_parse_found_returns_answer_without_marker():
    assert parse_sweep_marker("FOUND\nA red car ran the light.") == (True, "A red car ran the light.")


def test_parse_found_inline_marker():
    assert parse_sweep_marker("FOUND: a red car (cam1)") == (True, "a red car (cam1)")


def test_parse_not_found():
    assert parse_sweep_marker("NOT_FOUND") == (False, "")


def test_parse_malformed_keeps_text_and_not_found():
    assert parse_sweep_marker("maybe?") == (False, "maybe?")


def test_parse_found_strips_confidence_token():
    assert parse_sweep_marker("FOUND 0.92\nA red car ran the light.") == (
        True, "A red car ran the light.")


def test_parse_found_strips_inline_confidence():
    assert parse_sweep_marker("FOUND 0.9: a red car (cam1)") == (True, "a red car (cam1)")


def test_parse_found_rejects_below_confidence_threshold():
    assert parse_sweep_marker("FOUND 0.4\na red car", min_confidence=0.7) == (False, "")


def test_parse_found_accepts_at_confidence_threshold():
    assert parse_sweep_marker("FOUND 0.7\na red car", min_confidence=0.7) == (True, "a red car")


def test_parse_found_percentage_confidence():
    assert parse_sweep_marker("FOUND 90%\na red car", min_confidence=0.7) == (True, "a red car")
    assert parse_sweep_marker("FOUND 40%\na red car", min_confidence=0.7) == (False, "")


def test_parse_found_missing_confidence_passes_threshold():
    # No parsable confidence -> absence can't be thresholded -> accepted.
    assert parse_sweep_marker("FOUND\na red car", min_confidence=0.7) == (True, "a red car")
    assert parse_sweep_marker("FOUND: a red car (cam1)", min_confidence=0.7) == (
        True, "a red car (cam1)")


def test_parse_found_answer_digit_not_misread_as_confidence():
    # "2 cars" must not be parsed as confidence 0.02 and rejected.
    assert parse_sweep_marker("FOUND 2 cars collided at cam1", min_confidence=0.7) == (
        True, "2 cars collided at cam1")


def test_gate_rejects_low_confidence_found_as_not_found():
    g = SweepMarkerGate(min_confidence=0.7)
    out = "".join(g.feed(c) for c in "FOUND 0.4\na red car at cam1")
    out += g.flush()
    assert g.decision == "NOT_FOUND"
    assert out == ""


def test_gate_accepts_high_confidence_found_and_strips_number():
    g = SweepMarkerGate(min_confidence=0.7)
    out = "".join(g.feed(c) for c in "FOUND 0.95\na red car at cam1")
    out += g.flush()
    assert g.decision == "FOUND"
    assert out == "a red car at cam1"


def test_gate_streams_remainder_after_found_decision():
    g = SweepMarkerGate()
    out = "".join(g.feed(c) for c in "FOUND\nhello world")
    assert g.decision == "FOUND"
    assert out + g.flush() == "hello world"


def test_gate_suppresses_everything_on_not_found():
    g = SweepMarkerGate()
    out = "".join(g.feed(c) for c in "NOT_FOUND\nignored")
    assert g.decision == "NOT_FOUND"
    assert out + g.flush() == ""


def test_gate_resolves_found_without_trailing_newline_at_flush():
    g = SweepMarkerGate()
    out = "".join(g.feed(c) for c in "FOUND the car is at cam1")
    out += g.flush()
    assert g.decision == "FOUND"
    assert out == "the car is at cam1"


def test_gate_resolves_not_found_without_trailing_newline_at_flush():
    g = SweepMarkerGate()
    out = "".join(g.feed(c) for c in "NOT_FOUND")
    out += g.flush()
    assert g.decision == "NOT_FOUND"
    assert out == ""


def test_parse_skips_leading_thought_line_before_found():
    assert parse_sweep_marker("thought\nFOUND\nA red car ran the light.") == (True, "A red car ran the light.")


def test_parse_skips_thought_label_line_before_not_found():
    assert parse_sweep_marker("Thought:\nNOT_FOUND") == (False, "")


def test_parse_strips_inline_thought_before_found():
    assert parse_sweep_marker("thought FOUND: a red car (cam1)") == (True, "a red car (cam1)")


def test_parse_does_not_strip_thoughtful_real_answer():
    # "thoughtful" is not the reasoning word "thought" -> left intact, treated as plain.
    assert parse_sweep_marker("thoughtful summary of the scene") == (False, "thoughtful summary of the scene")


def test_gate_suppresses_thought_preamble_and_marker_streams_answer():
    g = SweepMarkerGate()
    out = "".join(g.feed(c) for c in "thought\nFOUND\nhello world")
    out += g.flush()
    assert g.decision == "FOUND"
    assert out == "hello world"
    assert "thought" not in out and "FOUND" not in out


def test_gate_suppresses_thought_preamble_on_not_found():
    g = SweepMarkerGate()
    out = "".join(g.feed(c) for c in "Thought\nNOT_FOUND")
    out += g.flush()
    assert g.decision == "NOT_FOUND"
    assert out == ""


def test_gate_strips_inline_thought_then_found():
    g = SweepMarkerGate()
    out = "".join(g.feed(c) for c in "thought FOUND the car is at cam1\n")
    out += g.flush()
    assert g.decision == "FOUND"
    assert out == "the car is at cam1"


def test_gate_blank_lead_line_before_marker():
    g = SweepMarkerGate()
    out = "".join(g.feed(c) for c in "\n\nFOUND\nthere it is")
    out += g.flush()
    assert g.decision == "FOUND"
    assert out == "there it is"


def test_history_chunks_newest_to_oldest():
    assert history_chunks(100, 30) == [(70, 100), (40, 70), (10, 40), (0, 10)]


def test_history_chunks_resume_from_start():
    assert history_chunks(100, 30, start=40) == [(10, 40), (0, 10)]


def test_history_chunks_empty_when_start_zero():
    assert history_chunks(100, 30, start=0) == []


def test_secs_ago():
    assert secs_ago(70, 100, 2.0) == 15.0   # 30 frames back at 2 fps


import asyncio
from app.chat_search import run_sweep


def _run(agen):
    async def go():
        out = []
        async for it in agen:
            out.append(it)
        return out
    return asyncio.run(go())


def test_run_sweep_found_in_history_stops_and_answers():
    # 100 history frames on cam0; cam frame value encodes index.
    snaps = {"cam0": [(float(i), bytes([i % 256])) for i in range(100)]}
    calls = {"n": 0}

    async def probe(frames, offsets=None):
        calls["n"] += 1
        # "found" when the window includes the oldest sentinel byte 70 (history chunk [70,100))
        flat = frames.get("cam0", [])
        found = any(b == bytes([70]) for b in flat)
        return (found, "here it is") if found else (False, "")

    def live_window(since):
        return {}, {}, since  # no live frames

    items = _run(run_sweep(snapshots=snaps, live_window=live_window, probe=probe,
                           fps=2.0, round_seconds=15, window_frames=4,
                           max_calls=12, max_seconds=99))
    tokens = "".join(it["delta"] for it in items if it["type"] == "token")
    result = [it for it in items if it["type"] == "result"][-1]
    assert tokens == "here it is"
    assert result["found"] is True


def test_run_sweep_exhausts_history():
    snaps = {"cam0": [(float(i), bytes([i % 256])) for i in range(20)]}

    async def probe(frames, offsets=None):
        return False, ""

    def live_window(since):
        return {}, {}, since

    items = _run(run_sweep(snapshots=snaps, live_window=live_window, probe=probe,
                           fps=2.0, round_seconds=5, window_frames=2,
                           max_calls=99, max_seconds=99))
    result = [it for it in items if it["type"] == "result"][-1]
    assert result["found"] is False
    assert result["exhausted"] is True


def test_run_sweep_hits_call_cap():
    snaps = {"cam0": [(float(i), bytes([i % 256])) for i in range(1000)]}

    async def probe(frames, offsets=None):
        return False, ""

    def live_window(since):
        return {"cam0": [b"L"]}, {"cam0": [since + 1]}, since + 1

    items = _run(run_sweep(snapshots=snaps, live_window=live_window, probe=probe,
                           fps=2.0, round_seconds=2, window_frames=2,
                           max_calls=4, max_seconds=99))
    result = [it for it in items if it["type"] == "result"][-1]
    assert result["found"] is False
    assert result["exhausted"] is False  # stopped by the cap, history not done


def test_run_sweep_time_cap_stops_with_exhausted_false():
    snaps = {"cam0": [(float(i), bytes([i % 256])) for i in range(1000)]}

    async def probe(frames, offsets=None):
        return False, ""

    def live_window(since):
        return {}, {}, since  # no live frames; history rounds drive the loop

    # Fake clock: first call is the start time, every later call is far past max_seconds.
    ticks = iter([0.0] + [1000.0] * 50)

    def fake_monotonic():
        return next(ticks)

    items = _run(run_sweep(snapshots=snaps, live_window=live_window, probe=probe,
                           fps=2.0, round_seconds=2, window_frames=2,
                           max_calls=999, max_seconds=30, monotonic=fake_monotonic))
    result = [it for it in items if it["type"] == "result"][-1]
    assert result["found"] is False
    assert result["exhausted"] is False  # stopped by the time cap, not history exhaustion


def test_gate_conversation_streams_answer():
    g = SweepMarkerGate()
    out = "".join(g.feed(c) for c in "CONVERSATION\nHello! I monitor the cameras.")
    out += g.flush()
    assert g.decision == "CONVERSATION"
    assert out == "Hello! I monitor the cameras."


def test_gate_plain_no_marker_is_emitted_not_dropped():
    g = SweepMarkerGate()
    out = "".join(g.feed(c) for c in "A red car is at cam1.")  # no marker, no newline
    out += g.flush()
    assert g.decision == "PLAIN"
    assert out == "A red car is at cam1."


def test_gate_plain_multiline_no_marker_emitted():
    g = SweepMarkerGate()
    out = "".join(g.feed(c) for c in "Two people are talking.\nThey look calm.")
    out += g.flush()
    assert g.decision == "PLAIN"
    assert out == "Two people are talking.\nThey look calm."


def _live_window_factory(snaps, window_frames):
    """A live_window mirroring LocalChatContext/StreamManager.sweep_live_window:
    only frames newer than the cursor, stride-picked to window_frames."""
    from app.chat_select import _stride_pick

    def live_window(since):
        newer = [(o, j) for o, j in snaps["cam0"] if o > since]
        if not newer:
            return {}, {}, since
        picked = _stride_pick(newer, window_frames)
        return ({"cam0": [j for _o, j in picked]},
                {"cam0": [o for o, _j in picked]}, newer[-1][0])

    return live_window


def _max_probe_span(snaps, start_since):
    spans = []

    async def probe(frames, offsets=None):
        offs = (offsets or {}).get("cam0", [])
        if offs:
            spans.append(max(offs) - min(offs))
        return False, ""

    _run(run_sweep(snapshots=snaps, live_window=_live_window_factory(snaps, 4),
                   probe=probe, fps=2.0, round_seconds=5, window_frames=4,
                   max_calls=99, max_seconds=99, start_since=start_since))
    return max(spans) if spans else 0.0


def test_run_sweep_live_cursor_seed_prevents_whole_buffer_dump():
    # 50 history frames at 2 fps spanning offsets 0..24.5s.
    snaps = {"cam0": [(i * 0.5, bytes([i % 256])) for i in range(50)]}
    # Old behaviour: live cursor at 0 -> first live round keeps the WHOLE buffer
    # (stride-picked), spanning ~all 24.5s.
    buggy = _max_probe_span(snaps, 0.0)
    # Fixed: seed the cursor at "now" (newest offset) -> first live round is empty,
    # every round stays within ~round_seconds.
    fixed = _max_probe_span(snaps, snapshots_newest_offset(snaps))
    assert buggy > 15.0
    assert fixed <= 5.5


def test_run_sweep_emits_log_lines():
    snaps = {"cam0": [(float(i), bytes([i % 256])) for i in range(40)]}
    logs = []

    async def probe(frames, offsets=None):
        return (False, "")

    def live_window(since):
        return {"cam0": [b"L", b"L"]}, {"cam0": [since + 1, since + 1]}, since + 1  # 2 fresh frames each round

    _run(run_sweep(snapshots=snaps, live_window=live_window, probe=probe,
                   fps=2.0, round_seconds=10, window_frames=2,
                   max_calls=3, max_seconds=99, log=logs.append))

    joined = "\n".join(logs)
    assert "sweep start" in joined
    assert "live round new=2" in joined
    assert "hist chunk=" in joined
    assert "sweep done" in joined


def test_segments_for_found_pads_by_one_round_seconds():
    calls = []

    class _Store:
        def list_range(self, source_id, lo, hi):
            calls.append((source_id, lo, hi))
            return ["ref"]

    out = segments_for_found(_Store(), "cam0", lo=20.0, hi=28.0, round_seconds=8.0)

    assert calls == [("cam0", 12.0, 36.0)]
    assert out == ["ref"]
