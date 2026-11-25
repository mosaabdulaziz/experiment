"""Unit tests for perception/engine/gie_gate.py — running a secondary GIE only on
the frames whose metadata is published (face_detect sample_only).

Pure stdlib (no gi), so these run on the host. The gate decides, ahead of the
SGIE, which frames are sampled; the probe downstream must follow that same
decision, or a published frame would carry no faces.
"""

from perception.engine.gie_gate import HIDE_OFFSET, SampleGate, hidden_id, restored_id


def test_first_frame_is_sampled_then_every_interval():
    g = SampleGate(sample_interval=0.1)
    decisions = [g.decide(0, i * 40_000_000, i * 0.04) for i in range(8)]   # 25 fps
    # 0.00 sampled; 0.04/0.08 too soon; 0.12 due; 0.16/0.20 too soon; 0.24 due
    assert decisions == [True, False, False, True, False, False, True, False]


def test_sources_keep_their_own_clock():
    g = SampleGate(sample_interval=0.1)
    assert g.decide(0, 0, 0.0) is True
    assert g.decide(1, 0, 0.0) is True
    assert g.decide(0, 1, 0.05) is False
    assert g.decide(1, 1, 0.05) is False


def test_take_returns_the_recorded_decision_once():
    g = SampleGate(sample_interval=0.1)
    g.decide(0, 7, 0.0)
    g.decide(0, 8, 0.04)
    assert g.take(0, 7) is True
    assert g.take(0, 8) is False
    assert g.take(0, 7) is None          # popped


def test_take_unknown_frame_is_none():
    assert SampleGate(sample_interval=0.1).take(3, 123) is None


def test_decisions_are_bounded_when_never_taken():
    g = SampleGate(sample_interval=0.1, maxlen=4)
    for i in range(10):
        g.decide(0, i, i * 0.04)
    assert g.take(0, 0) is None           # evicted
    assert g.take(0, 9) is not None       # newest kept


def test_hide_only_moves_the_sgies_input_objects():
    assert hidden_id(1, operate_on=1) == 1 + HIDE_OFFSET
    assert hidden_id(3, operate_on=1) == 3        # fire gie untouched
    assert hidden_id(-1, operate_on=1) == -1      # tracker-only objects untouched


def test_restore_undoes_hide_and_nothing_else():
    for uid in (1, 2, 3, -1):
        assert restored_id(hidden_id(uid, operate_on=1)) == uid
    assert restored_id(2) == 2


def test_the_gate_refuses_a_pipeline_it_cannot_follow():
    import types
    from perception.engine.gie_gate import GieGate
    el = types.SimpleNamespace(get_property=lambda n: 2)
    busy = types.SimpleNamespace(pipeline_preview=True, infer_interval=1, sample_interval=0.1)
    throttled = types.SimpleNamespace(pipeline_preview=False, infer_interval=2, sample_interval=0.1)
    assert GieGate.install(el, "sgie", busy, (1, 1), {}) is None
    assert GieGate.install(el, "sgie", throttled, (1, 1), {}) is None
    unsteerable = types.SimpleNamespace(get_property=lambda n: -1)
    ok = types.SimpleNamespace(pipeline_preview=False, infer_interval=1, sample_interval=0.1)
    assert GieGate.install(unsteerable, "sgie", ok, (1, 1), {}) is None
