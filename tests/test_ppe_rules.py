"""modules.ppe: per-stream rules in on_frame (eval_fps gate, per-frame rules, temporal filter by track id)."""
import types

import pytest

from perception.engine.frame import FrameOutput, FrameView, ObjectView
from perception.engine.modules import EngineServices, build_settings
from perception.modules.ppe import MODULE, Settings
from perception.modules.ppe.adapter import UNTRACKED
from perception.modules.ppe.compliance.types import ItemStatus

MUX = (1080, 720)
FPS = 20.0                                   # the default eval_fps: feeding at this rate evaluates every frame


def _module(streams=("cam-1",), **overrides):
    states = {sid: types.SimpleNamespace(module_state={}) for sid in streams}
    services = EngineServices(
        redis=None, redis_factory=lambda: None, emit=lambda *a, **k: None, request_clip=lambda *a, **k: None,
        stream_ids=list(streams), timer=lambda *a: None, telemetry=None, batch_size=1, states=states)
    mod = MODULE(build_settings(Settings, overrides, "ppe"), services)
    mod.on_built(None)
    return mod


def _worker(track_id, helmet="NO-Hardhat", conf=0.8, x=100):
    """A 300 px tall worker in mux pixels with one head box (helmet / bare head) in the head band; no vest box."""
    return [ObjectView(track_id, "person", 0.9, 0.9, x, 200, 100, 300, gie_name="pgie"),
            ObjectView(UNTRACKED, helmet, conf, conf, x + 25, 185, 50, 75, gie_name="pgie_ppe")]


def _run(mod, now, objects, stream="cam-1", frame_num=0):
    frame = FrameView(stream, None, 0, frame_num, now, now, mux_size=MUX, _objects=objects)
    mod.on_frame(frame, FrameOutput())
    return frame.scratch.get("ppe")          # None when the frame was skipped


def _rules(mod, stream="cam-1"):
    return mod.services.state("ppe", stream)["rules"]


def test_frames_faster_than_eval_fps_are_skipped():
    mod = _module()
    seen = [_run(mod, i / 30.0, _worker(1)) is not None for i in range(30)]     # one second at 30 fps
    assert sum(seen) == 20 and _rules(mod).evaluated == 20
    assert seen[:6] == [True, False, True, True, False, True]                    # 2 of every 3 frames


def test_frames_slower_than_eval_fps_are_all_evaluated():
    mod = _module()
    assert all(_run(mod, i / 10.0, _worker(1)) is not None for i in range(10))


def test_a_sustained_bare_head_becomes_a_new_violation_on_the_sixth_evaluated_frame():
    mod = _module()
    new = [_run(mod, i / FPS, _worker(1))[0].new_violations for i in range(8)]
    assert new == [[], [], [], [], [], ["helmet"], [], []]
    (sp,) = _run(mod, 8 / FPS, _worker(1))
    assert sp.result.track_id == 1 and sp.item_status["helmet"] is ItemStatus.ABSENT


def test_an_evaluated_frame_without_persons_gives_no_one_and_a_skipped_frame_gives_nothing():
    mod = _module()
    assert _run(mod, 0.0, []) == []
    assert _run(mod, 0.01, []) is None


def test_streams_are_kept_apart():
    mod = _module(streams=("cam-1", "cam-2"))
    fired = {"cam-1": [], "cam-2": []}
    for i in range(10):
        # the same track id on both cameras: a bare head on cam-1, a helmet on cam-2
        for sid, helmet in (("cam-1", "NO-Hardhat"), ("cam-2", "Hardhat")):
            (sp,) = _run(mod, i / FPS, _worker(1, helmet), stream=sid)
            fired[sid] += [i] if sp.new_violations else []
    assert fired == {"cam-1": [5], "cam-2": []}
    assert sp.item_status["helmet"] is ItemStatus.PRESENT                         # cam-2's last frame
    assert _rules(mod, "cam-1") is not _rules(mod, "cam-2")


def test_without_reid_a_tracker_id_switch_restarts_the_window():
    # With re-ID on (the default), the same switch in the same place is one worker: see test_ppe_reid.py.
    mod = _module(reid={"max_gap_s": 0})
    for i in range(5):                                                           # 5 absent frames as track 1
        assert not _run(mod, i / FPS, _worker(1))[0].new_violations
    fired = []
    for i in range(5, 12):                                                       # the tracker now calls them 2
        (sp,) = _run(mod, i / FPS, _worker(2))
        if i == 5:
            assert (sp.result.track_id, sp.window_stats["helmet"]["window_frames"]) == (2, 1)
        fired += [i] if sp.new_violations else []
    assert fired == [10]                                                          # the 6th frame as track 2


def test_a_stale_track_is_forgotten_and_starts_a_new_window():
    mod = _module(temporal={"forget_after_s": 1.0})
    for i in range(5):
        _run(mod, i / FPS, _worker(1))
    assert 1 in _rules(mod).filter._tracks
    t = 5 / FPS
    while t < 5 / FPS + 1.2:                                                     # 1.2 s with nobody in the frame
        _run(mod, t, [])
        t += 1 / FPS
    assert 1 not in _rules(mod).filter._tracks
    (sp,) = _run(mod, t, _worker(1))                                             # back, same id
    assert sp.window_stats["helmet"]["window_frames"] == 1 and not sp.new_violations


def test_a_track_seen_within_forget_after_s_keeps_its_window():
    mod = _module(temporal={"forget_after_s": 1.0})
    for i in range(5):
        _run(mod, i / FPS, _worker(1))
    _run(mod, 5 / FPS + 0.5, [])                                                 # gone for half a second
    (sp,) = _run(mod, 5 / FPS + 0.55, _worker(1))
    assert sp.window_stats["helmet"]["window_frames"] == 6 and sp.new_violations == ["helmet"]


@pytest.mark.parametrize("required, expected", [(["helmet"], {"helmet"}), (["vest"], {"vest"})])
def test_only_the_required_items_are_judged(required, expected):
    mod = _module(required_items=required)
    (sp,) = _run(mod, 0.0, _worker(1))
    assert set(sp.item_status) == expected
