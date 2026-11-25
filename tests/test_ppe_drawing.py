"""modules.ppe: what the overlay shows (PPE boxes drawn, never published; the person in violation in yellow)."""
import types

from perception.engine.frame import FrameOutput, FrameView, ObjectView, detection_rows
from perception.engine.modules import EngineServices, build_settings
from perception.modules.ppe import BOX_PPE_ITEM, BOX_PPE_VIOLATION, MODULE, PPE_KIND, Settings
from perception.modules.ppe.adapter import UNTRACKED
from perception.modules.zones.colors import BOX_RED

MUX = (1080, 720)
FPS = 20.0


def _module():
    services = EngineServices(
        redis=None, redis_factory=lambda: None, emit=lambda *a, **k: None, request_clip=lambda *a, **k: None,
        stream_ids=["cam-1"], timer=lambda *a: None, telemetry=None, batch_size=1,
        states={"cam-1": types.SimpleNamespace(module_state={})})
    mod = MODULE(build_settings(Settings, {}, "ppe"), services)
    mod.on_built(None)
    return mod


def _objects(head="NO-Hardhat", track_id=12):
    person = ObjectView(track_id, "person", 0.9, 0.9, 100, 200, 100, 300, gie_name="pgie")
    item = ObjectView(UNTRACKED, head, 0.8, 0.8, 125, 185, 50, 75, gie_name="pgie_ppe")
    other = ObjectView(3, "person", 0.9, 0.9, 600, 200, 100, 300, gie_name="pgie")         # nobody's violation
    for o in (person, other):                  # what the engine leaves before transform: relevant persons shown,
        o.draw = o.publish = True              # PPE labels (not in relevant_classes) hidden
    item.draw = item.publish = False
    return [person, item, other]


def _frame(now, objects):
    return FrameView("cam-1", None, 0, int(now * FPS), now, now, mux_size=MUX, _objects=objects)


def _step(mod, now, objects):
    """One published frame through the hooks in engine order."""
    frame = _frame(now, objects)
    out = FrameOutput()
    mod.on_frame(frame, out)
    mod.transform(frame)
    mod.analyze(frame, out)
    return frame, out


def test_ppe_boxes_are_drawn_above_persons_but_never_published():
    mod = _module()
    frame, _ = _step(mod, 0.0, _objects(head="Hardhat"))
    person, item, other = frame.objects
    assert (item.kind, item.draw, item.publish, item.layer) == (PPE_KIND, True, False, 2)
    assert (person.kind, person.draw, person.publish, person.layer) == ("object", True, True, 1)
    assert [row[0] for row in detection_rows(frame.objects, full=False)] == ["person", "person"]


def test_a_shadow_ppe_box_is_not_drawn():
    mod = _module()
    objs = _objects()
    objs[1].conf = -0.1
    frame, _ = _step(mod, 0.0, objs)
    assert (frame.objects[1].draw, frame.objects[1].publish) == (False, False)


def test_ppe_boxes_get_their_colour_and_a_caption_without_the_untracked_id():
    mod = _module()
    frame, out = _step(mod, 0.0, _objects(head="Hardhat"))
    note = out.annotations[frame.objects[1]]
    assert (note.color, note.text) == (BOX_PPE_ITEM, "Hardhat 0.80")


def test_the_person_in_violation_is_yellow_from_the_violation_on_and_others_are_left_alone():
    mod = _module()
    notes = []
    for i in range(8):
        frame, out = _step(mod, i / FPS, _objects())
        notes.append(out.annotations.get(frame.objects[0]))
        assert frame.objects[2] not in out.annotations                    # the other worker
    assert notes[:5] == [None] * 5                                         # before the 6th absent frame
    assert all((n.color, n.text) == (BOX_PPE_VIOLATION, "#12 no helmet") for n in notes[5:])


def test_the_highlight_holds_on_skipped_frames_and_clears_when_the_helmet_is_back():
    mod = _module()
    t = 0.0
    for i in range(6):
        t = i / FPS
        _step(mod, t, _objects())
    frame, out = _step(mod, t + 0.01, _objects())                           # faster than eval_fps: not evaluated
    assert out.annotations[frame.objects[0]].color == BOX_PPE_VIOLATION
    for i in range(1, 8):                                                   # helmet back on
        frame, out = _step(mod, t + i / FPS, _objects(head="Hardhat"))
    assert frame.objects[0] not in out.annotations


def test_yellow_beats_the_zone_colour_whichever_module_runs_first():
    for zones_first in (True, False):
        mod = _module()
        for i in range(6):
            frame, out = _frame(i / FPS, _objects()), FrameOutput()
            mod.on_frame(frame, out)
        mod.transform(frame)
        if zones_first:
            out.annotate(frame.objects[0], color=BOX_RED, priority=1)
        mod.analyze(frame, out)
        if not zones_first:
            out.annotate(frame.objects[0], color=BOX_RED, priority=1)
        assert out.annotations[frame.objects[0]].color == BOX_PPE_VIOLATION
