"""person_boxes and person_masks: the per-person meta fields, now modules."""
import numpy as np

from perception.engine.frame import FrameOutput, FrameView, ObjectView
from perception.engine.modules import EngineServices, load_modules
from perception.engine.nvds import InstanceMask
from perception.modules.person_boxes import PersonBoxes
from perception.modules.person_masks import PersonMasks, mask_outline_of


def _frame(objects, need_meta=True):
    f = FrameView(stream_id="cam-1", state=None, pad_index=0, frame_num=1, pts=0.0, now=0.0, need_meta=need_meta,
                  mux_size=(1000, 500))
    f.objects = objects
    return f


def _run(module, frame):
    out = FrameOutput()
    module.transform(frame)
    module.analyze(frame, out)
    return out


def _p(oid, label="person", box=(10, 20, 30, 40), conf=0.9, raw=None, **kw):
    return ObjectView(oid, label, conf, conf, *box, raw=raw, **kw)


def test_person_boxes_lists_each_drawn_person_in_order_and_skips_everything_else():
    hidden = _p(4)
    hidden.draw = False
    out = _run(PersonBoxes(None, None), _frame([_p(1), _p(2, "car"), _p(3, box=(1, 2, 3, 4)), hidden,
                                                _p(5, kind="face", layer=0)]))
    assert out.fields["person_boxes"] == [[10, 20, 30, 40, 1], [1, 2, 3, 4, 3]]


def test_person_boxes_is_silent_with_nobody_in_frame_or_on_a_frame_that_feeds_no_meta():
    assert _run(PersonBoxes(None, None), _frame([_p(2, "car")])).fields == {}
    assert _run(PersonBoxes(None, None), _frame([_p(1)], need_meta=False)).fields == {}


def test_a_person_a_later_filter_drops_is_still_reported_when_the_module_is_listed_first():
    frame = _frame([_p(1), _p(2)])
    mod = PersonBoxes(None, None)
    mod.transform(frame)
    frame.objects[1].draw = False                         # what a filter module does later in the same transform pass
    out = FrameOutput()
    mod.analyze(frame, out)
    assert out.fields["person_boxes"] == [[10, 20, 30, 40, 1], [10, 20, 30, 40, 2]]


def _with_mask(width=4, height=4, threshold=0.5, scores=None, oid=1):
    values = np.asarray(scores if scores is not None else [0.0] * 16, np.float32)
    return _p(oid, raw=object(), attrs={"mask": InstanceMask(values, width, height, threshold)})


def test_a_mask_becomes_a_normalised_outline_inside_the_box():
    scores = [0, 0, 0, 0,
              0, 1, 1, 0,
              0, 1, 1, 0,
              0, 0, 0, 0]
    pts = mask_outline_of(_with_mask(scores=scores), 1000, 500)
    assert pts and len(pts) % 2 == 0 and all(0.0 <= v <= 1.0 for v in pts)


def test_an_unreadable_or_missing_mask_gives_none_not_an_error():
    assert mask_outline_of(_p(1), 1000, 500) is None
    assert mask_outline_of(_with_mask(width=3), 1000, 500) is None                   # grid does not fit the values
    assert mask_outline_of(_with_mask(scores=[0.0] * 16), 1000, 500) is None         # nothing above the threshold


def test_person_masks_publishes_one_entry_per_drawn_person_with_a_mask_source():
    synthetic = _p(9)                                       # no pyds object behind it: nothing to read a mask from
    out = _run(PersonMasks(None, None),
               _frame([_with_mask(scores=[1] * 16, oid=1), _p(2, "car", raw=object()), synthetic]))
    (entry,) = out.fields["person_masks"]
    assert entry[0] == 1 and entry[1] == 0.9


def test_person_masks_reads_masks_of_people_only_on_frames_that_feed_meta(monkeypatch):
    import perception.modules.person_masks as pm
    monkeypatch.setattr(pm, "instance_mask", lambda raw: "mask")
    mod = PersonMasks(None, None)
    assert mod.extract(_frame([], need_meta=False)) is None
    read = mod.extract(_frame([]))
    person, car = _p(1, raw=object()), _p(2, "car", raw=object())
    read(person), read(car)
    assert person.attrs == {"mask": "mask"} and car.attrs == {}


def test_both_load_by_name_and_take_no_settings():
    services = EngineServices(redis=None, redis_factory=lambda: None, emit=lambda *a, **k: True,
                              request_clip=lambda *a, **k: None, stream_ids=[], timer=lambda i, f: None, telemetry=None)
    assert load_modules({"person_boxes": {}, "person_masks": {}}, services).names() == ["person_boxes", "person_masks"]
