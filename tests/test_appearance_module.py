"""appearance: the hooks that sample persons and hand crops to the worker (the worker itself: test_appearance.py)."""
from perception.engine.frame import FrameView, ObjectView
from perception.engine.modules import EngineServices, load_modules
from perception.modules.appearance import Appearance, Settings


class _Worker:
    def __init__(self, due=True):
        self.due, self.asked, self.submitted = due, [], []

    def wants(self, stream, boxes, now):
        self.asked.append((stream, boxes, now))
        return self.due

    def submit(self, stream, bgr, boxes, now):
        self.submitted.append((stream, bgr, boxes, now))


def _module(worker=None):
    mod = Appearance(Settings(), None)
    mod.worker = worker
    return mod


def _frame(objects, need_meta=True):
    f = FrameView(stream_id="cam-1", state=None, pad_index=0, frame_num=1, pts=0.0, now=5.0, need_meta=need_meta,
                  mux_size=(1000, 500), capture_size=(2000, 1000))
    f.objects = objects
    return f


def _person(oid, **kw):
    return ObjectView(oid, "person", 0.9, 0.9, 100, 100, 50, 200, **kw)


def test_the_drawn_people_are_sampled_in_capture_coordinates():
    worker, frame, mod = _Worker(), _frame([_person(7), ObjectView(8, "car", 0.9, 0.9, 0, 0, 10, 10)]), None
    mod = _module(worker)
    mod.transform(frame)
    assert mod.wants_pixels(frame) is True
    assert worker.asked == [("cam-1", [(200.0, 200.0, 100.0, 400.0, 7)], 5.0)]
    mod.on_pixels(frame, "BGR")
    assert worker.submitted == [("cam-1", "BGR", [(200.0, 200.0, 100.0, 400.0, 7)], 5.0)]


def test_a_frame_with_no_track_due_costs_no_copy_and_submits_nothing():
    worker, mod = _Worker(due=False), None
    mod = _module(worker)
    frame = _frame([_person(7)])
    mod.transform(frame)
    assert mod.wants_pixels(frame) is False
    mod.on_pixels(frame, "BGR")
    assert worker.submitted == []


def test_nothing_happens_without_people_without_meta_or_without_a_worker():
    worker = _Worker()
    mod = _module(worker)
    for frame in (_frame([]), _frame([_person(7)], need_meta=False)):
        mod.transform(frame)
        assert mod.wants_pixels(frame) is False
        mod.on_pixels(frame, "BGR")
    assert worker.asked == [] and worker.submitted == []
    frame = _frame([_person(7)])
    idle = _module(None)
    idle.transform(frame)
    assert idle.wants_pixels(frame) is False
    idle.on_pixels(frame, "BGR")


def test_a_person_a_later_filter_dropped_is_still_sampled_when_listed_first():
    worker = _Worker()
    mod = _module(worker)
    keep, drop = _person(1), _person(2)
    frame = _frame([keep, drop])
    mod.transform(frame)
    drop.draw = False                                       # a later transform (a filter) removes it
    mod.wants_pixels(frame)
    assert [b[4] for b in worker.asked[0][1]] == [1, 2]


def test_it_loads_by_name():
    services = EngineServices(redis=None, redis_factory=lambda: None, emit=lambda *a, **k: True,
                              request_clip=lambda *a, **k: None, stream_ids=[], timer=lambda i, f: None, telemetry=None)
    assert load_modules({"appearance": {"staff_model": "classifier", "crops_dir": "/c"}}, services).names() == [
        "appearance"]
