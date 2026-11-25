"""Detection domain model and the prompt-facing metadata renderer.

Detections gained a track_id and bbox for the cognitive-monitoring work; they used
to be bare (class, conf) tuples unpacked positionally, so the renderer is pinned
here against arity regressions.
"""
import cv2
import numpy as np
from app.cv_enricher import CVBatchContext, Detection, FrameMeta, CVEnricher


def test_detection_defaults_track_id_and_bbox_to_none():
    d = Detection("person", 0.91)
    assert d.cls == "person"
    assert d.conf == 0.91
    assert d.track_id is None
    assert d.bbox is None


def test_detection_carries_track_id_and_bbox():
    d = Detection("truck", 0.8, 17, (10.0, 20.0, 30.0, 40.0))
    assert d.track_id == 17
    assert d.bbox == (10.0, 20.0, 30.0, 40.0)


def test_format_metadata_counts_classes():
    ctx = CVBatchContext(frames=[], frame_metas=[
        FrameMeta(offset=1.0, detections=[
            Detection("person", 0.9, 1),
            Detection("person", 0.8, 2),
            Detection("truck", 0.7, 3),
        ]),
    ])
    out = ctx.format_metadata()
    assert "person×2" in out
    assert "truck" in out


def test_format_metadata_reports_empty_frames():
    ctx = CVBatchContext(frames=[], frame_metas=[FrameMeta(offset=0.0, detections=[])])
    assert "(no detections)" in ctx.format_metadata()


def test_has_detections_is_false_when_all_frames_empty():
    ctx = CVBatchContext(frames=[], frame_metas=[FrameMeta(offset=0.0, detections=[])])
    assert ctx.has_detections() is False


def test_annotate_single_returns_detection_objects():
    """Regression test: annotate_single must return Detection objects, not tuples.

    This caught a bug where cv.annotate_stream: true + detections caused AttributeError
    when log_cycle tried to access d.cls on bare tuples. This test calls annotate_single
    with a faked YOLO model to verify it actually returns Detection instances at runtime.
    """
    # Create fake YOLO model components
    class _FakeBox:
        def __init__(self, cls_idx, conf):
            self.cls = [cls_idx]
            self.conf = [conf]

    class _FakeResults:
        names = {0: "person", 1: "car"}
        def __init__(self, boxes, img):
            self.boxes = boxes
            self._img = img
        def plot(self, **kw):
            return self._img

    class _FakeModel:
        def __init__(self, boxes):
            self._boxes = boxes
        def __call__(self, img, conf=None, verbose=False):
            return [_FakeResults(self._boxes, img)]

    # Create enricher with CV disabled to avoid ultralytics import
    enricher = CVEnricher({"cv": {"enabled": False}, "sampling": {"jpeg_quality": 80}})

    # Manually enable and inject fake model
    enricher._enabled = True
    enricher._model = _FakeModel([_FakeBox(0, 0.95), _FakeBox(1, 0.85)])
    enricher._relevant = None  # Accept all classes

    # Create a minimal valid JPEG
    fake_image = np.zeros((100, 100, 3), dtype=np.uint8)
    ok, jpeg_buf = cv2.imencode(".jpg", fake_image, [cv2.IMWRITE_JPEG_QUALITY, 80])
    fake_jpeg = jpeg_buf.tobytes()

    # Call annotate_single with the faked setup
    jpeg_out, detections = enricher.annotate_single(fake_jpeg)

    # Verify detections are Detection instances, not tuples
    assert len(detections) == 2
    assert all(isinstance(d, Detection) for d in detections)

    # Verify values from the fake model are correctly transferred
    assert detections[0].cls == "person"
    assert detections[0].conf == 0.95
    assert detections[0].track_id is None
    assert detections[0].bbox is None

    assert detections[1].cls == "car"
    assert detections[1].conf == 0.85
    assert detections[1].track_id is None
    assert detections[1].bbox is None


def test_log_cycle_succeeds_with_detection_objects(tmp_path):
    """Verify log_cycle can process Detection objects without AttributeError,
    and that the JSON it writes actually carries the right cls/conf values.

    Tests that log_cycle successfully accepts and serializes Detection instances
    (as opposed to bare tuples which would cause AttributeError on d.cls/d.conf access).
    """
    # cv.enabled: False so __init__ never imports ultralytics or loads a model
    # (the real yolo11n.pt is gitignored and would otherwise be fetched over the
    # network on a clean checkout / in CI).
    enricher = CVEnricher({"cv": {"enabled": False}, "sampling": {}})
    enricher._enabled = True  # simulate an enabled enricher post-construction
    enricher._model = object()  # never called by log_cycle — only `enabled` reads it

    debug_log_path = tmp_path / "cv_debug.json"
    enricher._debug_log_path = str(debug_log_path)

    # Build the exact shape that Scheduler._cycle would create after annotate_single
    ctx = CVBatchContext(frames=[b"fake"], frame_metas=[
        FrameMeta(offset=1.0, detections=[
            Detection("car", 0.85),
            Detection("person", 0.9199),
        ])
    ])

    enricher.log_cycle(ctx)  # should succeed without AttributeError

    import json
    entries = json.loads(debug_log_path.read_text(encoding="utf-8"))
    assert len(entries) == 1
    frames = entries[0]["frames"]
    assert len(frames) == 1
    assert frames[0]["offset"] == "00:00:01"
    dets = frames[0]["detections"]
    assert dets == [
        {"cls": "car", "conf": 0.85},
        {"cls": "person", "conf": 0.92},  # rounded to 3 places
    ]
