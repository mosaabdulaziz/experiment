"""iris:meta entries carry the version of their shape, so a consumer can pin it."""
import json
import sys
import types
from unittest import mock

try:                                    # the DeepStream image has GStreamer; the CI test image does not
    import gi  # noqa: F401
except ImportError:
    sys.modules["gi"] = mock.MagicMock()
    sys.modules["gi.repository"] = mock.MagicMock()

from perception.engine import publisher
from perception.engine.frame import FrameOutput


class _Redis:
    def __init__(self):
        self.added = []

    def xadd(self, key, fields, maxlen=None, approximate=None):
        self.added.append((key, fields))

    def set(self, *a, **k):
        pass


def _publish(job_extra=None, fields=None):
    r = _Redis()
    gcfg = types.SimpleNamespace(r=r, maxlen=64, pipeline_preview=False, gpu_jpeg=False, jpeg_encoders=1,
                                 viewer_gated_preview=False, jpeg_quality=50)
    pub = publisher._FramePublisher(gcfg)
    try:
        st = types.SimpleNamespace(id="cam-1", meta_key="iris:meta:cam-1", jpeg_key="iris:jpeg:cam-1")
        job = {"st": st, "need_meta": True, "need_stream": False, "frame_bgr": None, "boxes": [], "frame_num": 3,
               "pts": 1.0, "dets": [["person", 0.9, 1]], "fields": fields or {}, "annotations": {},
               "publish_hooks": []}
        job.update(job_extra or {})
        pub._publish(job)
    finally:
        pub.stop()
    return r.added


def test_every_meta_entry_is_stamped_with_the_shape_version():
    ((key, fields),) = _publish()
    assert key == "iris:meta:cam-1" and fields["v"] == publisher.META_VERSION == "1"
    assert json.loads(fields["dets"]) == [["person", 0.9, 1]]


def test_a_module_field_is_published_beside_it_json_encoded():
    ((_, fields),) = _publish(fields={"faces": [[1, 2, 3, 4, 0.9, 7]]})
    assert json.loads(fields["faces"]) == [[1, 2, 3, 4, 0.9, 7]] and fields["v"] == "1"


def test_a_publish_hook_gets_the_clean_frame_after_the_entry_is_written():
    seen = []
    _publish({"frame_bgr": "PIXELS", "publish_hooks": [lambda clean: seen.append(clean)]})
    assert seen == ["PIXELS"]
