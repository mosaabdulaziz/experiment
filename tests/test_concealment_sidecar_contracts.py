"""GPU-free contracts for concealment-only platform opt-ins."""
from pathlib import Path

import pytest
import yaml

from perception.engine.frame import FrameOutput, detection_rows
from perception.modules.pose import Settings
from tests.test_pose_module import _detect, _frame, _person

ROOT = Path(__file__).resolve().parents[1]


def test_optional_defaults_remain_off():
    assert getattr(Settings(), "publish_detection_boxes", None) is False


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_box_rows_and_mux_shape_even_without_pose_observation(enabled, empty):
    frame = _frame([] if empty else [_person(7)])
    mod = _detect(publish_detection_boxes=enabled)
    mod.transform(frame)
    out = FrameOutput()
    mod.analyze(frame, out)
    assert frame.dets_with_boxes is enabled
    rows = detection_rows(frame.objects, frame.dets_with_boxes)
    if not empty:
        assert rows == ([["person", 0.9, 7, 100, 100, 200, 400]] if enabled
                        else [["person", 0.9, 7]])
        assert "pose" not in out.fields
    if enabled:
        assert out.fields["dets_shape"] == [1000, 500]
    else:
        assert "dets_shape" not in out.fields


def test_only_concealment_overlay_enables_opt_ins():
    overlay = yaml.safe_load((ROOT / "products/concealment_detection/sidecar.yaml").read_text(encoding="utf-8"))
    assert overlay["modules"]["pose"].get("publish_detection_boxes") is True
    assert "svr_archive_dir" not in overlay
    assert "svr_archive_dir_app" not in overlay
    for path in (ROOT / "products").glob("*/sidecar.yaml"):
        if path.parent.name != "concealment_detection":
            other = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            assert not (other.get("modules", {}).get("pose") or {}).get("publish_detection_boxes")
            assert not other.get("svr_archive_dir")
