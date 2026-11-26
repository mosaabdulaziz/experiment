"""Tests for tools/render_tracker_ab.py timeline parsing, scaling, and side-by-side MP4 rendering."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.render_tracker_ab import (
    TimelineCursor,
    id_color_bgr,
    load_track_timeline,
    parse_mux_size_arg,
    render_side_by_side,
    scale_box,
)


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def test_parse_mux_size_and_scale_box_across_resolutions():
    assert parse_mux_size_arg("1280x720") == (1280, 720)
    with pytest.raises(ValueError):
        parse_mux_size_arg("1280")

    from tools.render_tracker_ab import BoxObs

    box = BoxObs(oid=3, label="person", conf=0.9, left=640.0, top=360.0, width=160.0, height=180.0)
    x1, y1, x2, y2 = scale_box(box, mux_size=(1280, 720), panel_size=(640, 360))
    assert (x1, y1, x2, y2) == (320, 180, 400, 270)
    assert id_color_bgr(1) != id_color_bgr(2)


def test_load_track_timeline_supports_all_wire_formats(tmp_path: Path):
    # 1. person_boxes + 3-field dets (including shadow conf < 0)
    p1 = _write_jsonl(
        tmp_path / "pboxes.jsonl",
        [
            {
                "key": "iris:meta:cam-a",
                "fields": {
                    "frame_num": 1,
                    "pts": 0.5,
                    "dets": [["person", 0.91, 10], ["person", -0.1, 11]],
                    "person_boxes": [[100.0, 200.0, 60.0, 140.0, 10], [300.0, 220.0, 55.0, 130.0, 11]],
                },
            }
        ],
    )
    tl1 = load_track_timeline(p1)
    assert tl1.cam == "cam-a"
    assert len(tl1.frames) == 1
    assert tl1.frames[0].boxes[0].is_shadow is False
    assert tl1.frames[0].boxes[1].is_shadow is True

    # 2. 7-field dets + global_ids + iris:overlay proximity line
    p2 = _write_jsonl(
        tmp_path / "full.jsonl",
        [
            {
                "key": "iris:meta:cam-b",
                "fields": {
                    "frame_num": 1,
                    "pts": "1.200",
                    "dets": [["car", 0.85, 5, 400.0, 300.0, 180.0, 120.0]],
                    "global_ids": {"5": 42},
                },
            },
            {
                "key": "iris:overlay:cam-b",
                "fields": {
                    "pts": "1.200",
                    "w": 1280,
                    "h": 720,
                    "shapes": [
                        {"type": "line", "p1": [410.0, 410.0], "p2": [490.0, 420.0], "label": "0.35", "style": "latched"}
                    ],
                },
            },
        ],
    )
    tl2 = load_track_timeline(p2)
    assert tl2.mux_size == (1280, 720)
    assert tl2.frames[0].boxes[0].global_id == 42
    assert len(tl2.frames[0].lines) == 1
    assert tl2.frames[0].lines[0].style == "latched"


def test_timeline_cursor_tracks_spawns_after_warmup(tmp_path: Path):
    p = _write_jsonl(
        tmp_path / "cur.jsonl",
        [
            {
                "key": "iris:meta:cam-1",
                "fields": {
                    "frame_num": 1,
                    "pts": 0.2,
                    "dets": [["person", 0.9, 1, 100.0, 100.0, 50.0, 100.0]],
                },
            },
            {
                "key": "iris:meta:cam-1",
                "fields": {
                    "frame_num": 30,
                    "pts": 1.5,
                    "dets": [["person", 0.9, 2, 110.0, 100.0, 50.0, 100.0]],
                },
            },
        ],
    )
    tl = load_track_timeline(p)
    cur = TimelineCursor(tl, hold_seconds=0.4, trail_seconds=1.0)
    assert cur.at(0.25) is not None
    assert cur.seen_ids == {1}
    assert cur.spawns_after_warmup == set()

    assert cur.at(1.55) is not None
    assert cur.seen_ids == {1, 2}
    assert cur.spawns_after_warmup == {2}


def test_render_side_by_side_writes_valid_mp4(tmp_path: Path):
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    video_in = tmp_path / "clip.mp4"
    fps = 10.0
    writer = cv2.VideoWriter(str(video_in), cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 360))
    assert writer.isOpened()
    for f in range(25):
        img = np.full((360, 640, 3), 60, dtype=np.uint8)
        cx = 120 + f * 8
        cv2.rectangle(img, (cx, 120), (cx + 50, 260), (200, 200, 200), -1)
        writer.write(img)
    writer.release()

    before_rows = []
    after_rows = []
    for f in range(25):
        pts = round(f / fps, 3)
        x = 240.0 + f * 16.0  # mux=1280x720 scaled 2x over 640x360 video
        oid_b = 1 if f < 14 else 2
        before_rows.append(
            {
                "key": "iris:meta:cam-1",
                "fields": {
                    "frame_num": f + 1,
                    "pts": pts,
                    "dets": [["person", 0.92, oid_b, x, 240.0, 100.0, 280.0]],
                },
            }
        )
        conf_a = -0.1 if f in (13, 14) else 0.92
        after_rows.append(
            {
                "key": "iris:meta:cam-1",
                "fields": {
                    "frame_num": f + 1,
                    "pts": pts,
                    "dets": [["person", conf_a, 1, x, 240.0, 100.0, 280.0]],
                },
            }
        )

    b_path = _write_jsonl(tmp_path / "before.jsonl", before_rows)
    a_path = _write_jsonl(tmp_path / "after.jsonl", after_rows)
    out_mp4 = tmp_path / "out_ab.mp4"

    res = render_side_by_side(
        video_in,
        b_path,
        a_path,
        out_mp4,
        mux_size_override=(1280, 720),
        panel_width=480,
    )
    assert out_mp4.is_file()
    assert out_mp4.stat().st_size > 1000
    assert res["frames_rendered"] == 25
    assert res["before_unique_ids"] == 2
    assert res["after_unique_ids"] == 1
    assert res["before_spawns_after_1s"] == 1
    assert res["after_spawns_after_1s"] == 0

    cap = cv2.VideoCapture(str(out_mp4))
    assert cap.isOpened()
    assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 480 * 2 + 4
    assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 58 + 270 + 26
    cap.release()
