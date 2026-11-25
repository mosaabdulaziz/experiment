"""Pure-NumPy unit tests for benchmarks/eval_elevator_video.py."""
from __future__ import annotations

import numpy as np

from benchmarks.eval_elevator_video import (
    GalleryIndex,
    _should_save_frame,
    format_timestamp_hms,
    match_face_open_set,
    summarize_report,
)


def test_format_timestamp_hms():
    assert format_timestamp_hms(0.0) == "00:00.0"
    assert format_timestamp_hms(65.4) == "01:05.4"
    assert format_timestamp_hms(3661.2) == "01:01:01.2"


def test_match_face_open_set_enforces_threshold_and_margin():
    v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    gallery = GalleryIndex(names=["alice", "bob"], matrix=np.stack([v1, v2]))

    clear = match_face_open_set(np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32), gallery, 0.28, 0.05)
    assert clear["matched_name"] == "alice"
    assert clear["runner_up_name"] == "bob"

    tie = match_face_open_set(np.array([0.72, 0.70, 0.0, 0.0], dtype=np.float32), gallery, 0.28, 0.05)
    assert tie["matched_name"] is None
    assert tie["best_name"] == "alice"

    low = match_face_open_set(np.array([0.2, 0.0, 0.98, 0.0], dtype=np.float32), gallery, 0.28, 0.05)
    assert low["matched_name"] is None

    empty = match_face_open_set(v1, GalleryIndex(names=[], matrix=np.empty((0, 4), dtype=np.float32)))
    assert empty["matched_name"] is None and empty["similarity"] is None


def test_summarize_report_and_save_modes():
    frames = [
        {
            "frame_idx": 0,
            "timestamp_s": 0.0,
            "timestamp_hms": "00:00.0",
            "person_boxes": [[10, 10, 100, 300]],
            "faces": [
                {
                    "person_idx": 0,
                    "bbox": [30.0, 20.0, 70.0, 65.0],
                    "det_conf": 0.72,
                    "short_side_px": 40.0,
                    "plausible_kpts": True,
                    "yaw": 0.12,
                    "matched_name": "alice",
                    "similarity": 0.41,
                    "margin_gap": 0.22,
                    "runner_up_name": "bob",
                    "runner_up_sim": 0.19,
                    "yolo_iou": 0.85,
                }
            ],
            "yolo_only_faces": [],
        },
        {
            "frame_idx": 15,
            "timestamp_s": 0.5,
            "timestamp_hms": "00:00.5",
            "person_boxes": [[20, 10, 120, 310]],
            "faces": [],
            "yolo_only_faces": [{"person_idx": 0, "bbox": [40.0, 25.0, 70.0, 55.0], "yolo_conf": 0.31, "short_side_px": 30.0}],
        },
    ]

    rep = summarize_report(
        frame_records=frames,
        video_info={"path": "clip.mkv"},
        config_info={"det_conf": 0.25},
        gallery_entries=2,
        raw_faces_total=2,
        yolo_enabled=True,
        yolo_total_faces=2,
        yolo_both_matched=1,
    )
    s = rep["summary"]
    assert s["frames_sampled"] == 2
    assert s["frames_with_people"] == 2
    assert s["frames_with_faces"] == 1
    assert s["frames_with_people_but_zero_faces"] == 1
    assert s["recognized_hits"] == 1
    assert s["per_identity"]["alice"]["hits"] == 1
    assert s["yolo_comparison"]["recall_vs_yolo"] == 0.5
    assert len(rep["misses_and_hits_to_check"]["new_missed_vs_yolo"]) == 1
    assert _should_save_frame(frames[1], "misses") is True
    assert _should_save_frame(frames[0], "misses") is False
    assert _should_save_frame(frames[0], "faces") is True


def test_buffalo_comparison_and_renderer_diffs():
    from benchmarks.render_elevator_comparison import classify_frame_diff, compute_keep_ranges

    frames = [
        {
            "frame_idx": 10,
            "timestamp_s": 1.0,
            "timestamp_hms": "00:01.0",
            "person_boxes": [[10, 10, 100, 300]],
            "faces": [{"bbox": [0, 0, 30, 30], "det_conf": 0.8, "short_side_px": 30.0, "plausible_kpts": True, "matched_name": "emp8", "similarity": 0.31, "yolo_iou": None}],
            "yolo_faces": [],
            "yolo_only_faces": [],
        },
        {
            "frame_idx": 50,
            "timestamp_s": 5.0,
            "timestamp_hms": "00:05.0",
            "person_boxes": [[10, 10, 100, 300]],
            "faces": [{"bbox": [0, 0, 30, 30], "det_conf": 0.8, "short_side_px": 30.0, "plausible_kpts": True, "matched_name": "emp7", "similarity": 0.45, "yolo_iou": 0.8}],
            "yolo_faces": [{"bbox": [0, 0, 30, 30], "det_conf": 0.8, "short_side_px": 30.0, "matched_new": True, "matched_name": "emp7", "similarity": 0.35}],
            "yolo_only_faces": [],
        },
    ]
    rep = summarize_report(
        frame_records=frames,
        video_info={"path": "clip.mkv"},
        config_info={},
        gallery_entries=10,
        raw_faces_total=2,
        yolo_enabled=True,
        yolo_total_faces=1,
        yolo_both_matched=1,
        buffalo_enabled=True,
        old_gallery_entries=10,
    )
    bc = rep["summary"]["buffalo_comparison"]
    assert bc["old_recognized_hits"] == 1
    assert bc["frame_identity_both"] == 1
    assert bc["frame_identity_new_only"] == 1
    assert bc["frame_identity_old_only"] == 0

    lbl0, _ = classify_frame_diff(frames[0])
    lbl1, _ = classify_frame_diff(frames[1])
    assert "NEW RECOGNIZED (emp8)" in lbl0
    assert "BOTH MATCH: emp7" in lbl1
    assert compute_keep_ranges(frames, mode="diffs", fps=10.0, pad_s=0.5) == [(5, 15)]
    assert compute_keep_ranges(frames, mode="events", fps=10.0, pad_s=0.5) == [(5, 15), (45, 55)]
