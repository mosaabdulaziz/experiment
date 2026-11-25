"""Unit tests for batch_detect_plates (day-sorting and resume) and annotate_viewer (review tracking & discovery)."""

import json
from pathlib import Path
import pytest

pytest.importorskip("onnxruntime", reason="the CI test image leaves out onnxruntime (the test group, pyproject.toml)")

from mtmc.batch_detect_plates import _extract_date, _sort_videos_by_day_and_camera, _extract_camera_id
from mtmc.annotate_viewer import load_reviewed, save_reviewed, discover_folders


def test_extract_date():
    assert _extract_date("ch23_20260726000005.mp4") == "20260726"
    assert _extract_date("D18_20260726000000.mp4") == "20260726"
    assert _extract_date("ch23_20260727000005.mp4") == "20260727"
    assert _extract_date("2026-08-15_cam01.mp4") == "20260815"
    assert _extract_date("undated_video.mp4") == "99999999"


def test_sort_videos_by_day_and_camera():
    v1 = Path("ch23_20260727000005.mp4")
    v2 = Path("D18_20260727000000.mp4")
    v3 = Path("ch23_20260726000005.mp4")
    v4 = Path("D18_20260726000000.mp4")

    videos = [v1, v2, v3, v4]
    sorted_v = _sort_videos_by_day_and_camera(videos)

    # Expected order:
    # 1. 20260726 D18
    # 2. 20260726 ch23
    # 3. 20260727 D18
    # 4. 20260727 ch23
    names = [v.name for v in sorted_v]
    assert names == [
        "D18_20260726000000.mp4",
        "ch23_20260726000005.mp4",
        "D18_20260727000000.mp4",
        "ch23_20260727000005.mp4",
    ]


def test_reviewed_status_persistence(tmp_path):
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()

    # Initial state should be empty
    assert load_reviewed(labels_dir) == set()

    # Save reviewed images
    save_reviewed(labels_dir, {"frame00001.jpg", "frame00002.jpg"})

    # Load back
    rev = load_reviewed(labels_dir)
    assert rev == {"frame00001.jpg", "frame00002.jpg"}


def test_discover_folders_root_and_single(tmp_path):
    # Setup root with 2 video subfolders
    root_dir = tmp_path / "plate_crops"
    v1_dir = root_dir / "vid1"
    (v1_dir / "frames").mkdir(parents=True)
    (v1_dir / "labels").mkdir(parents=True)
    (v1_dir / "frames" / "frame000.jpg").write_text("fake img")

    v2_dir = root_dir / "vid2"
    (v2_dir / "frames").mkdir(parents=True)
    (v2_dir / "labels").mkdir(parents=True)
    (v2_dir / "frames" / "frame000.jpg").write_text("fake img")

    folders = discover_folders(root_arg=str(root_dir))
    assert len(folders) == 2
    assert folders[0][0] == "vid1"
    assert folders[1][0] == "vid2"

    # Single folder mode
    single_folders = discover_folders(frames_arg=str(v1_dir / "frames"))
    assert len(single_folders) == 1
    assert single_folders[0][0] == "vid1"
