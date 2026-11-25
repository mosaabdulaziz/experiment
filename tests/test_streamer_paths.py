"""CAM_FILES carries the shared server's absolute locations (/data/IRIS/...). On any other machine the same
clip lives under data/uploads/ (the fire samples under data/uploads/fire/part1/...): the streamer must
find it by name instead of failing with "missing file"."""
import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def load(tmp_path):
    spec = importlib.util.spec_from_file_location("stream_recordings_under_test", REPO / "test_streamer" / "stream_recordings.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.REPO = str(tmp_path)
    mod.UPLOADS_DIR = str(tmp_path / "data" / "uploads")
    return mod


def test_an_existing_absolute_path_is_used_as_is(tmp_path):
    f = tmp_path / "elsewhere" / "clip.mp4"
    f.parent.mkdir(); f.write_bytes(b"x")
    assert load(tmp_path)._resolve_path(str(f)) == str(f)


def test_a_server_path_that_does_not_exist_here_is_found_by_name_under_uploads(tmp_path):
    f = tmp_path / "data" / "uploads" / "fire" / "part1" / "bucketfire.mp4"
    f.parent.mkdir(parents=True); f.write_bytes(b"x")
    mod = load(tmp_path)
    assert mod._resolve_path("/data/IRIS/repo/surveillance-camera-sys/data/uploads/fire/part1/bucketfire.mp4") == str(f)


def test_a_clip_under_mtmc_videos_is_found_too(tmp_path):
    f = tmp_path / "mtmc" / "videos" / "D05-crop.mp4"
    f.parent.mkdir(parents=True); f.write_bytes(b"x")
    assert load(tmp_path)._resolve_path("/data/IRIS/x/D05-crop.mp4") == str(f)


def test_a_clip_that_is_nowhere_keeps_the_original_path_so_the_error_names_it(tmp_path):
    p = "/data/IRIS/repo/nothing/here.mp4"
    assert load(tmp_path)._resolve_path(p) == p


def test_bare_names_still_resolve_under_uploads(tmp_path):
    f = tmp_path / "data" / "uploads" / "camera_4.mp4"
    f.parent.mkdir(parents=True); f.write_bytes(b"x")
    assert load(tmp_path)._resolve_path("camera_4.mp4") == str(f)


def test_a_path_with_a_use_case_folder_resolves_under_uploads(tmp_path):
    f = tmp_path / "data" / "uploads" / "attendance" / "camera_4.mp4"
    f.parent.mkdir(parents=True); f.write_bytes(b"x")
    assert load(tmp_path)._resolve_path("attendance/camera_4.mp4") == str(f)


def test_an_old_bare_name_is_found_inside_a_use_case_folder(tmp_path):
    f = tmp_path / "data" / "uploads" / "attendance" / "camera_4.mp4"
    f.parent.mkdir(parents=True); f.write_bytes(b"x")
    assert load(tmp_path)._resolve_path("camera_4.mp4") == str(f)


def test_a_bare_name_in_both_places_prefers_the_direct_path(tmp_path):
    direct = tmp_path / "data" / "uploads" / "clip.mp4"
    deeper = tmp_path / "data" / "uploads" / "misc" / "clip.mp4"
    deeper.parent.mkdir(parents=True); deeper.write_bytes(b"x"); direct.write_bytes(b"y")
    assert load(tmp_path)._resolve_path("clip.mp4") == str(direct)


def test_a_clip_that_is_versioned_but_not_pulled_says_how_to_get_it(tmp_path):
    clip = tmp_path / "data" / "uploads" / "el-mohandes" / "1-D41-enter.mp4"
    clip.parent.mkdir(parents=True)
    (clip.parent / "1-D41-enter.mp4.dvc").write_text("outs: []\n")
    msg = load(tmp_path)._missing_file_message("cam-1", str(clip))
    assert "versioned with DVC but not pulled" in msg and "dvc pull data/uploads/el-mohandes/1-D41-enter.mp4.dvc" in msg


def test_a_clip_that_is_just_missing_keeps_the_plain_message(tmp_path):
    msg = load(tmp_path)._missing_file_message("cam-1", str(tmp_path / "nowhere.mp4"))
    assert msg.startswith("missing file for cam-1:") and "DVC" not in msg
