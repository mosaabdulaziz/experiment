"""tools/engine_golden.py: the normaliser drops what varies run to run, and the diff tolerates shed frames
but not changed values."""
import importlib.util
import json
import pathlib

SPEC = importlib.util.spec_from_file_location(
    "engine_golden", pathlib.Path(__file__).resolve().parents[1] / "tools" / "engine_golden.py")
golden = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(golden)


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _meta(n, dets, **extra):
    return {"key": "iris:meta:cam-1", "fields": {"frame_num": n, "pts": n / 10, "dets": dets, **extra}}


def test_entry_drops_wallclock_and_jpeg_and_rounds_floats():
    e = golden._entry("iris:meta:cam-1", {
        b"frame_num": b"7", b"dets": b'[["person", 0.912345, 3]]', b"jpeg": b"\xff\xd8...", b"ntp_ts": b"123"})
    assert e["fields"] == {"frame_num": 7.0, "dets": [["person", 0.912, 3]]}


def test_identical_runs_pass(tmp_path):
    rows = [_meta(n, [["person", 0.9, 1]]) for n in range(10)]
    a, b = _write(tmp_path / "a", rows), _write(tmp_path / "b", rows)
    assert golden.diff(a, b, 0.98, 3) == 0


def test_a_changed_value_in_a_shared_frame_fails(tmp_path):
    a = _write(tmp_path / "a", [_meta(n, [["person", 0.9, 1]]) for n in range(10)])
    rows = [_meta(n, [["person", 0.9, 1]]) for n in range(10)]
    rows[4] = _meta(4, [["person", 0.9, 1, 1, 2, 3, 4]])          # the 7-field shape leaked in
    assert golden.diff(a, _write(tmp_path / "b", rows), 0.98, 3) == 1


def test_a_few_shed_frames_are_tolerated_but_low_coverage_fails(tmp_path):
    full = [_meta(n, []) for n in range(100)]
    a = _write(tmp_path / "a", full)
    assert golden.diff(a, _write(tmp_path / "b", full[:-1]), 0.98, 3) == 0
    assert golden.diff(a, _write(tmp_path / "c", full[:50]), 0.98, 3) == 1


def test_a_stream_missing_from_one_run_fails(tmp_path):
    a = _write(tmp_path / "a", [_meta(0, [])])
    b = _write(tmp_path / "b", [{"key": "iris:meta:cam-2", "fields": {"frame_num": 0}}])
    assert golden.diff(a, b, 0.98, 3) == 1


def _sem(tmp_path, rows_a, rows_b, **kw):
    a, b = _write(tmp_path / "a", rows_a), _write(tmp_path / "b", rows_b)
    return golden.semantic_diff(a, b, kw.get("tol", 0.03), kw.get("window", 0.5))


def _meta_labels(n, labels):
    return {"key": "iris:meta:cam-1", "fields": {"frame_num": n, "pts": n / 10, "dets": [[l, 0.9, i] for i, l in enumerate(labels)]}}


def _alert(pts, kind="fire_smoke", cls="fire"):
    return {"key": "iris:overcrowd_alert:cam-1", "fields": {"pts": pts, "kind": kind, "class": cls}}


def test_semantic_ignores_value_jitter_but_not_missing_detections(tmp_path):
    a = [_meta_labels(n, ["person", "person"]) for n in range(100)]
    jittered = [{"key": r["key"], "fields": {**r["fields"], "pts": r["fields"]["pts"] + 0.003}} for r in a]
    assert _sem(tmp_path, a, jittered) == 0
    fewer = [_meta_labels(n, ["person"]) for n in range(100)]            # half the person detections gone
    assert _sem(tmp_path, a, fewer) == 1


def test_semantic_alerts_must_match_in_kind_and_stay_within_the_pts_window(tmp_path):
    base = [_meta_labels(n, ["fire"]) for n in range(10)]
    assert _sem(tmp_path, base + [_alert(20.3)], base + [_alert(20.5)]) == 0                    # 0.2 s of jitter
    assert _sem(tmp_path, base + [_alert(20.3)], base + [_alert(22.0)]) == 1                    # moved by 1.7 s
    assert _sem(tmp_path, base + [_alert(20.3)], base + [_alert(20.3, cls="smoke")]) == 1       # a different alert
    assert _sem(tmp_path, base + [_alert(20.3)], base) == 1                                     # the alert is gone


def _snap(key, value):
    return {"key": key, "snapshot": "hash", "value": value}


def test_hash_snapshots_compare_by_field_names_not_blob_digests(tmp_path):
    rows = [_meta_labels(0, ["vehicle"])]
    same = _sem(tmp_path, rows + [_snap("iris:vehicle:crops", ["cam-01:0"])], rows + [_snap("iris:vehicle:crops", ["cam-01:0"])])
    other = _sem(tmp_path, rows + [_snap("iris:vehicle:crops", ["cam-01:0"])], rows + [_snap("iris:vehicle:crops", ["cam-01:7"])])
    assert (same, other) == (0, 1)


def test_semantic_gives_tiny_streams_an_absolute_slack_but_not_unlimited(tmp_path):
    face = lambda n: {"key": "iris:face_rec:cam-1", "fields": {"pts": n, "face_rec": []}}
    assert _sem(tmp_path, [face(1), face(2)], [face(1)]) == 0                       # 2 vs 1 entries
    assert _sem(tmp_path, [face(i) for i in range(6)], [face(1)]) == 1              # 6 vs 1 is not noise


def test_blobs_are_kept_as_their_length_and_random_paths_are_dropped():
    e = golden._entry("iris:appearance:cam-1", {
        b"items": b'[{"crop_path": "bodies/ab12.jpg", "crop_quality": 0.35871, "reid": "' + b"A" * 400 + b'"}]'})
    assert e["fields"]["items"] == [{"crop_quality": 0.359, "reid": "<str:400>"}]


def test_a_snapshot_that_moved_out_with_a_module_is_not_compared(tmp_path):
    base = [_meta(0, [])]
    a = _write(tmp_path / "a", base + [_snap("iris:analytics:zones_applied", 0.0)])
    b = _write(tmp_path / "b", base)
    assert golden.diff(a, b, 0.98, 3) == 0
    c = _write(tmp_path / "c", base + [_snap("iris:other", 1)])
    assert golden.diff(a, c, 0.98, 3) == 1                      # any other missing snapshot still fails


def test_mask_outlines_are_kept_as_their_point_count():
    e = golden._entry("iris:meta:cam-1", {b"person_masks": b'[[7, 0.9, [0.1, 0.2, 0.3, 0.4]], [8, 0.8, null]]'})
    assert e["fields"]["person_masks"] == [[7, 0.9, 4], [8, 0.8, None]]


def test_semantic_snapshots_tolerate_number_jitter_but_not_a_change(tmp_path):
    base = [_meta(n, [["person", 0.9, 1]]) for n in range(10)]

    def scale(b, kind="fit"):
        return {"key": "iris:proximity:scale:cam-1", "snapshot": "string", "value": {"a": 0.254, "b": b, "kind": kind}}
    assert _sem(tmp_path, base + [scale(11.838)], base + [scale(11.851)]) == 0      # a learned value, run-to-run
    assert _sem(tmp_path, base + [scale(11.838)], base + [scale(15.0)]) == 1        # a real change
    assert _sem(tmp_path, base + [scale(11.838)], base + [scale(11.838, "other")]) == 1


def test_an_overlay_entry_s_arrival_time_is_wall_clock_and_not_compared():
    """EngineServices.overlay stamps each overlay entry with its frame's arrival (`at`, wall clock), so it differs
    on every run; an `at` anywhere else is kept."""
    e = golden._entry("iris:overlay:cam-1", {b"pts": b"1.5", b"at": b"1794240000.250", b"shapes": b"[]"})
    assert e["fields"] == {"pts": 1.5, "shapes": []}
    assert "at" in golden._entry("iris:mtmc:cam-1", {b"at": b"12.5"})["fields"]


def test_every_feature_s_zones_acknowledgement_is_not_compared(tmp_path):
    """The PPE, fire and fall zones acknowledge their zones version like the zones module does: a key a product gains
    when one of those modules learns zones, which nothing of that product reads."""
    base = [_meta(0, [])]
    a = _write(tmp_path / "a", base)
    for feature in ("ppe", "fire", "fall"):
        b = _write(tmp_path / "b", base + [_snap(f"iris:analytics:{feature}_zones_applied", 1)])
        assert golden.diff(a, b, 0.98, 3) == 0, feature
    c = _write(tmp_path / "c", base + [_snap("iris:other", 1)])
    assert golden.diff(a, c, 0.98, 3) == 1                      # any other new snapshot still fails
