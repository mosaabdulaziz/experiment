"""Tiny synthetic calibration regressions; no dataset inference or model downloads."""
from __future__ import annotations

from dataclasses import asdict, replace
import importlib
import inspect
import json
import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from calibration import calibrate, dataset_utils as du
from concealment.types import CameraZones, DetectorSettings
from conftest import make_skeleton_obs, make_frame
from test_evidence_and_clock import _sample_candidate


def _render_module(name):
    try:
        return importlib.import_module(f"calibration.{name}")
    except ImportError as exc:
        pytest.fail(f"Offline helpers must import without cv2/ultralytics: {exc}")


class _VideoBackend:
    """Tiny video/drawing dependency double; exercise real rendering orchestration."""
    FONT_HERSHEY_SIMPLEX = LINE_AA = INTER_AREA = INTER_LINEAR = CAP_PROP_FPS = CAP_PROP_POS_FRAMES = 0

    def __init__(self, count=90):
        self.count = count
        self.written = 0
        self.refs = []
        self.max_live = 0
        self.text = []

    def VideoCapture(self, path):
        remaining = iter(range(self.count))
        def read():
            if next(remaining, None) is None:
                return False, None
            return True, np.zeros((18, 32, 3), dtype=np.uint8)
        return SimpleNamespace(read=read, release=lambda: None, get=lambda prop: 15, set=lambda *a: None)

    def VideoWriter(self, *args):
        def write(image):
            self.written += 1
            self.refs.append(weakref.ref(image))
            self.max_live = max(self.max_live, sum(ref() is not None for ref in self.refs))
        return SimpleNamespace(write=write, release=lambda: None)

    def resize(self, image, size, **kwargs):
        return np.zeros((size[1], size[0], 3), dtype=np.uint8) if size == (640, 360) else image.copy()

    def putText(self, image, text, *args):
        self.text.append(text)

    def getTextSize(self, *args):
        return (100, 12), 0

    def __getattr__(self, name):
        return lambda *a, **kw: None


def test_render_helpers_import_without_optional_inference_dependencies():
    _render_module("render_demo")
    _render_module("render_rgb_demo")
    experiment = _render_module("experiment_inward_motion")
    assert "require_approach_vel" not in inspect.signature(experiment.make_inward_feature_extractor).parameters


def test_custom_render_handles_empty_frames_and_retains_bounded_images(tmp_path, monkeypatch):
    module = _render_module("render_demo")
    backend = _VideoBackend()
    monkeypatch.setattr(module, "cv2", backend)
    monkeypatch.setattr(module, "_build_store_canvas", lambda *a: np.zeros((18, 32, 3), dtype=np.uint8))
    monkeypatch.setattr(module, "_draw_person_and_pose", lambda **kw: None)
    monkeypatch.setattr(module, "_draw_hud_and_timeline", lambda **kw: None)
    pose = _write_pose(tmp_path / "1_custom.json", range(1, 90, 2))
    cfg = SimpleNamespace(detector=DetectorSettings(), camera_zones={})
    module.render_demo(pose, None, "custom", cfg, tmp_path)
    assert backend.written == 90
    assert backend.max_live <= 16, "contact-sheet collection must not retain every frame"


def test_rgb_render_retains_only_bounded_keyframes(tmp_path, monkeypatch):
    module = _render_module("render_rgb_demo")
    backend = _VideoBackend()
    monkeypatch.setattr(module, "cv2", backend)
    monkeypatch.setattr(module, "OUT_DIR", tmp_path)
    for name in ("_draw_zone_overlay", "_draw_track_on_rgb", "_draw_rgb_hud", "_draw_run_banner"):
        monkeypatch.setattr(module, name, lambda *a, **kw: None)
    frames = [make_frame(i / 15, (), frame_num=i) for i in range(90)]
    monkeypatch.setattr(module, "acl", SimpleNamespace(parse_pose_frame=lambda camera, entry: entry))
    result = module.render_rgb_video_demo(tmp_path / "video.mp4", "cam-1", CameraZones("cam-1"),
                                         DetectorSettings(), frames, 15, tmp_path / "out.mp4",
                                         tmp_path / "out.png", preferred_primary_tid=1)
    assert backend.written == result["frames"] == 90
    assert backend.max_live <= 16, "RGB contact sheet retained the whole video"


def test_cache_fingerprints_video_models_settings_and_rejects_legacy(tmp_path):
    module = _render_module("render_rgb_demo")
    fingerprint = getattr(module, "video_cache_fingerprint", None)
    assert callable(fingerprint)
    video, pose, detector = [tmp_path / name for name in ("video.mp4", "pose.pt", "det.pt")]
    for p in (video, pose, detector):
        p.write_bytes(b"original")
    def fp(settings=None):
        return fingerprint(video, "cam-1", pose, detector, settings=settings)
    original = fp()
    assert fp() == original
    assert fp({"pose_conf": .9}) != original
    for path in (video, pose, detector):
        before = fp()
        path.write_bytes(b"changed")
        assert fp() != before
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({"entries": [], "fps": 15, "shape": [1920, 1080]}))
    with pytest.raises(ValueError, match="[Ss]tale|[Ff]ingerprint"):
        module.extract_video_meta_entries(video, "cam-1", SimpleNamespace(ckpt_path=pose),
                                          SimpleNamespace(ckpt_path=detector), cache)


def _hud_settings():
    return DetectorSettings(dwell_min_s=3.0, reach_excursion_torsos=0.5,
                            body_zone_radius_torsos=0.4, max_conceal_elbow_deg=100.0, min_hold_s=0.9)


def test_custom_pose_hud_gates_and_missed_frame_count_come_from_the_run(tmp_path, monkeypatch):
    module = _render_module("render_demo")
    backend = _VideoBackend()
    monkeypatch.setattr(module, "cv2", backend)
    settings = _hud_settings()
    feat = module.extract_track_features(make_skeleton_obs(), [], CameraZones("cam-1"), (18, 32), settings)
    module._draw_hud_and_timeline(
        img=np.zeros((18, 32, 3), dtype=np.uint8), frame_idx=3, total_frames=10, pts=3.0,
        duration_s=10.0, camera="cam-1", clip_id="1_0101", split_name="dev", stage="REACH",
        dwell_s=1.0, hold_s=0.2, feat=feat, q_state="OK", q_conf=0.8, gt_intervals=[],
        candidate_event=None, candidate_record=None, settings=settings, missed_frames=4,
    )
    text = "\n".join(str(t) for t in backend.text)
    assert "min 0.50T" in text and "min 3.00s" in text and "4 unobserved frames" in text
    assert "min 0.32T" not in text and "0 missed frames" not in text


def test_rgb_hud_gates_come_from_the_run(tmp_path, monkeypatch):
    module = _render_module("render_rgb_demo")
    backend = _VideoBackend()
    monkeypatch.setattr(module, "cv2", backend)
    settings = _hud_settings()
    feat = module.extract_track_features(make_skeleton_obs(), [], CameraZones("cam-1"), (18, 32), settings)
    module._draw_rgb_hud(
        img=np.zeros((18, 32, 3), dtype=np.uint8), video_name="v.mp4", camera="cam-1", frame_idx=3,
        total_frames=10, pts=3.0, duration_s=10.0, primary_tid=1, stage="REACH", dwell_s=1.0,
        hold_s=0.2, feat=feat, q_state="OK", q_conf=0.8, candidate_event=None,
        candidate_record=None, settings=settings,
    )
    text = "\n".join(str(t) for t in backend.text)
    assert "0.50T" in text and "3.00s" in text and "0.28T" not in text


def test_experiment_default_never_preloads_holdout(tmp_path, monkeypatch):
    module = _render_module("experiment_inward_motion")
    pairs = tuple(_pair(tmp_path, f"1_stage_{i}") for i in range(4))
    rw = (_pair(tmp_path, "1_realworld"),)
    splits = du.DatasetSplits((), (), (), (), pairs, rw)
    monkeypatch.setattr(module, "discover_retails_splits", lambda root: splits)
    loaded = []
    monkeypatch.setattr(module, "_preload_positive_split", lambda pairs: loaded.extend(pairs) or [])
    monkeypatch.setattr(module, "_preload_negative_split", lambda files: [])
    monkeypatch.setattr(module, "evaluate_settings_on_splits", lambda *args: {
        "recall_all": 0, "recall_single_person": 0, "fa_per_span_hour": 0, "warning_fa_per_span_hour": 0})
    monkeypatch.setattr(module, "eval_rgb_videos", lambda cfg: [])
    monkeypatch.setattr(module, "OUT_DIR", tmp_path)
    monkeypatch.setattr(Path, "write_text", lambda *a, **kw: None)
    original = module.sm_mod.extract_track_features
    module.main()
    assert len(loaded) == 2 and not set(loaded).intersection(rw)
    assert module.sm_mod.extract_track_features is original


def test_inward_experiment_restores_both_extractors_on_failure(tmp_path, monkeypatch):
    module = _render_module("experiment_inward_motion")
    monkeypatch.setattr(module, "discover_retails_splits", lambda root: du.DatasetSplits((), (), (), (), (), ()))
    monkeypatch.setattr(module, "_preload_positive_split", lambda pairs: [])
    monkeypatch.setattr(module, "_preload_negative_split", lambda paths: [])
    sentinel = lambda *args: None
    monkeypatch.setattr(module.sm_mod, "extract_track_features", sentinel)
    original = module.feat_mod.extract_track_features
    def fail(*args):
        raise RuntimeError("evaluation failed")
    monkeypatch.setattr(module, "evaluate_settings_on_splits", fail)
    with pytest.raises(RuntimeError, match="evaluation failed"):
        module.main()
    assert module.feat_mod.extract_track_features is original
    assert module.sm_mod.extract_track_features is sentinel


def _write_pose(path, indices=(30,), people=1, conf=.9):
    obs = make_skeleton_obs(conf=conf)
    flat = [v for x, y, c in obs.kps for v in (y, x, c)]
    path.write_text(json.dumps({str(t): {str(i): flat for i in indices}
                               for t in range(1, people + 1)}), encoding="utf-8")
    return path


def _pair(tmp_path, stem, people=1, positive=True, indices=(30,), conf=.9):
    pose = _write_pose(tmp_path / f"{stem}.json", indices, people, conf)
    mask = np.zeros(max(indices) + 16, dtype=int)
    if positive:
        mask[list(indices)] = 1
    gt = tmp_path / f"{stem}.npy"
    np.save(gt, mask)
    return pose, gt


def test_sparse_loader_keeps_clip_origin(tmp_path):
    pose = _write_pose(tmp_path / "1_0101.json", (30, 60))
    frames, span, active = du.load_sparse_pose_frames(pose)
    assert next(f for f in frames if f.frame_num == 30).pts == 2
    assert frames[0].frame_num == 0 and not frames[0].tracks
    assert span == pytest.approx(61 / 15)
    assert active == pytest.approx(2 / 15)


def test_positive_preload_covers_mask_prefix_and_tail(tmp_path):
    pair = _pair(tmp_path, "1_0101", indices=(30,))
    frames = calibrate._preload_positive_split((pair,))[0][2]
    assert frames[0].frame_num == 0
    assert frames[-1].frame_num == 45  # mask ends at frame 45; no invented extra frames
    assert next(f for f in frames if f.tracks).pts == 2


def test_negative_sample_is_camera_day_stratified_and_order_independent():
    files = tuple(Path(f"{cam}_{day}12_{i}.json") for cam in (1, 2)
                  for day in ("0101", "0102") for i in range(3 if cam == 1 else 20))
    sample = calibrate._stratified_sample(files, 4)
    assert {(du.parse_camera_id(p.stem), du.parse_train_day(p.stem)) for p in sample} == {
        (f"cam-{cam}", day) for cam in (1, 2) for day in ("0101", "0102")}
    assert sample == calibrate._stratified_sample(tuple(reversed(files)), 4)
    assert calibrate._stratified_sample(files, 0) == []


def test_staged_split_is_disjoint_balanced_and_stratified(tmp_path):
    pairs = tuple(_pair(tmp_path, f"{cam}_{people}_{int(pos)}_{i}", people, pos)
                  for cam in (1, 2) for people in (1, 2) for pos in (False, True) for i in range(4))
    split = getattr(du, "partition_staged_pairs", None)
    assert callable(split), "staged development/test partition is missing"
    dev, test = split(pairs)
    assert len(dev) == len(test) == len(pairs) // 2
    assert set(dev).isdisjoint(test) and set(dev) | set(test) == set(pairs)
    for group in (dev, test):
        assert len({p.stem.rsplit("_", 1)[0] for p, _ in group}) == 8
    assert (dev, test) == split(tuple(reversed(pairs)))


def test_positive_false_alarms_include_duplicates_and_normal_clips(monkeypatch):
    candidate = replace(_sample_candidate(), pts=1, stage_timeline=())
    class Monitor:
        reset_counts, sequence_resets = {}, []

        def __init__(self, *args):
            pass
        def update(self, frame):
            return SimpleNamespace(candidates=(candidate, candidate))
    monkeypatch.setattr(calibrate, "ConcealmentMonitor", Monitor)
    clips = [("pos", "cam-1", [make_frame(1)], [(0, 2)], 3, 1),
             ("normal", "cam-1", [make_frame(1)], [], 3, 1)]
    metrics = calibrate.evaluate_settings_on_splits(DetectorSettings(), {}, clips, [])
    assert metrics.get("pos_false_alarms") == 3
    assert metrics.get("pos_normal_clip_false_alarms") == 2
    assert metrics.get("pos_normal_clips") == 1
    assert metrics.get("pos_span_hours") == pytest.approx(6 / 3600)


def test_evaluation_breaks_down_hits_and_false_alarms_by_stage_and_tier(monkeypatch):
    from collections import Counter

    def cand(tier, stage, pts):
        return replace(_sample_candidate(tier), terminal_stage=stage, pts=pts, stage_timeline=())

    per_camera = {
        "cam-a": ([cand("high", "RETURNED", 1), cand("low", "PARTIAL", 50)],
                  Counter({("gap", "REACH"): 2, ("cooldown", "IDLE"): 5}),
                  [(1.0, 7, "gap", "REACH"), (60.0, 7, "gap", "REACH")]),
        "cam-b": ([cand("medium", "HIDDEN", 100)], Counter(), []),
        "cam-c": ([cand("high", "RETURNED", 5), cand("low", "PARTIAL", 9)], Counter(), []),
    }

    class Monitor:
        def __init__(self, camera, *args):
            self.cands, self.reset_counts, self.sequence_resets = per_camera[camera]

        def update(self, frame):
            out, self.cands = self.cands, []
            return SimpleNamespace(candidates=tuple(out))

    monkeypatch.setattr(calibrate, "ConcealmentMonitor", Monitor)
    pos = [("p1", "cam-a", [make_frame(0)], [(0, 2)], 3600, 1),
           ("p2", "cam-b", [make_frame(0)], [(0, 2)], 3600, 1)]
    neg = [("n1", "cam-c", [make_frame(0)], 3600, 1800)]
    m = calibrate.evaluate_settings_on_splits(DetectorSettings(), {}, pos, neg)

    def row(tp, pos_fp, neg_fa):
        return {"tp": tp, "pos_fp": pos_fp, "neg_fa": neg_fa, "neg_fa_per_span_hour": float(neg_fa)}

    assert m["by_terminal_stage"] == {"RETURNED": row(1, 0, 1), "PARTIAL": row(0, 1, 1),
                                      "HIDDEN": row(0, 1, 0)}
    assert m["by_tier"] == {"high": row(1, 0, 1), "low": row(0, 1, 1), "medium": row(0, 1, 0)}
    assert m["by_stage_tier"]["RETURNED/high"] == row(1, 0, 1)
    assert m["recall_single_person"] == m["recall_single_person_warning"] == 0.5
    assert m["objective"] == pytest.approx(0.5 - calibrate.BALANCED_FA_WEIGHT * 2)
    assert m["objective_warning"] == pytest.approx(0.5 - calibrate.BALANCED_FA_WEIGHT * 1)
    assert m["event_hits"] == [["p1", 0]]
    assert m["neg_fa_by_clip"] == {"n1": 2}
    assert m["reset_counts"] == {"gap/REACH": 2, "cooldown/IDLE": 5}
    assert m["sequence_resets_in_gt"] == {"gap/REACH": 1}
    json.dumps(m)


def test_warning_recall_rematches_without_low_tier_candidates(monkeypatch):
    low_first = [replace(_sample_candidate("low"), pts=1, stage_timeline=()),
                 replace(_sample_candidate("medium"), pts=50, stage_timeline=())]

    class Monitor:
        reset_counts, sequence_resets = {}, []

        def __init__(self, *args):
            self.cands = list(low_first)

        def update(self, frame):
            out, self.cands = self.cands, []
            return SimpleNamespace(candidates=tuple(out))

    monkeypatch.setattr(calibrate, "ConcealmentMonitor", Monitor)
    pos = [("p1", "cam-1", [make_frame(0)], [(0, 2)], 3600, 1)]
    m = calibrate.evaluate_settings_on_splits(DetectorSettings(), {}, pos, [])
    assert m["recall_single_person"] == 1.0
    assert m["recall_single_person_warning"] == 0.0


def test_quantiles_exclude_ambiguous_bystanders_and_ineligible_poses(tmp_path):
    pairs = (_pair(tmp_path, "1_single", indices=(30, 45)),
             _pair(tmp_path, "1_multi", people=2, indices=(30, 45)),
             _pair(tmp_path, "1_bad", indices=(30, 45), conf=.01))
    q = du.compute_empirical_quantiles(pairs, (), {}, settings=DetectorSettings(min_track_s=0))
    assert q["positive_staged_gt"]["wrist_excursion_torsos"]["count"] == 2


def test_quantile_history_keeps_full_inward_window_and_samples_tail(tmp_path, monkeypatch):
    pose = _write_pose(tmp_path / "1_010112_1.json", range(750))
    seen = []
    real = du.extract_track_features
    def capture(obs, history, *args, **kwargs):
        seen.append((history[-1][0], history[-1][0] - history[0][0]))
        return real(obs, history, *args, **kwargs)
    monkeypatch.setattr(du, "extract_track_features", capture)
    du.compute_empirical_quantiles((), (pose,), {})
    assert max(t for t, _ in seen) >= 49
    assert max(span for _, span in seen) >= 4
    assert len(seen) <= 600


def test_proposed_zones_are_shelf_limited_unless_explicit(tmp_path, monkeypatch):
    pose = _write_pose(tmp_path / "1_010112_1.json", range(45))
    zones = du.propose_all_camera_zones((pose,), min_dwell_s=.1)
    assert not zones["cam-1"].zone_free
    assert du.propose_all_camera_zones((pose,), zone_free=True)["cam-1"].zone_free


def test_balanced_selection_reproducible_exact_settings():
    select = getattr(calibrate, "select_balanced_run", None)
    assert callable(select), "balanced objective must be reproducible outside the runner"
    rows = [{"params": asdict(replace(DetectorSettings(), min_hold_s=h)),
             "tuning_metrics": {"recall_single_person": .8, "fa_per_span_hour": 1.0}}
            for h in (.2, .4)]
    assert select(rows) == select(list(reversed(rows)))
    assert select(rows)["params"] in [r["params"] for r in rows]


def test_calibration_tunes_only_development_and_freezes_before_holdout(tmp_path, monkeypatch):
    pairs = tuple(_pair(tmp_path, f"1_stage_{i}") for i in range(4))
    realworld = (_pair(tmp_path, "1_realworld"),)
    splits = du.DatasetSplits((), (), (), (), pairs, realworld)
    monkeypatch.setattr(calibrate, "discover_retails_splits", lambda root: splits)
    monkeypatch.setattr(calibrate, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(calibrate, "propose_all_camera_zones", lambda *a, **k: {})
    quantile_pairs, evaluations = [], []
    def quantiles(p, *a, **k):
        quantile_pairs.extend(p)
        return {"positive_staged_gt": {}, "negative_normal_train": {}}
    monkeypatch.setattr(calibrate, "compute_empirical_quantiles", quantiles)
    monkeypatch.setattr(calibrate.itertools, "product", lambda *a: [(.25, 130, .5, .25, 1.5, False)])
    real_eval = calibrate.evaluate_settings_on_splits
    def evaluate(cfg, zones, clips, negatives):
        evaluations.append(tuple(c[0] for c in clips))
        return real_eval(cfg, zones, clips, negatives)
    monkeypatch.setattr(calibrate, "evaluate_settings_on_splits", evaluate)
    # Exercise orchestration without creating report artifacts.
    monkeypatch.setattr(calibrate, "_build_markdown_report", lambda data: "report")
    writes = []
    monkeypatch.setattr(Path, "write_text", lambda self, text, **kw: writes.append(self))
    result = calibrate.run_calibration(tmp_path)
    assert len(quantile_pairs) == 2, "quantiles must never consume staged holdout"
    dev_ids = {p.stem for p, _ in quantile_pairs}
    assert all(set(ids) == dev_ids for ids in evaluations[:2])
    assert all(not dev_ids.intersection(ids) for ids in evaluations[2:])
    assert result.get("provenance", {}).get("fingerprint")
    assert result["stage3_grid_search"].get("objective")
    assert all(p.parent != tmp_path / "reports" for p in writes), "never overwrite legacy artifacts"


def test_report_shows_span_and_active_exposure_rates():
    row = {"name": "P", "tuning_metrics": {
        "tp_events": 1, "total_events": 2, "tp_single_person": 1, "total_single_person": 2,
        "fa_per_span_hour": 1.0, "fa_per_active_hour": 4.0,
        "pos_false_alarms": 3, "pos_normal_clip_false_alarms": 1}}
    report = calibrate._build_markdown_report({
        "data_root": "synthetic", "splits": {},
        "stage1_proposed_zones": {"cameras": {}},
        "stage2_empirical_quantiles": {"positive_staged_gt": {}, "negative_normal_train": {}},
        "stage3_grid_search": {"selected_operating_points": [row]}, "stage4_locked_holdout": [],
    })
    assert "active-hour" in report and "4.000" in report


def test_report_marks_unverified_legacy_claims_and_has_no_fixed_denominators():
    report = calibrate._build_markdown_report({
        "data_root": "synthetic", "splits": {},
        "stage1_proposed_zones": {"cameras": {}},
        "stage2_empirical_quantiles": {"positive_staged_gt": {}, "negative_normal_train": {}},
        "stage3_grid_search": {"selected_operating_points": []}, "stage4_locked_holdout": [],
    })
    assert "unverified" in report.lower()
    assert "All 622" not in report and "All 43" not in report


def test_provenance_changes_with_settings_zones_sources_and_dataset(tmp_path):
    build = getattr(du, "build_provenance", None)
    assert callable(build), "reports need content-addressed provenance"
    pose = _write_pose(tmp_path / "1_0101.json")
    source = tmp_path / "source.py"
    source.write_text("version = 1", encoding="utf-8")
    cfg = DetectorSettings()
    zones = {"cam-1": CameraZones("cam-1", zone_free=True)}
    def run(settings=cfg, camera_zones=zones):
        return build(settings, camera_zones, {"dev": (pose,)}, source_paths=(source,))
    original = run()
    assert original == run()
    assert original != run(replace(cfg, min_hold_s=99))
    assert original != run(camera_zones={})
    source.write_text("version = 2", encoding="utf-8")
    assert original != run()
    changed_source = run()
    _write_pose(pose, (31,))
    assert changed_source != run()


def _fake_fsm_experiment(module, tmp_path, monkeypatch, metrics=None):
    tune = tuple(Path(f"1_0101{i:02d}_{i}.json") for i in range(3))
    holdout = (Path("1_0901_0.json"),)
    staged = tuple(_pair(tmp_path, f"1_1_{int(pos)}_{i}", positive=pos)
                   for pos in (False, True) for i in range(4))
    rw = (_pair(tmp_path, "1_realworld"),)
    splits = du.DatasetSplits(("0101",), ("0901",), tune, holdout, staged, rw)
    seen = {"pos": [], "neg": [], "zones": [], "gap": set()}

    def pos(pairs, gap_flush_s):
        seen["pos"].extend(pairs)
        seen["gap"].add(gap_flush_s)
        return []

    def neg(files, gap_flush_s):
        seen["neg"].extend(files)
        seen["gap"].add(gap_flush_s)
        return []

    monkeypatch.setattr(module, "discover_retails_splits", lambda root: splits)
    monkeypatch.setattr(module, "_preload_positive_split", pos)
    monkeypatch.setattr(module, "_preload_negative_split", neg)
    monkeypatch.setattr(module, "propose_all_camera_zones",
                        lambda files, **kw: seen["zones"].extend(files) or {})
    monkeypatch.setattr(module, "evaluate_settings_on_splits", lambda *a: dict(metrics or {
        "objective": 0.1, "objective_warning": 0.1, "recall_single_person": 0.5,
        "fa_per_span_hour": 1.0, "warning_fa_per_span_hour": 1.0, "event_hits": []}))
    staged_test = set(du.partition_staged_pairs(staged)[1])
    return seen, staged_test, rw, holdout


def test_fsm_experiment_loads_development_data_only(tmp_path, monkeypatch):
    module = _render_module("experiment_fsm_upgrade")
    seen, staged_test, rw, holdout = _fake_fsm_experiment(module, tmp_path, monkeypatch)
    module.run(["baseline"], out_dir=tmp_path)
    assert seen["pos"] and not set(seen["pos"]) & (staged_test | set(rw))
    assert seen["neg"] and not set(seen["neg"]) & set(holdout)
    assert seen["zones"] and not set(seen["zones"]) & set(holdout)
    assert (tmp_path / "fsm_upgrade_ablation.json").exists()


def test_fsm_experiment_preload_covers_longest_variant_track_loss(tmp_path, monkeypatch):
    module = _render_module("experiment_fsm_upgrade")
    seen, *_ = _fake_fsm_experiment(module, tmp_path, monkeypatch)
    monkeypatch.setitem(module.VARIANTS, "long_loss", {"lost_s": 2.5, "reid_max_gap_s": 2.5})
    module.run(["baseline", "long_loss"], out_dir=tmp_path)
    assert len(seen["gap"]) == 1 and next(iter(seen["gap"])) == pytest.approx(2.6)
    assert module.preload_gap_flush_s([DetectorSettings()]) == pytest.approx(1.6)


def test_fsm_experiment_rejects_unknown_knobs():
    module = _render_module("experiment_fsm_upgrade")
    with pytest.raises(ValueError, match="no_such_knob"):
        module.variant_settings(DetectorSettings(), {"no_such_knob": 1})


def test_fsm_experiment_raises_when_baseline_drifts(tmp_path, monkeypatch):
    module = _render_module("experiment_fsm_upgrade")
    _fake_fsm_experiment(module, tmp_path, monkeypatch)
    module.run(["baseline"], out_dir=tmp_path)
    module.run(["baseline"], out_dir=tmp_path)
    drifted = {"objective": 0.2, "objective_warning": 0.1, "recall_single_person": 0.5,
               "fa_per_span_hour": 1.0, "warning_fa_per_span_hour": 1.0, "event_hits": []}
    monkeypatch.setattr(module, "evaluate_settings_on_splits", lambda *a: drifted)
    with pytest.raises(module.BaselineMismatch):
        module.run(["baseline"], out_dir=tmp_path)


def test_fsm_baseline_rejects_new_settings_identity_with_metric_drift(tmp_path):
    module = _render_module("experiment_fsm_upgrade")
    path = tmp_path / "baseline.json"
    identity = {"settings_sha256": "legacy", "zones_sha256": "zones", "gap_flush_s": 1.6}
    module.check_baseline(identity, {"objective": 0.1}, path)
    original = path.read_bytes()
    with pytest.raises(module.BaselineMismatch):
        module.check_baseline({**identity, "settings_sha256": "added-default-off-knobs"},
                              {"objective": -99.0}, path)
    assert path.read_bytes() == original


def test_fsm_baseline_new_settings_identity_must_reproduce_pinned_metrics(tmp_path):
    module = _render_module("experiment_fsm_upgrade")
    path = tmp_path / "baseline.json"
    identity = {"settings_sha256": "legacy", "legacy_settings_sha256": "legacy-profile",
                "zones_sha256": "zones", "gap_flush_s": 1.6}
    metrics = {"objective": 0.1}
    module.check_baseline(identity, metrics, path)
    original = path.read_bytes()
    module.check_baseline({**identity, "settings_sha256": "added-default-off-knobs"}, metrics, path)
    assert path.read_bytes() == original


def test_fsm_legacy_settings_hash_accepts_only_default_upgrade_knobs():
    module = _render_module("experiment_fsm_upgrade")
    base = DetectorSettings()
    assert module.legacy_settings_hash(base) == module.legacy_settings_hash(replace(base))
    for changed in (
        replace(base, sequence_max_gap_s=0.8),
        replace(base, hand_transfer=True),
        replace(base, spatial_hidden_s=1.0),
        replace(base, dwell_min_s=base.dwell_min_s + 1.0),
    ):
        assert module.legacy_settings_hash(base) != module.legacy_settings_hash(changed)


def test_fsm_baseline_rejects_legacy_settings_drift_even_when_metrics_match(tmp_path):
    module = _render_module("experiment_fsm_upgrade")
    path = tmp_path / "baseline.json"
    identity = {"settings_sha256": "full-a", "legacy_settings_sha256": "legacy-a",
                "zones_sha256": "zones", "gap_flush_s": 1.6}
    metrics = {"objective": 0.1}
    module.check_baseline(identity, metrics, path)
    with pytest.raises(module.BaselineMismatch, match="legacy detector settings"):
        module.check_baseline({**identity, "settings_sha256": "full-b",
                               "legacy_settings_sha256": "legacy-b"}, metrics, path)


def test_fsm_baseline_accepts_historical_hash_as_normalized_legacy_hash(tmp_path):
    module = _render_module("experiment_fsm_upgrade")
    path = tmp_path / "baseline.json"
    historical = {"settings_sha256": "legacy-profile",
                  "zones_sha256": "zones", "gap_flush_s": 1.6}
    metrics = {"objective": 0.1}
    module.check_baseline(historical, metrics, path)
    original = path.read_bytes()
    current = {**historical, "settings_sha256": "full-with-default-upgrade-fields",
               "legacy_settings_sha256": "legacy-profile"}
    module.check_baseline(current, metrics, path)
    assert path.read_bytes() == original


def test_fsm_baseline_requires_new_reference_for_changed_preload(tmp_path):
    module = _render_module("experiment_fsm_upgrade")
    path = tmp_path / "baseline.json"
    identity = {"settings_sha256": "legacy", "zones_sha256": "zones", "gap_flush_s": 1.6}
    module.check_baseline(identity, {"objective": 0.1}, path)
    with pytest.raises(module.BaselineMismatch):
        module.check_baseline({**identity, "gap_flush_s": 2.6}, {"objective": -99.0}, path)


def test_fsm_experiment_new_output_uses_pinned_baseline_reference(tmp_path, monkeypatch):
    module = _render_module("experiment_fsm_upgrade")
    _fake_fsm_experiment(module, tmp_path, monkeypatch)
    module.run(["baseline"], out_dir=tmp_path)
    reference = tmp_path / "fsm_upgrade_baseline.json"
    original = reference.read_bytes()
    module.run(["baseline"], out_dir=tmp_path / "verified", baseline_reference=reference)
    assert reference.read_bytes() == original
    drifted = {"objective": 0.2, "objective_warning": 0.1, "recall_single_person": 0.5,
               "fa_per_span_hour": 1.0, "warning_fa_per_span_hour": 1.0, "event_hits": []}
    monkeypatch.setattr(module, "evaluate_settings_on_splits", lambda *a: drifted)
    with pytest.raises(module.BaselineMismatch):
        module.run(["baseline"], out_dir=tmp_path / "drifted", baseline_reference=reference)


def test_fsm_experiment_reports_single_person_event_gains_and_losses():
    module = _render_module("experiment_fsm_upgrade")
    base = {"objective": 0.5, "objective_warning": 0.4, "recall_single_person": 0.5,
            "fa_per_span_hour": 10.0, "warning_fa_per_span_hour": 4.0,
            "event_hits": [["a", 0], ["b", 0], ["multi", 0]]}
    var = {"objective": 0.6, "objective_warning": 0.4, "recall_single_person": 0.6,
           "fa_per_span_hour": 11.0, "warning_fa_per_span_hour": 5.0,
           "event_hits": [["a", 0], ["c", 1], ["multi", 1]]}
    delta = module.compare(base, var, single_person_stems={"a", "b", "c"})
    assert delta["events_gained"] == [["c", 1]] and delta["events_lost"] == [["b", 0]]
    assert delta["net_events"] == 0
    assert delta["delta_objective"] == pytest.approx(0.1)
    assert delta["warning_fa_change_frac"] == pytest.approx(0.25)


def _hits(n, extra=()):
    return {"event_hits": [[f"p{i}", 0] for i in range(n)] + list(extra),
            "neg_fa_by_clip": {f"n{i}": 1 for i in range(10)}}


def test_fsm_bootstrap_is_confident_only_for_broad_single_person_gains():
    module = _render_module("experiment_fsm_upgrade")
    single_gt = {f"p{i}": 1 for i in range(20)}
    span_h = {f"n{i}": 1.0 for i in range(10)}
    p = lambda base, var: module.bootstrap_p_improvement(base, var, single_gt, span_h)
    assert p(_hits(5), _hits(15)) >= 0.99
    assert 0.5 < p(_hits(5), _hits(6)) < 0.9
    assert p(_hits(5), _hits(5, extra=[["multi", 0]])) == 0.0
    costly = _hits(15)
    costly["neg_fa_by_clip"] = {f"n{i}": 200 for i in range(10)}
    assert p(_hits(5), costly) == 0.0


def test_fsm_split_half_fa_change_alternates_within_camera_day():
    module = _render_module("experiment_fsm_upgrade")
    stems = ["1_010101_0", "1_010101_1", "2_010101_0", "2_010101_1"]
    span_h = dict.fromkeys(stems, 1.0)
    base = {"neg_fa_by_clip": dict.fromkeys(stems, 2)}
    var = {"neg_fa_by_clip": {"1_010101_0": 2, "1_010101_1": 3,
                              "2_010101_0": 2, "2_010101_1": 3}}
    assert module.split_half_fa_change(base, var, span_h) == [0.0, 0.5]
    quiet = {"neg_fa_by_clip": dict.fromkeys(stems, 0)}
    assert module.split_half_fa_change(quiet, quiet, span_h) == [0.0, 0.0]
    assert module.split_half_fa_change(quiet, base, span_h) == [float("inf")] * 2


def test_fsm_adoption_requires_every_gate():
    module = _render_module("experiment_fsm_upgrade")
    ok = {"delta_objective": 0.010, "net_events": 2, "warning_fa_change_frac": 0.03,
          "bootstrap_p_improvement": 0.9, "split_half_fa_change": [0.0, 0.05]}
    assert module.adoption_verdict(ok)["adopt"]
    for key, bad in [("delta_objective", 0.0099), ("net_events", 1),
                     ("warning_fa_change_frac", 0.031), ("warning_fa_change_frac", None),
                     ("bootstrap_p_improvement", 0.89), ("split_half_fa_change", [0.0, 0.051])]:
        verdict = module.adoption_verdict({**ok, key: bad})
        assert not verdict["adopt"] and not all(verdict["checks"].values())


def test_fsm_experiment_ablates_zone_exit_grace_at_two_values():
    module = _render_module("experiment_fsm_upgrade")
    assert module.VARIANTS["zone_exit_0.2"] == {"zone_exit_grace_s": 0.2}
    assert module.VARIANTS["zone_exit_0.4"] == {"zone_exit_grace_s": 0.4}


def test_fsm_experiment_report_carries_adoption_verdict(tmp_path, monkeypatch):
    module = _render_module("experiment_fsm_upgrade")
    _fake_fsm_experiment(module, tmp_path, monkeypatch)
    monkeypatch.setitem(module.VARIANTS, "probe", {"hold_flicker_grace_s": 0.1})
    report = module.run(["probe"], out_dir=tmp_path)
    probe = report["variants"]["probe"]
    assert probe["vs_baseline"]["bootstrap_p_improvement"] == 0.0
    assert probe["vs_baseline"]["split_half_fa_change"] == [0.0, 0.0]
    assert probe["adoption"]["adopt"] is False


def test_fsm_finalizer_validates_authoritative_report_bundle():
    module = _render_module("finalize_fsm_upgrade")
    evidence = module.load_frozen_evidence()
    assert evidence["decision"] == "No upgrade knob passes all development adoption gates"
    assert len(evidence["ablation_reports"]) == 5


def test_fsm_finalizer_rejects_incomplete_or_tampered_evidence():
    module = _render_module("finalize_fsm_upgrade")
    reports = module.load_report_bundle()
    settings_hash, legacy_hash = module.deployed_settings_hashes()

    missing = json.loads(json.dumps(reports))
    del missing[module.QUALITY_REPORT]["variants"]["hand_transfer"]
    with pytest.raises(RuntimeError, match="variant contract"):
        module.validate_report_bundle(missing, settings_hash, legacy_hash)

    stale = json.loads(json.dumps(reports))
    stale[module.QUALITY_REPORT]["identity"]["zones_sha256"] = "stale"
    with pytest.raises(RuntimeError, match="data identity"):
        module.validate_report_bundle(stale, settings_hash, legacy_hash)

    tampered = json.loads(json.dumps(reports))
    tampered[module.QUALITY_REPORT]["variants"]["quality_grace_0.3"]["adoption"]["adopt"] = True
    with pytest.raises(RuntimeError, match="adoption verdict"):
        module.validate_report_bundle(tampered, settings_hash, legacy_hash)


def test_fsm_finalizer_claims_stable_run_once(tmp_path):
    module = _render_module("finalize_fsm_upgrade")
    run_dir = tmp_path / "fsm-final-heldout"
    module.claim_run(run_dir, {"decision": "frozen"})
    assert json.loads((run_dir / "started.json").read_text(encoding="utf-8")) == {
        "decision": "frozen"
    }
    with pytest.raises(RuntimeError, match="already claimed"):
        module.claim_run(run_dir, {"decision": "frozen"})
    assert "assert " not in inspect.getsource(module.validate_report_bundle)
