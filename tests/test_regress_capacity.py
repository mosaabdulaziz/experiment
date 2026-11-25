"""The capacity measurements (tools/regress/capacity.py): planning, overlay configs, summarising, engine protection. No docker, no GPU."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

from regress import capacity  # noqa: E402

REPO = os.path.join(os.path.dirname(__file__), "..")
BASE = '''# a golden config
extends: ../../products/attendance/sidecar.yaml

file_loop: false

sources:
  - {id: cam-1, uri: "file:///workspace/iris/data/uploads/attendance/enter-3.mp4", roles: [attendance], label: "golden enter-3"}
'''


def test_read_and_replicate_sources():
    assert capacity.read_sources(BASE)[0][0] == "cam-1"
    lines = capacity.replicate_sources(BASE, 3)
    assert len(lines) == 3
    assert "{id: cam-1," in lines[0] and "{id: cam-1-2," in lines[1] and "{id: cam-1-3," in lines[2]
    assert all('roles: [attendance]' in l and 'enter-3.mp4' in l for l in lines)


def test_a_base_without_one_line_sources_is_refused():
    with pytest.raises(ValueError):
        capacity.replicate_sources("extends: x.yaml\n", 2)


def test_overlay_paced_and_unpaced():
    paced = capacity.overlay_text("attendance", BASE, 1, True)
    free = capacity.overlay_text("attendance", BASE, 1, False)
    assert "extends: golden-attendance.yaml" in paced and "sink_sync" not in paced and "mux_sync_inputs" not in paced
    assert "sink_sync: false" in free and "mux_sync_inputs: 0" in free
    assert "sources:" not in paced                                            # one camera: the base's own source
    many = capacity.overlay_text("attendance", BASE, 4, True, {"pgie_config": "/workspace/iris/x.txt"})
    assert many.count("- {id:") == 4 and 'pgie_config: "/workspace/iris/x.txt"' in many


def test_run_names_are_distinct_and_carry_what_they_are():
    assert capacity.run_name("attendance", 4, True) == "cap-attendance-n4-rt"
    assert capacity.run_name("attendance", 1, False, "yolo11s") == "cap-attendance-yolo11s-n1-max"


def test_the_plan_has_ceiling_realtime_counts_and_the_largest_ceiling():
    runs = capacity.plan_limits(["a"], [2, 4])
    assert runs == [("a", 1, False), ("a", 1, True), ("a", 2, True), ("a", 4, True), ("a", 4, False)]
    assert capacity.plan_limits(["a"], [1]) == [("a", 1, False), ("a", 1, True)]


def result(name, ran=True, cams=(20.0,), drops=0.0, stalls=0.0, util=60.0):
    fps = {f"cam-{i + 1}": {"median": m, "p10": m, "min": m, "n": 9} for i, m in enumerate(cams)}
    return {"correctness": {"golden": {name: {"ran": ran, "entries": 10, "error": "boom"}}},
            "runtime": {name: {"fps": fps, "drops": {"qos_dropped_total": drops, "stall_episodes": stalls}, "boot_to_first_fps_s": 8.0,
                               "gpu_stages": {"gpu.infer.pgie": {"avg_ms": 18.0}}, "gpu_memory": {"added_mib": 1900},
                               "gpu_counters": {"nvml": {"gpu_util": {"median": util}, "power_w": {"median": 55.0}, "throttled_fraction": 0.1}}}}}


def test_entry_of_reads_a_replay():
    e = capacity.entry_of(result("x", cams=(20.0, 19.0)), "x")
    assert e["ran"] and e["cameras"] == 2 and e["aggregate_fps"] == 39.0 and e["min_cam_median"] == 19.0 and e["gpu_util"] == 60.0
    assert capacity.entry_of(result("x", ran=False), "x") == {"ran": False, "error": "boom"}


def test_realtime_needs_every_camera_at_95_percent_with_no_drops_or_stalls():
    ok = capacity.entry_of(result("x", cams=(20.0, 19.5)), "x")
    assert capacity.realtime_ok(ok, 20.0)
    assert not capacity.realtime_ok(capacity.entry_of(result("x", cams=(20.0, 18.0)), "x"), 20.0)
    assert not capacity.realtime_ok(capacity.entry_of(result("x", cams=(20.0, 20.0), drops=3.0), "x"), 20.0)
    assert not capacity.realtime_ok(capacity.entry_of(result("x", cams=(20.0, 20.0), stalls=1.0), "x"), 20.0)
    assert not capacity.realtime_ok({"ran": False}, 20.0)


def test_summary_finds_the_largest_count_kept_up_with_and_the_headroom():
    e = capacity.entry_of
    entries = {("c", 1, True): e(result("a", cams=(20.0,)), "a"), ("c", 1, False): e(result("b", cams=(28.0,)), "b"),
               ("c", 2, True): e(result("c2", cams=(20.0, 19.8)), "c2"), ("c", 4, True): e(result("c4", cams=(9.0, 9.0, 9.0, 9.0)), "c4"),
               ("c", 4, False): e(result("d", cams=(7.0, 7.0, 7.0, 7.0)), "d")}
    s = capacity.summarize(entries)["c"]
    assert s["source_fps"] == 20.0 and s["headroom_1cam"] == 1.4
    assert s["realtime_cameras_tested_ok"] == 2                      # 1 and 2 kept up, 4 did not
    assert s["aggregate_ceiling_fps"] == 28.0 and s["cameras_by_ceiling"] == 1.4
    assert "c" in capacity.render(capacity.summarize(entries))


def test_a_count_after_a_failure_is_not_counted_even_if_it_passes():
    e = capacity.entry_of
    entries = {("c", 1, True): e(result("a", cams=(20.0,)), "a"), ("c", 2, True): e(result("b", cams=(10.0, 10.0)), "b"),
               ("c", 4, True): e(result("c", cams=(20.0, 20.0, 20.0, 20.0)), "c")}
    assert capacity.summarize(entries)["c"]["realtime_cameras_tested_ok"] == 1


def test_fp16_copy_of_a_pgie_config():
    assert "network-mode=2" in capacity.fp16_config_text("[property]\nnetwork-mode=0\nbatch-size=1\n")
    assert "network-mode=0" not in capacity.fp16_config_text("[property]\nnetwork-mode=0\n")
    assert capacity.fp16_config_text("[property]\nbatch-size=1\n").rstrip().endswith("network-mode=2")


def test_trtexec_parsing():
    text = ("[I] Throughput: 61.2 qps\n[I] Latency: min = 15.9 ms, max = 17 ms, mean = 16.1 ms, median = 16.0 ms, percentile(99%) = 16.9 ms\n"
            "[I] GPU Compute Time: min = 15.5 ms, max = 16.5 ms, mean = 15.9 ms, median = 15.8 ms, percentile(99%) = 16.4 ms\n")
    assert capacity.parse_trtexec(text) == {"gpu_ms_median": 15.8, "throughput_qps": 61.2, "latency_ms_median": 16.0}
    assert capacity.parse_trtexec("nothing") == {}


def test_engine_backup_and_restore_put_the_originals_back(tmp_path):
    repo = tmp_path / "repo"
    (repo / "perception" / "models" / "DeepStream-Yolo").mkdir(parents=True)
    (repo / "model_b1_gpu0_fp32.engine").write_bytes(b"original")
    (repo / "model_b1_gpu0_fp32.engine.src").write_text("onnx=x")
    (repo / "perception" / "models" / "DeepStream-Yolo" / "a.onnx_b1_gpu0_fp16.engine").write_bytes(b"other")
    (repo / "perception" / "models" / "DeepStream-Yolo" / "a.onnx").write_bytes(b"not an engine")
    files = capacity.engine_files(str(repo))
    assert set(files) == {"model_b1_gpu0_fp32.engine", "model_b1_gpu0_fp32.engine.src", "perception/models/DeepStream-Yolo/a.onnx_b1_gpu0_fp16.engine"}
    backup = tmp_path / "bk"
    capacity.backup_engines(str(repo), str(backup))
    (repo / "model_b1_gpu0_fp32.engine").write_bytes(b"rebuilt for another model")        # what a model swap does
    (repo / "model_b1_gpu0_fp32.engine.src").write_text("onnx=y")
    capacity.restore_engines(str(repo), str(backup), files)
    assert (repo / "model_b1_gpu0_fp32.engine").read_bytes() == b"original" and (repo / "model_b1_gpu0_fp32.engine.src").read_text() == "onnx=x"


def test_overlays_are_written_next_to_the_golden_configs_and_removed(tmp_path):
    cfgs = tmp_path / "tests" / "configs"
    cfgs.mkdir(parents=True)
    (cfgs / "golden-attendance.yaml").write_text(BASE)
    runs = capacity.plan_limits(["attendance"], [1, 2])
    names = capacity.write_overlays(str(tmp_path), runs)
    assert all((cfgs / f"golden-{n}.yaml").exists() for n in names.values())
    capacity.remove_overlays(str(tmp_path), names)
    assert [p.name for p in cfgs.iterdir()] == ["golden-attendance.yaml"]


@pytest.mark.skipif(not os.path.exists(os.path.join(REPO, "tests", "configs", "golden-attendance.yaml")), reason="needs the repo's configs")
def test_every_default_capacity_config_can_be_replicated():
    for cfg in capacity.DEFAULT_CONFIGS:
        text = open(os.path.join(REPO, "tests", "configs", f"golden-{cfg}.yaml")).read()
        assert len(capacity.replicate_sources(text, 3)) == 3, cfg


def test_a_config_that_cannot_keep_up_with_one_camera_uses_the_clips_rate_not_its_own(monkeypatch):
    e = capacity.entry_of
    entries = {("fall", 1, True): e(result("a", cams=(17.4,)), "a"), ("fall", 1, False): e(result("b", cams=(17.6,)), "b")}
    s = capacity.summarize(entries, {"fall": 25.0})["fall"]
    assert s["source_fps"] == 25.0 and s["realtime_cameras_tested_ok"] == 0 and s["headroom_1cam"] == 0.7


def test_the_camera_ladder_stops_at_the_first_count_not_kept_up_with(monkeypatch):
    table = {"cap-attendance-n1-max": 28.0, "cap-attendance-n1-rt": 20.0, "cap-attendance-n2-rt": 13.0, "cap-attendance-n2-max": 12.0,
             "cap-attendance-n4-rt": 99.0, "cap-attendance-n4-max": 99.0}
    ran = []

    def fake_run(repo, names, out_json, timeout_s=0):
        name = names[0]
        ran.append(name)
        n = int(name.split("-n")[1].split("-")[0])
        return result(name, cams=tuple([table[name]] * n))
    monkeypatch.setattr(capacity, "run_replays", fake_run)
    monkeypatch.setattr(capacity, "write_overlays", lambda repo, runs, extra_for=None: {r: capacity.run_name(*r) for r in runs})
    monkeypatch.setattr(capacity, "remove_overlays", lambda repo, names: None)
    got = capacity.measure_one("r", "attendance", "attendance", [1, 2, 4], "out", src=20.0)
    assert ran == ["cap-attendance-n1-max", "cap-attendance-n1-rt", "cap-attendance-n2-rt", "cap-attendance-n2-max"]    # no 4 cameras
    s = capacity.summarize(got, {"attendance": 20.0})["attendance"]
    assert s["realtime_cameras_tested_ok"] == 1 and s["aggregate_ceiling_cameras"] == 2 and s["ceiling_fps_1cam"] == 28.0


def test_the_ladder_goes_on_while_every_camera_keeps_up(monkeypatch):
    ran = []

    def fake_run(repo, names, out_json, timeout_s=0):
        ran.append(names[0])
        n = int(names[0].split("-n")[1].split("-")[0])
        return result(names[0], cams=tuple([20.0] * n))
    monkeypatch.setattr(capacity, "run_replays", fake_run)
    monkeypatch.setattr(capacity, "write_overlays", lambda repo, runs, extra_for=None: {r: capacity.run_name(*r) for r in runs})
    monkeypatch.setattr(capacity, "remove_overlays", lambda repo, names: None)
    capacity.measure_one("r", "attendance", "attendance", [1, 2, 4], "out", src=20.0)
    assert ran == ["cap-attendance-n1-max", "cap-attendance-n1-rt", "cap-attendance-n2-rt", "cap-attendance-n4-rt", "cap-attendance-n4-max"]


def test_noisy_steps_are_carried_into_the_summary_and_marked_in_the_table():
    noisy = result("a", cams=(20.0,))
    noisy["noisy"] = True
    noisy["guardrails"] = {"reasons": ["GPU busy: 35 %"]}
    e = capacity.entry_of
    entries = {("c", 1, True): e(noisy, "a"), ("c", 1, False): e(result("b", cams=(28.0,)), "b")}
    s = capacity.summarize(entries, {"c": 20.0})
    assert s["c"]["noisy_steps"] == ["n1-rt"] and s["c"]["runs"]["n1-rt"]["noise_reasons"] == ["GPU busy: 35 %"]
    assert "c *" in capacity.render(s) and "lower bounds" in capacity.render(s)
    assert "*" not in capacity.render(capacity.summarize({("c", 1, True): e(result("a"), "a")}, {"c": 20.0}))


def test_resume_reuses_a_steps_results_file(tmp_path, monkeypatch):
    out = tmp_path / "step.json"
    out.write_text(json.dumps(result("cap-x-n1-max", cams=(30.0,))))
    called = []
    monkeypatch.setattr(capacity.subprocess, "run", lambda *a, **k: called.append(a))
    monkeypatch.setattr(capacity, "RESUME", True)
    got = capacity.run_replays("r", ["cap-x-n1-max"], str(out))
    assert not called and capacity.entry_of(got, "cap-x-n1-max")["aggregate_fps"] == 30.0


def test_restore_replaces_a_file_that_cannot_be_overwritten_and_removes_engines_that_were_not_there(tmp_path):
    repo = tmp_path / "repo"
    (repo / "perception" / "models" / "DeepStream-Yolo").mkdir(parents=True)
    (repo / "model_b1_gpu0_fp32.engine").write_bytes(b"original")
    (repo / "model_b1_gpu0_fp32.engine.src").write_text("onnx=x")
    files = capacity.engine_files(str(repo))
    backup = tmp_path / "bk"
    capacity.backup_engines(str(repo), str(backup))
    eng = repo / "model_b1_gpu0_fp32.engine"
    eng.write_bytes(b"rebuilt by a container")
    os.chmod(eng, 0o444)                                                    # what a root-owned file looks like to this user
    (repo / "model_b2_gpu0_fp32.engine").write_bytes(b"built during the swap")
    (repo / "perception" / "models" / "DeepStream-Yolo" / "other.onnx_b1_gpu0_fp16.engine").write_bytes(b"also new")
    out = capacity.restore_engines(str(repo), str(backup), files)
    assert eng.read_bytes() == b"original" and out["restored"] == 2
    assert sorted(out["removed"]) == ["model_b2_gpu0_fp32.engine", "perception/models/DeepStream-Yolo/other.onnx_b1_gpu0_fp16.engine"]
    assert not (repo / "model_b2_gpu0_fp32.engine").exists()


def test_a_restore_that_did_not_reproduce_the_backup_raises(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "model_b1_gpu0_fp32.engine").write_bytes(b"original")
    files = capacity.engine_files(str(repo))
    backup = tmp_path / "bk"
    capacity.backup_engines(str(repo), str(backup))
    monkeypatch.setattr(capacity.shutil, "copy2", lambda s, d: open(d, "wb").write(b"corrupt"))
    with pytest.raises(RuntimeError):
        capacity.restore_engines(str(repo), str(backup), files)


def test_capacity_metrics_are_flattened_and_have_a_family():
    from regress import metrics
    doc = {"capacity": {"attendance": {"ceiling_fps_1cam": 28.5, "realtime_cameras_tested_ok": 1, "aggregate_ceiling_fps": 26.1},
                        "attendance@af5": {"ceiling_fps_1cam": 120.0, "realtime_cameras_tested_ok": 4}}}
    flat = metrics.flatten(doc)
    assert flat["capacity.attendance.ceiling_fps"].value == 28.5 and flat["capacity.attendance.ceiling_fps"].direction == "higher"
    assert "capacity.attendance.realtime_cameras" not in flat and "capacity.attendance.aggregate_ceiling_fps" not in flat     # depends on the ladder
    assert flat["capacity.attendance@af5.realtime_cameras"].value == 4 and "capacity.attendance@af5.ceiling_fps" not in flat  # capped at the rate
    assert metrics.family("capacity.attendance@af5.ceiling_fps") == "capacity.*.ceiling_fps"


def test_a_capacity_results_file_is_compared_like_any_other(monkeypatch, tmp_path):
    from regress import baseline, compare, machine, tolerances
    monkeypatch.setattr(machine, "fingerprint", lambda repo: {"machine_id": "m1", "git": {"commit": "c", "short": "c", "branch": "b", "dirty": False}})
    summary = {"attendance": {"source_fps": 20.0, "ceiling_fps_1cam": 28.5, "realtime_cameras_tested_ok": 1, "noisy_steps": [], "runs": {}}}
    ref = capacity.results_doc(str(tmp_path), summary)
    assert ref["schema"] == "regress.v1" and ref["capacity"]["attendance"]["ceiling_fps_1cam"] == 28.5 and "runs" not in ref["capacity"]["attendance"]
    doc, _ = baseline.build([ref])
    tol = {"default": {"rel": 0.10, "abs": 0.0}, "rules": [{"match": "capacity.*.ceiling_fps", "rel": 0.03, "abs": 0.0}], "source": None}
    same = compare.compare(ref, doc, tol)
    assert same["verdict"] == "PASS"
    slower = capacity.results_doc(str(tmp_path), {"attendance": dict(summary["attendance"], ceiling_fps_1cam=24.0)})       # -16 %
    rep = compare.compare(slower, doc, tol)
    assert any(r["id"] == "capacity.attendance.ceiling_fps" and r["status"] == "WARN" for r in rep["rows"])
    nearly = capacity.results_doc(str(tmp_path), {"attendance": dict(summary["attendance"], ceiling_fps_1cam=28.0)})        # -1.8 %: inside 3 %
    assert compare.compare(nearly, doc, tol)["verdict"] == "PASS"


def test_the_analysis_rate_cells_are_labelled_and_tagged_separately(monkeypatch):
    seen = []

    def fake_run(repo, names, out_json, timeout_s=0):
        seen.append(names[0])
        n = int(names[0].split("-n")[1].split("-")[0])
        return result(names[0], cams=tuple([5.0] * n))
    monkeypatch.setattr(capacity, "run_replays", fake_run)
    monkeypatch.setattr(capacity, "write_overlays", lambda repo, runs, extra_for=None: {r: capacity.run_name(*r) for r in runs})
    monkeypatch.setattr(capacity, "remove_overlays", lambda repo, names: None)
    monkeypatch.setattr(capacity, "source_rate", lambda repo, base: 20.0)
    s = capacity.measure_limits("r", ["attendance"], [1, 2], "out", analysis_fps=5)
    assert list(s) == ["attendance@af5"] and all("@" not in n and "-af5-" in n for n in seen)      # file and compose names never see the label
    assert seen == ["cap-attendance-af5-n1-rt", "cap-attendance-af5-n2-rt"]                      # no un-paced steps at a fixed analysis rate
    assert s["attendance@af5"]["realtime_cameras_tested_ok"] == 2


def test_a_one_camera_run_is_not_compared_on_cameras_with_a_baseline_that_climbed_higher():
    from regress import baseline, compare, machine
    import regress.capacity as cap
    ladder = {"barber": {"source_fps": 25.0, "ceiling_fps_1cam": 94.0, "realtime_cameras_tested_ok": 2, "noisy_steps": [], "runs": {}}}
    short = {"barber": {"source_fps": 25.0, "ceiling_fps_1cam": 93.5, "realtime_cameras_tested_ok": 1, "noisy_steps": [], "runs": {}}}
    monkey = lambda repo: {"machine_id": "m1", "git": {"commit": "c", "short": "c", "branch": "b", "dirty": False}}   # noqa: E731
    machine_fingerprint, machine.fingerprint = machine.fingerprint, monkey
    try:
        ref, new = cap.results_doc("x", ladder), cap.results_doc("x", short)
    finally:
        machine.fingerprint = machine_fingerprint
    doc, _ = baseline.build([ref])
    tol = {"default": {"rel": 0.10, "abs": 0.0}, "rules": [{"match": "capacity.*.ceiling_fps", "rel": 0.03, "abs": 0.0}], "source": None}
    assert compare.compare(new, doc, tol)["verdict"] == "PASS"
