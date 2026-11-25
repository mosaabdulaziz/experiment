"""The compare / noise / baseline half of the regression harness (tasks #228 and #230). Synthetic results files only:
no docker, no GPU. The shape is the one `regress run` writes (schema regress.v1)."""
import json
import os
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

from regress import baseline, cli, compare, metrics, noise, tolerances  # noqa: E402


def result(machine="m1", tests=100, failing=0, boot=8.0, fps=20.0, drops=0.0, sha="aaa", entries=50, dec_ms=20.0,
           sm=0.5, rebuilt=False, noisy=False, commit="c1", dirty=False, golden=True, ran=True, img=1000,
           status="IDENTICAL"):
    r = {"schema": "regress.v1", "machine_id": machine, "noisy": noisy,
         "fingerprint": {"machine_id": machine, "gpu": "G", "driver": "1", "cuda": "13",
                         "git": {"commit": commit, "short": commit, "branch": "b", "dirty": dirty}},
         "correctness": {"unit": {"core": {"ran": True, "tests": tests, "failures": failing, "errors": 0, "skipped": 0,
                                           "time_s": 80.0, "wall_s": 80.0, "failed": [], "slowest": []}}},
         "runtime": {}, "build": {"images": {"img:1": img}, "frontend_bundle": {"total_bytes": 1000, "js_bytes": 800,
                                                                                  "css_bytes": 100, "largest_chunk": {"file": "a.js", "bytes": 500}},
                                  "lock_packages": {"uv.lock": 10}},
         "verdict": {"pass": failing == 0, "reasons": [], "known_failures": []}}
    if golden:
        r["correctness"]["golden"] = {"att": {"ran": ran, "entries": entries, "output_sha256": sha, "output_file": "x.jsonl",
                                              "vs_repo_baseline": {"status": status, "ok": status == "IDENTICAL"}}}
        r["runtime"]["att"] = {"boot_to_first_fps_s": boot, "engine": {"cached": not rebuilt, "rebuilt": rebuilt,
                                                                       "cached_elements": [], "rebuilt_elements": []},
                               "fps": {"cam-1": {"median": fps, "p10": fps, "min": fps, "n": 9}},
                               "drops": {"qos_dropped_total": drops, "publisher_drops": 0.0, "pub_shed_total": 0.0, "stall_episodes": 0.0},
                               "gpu_stages": {"gpu.decode": {"busy": 40.0, "avg_ms": dec_ms, "max_ms": 30.0, "n": 99}},
                               "gpu_total_busy": 40.0, "gpu_memory": {"before_mib": 1, "peak_mib": 101, "added_mib": 100},
                               "gpu_counters": {"dcgm_valid": True, "dcgm": {"sm_active": {"median": sm, "p95": sm, "max": sm, "mean": sm, "n": 9}},
                                                "nvml": {"gpu_util": {"median": 60.0, "p95": 61.0, "max": 61.0, "mean": 60.0, "n": 9}}}}
    return r


def base_of(res):
    return compare.load_baseline_from(res) if hasattr(compare, "load_baseline_from") else _as_baseline(res)


def _as_baseline(res, tmp=None):
    flat = metrics.flatten(res)
    golden = {c: {"sha256": g["output_sha256"], "output": g["output_file"]}
              for c, g in res["correctness"].get("golden", {}).items() if g.get("ran")}
    return {"schema": "regress.baseline.v1", "machine_id": res["machine_id"], "fingerprint": res["fingerprint"],
            "noisy": res["noisy"], "metrics": {k: {"value": m.value, "kind": m.kind, "direction": m.direction, "unit": m.unit}
                                               for k, m in flat.items()}, "golden": golden, "_dir": None}


TOL = {"default": {"rel": 0.10, "abs": 0.0}, "rules": [], "source": None}


def run(new, base, **kw):
    return compare.compare(new, _as_baseline(base), kw.pop("tol", TOL), **kw)


def run_(new, base, **kw):
    return run(new, base, **kw)


def status(rep, mid):
    return next(r["status"] for r in rep["rows"] if r["id"] == mid)


def test_identical_runs_pass():
    rep = run(result(), result())
    assert rep["verdict"] == "PASS" and rep["exit"] == 0 and rep["counts"]["FAIL"] == 0 and rep["counts"]["WARN"] == 0


def test_a_rebuilt_engine_leaves_the_boot_time_out():
    assert "runtime.att.boot_s" not in metrics.flatten(result(rebuilt=True))
    rep = run(result(rebuilt=True, boot=300.0), result())            # 300 s is not a regression: it was an engine build
    assert all(r["id"] != "runtime.att.boot_s" for r in rep["rows"])


# ── correctness fails ────────────────────────────────────────────────────────────────────────────────────────────────
def test_more_failing_tests_than_the_baseline_fails():
    rep = run(result(failing=2), result())
    assert status(rep, "unit.core.failing") == "FAIL" and rep["exit"] == 1


def test_fewer_tests_than_the_baseline_fails():
    rep = run(result(tests=90), result(tests=100))
    assert status(rep, "unit.core.tests") == "FAIL" and rep["exit"] == 1


def test_more_tests_is_fine():
    assert status(run(result(tests=120), result(tests=100)), "unit.core.tests") == "PASS"


def test_the_same_failures_as_a_baseline_that_had_them_do_not_fail():
    """The Orin baseline carries one known failing test (an older ffmpeg): the same failure is not a regression."""
    assert status(run(result(failing=1), result(failing=1)), "unit.core.failing") == "PASS"


def test_a_golden_replay_that_did_not_run_fails():
    rep = run(result(ran=False), result())
    assert status(rep, "golden.att.output") == "FAIL" and rep["exit"] == 1


def test_a_config_the_run_did_not_cover_is_not_a_failure_unless_strict():
    new = result()
    new["correctness"]["golden"]["other"] = new["correctness"]["golden"].pop("att")      # a partial run: only `other` was replayed
    rep = run(new, result())
    assert status(rep, "golden.att.output") == "INFO" and rep["exit"] == 0
    assert status(run(new, result(), strict=True), "golden.att.output") == "FAIL"
    assert run(new, result(), strict=True)["exit"] == 1


def test_a_config_that_was_asked_for_and_did_not_run_still_fails_even_in_a_partial_run():
    new = result(ran=False)
    new["correctness"]["golden"]["other"] = dict(new["correctness"]["golden"]["att"], ran=True)
    assert status(run(new, result()), "golden.att.output") == "FAIL"


def test_a_missing_golden_section_is_noted_not_failed():
    rep = run(result(golden=False), result())
    assert rep["exit"] == 0 and any("no golden replay" in n for n in rep["notes"])


def test_identical_golden_output_passes():
    assert status(run(result(sha="aaa"), result(sha="aaa")), "golden.att.output") == "PASS"


def test_a_changed_golden_output_is_judged_by_the_diff(tmp_path):
    b, n = tmp_path / "b.jsonl", tmp_path / "n.jsonl"
    b.write_text("1\n")
    n.write_text("2\n")
    base, new = result(sha="aaa"), result(sha="bbb")
    base["correctness"]["golden"]["att"]["output_file"] = str(b)
    new["correctness"]["golden"]["att"]["output_file"] = str(n)
    assert status(run(new, base, golden_diff=lambda c, x, y: (True, "SEMANTICALLY EQUAL")), "golden.att.output") == "PASS"
    rep = run(new, base, golden_diff=lambda c, x, y: (False, "3 difference(s)"))
    assert status(rep, "golden.att.output") == "FAIL" and rep["exit"] == 1


def test_a_changed_output_without_the_baseline_output_file_only_warns():
    rep = run(result(sha="bbb"), result(sha="aaa"))
    assert status(rep, "golden.att.output") == "WARN" and rep["exit"] == 0


# ── efficiency warns ─────────────────────────────────────────────────────────────────────────────────────────────────
def test_a_slower_stage_beyond_the_tolerance_warns_and_a_small_one_passes():
    rep = run(result(dec_ms=25.0), result(dec_ms=20.0))                  # +25% > 10%
    assert status(rep, "runtime.att.stage.gpu.decode.avg_ms") == "WARN" and rep["exit"] == 0 and rep["verdict"] == "WARN"
    assert status(run(result(dec_ms=21.0), result(dec_ms=20.0)), "runtime.att.stage.gpu.decode.avg_ms") == "PASS"


def test_strict_turns_warnings_into_failures():
    rep = run(result(dec_ms=25.0), result(dec_ms=20.0), strict=True)
    assert rep["exit"] == 1 and status(rep, "runtime.att.stage.gpu.decode.avg_ms") == "FAIL"


def test_lower_fps_warns_and_higher_fps_is_an_improvement():
    assert status(run(result(fps=15.0), result(fps=20.0)), "runtime.att.fps_p10") == "WARN"
    assert status(run(result(fps=25.0), result(fps=20.0)), "runtime.att.fps_p10") == "IMPROVED"


def test_a_drop_where_there_were_none_warns():
    assert status(run(result(drops=1.0), result(drops=0.0)), "runtime.att.drops") == "WARN"


def test_a_utilisation_counter_that_moves_either_way_warns():
    assert status(run(result(sm=0.3), result(sm=0.5)), "counters.att.sm_active.median") == "WARN"
    assert status(run(result(sm=0.7), result(sm=0.5)), "counters.att.sm_active.median") == "WARN"


def test_a_bigger_image_and_a_changed_package_count_warn():
    rep = run(result(img=1300), result(img=1000))
    assert status(rep, "build.image.img:1") == "WARN"
    new = result()
    new["build"]["lock_packages"]["uv.lock"] = 11
    assert status(run(new, result()), "build.lock.uv.lock") == "WARN"


def test_the_absolute_tolerance_wins_when_it_is_larger():
    tol = {"default": {"rel": 0.01, "abs": 0.0}, "rules": [{"match": "runtime.*.drops", "rel": 0.0, "abs": 5}], "source": None}
    assert status(run(result(drops=3.0), result(drops=0.0), tol=tol), "runtime.att.drops") == "PASS"
    assert status(run(result(drops=6.0), result(drops=0.0), tol=tol), "runtime.att.drops") == "WARN"


# ── comparability ────────────────────────────────────────────────────────────────────────────────────────────────────
def test_another_machine_is_not_directly_comparable_but_unit_suites_still_count():
    rep = run(result(machine="m2", dec_ms=99.0, sha="zzz", tests=90), result(machine="m1"))
    assert not rep["comparable"] and any("not directly comparable" in n for n in rep["notes"])
    assert status(rep, "runtime.att.stage.gpu.decode.avg_ms") == "INFO"
    assert status(rep, "golden.att.output") == "INFO"
    assert status(rep, "unit.core.tests") == "FAIL" and rep["exit"] == 1


def test_a_noisy_run_is_advisory():
    rep = run(result(noisy=True, dec_ms=99.0), result())
    assert status(rep, "runtime.att.stage.gpu.decode.avg_ms") == "INFO" and rep["exit"] == 0
    assert any("NOISY" in n for n in rep["notes"])


def test_a_different_driver_is_noted():
    new = result()
    new["fingerprint"]["driver"] = "2"
    assert any("driver differs" in n for n in run(new, result())["notes"])


def test_render_lists_failures_first_and_hides_passes_unless_asked():
    rep = run(result(failing=1, dec_ms=25.0), result())
    text = compare.render(rep)
    assert text.index("FAIL") < text.index("WARN") and "PASS  " not in text.split("\n")[1]
    assert "->  FAIL" in text and "PASS" in compare.render(rep, show_all=True)


# ── tolerances ───────────────────────────────────────────────────────────────────────────────────────────────────────
def test_tolerance_rules_match_by_id_or_family_and_first_wins():
    tol = {"default": {"rel": 0.1, "abs": 0}, "rules": [{"match": "runtime.*.fps_p10", "rel": 0.5, "abs": 0},
                                                         {"match": "runtime.*", "rel": 0.01, "abs": 0}], "source": None}
    assert tolerances.allowed(tol, "runtime.att.fps_p10", metrics.family("runtime.att.fps_p10"), 20.0) == 10.0
    assert tolerances.allowed(tol, "runtime.att.boot_s", metrics.family("runtime.att.boot_s"), 10.0) == pytest.approx(0.1)
    assert tolerances.allowed(tol, "build.x", "build.x", 10.0) == 1.0


def test_load_tolerances_without_a_file_uses_the_default(tmp_path):
    assert tolerances.load(str(tmp_path / "none.json"))["default"] == tolerances.DEFAULT


def test_family_names():
    assert metrics.family("runtime.attendance.fps_p10") == "runtime.*.fps_p10"
    assert metrics.family("runtime.a.stage.gpu.decode.busy") == "runtime.*.stage.gpu.decode.busy"
    assert metrics.family("build.image.x:latest") == "build.image.*"
    assert metrics.family("build.frontend.total_bytes") == "build.frontend.total_bytes"


# ── noise ────────────────────────────────────────────────────────────────────────────────────────────────────────────
RUNS = [result(boot=8.0, dec_ms=20.0), result(boot=8.4, dec_ms=20.0),
        result(boot=7.8, dec_ms=22.0, sha="bbb", status="1 difference(s)"),
        result(boot=8.1, dec_ms=20.5), result(boot=8.2, dec_ms=20.1)]


def test_noise_reports_the_spread_and_leaves_out_nothing_that_moved():
    an = noise.analyze(RUNS)
    boot = an["metrics"]["runtime.att.boot_s"]
    assert boot["n"] == 5 and boot["min"] == 7.8 and boot["max"] == 8.4 and boot["median"] == 8.1
    assert boot["spread_rel"] == pytest.approx(0.6 / 8.1, abs=1e-3)
    assert an["golden"]["att"]["distinct"] == 2 and an["golden"]["att"]["statuses"] == {"IDENTICAL": 4, "1 difference(s)": 1}


STABLE = [result(boot=8.0, dec_ms=20.0), result(boot=8.4, dec_ms=20.0), result(boot=7.8, dec_ms=22.0, sha="bbb"),
          result(boot=8.1, dec_ms=20.5), result(boot=8.2, dec_ms=20.1)]       # the same golden status every time


def test_suggested_tolerance_is_never_tighter_than_the_noise():
    an = noise.analyze(STABLE)
    rules = {r["match"]: r for r in noise.suggest(an)["rules"]}
    boot = rules["runtime.*.boot_s"]
    assert boot["rel"] >= an["metrics"]["runtime.att.boot_s"]["spread_rel"] * noise.SAFETY
    dec = rules["runtime.*.stage.gpu.decode.avg_ms"]
    assert dec["rel"] >= an["metrics"]["runtime.att.stage.gpu.decode.avg_ms"]["spread_rel"] * noise.SAFETY
    assert all(r["rel"] >= noise.MIN_REL for r in rules.values())
    # a metric that never moved still gets the 2% floor, and correctness metrics get no tolerance at all
    assert "unit.*.tests" not in rules and "build.lock.*" not in rules


def test_noise_render_mentions_a_flaky_config():
    an = noise.analyze(RUNS)
    assert "FLAKY" in noise.render(an, noise.suggest(an))


def test_the_raw_hash_is_not_the_determinism_measure():
    """Identical outputs carry volatile fields, so every run's raw hash differs: only the replay's own diff counts."""
    runs = [result(sha=f"h{i}") for i in range(4)]
    assert noise.analyze(runs)["golden"]["att"] == {"distinct": 1, "statuses": {"IDENTICAL": 4}}


def test_a_metric_that_is_unstable_between_identical_runs_is_not_gated():
    # fire's face stage ran in some runs and idled in others: 0.6 ms vs 15 ms
    runs = [result(dec_ms=v) for v in (20.0, 20.5, 5.0, 20.2, 4.8)]
    tol = noise.suggest(noise.analyze(runs))
    off = [r for r in tol["rules"] if r.get("gate") is False]
    assert [r["match"] for r in off] == ["runtime.att.stage.gpu.decode.avg_ms"] and "apart between identical runs" in off[0]["why"]
    assert tol["rules"][0].get("gate") is False                               # first: the first match wins
    assert not any(r["match"] == "runtime.*.stage.gpu.decode.avg_ms" for r in tol["rules"])   # and not in the family's tolerance
    rep = run_(result(dec_ms=99.0), result(dec_ms=20.0), tol=tol)
    assert status(rep, "runtime.att.stage.gpu.decode.avg_ms") == "INFO" and rep["exit"] == 0


def test_a_small_absolute_spread_is_not_instability():
    runs = [result(drops=v) for v in (0.0, 0.0, 1.0, 0.0, 0.0)]                # 1 drop: infinite relative, 1 absolute
    assert [r for r in noise.suggest(noise.analyze(runs))["rules"] if r.get("gate") is False] == []


def test_a_flaky_golden_config_only_warns(tmp_path):
    b, n = tmp_path / "b.jsonl", tmp_path / "n.jsonl"
    b.write_text("1\n")
    n.write_text("2\n")
    base = _as_baseline(result(sha="aaa"))
    base["golden"]["att"]["output"] = str(b)
    base["golden"]["att"]["deterministic"] = False
    new = result(sha="bbb")
    new["correctness"]["golden"]["att"]["output_file"] = str(n)
    rep = compare.compare(new, base, TOL, golden_diff=lambda c, x, y: (False, "1 difference(s)"))
    assert status(rep, "golden.att.output") == "WARN" and rep["exit"] == 0
    assert "flaky" in next(r["note"] for r in rep["rows"] if r["id"] == "golden.att.output")


def test_flatten_drops_a_stalled_counter_window_even_in_an_old_result_file():
    r = result()
    st = lambda m: {"median": m, "p95": m, "max": m, "mean": m, "n": 9}      # noqa: E731
    r["runtime"]["att"]["gpu_counters"]["dcgm"] = {"gr_engine_active": st(0.95), "sm_active": st(0.0)}
    assert not any(k.startswith("counters.att.sm_active") for k in metrics.flatten(r))
    assert any(k.startswith("counters.att.gr_engine_active") for k in metrics.flatten(r))


# ── baseline ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_baseline_is_the_median_and_keeps_the_most_frequent_output(tmp_path):
    runs = []
    for i, r in enumerate(RUNS):
        f = tmp_path / f"o{i}.jsonl"
        f.write_text(r["correctness"]["golden"]["att"]["output_sha256"] + "\n")
        r["correctness"]["golden"]["att"]["output_file"] = str(f)
        runs.append(r)
    doc, keep = baseline.build(runs)
    assert doc["metrics"]["runtime.att.boot_s"]["value"] == 8.1
    g = doc["golden"]["att"]
    assert g["sha256"] == "aaa" and g["deterministic"] is False and g["statuses"] == {"IDENTICAL": 4, "1 difference(s)": 1}
    assert open(keep["att"]).read().strip() == "aaa"
    path = baseline.write(str(tmp_path), doc, keep)
    assert path.endswith(os.path.join("benchmarks", "baselines", "m1", "baseline.json"))
    assert (tmp_path / "benchmarks" / "baselines" / "m1" / "golden" / "att.jsonl").read_text().strip() == "aaa"
    loaded = compare.load_baseline(path)
    assert loaded["_dir"] == os.path.dirname(path) and loaded["golden"]["att"]["output"] == "golden/att.jsonl"


def test_baseline_refuses_dirty_noisy_failing_or_mixed_commit_runs():
    assert baseline.check([result(), result()], allow_noisy=False) == []
    assert any("dirty" in r for r in baseline.check([result(dirty=True)], False))
    assert any("not on an idle" in r for r in baseline.check([result(noisy=True)], False))
    assert baseline.check([result(noisy=True)], True) == []
    assert any("fails correctness" in r for r in baseline.check([result(failing=1)], False))
    assert any("different commits" in r for r in baseline.check([result(commit="a"), result(commit="b")], False))


@pytest.mark.skipif(shutil.which("git") is None, reason="needs the git binary (the CI image has none)")
def test_baseline_record_refuses_a_dirty_tree(tmp_path, capsys):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "f").write_text("1")
    subprocess.run(["git", "-C", str(tmp_path), "add", "f"], check=True)
    assert baseline.record(str(tmp_path), 1, [], allow_dirty=False, allow_noisy=False) == 2
    assert "dirty tree" in capsys.readouterr().err


def test_compare_from_a_results_file_as_the_baseline(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(result()))
    b.write_text(json.dumps(result(dec_ms=30.0)))
    base = compare.load_baseline(str(a))
    assert base["_from_results"] and base["golden"]["att"]["sha256"] == "aaa"
    rep = compare.compare(json.load(open(b)), base, TOL)
    assert rep["verdict"] == "WARN"


# ── the commands ─────────────────────────────────────────────────────────────────────────────────────────────────────
def test_cli_compare_exit_codes(tmp_path, capsys):
    base, ok, bad = tmp_path / "base.json", tmp_path / "ok.json", tmp_path / "bad.json"
    base.write_text(json.dumps(result()))
    ok.write_text(json.dumps(result(dec_ms=20.5)))
    bad.write_text(json.dumps(result(tests=10)))
    tol = tmp_path / "t.json"
    tol.write_text(json.dumps({"default": {"rel": 0.1, "abs": 0}, "rules": []}))
    assert cli.main(["compare", str(ok), "--baseline", str(base), "--tolerances", str(tol)]) == 0
    assert "->  PASS" in capsys.readouterr().out
    assert cli.main(["compare", str(bad), "--baseline", str(base), "--tolerances", str(tol)]) == 1
    assert "->  FAIL" in capsys.readouterr().out
    assert cli.main(["compare", str(ok), "--baseline", str(tmp_path / "missing.json")]) == 2


def test_cli_noise_writes_the_tolerance_file(tmp_path, capsys):
    files = []
    for i, r in enumerate(RUNS):
        p = tmp_path / f"r{i}.json"
        p.write_text(json.dumps(r))
        files.append(str(p))
    out = tmp_path / "tol.json"
    assert cli.main(["noise", *files, "--tolerances", str(out)]) == 0
    data = json.loads(out.read_text())
    assert data["rules"] and data["runs"] == 5 and "identical runs" in capsys.readouterr().out


def test_a_flaky_config_has_none_of_its_efficiency_gated():
    runs = [result(), result(), result(status="1 difference(s)"), result(), result()]
    tol = noise.suggest(noise.analyze(runs))
    assert {"runtime.att.*", "counters.att.*"} <= {r["match"] for r in tol["rules"] if r.get("gate") is False}
    rep = run_(result(dec_ms=99.0), result(dec_ms=20.0), tol=tol)
    assert status(rep, "runtime.att.stage.gpu.decode.avg_ms") == "INFO"
    assert status(rep, "build.image.img:1") == "PASS"                      # build metrics are not the config's


def test_golden_outputs_are_found_relative_to_the_checkout_that_wrote_the_results(tmp_path):
    root = tmp_path / "co"
    (root / "out").mkdir(parents=True)
    (root / "out" / "n.jsonl").write_text("2\n")
    b = tmp_path / "b.jsonl"
    b.write_text("1\n")
    base = _as_baseline(result(sha="aaa"))
    base["golden"]["att"]["output"] = str(b)
    new = result(sha="bbb")
    new["correctness"]["golden"]["att"]["output_file"] = "out/n.jsonl"            # relative, as `regress run` writes it
    seen = []
    rep = compare.compare(new, base, TOL, golden_diff=lambda c, x, y: (seen.append((x, y)) or (True, "ok")), outputs_root=str(root))
    assert seen == [(str(b), str(root / "out" / "n.jsonl"))] and status(rep, "golden.att.output") == "PASS"


def test_a_changed_golden_output_after_an_engine_rebuild_only_warns(tmp_path):
    """Two checkouts of the same code each built their own TensorRT engine and gave two different, stable outputs."""
    b, n = tmp_path / "b.jsonl", tmp_path / "n.jsonl"
    b.write_text("1\n")
    n.write_text("2\n")
    base = _as_baseline(result(sha="aaa"))
    base["golden"]["att"]["output"] = str(b)
    new = result(sha="bbb", rebuilt=True)
    new["correctness"]["golden"]["att"]["output_file"] = str(n)
    rep = compare.compare(new, base, TOL, golden_diff=lambda c, x, y: (False, "85 mismatched"))
    row = next(r for r in rep["rows"] if r["id"] == "golden.att.output")
    assert row["status"] == "WARN" and "engine was rebuilt" in row["note"] and rep["exit"] == 0
    new2 = result(sha="bbb")
    new2["correctness"]["golden"]["att"]["output_file"] = str(n)
    rep = compare.compare(new2, base, TOL, golden_diff=lambda c, x, y: (False, "85 mismatched"))                 # not rebuilt: a real FAIL
    assert rep["exit"] == 1 and status(rep, "golden.att.output") == "FAIL"


def test_compare_prefers_the_machines_own_tolerances(tmp_path, monkeypatch, capsys):
    repo = tmp_path / "repo"
    (repo / "benchmarks" / "baselines" / "m1").mkdir(parents=True)
    (repo / "benchmarks" / "tolerances.json").write_text(json.dumps({"default": {"rel": 0.5, "abs": 0}, "rules": []}))
    base, new = tmp_path / "base.json", tmp_path / "new.json"
    base.write_text(json.dumps(result()))
    new.write_text(json.dumps(result(dec_ms=25.0)))                              # +25%
    monkeypatch.setattr(cli, "REPO", str(repo))
    assert cli.main(["compare", str(new), "--baseline", str(base)]) == 0
    assert "0 warn" in capsys.readouterr().out                                    # inside the global 50%
    (repo / "benchmarks" / "baselines" / "m1" / "tolerances.json").write_text(json.dumps({"default": {"rel": 0.05, "abs": 0}, "rules": []}))
    assert cli.main(["compare", str(new), "--baseline", str(base)]) == 0         # a warning does not fail
    assert " 1 warn" in capsys.readouterr().out                                   # but the machine's own 5% sees +25%


def test_an_engine_rebuilt_by_another_config_of_the_same_run_downgrades_a_changed_output_to_a_warning(tmp_path):
    new = result(sha="b" * 64, status="1 difference(s)")
    new["runtime"]["other"] = dict(new["runtime"]["att"], engine={"cached": False, "rebuilt": True, "cached_elements": [], "rebuilt_elements": ["pgie"]})
    new["runtime"]["att"]["engine"] = {"cached": True, "rebuilt": False, "cached_elements": [], "rebuilt_elements": []}
    (tmp_path / "a.jsonl").write_text("{}\n")
    (tmp_path / "b.jsonl").write_text("{}\n")
    base = _as_baseline(result())
    base["_dir"] = str(tmp_path)
    base["golden"]["att"]["output"] = "a.jsonl"
    new["correctness"]["golden"]["att"]["output_file"] = "b.jsonl"
    rep = compare.compare(new, base, TOL, golden_diff=lambda cfg, a, b: (False, "1 difference(s)"), outputs_root=str(tmp_path))
    assert status(rep, "golden.att.output") == "WARN"


def test_a_config_without_a_repo_baseline_is_non_deterministic_when_its_raw_outputs_differ():
    rs = []
    for sha in ("a" * 64, "b" * 64, "a" * 64):
        r = result(sha=sha)
        r["correctness"]["golden"]["att"]["vs_repo_baseline"] = {"status": "no baseline", "ok": True}
        rs.append(r)
    doc, _ = baseline.build(rs)
    assert doc["golden"]["att"]["deterministic"] is False
    same = [result(sha="a" * 64) for _ in range(3)]
    for r in same:
        r["correctness"]["golden"]["att"]["vs_repo_baseline"] = {"status": "no baseline", "ok": True}
    assert baseline.build(same)[0]["golden"]["att"]["deterministic"] is True


def test_a_metric_only_some_runs_have_is_kept_when_two_do_and_dropped_when_one_does():
    runs = [result(), result(), result(), result()]
    for i, r in enumerate(runs):
        r["runtime"]["att"]["resources"] = {"cpu_cores": 0.6 + 0.01 * i} if i >= 2 else {}      # the harness gained it after the first two runs
    runs[0]["runtime"]["att"]["flight"] = {"cpu_pct_max": 50}                                    # only one run has this one
    doc, _ = baseline.build(runs)
    assert doc["metrics"]["runtime.att.res.cpu_cores"]["value"] == 0.625                        # median of 0.62 and 0.63, the two runs that have it
    assert "runtime.att.flight_cpu_pct" not in doc["metrics"]
    assert "runtime.att.fps_p10" in doc["metrics"]
