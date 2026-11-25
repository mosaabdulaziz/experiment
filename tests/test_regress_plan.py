"""`regress plan`: changed files -> tier -> configs -> the exact commands (tools/regress/plan.py, benchmarks/tiers.json)."""
import json
import os
import shutil
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

from regress import cli, plan  # noqa: E402

RULES = plan.load_rules(os.path.join(os.path.dirname(__file__), "..", "benchmarks", "tiers.json"))


def test_docs_and_frontend_changes_need_no_gpu():
    p = plan.classify(["docs/DESIGN.md", "frontend/src/App.tsx", "README.md", "app/main.py"], RULES)
    assert p["tier"] == 0 and p["configs"] == []
    assert [s["what"] for s in plan.commands(p)] == ["unit suites and build sizes against the baseline"]


def test_a_role_module_change_runs_that_roles_replay_and_ceiling_only():
    p = plan.classify(["perception/modules/ppe/detector.py", "perception/modules/ppe/README.md"], RULES)
    assert p["tier"] == 1 and p["configs"] == ["ppe"]
    cmds = [s["cmd"] for s in plan.commands(p)]
    assert any("--configs ppe" in c and "run --only golden" in c for c in cmds) and any("capacity --configs ppe --cameras 1" in c for c in cmds)
    assert not any("soak" in c for c in cmds)


def test_several_roles_unite_their_configs_and_the_highest_tier_wins():
    p = plan.classify(["perception/modules/ppe/a.py", "perception/modules/fire_alerts/b.py", "docs/x.md"], RULES)
    assert p["tier"] == 1 and p["configs"] == ["fire", "ppe"]
    p2 = plan.classify(["perception/modules/ppe/a.py", "perception/engine/probe.py"], RULES)
    assert p2["tier"] == 2 and p2["configs"] == plan.ALL_CONFIGS


def test_the_shared_engine_and_the_images_run_everything_and_images_add_a_note():
    p = plan.classify(["perception/engine/gst_build.py"], RULES)
    assert p["tier"] == 2 and p["notes"] == []
    q = plan.classify(["perception/pipeline/Dockerfile"], RULES)
    assert q["tier"] == 2 and any("re-recording the baseline" in n for n in q["notes"])


def test_recording_and_incident_code_adds_the_soak_smoke():
    p = plan.classify(["perception/engine/recording.py"], RULES)
    assert p["tier"] == 3
    steps = plan.commands(p)
    assert any("soak-compare" in s["cmd"] for s in steps) and sum(s["minutes"] for s in steps) > 75


def test_a_file_no_rule_matches_is_a_shared_change_and_says_so():
    p = plan.classify(["somewhere/new/thing.py"], RULES)
    assert p["tier"] == 2 and "no rule matches" in p["files"][0]["why"]


def test_a_product_without_a_golden_only_runs_the_unit_suites_and_notes_it():
    p = plan.classify(["products/concealment_detection/main.py"], RULES)
    assert p["tier"] == 0 and any("no golden" in n for n in p["notes"])


def test_proximity_has_a_golden_replay_and_a_ceiling_now_and_says_what_it_needs():
    p = plan.classify(["perception/modules/proximity/zones.py"], RULES)
    assert p["tier"] == 1 and p["configs"] == ["proximity"] and any("proximity-models" in n for n in p["notes"])
    cmds = [s["cmd"] for s in plan.commands(p)]
    assert any("--configs proximity" in c and "run --only golden" in c for c in cmds) and any("capacity --configs proximity" in c for c in cmds)
    assert "proximity" in plan.ALL_CONFIGS and "proximity" in plan.CEILING_CONFIGS


def test_golden_config_files_map_to_their_config():
    assert plan.classify(["tests/configs/golden-barber-faces.yaml"], RULES)["configs"] == ["barber-faces"]
    assert plan.classify(["tests/configs/golden-barber.yaml"], RULES)["configs"] == ["barber"]
    assert plan.classify(["tests/configs/golden-fall_detection.yaml"], RULES)["configs"] == ["fall_detection"]


def test_durations_add_up_for_the_default_costs():
    p = plan.classify(["perception/modules/ppe/a.py"], RULES)
    mins = [s["minutes"] for s in plan.commands(p)]
    assert mins == [plan.UNIT_BUILD_MIN, plan.HARNESS_OVERHEAD_MIN + plan.GOLDEN_MIN_PER_CONFIG, plan.CEILING_MIN_PER_CONFIG]


def test_every_config_a_rule_names_exists_and_every_rule_has_a_valid_tier():
    for r in RULES["rules"]:
        assert r["tier"] in (0, 1, 2, 3) and r["match"], r
        if isinstance(r.get("configs"), list):
            assert set(r["configs"]) <= set(plan.ALL_CONFIGS), r
        if r["tier"] == 1:
            assert r["configs"], r


def test_the_command_with_files_prints_the_plan_and_the_json_form_parses(capsys):
    assert cli.main(["plan", "perception/modules/ppe/a.py"]) == 0
    out = capsys.readouterr().out
    assert "tier 1" in out and "--configs ppe" in out and "total about" in out
    assert cli.main(["plan", "--json", "docs/a.md"]) == 0
    assert json.loads(capsys.readouterr().out)["tier"] == 0


def test_without_files_it_reads_the_diff_against_the_base(monkeypatch, capsys):
    monkeypatch.setattr(plan, "changed_files", lambda base, repo=".": ["perception/engine/probe.py"] if base == "origin/main" else [])
    assert cli.main(["plan"]) == 0 and "tier 2" in capsys.readouterr().out
    assert cli.main(["plan", "--base", "nothing"]) == 0 and "nothing to check" in capsys.readouterr().out


@pytest.mark.skipif(shutil.which("git") is None, reason="needs the git binary (the CI image has none)")
def test_changed_files_are_this_branchs_own_not_what_landed_on_main_since(tmp_path):
    import subprocess

    def git(*a):
        subprocess.run(["git", "-C", str(tmp_path), "-c", "user.email=t@t", "-c", "user.name=t", *a], check=True, capture_output=True)
    git("init", "-q", "-b", "main")
    (tmp_path / "base.txt").write_text("1")
    git("add", "-A"); git("commit", "-q", "-m", "base")
    git("checkout", "-q", "-b", "feat")
    (tmp_path / "mine.txt").write_text("1")
    git("add", "-A"); git("commit", "-q", "-m", "mine")
    git("checkout", "-q", "main")
    (tmp_path / "theirs.txt").write_text("1")                      # lands on main after the branch was cut
    git("add", "-A"); git("commit", "-q", "-m", "theirs")
    git("checkout", "-q", "feat")
    (tmp_path / "wip.txt").write_text("x")                          # a tracked file edited, not yet committed
    (tmp_path / "base.txt").write_text("2")
    assert plan.changed_files("main", str(tmp_path)) == ["base.txt", "mine.txt"]


def test_without_the_git_binary_plan_says_so_instead_of_a_traceback(monkeypatch):
    def no_git(*a, **k):
        raise FileNotFoundError(2, "No such file or directory", "git")
    monkeypatch.setattr(plan.subprocess, "run", no_git)
    with pytest.raises(SystemExit) as e:
        plan.changed_files("origin/main", ".")
    assert "git" in str(e.value) and "installed" in str(e.value)


def test_top_level_files_the_replays_do_not_use_need_no_gpu():
    assert plan.classify([".gitignore", "docker-compose.shelf-live.yml", "data/catalog.yaml"], RULES)["tier"] == 0
    assert plan.classify(["docker-compose.yml"], RULES)["tier"] == 2
    assert plan.classify(["mtmc/audit_vehicle_detector.py"], RULES)["configs"] == ["multicam"]
