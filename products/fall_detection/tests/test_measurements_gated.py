"""The measurement ACs: harnesses that run where their prerequisites exist, and skip cleanly elsewhere.

ACs: AC-F1.2, AC-N1.1, AC-N2.1, AC-N3.1, AC-N4.1, AC-N5.1, AC-F29.1.
These are experiments on live infrastructure (Docker+GPU sidecar, the RTX 5090 deployment machine,
the fall corpus) — the unit-testable claim of each is "the harness exists, takes the parameter the
requirement names, and reports the numbers the requirement asks for"; the run itself is gated and
its evidence is recorded (Phase 1 testability audit). Nothing here fails on a dev box without a
GPU, a corpus or a recorded run.
"""
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

PRODUCT = Path(__file__).resolve().parents[1]
REPO = PRODUCT.parents[1]
CORPUS = Path(os.environ.get("FALL_CORPUS", r"F:\fall_detection\data"))
# where a machine that ran the gated measurements records their output (the Phase 3 evidence dir)
EVIDENCE = Path(os.environ["FALL_MEASUREMENTS"]) if os.environ.get("FALL_MEASUREMENTS") else None


def _gpu_name():
    if shutil.which("nvidia-smi") is None:
        return ""
    out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                         capture_output=True, text=True)
    return out.stdout.strip()


# ---------------------------------------------------------------- AC-F29.1 (golden baseline)
def test_the_golden_baseline_carries_the_pose_field_with_17_keypoints():
    baseline = REPO / "tests" / "baselines" / "golden-fall_detection.before.jsonl"
    if not baseline.is_file():
        pytest.skip("the golden run has not been recorded on this machine "
                    "(tools/engine_golden_run.sh needs Docker+GPU)")
    entries = [json.loads(line) for line in baseline.read_text(encoding="utf-8").splitlines() if line.strip()]
    poses = [json.loads(e["pose"]) if isinstance(e.get("pose"), str) else e.get("pose")
             for e in entries]
    poses = [p for p in poses if p]
    assert poses, "the baseline has no entry with a pose field"
    for p in poses:
        assert len(p["shape"]) == 2
        for item in p["items"]:
            assert len(item["kps"]) == 17
            assert all(len(kp) == 3 for kp in item["kps"])


# ---------------------------------------------------------------- AC-F1.2 (live GIE evidence)
def test_the_recorded_sidecar_run_names_the_pose_gie():
    if EVIDENCE is None:
        pytest.skip("set FALL_MEASUREMENTS to the directory holding the recorded sidecar run")
    logs = list(EVIDENCE.glob("*sidecar*.log")) + list(EVIDENCE.glob("*fall*log*"))
    assert logs, f"no recorded sidecar log in {EVIDENCE}"
    text = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in logs)
    assert "sgie_pose" in text                      # the engine names the GIE it built
    assert "interval" in text.lower()               # ...and its interval (AC-F1.2)


# ---------------------------------------------------------------- AC-N1.1 (GPU budget)
def test_the_gpu_budget_harness_exists_and_the_recorded_delta_names_the_pose_stage():
    harness = REPO / "tools" / "engine_gpu_budget.sh"
    assert harness.is_file(), "tools/engine_gpu_budget.sh is the requirement's own harness"
    if EVIDENCE is None:
        pytest.skip("set FALL_MEASUREMENTS to the directory holding the recorded budget runs")
    budget = EVIDENCE / "gpu_budget.txt"
    assert budget.is_file(), "the before/after budget was not recorded"
    text = budget.read_text(encoding="utf-8")
    assert "pose" in text.lower()
    # the AC's demanded numbers: each stage's busy% printed, and the pose delta recorded
    assert re.search(r"busy\s*%?", text, re.I), "no busy% column in the recorded budget"
    assert re.search(r"pose.*delta|delta.*pose", text, re.I | re.S), "no pose-stage delta recorded"


# ---------------------------------------------------------------- AC-N2.1 + AC-N4.1 (corpus replay)
def test_the_corpus_replay_harness_reports_the_action_path_and_skips_when_the_corpus_is_absent():
    import measure                                            # ModuleNotFoundError == expected red
    if not CORPUS.is_dir():
        # the requirement itself: the harness takes the corpus root as a parameter and skips
        report = measure.action_path_eval(CORPUS, interval=2)
        assert report["skipped"] is True
        pytest.skip(f"the corpus at {CORPUS} is not on this machine")
    reports = {}
    for interval in (2, 1):
        report = measure.action_path_eval(CORPUS, interval=interval)
        if report.get("skipped"):
            pytest.skip(report["reason"])
        assert {"precision", "recall", "false_alarms_per_hour", "action_path_share"} <= set(report)
        # AC-N4.1's attribution clause: the candidate is on the right camera and track. The
        # spatial gate can only be checked where the ground truth carries boxes; where it does
        # not, the report says how many matches could not be checked instead of claiming them.
        assert report["attributed"] == report["attribution_checked"]
        assert report["attributed"] + report["attribution_unavailable"] == report["true_positives"]
        reports[interval] = report
    if EVIDENCE is not None:                                  # the comparison is written down
        (EVIDENCE / "interval_sweep.json").write_text(json.dumps(reports, indent=2))


# ---------------------------------------------------------------- AC-N3.1 (decision-to-alert latency)
def test_the_onset_to_alert_latency_is_measured_end_to_end_on_replayed_footage():
    import measure
    if not CORPUS.is_dir():
        assert measure.latency(CORPUS)["skipped"] is True
        pytest.skip(f"the corpus at {CORPUS} is not on this machine")
    report = measure.latency(CORPUS)
    if report.get("skipped"):
        pytest.skip(report["reason"])
    assert "onset_to_alert_s" in report
    if not report["matched_falls"]:
        # nothing was detected, so there is no latency to state: the report says so rather than
        # claiming a zero-second alert
        assert report["onset_to_alert_s"] is None
        pytest.skip("no fall was matched on this corpus, so no latency was measured")
    assert report["onset_to_alert_s"] < 5.0                   # the clip window is ±5 s


# ---------------------------------------------------------------- AC-N5.1 (classifier cost, RTX 5090)
def test_the_classifier_cost_is_recorded_on_the_deployment_card():
    if "5090" not in _gpu_name():
        pytest.skip(f"this machine's GPU is {_gpu_name() or 'absent'} — ruling A9: N5 is answered "
                    "on the RTX 5090 deployment machine")
    import measure
    report = measure.classifier_cost(device="cuda:0", concurrent_tracks=(1, 4, 8))
    if report.get("skipped"):
        pytest.skip(report["reason"])
    assert {"ms_per_forward", "concurrent_tracks", "gpu_util"} <= set(report)
    if EVIDENCE is not None:
        (EVIDENCE / "classifier_timing.json").write_text(json.dumps(report, indent=2))
