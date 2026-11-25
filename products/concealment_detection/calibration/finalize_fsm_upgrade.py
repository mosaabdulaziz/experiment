"""Validate frozen FSM evidence and perform the single held-out calibration run."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Callable

PRODUCT_DIR = Path(__file__).resolve().parents[1]
if str(PRODUCT_DIR) not in sys.path:
    sys.path.insert(0, str(PRODUCT_DIR))

from calibration.calibrate import run_calibration
from calibration.dataset_utils import canonical_hash
from calibration.experiment_fsm_upgrade import VARIANTS, adoption_verdict, legacy_settings_hash
from config import CONFIG_PATH, load_config


DEMO_DIR = Path(__file__).resolve().parent / "reports" / "demo"
RUN_DIR = Path("F:/IRIS/scratch/fsm-final-heldout")
SOLO_REPORT = "fsm_upgrade_ablation.json"
LONG_LOSS_REPORT = "long_loss/fsm_upgrade_ablation.json"
ZONE_EXIT_REPORT = "zone_exit/fsm_upgrade_ablation.json"
REVIEWED_REPORT = "reviewed/fsm_upgrade_ablation.json"
QUALITY_REPORT = "quality_fixed/fsm_upgrade_ablation.json"

_LONG_LOSS_VARIANTS = {"baseline", "long_loss", "gap_2.0_long_loss"}
_ZONE_EXIT_VARIANTS = {"baseline", "zone_exit_0.2", "zone_exit_0.4"}
_REVIEWED_VARIANTS = {
    "baseline",
    "quality_grace_0.3",
    "quality_grace_0.6",
    "spatial_hidden_0.7",
    "spatial_hidden_1.0",
    "spatial_hidden_1.5",
    "hand_transfer",
    "hand_transfer_meet_0.5",
}
_QUALITY_VARIANTS = {
    "baseline",
    "quality_grace_0.3",
    "quality_grace_0.6",
    "hand_transfer",
    "hand_transfer_meet_0.5",
}
_REPORT_SPECS = {
    SOLO_REPORT: (set(VARIANTS) - _LONG_LOSS_VARIANTS - _ZONE_EXIT_VARIANTS) | {"baseline"},
    LONG_LOSS_REPORT: _LONG_LOSS_VARIANTS,
    ZONE_EXIT_REPORT: _ZONE_EXIT_VARIANTS,
    REVIEWED_REPORT: _REVIEWED_VARIANTS,
    QUALITY_REPORT: _QUALITY_VARIANTS,
}
_STANDARD_REPORTS = {SOLO_REPORT, ZONE_EXIT_REPORT, REVIEWED_REPORT, QUALITY_REPORT}
_DATA_IDENTITY_KEYS = ("zones_sha256", "positives", "negatives", "tolerance_s")
_PRELOAD_DIAGNOSTICS = {"by_stage_tier", "by_tier", "reset_counts"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def deployed_settings_hashes() -> tuple[str, str]:
    settings = load_config(CONFIG_PATH, env={}).detector
    return canonical_hash(asdict(settings)), legacy_settings_hash(settings)


def load_report_bundle() -> dict[str, dict]:
    return {
        name: json.loads((DEMO_DIR / name).read_text(encoding="utf-8"))
        for name in _REPORT_SPECS
    }


def validate_report_bundle(
    reports: dict[str, dict],
    settings_hash: str,
    legacy_hash: str,
) -> dict:
    _require(set(reports) == set(_REPORT_SPECS), "authoritative report bundle is incomplete")
    reference_identity = reports[SOLO_REPORT]["identity"]
    reference_metrics = reports[SOLO_REPORT]["variants"]["baseline"]["metrics"]

    for name, expected_variants in _REPORT_SPECS.items():
        report = reports[name]
        identity = report["identity"]
        _require(identity.get("settings_sha256") == settings_hash,
                 f"{name}: deployed settings identity changed")
        _require(identity.get("legacy_settings_sha256") == legacy_hash,
                 f"{name}: normalized settings identity changed")
        _require(set(report["variants"]) == expected_variants,
                 f"{name}: variant contract changed")
        for variant_name, variant in report["variants"].items():
            _require(variant["overrides"] == VARIANTS[variant_name],
                     f"{name}/{variant_name}: override contract changed")
            if variant_name == "baseline":
                _require(variant["settings_sha256"] == settings_hash,
                         f"{name}: baseline settings identity changed")
            expected_adoption = adoption_verdict(variant["vs_baseline"])
            _require(variant["adoption"] == expected_adoption,
                     f"{name}/{variant_name}: adoption verdict is inconsistent")
            _require(not expected_adoption["adopt"],
                     f"{name}/{variant_name}: adoptable setting is not frozen")

        for key in _DATA_IDENTITY_KEYS:
            _require(identity.get(key) == reference_identity.get(key),
                     f"{name}: development data identity changed at {key}")
        expected_flush = 2.6 if name == LONG_LOSS_REPORT else 1.6
        _require(identity.get("gap_flush_s") == expected_flush,
                 f"{name}: preload identity changed")

        baseline_metrics = report["variants"]["baseline"]["metrics"]
        if name in _STANDARD_REPORTS:
            _require(baseline_metrics == reference_metrics,
                     f"{name}: standard baseline metrics changed")
        else:
            comparable = {k: v for k, v in baseline_metrics.items()
                          if k not in _PRELOAD_DIAGNOSTICS}
            reference_comparable = {k: v for k, v in reference_metrics.items()
                                    if k not in _PRELOAD_DIAGNOSTICS}
            _require(comparable == reference_comparable,
                     f"{name}: long-loss baseline outcomes changed")

    return reference_metrics


def load_frozen_evidence() -> dict:
    settings = load_config(CONFIG_PATH, env={}).detector
    settings_hash = canonical_hash(asdict(settings))
    legacy_hash = legacy_settings_hash(settings)
    reports = load_report_bundle()
    validate_report_bundle(reports, settings_hash, legacy_hash)
    return {
        "decision": "No upgrade knob passes all development adoption gates",
        "settings": asdict(settings),
        "settings_sha256": settings_hash,
        "legacy_settings_sha256": legacy_hash,
        "yaml_sha256": hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
        "ablation_reports": [
            {
                "path": str(DEMO_DIR / name),
                "sha256": hashlib.sha256((DEMO_DIR / name).read_bytes()).hexdigest(),
                "identity": reports[name]["identity"],
            }
            for name in _REPORT_SPECS
        ],
    }


def claim_run(run_dir: Path, manifest: dict) -> None:
    try:
        run_dir.mkdir(parents=False, exist_ok=False)
    except FileExistsError as exc:
        raise RuntimeError(f"held-out calibration already claimed at {run_dir}") from exc
    (run_dir / "started.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def run_frozen_calibration(
    run_dir: Path = RUN_DIR,
    calibration_runner: Callable[[], dict] = run_calibration,
) -> dict:
    manifest = load_frozen_evidence()
    runtime_settings = load_config().detector
    _require(asdict(runtime_settings) == manifest["settings"],
             "runtime detector settings differ from the frozen YAML")
    claim_run(run_dir, manifest)
    print(f"[Freeze] settings={manifest['settings_sha256']}; ledger={run_dir}", flush=True)

    result = calibration_runner()
    baseline = result["stage3_grid_search"]["selected_operating_points"][0]
    _require(baseline["settings_sha256"] == manifest["settings_sha256"],
             "Stage 3 baseline differs from the frozen profile")
    _require(hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest() == manifest["yaml_sha256"],
             "deployed YAML changed during held-out calibration")
    completed = {
        "output_dir": result["output_dir"],
        "settings_sha256": manifest["settings_sha256"],
        "legacy_settings_sha256": manifest["legacy_settings_sha256"],
        "yaml_sha256": manifest["yaml_sha256"],
    }
    (run_dir / "completed.json").write_text(json.dumps(completed, indent=2), encoding="utf-8")
    print(f"[Verified] output={result['output_dir']}; deployed YAML unchanged", flush=True)
    return result


def main() -> None:
    run_frozen_calibration()


if __name__ == "__main__":
    main()
