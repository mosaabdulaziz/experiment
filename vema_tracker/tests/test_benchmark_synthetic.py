"""End-to-end synthetic SOTA benchmark and ablation tests."""

from __future__ import annotations

import numpy as np

from vema_tracker.baselines import TRACKER_NAMES
from vema_tracker.core.config import VEMAConfig
from vema_tracker.datasets.synthetic import generate_synthetic_suite
from vema_tracker.eval.ablation import ABLATIONS, run_ablation
from vema_tracker.eval.runner import run_tracker_on_sequence


def test_all_trackers_on_synthetic_suite():
    seqs = generate_synthetic_suite(seed=42)
    cfg = VEMAConfig()

    results_by_tracker: dict[str, dict[str, dict]] = {}
    for tname in TRACKER_NAMES:
        results_by_tracker[tname] = {}
        for seq in seqs:
            _, metrics, ms_pf = run_tracker_on_sequence(tname, seq, config=cfg)
            d = metrics.to_dict()
            assert 0.0 <= d["HOTA"] <= 100.0
            assert 0.0 <= d["AssA"] <= 100.0
            assert 0.0 <= d["IDF1"] <= 100.0
            assert np.isfinite(d["MOTA"])
            assert ms_pf >= 0.0
            results_by_tracker[tname][seq.name] = d

    # VEMA should match or beat SORT on partial_occlusion_truncation (IDSW and AssA)
    vema_occ = results_by_tracker["vema"]["partial_occlusion_truncation"]
    sort_occ = results_by_tracker["sort"]["partial_occlusion_truncation"]
    assert vema_occ["IDSW"] <= sort_occ["IDSW"]
    assert vema_occ["AssA"] >= sort_occ["AssA"]
    assert vema_occ["IDF1"] >= 90.0


def test_ablation_runner_synthetic():
    rows = run_ablation("synthetic", verbose=False)
    assert len(rows) == len(ABLATIONS)
    assert rows[0]["variant"] == "full_vema"
    assert all(np.isfinite(r["HOTA"]) for r in rows)
