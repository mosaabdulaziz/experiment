"""Unit tests for tools/tracker_ab_eval.py (Tiers 1-4)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.tracker_ab_eval import (
    compute_intrinsic_metrics,
    evaluate_ab,
    evaluate_product_oracle,
    parse_jsonl_recording,
    parse_telemetry_log,
    render_markdown,
)


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def _meta_entry(cam: str, fnum: int, pts: float, dets: list, gids: dict | None = None) -> dict:
    fields = {"frame_num": fnum, "pts": round(pts, 3), "dets": dets}
    if gids is not None:
        fields["global_ids"] = gids
    return {"key": f"iris:meta:{cam}", "fields": fields}


def test_tier1_telemetry_parsing():
    log = "\n".join([
        "[cam-1] 24.0 fps",
        "[telemetry] gpu.infer.pgie n=100 total=1500.0ms busy=30.0% avg=15.00ms max=22.00ms thread=gpu window=5.0s",
        "[telemetry] gpu.tracker n=100 total=800.0ms busy=16.0% avg=8.00ms max=14.00ms thread=gpu window=5.0s",
        "[cam-1] 25.0 fps",
        "[telemetry] gpu.infer.pgie n=125 total=1500.0ms busy=30.0% avg=12.00ms max=16.00ms thread=gpu window=5.0s",
        "[telemetry] gpu.tracker n=125 total=250.0ms busy=5.0% avg=2.00ms max=3.50ms thread=gpu window=5.0s",
        "[cam-1] 25.2 fps",
        "[cam-1] 25.1 fps",
        "[telemetry] gauge qos_dropped_total=0",
        "[telemetry] gauge publisher_drops=0",
        "[telemetry] gauge source_stall_episodes_total_cam-1=0",
    ])
    rt = parse_telemetry_log(log)
    # First window dropped -> steady window only has 125 batches, avg=2.00ms
    assert rt["gpu_tracker"]["avg_ms"] == pytest.approx(2.0, abs=1e-4)
    assert rt["gpu_tracker"]["max_ms"] == pytest.approx(3.5, abs=1e-4)
    assert rt["gpu_tracker"]["busy_pct"] == pytest.approx(5.0, abs=1e-4)
    assert rt["overall_fps"] == pytest.approx(25.15, abs=0.05)
    assert rt["total_drops"] == 0
    assert rt["crash_count"] == 0


def test_tier2_intrinsic_metrics_penalize_fragmented_jittery_tracks(tmp_path: Path):
    # Smooth continuous single track (ID 1) over 90 frames @ 30fps (3.0s)
    smooth_rows = [
        _meta_entry("cam-1", f, f / 30.0, [["person", 0.9, 1, 100.0 + 4.0 * f, 200.0, 60.0, 140.0]])
        for f in range(90)
    ]
    # Fragmented, breathing, jittery tracks (ID switches at f=30, f=60 + short 5-frame flicker ID 99)
    frag_rows = []
    for f in range(90):
        if f < 30:
            oid = 1
        elif f < 60:
            oid = 2
        else:
            oid = 3
        jitter_y = 6.0 if (f % 2 == 0) else -6.0
        w = 60.0 * (1.25 if f % 2 == 0 else 0.78)
        dets = [["person", 0.9, oid, 100.0 + 4.0 * f, 200.0 + jitter_y, w, 140.0]]
        if 10 <= f < 15:
            dets.append(["person", 0.7, 99, 500.0, 300.0, 50.0, 120.0])
        frag_rows.append(_meta_entry("cam-1", f, f / 30.0, dets))

    smooth_m = compute_intrinsic_metrics(parse_jsonl_recording(_write_jsonl(tmp_path / "smooth.jsonl", smooth_rows)))
    frag_m = compute_intrinsic_metrics(parse_jsonl_recording(_write_jsonl(tmp_path / "frag.jsonl", frag_rows)))

    assert smooth_m["unique_ids"] == 1
    assert smooth_m["flicker_rate_pct"] == 0.0
    assert smooth_m["stitchable_breaks"] == 0
    assert frag_m["unique_ids"] == 4
    assert frag_m["unique_id_inflation"] > smooth_m["unique_id_inflation"]
    assert frag_m["flicker_rate_pct"] > smooth_m["flicker_rate_pct"]
    assert frag_m["stitchable_breaks"] == 2
    assert frag_m["center_jerk_rms"] > smooth_m["center_jerk_rms"]
    assert frag_m["log_area_volatility"] > smooth_m["log_area_volatility"]


def test_tier3_product_oracles_and_gates(tmp_path: Path):
    # 1. Zones oracle: NvDCF switches ID inside zone and double-fires alert; VEMA-3D fires once.
    nvdcf_zones = _write_jsonl(tmp_path / "z_a.jsonl", [
        _meta_entry("cam-1", 1, 1.0, [["person", 0.9, 10, 300.0, 200.0, 60.0, 140.0]]),
        {"key": "iris:zone_alert:cam-1", "fields": {"pts": "1.000", "object_id": 10, "zone": "z1"}},
        _meta_entry("cam-1", 30, 2.0, [["person", 0.9, 11, 304.0, 200.0, 60.0, 140.0]]),
        {"key": "iris:zone_alert:cam-1", "fields": {"pts": "2.000", "object_id": 11, "zone": "z1"}},
    ])
    vema_zones = _write_jsonl(tmp_path / "z_b.jsonl", [
        _meta_entry("cam-1", 1, 1.0, [["person", 0.9, 10, 300.0, 200.0, 60.0, 140.0]]),
        {"key": "iris:zone_alert:cam-1", "fields": {"pts": "1.000", "object_id": 10, "zone": "z1"}},
        _meta_entry("cam-1", 30, 2.0, [["person", 0.9, 10, 304.0, 200.0, 60.0, 140.0]]),
    ])
    z_report = evaluate_ab(nvdcf_zones, vema_zones, "zones")
    assert z_report["oracle"]["nvdcf"]["duplicate_retriggers"] == 1
    assert z_report["oracle"]["vema3d"]["duplicate_retriggers"] == 0
    assert z_report["oracle"]["passed"] is True
    assert z_report["gates"]["overall_verdict"] == "GO"
    assert "VEMA-3D Migration Scorecard" in render_markdown(z_report)

    # 2. Multicam oracle: plates & global IDs.
    mc_a = parse_jsonl_recording(_write_jsonl(tmp_path / "mc_a.jsonl", [
        _meta_entry("cam-01", 1, 1.0, [["vehicle", 0.9, 1, 100, 100, 120, 80]], {"1": 101}),
        _meta_entry("cam-01", 2, 1.1, [["vehicle", 0.9, 2, 105, 100, 120, 80]], {"2": 102}),
        {"key": "iris:mtmc:plates", "fields": {"global_id": 101, "text": "ABC123"}},
    ]))
    mc_b = parse_jsonl_recording(_write_jsonl(tmp_path / "mc_b.jsonl", [
        _meta_entry("cam-01", 1, 1.0, [["vehicle", 0.9, 1, 100, 100, 120, 80]], {"1": 101}),
        _meta_entry("cam-01", 2, 1.1, [["vehicle", 0.9, 1, 105, 100, 120, 80]], {"1": 101}),
        {"key": "iris:mtmc:plates", "fields": {"global_id": 101, "text": "ABC123"}},
        {"key": "iris:mtmc:plates", "fields": {"global_id": 101, "text": "XYZ789"}},
    ]))
    mc_oracle = evaluate_product_oracle("multicam", mc_a, mc_b)
    assert mc_oracle["passed"] is True
    assert mc_oracle["vema3d"]["unique_local_ids"] < mc_oracle["nvdcf"]["unique_local_ids"]

    # 3. Attendance oracle: named employees & fewer session splits.
    att_a = parse_jsonl_recording(_write_jsonl(tmp_path / "att_a.jsonl", [
        {"key": "iris:face_rec:cam-1", "fields": {"pts": "1.0", "face_rec": [{"name": "Alice", "parent_id": 1, "confirmed": True}]}},
        {"key": "iris:face_rec:cam-1", "fields": {"pts": "5.0", "face_rec": [{"name": "Alice", "parent_id": 2, "confirmed": True}]}},
    ]))
    att_b_ok = parse_jsonl_recording(_write_jsonl(tmp_path / "att_b_ok.jsonl", [
        {"key": "iris:face_rec:cam-1", "fields": {"pts": "1.0", "face_rec": [{"name": "Alice", "parent_id": 1, "confirmed": True}]}},
        {"key": "iris:face_rec:cam-1", "fields": {"pts": "5.0", "face_rec": [{"name": "Alice", "parent_id": 1, "confirmed": True}]}},
    ]))
    att_b_missing = parse_jsonl_recording(_write_jsonl(tmp_path / "att_b_miss.jsonl", [
        {"key": "iris:face_rec:cam-1", "fields": {"pts": "1.0", "face_rec": [{"name": "unknown", "parent_id": 1, "confirmed": False}]}},
    ]))
    assert evaluate_product_oracle("attendance", att_a, att_b_ok)["passed"] is True
    assert evaluate_product_oracle("attendance", att_a, att_b_ok)["vema3d"]["session_splits"] == 0
    assert evaluate_product_oracle("attendance", att_a, att_b_missing)["passed"] is False


def test_tier4_efficiency_gate_rejects_slower_tracker(tmp_path: Path):
    rows = _write_jsonl(tmp_path / "base.jsonl", [
        _meta_entry("cam-1", f, f / 30.0, [["person", 0.9, 1, 200.0 + f, 200.0, 60.0, 140.0]])
        for f in range(40)
    ])
    fast_log = tmp_path / "fast.log"
    slow_log = tmp_path / "slow.log"
    fast_log.write_text(
        "[cam-1] 25.0 fps\n"
        "[telemetry] gpu.tracker n=50 total=100.0ms busy=2.0% avg=2.00ms max=3.00ms thread=gpu window=5.0s\n"
        "[telemetry] gpu.tracker n=50 total=100.0ms busy=2.0% avg=2.00ms max=3.00ms thread=gpu window=5.0s\n",
        encoding="utf-8",
    )
    slow_log.write_text(
        "[cam-1] 25.0 fps\n"
        "[telemetry] gpu.tracker n=50 total=600.0ms busy=12.0% avg=12.00ms max=18.00ms thread=gpu window=5.0s\n"
        "[telemetry] gpu.tracker n=50 total=600.0ms busy=12.0% avg=12.00ms max=18.00ms thread=gpu window=5.0s\n",
        encoding="utf-8",
    )
    report = evaluate_ab(rows, rows, "attendance", log_nvdcf_path=fast_log, log_vema3d_path=slow_log)
    assert report["gates"]["efficiency_checked"] is True
    assert report["gates"]["efficiency_gate"] is False
    assert report["gates"]["overall_verdict"] == "NO_GO"
