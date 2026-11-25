# Offline Calibration & Verification Report (`products/concealment_detection`)

## 1. Dataset Splits & Leakage Isolation
- **Data Root:** `F:\IRIS\data\RetailS`
- **Stage 1 & 3 Negative Tuning Split (`RetailS_train` Days 1–8):** `618` files across days `0418, 0419, 0420, 0421, 0422, 0423, 0424, 0425` (`36` stratified files / `20.05h` span camera-hours evaluated in grid search).
- **Stage 2 & 3 Positive Tuning Split (`RetailS_test_staged`):** `624` clips (`622` positive events; `459` single-person events).
- **Stage 4 Locked Held-Out Positive Split (`RetailS_test_realworld` / `PoseLift/test`):** `47` clips (`43` positive events; `25` single-person events).
- **Stage 4 Locked Held-Out Negative Split (`RetailS_train` Days 9–11):** `324` files across days `0426, 0428, 0429` (`36` stratified files / `21.35h` span camera-hours).

---

## 2. Stage 1 — Proposed Per-Camera Shelf Zones
Stationary shopper dwell heatmaps (`propose_zones_from_dwell`) on `RetailS_train` Days 1–8 produced proposed shelf zones saved in [proposed_zones.yaml](file:///f:/IRIS/surveillance-new/products/concealment_detection/calibration/reports/proposed_zones.yaml):

| Camera | Proposed Zones | Polygon Bounds `(x1, y1) -> (x2, y2)` |
|---|---|---|
| `cam-1` | 2 | `proposed_shelf_1`: `(0.00, 0.07) -> (0.62, 1.00)`, `proposed_shelf_2`: `(0.47, 0.28) -> (1.00, 1.00)` |
| `cam-2` | 1 | `proposed_shelf_1`: `(0.00, 0.17) -> (1.00, 1.00)` |
| `cam-3` | 2 | `proposed_shelf_1`: `(0.07, 0.00) -> (1.00, 1.00)`, `proposed_shelf_2`: `(0.00, 0.68) -> (0.33, 1.00)` |
| `cam-4` | 1 | `proposed_shelf_1`: `(0.07, 0.07) -> (1.00, 1.00)` |
| `cam-5` | 1 | `proposed_shelf_1`: `(0.00, 0.07) -> (1.00, 1.00)` |
| `cam-6` | 1 | `proposed_shelf_1`: `(0.17, 0.00) -> (1.00, 1.00)` |

---

## 3. Stage 2 — Empirical Feature Quantiles (`RetailS_test_staged` GT=1 vs `RetailS_train` GT=0)

| Feature / Metric | Split | Count | p10 | p25 | Median (p50) | p75 | p90 |
|---|---|---:|---:|---:|---:|---:|---:|
| `event_duration_s` | Staged Positive (`gt=1`) | 250 | 1.2667 | 1.5333 | **2.0333** | 2.6667 | 3.3333 |
| `pre_event_lead_s` | Staged Positive (`gt=1`) | 250 | 0.6667 | 0.8 | **1.0** | 1.3333 | 1.6667 |
| `wrist_excursion_torsos` | Staged Positive (`gt=1`) | 9556 | 0.1866 | 0.3537 | **0.5925** | 0.9226 | 1.2177 |
| `wrist_excursion_torsos` | Normal Train (`gt=0`) | 17857 | 0.142 | 0.3129 | **0.5541** | 0.8885 | 1.1823 |
| `min_elbow_deg` | Staged Positive (`gt=1`) | 9556 | 52.1588 | 87.6215 | **114.593** | 141.2075 | 158.5513 |
| `min_elbow_deg` | Normal Train (`gt=0`) | 17857 | 26.5097 | 61.717 | **101.2217** | 147.1546 | 163.323 |
| `wrist_to_body_torsos` | Staged Positive (`gt=1`) | 9556 | 0.0909 | 0.1588 | **0.2469** | 0.3555 | 0.5267 |
| `wrist_to_body_torsos` | Normal Train (`gt=0`) | 17857 | 0.0967 | 0.1657 | **0.2451** | 0.3493 | 0.4879 |

---

## 4. Stage 3 — Calibration Grid Search & Operating Points (`RetailS_test_staged` + `RetailS_train` Days 1–8)

| Operating Point | `reach_excursion` | `max_elbow_deg` | `body_radius` | `min_hold_s` | `short_grace_s` | `emit_low_tier` | Staged Recall (All 622) | Staged Recall (Single-Person 461) | Train FA / Span Hour | Train Warning FA / Span Hour |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| **Baseline (Un-tuned Default)** | `0.35` | `135.0` | `0.55` | `0.35` | `1.2` | `True` | **88.8%** (552/622) | **87.2%** (400/459) | **102.69/h** | **70.47/h** |
| **Recommended Balanced (PoC Default)** | `0.32` | `140.0` | `0.5` | `0.35` | `1.5` | `False` | **86.2%** (536/622) | **84.5%** (388/459) | **91.12/h** | **91.12/h** |
| **High-Sensitivity (Max Recall)** | `0.25` | `150.0` | `0.6` | `0.25` | `1.5` | `True` | **96.8%** (602/622) | **96.5%** (443/459) | **121.14/h** | **96.45/h** |
| **Strict Low-FA (Conservative)** | `0.4` | `130.0` | `0.5` | `0.5` | `1.5` | `False` | **70.7%** (440/622) | **68.2%** (313/459) | **71.07/h** | **71.07/h** |

---

## 5. Stage 4 — Locked Held-Out Verification (`RetailS_test_realworld` + `RetailS_train` Days 9–11)

| Operating Point | Real-World Recall (All 43) | Real-World Recall (Single-Person 34) | Held-Out FA / Span Hour | Held-Out Warning FA / Span Hour | 95% Upper Bound FA / Span Hour |
|---|---:|---:|---:|---:|---:|
| **Baseline (Un-tuned Default)** | **69.8%** (30/43) | **64.0%** (16/25) | **92.69/h** | **64.54/h** | `96.19/h` |
| **Recommended Balanced (PoC Default)** | **69.8%** (30/43) | **64.0%** (16/25) | **81.68/h** | **81.68/h** | `84.97/h` |
| **High-Sensitivity (Max Recall)** | **88.4%** (38/43) | **84.0%** (21/25) | **106.32/h** | **85.99/h** | `110.06/h` |
| **Strict Low-FA (Conservative)** | **51.2%** (22/43) | **48.0%** (12/25) | **65.48/h** | **65.48/h** | `68.43/h` |

---

## 6. Recommended `detector:` Thresholds for `concealment_detection.yaml`

```yaml
detector:
  min_track_s: 0.25
  min_keypoints: 6
  keypoint_conf_threshold: 0.3
  min_box_height_px: 48.0
  min_torso_px: 25.0
  min_core_conf_enter: 0.3
  min_core_conf_exit: 0.35
  degrade_max_s: 1.0
  recover_s: 0.4
  lost_s: 1.5
  camera_stale_s: 3.0
  reid_max_gap_s: 1.5
  reid_max_dist_torsos: 1.2
  dwell_min_s: 1.6
  short_track_grace_s: 1.5
  reach_excursion_torsos: 0.32
  reach_window_s: 2.5
  body_zone_radius_torsos: 0.5
  bag_zone_radius_torsos: 0.5
  max_conceal_elbow_deg: 140.0
  min_hold_s: 0.35
  hidden_s: 0.4
  hidden_conf_drop_fraction: 0.45
  baseline_window_s: 2.0
  cooldown_s: 8.0
  emit_low_tier: false
```
