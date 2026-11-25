# Offline Calibration & Verification Report

Run fingerprint: `9c0c74134fd77aa1b9b18e817eb8d029ca85e1f5c74aa3c228b782c9b1b4783c`. Results apply only to the serialized settings, zones, sources and dataset identities in this run; older artifacts are not revalidated or rewritten.

## Splits
{
  "train_tune_days": [
    "0418",
    "0419",
    "0420",
    "0421",
    "0422",
    "0423",
    "0424",
    "0425"
  ],
  "train_holdout_days": [
    "0426",
    "0428",
    "0429"
  ],
  "train_tune_files": 618,
  "train_tune_sampled_files": 36,
  "train_tune_span_hours": 21.92333333333333,
  "train_holdout_files": 324,
  "train_holdout_sampled_files": 36,
  "train_holdout_span_hours": 18.082259259259256,
  "staged_clips": 312,
  "staged_test_clips": 312,
  "staged_events": 311,
  "staged_single_person_events": 229,
  "realworld_clips": 47,
  "realworld_events": 43,
  "realworld_single_person_events": 25
}

Staged development/test: frozen camera/class/person-count stratified 50/50. Train negatives: camera/day-stratified sample. Real-world is final evaluation only.

## Zone mode
- cam-1: shelf-limited; 2 proposed polygons.
- cam-2: shelf-limited; 2 proposed polygons.
- cam-3: shelf-limited; 2 proposed polygons.
- cam-4: shelf-limited; 1 proposed polygons.
- cam-5: shelf-limited; 1 proposed polygons.
- cam-6: shelf-limited; 1 proposed polygons.

## Empirical quantiles
Eligible single-person positive GT only; time-based production history and representative normal samples.
```json
{
  "positive_staged_gt": {
    "event_duration_s": {
      "count": 179,
      "p10": 1.2,
      "p25": 1.5667,
      "p50": 2.0,
      "p75": 2.5,
      "p90": 2.8133
    },
    "pre_event_lead_s": {
      "count": 179,
      "p10": 0.6,
      "p25": 0.8,
      "p50": 1.0,
      "p75": 1.2667,
      "p90": 1.4
    },
    "wrist_excursion_torsos": {
      "count": 5277,
      "p10": 0.2277,
      "p25": 0.3587,
      "p50": 0.5179,
      "p75": 0.7438,
      "p90": 0.9583
    },
    "min_elbow_deg": {
      "count": 5277,
      "p10": 76.9374,
      "p25": 97.984,
      "p50": 123.0206,
      "p75": 151.0112,
      "p90": 168.1475
    },
    "wrist_to_body_torsos": {
      "count": 5277,
      "p10": 0.1161,
      "p25": 0.198,
      "p50": 0.2891,
      "p75": 0.4648,
      "p90": 1.5245
    },
    "active_wrist_conf": {
      "count": 5277,
      "p10": 0.3902,
      "p25": 0.6528,
      "p50": 0.791,
      "p75": 0.8462,
      "p90": 0.8789
    }
  },
  "negative_normal_train": {
    "wrist_excursion_torsos": {
      "count": 12980,
      "p10": 0.1442,
      "p25": 0.2472,
      "p50": 0.424,
      "p75": 0.624,
      "p90": 0.8656
    },
    "min_elbow_deg": {
      "count": 12980,
      "p10": 37.5859,
      "p25": 71.775,
      "p50": 109.9253,
      "p75": 152.5612,
      "p90": 171.0487
    },
    "wrist_to_body_torsos": {
      "count": 12980,
      "p10": 0.1506,
      "p25": 0.2338,
      "p50": 0.3475,
      "p75": 0.5827,
      "p90": NaN
    },
    "active_wrist_conf": {
      "count": 12980,
      "p10": 0.3079,
      "p25": 0.434,
      "p50": 0.6709,
      "p75": 0.8306,
      "p90": 0.8765
    }
  }
}
```

## Development objective
single-person recall - 0.035 * normal-train FA/span-hour; canonical settings tie-break.

## Development
| Point | Recall all | Recall single-person | Labeled-split FP | Normal labeled-clip FP | Train FA/span-hour | Train FA/active-hour |
|---|---|---|---|---|---|---|
| Baseline (Current Config) | 127/311 | 89/229 | 4 | 0 | 24.312 | 170.107 |
| Recommended Balanced (Development Selected) | 118/311 | 84/229 | 1 | 0 | 20.024 | 140.107 |
| High-Sensitivity (Max Recall) | 184/311 | 128/229 | 10 | 0 | 38.954 | 272.555 |
| Strict Low-FA (Conservative) | 151/311 | 104/229 | 6 | 0 | 26.547 | 185.746 |

## Frozen real-world / train holdout
| Point | Recall all | Recall single-person | Labeled-split FP | Normal labeled-clip FP | Train FA/span-hour | Train FA/active-hour |
|---|---|---|---|---|---|---|
| Baseline (Current Config) | 6/43 | 4/25 | 4 | 3 | 32.573 | 148.626 |
| Recommended Balanced (Development Selected) | 7/43 | 5/25 | 4 | 3 | 25.882 | 118.093 |
| High-Sensitivity (Max Recall) | 13/43 | 6/25 | 7 | 5 | 53.478 | 244.009 |
| Strict Low-FA (Conservative) | 9/43 | 5/25 | 4 | 3 | 36.057 | 164.523 |

## Frozen staged test
| Point | Recall all | Recall single-person | Labeled-split FP | Normal labeled-clip FP | Train FA/span-hour | Train FA/active-hour |
|---|---|---|---|---|---|---|
| Baseline (Current Config) | 143/311 | 104/230 | 2 | 0 | 0.000 | 0.000 |
| Recommended Balanced (Development Selected) | 116/311 | 86/230 | 0 | 0 | 0.000 | 0.000 |
| High-Sensitivity (Max Recall) | 209/311 | 152/230 | 7 | 1 | 0.000 | 0.000 |
| Strict Low-FA (Conservative) | 162/311 | 119/230 | 3 | 0 | 0.000 | 0.000 |

## Exact recommended development-selected settings
```yaml
detector:
  at_body_requires_flex: true
  bag_zone_radius_torsos: 0.5
  baseline_window_s: 2.0
  body_zone_radius_torsos: 0.5
  camera_stale_s: 3.0
  cooldown_s: 8.0
  cooldown_scope: track
  degrade_max_s: 1.0
  dwell_min_s: 1.6
  emit_low_tier: false
  exit_min_hidden_frac: 0.0
  fast_dropin_min_hold_s: 0.0
  hand_transfer: false
  hand_transfer_meet_torsos: 0.35
  hidden_conf_drop_fraction: 0.45
  hidden_s: 0.4
  hold_flicker_grace_s: 0.0
  insufficient_grace_s: 0.0
  inward_min_torsos: 0.12
  keypoint_conf_threshold: 0.3
  lost_s: 1.5
  max_conceal_elbow_deg: 130.0
  min_box_height_px: 48.0
  min_core_conf_enter: 0.3
  min_core_conf_exit: 0.35
  min_hold_s: 0.5
  min_keypoints: 6
  min_torso_px: 25.0
  min_track_s: 0.25
  partial_min_near_body_s: 0.0
  reach_excursion_torsos: 0.4
  reach_expiry_s: 0.0
  reach_seed_max_speed_torsos_s: 0.6
  reach_seeds_dwell: false
  reach_window_s: 2.5
  recover_s: 0.4
  reid_max_dist_torsos: 1.2
  reid_max_gap_s: 1.5
  sequence_max_gap_s: 0.5
  short_track_grace_s: 1.5
  spatial_hidden_s: 0.0
  zone_exit_grace_s: 0.0
  zone_persist_across_shelves: false

```