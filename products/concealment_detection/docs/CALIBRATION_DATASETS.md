# Calibration Datasets: Verified Inspection of `RetailS.zip` and `PoseLift.zip`

**Status:** verified against the actual archives at `F:\IRIS\data\RetailS.zip` and `F:\IRIS\data\PoseLift.zip` on 2026-11-02 using `inspect_pose_datasets.py` and targeted read-only probes. Section 4's split plan was re-checked on 2026-11-05 against the code that now implements it (`calibration/dataset_utils.py`, `calibration/calibrate.py`) and updated on 2026-11-13 with the locked held-out calibration run `calibration/reports/run-1794576781479912800/`.
**Applies to:** [`HIGH_LEVEL_DESIGN.md`](HIGH_LEVEL_DESIGN.md) (sections 6, 13b, 14, 18).

---

## 1. Bottom-line verdict

**Yes — `RetailS.zip` and `PoseLift.zip` are sufficient to build and calibrate the offline pose-first detector (false-alarm rate per camera-hour, shelf-zone dwell proposals, and kinematic recall), provided we make four specific adjustments to our adapter and state machine that only showed up when inspecting the actual files:**

1. **Do not treat `PoseLift/test` and `RetailS_test_realworld` as two separate datasets.** All 47 pose JSONs and all 47 `.npy` frame masks in `RetailS_test_realworld` and `PoseLift/test` are **byte-for-byte identical** (same filenames, identical MD5 hashes, same 43 real shoplifting events across 41 positive and 6 normal clips). Treating them as separate sets would double-count or leak test data.
2. **Swap `(col0, col1)` to `(x = col1, y = col0)` in the dataset adapter.** Both datasets store each keypoint triple as `[y, x, conf]` (`col0` in `[0, 1080]` increasing from nose to ankle; `col1` in `[0, 1920]`), contrary to the RetailS README's `(x, y, conf)` description. Reading them unswapped rotates every person by 90 degrees.
3. **Allow trimmed/short-lead-in tracks to reach `CANDIDATE` without a multi-second prior `DWELL` gate.** While `RetailS_train` has continuous multi-minute recordings (**552.2 span camera-hours** across 942 files), the test clips in `RetailS_test_staged` (624 clips) and `RetailS_test_realworld` (47 clips) were tightly trimmed by the dataset authors around the ~2 s concealment action: the median pre-event lead-in before `mask == 1` is **1.00 s (15 frames)** in staged clips and **0.47 s (7 frames, min 0)** in real-world clips. A hard 2–3 s `DWELL` prerequisite before `REACH -> AT_BODY` would reject almost every test clip before the concealment frame arrives (and in live video would fail whenever a tracker ID switches near a shelf).
4. **Define `AT_BODY` with elbow flexion + medial/torso proximity, not wrist-to-hip distance alone.** When a person stands or walks with arms relaxed at their sides, their wrists naturally hang right next to their hips (`lhip`/`rhip`): across 458 single-person staged clips -- that probe's own eligible-clip count, not the calibration denominator, which §3 records as **459** -- median minimum wrist-to-hip distance is `0.068` torso lengths inside the labeled event and `0.064` outside it (AUROC ~0.50). Requiring **elbow flexion (`elbow_angle < max_conceal_elbow_deg`)** together with wrist proximity to the torso/waist/chest separates concealment from resting arms (single-feature frame AUROC jumps from `0.50` to `0.64` even before sequence ordering).

---

## 2. Exact inventory of the two archives

| Split / Archive | Pose JSON files | `.npy` mask files | Uncompressed size | Frame span (at 15 fps) | Active-person frames | Events (`mask == 1` runs) |
|---|---:|---:|---:|---:|---:|---:|
| `RetailS.zip` — `RetailS_train` | 942 | 0 | 4,443.4 MB | 29,820,124 (**552.2 cam-h**) | 6,349,731 (**117.6 cam-h**) | 0 (all normal) |
| `RetailS.zip` — `RetailS_test_staged` | 624 | 624 | 25.3 MB | 40,913 (0.76 cam-h) | 40,400 | **622** (in 622 positive clips; 2 clips normal-only) |
| `RetailS.zip` — `RetailS_test_realworld` | 47 | 47 | 2.5 MB | 4,846 (0.09 cam-h) | 3,721 | **43** (in 41 positive clips; 6 clips normal-only) |
| `PoseLift.zip` — `pose/train` | 104 | 0 | 43.3 MB | ~79,342 (1.47 cam-h) | — | 0 (all normal) |
| `PoseLift.zip` — `pose/test` + `gt` | 47 | 47 | 2.5 MB | 4,846 (0.09 cam-h) | 3,721 | **43** (**identical to `RetailS_test_realworld`**) |

### Coverage per camera and day in `RetailS_train`
- **By camera (`1` to `6`):**
  - Cam 1: 159 files, **92.7 h**
  - Cam 2: 159 files, **90.8 h**
  - Cam 3: 156 files, **90.3 h**
  - Cam 4: 156 files, **95.0 h**
  - Cam 5: 156 files, **90.7 h**
  - Cam 6: 156 files, **92.7 h**
- **By date (`MMDD` in filename `<cam>_<MMDDHH>_<part>.json`):** 11 days (`0418` to `0429`, excluding `0427`), 7.3 h to 64.2 h per day.
- **Implication for false-alarm calibration:** 552.2 span camera-hours (~92 h per camera) comfortably exceeds the 100 camera-hour bar in `dataset.md` section 5. With zero false alarms in 92 hours on a camera, the 95% upper confidence bound (rule of three) is `0.033` false alarms per camera-hour.

---

## 3. Verification table (section 5 of `dataset.md` filled with measured values)

| Check | Measured value | Consequence for the design |
|---|---|---|
| Camera-hours in `RetailS.zip` | **552.2 span camera-hours** (117.6 active-person hours) across 942 train files, ~92 h per camera | Full normal calibration is viable, including per-camera and leave-one-camera-out splits and day-based (`0418`–`0425` train/dev vs `0426`–`0429` test) normal holdout. |
| JSON schema & coordinate axes | `{person_id: {frame_id: {"keypoints": [51 floats], "scores": null}}}`. Each triple is **`[y, x, conf]`**: `col0` (`y`) in `[0, 1080]`, `median(ankle - nose) = +212..+248 px`; `col1` (`x`) in `[0, 1920]`. | Adapter swaps `(col0, col1)` -> `(x, y)` and derives person bbox `[min_x, min_y, max_x, max_y]` from confident keypoints. |
| Wrist confidence distribution & repeat rate | `conf_frac_zero = 0.0`, `conf_frac_one = 0.0`, `wrist_conf_repeat_rate = 0.28%–0.50%`, `wrist_conf < 0.30` on `22.0%` (train), `11.5%` (staged), `24.5%` (real). | Confidences are continuous floats, not constant fills. However, 8-frame HRNet smoothing damps sudden confidence drops (single-frame drop AUROC is `0.522` on staged, `0.554` on real). Keep `RETURNED` (kinematic hold + exit) as a full first-class path alongside `HIDDEN` (sustained confidence drop). |
| Event lengths & lead-in / tail | Staged (622 events): event length min/median/max = **2 / 31 / 91 frames** (`0.13 / 2.07 / 6.07 s`); pre-event lead-in median = **15 frames (`1.00 s`)**. Real (43 events): event length **10 / 27 / 180 frames** (`0.67 / 1.80 / 12.0 s`); pre-event lead-in median = **7 frames (`0.47 s`, min `0`)**, post-event tail median = **7 frames (`0.47 s`)**. | Labels mark the concealment action itself inside tightly trimmed clips. The state machine must not require a multi-second `DWELL` before `REACH` when a track starts mid-scene (see Section 4). |
| Mask length vs pose frames | Staged: `0 / 624` mismatches (`len(mask) == max_frame + 1`). Real: `4 / 47` files have `max_frame < len(mask) - 1` only because the person left before the clip ended. | Frame keys are 0-based indices directly into the `.npy` mask; no offset adjustment needed. |
| Per-person vs video-level labels | Masks are 1D per-frame arrays (`0/1`), not per-person, so **clip counts and event counts are different numbers and only one of them is a recall denominator.** Measured again on 2026-11-05 from the extracted archives: staged -- **461 of 624 clips (73.9%)** hold exactly one person id at any point in the clip; **459** of those are positive and each contributes exactly one labeled run, so the single-person staged denominator is **459 events**. Real-world -- **27 of 47 clips** hold exactly one id, **25** of them positive, giving **25 events**. A different question -- "exactly one person present *during the labeled window*" -- gives 492 staged clips (79.1% of the 622 positive clips, 492 events) and 27 real clips (28 events, because some of those clips contain more than one run). | Calibrate on the **461 single-person staged clips / 459 events** and **27 / 25** real-world, because that is the definition `partition_staged_pairs` stratifies on (`tracks == 1`) and the one `calibration_results.json` writes into `splits.staged_single_person_events` and `splits.realworld_single_person_events`; report full-set recall (622 staged events, 43 real events) alongside it. A table header must never pair a clip count with an event denominator. |
| `RetailS_test_realworld` vs `PoseLift/test` | **47 / 47 pose JSONs and 47 / 47 `.npy` masks have identical MD5 hashes.** | Deduplicate: use `RetailS_test_realworld` once as the held-out real test set (never tune on it). |
| Camera images / `.pkl` files | None in either zip. | Shelf-front zones on `RetailS` are generated by the dwell-heatmap zone-proposal tool on `RetailS_train` per camera (`1..6`), alongside the zone-free ablation mode. |

---

## 4. Refinements propagated to `HIGH_LEVEL_DESIGN.md`

1. **`DWELL` as tier promoter / continuous-stream gate rather than hard blocker for short tracks (`HIGH_LEVEL_DESIGN.md` §6.1):**
   - Tracks with `track_age >= dwell_min_s` must satisfy `DWELL` in a shelf zone before `REACH -> AT_BODY`.
   - Tracks born inside a shelf zone (or in zone-free mode) that begin `REACH -> AT_BODY` within `short_track_grace_s` of track birth (covering trimmed benchmark clips and live tracker ID switches near a shelf) may enter `REACH -> AT_BODY` directly; on continuous streams their tier is `REVIEW (medium)` unless re-ID links them to a prior dwelling track (`REVIEW (high)`).
2. **Anatomical `AT_BODY` predicate (`HIGH_LEVEL_DESIGN.md` §6.1 & §14):**
   - `AT_BODY` requires the wrist to enter a torso-normalized target zone (front waist/pocket, chest/jacket, under-arm, or worn bag) **while the ipsilateral elbow is flexed (`elbow_angle < max_conceal_elbow_deg`) or the wrist crosses medial to the shoulder–hip line**, preventing false triggers from straight arms hanging past the hips during normal standing/walking.
3. **Fast quality-gate warm-up (`HIGH_LEVEL_DESIGN.md` §6.2):**
   - `min_track_s` defaults to `0.4 s` (6 frames at 15 fps, 10 frames at 25 Hz) with at least 6 confident keypoints including shoulders and hips (`torso_px >= min_torso_px`, where the 10th percentile in `RetailS` is `48–68 px`).
   - **What actually ships is `min_track_s: 0.25`, not `0.4`.** `DetectorSettings.min_track_s` keeps the `0.4` default, but `run_calibration` pins `min_track_s=.25` (and `dwell_min_s=1.6`) for the entire Stage 3 grid, so all three stored tuned rows -- "Recommended Balanced", "High-Sensitivity" and "Strict Low-FA" -- record `0.25`, and `concealment_detection.yaml` ships that. The warm-up bar the design asks for is therefore the one in `types.py`; the live service runs a third of a second shorter, and the yaml's `dwell_min_s: 2.0` plus `inward_min_torsos: 0.12` were set after that run rather than by it. Re-measure the false-alarm cost of `0.25` versus `0.4` before treating the shipped value as the calibrated one.
4. **Strict split hygiene (`HIGH_LEVEL_DESIGN.md` §13b) -- as implemented today:**
   - **Normals:** `discover_retails_splits` sorts the 11 `RetailS_train` day codes and takes the first eight as development (`0418`–`0425`, 618 files) and the remaining three as the frozen false-alarm holdout (`0426`, `0428`, `0429`, 324 files). Shelf zones are proposed from development files only (`propose_all_camera_zones(..., max_files_per_camera=15)`).
   - **Normal evaluation is sampled, so every false-alarm figure is a sampled measure.** `run_calibration(train_tune_sample_size=36, train_holdout_sample_size=36)` scores 36 development and 36 holdout files chosen by `stratified_sample_files` (equal allocation across camera x day with capacity redistribution); the locked run (`calibration/reports/run-1794576781479912800/`) records **21.92 span hours** (`3.13` active hours) development and **18.08 span hours** (`3.96` active hours) holdout out of the archive's 552.2 (while the historical pre-split `calibration/reports/calibration_results.json` recorded **20.05** and **21.35** span hours). A camera-hour claim must quote the span from the artifact, never the archive total; widening to the full archive means raising both sample sizes and re-running.
   - **Positives:** `partition_staged_pairs` produces the frozen 50/50 staged split -- stratified by camera, positive/normal and single/multi-person (`tracks == 1`); inside a stratum the order is the SHA-256 of the filename, so the split is reproducible without process-dependent RNG state; odd strata alternate the extra clip so the halves stay within one clip of each other (`staged_dev` = **312 clips / 311 events / 229 single-person events**; `staged_test` = **312 clips / 311 events / 230 single-person events**). Stage 2 quantiles and the Stage 3 grid see the development half only (`max_staged=250` sampled positive frames of features); the test half is scored exactly once, in Stage 4, and never enters selection.
   - **Order of writes:** the complete grid and the frozen choice are persisted to `frozen_selection.json` *before* any holdout clip is opened, so a holdout number cannot be improved by re-running selection afterwards.
   - **Real-world:** `RetailS_test_realworld` (47 clips, 43 events; `PoseLift/test` dropped as the exact duplicate) is a Stage 4 holdout. Because `experiment_inward_motion.py` historically screened variants against it, it is **not** an untouched test set in the strict sense -- see §18 of `HIGH_LEVEL_DESIGN.md`.
   - **Historical pre-split artifact vs. locked split run:** `calibration/reports/run-1794576781479912800/` implements this complete leak-free split (recording both `stage4_staged_test` and `stage4_realworld`). The older `calibration/reports/calibration_results.json` (`staged_clips: 624`, no `stage4_staged_test` key) remains untouched as a historical pre-split artifact.
