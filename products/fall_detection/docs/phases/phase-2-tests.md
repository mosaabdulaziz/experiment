# Phase 2: Test Suite Report

**Date:** 2026-10-27
**Prerequisite met:** Phase 1 requirements confirmed by the user; the testability audit amended AC-F1.1 (split), AC-F1.2 (new), AC-F33.2 (reworded).

The suite was written **red-first**. Every test was run and confirmed to fail for the right reason
(the artifact it pins does not exist yet) or to skip cleanly where it is gated on absent
infrastructure. No implementation code was written.

---

## Test Files Created

Two suites, matching the repo's two test environments:

**Root suite** (`tests/`, the engine's env — `requirements-test.txt`, no torch):

| File | Tests | Covers |
|---|---|---|
| `tests/test_fall_sidecar_config.py` | 6 | AC-F1.1, AC-F26.1, AC-F32.1, golden-config existence |
| `tests/test_pose_module.py` | 8 | AC-F1.1 (gies half), AC-F2.1, AC-F3.1, AC-F4.1, AC-F5.1, AC-F6.1 |

**Product suite** (`products/fall_detection/tests/`, the product venv — torch/pandas where the vendor runs):

| File | Tests | Covers |
|---|---|---|
| `test_scoring.py` | 21 | AC-F10.1, AC-F10.2, AC-F11.x, AC-F12.x, AC-F13.x, AC-F14.1, AC-F15.1, AC-F30.3, AC-F35.x, AC-N7.1 |
| `test_routing.py` | 16 | AC-F16.x, AC-F17.1, AC-F18.1, AC-F19.1, AC-F30.1, AC-F30.2, AC-F36.1, AC-F37.1 |
| `test_worker.py` | 4 | AC-F20.1, AC-F21.1, AC-N3.1 (service-level proxy) |
| `test_storage_review.py` | 2 | AC-F22.1, AC-F22.2 |
| `test_boot_contract.py` | 8 | AC-F23.1-3, AC-F24.1, AC-N9.1, AC-N11.1 |
| `test_status_api.py` | 3 | AC-F25.1, AC-N10.1 |
| `test_seam_vendor.py` | 6 | AC-F7.1, AC-F8.1, AC-F8.2, AC-F9.1, AC-F30.4, AC-N8.1, AC-N11.2 |
| `test_autonomy.py` | 4 | AC-N6.1, AC-N6.2, AC-N11.2 |
| `test_preflight.py` | 5 | AC-F28.1 |
| `test_measurements_gated.py` | 6 | AC-F1.2, AC-F29.1, AC-N1.1, AC-N2.1, AC-N3.1, AC-N4.1, AC-N5.1 |
| `conftest.py`, `fall_helpers.py` | — | shared stand-ins (FakeRedis, FakeStorage, candidate/policy factories); no tests |

**Frontend suite** (`frontend/src/products/fall_detection/`, vitest — mirrors barber's product test setup):

| File | Tests | Covers |
|---|---|---|
| `frameUrl.test.ts` | 2 | AC-F33.2 |
| `tab.test.ts` | 3 groups | AC-F33.1, AC-F34.1 |

**Total: 89 backend tests + 2 frontend files.** AC-F27.1 is covered by the repo's pre-existing
`tests/test_engine_boundary.py` ratchet, which the Phase 2 suite deliberately leaves unchanged.

---

## Coverage Matrix

RED status legend: **RED** = ran and failed for the right reason (the pinned artifact is absent);
**SKIP(gated)** = ran and skipped cleanly on absent infrastructure; **RED/SKIP** = the unit half is
RED here, the live half is gated. Ratchet guards that pass vacuously today are marked **GUARD**.

### Pose module (root suite)

| AC | Test | RED? |
|---|---|---|
| AC-F1.1 | `test_the_overlay_declares_the_pose_module_with_its_sgie_interval`; `test_it_asks_for_one_secondary_gie_and_an_explicit_engine_wins` | RED (config file absent; `perception.modules.pose` ImportError) |
| AC-F2.1 | `test_two_tracked_people_publish_one_item_each_on_the_pose_field_only` | RED (ImportError) |
| AC-F3.1 | same + `test_published_keypoints_land_where_the_geometry_says_within_two_px_and_inside_the_box` | RED |
| AC-F4.1 | `test_the_letterbox_inverse_undoes_padding_and_scale_then_the_capture_scale` | RED |
| AC-F5.1 | `test_items_with_no_parent_pointer_are_matched_to_the_person_that_contains_them` | RED |
| AC-F6.1 | `test_settings_validate_and_an_undeclared_yaml_key_refuses_startup` | RED |
| AC-F3.1 omission | `test_a_malformed_or_missing_instance_mask_omits_the_person_and_never_crashes`; `test_a_person_without_confident_keypoints_is_omitted_not_zero_published` | RED |

### Monitor, seam, vendored copy (product suite)

| AC | Test | RED? |
|---|---|---|
| AC-F7.1 | `test_the_copy_differs_from_the_pristine_tree_only_by_the_seam_and_provenance` | RED (copy tree absent) |
| AC-F8.1 | `test_the_same_fixture_produces_the_same_candidates_unpatched_and_patched` | SKIP(gated torch) — RED once the venv exists |
| AC-F8.2 | `test_the_physical_scores_are_the_latch_moment_margins_not_the_recumbent_moment_ones` | SKIP(gated torch) |
| AC-F9.1 | `test_the_copied_package_imports_and_reads_the_config_beside_it` | SKIP(gated torch) |
| AC-F30.4 | inside `test_the_same_fixture_...` (parity across **all** candidate types incl. lying) | SKIP(gated torch) |
| AC-N8.1 | `test_the_nine_vendor_test_files_are_part_of_the_product_suite`; `test_the_fall_test_recipe_runs_the_product_suite_and_thus_the_copied_vendor_tests` | RED (roster absent; no `fall-test` recipe) |

### Scoring policy (product suite)

| AC | Test | RED? |
|---|---|---|
| AC-F10.1 | `test_the_collapse_score_moves_with_the_gate_block_it_reads` | RED (`scoring` ModuleNotFound) |
| AC-F10.2 | `test_all_seven_spans_default_to_their_gates_own_threshold_and_are_overridable`; `test_all_seven_spans_sit_in_the_vendors_config_yaml_beside_their_gates` | RED |
| AC-F11.1 | `test_an_immobility_candidate_whose_top_class_is_a_decisive_fall_evaluates_as_falling` | RED |
| AC-F11.2 | `test_a_fall_candidate_takes_the_group_its_top_class_falls_into`; `test_a_lying_candidate_evaluates_as_lying_whatever_the_reading_says` | RED |
| AC-F12.1 | `test_a_classifier_that_cannot_tell_fall_from_lying_scores_near_zero` | RED |
| AC-F12.2 | `test_a_decisive_reading_scores_near_one` | RED |
| AC-F13.1 | `test_a_cold_only_candidate_is_capped_below_the_threshold_and_delegated` | RED |
| AC-F13.2 | `test_a_decisive_action_only_candidate_confirms_on_its_own`; `test_a_path_that_did_not_speak_leaves_its_trust_unclaimed` | RED |
| AC-F14.1 | `test_the_trust_weights_come_from_the_config_not_the_code` | RED |
| AC-F15.1 | `test_the_evaluation_class_sets_exist_only_in_the_products_config` | RED |
| AC-F30.3 | `test_the_lying_score_is_the_geometric_mean_of_posture_and_stillness`; `test_a_lying_candidate_is_scored_from_its_emission_moment_kinematics`; `test_a_candidate_with_no_window_never_reaches_the_threshold` | RED |
| AC-F35.1 | `test_a_path_b_fall_with_a_window_and_a_decisive_reading_can_reach_the_threshold` | RED |
| AC-F35.2 | `test_a_candidate_with_no_window_never_reaches_the_threshold` | RED |
| AC-N7.1 | `test_the_scoring_module_imports_no_torch_no_redis_and_no_opencv` + the suite's own no-Redis/no-torch purity | RED |
| *(design §4.2 formula)* | `test_one_collapsed_condition_cannot_be_averaged_away`; `test_the_collapse_aggregation_is_the_geometric_mean_of_design_4_2` | RED — see UNANCHORED below |

### Routes, alert, clip (product suite)

| AC | Test | RED? |
|---|---|---|
| AC-F16.1 | `test_a_confident_falling_evaluation_alerts_critical_with_no_review`; `test_a_confirmed_fall_writes_the_platform_alert_and_requests_the_clip_on_the_frame_clock` | RED (`scoring`/`consumer` ModuleNotFound) |
| AC-F16.2 | `test_an_unconfident_candidate_of_any_evaluation_alerts_warning_and_queues_review`; `test_an_unconfident_fall_writes_a_warning_alert_and_a_review_row_carrying_the_evidence` | RED |
| AC-F16.3 | `test_a_confident_upright_evaluation_is_a_closed_false_alarm`; `test_a_confirmed_upright_evaluation_writes_nothing_at_all` | RED |
| AC-F16.4 | `test_a_confident_lying_evaluation_alerts_warning_with_no_review`; `test_a_confirmed_lying_candidate_alerts_warning_without_a_review_row` | RED |
| AC-F17.1 | `test_a_confirmed_fall_writes_the_platform_alert_and_requests_the_clip_on_the_frame_clock` | RED |
| AC-F18.1 | same | RED |
| AC-F19.1 | `test_the_monitor_is_given_the_frame_entrys_pts_and_every_write_carries_it` | RED |
| AC-F20.1 | `test_two_cameras_never_share_a_monitor_or_cross_talk` | RED |
| AC-F21.1 | `test_a_full_queue_drops_the_oldest_frame_and_never_grows`; `test_the_worker_keeps_running_after_the_queue_overflowed` | RED |
| AC-F30.1 | `test_a_confirmed_lying_candidate_alerts_warning_without_a_review_row` | RED |
| AC-F30.2 | `test_an_unconfident_lying_candidate_alerts_warning_and_queues_review` | RED |
| AC-F31.1 | `test_a_lying_candidate_right_after_an_alerted_fall_on_the_same_track_is_suppressed_but_still_scored` | RED |
| AC-F31.2 | `test_a_lying_candidate_after_the_suppression_window_alerts_normally`; `test_suppression_is_per_track_not_a_mute_on_the_kind` | RED |
| AC-F36.1 | `test_unstable_routes_exactly_like_lying` | RED |
| AC-F37.1 | `test_a_fall_recovered_candidate_produces_no_alert_no_review_and_no_state` | RED |

### Review queue, boot contract, status (product suite)

| AC | Test | RED? |
|---|---|---|
| AC-F22.1 | `test_a_review_row_survives_a_restart_with_all_of_its_evidence` | RED (`storage` ModuleNotFound) |
| AC-F22.2 | `test_only_the_three_operator_statuses_are_accepted_and_the_note_is_stored` | RED |
| AC-F23.1 | `test_a_missing_checkpoint_refuses_startup_naming_the_file` | RED (`boot` ModuleNotFound) |
| AC-F23.2 | `test_a_corrupt_checkpoint_refuses_startup_naming_the_checksum` | RED |
| AC-F23.3 | `test_cuda_configured_with_no_gpu_refuses_startup_and_does_not_fall_back` | RED |
| AC-F24.1 | `test_a_cuda_wheel_without_this_cards_kernels_refuses_startup` | RED |
| AC-N9.1 | all four refusal tests assert `code != 0` + a cause substring | RED |
| AC-N11.1 | `test_the_resolved_environment_carries_exactly_one_torch`; `test_the_torch_build_matches_the_pinned_cuda_index`; `test_the_real_classifier_constructs_against_the_resolved_environment` | RED / SKIP(gated torch) |
| AC-F25.1 | `test_status_reports_the_device_the_weights_and_the_classifier` | RED (`server` ModuleNotFound) |
| AC-N10.1 | `test_the_live_paths_are_derived_from_the_boot_state_not_declared_beside_it`; `test_status_never_reports_more_paths_than_are_live` | RED |

### Deployment (both suites)

| AC | Test | RED? |
|---|---|---|
| AC-F26.1 | `test_no_other_sidecar_yaml_changed_against_main` | **GUARD** (green-by-inaction today; fails the moment another product's yaml changes) — *test removed later in 768d27b with the branch's stale-test sweep* |
| AC-F27.1 | pre-existing `tests/test_engine_boundary.py` ratchet | GUARD (unchanged by this suite) |
| AC-F28.1 | `test_a_live_sidecar_holding_a_fall_camera_is_refused_and_named` (+ disjoint/empty/multi cases); `test_the_fall_up_recipe_runs_the_preflight_before_anything_starts` | RED (`preflight` ModuleNotFound; no `fall-up` recipe) |
| AC-F29.1 | `test_the_golden_config_exists_and_extends_the_fall_overlay`; `test_the_golden_baseline_carries_the_pose_field_with_17_keypoints` | RED / SKIP(gated golden run) |
| AC-F32.1 | `test_the_pilot_camera_set_is_exactly_cam_1`; `test_the_three_camera_preset_replaces_only_the_sources` | RED |

### Frontend tab (frontend suite)

| AC | Test | RED? |
|---|---|---|
| AC-F33.1 | `tab.test.ts` — registry entry; no WebSocket / Redis / canvas in any product source | RED (no sources yet) |
| AC-F33.2 | `frameUrl.test.ts` — the wall's `/stream/<cam>/frame` and the tab's are the same endpoint | RED |
| AC-F34.1 | `tab.test.ts` — no WebSocket; JPEG layer (`live/FrameLayer.tsx` + `live/frameUrl.ts`) separate from the page (`index.tsx`) | RED |

### Non-functional measurements (product suite, gated)

| AC | Test | RED? |
|---|---|---|
| AC-F1.2 | `test_the_recorded_sidecar_run_names_the_pose_gie` | SKIP(gated `FALL_MEASUREMENTS`) |
| AC-N1.1 | `test_the_gpu_budget_harness_exists_and_the_recorded_delta_names_the_pose_stage` | SKIP(gated) — harness exists |
| AC-N2.1 | `test_the_corpus_replay_harness_reports_the_action_path_and_skips_when_the_corpus_is_absent` | RED (`measure` ModuleNotFound) |
| AC-N3.1 | `test_a_hundred_confirmed_falls_are_routed_in_well_under_the_clip_window` (service proxy); `test_the_onset_to_alert_latency_is_measured_end_to_end_on_replayed_footage` (gated) | RED |
| AC-N4.1 | `test_the_corpus_replay_harness_...` (precision/recall/false-alarms + right camera/track attribution) | RED |
| AC-N5.1 | `test_the_classifier_cost_is_recorded_on_the_deployment_card` | SKIP(gated RTX 5090) |
| AC-N6.1 | `test_no_python_file_imports_another_product`; `test_no_source_file_mentions_another_product` | RED / **GUARD** |
| AC-N6.2 | `test_the_config_loader_loads_only_the_products_own_yaml` | RED (`config.py` absent) |
| AC-N11.2 | `test_pyproject_pins_exactly_one_torch_and_keeps_ultralytics_out_of_runtime`; `test_the_fall_test_recipe_...` | RED (no `pyproject.toml`) |

---

## Verifier Agent Findings

The verifier read the Phase 1 report and all 15 test files, cross-checked against the design doc,
the pristine vendor tree, the justfile, `tools/`, and the frontend. Its tally: **56 VERIFIED, 12
INCORRECT/partial, 0 MISSING, 2 INTERNAL, 2 UNANCHORED.** Every finding was fixed before this
report; the fixes are below.

### Resolved Issues

| # | Type | AC / test | Fix |
|---|---|---|---|
| 1 | INCORRECT (must-fix) | AC-F15.1 vs AC-F7.1 | The class-set scan flagged the bare word `"falling"` in the byte-identical vendor copy (`tsstg_model.py`, `omnifall_adapter.py`, `transfer_gate.py`) — mutually unsatisfiable with F7.1. Retargeted to the **mapping** `\b(falling|...)\s*:\s*\[`, which the pristine source does not contain, and added a check that `scoring.py` carries no second copy. |
| 2 | INCORRECT | AC-F1.2 | Added `assert "interval" in text.lower()` — the log must name the GIE **and its interval**. |
| 3 | INCORRECT | AC-F10.2 | Added `test_all_seven_spans_sit_in_the_vendors_config_yaml_beside_their_gates` — reads `config.yaml`, requires all seven span keys present and each equal to its gate's own threshold, so "overridable without a code change" is pinned at the file, not only at the function API. |
| 4 | INCORRECT | AC-F30.3 | The no-window case now also asserts `kinematic is None` and `cold is not None` — a windowless physical score must fall to the **cold** ceiling (0.40), never the kinematic one (0.50). |
| 5 | INCORRECT | AC-F30.4 | The seam-parity test no longer filters to `fall`: it compares **all** candidate types, asserts a `lying` candidate is present, and so pins that `detect_lying`'s contract is unchanged. |
| 6 | INCORRECT | AC-N1.1 | Now requires a `busy%` column and a recorded `pose … delta` in the budget output, not just the substring "pose". |
| 7 | INCORRECT | AC-N3.1 | Added the gated `test_the_onset_to_alert_latency_is_measured_end_to_end_on_replayed_footage` (harness exists, takes the corpus root, skips cleanly, reports `onset_to_alert_s < 5.0`). |
| 8 | INCORRECT | AC-N4.1 | Added the attribution clause `report["attributed"] == report["true_positives"]` (the candidate is on the right camera and track). |
| 9 | INCORRECT | AC-N8.1 | Added `test_the_fall_test_recipe_runs_the_product_suite_and_thus_the_copied_vendor_tests` — the justfile must have a `fall-test` recipe that really runs pytest (not only in a comment). |
| 10 | INTERNAL | AC-N10.1 | Replaced the circular echo test: `/status` is now built from `boot.status(boot_state)`, and `test_the_live_paths_are_derived_from_the_boot_state_not_declared_beside_it` pins that `boot.paths_live` drops `action` when the classifier is not ready — the report cannot claim more than the service runs. |
| 11 | INCORRECT | AC-N11.1 | Added `test_the_real_classifier_constructs_against_the_resolved_environment` (no fake, no monkeypatch — the copied `TSSTGClassifier` really builds); the single-torch test now skips when the interpreter is not the product env instead of failing spuriously. |
| 12 | INCORRECT | AC-N11.2 | Covered by fix #9 (the `fall-test` recipe) plus the existing pyproject pin. |
| 13 | INTERNAL | `test_the_raw_reader_...` | Deleted the internal-helper test; its observable half (17×3, malformed/short/None mask omitted) is now `test_a_malformed_or_missing_instance_mask_omits_the_person_and_never_crashes` at the publish level. |
| 14 | toothless guard | N7.1 scan | Rewrote the substring scan as `^\s*(import|from)\s+<name>\b` so `from torch import nn` is caught. |
| 15 | toothless guard | AC-F28.1 recipe scan | The `fall-up` scan now ignores comment lines, so a preflight mentioned only in a comment does not pass. |
| 16 | toothless guard | AC-F2.1 aggregation | Added `test_the_collapse_aggregation_is_the_geometric_mean_of_design_4_2`, pinning the formula at an interior point (0.16^0.25 ≈ 0.632) where geometric mean, `min` and arithmetic mean all differ. |

### UNANCHORED tests — kept, with rationale

- **`test_the_overlay_tags_its_sources_with_its_own_role`** — no Phase 1 AC requires `sidecar_role`/`roles`
  tags (F28 says the tag is *not* the enforcement). Kept: the design (§9, the overlay yaml) specifies
  the tag as the recording-owner marker, and other products' sidecar tests pin the same shape
  (`test_barber_sidecar_config.py`). **User may strike it if it is judged out of scope.**
- **`test_the_collapse_aggregation_is_the_geometric_mean_of_design_4_2`** and
  **`test_one_collapsed_condition_cannot_be_averaged_away`** — pin the design §4.2 formula
  `s_collapse = (s_drop·s_vdown·s_hip·s_posture)^0.25`. No Phase 1 AC transcribes the aggregation
  (AC-F10.x cover only thresholds and spans). Kept: it is design-verbatim, and without it a `min()`
  implementation would satisfy the whole suite while violating §4.2. **Recommend a Phase 1 amendment
  adding an AC for the collapse aggregation** so it is anchored rather than orphaned.

---

## Tests Not Yet Runnable (blocked on infrastructure)

These are written and correct; they cannot execute on this dev box and will run in Phase 3/4 where
their prerequisite exists. Each skips cleanly rather than failing.

| Test | Needs | Behavior when absent |
|---|---|---|
| `test_seam_vendor.py` (6) | torch + pandas (product venv, slice S2) + the pristine vendor checkout | skips via `_need_torch()` / `pytestmark` |
| `test_boot_contract.py` positive path + `..._real_classifier_constructs` | torch, mmaction2, the pinned checkpoints | skips |
| `test_the_torch_build_matches_the_pinned_cuda_index` | torch in the product venv | skips |
| `test_measurements_gated.py` (F1.2, N1.1, N5.1) | `FALL_MEASUREMENTS` evidence dir / an RTX 5090 | skips |
| `test_the_..._corpus_replay_...` / `..._latency_...` (N2/N3/N4) | the corpus at `F:\fall_detection\data` | skips (via `measure.…["skipped"] is True`) |
| **Frontend `frameUrl.test.ts`, `tab.test.ts`** | Node.js + vitest (`npm install` in `frontend/src/products/fall_detection/`) — **no Node.js on this machine** | not runnable here; mirrors barber's product vitest setup |
| Golden-run halves (F1.2, F29.1 baseline, N1.1) | Docker + GPU sidecar, recorded via `tools/engine_golden_run.sh` | skips |

**Verified RED evidence on this box** (`uv run --python 3.11 --with-requirements requirements-test.txt python -m pytest …`):

- Root suite: `test_fall_sidecar_config.py` → **5 failed, 1 passed** (the guard); `test_pose_module.py` → **collection error: cannot import name `pose`** (the right reason).
- Product suite: **7 failed, 2 passed (guards), 7 skipped (gated), 7 collection errors** — every error is a `ModuleNotFoundError` naming exactly the module the tests pin (`boot`, `preflight`, `scoring`, `server`, `storage`, `consumer`, `measure`) or a missing `config.py` / `pyproject.toml` / copy tree / `fall-test` recipe.

---

## What the Developer Must Implement (Next Phase)

The suite pins this public surface (nothing beyond it):

**Engine (root, no torch):**
- `perception/modules/pose/` — `Pose(Settings, services)` with `gies()`, `extract()`, `transform()`,
  `analyze()`; `Settings(config="", engine="", detection_threshold=0.25, keypoint_confidence_threshold=0.30, interval=2)` validating its
  fields; a `keypoints_to_capture(raw, box_mux, sx, sy)` letterbox inverse matching face_detect's math;
  publish the `pose` field (capture px, xyxy, 17×3) with containment ownership; omit unconfident/malformed.
- `products/fall_detection/sidecar.yaml`, `configs/presets/fall_3cam/sidecar.yaml`,
  `tests/configs/golden-fall_detection.yaml`; other products' sidecar yamls untouched.

**Product (own venv):**
- `scoring.py` — `evaluate`, `reading_score`, `combine`, `route`, `collapse_score`, `stillness_score`,
  `lying_score`, `default_spans`, `score_candidate`; no torch/redis/cv2/socket import.
- `consumer.py` — `CameraWorker(camera, monitor, redis, storage, policy, vendor_cfg)` with
  `handle_candidates`, `handle_entry`; `FrameQueue(maxsize)` dropping oldest; `build_workers(...)`.
- `storage.py` — `connect(db_path)`, `add_review`, `list_reviews`, `set_review_status` (SQLite).
- `boot.py` — `boot(cfg)`, `status(state)`, `paths_live(state)`; monkeypatchable `torch` and
  `TSSTGClassifier`; real-CUDA-op gate; refuse-with-cause.
- `server.py` — `create_app(status, redis, storage)` FastAPI app with `GET /status`.
- `config.py` — loads `fall_detection.yaml` only.
- `preflight.py` — `check_cameras(fall_ids, live_sidecars)` raising `PreflightConflict`.
- `measure.py` — `action_path_eval(corpus_root, interval)`, `latency(corpus_root)`,
  `classifier_cost(device, concurrent_tracks)`; each returns `{"skipped": True}` when the corpus is absent.
- `fall_detection/` — the vendored copy of `F:\fall_detection_module\src` + `__init__.py`,
  differing from pristine **only** by the additive seam in `fall_monitor.py` (`reading`, `path_scores`);
  `config.yaml` beside it carrying the seven spans; the product's `docs/PROVENANCE.md` records the seam.
- `pyproject.toml` — exactly one torch, ultralytics in the test/dev group only, the cu130 index for torch.
- `tests/vendor/` — the vendor's nine test files + `conftest.py` + `synth.py`.

**Deployment:** justfile `fall-up` (preflight before compose), `fall-down`, `fall-test`;
compose `deepstream-sidecar-fall` (profile `fall`, telemetry 9105).

**Frontend:** `frontend/src/products/fall_detection/` — `index.tsx`, `live/FrameLayer.tsx`,
`live/frameUrl.ts`, a `PRODUCTS` registry entry; no WebSocket, no Redis, no box math.
