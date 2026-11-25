# Phase 1: Requirements and Acceptance Criteria

**Date:** 2026-10-27 · **Prerequisite met:** Phase 0 closed — all ten ambiguities ruled (phase-0-reading-notes.md, "Rulings").
**Numbering:** the requirements doc's stable ids are reused (`F`, `N`, `X`). Three ids are **new from the Phase 0 rulings**: F35 (A1), F36 (A2), F37 (A3). Rulings A4/A6/A7 add acceptance criteria to F8/F10/F30. AC ids are `AC-<req>.<n>`.
**Delivery scope:** the first release (F1–F37, N1–N10). X1–X8 (the live annotation overlay) are the **next release** — out of scope here, listed under Explicitly Out of Scope.

## Functional Requirements

| ID | Requirement (SHALL statement) | Source | AC IDs |
|----|-------------------------------|--------|--------|
| F1 | The fall sidecar SHALL run exactly one secondary GIE (`sgie_pose`) on tracked persons' crops when the `pose` module is enabled; no new full-frame inference. | [DOC: FALL_DETECTION_REQUIREMENTS.md, F1; NEXT_MODULES.md:41-53] | AC-F1.1, AC-F1.2 *(split in Phase 2 audit)* |
| F2 | The pose module SHALL publish keypoints as the JSON-encoded `pose` field on the frame's own `iris:meta:<cam>` stream; no other stream is written for pose. | [DOC: REQUIREMENTS F2; design §1.3 refinement 9; publisher.py:348-349] | AC-F2.1 |
| F3 | The `pose` field SHALL be exactly `{"shape": [capture_w, capture_h], "items": [{"id": <tracker id>, "box": [x1,y1,x2,y2], "kps": [[x,y,conf] × 17]}]}`; persons without confident keypoints are omitted, not zero-published. | [DOC: REQUIREMENTS F3; design §3.4; fall_monitor.py:120-151] | AC-F3.1 |
| F4 | Box and keypoints SHALL be in one space — capture pixels — via the letterbox inverse (`landmarks_to_capture` math) then mux→capture scaling. | [DOC: REQUIREMENTS F4; design §3.2; face_detect/__init__.py:60-76] | AC-F4.1 |
| F5 | The module SHALL NOT rely on the SGIE's `parent_id`; ownership is the containment match reused from `face_detect/owner.py`. | [DOC: REQUIREMENTS F5; face_detect/__init__.py:64-65; owner.py:117-158] | AC-F5.1 |
| F6 | The pose module SHALL declare its dependencies and every tunable as a `Settings` field; an undeclared yaml key refuses startup. | [DOC: REQUIREMENTS F6; modules/README.md:115] | AC-F6.1 |
| F7 | `products/fall_detection/fall_detection/` SHALL be a copy of `F:\fall_detection_module\src` differing **only** by the F8 seam and `PROVENANCE.md`; the vendor checkout stays pristine (ruling A5). | [DOC: REQUIREMENTS F7; plan decision 4; Phase 0 ruling A5] | AC-F7.1 |
| F8 | The seam SHALL be additive: the same fixture produces the same candidates (ids, times, types, boolean gates) unpatched and patched; every candidate additionally carries `reading` (`probs`, `top`) and `path_scores` (`action`, `kinematic`, `cold`). The physical score is measured **at packaging time**, over the gate of the path that latched; the reading is attached **at packaging time** (ruling A4, corrected in Phase 3 Round 2 — see below). | [DOC: REQUIREMENTS F8; design §4.2; fall_monitor.py:740-772, 999-1009; Phase 0 ruling A4] | AC-F8.1, AC-F8.2 |
| F9 | The vendor's master config SHALL sit at `products/fall_detection/config.yaml` (beside the package, where the import-time loader looks); the product's policy SHALL live in `fall_detection.yaml`; the copy imports without `FileNotFoundError`. | [DOC: REQUIREMENTS F9; fall_monitor.py:58-72] | AC-F9.1 |
| F10 | Physical-score thresholds SHALL be read from the same config blocks the boolean gates use (`path_b`, `path_c`, `lying`, `immobility`) — no second copy; the seven spans sit beside them, **default = the gate's own threshold**, all configurable (ruling A6, "until we benchmark and tune them"). The collapse score SHALL be the **geometric mean** of its four margins. | [DOC: REQUIREMENTS F10; design §4.2, §13.3; Phase 0 ruling A6] | AC-F10.1, AC-F10.2, AC-F10.3 |
| F11 | The evaluation SHALL be decided detector-first: a `fall` candidate takes the group of its reading's top class; an `immobility` candidate is `immobile` unless the top class is a decisive `Fall` (6), then `falling`; a `lying` candidate is `lying`. | [DOC: REQUIREMENTS F11; design §6.2] | AC-F11.1, AC-F11.2 |
| F12 | The reading's score SHALL be `clip((support − oppose) / margin_span, 0, 1)` over the evaluation's class set. | [DOC: REQUIREMENTS F12; plan §The reading's confidence] | AC-F12.1, AC-F12.2 |
| F13 | The paths SHALL combine as `1 − Π(1 − trust·score)` over the paths that spoke; a path that did not speak leaves its trust unclaimed (no renormalization). | [DOC: REQUIREMENTS F13; plan decision 7] | AC-F13.1, AC-F13.2 |
| F14 | The trust weights SHALL be configuration (`fall_detection.yaml`), not code; changing one changes that path's ceiling. | [DOC: REQUIREMENTS F14; plan §The seam Option A] | AC-F14.1 |
| F15 | The evaluation class sets SHALL exist in exactly one place: the product's config — never in the vendor's copy. | [DOC: REQUIREMENTS F15; design §6.2 refinement] | AC-F15.1 |
| F16 | Routing SHALL follow the table: score ≥ `confirm_threshold` → `falling`/`immobile` alert `critical`, no review; `lying` alert `warning`, no review; `upright` no alert, no review (closed false alarm). Score < threshold → `warning` alert **and** a review row, for any evaluation. | [DOC: REQUIREMENTS F16; plan decision 2; design §5.3] | AC-F16.1 … AC-F16.4 |
| F17 | The alert SHALL use the platform's contract as it is: `XADD iris:alert:<cam>` with `v=1`, `kind=fall`, `severity ∈ {critical, warning}`, `category=safety`, `message` naming the evaluation and the confidence, `pts` — readable by the app with no module code. | [DOC: REQUIREMENTS F17; modules.py:99-111; module_alerts.py:41-54] | AC-F17.1 |
| F18 | The clip SHALL be requested via `XADD iris:svr_request:<cam> {pts}` with the sidecar's frame clock; the resulting clip covers the fall (anchor ± `pad_seconds` 5.0). | [DOC: REQUIREMENTS F18; recordings.py:69-70; engine/main.py:258-289] | AC-F18.1 |
| F19 | The monitor's `t` SHALL be the frame entry's `pts` (the sidecar's clock), never the wall clock; alert and clip request carry the same clock. | [DOC: REQUIREMENTS F19; plan §The consumer] | AC-F19.1 |
| F20 | There SHALL be exactly one `FallMonitor` per camera; two cameras reporting the same track id never cross-talk. | [DOC: REQUIREMENTS F20; FallMonitor.tracks keyed by id alone] | AC-F20.1 |
| F21 | The per-camera queue SHALL be bounded and drop the **oldest** frame when full; the monitor keeps running. | [DOC: REQUIREMENTS F21; plan §The monitor runs on its own thread] | AC-F21.1 |
| F22 | The review queue SHALL persist in SQLite with the candidate's evidence (evaluation, combined score, per-path scores, reading's top class, clip time range); it survives a restart; only `acknowledged`/`confirmed`/`dismissed` + note are accepted. | [DOC: REQUIREMENTS F22; design §5.4; fall_monitor.py:511-522] | AC-F22.1, AC-F22.2 |
| F23 | The service SHALL refuse to start (non-zero exit, message naming the cause) on: a missing checkpoint; a corrupt/checksum-mismatched checkpoint; `classifier.device: cuda:0` with no working GPU. No CPU fallback. | [DOC: REQUIREMENTS F23; plan §The boot contract] | AC-F23.1 … AC-F23.3 |
| F24 | The start-up GPU check SHALL run a real CUDA operation on the device, not `cuda.is_available()`. | [DOC: REQUIREMENTS F24; plan risk 4] | AC-F24.1 |
| F25 | `GET /status` SHALL report `device`, `weights_verified`, `classifier_ready`, and which paths are live. | [DOC: REQUIREMENTS F25; design §8] | AC-F25.1 |
| F26 | The fall product SHALL own its sidecar overlay; `products/attendance/sidecar.yaml`, `products/barber/sidecar.yaml` and `perception/pipeline/sidecar.yaml` SHALL be byte-identical to `main`, asserted by a test. | [DOC: REQUIREMENTS F26; plan decision 6; ARCHITECTURE.md:422] | AC-F26.1 |
| F27 | `perception/engine/**` and `app/**` SHALL be untouched; the boundary ratchet test passes and its recorded counts do not increase. | [DOC: REQUIREMENTS F27; tests/test_engine_boundary.py] | AC-F27.1 |
| F28 | `just fall-up` SHALL refuse to start beside a live sidecar holding any of the fall cameras, and name it (preflight is the enforcement; the `roles` tag is not). | [DOC: REQUIREMENTS F28; sources.py:52-74; plan risk 5] | AC-F28.1 |
| F29 | The fall product SHALL have a golden config `tests/configs/golden-fall_detection.yaml` whose baseline contains the `pose` field with 17 keypoints per person. | [DOC: REQUIREMENTS F29; modules/README.md:120-121] | AC-F29.1 |
| F30 | The vendor's `lying` candidate SHALL alert from day one on the same route table (confirmed → `warning`, no review; below → `warning` + review); its evaluation is `lying`; its physical score is `(s_posture · s_still)^0.5` against the vendor's own `lying` thresholds, computed **from the vendor's built-in emission-moment kinematics** (ruling A7), trusted at the `kinematic` ceiling with a window, `cold` without — no new trust value. | [DOC: REQUIREMENTS F30; design §13.3; Phase 0 ruling A7] | AC-F30.1 … AC-F30.4 |
| F31 | A `lying` candidate SHALL be suppressed (not alerted, not queued — still scored) while its track has an alerted `fall` within `suppress_after_fall_s` (default 10 s); suppression is per track and per fall event, not a mute on the kind. | [DOC: REQUIREMENTS F31; design §13.3] | AC-F31.1, AC-F31.2 |
| F32 | The pilot's camera set SHALL be exactly `cam-1`; the 3-camera set is reachable by configuration alone (`configs/presets/fall_3cam/sidecar.yaml`). | [DOC: REQUIREMENTS F32; design §13.1] | AC-F32.1 |
| F33 | The fall product SHALL have its own frontend tab (registered in the products registry) drawing the engine-baked JPEG (`GET /stream/<cam>/frame`); the tab adds no Redis contract, no WebSocket, no box math. | [DOC: REQUIREMENTS F33; design §13.4; products.ts registry] | AC-F33.1, AC-F33.2 |
| F34 | The first release SHALL contain no WebSocket; the tab's JPEG layer and its (future) chip layer SHALL be separate, so slices B1/B2 land later without changing the tab's page, the engine, the app, the pose module, the monitor or the scoring. | [DOC: REQUIREMENTS F34; design §13.5] | AC-F34.1 |
| **F35** *(new, ruling A1)* | For **every** candidate type, the service SHALL fill the `action` path score with the reading's margin **whenever a classifier window existed when the candidate was emitted**; with no window the `action` slot stays `None`. | [Phase 0 ruling A1, refining design §6.4] | AC-F35.1, AC-F35.2 |
| **F36** *(new, ruling A2)* | A candidate whose evaluation is `unstable` SHALL route exactly like `lying`: confirmed → `warning` alert, no review; below threshold → `warning` alert + review row. | [Phase 0 ruling A2, extending design §5.3] | AC-F36.1 |
| **F37** *(new, ruling A3)* | The service SHALL ignore `fall_recovered` candidates: no alert, no review row, no persisted state; the behavior is pinned by a test. | [Phase 0 ruling A3; fall_monitor.py:850-857] | AC-F37.1 |

## Non-Functional Requirements

| ID | Requirement | Source | AC IDs |
|----|-------------|--------|--------|
| N1 | The sidecar's GPU cost of the feature SHALL be measured with `tools/engine_gpu_budget.sh` before/after; the pose stage's delta is recorded as the feature's cost. | [DOC: REQUIREMENTS N1; NEXT_MODULES.md:91] | AC-N1.1 |
| N2 | The pilot SHALL run at 12.5 Hz (`interval: 2`); the corpus replay at `interval: 2` vs `interval: 1` (action-path contribution + false alarms/hour) is a required experiment before shipping. | [DOC: REQUIREMENTS N2; plan decision 5] | AC-N2.1 |
| N3 | Decision-to-alert latency SHALL be under ~5 s end to end on replayed footage (the clip window is ±5 s). | [DOC: REQUIREMENTS N3; alert_clips.py:120] | AC-N3.1 |
| N4 | The action path's accuracy SHALL be measured on **our own** pose stream (precision, recall, false alarms/hour) using the corpus at `F:\fall_detection\data`; the harness takes the corpus root as a parameter and **skips when absent**. | [DOC: REQUIREMENTS N4; plan §Focus areas] | AC-N4.1 |
| N5 | The classifier's cost SHALL be measured on the card it shares with the pose GIE: ms/forward, concurrent tracks, whole-card view — **on the RTX 5090 deployment machine** (ruling A9; dev box Quadro RTX 3000 cannot answer it). Torch wheels for the deployment machine are cu128+ (`sm_120`); cu126 wheels lack Blackwell kernels. | [DOC: REQUIREMENTS N5; plan risk 4; Phase 0 ruling A9] | AC-N5.1 |
| N6 | The product SHALL be autonomous: no import of any other product's code; Redis shapes re-declared; its own yaml loader (borrowing only the `_merge`/`_read` pattern) and its own copy of the generic `connect()` helper. | [DOC: REQUIREMENTS N6; barber/acl.py:1-16] | AC-N6.1, AC-N6.2 |
| N7 | The scoring policy SHALL be unit-testable with no Redis, no torch, no GPU, no camera — covering the margin, every path's ceiling, the combination and every route. | [DOC: REQUIREMENTS N7; plan §The seam Option A] | AC-N7.1 |
| N8 | The vendor's own **nine** test files (Phase 0 correction; the doc says eight) SHALL pass against the copy under `just fall-test`, with the seam applied; the copied `conftest.py`'s path lines are wiring (ruling A10); tests needing absent weights/corpus skip cleanly. | [DOC: REQUIREMENTS N8; Phase 0 rulings A10, doc-inaccuracy 3] | AC-N8.1 |
| N9 | Every refused start-up SHALL exit non-zero with a message naming the cause. | [DOC: REQUIREMENTS N9; plan §The boot contract] | AC-N9.1 |
| N10 | The service SHALL never run in a degraded mode the operator cannot see: `/status` reports the classifier's state, the device and the live paths, and no code path runs fewer paths than reported. | [DOC: REQUIREMENTS N10; plan risk 3] | AC-N10.1 |
| N11 | **Dependency ruling (user, 2026-10-27):** the fall product's `pyproject.toml` adopts the surveillance repo's dependency conventions (uv-managed, repo's pins where they overlap); the vendor's non-conflicting deps are copied so the module works: **torch (exactly one version, cu128+ index for the deployment machine), mmengine, mmcv-lite, mmaction2** (imported as `mmaction`, tsstg_model.py:21-23 — runtime-required), **numpy, pandas, opencv, pyyaml**. `ultralytics` is a **test-only** dep (imported solely by the vendor CLI `run()`, fall_monitor.py:1054, and one vendor test, test_fall_monitor.py:230); `lapx` is excluded unless verification shows a test needs it. The resolved venv SHALL contain exactly one torch distribution — no duplicate or conflicting torch installs. **Amendment (user, 2026-10-27, Round 1):** the opencv distribution is **`opencv-contrib-python>=4.9.0`**, not `opencv-python` — mmaction2 1.2.0 hard-depends on the contrib build, and two opencv wheels install over the same `cv2` files; the repo's 4.9.0 floor is kept. | [Phase 1 dependency ruling + Round 1 amendment; vendor requirements.txt:11-18; plan slice S2, risk 4] | AC-N11.1, AC-N11.2 |

## Acceptance Criteria

Each AC is binary (pass/fail by running something). G/W/T transcribed from the requirements doc; ruling-derived ACs marked *(ruling)*.

### Pose module
- **AC-F1.1** *(reworded in Phase 2 audit — the config-testable half)* Given `products/fall_detection/sidecar.yaml`, when it is loaded through `perception.engine.sources.load_sidecar_config`, then the merged config enables the `pose` module declaring exactly one secondary GIE `sgie_pose` with its `interval`, and adds no new primary/full-frame inference stage.
- **AC-F1.2** *(new in Phase 2 audit — the live-evidence half, gated on Docker+GPU like N1)* Given the fall sidecar running with the `pose` module enabled, when a person is detected and tracked, then the pose GIE runs on that person's crop and the engine's log names `sgie_pose` and its interval; the evidence is a recorded golden run.
- **AC-F2.1** Given a frame with two tracked people, when the entry is published on `iris:meta:<cam>`, then it carries a `pose` field with one item per person with confident keypoints, and no other Redis stream is written for the pose.
- **AC-F3.1** Given a golden replay of one frame, when the `pose` field is read, then `shape` equals the capture size; every item's `id` is a tracker id present in the frame's `dets`; every `box` has `x1 < x2`, `y1 < y2` and lies inside the capture rectangle; every `kps` has 17 entries of 3 numbers.
- **AC-F4.1** Given a person whose network-input keypoints are known, when the module publishes, then the keypoints equal the expected capture-pixel positions within 2 px per coordinate, and every keypoint lies inside the published `box` expanded by 10 %.
- **AC-F5.1** Given a frame whose SGIE items carry no `parent_id`, when the module publishes, then each item's keypoints belong to the person whose box contains them, and no item is dropped for a missing parent pointer.
- **AC-F6.1** Given the module folder, when `pytest tests/test_module_requirements.py` runs, then it passes, and a yaml key the module does not declare refuses startup.

### Monitor and seam
- **AC-F7.1** Given `products/fall_detection/fall_detection/`, when diffed against `F:\fall_detection_module\src`, then the trees differ only by the F8 seam edit and `PROVENANCE.md`.
- **AC-F8.1** Given a fixture that produces a fall candidate, when run through the unmodified vendor code and through the patched copy, then both produce the same candidates (ids, times, types), and the patched one additionally carries `reading` (`probs`, `top`) and `path_scores` (`action`, `kinematic`, `cold`).
- **AC-F8.2** *(ruling A4, corrected in Phase 3 Round 2)* Given a fall that latches at t₀ and is packaged at t₀+Δ, when the candidate is emitted, then its `path_scores.kinematic`/`cold` equal the collapse margins over the **latched path's own gate** (`path_b` / `path_c`) computed from the candidate's **packaging-time** kinematics, and its `reading` is the smoothed vector at packaging time.

  > **Phase 3 Round 2 correction (measured, pending the user's ratification).** A4 ruled the
  > physical scores at **latch** time, on the premise that by packaging the collapse "has slid out
  > of the 2.5 s rolling window". Measurement on the vendor's own fixture refutes both halves:
  > the packaging happens **0.48–0.56 s** after the latch (`post_s` = 3 s is the latch's *maximum*
  > > lifetime; the candidate is packaged as soon as a 0.5 s confirmed-posture run with low motion
  > > exists), so the window still remembers the collapse; and a gate fires **at the crossing**, so
  > > the crossed quantity's margin there is zero by construction — the geometric mean cannot average
  > > a zero away. Measured collapse margins on the same fall: **latch 0.32 (path B) / 0.00
  > > (path A)** vs **packaging 0.92**. The latch-time reading left the kinematic path structurally
  > > worthless (its 0.50 ceiling unreachable), and the packaging-time reading is the same moment
  > > and the same quantities the lying and immobility scores already use (ruling A7).
- **AC-F9.1** Given the fall product's folder, when the copied package is imported, then it imports without `FileNotFoundError`, the vendor's thresholds come from `products/fall_detection/config.yaml`, and the product's policy from `fall_detection.yaml`.
- **AC-F10.1** Given the vendor's config, when the physical score is computed, then its thresholds are the same numbers the boolean gate uses (`path_b` for kinematic, `path_c` for cold), read from that same config — not a second copy.
- **AC-F10.2** *(ruling A6)* Given a fresh copy, when the spans are read, then all seven (`drop_span`, `vdown_span`, `hip_span`, `angle_span`, `aspect_span`, `motion_span`, `disp_span`) exist in `config.yaml`, each defaulting to its gate's own threshold (a quantity at 2× its threshold scores 1.0), and each is overridable without a code change.
- **AC-F10.3** Given the four physical margins of a collapse (`s_drop`, `s_vdown`, `s_hip`, `s_posture`), when the collapse score is aggregated, then it is their **geometric mean** (`(s_drop · s_vdown · s_hip · s_posture) ^ 0.25`) — the one aggregate where a single collapsed condition cannot be averaged away, which is what a conjunction means.

### Scoring policy
- **AC-F11.1** Given an `immobility` candidate whose reading's top class is `Fall`, when the evaluation is computed, then it is `falling`, not `immobile`.
- **AC-F11.2** Given a `fall` candidate whose reading's top class is `Lying`, when the evaluation is computed, then it is `lying`.
- **AC-F12.1** Given `p[6]=0.45` vs `p[3]=0.42` and `margin_span=0.80`, when the reading is computed, then it is ≤ 0.05.
- **AC-F12.2** Given `p[6]=0.95` vs `p[3]=0.02`, when the reading is computed, then it is ≥ 0.90.
- **AC-F13.1** Given a candidate whose only path is cold with physical score 1.0, when combined with `trust.cold=0.40`, then the score is 0.40 — below 0.70 — and the candidate is delegated.
- **AC-F13.2** Given a candidate with an action score of 0.95 and no other path, when combined, then the score is 0.95 — at/above threshold — and the candidate is auto-confirmed.
- **AC-F14.1** Given `trust.action=0.60` in config, when the combined score is computed, then the action path's ceiling is 0.60 and an action-only candidate cannot reach a 0.70 threshold.
- **AC-F15.1** Given the product's config and the vendor's copy, when both are searched for the class groups (`falling: [6]`, `upright: [0,1,2,4,5]`, …), then they are defined in the product's config only.
- **AC-F35.1** *(ruling A1)* Given a fall candidate latched by Path B only (no Path A) with a classifier window at emission and a decisive reading, when scored, then the `action` slot is filled with the margin and the candidate can reach the threshold.
- **AC-F35.2** *(ruling A1)* Given any candidate emitted with no classifier window, when scored, then the `action` slot is `None` and the combined score cannot reach `confirm_threshold` (cold ceiling 0.40).

### Routes, alert, clip
- **AC-F16.1** Given combined 0.90 and evaluation `falling`, when routed, then a `critical` alert is raised immediately and no review row is created.
- **AC-F16.2** Given combined 0.50 and evaluation `falling`, when routed, then a `warning` alert is raised immediately and a review row is created.
- **AC-F16.3** Given combined 0.90 and evaluation `upright`, when routed, then no alert and no review row — a confirmed false alarm, closed.
- **AC-F16.4** Given combined 0.90 and evaluation `lying`, when routed, then a `warning` alert is raised immediately and no review row is created.
- **AC-F36.1** *(ruling A2)* Given a candidate evaluated `unstable`, when routed, then it behaves exactly like `lying`: confirmed → `warning`, no review; below threshold → `warning` + review.
- **AC-F17.1** Given a confirmed fall, when the alert is written, then the entry carries `v=1`, `kind=fall`, `severity ∈ {critical, warning}`, `category=safety`, a `message` naming the evaluation and the confidence, and `pts` — and the app turns it into an event with no module code.
- **AC-F18.1** Given a confirmed fall at the frame whose `pts` is T, when the clip is requested, then `iris:svr_request:<cam>` carries `pts=T` and the sidecar's clip covers the fall.
- **AC-F19.1** Given a frame published with `pts=1234.5`, when the monitor is updated, then it is given `t=1234.5`, and the alert and the clip request carry the same clock.
- **AC-F20.1** Given two cameras reporting the same track id, when both are monitored, then neither camera's track state is affected by the other.
- **AC-F21.1** Given a camera whose worker is slower than the frame rate, when the queue is full, then the oldest frame is dropped, the queue never grows without bound, and the monitor keeps running.
- **AC-F22.1** Given a candidate below the threshold, when the service restarts, then the review row is still there with the evaluation, combined score, per-path scores and the reading's top class.
- **AC-F22.2** Given a review row, when marked `acknowledged`/`confirmed`/`dismissed`, then the status is stored with the operator's note, and no other status is accepted.
- **AC-F30.1** Given a `lying` candidate at/above `confirm_threshold`, when routed, then a `warning` alert is raised and no review row is created.
- **AC-F30.2** Given a `lying` candidate below the threshold, when routed, then a `warning` alert is raised and a review row is created.
- **AC-F30.3** Given a `lying` candidate, when scored, then its evaluation is `lying` and its physical score is `(s_posture · s_still)^0.5` against the vendor's `angle_in`/`aspect_in`/`max_motion`, trusted at the kinematic ceiling with a window and the cold ceiling without.
- **AC-F30.4** *(ruling A7)* Given the lying score's inputs, when computed, then they are the vendor's own emission-moment kinematics (the values its evidence block carries) — `detect_lying`'s contract is not changed.
- **AC-F31.1** Given a track that produced an alerted `fall` within `suppress_after_fall_s` (10 s), when a `lying` candidate for that track is scored, then it is not alerted and not queued.
- **AC-F31.2** Given a track whose alerted `fall` is older than `suppress_after_fall_s`, when a `lying` candidate is scored, then it alerts normally — suppression is per track and per fall event, not per kind.
- **AC-F37.1** *(ruling A3)* Given a `fall_recovered` candidate arrives from the monitor, when the worker processes it, then no alert is written, no review row is created, and nothing is persisted for it.

### Boot contract
- **AC-F23.1** Given a missing checkpoint, when the service starts, then it exits non-zero naming the missing file.
- **AC-F23.2** Given a corrupted checkpoint, when the service starts, then it exits non-zero naming the checksum mismatch.
- **AC-F23.3** Given `classifier.device: cuda:0` on a machine with no GPU, when the service starts, then it exits non-zero naming the device — and does not fall back to the CPU.
- **AC-F24.1** Given a torch build whose CUDA reports available but whose kernels do not match the card, when the service starts, then the check (a real CUDA op) fails and the service refuses to start.
- **AC-F25.1** Given a running service, when `GET /status` is called, then the response carries `device`, `weights_verified`, `classifier_ready` and which paths are live.

### Deployment
- **AC-F26.1** Given the change, when `git diff` is inspected, then `products/attendance/sidecar.yaml`, `products/barber/sidecar.yaml` and `perception/pipeline/sidecar.yaml` are byte-identical to `main`, and a test asserts it.
- **AC-F27.1** Given the change, when `pytest tests/test_engine_boundary.py` runs, then it passes and the recorded ratchet counts have not gone up.
- **AC-F28.1** Given a live attendance sidecar holding one of the fall cameras, when `just fall-up` runs, then it refuses to start and names the sidecar it found.
- **AC-F29.1** Given the fall product, when the golden replay runs, then `tests/configs/golden-fall_detection.yaml` produces its baseline, and the baseline contains the `pose` field with 17 keypoints per person.
- **AC-F32.1** Given the pilot deployment, when its camera set is read, then it is exactly `cam-1` (the 3-camera preset exists at `configs/presets/fall_3cam/sidecar.yaml` by configuration alone).

### Frontend tab
- **AC-F33.1** Given the fall product's tab, when opened for a camera, then it shows `GET /stream/<cam>/frame` and adds no Redis contract, no WebSocket and no box math.
- **AC-F33.2** *(reworded in Phase 2 audit — the next-release half was untestable in this delivery)* Given the fall product's tab and the dashboard wall, when both render a camera, then they draw the **same** engine-baked frame endpoint (`GET /stream/<cam>/frame`), so a condition baked onto a box by the engine appears identically in both.
- **AC-F34.1** Given the first release, when it ships, then no WebSocket exists and the tab's JPEG layer and chip layer are separate, so the scene slices (B1/B2) can land later without changing the tab's page.

### Non-functional
- **AC-N1.1** Given the golden replay before/after the module is enabled, when `tools/engine_gpu_budget.sh` runs, then each stage's busy% is printed and the pose stage's delta is recorded.
- **AC-N2.1** Given the fall corpus, when replayed at `interval: 2` and `interval: 1`, then both runs report the action path's contribution and false alarms/hour, and the comparison is written down before shipping.
- **AC-N3.1** Given a fall in replayed footage, when measured end to end, then onset-to-alert is under 5 s.
- **AC-N4.1** Given the corpus at `F:\fall_detection\data` (skipped when absent), when replayed through sidecar → `pose` → service, then precision, recall and false alarms/hour are reported, and the candidate is on the right camera and track.
- **AC-N5.1** Given the service running on the **RTX 5090 deployment machine** with N concurrent tracks, when forwards are timed, then ms/forward, concurrent tracks and the whole-card view are recorded.
- **AC-N6.1** Given the fall product's code, when its imports are listed, then no `products.<other>` module is among them.
- **AC-N6.2** Given the product's config loader, when read, then it loads `fall_detection.yaml` and nothing from another product's folder — in particular it does not extend the barber's yaml.
- **AC-N7.1** Given the scoring module, when unit tested, then the tests need no Redis, no torch and no GPU, and cover the margin, each path's ceiling, the combination and every route.
- **AC-N8.1** Given the copied package, when `just fall-test` runs, then the vendor's nine test files pass against the copy with the seam applied (weight/corpus-gated tests skip when absent).
- **AC-N9.1** Given each refused start-up in AC-F23.1–3 and AC-F24.1, when the service exits, then the message names the cause and the exit code is non-zero.
- **AC-N10.1** Given a running service, when `/status` is read, then the classifier's state, device and live paths are reported, and no code path runs with fewer paths than reported.
- **AC-N11.1** Given `products/fall_detection` synced with uv, when the resolved environment is listed, then exactly one torch distribution is installed and its build matches the pinned index (cu128+ on the deployment machine); `import fall_detection` + `TSSTGClassifier` construction succeed against it. *(ruling)*
- **AC-N11.2** Given the product's `pyproject.toml`, when read, then `ultralytics` appears only in the test/dev dependency group (not runtime), and the vendor's nine test files still pass under `just fall-test` with that arrangement. *(ruling)*

## Explicitly Out of Scope (will NOT be implemented in this delivery)

**The next release (designed, deferred — the first release must not preclude it):**
- X1–X8: the live annotation overlay — `iris:fall_state:<cam>` publication (slice N1), `perception/modules/fall_overlay/` (N2), the magenta/priority-12 styling, hold/clear behavior (N3). F34 pins the separability now; nothing else.
- Option B slices B1/B2: `/fall_detection/ws` scene WebSocket and DOM chips.

**From the docs' out-of-scope lists (design §12, requirements §5):**
- two products on the same cameras; ST-GCN fine-tuning / transfer gate; automatic CPU fallback; `pipeline_preview: false`; skeleton or keypoints in the UI; a DOM overlay on the platform's wall; any change to `app/**`, `perception/engine/**`, or another product's `sidecar.yaml`; a live skeleton overlay on attendance tiles.

**From the Phase 0 rulings:**
- pushing the seam edit upstream to `F:\fall_detection_module` (ruling A5 — the vendor checkout stays pristine; upstream is a later user-driven step);
- alerting or persisting `fall_recovered` (ruling A3);
- F31's overlay-visible half ("the box still carries the lying condition") belongs to X-series/next release — in this delivery suppression silences alert + review only.

## Deferred Ambiguities — Resolution Status

All ten Phase 0 ambiguities were **resolved by the user on 2026-10-27** (phase-0-reading-notes.md, "Rulings"); none remains deferred. Binding outcomes folded into the tables above: A1→F35, A2→F36, A3→F37, A4→AC-F8.2, A5→F7, A6→AC-F10.2, A7→AC-F30.4, A8→F22's surface (queue in the product's own tab), A9→N5, A10→N8.

**Environment facts of record (ruling A9):** deployment GPU = RTX 5090 (`sm_120`, torch cu128+); dev machine = Quadro RTX 3000 6 GB (verified by `nvidia-smi`) — GPU-dependent positive paths are verified on the deployment machine; refusal paths (F23/F24) are testable anywhere by hiding CUDA.

**Dependency ruling of record (user, 2026-10-27 → N11):** the fall product adopts the surveillance system's requirements/wheel conventions; non-conflicting vendor deps are copied; exactly one torch version is installed. Verified by import audit: `mmaction` (mmaction2) is runtime-required by tsstg_model.py:21-23; `ultralytics` is imported only inside the CLI `run()` (fall_monitor.py:1054) and one vendor test (test_fall_monitor.py:230), so it is test-only; no file imports `lapx`/`lap` directly.

**Doc errata carried into Phase 1** (Phase 0 §"Minor doc inaccuracies"): compose config passes via `command:` like the barber's (not a `SIDECAR_CONFIG` env); N8 counts nine vendor test files, not eight; the publisher's label is lowercase `person …`.
