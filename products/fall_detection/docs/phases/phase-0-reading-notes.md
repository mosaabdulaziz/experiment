# Phase 0: Reading Notes

**Date:** 2026-10-27 · **Workflow:** rigorous-software-development · **Status:** CLOSED — all ten questions ruled by the user 2026-10-27 (see "Rulings" at the bottom)

## Documents Read

| Document | Path | Purpose |
|---|---|---|
| the design | `products/fall_detection/docs/FALL_DETECTION_DESIGN.md` (read at the repo root, commit `e4653ff`) | component-by-component design, C1–C8; §13 carries the binding rulings and supersedes §§1–12 where they conflict |
| the requirements | `products/fall_detection/docs/FALL_DETECTION_REQUIREMENTS.md` (read at the repo root, commit `e4653ff`) | F1–F34, N1–N10, X1–X8 with Given/When/Then acceptance criteria |
| the working plan | `C:\Users\Maha Mamdouh\.qoder\plans\polished-glen-newt.md` (outside the repo) | 16 locked decisions, the build order S0–S6 + N1–N3, risks |
| the vendor's code | `F:\fall_detection_module\src\` (7 files, ~3,100 lines) + `config.yaml` + `tests/` | the `FallMonitor`, the keypoint adapter, the TSSTG classifier |
| the second vendor checkout | `F:\fall_detection\` (commit `76f7427`) | weights + corpus; `src/` verified byte-identical to `F:\fall_detection_module\src` (only `__pycache__` differs) |

## Verification Performed (evidence, not trust)

Every load-bearing code citation in the docs was checked against the actual files:

**IRIS repo — all CONFIRMED** (highlights):
- the engine's `emit` is a bare `xadd` (`perception/engine/gst_build.py:418-424`); the publisher JSON-encodes module fields (`perception/engine/publisher.py:348-349`) — the design's refinement 9 is correct.
- `landmarks_to_capture` (`perception/modules/face_detect/__init__.py:60-76`), the instance-mask payload (:48-57), the unset `parent_id` finding (:64-65), the `GieSpec(role="secondary")` declaration (:126-130), the containment match (`owner.py:117-158`) — all exist as described.
- `services.alert()` field set (`perception/engine/modules.py:99-111`), `Module.control()` (:176-178), the `ControlPoller` fires only on change at 1 Hz (`perception/engine/control.py:38-42`), `FrameOutput.annotate(..., priority=)` (`perception/engine/frame.py:126-127`) — confirmed; `color` is any BGR tuple, so magenta is available.
- `GieGate.install` returns `None` under `pipeline_preview` (`gie_gate.py:89-93`) — confirmed.
- the barber's patterns: `_Cursor.read` (`acl.py:163-169`), `_merge`/`_read` with the extends cycle guard (`config.py:64-79`), `connect()` with WAL + busy timeout (`storage.py:27-38`), `create_app` (`server.py:138`) — confirmed.
- the sidecar overlay loader (`sources.py:185-207`, deep merge, lists replace outright) — confirmed.
- `tests/test_product_declarations.py`, `tests/test_module_requirements.py`, `tests/test_engine_boundary.py` (ratchet file `tests/engine_boundary_ratchet.json`), `tests/configs/golden-barber.yaml` + baseline — all exist.
- the parser auto-build (`perception/pipeline/Dockerfile:50-52`) — confirmed; `BASE_MODULES` only installs pip requirements, module code ships via `COPY perception/modules/`.
- telemetry port **9105 is free** (9100–9104 taken) — confirmed.
- `perception/NEXT_MODULES.md` reserves **magenta, priority 12** for fall (:109) — confirmed.
- `GET /stream/{source_id}/frame` (`app/server.py:847-875`), the closed severity set (`app/severity.py:8-9`), the SVR contract (`app/usecases/recordings.py:69-70`, `perception/engine/main.py:258-289`), `pad_seconds: 5.0` (`app/alert_clips.py:120`) — confirmed.

**Vendor code — all CONFIRMED:**
- `ROOT`/`load_cfg()` at import (`fall_monitor.py:58-72`); the bare `except Exception: self.classifier = None` (:459-469); `review_event` with the three statuses (:511-522); the smoothed 8-class vector (:684-689); the path latch (:740-772); the candidate packaging `rec` (:999-1009); `update_ref_scale` reads xyxy (:120-151); `detect_lying` (:303-319); the immobility candidates (:887-893, :914-920, :928-934); `fall_recovered` (:850-857).
- `prenormalize_2d` normalizes against `img_shape=(h,w)` (`keypoint_adapter.py:111-128`); `resample_window_25hz` needs ≥6 observations, span ≥1.16 s, no gap >1.0 s (:62-108).
- `WEIGHT_SHA256_REGISTRY` (`tsstg_model.py:122-131`); `TSSTGClassifier(weight_dir, variant, streams, num_classes, device, projection_cfg)` (:326-335); `FallMonitor(classifier=...)` is accepted and bypasses the bare except (:459-460).
- `FallMonitor` defaults: `img_shape=(240,320)` = (h,w), `buffer_s=30`, `eval_every_s=0.5`, `device="cpu"` — all as the docs describe.
- `config.yaml` has `lying {angle_in: 35.0, aspect_in: 0.50, max_motion: 1.0}`, `path_a/b/c` blocks, `immobility.disp_thresh: 0.035` — confirmed. It has **no span keys yet** (they are the design's addition).
- **The two weights hash-match the registry exactly** (fresh `sha256sum`: `484a394a…` joint, `f18eb408…` jm).

**Minor doc inaccuracies found (none load-bearing):**
1. the barber compose service passes its config path in `command:` (`${BARBER_SIDECAR_CONFIG:-…}`), not through a `SIDECAR_CONFIG` env var, and its telemetry port is 9103 — the fall service should copy the actual mechanism (`command:` + `FALL_SIDECAR_CONFIG`), port 9105.
2. the publisher's box label is lowercase `person 0.95 #7` (the raw class label), not `Person 0.95 #7` as X2 quotes — the AC's intent is unaffected.
3. N8 says "eight test files"; the vendor ships **nine** (`test_adversarial_critiques`, `test_config_source_of_truth`, `test_evaluate_benchmark`, `test_fall_monitor`, `test_keypoint_adapter`, `test_omnifall_adapter`, `test_phase1_improvements`, `test_transfer_gate`, `test_tsstg_model`) plus `conftest.py` and `synth.py`.
4. §5.1's "a consumer that is back inside 30 s loses nothing that matters" overstates it: `iris:meta:<cam>` has `maxlen≈64` (~2.5 s of frames), so any restart loses in-flight history; the true mitigation is that the monitor's resampler rejects windows with >1 s gaps and re-accumulates — the behavior is safe, the sentence is loose.

## Components Identified

| # | Component | Responsibility | Inputs → Outputs | Key constraints |
|---|---|---|---|---|
| C1 | pose model (`perception/models/DeepStream-Yolo-Pose/`) | yolo11n-pose → ONNX with `[N,4]` boxes, `[N]` scores, `[N,17,3]` kps in network-input px (the exported file carries a batch axis in front of each — corrected in Round 4); C++ parser `NvDsInferParseYoloPose` packs 51 floats + net size into the instance-mask field | ONNX export ← `export_pose.py`; parser ← face parser pattern | numeric equality vs `model.predict()`; letterbox `maintain-aspect-ratio=1`, `symmetric-padding=1` |
| C2 | pose module (`perception/modules/pose/`) | secondary GIE `sgie_pose`; crop→capture mapping; publishes `pose` field | instance mask → `obj.attrs["keypoints_raw"]` → capture px → `out.fields["pose"]` | capture-px contract (F4); containment ownership, no `parent_id` (F5); `interval: 2`; Settings-declared yaml keys (F6) |
| C3 | vendored monitor (`products/fall_detection/fall_detection/`) | copy of `F:\fall_detection_module\src` + one additive seam edit + `PROVENANCE.md`; `config.yaml` (vendor's + spans) beside it | candidates gain `reading` and `path_scores` | package one level below product root (import-time `ROOT/config.yaml`); boolean gates untouched (F8); vendor tests run against the copy (N8) |
| C4 | fall product service (`products/fall_detection/`) | host process: Redis consumer → bounded per-camera queues → per-camera workers (`FallMonitor.update`) → scoring → routes; FastAPI (`/status`, `/candidates`, `/review/<id>`); SQLite review queue | `iris:meta:<cam>` `pose` field → alerts, clip requests, review rows | fork the barber's shape, import nothing from other products (N6); one monitor per camera (F20); frame `pts` is the clock (F19); boot gate (F23–F25) |
| C5 | scoring policy (`products/fall_detection/scoring.py` + `fall_detection.yaml`) | evaluation lookup, margin, combination `1-Π(1-trust·score)`, routes, suppression | candidate + policy → evaluation, combined score, route | pure functions, no I/O (N7); class sets in exactly one place (F15); trust weights are config (F14) |
| C6 | boot contract | verify weights (SHA-256), real CUDA op, build classifier, hand to `FallMonitor(classifier=…)`; refuse to start otherwise | config → live classifier or non-zero exit | no CPU fallback, no degraded mode (F23, F24, N9, N10) |
| C7 | deployment | `products/fall_detection/sidecar.yaml` (cam-1, role `fall_detection`, pose module), compose service `deepstream-sidecar-fall` (profile `fall`, telemetry 9105), `just fall-up/down/test/stgcn-model` with the exclusivity preflight, `configs/presets/fall_3cam/`, golden config | — | other products' yamls byte-identical (F26); engine boundary ratchet unchanged (F27) |
| C8 | frontend tab (`frontend/src/products/fall_detection/`) + registry entry | draws `GET /stream/<cam>/frame` (engine-baked JPEG) | — | no WebSocket, no box math in the first release (F33, F34) |
| (next release) | `iris:fall_state:<cam>` + `perception/modules/fall_overlay/` | per-track state published by the service; engine-side annotation via `control()` | state key → magenta label at priority 12 | X1–X8; **not in the first release's scope** |

## Data Flows

1. **Frame → keypoints:** camera → pgie(person) → tracker → `sgie_pose` (interval 2) → parser → instance mask → module `extract`/`transform` (crop→capture px) → `analyze` → publisher JSON-encodes → `XADD iris:meta:<cam>` field `pose` (maxlen ≈64).
2. **Keypoints → decision:** consumer thread `XREAD` (tail on restart) → parse `pose` → bounded per-camera queue (drop oldest) → worker: `FallMonitor.update(pts, ids, boxes_xyxy, kps)` → candidates (fall / lying / immobility / fall_recovered) → `scoring.score(candidate, policy)` → evaluation + combined score.
3. **Decision → routes:** confirmed → `XADD iris:alert:<cam>` (kind=fall, severity critical|warning, category safety) + `XADD iris:svr_request:<cam> {pts}`; below threshold → warning alert + clip + SQLite review row; confirmed `upright` → closed silently; suppressed lying-after-fall → scored, not alerted, not queued.
4. **Alert → app (unchanged):** `app/usecases/module_alerts.py` turns the entry into a platform event; the clip resolves around `pts ± 5 s`.
5. **Next release:** service → `SET iris:fall_state:<cam>` → ControlPoller (1 Hz, own thread) → `fall_overlay` module → `annotate(magenta, priority 12)` → baked JPEG → `/stream/<cam>/frame` → dashboard wall + the fall tab.

## Explicit Constraints (verbatim anchors)

- "Nothing in `perception/engine/` changes. Nothing in `app/` changes. No other product's `sidecar.yaml` changes." (design §1) — enforced by F26, F27 and the ratchet test.
- "The `pose` field is written in capture pixels" (design §3.2) — box xyxy, kps 17×3, `shape=[capture_w, capture_h]`; the service passes `img_shape=(shape[1], shape[0])`.
- "The vendor's code is copied into the product, not depended on" (decision 4); "nothing in the product imports another product's code — the shapes are re-declared" (N6).
- "The pilot starts at 12.5 Hz (`interval: 2`); 25 Hz is a required experiment before shipping" (decision 5, N2).
- "The threshold decides whether a human is needed, not how urgent the event is" (decision 2); severity: critical for confirmed falling/immobile, warning for lying and for everything delegated.
- `confirm_threshold: 0.70`, `margin_span: 0.80`, trust action/kinematic/cold = 1.00/0.50/0.40 (decision 13, F13, F14) — configurable, measured before shipping.
- "The classifier is mandatory at boot… no GPU = refuse to start… the check runs a real CUDA op" (F23, F24).
- Pilot = `cam-1` only (F32); `suppress_after_fall_s: 10` (F31); `state_hold_s: 10` (X5, next release).
- The shipped config is `cuda:0`; a GPU-less dev box may configure `cpu`, but there is no automatic fallback (out of scope).
- Vendor deps: torch (CUDA wheel, index pinned), mmengine, mmcv-lite, mmaction2; the vendor supports Python 3.10–3.12 and was verified working on 3.14.4 + torch 2.14.1+cu126 in `F:\fall_detection\.venv`.

## Ambiguities and Contradictions

Ten items were found and put to the user; all ten were ruled on 2026-10-27 (see Rulings). The full analysis is preserved here as the reasoning of record.

### A1. When does the `action` path speak for a **fall** candidate? (the one real scoring gap)

The docs give two rules that disagree at the margin:
- design §6.4's table: the action path "speaks" when **Path A latched** (top class Fall, full window, kinematic trigger).
- design §6.4's immobility paragraph and §13.3's lying rule: the reading counts **"when a window exists"** — no Path-A-style gate.

The disagreement is observable: a fall candidate latched by **Path B only** (kinematic collapse with weak classifier support — Path B does *not* require the top class to be Fall) has a full window and a reading. Under reading (a) its action slot stays `None`, the combined score is capped at 0.50, and it is **always delegated** even if the classifier is 95 % decisive about the evaluation. Under reading (b) the decisive reading confirms it.

**Recommendation (ACCEPTED):** (b) — fill the action slot whenever a window existed when the candidate was emitted, for every candidate type. The cold path's ceiling (0.40) still guarantees a no-window candidate is always delegated. Tradeoff accepted: a fall candidate Path A refused can still be auto-confirmed by a decisive reading — which is decision 8's intent.

### A2. The `unstable` evaluation has no route

`evaluations.unstable: [7]` exists in the policy config (design §6.1), and a fall candidate **can** be evaluated `unstable`: Path B accepts `p_unstable ≥ 0.35` as action support, so a latched fall whose reading's top class is Instability evaluates to `unstable`. The route table (§5.3, F16) has rows for falling / immobile / lying / upright only.

**Recommendation (ACCEPTED):** a confirmed `unstable` routes exactly like a confirmed `lying` — `warning` alert, no review; below threshold — `warning` + review. Not folded into `upright`'s silent close: a staggering person is precisely who should not be silently closed.

### A3. The vendor emits a fourth candidate type the docs never mention: `fall_recovered`

`fall_monitor.py:850-857`: when the fall latch expires and the person is upright again, the monitor emits `type='fall_recovered'`, `priority='low'`, `status='recovered'`. No route, no scoring rule, no requirement covers it. It **will** arrive at the service.

**Recommendation (ACCEPTED):** the service **ignores** it (no alert, no review row, no state), pinned by a unit test. It is the vendor's audit signal, not an event.

### A4. When are the fall candidate's physical scores computed — at latch time or at packaging time?

A fall candidate latches at t₀ (paths speak, `pending_fall_t0` set) but is packaged up to `post_s = 3 s` later, in a **different `update()` call**, after post-fall stability confirmation. The gate quantities (`curr_drop`, `curr_vpk`, `curr_hr`, posture) at packaging time are the *recumbent* person's, not the *collapse's* — the collapse has slid out of the 2.5 s rolling window, so computing the margin at packaging time would score every real fall near zero.

**RULING (user + correction):** the **reading** is attached at packaging/alert time (the current `probs_smooth`, as §4.2 describes) — the user confirmed this ("register the action when the system fires its alert"). The **physical (kinematic/cold) scores are computed at latch time**, stored on the track beside `d['fall_paths']`, and copied onto the `rec` at packaging — this half is the agent's technical correction, explained to and left unopposed by the user; it is non-negotiable for correctness. Timing facts confirmed for the user: candidate lands ~1–4 s (typically ~3 s) after fall onset; the clip (±5 s around the alerting frame's `pts`) contains the fall itself.

### A5. Where is the seam edit physically made — vendor checkout, copy, or both?

F7's AC (copy differs from vendor **only** by the seam) is only testable if the vendor checkout stays clean.

**Recommendation (ACCEPTED):** copy first, edit only inside `products/fall_detection/fall_detection/`; `F:\fall_detection_module` stays untouched as F7's diff baseline; the upstream push is a separate, later, user-driven step, out of this task's scope. The span keys are added to the **copied** `products/fall_detection/config.yaml` only.

### A6. The span values are specified nowhere

The score formulas name seven spans (`drop_span`, `vdown_span`, `hip_span`, `angle_span`, `aspect_span`, `motion_span`, `disp_span`) but no document gives numbers.

**Recommendation (ACCEPTED):** ship defaults of **span = the gate's own threshold** (score 1.0 at 2× threshold), all seven config keys in the copied `config.yaml` — explicitly a guess until the N4 corpus replay calibrates them ("until we benchmark and tune them" — the user).

### A7. The lying candidate's physical score: which angle/aspect/motion values?

`detect_lying` decides on **run statistics**, but §13.3's formulas don't say whether the score reads run values or emission-moment values.

**RULING (user):** "use the built-in system's calculations, following the rule of minimal code changes" — i.e. the emission-moment kinematics the vendor's own evidence block already carries (`curr_angle`, `curr_aspect`, `motion` at packaging). Matches the recommendation: one packaging point, additive seam, no change to `detect_lying`'s return contract. Tradeoff accepted: a person rising at the exact emission frame scores lower; bounded by the 5 s cooldown.

### A8. The review queue's operator surface — the doc's own open question

**Recommendation (ACCEPTED):** the queue lives in the fall product's **own tab** (list with acknowledge/confirm/dismiss). Does not block the service; the API and the three statuses are identical either way.

### A9. Which GPU is the deployment machine?

The plan's risk 4 said "RTX 5090"; `perception/NEXT_MODULES.md:91` says "RTX 5050 8 GB".

**RULING (user):** the deployment GPU is the **RTX 5090**. Agent verification (fresh `nvidia-smi`): **this dev machine is a Quadro RTX 3000, 6 GB** — so dev ≠ deployment, and `NEXT_MODULES.md`'s 5050 claim is stale. Consequences recorded for Phase 1/2:
1. **cu126 wheels do not carry `sm_120` (Blackwell) kernels** — the vendor venv's torch `2.14.1+cu126` would pass `cuda.is_available()` and fail at the first matmul on the 5090 (exactly the plan's risk 4). The fall product's torch index must be pinned to a **cu128+** wheel for the deployment machine; the dev box (Turing `sm_75`) runs either.
2. **GPU-dependent gates (N1, N5, the positive boot path) can only be verified on the 5090 machine.** The refusal paths (F23/F24) are testable anywhere by hiding CUDA, and the test suite will be designed that way.

### A10. Vendor-test wiring under `just fall-test`

The vendor's `pytest.ini` uses `pythonpath = . src`; its `conftest.py` inserts `ROOT`, `ROOT/src`, `ROOT/tests` into `sys.path`; the copy's package dir is `fall_detection/`, not `src/`.

**Recommendation (ACCEPTED):** the fall product gets its own pytest config; the copied `conftest.py`'s path lines are wiring, not a vendor-code edit (`tests/` sits outside the `src/` tree F7 diffs). Weight/corpus-gated vendor tests skip cleanly when absent, mirroring N4's rule.

## Scope Boundary (What Is NOT in the Docs — will not be built)

From the docs' own out-of-scope lists (design §12, requirements §5), plus what Phase 0 confirms nothing asks for:
- two products on the same cameras; ST-GCN fine-tuning / transfer gate; automatic CPU fallback; `pipeline_preview: false`; skeleton or keypoints in the UI; a DOM overlay on the platform's wall; any change to `app/**`, `perception/engine/**`, or another product's `sidecar.yaml`.
- the next release (X1–X8: `iris:fall_state`, `fall_overlay` module, state hold/clear, unconfirmed styling) — **designed, not built now**; the first release must merely not preclude it.
- Option B's WebSocket (`/fall_detection/ws`) and DOM chips — later slices B1/B2 (F34 pins the separability now).
- Not asked for anywhere, so not built: pushing the seam upstream (A5), alerting `fall_recovered` (A3), any multi-site/branch config for fall, any auth on the fall API beyond what the barber's pattern provides, i18n for the tab beyond the registry entry's shape.

## Rulings (2026-10-27, user)

| # | Ruling |
|---|---|
| A1 | **Confirmed** — action slot filled whenever a window existed at candidate emission, for every candidate type |
| A2 | **Confirmed** — `unstable` routes like `lying` (warning; review when below threshold) |
| A3 | **Confirmed** — `fall_recovered` ignored, pinned by test |
| A4 | **Reading at alert time** (user); **physical scores at latch time** (agent's correction, unopposed — required for correctness); the ~3 s detection latency and clip-contains-the-fall behavior confirmed to the user |
| A5 | **Confirmed** — edit only the IRIS copy; vendor checkout pristine; upstream push out of scope |
| A6 | **Confirmed** — span defaults = the gate's own threshold, configurable, "until we benchmark and tune them" |
| A7 | **Confirmed** — the vendor's built-in emission-moment calculations (evidence kinematics), minimal changes |
| A8 | **Confirmed** — review queue in the fall product's own tab |
| A9 | **Deployment GPU = RTX 5090** (`sm_120`, needs cu128+ torch); dev machine verified = Quadro RTX 3000 6 GB; `NEXT_MODULES.md`'s 5050 claim is stale |
| A10 | **Confirmed** — copied conftest path lines are wiring; gated vendor tests skip when weights/corpus absent |
