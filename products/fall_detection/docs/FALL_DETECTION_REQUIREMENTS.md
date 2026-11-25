# Fall detection on the IRIS cameras — requirements

**Status:** draft for review, 2026-10-27 · **Audience:** supervisor + the implementer
**Design:** [`FALL_DETECTION_DESIGN.md`](FALL_DETECTION_DESIGN.md) · **Plan:** `~/.qoder/plans/polished-glen-newt.md`

---

## 1. How to read this document

Every requirement has an **id**, a **statement**, a **source** (where the requirement comes from —
the plan's locked decision, the repo's own rule, or the vendor's contract) and a **binary acceptance
criterion** in Given/When/Then. "Binary" means it can be answered yes or no by running something; a
requirement that cannot be falsified is not a requirement.

The numbering is stable: `F` functional, `N` non-functional, `X` the next release's overlay.

**The rule that governs all of them:** the feature is **additive**. No change to
`perception/engine/**`, no change to `app/**`, and no change to another product's `sidecar.yaml`.
A requirement that could only be met by changing one of those is a design error, not a
requirement.

---

## 2. Functional requirements

### The pose module

**F1 — The sidecar runs a pose model on every tracked person.**
*Source:* `perception/NEXT_MODULES.md:41-53`; the plan's C1/C2.
The fall product's sidecar overlay lists a `pose` module, and the module adds **one secondary GIE**
that runs on the tracked persons' crops. No new full-frame (primary) inference is added.

> **Given** the fall sidecar running with the `pose` module enabled
> **When** a person is detected and tracked
> **Then** the pose GIE runs on that person's crop, and the engine's log names the element
> (`sgie_pose`) and its `interval`.

**F2 — The pose module publishes the keypoints as the `pose` field on `iris:meta:<cam>`.**
*Source:* the plan's decision 9; `perception/modules/person_boxes/__init__.py:24-27` (the pattern).
The field is written on the frame's own stream, not on a new stream, and it is JSON-encoded by the
publisher (`perception/engine/publisher.py:348-349`).

> **Given** a frame with two tracked people
> **When** the entry is published on `iris:meta:<cam>`
> **Then** the entry carries a `pose` field whose `items` have one entry per person with confident
> keypoints, and **no** other Redis stream is written for the pose.

**F3 — The `pose` field's shape is exactly this.**
*Source:* the design's §3.4; the vendor's `sanitize_keypoints_17` and `update_ref_scale`
(`F:\fall_detection_module\src\fall_monitor.py:120-150`).

```json
"pose": {"shape": [1280, 720],
         "items": [{"id": 42, "box": [x1, y1, x2, y2], "kps": [[x, y, conf] x 17]}]}
```

* `shape` is `[capture_w, capture_h]`;
* `id` is the DeepStream tracker id — the same id the engine prints as `#<oid>` on the box;
* `box` is **xyxy**, in capture pixels;
* `kps` are 17 keypoints in the vendor's COCO order, in capture pixels;
* a person with no confident keypoints is **omitted**, not published with zeros.

> **Given** a golden replay of one frame
> **When** the `pose` field is read
> **Then** `shape` equals the capture size, every item's `id` is a tracker id present in the frame's
> `dets`, every `box` satisfies `x1 < x2` and `y1 < y2` and lies inside the capture rectangle, and
> every `kps` has 17 entries of 3 numbers.

**F4 — The keypoints and the box are in the same space, and that space is capture pixels.**
*Source:* the design's §3.2; the vendor's normalization (`src/keypoint_adapter.py:111-128`).
The network's input coordinates are mapped through the crop's letterbox and then through
mux → capture, using the shipped `landmarks_to_capture` math (`perception/modules/face_detect/__init__.py:60-76`).

> **Given** a person whose network-input keypoints are known
> **When** the module publishes the `pose` field
> **Then** the published keypoints equal the expected capture-pixel positions, within 2 px per
> coordinate, and every keypoint lies inside the published `box` (expanded by 10 %).

**F5 — The module does not depend on the secondary GIE's `parent_id`.**
*Source:* `perception/modules/face_detect/__init__.py:61-65` finds it unset on this pipeline;
`perception/NEXT_MODULES.md:58` says it should be set. The owner is found by containment, reusing
the shipped match (`perception/modules/face_detect/owner.py:117-158`).

> **Given** a frame whose SGIE items carry no `parent_id`
> **When** the module publishes the `pose` field
> **Then** each item's keypoints are the ones of the person whose box contains them, and no item is
> dropped for a missing parent pointer.

**F6 — The module declares its own dependencies and its own settings.**
*Source:* `perception/modules/README.md:74-86` and the `Settings` rule.

> **Given** the module folder
> **When** `tests/test_module_requirements.py` runs
> **Then** it passes, and a yaml key the module does not declare refuses startup.

### The monitor and the seam

**F7 — The vendor's monitor is copied into the product, and the copy is used as it is.**
*Source:* the plan's decision 4.

> **Given** `products/fall_detection/fall_detection/`
> **When** it is compared with `F:\fall_detection_module\src`
> **Then** the trees differ **only** by the recorded seam edits of F8 (and its `PROVENANCE.md`).

**F8 — The seam is additive: it adds the reading and the path scores to the candidate, and changes
no behaviour.** The recorded edits besides it are two `ROOT` path rewrites (the same path:
`parents[1]` *is* `parent.parent`, so the repo's own path rule holds for the copy) and a bounded
state block at the top of the dedup section -- past `events.state_cap` the dedup ledgers halve by
insertion order, no clock arithmetic, so a PTS restart cannot corrupt it (added 2026-10-29).
*Source:* the plan's §The seam; the design's §4.2.

> **Given** a fixture that produces a fall candidate
> **When** it is run through the unmodified vendor code and through the patched copy
> **Then** both produce **the same candidates**, with the same ids, times and types; and the patched
> one additionally carries `reading` (`probs`, `top`) and `path_scores` (`action`, `kinematic`,
> `cold`).

**F9 — The vendor's master config sits beside the copied package, and the product's own config has
its own name.**
*Source:* the vendor's loader (`F:\fall_detection_module\src\fall_monitor.py:58-72`): `CFG = load_cfg()`
runs at import time and resolves `ROOT/config.yaml`, where `ROOT` is the package's parent's parent.

> **Given** the fall product's folder
> **When** the copied package is imported
> **Then** it imports without a `FileNotFoundError`, the vendor's thresholds are read from
> `products/fall_detection/config.yaml`, and the product's own policy is read from
> `products/fall_detection/fall_detection.yaml`.

**F10 — The score's thresholds and spans live with the gates they measure.**
*Source:* the design's §4.2 — one source of truth per quantity.

> **Given** the vendor's config
> **When** the physical score is computed
> **Then** its thresholds are the same numbers the boolean gate uses (`path_b` for the kinematic path,
> `path_c` for the cold path), read from that same config — not from a second copy in the product.

### The scoring policy

**F11 — The score is the confidence in the evaluation, and the evaluation is decided by the detector
first.**
*Source:* the plan's decision 7 and §The evaluation.

> **Given** a candidate of type `immobility` whose reading's top class is `Fall`
> **When** the evaluation is computed
> **Then** it is `falling`, not `immobile`.
>
> **Given** a candidate of type `fall` whose reading's top class is `Lying`
> **When** the evaluation is computed
> **Then** it is `lying`.

**F12 — The reading's score is the decision margin, and a tie scores near zero.**
*Source:* the plan's §The reading's confidence.

```python
support = max(p[c] for c in evaluations[verdict])
oppose  = max(p[c] for c in range(len(p)) if c not in evaluations[verdict])
reading = clip((support - oppose) / margin_span, 0, 1)
```

> **Given** `p[6] = 0.45` against `p[3] = 0.42` (a Fall-vs-Lying tie) and `margin_span = 0.80`
> **When** the reading is computed
> **Then** it is ≤ 0.05.
>
> **Given** `p[6] = 0.95` against `p[3] = 0.02`
> **When** the reading is computed
> **Then** it is ≥ 0.90.

**F13 — The paths combine as `1 - Π(1 - trust·score)`, and a path that did not speak leaves its
trust unclaimed.**
*Source:* the plan's decision 7 and §The combination.

> **Given** a candidate whose only path is the cold path, with a physical score of 1.0
> **When** the combined score is computed with `trust.cold = 0.40`
> **Then** it is 0.40 — below the 0.70 threshold — and the candidate is delegated.
>
> **Given** a candidate with an action score of 0.95 and no other path
> **When** the combined score is computed
> **Then** it is 0.95 — at or above the threshold — and the candidate is auto-confirmed.

**F14 — The trust weights are configuration, not code.**
*Source:* the plan's §The seam (Option A).

> **Given** `trust.action = 0.60` in the product's config
> **When** the service computes the combined score
> **Then** the action path's ceiling is 0.60 — and a candidate with only the action path can no
> longer reach a 0.70 threshold.

**F15 — The class sets are the product's, and they exist in exactly one place.**
*Source:* the design's §6.2 (a refinement of the plan: the service computes the evaluation).

> **Given** the product's config and the vendor's copy
> **When** both are searched for the class groups (`falling: [6]`, `upright: [0,1,2,4,5]`, …)
> **Then** the groups are defined in the product's config only.

### The routes, the alert and the clip

**F16 — The threshold decides whether a human is needed; the severity decides the urgency.**
*Source:* the plan's decision 2 and §The routes.

> **Given** a candidate with a combined score of 0.90 and the evaluation `falling`
> **When** it is routed
> **Then** an alert with `severity=critical` is raised immediately and **no** review row is created.
>
> **Given** a candidate with a combined score of 0.50 and the evaluation `falling`
> **When** it is routed
> **Then** an alert with `severity=warning` is raised immediately **and** a review row is created.
>
> **Given** a candidate with a combined score of 0.90 and the evaluation `upright`
> **When** it is routed
> **Then** **no** alert is raised and **no** review row is created — a confirmed false alarm, closed.
>
> **Given** a candidate with a combined score of 0.90 and the evaluation `lying`
> **When** it is routed
> **Then** an alert with `severity=warning` is raised immediately and **no** review row is created.

**F17 — The alert uses the platform's contract, used as it is.**
*Source:* the plan's decision 10; `perception/engine/modules.py:99-111` → `app/usecases/module_alerts.py:41-54`.

> **Given** a confirmed fall
> **When** the alert is written to `iris:alert:<cam>`
> **Then** the entry carries `v=1`, `kind=fall`, `severity` ∈ {`critical`, `warning`},
> `category=safety`, a `message` naming the evaluation and the confidence, and `pts` — and the app
> turns it into an event with no code of the module's own.

**F18 — The clip is requested on the platform's contract, with the sidecar's frame clock.**
*Source:* `app/usecases/recordings.py:63-72`; `perception/engine/main.py:258-289`.

> **Given** a confirmed fall at the frame whose `pts` is `T`
> **When** the clip is requested
> **Then** `iris:svr_request:<cam>` carries `pts=T`, and the sidecar's clip covers the fall.

**F19 — The monitor's `pts` is the sidecar's frame clock, not the wall clock.**
*Source:* the plan's §The consumer.

> **Given** a frame published with `pts = 1234.5`
> **When** the monitor is updated
> **Then** it is given `t = 1234.5`, and the alert and the clip request carry the same clock.

**F20 — One `FallMonitor` per camera.**
*Source:* `FallMonitor.tracks` is keyed by the track id alone.

> **Given** two cameras reporting the same track id
> **When** both are monitored
> **Then** neither camera's track state is affected by the other.

**F21 — The consumer's queue is bounded and drops the oldest frame when it is full.**
*Source:* the plan's §The monitor runs on its own thread.

> **Given** a camera whose worker is slower than the frame rate
> **When** the queue is full
> **Then** the oldest frame is dropped, the queue never grows without bound, and the monitor keeps
> running.

**F22 — The review queue persists the candidate and the operator's decision.**
*Source:* the design's §5.4; the vendor's `review_event` (`fall_monitor.py:511-522`).

> **Given** a candidate below the threshold
> **When** the service restarts
> **Then** the review row is still there, with the evaluation, the combined score, the per-path
> scores and the reading's top class.
>
> **Given** a review row
> **When** it is marked `acknowledged`, `confirmed` or `dismissed`
> **Then** the status is stored with the operator's note, and no other status is accepted.

**F30 — The vendor's `lying` candidate is an alert of its own.**
*Source:* the user's decision (2026-10-27); the design's §13.3.

> **Given** a `lying` candidate with a combined score at or above `confirm_threshold`
> **When** it is routed
> **Then** an alert with `severity=warning` is raised immediately and **no** review row is created.
>
> **Given** a `lying` candidate with a combined score below `confirm_threshold`
> **When** it is routed
> **Then** an alert with `severity=warning` is raised immediately **and** a review row is created.
>
> **Given** a `lying` candidate
> **When** it is scored
> **Then** its evaluation is `lying` and its physical score is `(s_posture · s_still) ^ 0.5` against
> the vendor's own `lying` thresholds (`angle_in`, `aspect_in`, `max_motion`) — trusted at the
> `kinematic` ceiling with a window and at the `cold` ceiling without one, with no new trust value.

**F31 — A fall's aftermath is not reported twice.**
*Source:* the user's decision (2026-10-27); the design's §13.3.

> **Given** a track that produced an alerted `fall` within `suppress_after_fall_s` (10 s)
> **When** a `lying` candidate for that track is scored
> **Then** it is **not** alerted and **not** queued for review.
>
> **Given** a track whose alerted `fall` is older than `suppress_after_fall_s`
> **When** a `lying` candidate for that track is scored
> **Then** it alerts normally — the suppression is per track and per fall event, not per kind.
>
> **Given** a suppressed `lying` candidate
> **When** the overlay draws the state
> **Then** the box still carries the `lying` condition — the suppression silences the event, not the
> picture.

### The boot contract

**F23 — The classifier is mandatory: the service refuses to start without it.**
*Source:* the plan's §The boot contract; the vendor's silent degradation
(`fall_monitor.py:459-469`).

> **Given** a missing checkpoint
> **When** the service starts
> **Then** it exits non-zero with a message naming the missing file.
>
> **Given** a corrupted checkpoint
> **When** the service starts
> **Then** it exits non-zero with a message naming the checksum mismatch.
>
> **Given** `classifier.device: cuda:0` on a machine with no GPU
> **When** the service starts
> **Then** it exits non-zero with a message naming the device — and it does **not** fall back to the
> CPU.

**F24 — The start-up check runs a real CUDA operation, not `cuda.is_available()`.**
*Source:* the plan's risk 4 (a wheel without this GPU's kernels reports a healthy CUDA).

> **Given** a torch build whose CUDA is present but whose kernels do not match the card
> **When** the service starts
> **Then** the check fails and the service refuses to start.

**F25 — The classifier's state is reported, not assumed.**
*Source:* the plan's §The boot contract.

> **Given** a running service
> **When** `GET /status` is called
> **Then** the response carries `device`, `weights_verified`, `classifier_ready` and which paths are
> live.

### The deployment

**F26 — The fall product owns its sidecar, and no other product's configuration changes.**
*Source:* the plan's decision 6; `perception/ARCHITECTURE.md:422`.

> **Given** the change
> **When** `git diff` is inspected
> **Then** `products/attendance/sidecar.yaml`, `products/barber/sidecar.yaml` and
> `perception/pipeline/sidecar.yaml` are byte-identical to `main` (a guard test asserted this
> until it was removed with the branch's stale-test sweep; the check is manual now).

**F27 — The engine and the app are untouched.**
*Source:* `tests/test_engine_boundary.py`.

> **Given** the change
> **When** `pytest tests/test_engine_boundary.py` runs
> **Then** it passes, and the recorded counts have not gone up.

**F28 — The fall deployment refuses to run beside a live attendance sidecar.**
*Source:* the plan's risk 5; `perception/engine/sources.py:52-74` (a `roles` tag does not exclude a
second process).

> **Given** a live attendance sidecar holding one of the fall cameras
> **When** `just fall-up` runs
> **Then** it refuses to start and names the sidecar it found.

**F29 — The fall product has a golden config.**
*Source:* `perception/modules/README.md:120`; the golden-config pattern
(`tests/configs/golden-barber.yaml`).

> **Given** the fall product
> **When** the golden replay runs
> **Then** `tests/configs/golden-fall_detection.yaml` produces its baseline, and the baseline
> contains the `pose` field with 17 keypoints per person.

**F32 — The pilot runs on `cam-1` alone.**
*Source:* the user's decision (2026-10-27); the design's §13.1.

> **Given** the pilot deployment
> **When** its camera set is read
> **Then** it is exactly `cam-1` — one camera, so a regression is attributable to the pose GIE.

### The fall product's tab

**F33 — The fall product has its own tab, and it draws the engine-baked JPEG.**
*Source:* the user's decision (2026-10-27); the design's §13.4; Option A of the design's §10.

> **Given** the fall product's tab
> **When** it is opened for a camera
> **Then** it shows `GET /stream/<cam>/frame` — the published JPEG, with the condition already drawn
> on the person's box — and the tab adds no Redis contract, no WebSocket and no box math.
>
> **Given** the tab
> **When** a condition is on a box
> **Then** it is visible in the tab exactly as it is on the dashboard wall, because it is the same
> frame.

**F34 — The WebSocket scene is a later slice, and it lands on the same state.**
*Source:* the user's decision (2026-10-27); the design's §13.5; Option B of the design's §10.

> **Given** the first release
> **When** it ships
> **Then** no WebSocket exists, and the tab's JPEG layer and its chip layer are separate — so the
> scene can be added later without changing the tab's page.
>
> **Given** the scene slice, once it has landed
> **When** it publishes the scene
> **Then** it publishes the same per-track map the engine-baked label reads — the id, the box in
> capture pixels, the evaluation, the confidence, the ts — over the fall product's own
> `/fall_detection/ws`, and `perception/engine/**`, `app/**`, the pose module, the monitor and the
> scoring policy are all unchanged.

---

## 3. Non-functional requirements

**N1 — The sidecar's GPU cost of the feature is measured, and it is bounded by the camera count.**
*Source:* the plan's §Focus areas; `perception/NEXT_MODULES.md:89-91` (the GPU is the top risk).

> **Given** the golden replay, before and after the module is enabled
> **When** `tools/engine_gpu_budget.sh` runs
> **Then** each GPU stage's busy% is printed, and the pose stage's delta is recorded as the cost of
> the feature.

**N2 — The pilot runs at 12.5 Hz, and 25 Hz is an experiment before shipping.**
*Source:* the plan's decision 5.

> **Given** the fall corpus
> **When** it is replayed at `interval: 2` and at `interval: 1`
> **Then** both runs report the action path's contribution and the false alarms per hour, and the
> comparison is written down before the product ships.

**N3 — The decision-to-alert latency is under ~5 s.**
*Source:* the plan's §Focus areas; the clip window (`pad_seconds: 5.0`).

> **Given** a fall in the replayed footage
> **When** the alert is measured end to end
> **Then** the time from the fall's onset to the alert is under 5 s.

**N4 — The action path's accuracy is measured on our own stream, not the vendor's numbers.**
*Source:* the plan's §Focus areas.

> **Given** the labelled corpus (present at `F:\fall_detection\data`, skipped when absent)
> **When** it is replayed through the sidecar → the `pose` field → the service
> **Then** the precision, the recall and the false alarms per hour are reported, and the candidate
> is on the right camera and the right track.

**N5 — The service's classifier cost is measured on the card it shares with the pose GIE.**
*Source:* the plan's §Focus areas.

> **Given** the service running with N concurrent tracks
> **When** the forwards are timed on `cuda:0`
> **Then** the milliseconds per forward, the number of concurrent tracks and the whole-card view are
> recorded.

**N6 — The product is autonomous: it imports nothing from another product.**
*Source:* the anti-corruption-layer pattern (`products/barber/acl.py:1-16`); the repo's own idiom —
the barber's ACL and storage are themselves forks of the attendance's, with the generic helpers kept
and everything else rewritten (`products/barber/acl.py:1-16`, `products/barber/storage.py:1-6`).

> **Given** the fall product's code
> **When** its imports are listed
> **Then** no `products.<other>` module is among them — the Redis shapes are re-declared, the yaml
> loader is the product's own, and the database connection is the product's own copy of the generic
> helper.
>
> **Given** the fall product's config loader
> **When** it is read
> **Then** it loads `products/fall_detection/fall_detection.yaml` and nothing from another product's
> folder — in particular it does **not** extend the barber's yaml.

**N7 — The policy is testable without a GPU, a camera or Redis.**
*Source:* the plan's §The seam (Option A) — the scoring maths is inside the IRIS repo.

> **Given** the scoring module
> **When** it is unit tested
> **Then** the tests need no Redis, no torch and no GPU, and they cover the margin, each path's
> ceiling, the combination and every route.

**N8 — The vendor's own tests run against the copy.**
*Source:* the plan's §Where the vendor code comes from.

> **Given** the copied package
> **When** `just fall-test` runs
> **Then** the vendor's nine test files pass against the copy, with the seam edit applied.

**N9 — The product's start-up failure is loud and specific.**
*Source:* the plan's §The boot contract.

> **Given** each of the three refused start-ups in F23 and F24
> **When** the service exits
> **Then** the message names the cause, and the exit code is non-zero.

**N10 — Nothing runs in a degraded mode that the operator cannot see.**
*Source:* the plan's risk 3.

> **Given** a running service
> **When** `/status` is read
> **Then** the classifier's state, the device and the live paths are reported, and there is no code
> path in which the service runs with fewer paths than it reports.

---

## 4. The next release — the live annotation overlay

**X1 — The condition appears on the person's own box, in the published video.**
*Source:* the user's request (2026-10-26); the plan's §The next release.

> **Given** a confirmed `falling` candidate on track 12
> **When** the published JPEG of that camera is fetched
> **Then** the box of track 12 carries the label `FALLING 0.93 #12`, and no other box's label
> changes.

**X2 — The default label is unchanged for every other person.**
*Source:* the same.

> **Given** a frame with two people, one of them in a `falling` state
> **When** the published JPEG is fetched
> **Then** the other person's box still carries the default `Person 0.95 #<oid>` label.

**X3 — The condition is drawn in the colour the repo already reserves for a fall.**
*Source:* `perception/NEXT_MODULES.md:104-111` — magenta, priority 12.

> **Given** a person in any of the three states
> **When** the box is drawn
> **Then** its colour is the reserved fall colour and its annotation priority is 12, and no new
> colour is introduced.

**X4 — The condition follows the person.**
*Source:* the plan's §The tracking.

> **Given** a person who moves across the frame while in a state
> **When** consecutive frames are fetched
> **Then** the annotation stays on the same person's box, matched by the tracker id.

**X5 — A condition ages off instead of freezing on the box.**
*Source:* the plan's §The live state.

> **Given** a person in a `falling` state whose track then disappears
> **When** more than `state_hold_s` (10 s) has passed
> **Then** the box is gone and no annotation is left behind.

> **Given** a person in a `falling` state and the fall service stops publishing
> **When** more than `state_hold_s` has passed
> **Then** the annotation is gone — the state is not frozen on the box.

**X6 — A delegated (unconfirmed) state is drawn, and marked as such.**
*Source:* the plan's §The defaults.

> **Given** a candidate below the threshold
> **When** the overlay draws it
> **Then** the annotation is visible and says it is unconfirmed — a delegated state is visibly
> different from a confirmed one.

**X7 — No frontend work is required for the overlay to appear.**
*Source:* the plan's decision 8 — the simple way is first.

> **Given** the overlay module enabled and the service publishing the state
> **When** the platform's existing dashboard wall is opened
> **Then** the condition is visible on the camera's tile, with no change to `app/**` and no change to
> the frontend's shared code.

**X8 — The module never blocks on Redis.**
*Source:* `perception/modules/README.md:45`; the `control()` mechanism
(`perception/engine/modules.py:176-178`, `perception/engine/control.py`).

> **Given** Redis is unreachable
> **When** frames are processed
> **Then** the pipeline keeps running and the module keeps its last known state, and the read happens
> on the control poller's thread, never in a hook.

---

## 5. Out of scope

These are explicitly **not** required, and their absence is not a defect.

* Running two products at once on the same cameras.
* Fine-tuning the ST-GCN, or any change to the published weights.
* An automatic CPU fallback for the classifier.
* `pipeline_preview: false`.
* A live skeleton overlay, or the 17 keypoints, in the UI.
* **A DOM overlay on the platform's dashboard wall** — the condition is baked into the published JPEG
  by the engine, so the wall shows it for free. The fall product's **own** tab may carry a chip
  later (F34); the platform's wall does not.
* A change to `app/**`, to `perception/engine/**`, or to another product's `sidecar.yaml`.
* A live skeleton overlay on the attendance tiles.

---

## 6. The decisions, and the one question still open

Three of the four questions are answered; the answers are in the design's §13, and they are binding:

1. the pilot's camera set — **`cam-1` only** (F32);
2. the confirm threshold's first value — **`0.70`**, configurable and measured before shipping
   (F13, N4);
3. `lying` alerts from day one — with the fall's aftermath suppressed (F30, F31).

The one question still open is the **review queue's operator surface**: a list in the fall product's
own tab (the recommendation), a separate page, or a read-only list in the existing dashboard. It is
a frontend decision and it does not block the service — the queue, the three statuses and the API
are the same either way (F22).

---

## 7. The verification commands

```bash
pytest tests/test_product_declarations.py -v   # the shared file still has modules: {}
pytest tests/test_module_requirements.py -v     # the pose module declares what it imports
pytest tests/test_engine_boundary.py -v         # the engine gained nothing
just fall-test                                  # the service's own suite + the vendor's tests
just fall-up                                    # the deployment: preflight, service, sidecar
```

There is **no weights-fetching recipe**: the vendor's checkpoints have no public download this repo
has verified. The two the config loads are **committed** under `products/fall_detection/weights/`
(a team-lead decision), and the boot gate verifies their SHA-256 at every start.

And the manual ones, in order:

* **Model:** the export's numbers against `model.predict()`.
* **Pipeline:** `sgie_pose enabled, …, interval=2` in the log; the derived engine file name (no
  rebuild on the second boot); `iris:meta:cam-1` carries a `pose` field with 17 keypoints per person.
* **Product isolation:** `git diff` shows no change to the other products' sidecar yamls.
* **Boot:** a corrupted `.pth`, a missing `.pth`, and `classifier.device: cuda:0` with no GPU — all
  three refuse to start, naming the cause.
* **GPU:** `nvidia-smi` shows the service's process holding a CUDA context; a forward is timed.
* **Load:** `tools/engine_gpu_budget.sh products/fall_detection/sidecar.yaml` before and after.
* **Decision:** the corpus replayed through the sidecar → the `pose` field → the service; the
  false-positive rate on non-fall footage is the number that decides shippability.
* **Overlay (next release):** the published JPEG shows the condition on the right person's box.
