# VLM Vehicle Association — Record/Replay Offline Evaluation Plan

> **Date**: 2026-09-29
> **Goal**: Answer, with numbers instead of live-run vibes, whether a VLM (the current free
> `qwen3-vl:4b` on a contended remote GPU, or a future paid API) can validate cross-camera
> vehicle association well enough to *not* need topology as the sole source of truth — before
> we ask for budget for a paid model API. Live pipeline runs cannot answer this today because
> remote GPU contention (see `BOTTLENECKS` investigation, 2026-09-29 session) makes VLM
> latency swing from ~2s to 800s+, so live-run failures conflate "the model isn't good enough"
> with "the model didn't get its turn on a shared GPU in time." This plan decouples the two.

---

## Core idea

Split one live-latency-bound problem into two decoupled, independently-repeatable phases:

1. **Record** (real-time, full pipeline, full video duration): run the actual DeepStream +
   vehicle-detector sidecar over the complete recorded footage, with the VLM **disabled**, and
   log *every* cross-camera candidate comparison the associator's decision point would ever
   face — including ones today's topology gate would silently drop — together with the crops
   involved. This runs at full pipeline speed because nothing is waiting on a remote model.
2. **Replay** (offline, no real-time constraint): feed the recorded candidate pairs (crops +
   metadata) to one or more "verdict providers" (current local VLM, and later a paid API
   adapter) with no deadline pressure, then re-run the *clustering* decision (which candidate
   pairs get merged into one global identity) entirely offline, and score the result against a
   ground-truth mapping we already trust.

Because the record phase captures the full candidate universe once, we can re-run replay+scoring
as many times as we want — against different models, different thresholds, different policies
(topology+VLM vs. VLM-alone vs. re-id+VLM) — without ever touching a live camera feed again.

```
┌─────────────┐   full video    ┌──────────────────┐   candidate    ┌─────────────────────┐
│ DeepStream + │ ─── (once) ──▶ │ Phase 1: RECORD   │ ── trace ────▶ │ Phase 2: REPLAY      │
│ vehicle pgie │                │ (VLM disabled,    │  JSONL +       │ (any verdict         │
│ + re-id      │                │  topology logged   │  crops on      │  provider, offline,  │
│ (this repo)  │                │  but NOT enforced) │  disk          │  no deadline)        │
└─────────────┘                └──────────────────┘                └──────────┬───────────┘
                                                                                │ verdicts
       ┌─────────────┐   deterministic  ┌──────────────────┐                  ▼
       │ DeepStream + │ ─── (once) ───▶ │ Phase 0: GOLD     │        ┌─────────────────────┐
       │ plate pgie   │                │ LABELS (topology+  │        │ Phase 3: CLUSTER +   │
       │ (known-good) │                │ plate = trusted)   │───────▶│ SCORE against gold   │
       └─────────────┘                └──────────────────┘  gold    └──────────┬───────────┘
                                                                                │
                                                                                ▼
                                                                    ┌─────────────────────┐
                                                                    │ Phase 4: DECISION     │
                                                                    │ report (accuracy,     │
                                                                    │ latency, $ estimate)  │
                                                                    └─────────────────────┘
```

---

## Ground rules ("full mimic, nothing silently dropped")

- Phase 1 must **not** apply the topology adjacency filter (`is_valid_transition`) as a
  candidate-generation gate. Every cross-camera pair that clears the *physical* timing gate
  (`min_disappearance_s` / `camera_gaps` / `max_inactive_ttl_s`) gets logged, whether or not
  today's topology graph would allow it. Topology becomes one **logged feature** on the trace
  row (`topology_would_allow: bool`), not a filter — that is exactly the "keep it, but not the
  only source of truth" requirement.
- The timing/disappearance gate stays as a candidate bound in Phase 1 (it is physically
  motivated — a vehicle cannot be the same object 140s apart on the same short loop — not an
  assumption about which camera graph edges exist). `max_inactive_ttl_s: 600.0` already exceeds
  every clip's ~143s duration, so in practice this gate does not hide anything within one loop.
- Every recorded candidate keeps a representative crop (see `_representative_crop` in
  `sidecar/mtmc/_associate.py:553`) **and** the full crop set already buffered on the tracklet
  (`tr.get_crops()`), so Phase 2 can retry with a different crop-selection strategy without a
  second record pass.
- One record pass covers the **entire** video duration for the **5 configured chain sources**
  (cam-01…cam-05, D41→D16→D52→CH05→CH23). Cameras 06–13 are not wired up in the current
  `sidecar.yaml` for this exercise — there is no negative-control data from them, so Phase 4's
  false-merge check is scoped to "did the policy ever merge two gold-distinct vehicles within
  these 5 cameras," not a broader cross-camera universe.
- Nothing here is "mocked" data — every crop and embedding comes from the real detector and
  real video frames. "Mock" only means the *adjudication timing* is decoupled from the *live
  clock*, not that the inputs are synthetic.

---

## Phase 0 — Regenerate the trusted gold-standard mapping (human-judged, not auto-trusted)

The pasted association table (ID 3/10 across 5 cams, ID 9 across 2 cams, ID 25/30 single-cam)
was a momentary UI snapshot, not a durable artifact, and only covers the 5 configured chain
cameras (cam-01…cam-05 — no cam-06…13 in this exercise). The gold mapping is not something an
agent gets to auto-certify: the deliverable is a **human-reviewable evidence package**, and the
user is the one who confirms it.

**Tasks**
1. Temporarily point `pgie_config` back to the plate detector (`config_infer_primary_plate.txt`,
   already modified on this branch per `git status`) and revert `mtmc.tracked_classes` /
   `stabilize_labels` accordingly — i.e. the topology+plate configuration the user confirmed
   was "reasoning very good and associating right."
2. Run the full pipeline once, end-to-end, over one full loop of all 5 chain videos
   (`products/el-mohandes/videos/1-D41-enter.mp4` … `5-CH23-enter.mp4`, ~143s each).
3. Instead of reading the transient `/api/tracking/tracks` UI list, capture the sidecar's own
   `global_tracks` state at the end of the run (add a one-off debug dump, or tap the existing
   `[mtmc] MATCH SUCCESS` / `[mtmc] NEW GLOBAL TRACK` log lines with `grep`) into a proposed
   `data/gold_labels/proposed_mapping.json`:
   ```json
   {
     "global_id": 3,
     "members": [{"cam_id": "cam-01", "local_id": 12, "first_pts": 4.1, "last_pts": 9.8},
                 {"cam_id": "cam-02", "local_id": 7,  "first_pts": 14.2, "last_pts": 20.0},
                 ...]
   }
   ```
4. Build a small **review artifact** for the user from this proposal — one section per
   `global_id`, showing the representative crop image for each member (`cam_id`/`local_id`)
   side by side with its timestamp, plus a plain list of which cam/local_id pairs the pipeline
   claims are the same vehicle vs. singletons. A static HTML page (thumbnail grid, grouped by
   proposed global_id) is the easiest format to scan quickly; a markdown table with embedded
   crop paths also works if the user prefers reading it in-terminal/editor.
5. **The user reviews this artifact directly and tells the agent which groupings are correct,
   which are wrong, and how to fix any wrong ones** (split a group, move a member, merge two
   groups) — this judgment call is not automatable and is not this plan's job to guess at.
   Only after that confirmation does the agent write the corrected result to
   `data/gold_labels/gold_mapping.json`.
6. Revert `pgie_config` / `tracked_classes` back to the vehicle-detector configuration
   afterward (Phase 1 needs the vehicle model, not plates).

**Output**: `data/gold_labels/gold_mapping.json` (trusted only once the user has signed off on
the review artifact from step 4 — treat any later change to this file as a reviewable event
that goes through the same human check, not something an agent regenerates and swaps in
silently).

---

## Phase 1 — Candidate recorder (code)

**New config flag** (`sidecar/sidecar.yaml`, `configs/presets/multicam/sidecar.yaml`):
```yaml
mtmc:
  trace_mode: true                       # candidate-recording mode (Phase 1 only)
  trace_output_dir: "data/mtmc_trace"    # JSONL + crop archive root
```

**Code changes** (`sidecar/mtmc/_associate.py`, `GlobalVehicleAssociation.get_or_create_global_id`):
- When `trace_mode` is on:
  - Do not early-`continue` on `not valid_topo` — still compute `valid_topo` and carry it as a
    field, but keep evaluating the candidate for timing/re-id.
  - For every candidate that clears the timing gate (topology-blind), emit one trace row
    (schema below) regardless of which re-id bucket it falls into (gate-reject / ambiguous /
    match) — do not stop at the first `best_candidate`; log **all** eligible candidates for
    that tracklet, not just the best one (the "full mimic" requirement — a policy change later
    might prefer a different candidate than today's greedy best-sim pick).
  - Skip the VLM call entirely (`self._vlm` stays `None`/disabled via config) and skip the
    actual merge/`add_camera_observation` — Phase 1 only ever creates new tracks, one per
    real per-camera tracklet, so the recorded trace reflects the un-merged ground truth of
    which crops came from which physical local track. (This also means Phase 1 numbering is
    just "cam_id + local_id," no cross-camera identity claim at all — identity only gets
    decided in Phase 3.)
  - Persist the representative crop (`_representative_crop`) plus up to `reid_max_crops` crops
    per **tracklet** (not per pair — a tracklet's crops are reused across every candidate row
    that references it) once, keyed by `f"{cam_id}_{local_id}"`, under
    `trace_output_dir/crops/`. Reuse `sidecar/vlm_arbiter.py`'s `_to_b64`/`_shrink` conventions
    for consistency with Phase 2, but store as `.jpg` files on disk (not embedded strings) —
    Phase 2 will read them directly.

**Trace row schema** (`trace_output_dir/candidates.jsonl`, one line per candidate pair per
frame it was evaluated — dedupe in Phase 2/3 by `(cam_id, local_id, other_gid_slot)` keeping the
last/best row):
```json
{
  "pts": 87.32,
  "cam_id": "cam-03", "local_id": 41,
  "cand_cam_id": "cam-02", "cand_local_id": 17, "cand_track_key": "cam-02_17",
  "elapsed_s": 6.1, "min_disappearance_s": 5.0,
  "reid_sim": 0.21, "reid_gate_threshold": 0.15, "reid_sim_threshold": 0.35,
  "bucket": "ambiguous",
  "topology_would_allow": true,
  "crop_a": "crops/cam-03_41.jpg", "crop_b": "crops/cam-02_17.jpg"
}
```

**Tests**: extend `tests/test_sidecar_mtmc.py` with a `trace_mode` test asserting (a) a
topology-invalid pair still gets a trace row, (b) no merge/VLM call happens in this mode, (c)
crop files are written once per track key not once per row.

---

## Phase 2 — Full-duration recording run

1. Bring up the vehicle-detector config with `trace_mode: true`, VLM arbiter left in config but
   the arbiter object itself not constructed (or constructed with `enabled: false` — cheaper:
   just skip `VLMArbiter(...)` construction in `sidecar.py` when `trace_mode` is set, so there
   is zero remote dependency for this run).
2. Run the full pipeline for **at least one full loop of every source's video** — since all
   clips are ~143s and file-loop wraps with a pts discontinuity (`sidecar.py:2120` handles
   this), run for a fixed wall-clock window comfortably over one loop (e.g. 170s) per camera,
   or key off pts wraparound to stop cleanly exactly at one loop per camera.
3. Confirm output volume: expect `data/mtmc_trace/candidates.jsonl` non-empty, a `crops/`
   directory with one pair of files per distinct tracklet across all 13 cameras, and log a
   summary count of candidates per bucket (gate-reject / ambiguous / match) and per
   topology-allowed vs. not, so an obviously-broken run (e.g. 0 candidates) is caught before
   moving to Phase 3.
4. Archive this trace (it is the expensive, non-reproducible-without-a-live-run artifact) —
   e.g. `data/mtmc_trace/run_2026-09-29/` — before iterating on Phase 2/3 so re-scoring never
   requires re-running the pipeline.

---

## Phase 3 — Offline verdict replay (pluggable provider)

**New script**: `scripts/offline_vlm_replay.py` (or `mtmc/offline_vlm_replay.py` next to the
existing remote `mtmc/compare_vehicles.py`/`benchmark_encoder_vs_vlm.py` scaffolding — reuse
their prompt/schema conventions where sensible, they already encode the comparison prompt this
repo's `vlm_arbiter.py` was based on).

- Define a small `VerdictProvider` interface: `compare(crop_a_path, crop_b_path, context) ->
  {"decision": "same"|"different"|"uncertain", "confidence": float|None, "latency_s": float}`.
- **Provider A (baseline, free, current)**: wraps `sidecar/vlm_arbiter.py`'s `_compare`/
  `_compare_stacked` logic directly (import and reuse, don't reimplement), but run with **no
  `request_timeout_s` deadline pressure** (or a much larger one) and **no orphaning risk**,
  since nothing is waiting on this in real time. This isolates "is the model's reasoning good
  enough" from "did it answer before the tracklet aged out," which was exactly the confound the
  live runs couldn't separate.
  - **Host choice — checked 2026-09-29**: this local machine already has Ollama running
    (`systemctl` active) with the *exact same* `qwen3-vl:4b` model pulled locally (plus several
    other size/context variants: `qwen3-vl:8b`, `qwen3-vl:4b-ctx16k/32k`, `qwen2.5vl:3b`
    variants, `moondream`), and its GPU (RTX 5050 Laptop, 8GB VRAM) was idle at check time (80
    MiB used, 38% util) — i.e. currently **far less contended** than the shared remote
    workstation (100.102.162.114), which had 13 other users and 89-100% GPU util at last check.
    Default Phase 3 to the **local** host (`http://127.0.0.1:11434`) for that reason — no
    network hop, no other tenants, same model weights so no accuracy difference expected, only
    a speed difference.
  - Caveat: this local GPU is the same one DeepStream + the re-id model use during Phase 1/2
    recording. Don't run Phase 3 concurrently with a live recording pass on this machine —
    sequence it strictly after Phase 2's containers are stopped, so Ollama gets the full 8GB.
  - Keep the remote host wired in as `Provider A-remote` (same interface, `host` swapped) for
    two things worth having even though it's not the default: (a) a sanity cross-check that
    local vs. remote give identical verdicts on the same crop pairs (same weights should agree
    at `temperature=0.0`; a disagreement would be worth investigating), and (b) as a fallback
    if the local GPU is needed for something else. Whichever host is actually faster/more
    available on the day of the Phase 3 run is fine to use as the reported default — this is a
    replay tool, not a fixed pipeline, so picking whichever is free that day is a reasonable
    call for whoever runs it.
- **Provider B (re-id-only baseline, for comparison)**: just threshold `reid_sim` from the trace
  row — no model call. This quantifies how much (if anything) the VLM adds over pure re-id.
- **Provider C (future paid API, stubbed)**: same interface, adapter left unimplemented behind
  a `NotImplementedError` with a docstring noting the pending decision (API keys/budget) —
  filled in only once the team lead approves spending, using Provider A's results as the
  baseline it needs to beat.
- Run every distinct `(track_key_a, track_key_b)` pair from `candidates.jsonl` through the
  chosen provider(s) exactly once (dedupe first — a pair may appear on many frames as the
  tracklet accumulates observations; only the final/representative crop matters), logging
  verdict + latency (+ token/cost fields for Provider C) to
  `data/mtmc_trace/run_.../verdicts_<provider>.jsonl`.
- Because this has no deadline, it is fine (and useful) to run Provider A **multiple times**
  over the same pairs to measure verdict stability/consistency at `temperature=0.0` — any
  disagreement across repeats on the *same* image pair is itself a data point about model
  reliability.

---

## Phase 4 — Clustering policies + scoring against gold

Generalize the existing batch clusterer (`sidecar/mtmc/_associate.py:650`,
`associate_cameras` / `_UnionFind` / `_camera_conflict`) into a policy-parameterized version that
accepts an arbitrary "should these two tracklets merge" predicate instead of only a similarity
threshold, so each policy below is one predicate swap, not new clustering code:

| Policy | Merge predicate | Purpose |
|---|---|---|
| `topology_reid` (today's pre-VLM baseline) | `topology_would_allow AND reid_sim >= reid_sim_threshold` | Reproduces current deterministic-ish behavior for comparison |
| `reid_only` | `reid_sim >= reid_sim_threshold` (no topology) | Does re-id alone hold up without the topology crutch? |
| `topology_vlm` | `topology_would_allow AND verdict == same` | Topology gates candidates, VLM is sole merge arbiter |
| `vlm_only` | `verdict == same` (any camera pair, any topology) | Full "VLM as sole source of truth," per the user's stretch goal |
| `gated_vlm` (current shipped design) | `(reid_sim >= gate_threshold) AND verdict == same` | What's actually running today, minus the live-latency confound |

For each policy, run the clusterer over the full candidate set, then score the resulting
clusters against `data/gold_labels/gold_mapping.json` using standard clustering metrics:
- **Pairwise precision/recall/F1** (did the policy correctly reunite every pair of tracklets
  that gold says are the same vehicle, without incorrectly merging tracklets gold says are
  different vehicles — e.g. gold IDs 25/30, single-camera-only vehicles, must never end up
  clustered with 3/10/9 or with each other).
- **Chain-completion rate**: for gold IDs 3/10 (5-camera chains) and 9 (2-camera), did the
  policy's cluster contain *all* the gold members, or only some (partial reunification)?
  Report per-gold-ID, not just aggregated.
- **False-merge count**: any cluster containing tracklets from two different gold IDs (e.g. a
  cluster mixing members of gold ID 3 and gold ID 25) — this is the number that would most
  worry a stakeholder about a wrong paid-API purchase decision.

Output one scoring table per policy into `data/mtmc_trace/run_.../scoring_report.md`.

---

## Phase 5 — Decision report

Roll Phase 3 (latency, consistency, $ estimate once Provider C exists) and Phase 4 (accuracy per
policy) into one short markdown report, `VLM_ASSOCIATION_DECISION_REPORT.md`, answering directly:

1. With GPU contention removed from the equation, does `gated_vlm` (what's actually deployed)
   reach gold-equivalent accuracy? If yes — the fix needed is purely about the live-latency
   problem (async merge event / pre-filter / dedicated GPU), not the model's judgment quality,
   and a paid API would not change the outcome.
2. Does `vlm_only` (no topology, no re-id) hold up on its own? This is the direct answer to "can
   we stop treating topology as the only source of truth."
3. How much does `reid_only` recover by itself (cheap, no model calls) — is there a cheaper
   pre-filter win available regardless of which model backs the VLM step (Option 2 from the
   2026-09-29 investigation)?
4. Only if (1) shows the *local* model's judgment itself is the bottleneck (not the timing) does
   this report recommend pursuing a paid API for Provider C, with the accuracy delta it would
   need to justify the cost, computed directly from this same harness once C is filled in.

---

## New/changed files (for the executing agent(s))

- `sidecar/mtmc/_associate.py` — `trace_mode` branch in `get_or_create_global_id`, all-candidate
  logging instead of best-only, crop persistence helper.
- `sidecar/sidecar.py` — skip constructing `VLMArbiter` when `mtmc.trace_mode` is true.
- `sidecar/sidecar.yaml`, `configs/presets/multicam/sidecar.yaml` — `trace_mode` /
  `trace_output_dir` keys (default `false` / unset so normal runs are unaffected).
- `scripts/offline_vlm_replay.py` — new, Phase 3 driver + `VerdictProvider` interface + Providers
  A/B (C stubbed).
- `sidecar/mtmc/_associate.py` (or a new `sidecar/mtmc/cluster_policies.py`) — generalize
  `associate_cameras` into a predicate-parameterized clusterer, reused by Phase 4.
- `scripts/score_clusters.py` — Phase 4 scoring against `gold_mapping.json`, emits
  `scoring_report.md` per policy.
- `data/gold_labels/gold_mapping.json` — Phase 0 output (trusted, hand-verified).
- `data/mtmc_trace/run_.../` — Phase 1/2 output archive (candidates.jsonl + crops/).
- `tests/test_sidecar_mtmc.py` — `trace_mode` coverage.
- `VLM_ASSOCIATION_DECISION_REPORT.md` — Phase 5 final output (not written yet — this plan's
  deliverable).

## Explicit non-goals for this pass

- Not changing the live/deployed merge behavior at all — `trace_mode` defaults off, nothing
  here touches the currently-running `gated_vlm` design until Phase 5's findings pick a
  direction.
- Not implementing Provider C (paid API) — stub only, filled in after budget approval and only
  if Phase 5 shows it's actually needed.
- Not solving the live-latency/orphaning problem itself (async merge event, pre-filter, GPU
  procurement) — this plan produces the *evidence* for which of those is worth doing, it
  doesn't do them.

## Open dependency

Phase 0 requires briefly reverting to the plate-detector config to regenerate a trusted,
file-backed gold mapping — flag if that is not acceptable (e.g. if the plate config no longer
runs cleanly on this branch) so Phase 0 can fall back to manual crop-based annotation instead.
