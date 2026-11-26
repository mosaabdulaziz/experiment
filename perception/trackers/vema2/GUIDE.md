# VEMA v2 (`libnvds_vema2_tracker.so`) — Architecture & Threshold Tuning Guide

VEMA v2 (**Derived Continuous Polar-EMA + Scalar NSA Kalman + NIS Maneuver Boost**) is IRIS's production CPU DeepStream `NvMOT` tracker replacing NVIDIA `NvDCF` across `attendance`, `zones`, `multicam`, `proximity`, and `barber`.

This guide explains how VEMA v2 works, what every YAML threshold does, how DeepStream detector/pipeline settings interact with tracker thresholds, how to diagnose and tune edge cases, and how to visually approve changes side by side against `NvDCF`.

---

## 1. Why VEMA v2 & How It Works

### Failure Modes VEMA v2 Solves
- **NvDCF CPU/GPU overhead**: On `barber`, `attendance`, and `zones`, VEMA v2 cuts `gpu.tracker` stage latency by **`5.9x–10.3x`** (`0.08–0.13 ms` vs `0.50–1.35 ms`) with zero GPU memory footprint.
- **Box scale blowups (OC-SORT / linear Kalman `ds/dt`)**: When a caped barber customer or partially occluded passer-by changes box height suddenly, linear Kalman scale velocity extrapolates huge phantom boxes across the frame. VEMA v2 holds scale velocity at **`dw/dt = dh/dt = 0`** and updates size gently with a trust-weighted EMA.
- **Crossing / leaning ID swaps (pure-IoU ByteTrack)**: When a barber leans over a seated customer or two people cross, IoU alone becomes ambiguous. VEMA v2 scores candidates by combining spatial overlap with a directional/speed kinematic cost weighted by directional coherence $R$.
- **Border exit lingering**: Departing subjects coast at screen boundaries under unconstrained shadow trackers. VEMA v2 terminates border exits quickly (`border_exit_lost`) and suppresses border shadow emissions (`k_lost >= 1`).

### Core Mathematical Design (Single Physical Parameter: `tau_mem`)
Instead of 10+ disconnected decay magic numbers, every kinematic gain derives analytically from **`tau_mem`** (`0.15 s` default) and the frame interval `dt`:

1. **Continuous Decay Rate**: $\alpha_{dt} = 1 - \exp(-dt / \tau_{\text{mem}})$.
2. **Continuous Health Score $H_t \in [0, 1]$**: Maps detector confidence cleanly between `tau_low` and `tau_high` so low-confidence occluded detections down-weight kinematic updates instead of corrupting velocity.
3. **Overlap / Occlusion & Size-Stability Weights ($w_{\text{ema}} = w_{\text{conf}} \cdot w_{\text{occ}} \cdot w_{\text{size}}$)**: When another confirmed track overlaps the detection (`max_other_iou`) or box area jumps abruptly (cape drape, leaning barber), kinematic absorption slows down automatically so clean pre-crossing velocity is preserved.
4. **Decoupled Polar Velocity + Scalar NSA Kalman Position**:
   - Scalar speed `s_ema` (px/s) and unit-disk mean resultant heading `m_vec` (coherence $R = \|\mathbf{m}\| \in [0, 1]$) decay along exact time gaps using the trust-weighted anchor `(p_anchor, t_anchor)`.
   - Position variance `p_var` runs a scalar NSA Kalman filter modulated by observation trust, with a temporary **NIS maneuver boost** ($\eta \ge 1$) when an unoccluded stationary object legitimately accelerates (e.g., parked car pulls away, seated customer stands up).
5. **4-Stage ByteTrack-Parity Association**:
   - **Stage 1**: Confirmed + lost tracks $\leftrightarrow$ High-confidence detections (`conf >= tau_high`, gate `iou >= iou_thresh_high`) using additive row-scale-invariant cost $(1 - \text{IoU}) + \text{tau\_iou\_high} \cdot \rho \cdot c_{\text{kin}}$.
   - **Stage 2**: Remaining active (`k_lost == 0`) confirmed tracks $\leftrightarrow$ Low-confidence detections (`tau_low <= conf < tau_high`, gate `iou >= iou_thresh_low`). **Never spawns new tracks.**
   - **Stage 3**: Remaining lost (`k_lost > 0`) confirmed tracks $\leftrightarrow$ Unmatched high-conf detections using anisotropic uncertainty ellipses along/perpendicular to heading $\hat{u}$.
   - **Stage 4**: Tentative tracks $\leftrightarrow$ Remaining high-conf detections (pure IoU).

---

## 2. Parameter Reference & Tuning Principles

Config files live in `perception/pipeline/config_tracker_vema2*.yml` and are parsed by [`vema2_tracker.cpp`](../../../perception/trackers/vema2/src/vema2_tracker.cpp#L377).

### 2.1 Detector & Association Contract

| Parameter | Default | Typical Range | What It Controls & How to Tune It |
| :--- | :---: | :---: | :--- |
| `tau_high` | `0.60` | `0.25 – 0.65` | **Stage 1 / 3 / 4 association split**. Detections $\ge \tau_{\text{high}}$ run in Stage 1 (active), Stage 3 (lost re-association), and Stage 4 (tentative), while $[\tau_{\text{low}}, \tau_{\text{high}})$ only rescue active (`k_lost == 0`) tracks in Stage 2.<br>**Rule**: Match your detector's `pre-cluster-threshold` (`0.25` for `YOLO26l-seg`) when lost occluded people need to re-attach at low confidence, and raise `new_track_thresh` to control spawns. |
| `new_track_thresh` | `0.0` (=`tau_high`) | `0.40 – 0.65` | **Track-creation confidence floor**. An unmatched detection in Step 7 can only spawn a new track if $\text{conf} \ge \max(\tau_{\text{high}}, \text{new\_track\_thresh})$. Decoupling this from `tau_high` (e.g., `tau_high: 0.25, new_track_thresh: 0.43` in `barber`) lets **confirmed** seated/caped customers re-attach at `0.25–0.42` after barber crossings while continuing to block `< 0.43` mirror reflections and background clutter from spawning new IDs. |
| `tau_low` | `0.15` | `0.10 – 0.25` | **Low-confidence rescue floor**. Keeps an already-active (`k_lost == 0`) track alive through brief partial occlusions or confidence dips without ever spawning a new ID. Raise only if heavy background clutter (`0.15–0.25`) steals active tracks from nearby people. |
| `iou_thresh_high` | `0.25` | `0.15 – 0.35` | Minimum IoU for Stage 1 & Stage 4, and weight factor on Stage 1 kinematic cost `(1 - IoU) + iou_thresh_high * rho * c_kin`. Lower (`0.20`) when cameras run low effective FPS (`process_fps` skip) so fast objects bridge wider inter-frame jumps. |
| `iou_thresh_low` | `0.15` | `0.10 – 0.25` | Minimum IoU for Stage 2 low-confidence rescue. Slightly looser than `iou_thresh_high` because occluded boxes shrink or shift slightly. |
| `check_class_match` | `true` | `true \| false` | Require identical detector `classId` between track and detection. Set **`false`** on multi-class vehicle products (`proximity`, where forklifts flicker between `car` and `truck`); keep **`true`** everywhere else. |

### 2.2 Motion Memory & Timing

| Parameter | Default | Typical Range | What It Controls & How to Tune It |
| :--- | :---: | :---: | :--- |
| `tau_mem` | `0.15` | `0.10 – 0.25` | **Single kinematic memory time constant (seconds)**. Controls EMA velocity/heading reactivity and noise floor scaling. Rarely needs changing: `0.15 s` handles both `30 fps` MOT benchmarks and `15–25 fps` CCTV. Raise toward `0.22–0.25` only on ultra-noisy detections; lower toward `0.10` for hyper-agile fast turns at high FPS. |
| `dt_aware` | `true` | `true` | Uses real GStreamer frame PTS deltas (`ts_ns - last_ts_ns`) normalized against `ref_fps`. **Always keep `true`** in DeepStream so stream hiccups or `15/20/25/30 fps` cameras integrate physical speed (px/s) cleanly. |
| `ref_fps` | `30.0` | `30.0` | Reference frame rate scaling `k_lost += dt * ref_fps`. Keep at `30.0` so frame counters (`max_lost_frames: 600` = `20 s`, `border_exit_lost: 4` = `0.13 s`) have uniform wall-clock meaning across 15/20/25/30 FPS cameras! |

### 2.3 Track Birth, Duplicate Suppression & Re-ID Memory

| Parameter | Default | Typical Range | What It Controls & How to Tune It |
| :--- | :---: | :---: | :--- |
| `probation_frames` | `2` | `1 – 4` | Consecutive hits (`hits >= probation_frames`) required before a tentative track becomes `Confirmed` and emits downstream. **Frame 1 bootstraps immediately** (`frame_idx == 1`).<br>**DeepStream parity note**: NvDCF's `probationAge: 2` means `age > 2` (**3 hits**), which is why `barber` uses `probation_frames: 3`. Keep `1–2` when `pgie` skips frames (`interval >= 2`, `multicam`, `proximity`). |
| `early_term_lost` | `2` | `2 – 30` | Maximum `k_lost` ticks a *Tentative* track survives before pruning. **CRITICAL PITFALL** with `process_fps` (`pgie interval > 0`): skipped batches keep inference boxes unchanged OR drop unconfirmed detections depending on pipeline cadence! See Section 3 below. |
| `min_iou_diff_new` | `0.70` | `0.35 – 0.70` | **Spawn suppression gate**. An unmatched high-conf detection is suppressed (won't spawn a duplicate ID) if `IoU(det, active_confirmed_track) > min_iou_diff_new`.<br>**Rule**: `0.37` matches IRIS NvDCF's `minIouDiff4NewTarget: 0.3686`, cutting double-box flicker on `YOLO26l-seg` (`barber`, `proximity`). Keep `0.70` in dense `YOLO11x` / MOT crowd scenes where side-by-side people legitimately overlap by `0.40–0.65`. |
| `max_lost_frames` | `600` | `60 – 600` | Internal re-ID memory window in `30 fps`-equivalent frames (`600` = `20 s`). Lets stationary seated people (backs turned in `attendance` or occluded behind a barber) reclaim their original track ID without emitting phantom boxes the entire time. |
| `max_targets_per_stream` | `120` | `120 – 150` | Advertised to DeepStream `NvMOTQuery`. Active tracks are always emitted *before* shadow tracks so capacity caps never drop real detections. |

### 2.4 Shadow Coasting & Border Hygiene

| Parameter | Default | Typical Range | What It Controls & How to Tune It |
| :--- | :---: | :---: | :--- |
| `emit_shadow_tracks` | `true` | `true \| false` | Whether lost confirmed tracks (`k_lost > 0`) are emitted with negative detector confidence (`conf < 0`, `associatedObjectIn = nullptr`). Needed in IRIS for smooth previews and `proximity` `SteadyBoxes`. |
| `max_shadow_emit_lost` | `1` | `1 – 20` | Maximum `k_lost` (in `30 fps` units, so `3` = `0.10 s`, `16` = `0.53 s`) during which a lost track emits a shadow box. Keeps `max_lost_frames: 600` alive internally for re-ID while capping visible ghost boxes!<br>**Warning**: On `attendance`/`zones`, setting this $> 2$ can fail `border_shadow_linger_s <= 0.25 s` if a track stops slightly inside `border_exit_margin_px`. |
| `border_exit_lost` | `4` | `4 – 8` | Prunes lost confirmed tracks touching an image edge within `border_exit_margin_px` after `border_exit_lost` lost units (~0.13–0.27 s). Plus, **any border-touching lost track (`k_lost >= 1`) never emits a shadow box**. |
| `border_exit_margin_px` | `24.0` | `16.0 – 32.0` | Pixel inset from frame edges (`0..img_width`, `0..img_height`) defining the border exit zone. |
| `min_visible_frac` | `0.25` | `0.25` | Immediately prunes lost tracks whose predicted box area inside frame bounds drops below `25%` for `k_lost > 1`. |
| `img_width`, `img_height` | `1080x720` | Match mux | Fallback dimensions if `NvMOTFrame` buffer dims aren't populated (at runtime [`nvmot_context.cpp`](../../../perception/trackers/vema2/src/nvmot_context.cpp#L220) reads actual mux buffer width/height). |

---

## 3. Critical Pipeline Interactions (Read Before Changing Thresholds)

### Pitfall A: `pgie` Skip Interval (`process_fps` in `proximity` & `multicam`)
When a product sets `process_fps` lower than `stream_fps` (e.g., `proximity`: `5/2.5` -> `interval = 2`, analyzing every `3rd` batch; `multicam`: `5/1.67` -> `interval = 3`, analyzing every `4th` batch), wall time between fresh detector frames can be **3–4x** longer than `1 / stream_fps`!
- Because `k_lost` increments in `ref_fps = 30.0` units (`dt * 30`), a `0.60 s` gap adds **`k_lost += 18`** in a single skipped cycle.
- Therefore:
  - `early_term_lost` MUST be `>= 20–30` on `interval >= 2` pipelines (otherwise a Tentative track dies during the very first detector skip before receiving its 2nd hit).
  - `probation_frames` should stay `1–2` (counting confirmed detections, not raw frames).
  - `max_shadow_emit_lost` (`10–16`) bridges short detector skips cleanly without holding ghosts for seconds.

### Pitfall B: Detector Floor vs `tau_high` vs `new_track_thresh`
Never set `tau_high` blindly high or blindly low without checking both the PGIE config and the product module:
- `YOLO11x` (`attendance`, `zones`): Person detections sit cleanly at `>= 0.60` (`tau_high: 0.60`).
- `YOLO26l-seg` (`barber`, `proximity`): PGIE `pre-cluster-threshold = 0.25`.
  - On **`barber`**, heavily caped or barber-occluded seated customers frequently re-emerge at `0.25–0.42`, while mirror reflections also fire around `0.25–0.32`. Setting **`tau_high: 0.25`** with **`new_track_thresh: 0.43`** allows **confirmed** customers to re-attach cleanly across `[0.25, 1.0]` while keeping new track creation strict (`>= 0.43`) so mirror ghosts do not spawn.
  - On **`proximity`**, `modules.proximity.min_confidence = 0.40` drops sub-`0.40` evidence downstream anyway, so **`tau_high: 0.42`** aligns directly with the safety module.

### Pitfall C: NvDCF `probationAge` Off-by-One Semantics
DeepStream's `TargetManagement.probationAge: 2` confirms only when `age > 2` (i.e. **3 detections**). VEMA v2 confirms at `hits >= probation_frames`. To match NvDCF's flicker filter (`barber`), set `probation_frames: 3`, not `2`.

---

## 4. Shipped Per-Product Configuration Matrix

| Knob | [Base / Attendance / Zones](../../pipeline/config_tracker_vema2.yml) | [Barber](../../pipeline/config_tracker_vema2_barber.yml) | [Proximity](../../pipeline/config_tracker_vema2_proximity.yml) | [Vehicle / Multicam](../../pipeline/config_tracker_vema2_vehicle.yml) |
| :--- | :---: | :---: | :---: | :---: |
| **PGIE Model** | `YOLO11x` (1080x720) | `YOLO26l-seg` (r384) | `YOLO26l-seg` (1280) | `vehicle_1cls` (1280x720) |
| **`tau_high` / `tau_low`** | `0.60 / 0.15` | `0.25 / 0.15` | `0.42 / 0.15` | `0.65 / 0.15` |
| **`new_track_thresh`** | `0.0` (=`0.60`) | `0.43` | `0.0` (=`0.42`) | `0.0` (=`0.65`) |
| **`iou_thresh_high` / `low`** | `0.25 / 0.15` | `0.25 / 0.15` | `0.25 / 0.15` | `0.20 / 0.10` |
| **`probation_frames`** | `2` | `3` | `2` | `1` |
| **`early_term_lost`** | `2` | `2` | `20` | `30` |
| **`min_iou_diff_new`** | `0.70` | `0.37` | `0.37` | `0.70` |
| **`max_lost_frames`** | `600` (20 s) | `600` (20 s) | `600` (20 s) | `500` (16.7 s) |
| **`check_class_match`** | `true` | `true` | `false` | `true` |
| **`max_shadow_emit_lost`** | `1` | `3` | `16` | `10` |
| **`border_exit_lost` / `margin`** | `4 / 24 px` | `4 / 24 px` | `8 / 24 px` | `6 / 28 px` |

---

## 5. Symptom -> Knob Troubleshooting Playbook

| Observed Problem | Root Cause | Fix (in Priority Order) |
| :--- | :--- | :--- |
| **Low total detections (`total_detections` < 95% of NvDCF)** | `tau_high` too strict for re-acquiring lost low-conf targets, or `probation_frames` too high | Lower `tau_high` (`0.43 -> 0.25`) while keeping `new_track_thresh: 0.43` to guard spawns; verify `probation_frames` vs `pgie interval` |
| **High `flicker_rate_pct` / short `< 1 s` IDs** | Spurious detections / mirror reflections / duplicate boxes spawning | Raise `new_track_thresh` (e.g. `0.43 -> 0.50`), raise `probation_frames` (`2 -> 3`), or lower `min_iou_diff_new` (`0.70 -> 0.37`) |
| **ID switches when barber leans over / people cross** | Low-IoU threshold too loose or insufficient lost memory | Raise `iou_thresh_high` (`0.25 -> 0.30`) slightly or verify `dt_aware: true` |
| **Standing customer / occluded track fails to re-ID after 5–15 s** | `max_lost_frames` too short or `tau_high` drops re-appearing box | Raise `max_lost_frames` (`30 -> 600` at `ref_fps: 30.0`) and keep `tau_high` at detector floor |
| **Ghost boxes linger at exits/doors (`border_shadow_linger_s > 0.25 s`)** | `max_shadow_emit_lost` too large near borders | Keep `max_shadow_emit_lost` small (`1–3` on 25–30 fps) and `border_exit_margin_px >= 24.0` |
| **Forklift / car ID splits mid-turn (`proximity`)** | Class flips `car <-> truck` | Confirm `check_class_match: false` |
| **Two side-by-side walkers merge onto one ID** | `min_iou_diff_new` too small in dense crowd | Raise `min_iou_diff_new` (`0.37 -> 0.60–0.70`) |
| **Vehicle misses fast turn association under frame skipping** | Fast displacement drops IoU below `0.25` | Lower `iou_thresh_high` to `0.20` and `iou_thresh_low` to `0.10` |

---

## 6. Build, Test, Score & Visually Approve Side-by-Side Videos

### Step A — Rebuild Sidecar `.so` & Run C++/Python Parity Tests (when C++ changes)
`libnvds_vema2_tracker.so` is compiled into the `deepstream-sidecar` Docker image (`perception/pipeline/Dockerfile`). Whenever `perception/trackers/vema2/*.{hpp,cpp}` changes, rebuild the sidecar image before running DeepStream benchmarks:
```bash
# On the DeepStream server after `git pull`:
docker compose build deepstream-sidecar

# Optional standalone C++ unit verification (inside container or with DeepStream headers):
cd perception/trackers/vema2 && g++ -O3 -std=c++17 -Wall -Wextra -Werror -Iinclude -I/opt/nvidia/deepstream/deepstream/sources/includes src/vema2_tracker.cpp tests/test_vema2_cpp.cpp -o /tmp/test_vema2 && /tmp/test_vema2 && cd ../../..
pytest vema_tracker/tests/ tests/test_tracker_ab_eval.py tests/test_render_tracker_ab.py -q
```

### Step B — Run Automated NvDCF vs VEMA v2 A/B Scorecard
```bash
# Example: Barber product golden gate check
docker compose rm -sf deepstream-sidecar >/dev/null 2>&1 || true
GOLDEN_IDLE=180 GOLDEN_MAX=1800 tools/engine_golden_run.sh tests/configs/golden-barber.yaml /tmp/barber_nvdcf.jsonl
docker compose logs --no-log-prefix deepstream-sidecar > /tmp/barber_nvdcf.log

docker compose rm -sf deepstream-sidecar >/dev/null 2>&1 || true
GOLDEN_IDLE=180 GOLDEN_MAX=1800 tools/engine_golden_run.sh tests/configs/golden-barber-vema2.yaml /tmp/barber_vema2.jsonl
docker compose logs --no-log-prefix deepstream-sidecar > /tmp/barber_vema2.log

python3 tools/tracker_ab_eval.py --product barber \
  --nvdcf /tmp/barber_nvdcf.jsonl --log-nvdcf /tmp/barber_nvdcf.log \
  --vema3d /tmp/barber_vema2.jsonl --log-vema3d /tmp/barber_vema2.log \
  --markdown -
```

### Step C — Render Side-by-Side Before/After Video for Eye Approval
Use [`tools/render_tracker_ab.py`](../../../tools/render_tracker_ab.py) directly on the server. Each panel draws **unique golden-angle colors per ID**, **`1.2 s` motion trails**, **dashed boxes for shadow coasting**, **`NEW` yellow flashes for track spawns after `1 s`**, and a **live HUD counter** comparing `Unique IDs` and `New Spawns` in real time.

1. **One-command run + render on a golden clip** (e.g., `barber`, `attendance`, `proximity`, `zones`, `multicam`):
```bash
python3 tools/render_tracker_ab.py run \
  --product barber \
  --out /tmp/barber_before_after.mp4
```

2. **One-command run + render on a custom event window of any `.mp4` on the server** (e.g., testing `camera01_2250-2340.mp4` around the `t=890–930 s` barber leaning crossing):
```bash
python3 tools/render_tracker_ab.py run \
  --product barber \
  --video /path/on/server/camera01_2250-2340.mp4 \
  --start 890 --duration 40 \
  --out /tmp/barber_crossing_before_after.mp4
```
*(This writes `/tmp/barber_crossing_before_after.mp4`, `/tmp/barber_crossing_before_after.scorecard.md`, and `/tmp/barber_crossing_before_after.scorecard.json`.)*

3. **Render immediately from already-recorded `.jsonl` files** (fast, no DeepStream re-run):
```bash
python3 tools/render_tracker_ab.py render \
  --video data/uploads/golden/barber.mp4 \
  --before /tmp/barber_nvdcf.jsonl \
  --after /tmp/barber_vema2.jsonl \
  --config tests/configs/golden-barber.yaml \
  --out /tmp/barber_before_after.mp4
```
