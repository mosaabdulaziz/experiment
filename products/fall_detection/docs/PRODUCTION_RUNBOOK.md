# Production runbook — fall detection on the 5090

Everything below runs **on the deployment server** (the RTX 5090, Ubuntu, Docker), from the
repository root. The commands are copy-pasteable in order; each block says what it proves.

Set once per shell:

```bash
cd /path/to/iris
export FALL_CORPUS=/data/fall_corpus          # the labelled corpus (no default on Linux)
export FALL_MEASUREMENTS=/data/fall_evidence  # where the recorded runs go (optional)
```

---

## 1. Once, before the first run

```bash
# 1. the tooling, the frontend and .env
just bootstrap

# 2. the two checkpoints -- they are in git now, under products/fall_detection/weights/.
#    The boot gate verifies their SHA-256 at every start and refuses on a mismatch.

# 3. the dashboard's dependencies -- the lockfile is committed, so `npm ci`, not `npm install`
cd frontend && npm ci && cd -

# 4. rebuild the image, so the pose parser compiles against DeepStream's REAL header
docker compose --profile fall build deepstream-sidecar-fall
```

**4b.** The pose ONNX and the checkpoints are versioned with DVC: `just data-pull models-fall` fetches
`perception/models/DeepStream-Yolo-Pose/yolo26x-pose.onnx` and `products/fall_detection/weights/`.
Its TensorRT engine is built by nvinfer on the sidecar's first start.

---

## 2. Bring it up

```bash
just fall-up      # uv sync -> preflight -> redis/minio/go2rtc -> the service -> the sidecar -> the PLAYING wait
just fall-log     # tail the service log (.run/fall.log)
```

`just fall-up` **exits non-zero** if the service never answers `/status` or the sidecar never
reaches `pipeline PLAYING`. That is the first gate: it refuses to start on a GPU-less box, a
missing checkpoint or a wrong SHA-256, in seconds — before the minutes of TensorRT build.

It also refuses when another sidecar already holds the fall product's camera (`cam-1`). That
is not a fault: one process reads a camera. Stop the other sidecar first.

---

## 3. Prove it is alive (the checks that must pass on the server)

```bash
# the service's own status: device, weights_verified, classifier_ready, the live paths
curl -sf http://localhost:8004/status | python3 -m json.tool

# the sidecar's log: the GIE it built and the interval it read from the config file
docker compose --profile fall logs deepstream-sidecar-fall | grep -Ei "sgie_pose|secondary-reinfer-interval|PLAYING"

# the pose field on the camera's own meta stream: 17 keypoints per person
docker compose exec -T redis redis-cli --scan --pattern 'iris:meta:cam-1*' | head

# the dashboard's own service -- /stream/<cam>/frame and the shell's product list come from it
docker compose up -d iris
```

**Expected:** `device: cuda:0`, `weights_verified: true`, `classifier_ready: true`; the engine log
names `sgie_pose` and the interval it read (2); `iris:meta:cam-1` carries a `pose` field whose
items each hold 17 keypoints.

The dashboard itself:

```bash
cd frontend && npm run dev        # http://<host>:5180 -> Fall Detection
```

**Do not use `just up` for this.** It starts the surveillance stack, including
`deepstream-sidecar` on `cam-1`, which the fall preflight then refuses to sit beside (§2).
`docker compose up -d iris` + the dev server is the dashboard without the other sidecar.

What the tab tells you, in its own words:

| On screen | Means |
|---|---|
| a red bar, `the fall service: …` | the service itself is not answering — status, cameras and the review queue are stale |
| a red bar, `the events feed: …` | only `/events` is failing; the frame, the badges and the queue are live, and the event list is the last good one. It clears on its own when `/events` answers again |
| a `candidate` row chip in **amber** | waiting for a verdict |
| a `confirmed` row chip in **green** | an operator confirmed it |
| `acknowledged` / `dismissed` chips in **grey** | seen without a confirmation — a rejection must never look like one |

---

## 4. The test suites on the server

```bash
just fall-test   # the product's own suite + the copied vendor's nine test files
just test        # the repo's root suite
just check       # the root suite + the barber's + the frontend build

# the two dashboard suites (barber + fall; attendance has no tests, and `--if-present` skips it)
cd frontend && npm run test --workspaces --if-present && cd -
```

On the server the **corpus replays run** (they are deselected only on a dev box, where the GPU is
not the deployment card). `just fall-test` needs the corpus at `$FALL_CORPUS`; without it those
two tests skip with the reason.

The gated measurements run too, once `FALL_MEASUREMENTS` is set (§5).

---

## 5. The measurements the server owes before the pilot

```bash
# N1 -- the GPU budget: the pose stage's share of the card.
#      The "before" is the pipeline without the pose module, the "after" is with it.
tools/engine_gpu_budget.sh tests/configs/golden-attendance.yaml
tools/engine_gpu_budget.sh tests/configs/golden-fall_detection.yaml

# N2 + N4 -- the corpus replay: precision, recall, false alarms/hour, the action path's share
(cd products/fall_detection && .venv/bin/python -c "import json, measure; print(json.dumps(measure.action_path_eval(), indent=2))")

# N3 -- the onset-to-alert latency, end to end on the replayed footage
(cd products/fall_detection && .venv/bin/python -c "import json, measure; print(json.dumps(measure.latency(), indent=2))")

# N5 -- the classifier's cost on the 5090 (ms/forward, concurrent tracks, card utilisation)
(cd products/fall_detection && .venv/bin/python -c "import json, measure; print(json.dumps(measure.classifier_cost(), indent=2))")

# F29 -- the golden baseline: the recorded sidecar run that carries the pose field
tools/engine_golden_run.sh tests/configs/golden-fall_detection.yaml tests/baselines/golden-fall_detection.before.jsonl
```

(The three `python -c` calls run **inside** `products/fall_detection`, which is where `measure.py`
and the vendored `fall_detection` package live; `$FALL_CORPUS` decides which corpus is replayed.)

Save the outputs into `$FALL_MEASUREMENTS` (`gpu_budget.txt`, `interval_sweep.json`, the sidecar
log) — the gated tests read that directory and skip without it.

Then the gated tests stop skipping:

```bash
FALL_MEASUREMENTS=/data/fall_evidence just fall-test
```

---

## 6. Watch it on a video (no camera on the box)

The engine reads whatever URI a source names, and `file://` is a source like any other: it
plays the file, loops it (`file_loop` is on in the shared settings), and everything after it —
pose, the `pose` field, the service, the tab, the alarms — runs exactly as on a camera. Use this
to see the product work, not to measure it: the numbers in §5 need the labelled corpus.

1. Put the clip **inside the checkout** — the sidecar mounts the repo at `/workspace/iris`, so a
   path outside it does not exist in the container:

   ```bash
   cp /path/to/your_clip.mp4 data/uploads/
   ```

2. An overlay that replaces only the source and inherits every tuned knob:

   ```bash
   cat > products/fall_detection/sidecar.local.yaml <<'EOF'
   # A local clip instead of a camera -- for a visual run, not for production.
   extends: sidecar.yaml
   sources:
     - {id: cam-1, uri: "file:///workspace/iris/data/uploads/your_clip.mp4", roles: [fall_detection]}
   EOF
   ```

   The id stays `cam-1` on purpose: the service's own config
   (`products/fall_detection/fall_detection.yaml`) watches `cam-1` and reads its meta stream.

3. Bring the service and the sidecar up with the overlay (the recipe exports `.env` and runs
   compose; the overlay is what the compose service is told to read):

   ```bash
   FALL_SIDECAR_CONFIG=products/fall_detection/sidecar.local.yaml just fall-up
   ```

4. The dashboard, in another shell:

   ```bash
   docker compose up -d iris          # /stream/<cam>/frame and the shell's product list
   cd frontend && npm run dev         # http://<host>:5180 -> Fall Detection
   ```

**Expected:** `curl -sf localhost:8004/cameras` answers `["cam-1"]`; the Live page's tile draws
the annotated clip; people walking through it produce events with per-path scores; a fall in the
clip raises the alarm card and, once scored above the review threshold, a `candidate` row on the
Review page that you can confirm or dismiss.

The clip loops, so the same moments repeat — that is the file playing again, not a second event
with a new id.

When the camera is real, skip this section and point the product's own `sidecar.yaml` source at
it (the go2rtc path, or the camera's RTSP URI directly) — nothing else changes.

---

## 7. Stop, and roll back

```bash
just fall-down    # stops the service and the fall sidecar (redis/minio/go2rtc keep running)
just down         # stops the shared containers too
just clean        # down + prune this project's docker images
```

`just fall-down` is safe to run twice; it is also the recovery when `just fall-up` stops
half-way ("the service is up and idle; stop it with: just fall-down").

---

## 8. If something refuses

| Symptom | What it means | Where to look |
|---|---|---|
| the service exits in seconds | the boot gate refused: no CUDA op on `cuda:0`, a missing checkpoint, or a SHA-256 mismatch | `just fall-log` — the message names the cause |
| "the fall sidecar is not PLAYING" | nvinfer could not build or run the engine | `docker compose --profile fall logs deepstream-sidecar-fall` |
| "deepstream-sidecar-fall holds cam-1" | another sidecar already has the camera — the fall deployment replaces the other one on its cameras | stop that container first, then `just fall-up` |
| the service starts but nothing alerts | no `pose` field is reaching it | `redis-cli --scan --pattern 'iris:meta:*'` and the sidecar's log |
| the tab's bar says `the events feed: …` while the page is otherwise live | `/events` is failing (route or database); the list on screen is the last good one | `curl -s 'localhost:8004/events?since=0'` and `just fall-log` |
| every call in the tab answers 404 | the dashboard's proxy and the service disagree about the path — the service's routes are bare (`/status`), so the proxy strips all of `/fall_detection/api` | `frontend/vite.config.ts`, the `/fall_detection/api` rule |
| a test fails on the server only | the GPU-bound ones (the corpus replays, the classifier cost) | they print `skipped` with the reason when a prerequisite is missing |
