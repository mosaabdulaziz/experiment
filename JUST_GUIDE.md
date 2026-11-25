# `just` Guide

How to drive this repo's dev environment day to day. For what's actually
happening under the hood (architecture, data layout, enrollment, adding
cameras), see [`products/attendance/RUNBOOK.md`](products/attendance/RUNBOOK.md)
— this file is the "which command do I run" cheat sheet.

Everything below runs from this repo's own root:

```bash
cd surveillance-camera-sys
```

## Cheat sheet

| I want to... | Run |
|---|---|
| Set up a fresh clone | `just bootstrap` |
| Fetch models, golden clips and sample data (DVC, [DATA.md](DATA.md)) | `just data-mount` then `just data-pull [groups]` |
| Switch the active UI | `just frontend-use iris` or `just frontend-use attendance` |
| Start, no rebuild (the common case) | `just up "cam-15,cam-18"` |
| Start, rebuilding images/deps first | `just up-build "cam-15,cam-18"` |
| Start with fire/attendance split into separate containers | `just up-roles "..."` / `just up-build-roles "..."` |
| Stop everything | `just down` |
| Tail all logs | `just logs` |
| Wipe stale test data, keep employees/photos | `just clear` |
| Restart the video replay from 00:00 | `just replay` |
| Stop + prune docker images (keeps real data) | `just clean` |

## First time on a machine

```bash
just bootstrap
```

Checks required tools (`docker`, the compose plugin, `just`, `python3`,
`node`, `ffmpeg`, `nvidia-smi`), initializes the `frontend` submodule if
needed, creates `.env` from `.env.example`, and checks for the non-git data
(videos, face photos, `attendance.db`). On this dev VM it finds the
supervisor bundle automatically and extracts it — no manual step. On any
other machine, it prints the exact `scp`/`tar` commands to fetch it.

Safe to re-run anytime — it only fixes what's actually missing or wrong,
and tells you plainly if something needs your attention (a real secret in
`.env`, for instance). You don't have to run it before every `up` — `up`
and friends run a fast version of the same checks themselves and refuse to
start with a clear error if something's missing, instead of failing deep
inside a docker-compose or Python traceback.

## Picking a frontend

Only one frontend runs at a time. Two variants live as branches of the same
`frontend` submodule:

| Variant | Branch | Port |
|---|---|---|
| `iris` | `feat/attendance-page` | `:5173/attendance` |
| `attendance` | `feat/fire-smoke-tab` | `:5180` |

```bash
just frontend-use iris
just frontend-use attendance
```

This checks out the matching branch inside `frontend/` and **stages** (does
not commit) the updated submodule pointer in this repo. Commit it yourself
when you're happy with the switch:

```bash
just frontend-use attendance
git commit -m "Switch active frontend to attendance"
just up "cam-15,cam-18"
```

Skipping the commit is fine too — everything still works off what's
actually on disk in `frontend/`; the pointer just stays uncommitted until
you do.

## Starting the stack: four combinations

Two independent choices — rebuild or not, and which DeepStream topology —
give four start recipes. Rebuilding and the topology are **not** tied to
which frontend is active; pick each independently for what you're doing.

| | No rebuild | Rebuild first |
|---|---|---|
| **Single shared sidecar** (handles fire *and* attendance roles together, per-camera `roles:` tags in `sidecar.yaml`) | `just up [cams]` | `just up-build [cams]` |
| **Split into `deepstream-sidecar-fire` + `deepstream-sidecar-attendance`** | `just up-roles [cams]` | `just up-build-roles [cams]` |

- Use the **rebuild** variants after pulling backend code changes, or on a
  fresh clone's first run (also builds the TensorRT engines — slow the
  first time, fast after).
- Use the **roles** variants only when you specifically want fire and
  attendance perception running as separate containers. Otherwise the
  single-sidecar path is simpler and is what the plain `up`/`up-build`
  recipes give you.
- `up`/`up-build` and `up-roles`/`up-build-roles` are **mutually
  exclusive** against the same camera set — don't run both at once.
- `[cams]` is a comma-separated cam-id list, e.g. `"cam-9"`,
  `"cam-15,cam-18"`. **Don't omit it** — it silently defaults to `cam-9`,
  an older test camera, if you do.

Whichever you run, the final line tells you the dashboard URL for
whichever frontend is currently active:

```
up (cam-15,cam-18). Dashboard: http://localhost:5180
```

## Stopping

```bash
just down
```

Stops the docker-compose stack (whichever topology is up — `down` tears
down both profile-less and `roles`-profile services) and the three host
processes (CHIRLA streamer, products/attendance, frontend dev server).
Also sweeps orphaned processes/publishers this repo started that a crashed
run might have left behind. Safe to run even if nothing's up.

## Maintenance

- **`just clear`** — wipes `sighting`, `desk_interval`, `violation` rows
  only. Keeps `employee`, photos, and CHIRLA data. Run before a clean
  verification run so old sessions' data doesn't bleed into it.
- **`just clean`** — `down`, plus prunes this project's docker images.
  Still never touches `attendance.db`, face photos, or CHIRLA data — those
  took real work to build.
- **`just replay [cams]`** — restarts the video replay from 00:00 without
  touching docker or the other host processes. No argument replays
  whatever's currently streaming.
- **`just logs`** — tails the docker-compose stack and all three host
  process logs together.

## Common workflows

**New teammate, fresh clone:**
```bash
git clone --recurse-submodules <repo-url>
cd surveillance-camera-sys
git checkout feat/fire-smoke-attendance
just bootstrap
just up-build "cam-15,cam-18"
```

**Daily start/stop, nothing changed:**
```bash
just up "cam-15,cam-18"
# ... work ...
just down
```

**Pulled new backend code:**
```bash
git pull
just up-build "cam-15,cam-18"
```

**Switching which UI you're testing against:**
```bash
just down
just frontend-use attendance   # or iris
just up "cam-15,cam-18"
```

**Testing the fire/attendance roles split:**
```bash
just down
just up-roles "cam-15,cam-18"
```

**Before a demo or a dedicated verification run:**
```bash
just clear
just up "cam-15,cam-18"
```

## Troubleshooting

Start with `just logs`. Host-process logs also live individually under
`.run/*.log`. For anything pipeline-specific (nobody recognized, a wedged
camera stream, recognition thresholds), see
[`RUNBOOK.md`'s Troubleshooting section](products/attendance/RUNBOOK.md#troubleshooting).
