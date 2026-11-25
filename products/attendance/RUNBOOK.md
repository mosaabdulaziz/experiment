# Attendance + Productivity Monitoring — Runbook

Operational guide: how to run the pipeline, and how to put **new data** through
it (new videos, new cameras, new people). Architecture and design history live
in [PIPELINE.md](PIPELINE.md); this file is the "I just want to drive it" layer.

## Running it (on the dev VM)

Everything is orchestrated from this repo's own root via its `justfile`:

```bash
cd surveillance-camera-sys
just up "cam-15,cam-18"       # start everything (no rebuild)
just up-build "cam-15,cam-18" # same, but rebuild docker images + venv/node_modules first
just down                     # stop everything
just logs                     # tail all logs (containers + host processes) at once
just clear                    # wipe the sighting + desk-interval log only (keeps
                              # employees, photos, videos) — do this before a clean test run
```

The cams argument is a comma-separated cam-id list. **Don't omit it** — the
default is `cam-9`, an older seq_004 test camera. The two productivity-
monitoring cameras are:

| cam id | source | zones |
|--------|--------|-------|
| cam-15 | CHIRLA seq_020 camera 4 (enrollment source video) | 4 seats |
| cam-18 | CHIRLA seq_020 camera 3 (open-plan room, elevated) | 5 seats |

Then open whichever URL `just up` printed (replace `localhost` with the
machine's IP if browsing remotely):

- Attendance variant (`:5180`) — dashboard, plus `/#/monitoring` for
  productivity monitoring: live video with zone/state overlays, per-employee
  working/away timers, the zone editor (✎ on a camera), and the policy knobs
  (debounce, sustained-phone time, seated aspect, grace)
- Iris variant (`:5173/attendance`) — the original iris-frontend attendance page

Switch which one is active with `just frontend-use iris` or
`just frontend-use attendance` before `just up`.

## What lives where (and what is NOT in git)

Two repos: this one (`surveillance-camera-sys`, orchestration root — the
`justfile` lives here now) and `iris-frontend` (wired in as the `frontend`
submodule, holding both UI variants as branches — `feat/attendance-page` is
the original iris UI, `feat/fire-smoke-tab` is the attendance/fire-smoke UI,
currently active). Switch between them with `just frontend-use iris` or
`just frontend-use attendance`. Only one runs at a time.

```
surveillance-camera-sys/                 branch feat/fire-smoke-attendance
├── justfile                             orchestrates everything
├── frontend/                            submodule -> iris-frontend, active branch = feat/fire-smoke-tab
├── data/uploads/*.mp4                   test videos; the seq_020 ones NOT in git
├── data/face_photos/                    NOT in git -- enrollment photos
│   ├── person-N.jpg                     primary photo per person
│   └── multi/person-N/*.jpg             extra per-angle crops (multi-embedding)
├── data/chirla_converted/               NOT in git -- all converted CHIRLA seqs (4.4G)
└── products/attendance/data/             NOT in git --
    ├── attendance.db                    employee rows + sighting/desk-interval log
    ├── zones.json                       seat zones (self-seeds from zones.seed.json)
    └── policy.json                      the 4 policy knobs (self-seeds from defaults)
```

**The `employee` table is load-bearing**: nothing in the code recreates it. The
justfile's registration step only UPDATEs names on existing rows, and any
sighting whose face-id has no employee row is dropped as "unknown". On a fresh
machine you must bring `attendance.db` (or insert rows — see "Enrolling a new
person").

### The bundle

`attendance-supervisor-bundle.tar.gz` (on the dev VM, at
`/data/IRIS/kw-repo/attendance-supervisor-bundle.tar.gz`) is a plain tar.gz —
not a git thing — that packs exactly the non-git files above, laid out
relative to this repo's own root:

- `data/uploads/` — the two seq_020 videos (cam-15 / cam-18, ~340M)
- `data/face_photos/` — all enrollment photos incl. the `multi/` angle crops
- `products/attendance/data/attendance.db` — a **clean** copy: the 10 employee
  rows kept, the sighting/desk-interval log stripped, so a new machine starts
  with an empty table

Why: cloning this repo alone gives you a pipeline that can't run — no
videos and no enrolled people (every sighting dropped as "unknown"). `just
bootstrap` checks for exactly this. On the dev VM itself it finds the bundle
at the fixed path above and extracts it automatically — no manual step at
all. Anywhere else, it names exactly what's missing and prints the
`scp`/`tar` commands to fetch it.

## Setting up on your own machine

Prerequisites:

- An NVIDIA GPU with driver installed — DeepStream won't run without it
- Docker + the compose plugin + nvidia-container-toolkit with CDI enabled
  (the recipes use `--device nvidia.com/gpu=all`)
- `just`, Python 3.10+, Node 18+, ffmpeg
- Internet for the first build, plus repo access: an ssh key registered on
  gitlab.urca.ai (surveillance-camera-sys, iris-frontend)

1. Clone this repo with its submodule:

   ```bash
   git clone --recurse-submodules ssh://git@gitlab.urca.ai:2222/Mohamed.Abdulhafez/surveillance-camera-sys.git
   cd surveillance-camera-sys
   git checkout feat/fire-smoke-attendance
   just bootstrap
   ```

   `bootstrap` checks required tools, initializes the submodule if you
   didn't clone with `--recurse-submodules`, creates `.env` from the
   template, and tells you exactly what's still missing (it will report the
   bundle below as missing — that's expected at this point).

2. Copy the bundle from the dev VM and extract it here — it overlays the
   face photos, the two seq_020 videos, and the 10-employee `attendance.db`
   into the right places:

   ```bash
   scp <user>@10.2.0.60:/data/IRIS/kw-repo/attendance-supervisor-bundle.tar.gz .
   tar xzf attendance-supervisor-bundle.tar.gz -C .
   just bootstrap   # re-run -- should report everything ok now
   ```

3. `just up-build "cam-15,cam-18"`. The first run is slow (builds the
   DeepStream sidecar image + TensorRT engines); after that, plain
   `just up` / `just down`.

4. Open the dashboard URL `just up` prints (`:5180` for the attendance
   variant, `:5173/attendance` for the iris variant). Seat zones and policy
   knobs self-seed from the committed defaults — nothing else to configure.

Caveats: the bundle's `attendance.db` is a snapshot — if people are enrolled
or renamed on the dev VM later, regenerate the bundle before handing it over
again. Other CHIRLA sequences are not in the bundle (4.4G); scp the specific
`data/chirla_converted/<seq>/` folder wanted and follow "Adding a new video
as a new camera".

## Adding a new video as a new camera

Six touchpoints, in pipeline order. Use the next free cam-id (cam-19, …).

1. **Convert** to h264 mp4 (CHIRLA ships .avi; the sidecar wants h264):

   ```bash
   ffmpeg -i input.avi -c:v libx264 -crf 23 -an output_h264.mp4
   ffprobe -v error -show_entries format=duration -of csv=p=0 output_h264.mp4
   ```

   Note the duration — step 3 needs it. (All CHIRLA sequences are already
   converted under `data/chirla_converted/<seq>/`.)

2. **Publish it**: copy the mp4 into `data/uploads/`, then map it in
   `test_streamer/stream_recordings.py` → `CAM_FILES`:

   ```python
   "cam-19": "my_new_video_h264.mp4",
   ```

3. **Register it with products/attendance** in `products/attendance/cameras.py`:
   add the same cam-id to `CAM_FILENAMES` (source filename, used as a label)
   and to `CAM_DURATIONS` (the exact ffprobe duration — the streamer loops the
   file forever, and this is how real elapsed time is modded into "position in
   file" for the wall-clock anchoring). The simulated recording start date is
   pinned in `server.py` (`recording_start`) — all cams share it; the daily
   dashboard view is pinned to that spike day.

4. **Feed it to DeepStream** in `perception/pipeline/sidecar.yaml` → `sources:`:

   ```yaml
   - {id: cam-19, uri: "rtsp://go2rtc:8554/raw-cam-19"}
   ```

   ...and add the matching `raw-cam-19:` placeholder + derived `cam-19:`
   entry to `go2rtc.yaml` (see cam-9..18 there for the pattern) if it isn't
   already declared.

5. **Show it in the UI**: both variants live in the `frontend` submodule now
   (one branch each) — on `feat/fire-smoke-tab`, add the cam-id to
   `ATTENDANCE_CAM_IDS` and a label to `CAM_LABELS` in
   `frontend/src/components.tsx`; on `feat/attendance-page`, add it to
   `ATTENDANCE_CAM_IDS` in `frontend/src/pages/AttendancePage.tsx` if the
   :5173 page should show it too. `just frontend-use <variant>` switches
   which branch is checked out to edit.

6. **Restart and draw zones**: `just down && just up cam-19` (sidecar.yaml is
   read at container start, so a restart is required). Then open
   `#/monitoring`, hit ✎ on the camera, and click out each seat polygon —
   saved to `products/attendance/data/zones.json` and hot-reloaded within
   seconds, no restart. To make zones survive a fresh checkout, copy the
   camera's block into the committed `products/attendance/zones.seed.json`
   (the seed is only used when `data/zones.json` doesn't exist yet).

A camera with no zones still produces attendance (sightings, first/last seen);
zones are only needed for the working/away desk monitoring.

## Enrolling everyone in a video (no photos)

To start from a clean roster made of whoever appears in a clip (2026-11-02):

```bash
just attendance-enroll-video                                   # data/uploads/attendance/Bus/bus-cabin.mp4
just attendance-enroll-video <video> "Staff" "--link-sim 0.5"  # video, name prefix, extra flags
```

Faces are detected in every frame, followed with IoU tracklets, and the tracklets that are the same face are merged
by ArcFace similarity (two faces in one frame are never merged). Each person becomes an employee "Person N" (nearest
face first) with their best crop as `data/face_photos/person-N.jpg` and a few more under `multi/person-N/`.
The old roster (`attendance.db`, `face_photos`) is copied to `products/attendance/data/roster-backup-<time>/` first; copy
it back to undo. Check `products/attendance/data/enrolment-report/contact_sheet.jpg` (one row per person; two rows
for one person: lower `--link-sim`; two people in one row: raise it) and `enrolment.json`.
Then `SIDECAR_CONFIG=tests/configs/attendance-bus.yaml just up cam-22`. The sidecar picks up a newly registered
gallery within 30 s, so names appear about half a minute after the stack starts. Faces under 24 px are ignored by
`face_detect` (`min_pixels`); `tests/configs/attendance-bus.yaml` lowers it to 16 for the far passenger.

### Bus whitelist (only allowed people on the bus)

`just attendance-bus [denied]` is the whole demo in one command: enrol from the clip if the roster is empty, start the
stack on it, then apply a whitelist. `denied` (default "Person 2", `""` for nobody) is marked not allowed.

It reuses the restricted-zone machinery (zones.py, violations.py), nothing new is detected: `bus_access.py` turns the
access check on for `cam-22` with `allowed_roles: [passenger]` (see below for where it applies) and gives every
employee the role `passenger`. A recognised person with any other role (or none) opens a `role_not_allowed`
violation, a body nobody recognises opens `unidentified` (a stranger on the bus); both appear in the dashboard's
violations panel after the debounce (3 s, `violation_debounce_seconds`) and close when the person is allowed again.

```bash
just attendance-bus-deny  "Person 3"   # role -> visitor: a violation opens within seconds, no restart
just attendance-bus-allow "Person 3"   # role -> passenger: it closes
just attendance-bus-access             # who is allowed, today's violations
```

**Where the check applies.** Per camera, in the attendance policy: `access_control: {"cam-22": {"allowed_roles":
["passenger"]}}` (`PUT /api/attendance/policy`). A camera listed there is checked **inside the restricted zones drawn
on it** (Cameras page, each zone with its own allowed roles) and, **when none is drawn, on the whole stream**.
A camera that is not listed is not checked unless it has a drawn restricted zone, as before, so the real cameras do not
start alerting on every employee that has no role. The bus demo draws no zone, so the whole cabin is checked; draw a
restricted zone on `cam-22` and only that area is (try one over one seat: a person denied outside it raises nothing).
The alerts page shows the scope of each camera.

**Alerts in the dashboard.** A new violation pops up at the side of every page (15 s, with the person's photo; a
stranger shows "unrecognised person") and the bell in the left rail shows how many are open. The bell opens the alerts
page, `#/attendance/alerts`: the alerts open now, the whitelist (who is authorised, with an allow/deny button per
person, which sets their role) and the violations log. The popup reads the same violations the engine already writes,
so it works for any restricted zone, not only the bus.

The recipe removes the saved zone before the stack starts and applies it once everyone has been recognised: the
sidecar loads the face gallery ~30 s after start, and with the zone live earlier every body would be flagged
`unidentified` just for not being recognised yet.

## Enrolling a new person

Names here are simulation placeholders — CHIRLA subjects are anonymized.

1. **Primary photo**: a clear, frontal crop as `data/face_photos/person-N.jpg`
   (next free N). To harvest candidates from a video, see PIPELINE.md's
   face-sweep notes — that's how the seq_020 roster was built.
2. **Optional but recommended** — extra angles: more crops of the same person
   (profile, distance, lighting) under `data/face_photos/multi/person-N/`.
   These are appended as extra gallery embeddings and are what make
   recognition survive poor angles.
3. **Justfile roster**: in `_register_people`, append the display name to the
   `names` array and N to the `for id in …` list.
4. **Employee row** (one-time, from the parent dir):

   ```bash
   python3 -c "import sqlite3; c = sqlite3.connect('surveillance-camera-sys/products/attendance/data/attendance.db'); c.execute(\"INSERT INTO employee (id, name, gallery_rec_id, active) VALUES ('emp-N', 'Display Name', 'person-N', 1)\"); c.commit()"
   ```

   `gallery_rec_id` must equal the face-id (`person-N`) and the display name
   must match the justfile's — the name-based fallback match depends on it.
5. `just up …` — registration re-runs idempotently on every up (the redis
   gallery is wiped on every down by design).

## Using other CHIRLA sequences

Nine sequences are already converted and ready to stream — skip step 1 above
and start at step 2 (copy the file into `data/uploads/` and wire up a cam-id).
On the dev VM:

```
/data/IRIS/kw-repo/surveillance-camera-sys/data/chirla_converted/
├── seq_000/  seq_001/  seq_002/  seq_006/  seq_007/
├── seq_020/            <- the enrolled sequence (cam-15..18 come from here)
└── seq_024/  seq_025/  seq_026/
```

Each holds all 7 synchronized camera angles as `camera_N_<timestamp>_h264.mp4`.
(seq_004 isn't in this dir — its conversions were done earlier and sit directly
in `data/uploads/` as cam-3, cam-9..14.)

The raw dataset (original .avi, incl. annotations) is at
`/data/IRIS/data/CHIRLA_data/CHIRLA/videos/<seq>/`, and the CHIRLA repo with
its downloader tooling at `/data/IRIS/data/CHIRLA` — only needed for a
sequence not converted yet.

Note the enrolled roster was built from seq_020: people in other sequences are
mostly different subjects and will (correctly) show as unrecognized until
enrolled.

## Real live cameras (RTSP) — not yet

The sidecar itself takes any RTSP uri, but the attendance layer currently
assumes looped recordings: `server.py` pins a simulated `recording_start`,
wall-clock anchoring mods elapsed time by `CAM_DURATIONS`, and the consumer
only watches cams listed in `CAM_FILENAMES`. Pointing at a real camera needs
those three spots generalized (real wall-clock = now, no looping) — small,
contained changes, but code changes nonetheless.

## Upgrading / Re-enrolling the face gallery (`buffalo_l` → `auraface`)

The default face recognizer migrated from InsightFace `buffalo_l` (`w600k_r50.onnx`) to `auraface` (`glintr100.onnx`), paired with `face_det_lite.onnx` (`threshold: 0.28`, `margin: 0.05`). Because both models output 512-D vectors, **existing Redis galleries built under `buffalo_l` (`attendance:face:gallery`) load without error but produce meaningless cosine scores** against `auraface` probe embeddings.

Before deploying or running after this upgrade against an existing/persisted Redis instance:

```bash
# Drop stale buffalo_l embeddings if Redis persisted across upgrades, then re-enroll from data/face_photos/
docker exec "$(docker compose ps -q redis)" redis-cli DEL attendance:face:gallery
just _register_people   # or run products/attendance/reregister_all.py inside the sidecar container
```

`just down && just up ...` recreates Redis cleanly and automatically runs `products/attendance/reregister_all.py` via `_register_people`.

## Troubleshooting

- `just logs` first. Host-process logs live in `<parent>/.run/*.log`.
- Stream up but frames frozen / cam wedged: `just down && just up …` — the
  `replay` recipe also self-heals a wedged sidecar (frame-hash check + restart).
- Nobody recognized at all: check the registration step ran (it prints
  `registered <name> (person-N)` on every up) and that `attendance.db` has the
  employee rows (see above — an empty db drops every sighting).
- Recognition thresholds and vote settings: `perception/pipeline/sidecar.yaml`
  (`face_recognition_*`, `sample_fps`); monitoring behavior knobs: the
  settings panel on `#/monitoring` (persisted in `data/policy.json`).
