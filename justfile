# Dev environment orchestration for the products/attendance spike.
# See products/attendance/PIPELINE.md for the full architecture writeup and
# manual runbook this just wraps.
#
# Design (Option B, agreed 2026-08-30): docker-compose owns the GPU-heavy
# stack (redis, minio, deepstream-sidecar, iris app) exactly as before --
# this file does NOT containerize the other three pieces (CHIRLA streaming,
# products/attendance, the frontend dev server). They still run as plain host
# processes, tracked here via PID files in .run/, so hot-reload and normal
# host-level debugging stay exactly as they are today. `just` is only a thin
# orchestration layer on top, not a new runtime.
#
# 2026-09-07: this file used to live one directory up (an untracked parent
# dir holding this repo, iris-frontend, and attendance-frontend as three
# separate sibling clones). Moved in here and tracked in git so a plain
# `git clone --recurse-submodules` of this repo is enough on its own -- the
# frontend is now purely the `frontend` submodule, on whichever branch is
# checked out (`feat/attendance-page` = iris UI, `feat/fire-smoke-tab` =
# attendance UI; switch with `just frontend-use iris|attendance`). Only one
# frontend runs at a time now -- the old simultaneous :5173 + :5180 dev
# servers are gone along with the separate attendance-frontend checkout.
#
# Recipes:
#   just bootstrap          one-time setup: check tools, init the frontend
#                           submodule, create .env, verify required data is
#                           present -- run this first on a fresh clone
#   just frontend-use <v>   switch the active frontend: iris | attendance | barber
#   just up-build [cams]    full rebuild (docker images + venv/node_modules) and start
#   just up [cams]          start everything without rebuilding
#   just down               stop everything (docker compose down -- go2rtc included)
#   just clean              down, plus prune this project's docker images -- does NOT
#                           touch attendance.db, face photos, or CHIRLA data; those
#                           took real work to build, never wiped silently
#   just clear              wipe stale simulation state ONLY (`sighting`, `desk_interval`,
#                           `violation` -- not employees, photos, or CHIRLA data) so the
#                           next run starts clean, e.g. before a dedicated verification run
#   just logs               tail every log (containers + the 3 host processes) at once
#   just barber-up [clips]  barber shop stack: barber sidecar + products/barber (:8003),
#                           optionally streaming test clips (see products/barber/README.md);
#                           the dashboard is served by `just up` (barber is one of its products)
#                           e.g. just barber-up barber-c,barber-l,barber-r
#   just barber_site=palm-hills barber-up   the Palm Hills shop (7 cameras, its test clips by default)
#   just barber-down        stop products/barber, its streamer and the barber sidecar
#   just barber-streamer-restart   resync the running site's synced stream group (barber-c/l/r or ph-1..7) on demand
#   just barber-restart [hard]     restart products/barber only (never the GPU sidecar): soft keeps the
#                           day's data; hard clears it (visits, waits, alerts, cards, door counts, tickets)
#                           and, with test clips streaming, restarts them from the beginning too
#   just barber-test        products/barber's own tests (no GPU/redis needed)
#   just fall-up            fall detection: the service (:8004) and the fall sidecar (profile
#                           `fall`, telemetry 9105), after the camera-exclusivity preflight
#   just fall-down          stop products/fall_detection and the fall sidecar
#   just fall-log           follow the fall service's log
#   just fall-test          products/fall_detection's own tests (its policy, its service and the
#                           copied vendor's tests; no GPU/redis needed)
#   just concealment-check  preflight check: .venv, config/zones, DVC models, camera exclusivity
#   just concealment-up     concealment detection: shared infra + service (:8005) + sidecar (profile
#                           `concealment`, telemetry 9107) + dashboard URL
#   just concealment-down   stop products/concealment_detection and deepstream-sidecar-concealment
#   just concealment-log    follow the concealment service's log
#   just concealment-sidecar-log  follow deepstream-sidecar-concealment container logs
#   just concealment-test   products/concealment_detection's own tests (no GPU/redis needed)
#   just logs               (also takes a service name to follow just that one)
#
# Preset use cases (configs/presets/<name>/, see "Presets" at the bottom):
#   just multi | multicam   13-camera multi-camera vehicle tracking (MTMC + Re-ID): runs the
#                           `multicam` compose profile on products/multicam/sidecar.yaml
#   just vlm                single-camera VLM cognitive monitoring
# A preset swaps the app's config.yaml and points the pipeline at its own overlay through
# SIDECAR_CONFIG (the shared perception/pipeline/sidecar.yaml is never overwritten);
# `up`/`up-roles` restore the tracked config.yaml first, so switching back and forth
# between the attendance/fire stack and a preset is safe.
#
# [cams] is an optional comma-separated cam-id list, defaults to "cam-9" (see
# the `cams` variable below for why). Examples: `just up cam-11`,
# `just up "cam-9,cam-11"`.

frontend := "frontend"
run_dir := ".run"

# Which camera(s) to stream -- comma-separated cam-ids, passed straight
# through to stream_recordings.py's --ids. Default cam-9: the CHIRLA seq_004
# camera where Person-8 is confirmed absent (per PIPELINE.md), so a fresh
# `up` demonstrates a real present/absent split rather than everyone always
# showing present. Override per-invocation, e.g.:
#   just up cam-11
#   just up "cam-9,cam-11"
#   just up live      real cameras only (cam-1/2/3/6), no test streamer --
#                     the streamer refuses those ids (2026-09-13)
cams := "cam-9"
# The barber shop (products/barber/site.sh): empty = the running site (.run/barber-site), else bekky.
# barber-up: empty = bekky; barber-restart: empty = the running site (another one is refused: barber-down + barber-up).
# Override per call: `just barber_site=palm-hills barber-up`. An unknown name stops both before anything starts.
barber_site := ""

# Required host tools, checked by both `bootstrap` and `_preflight`.
_required_tools := "docker just python3 node ffmpeg nvidia-smi"

# Well-known location of the supervisor bundle on the dev VM (see RUNBOOK.md's
# "The bundle" section). `bootstrap` checks here first and auto-extracts if
# found -- covers the common case (a fresh clone on this same machine) with
# zero manual steps. A clone on a genuinely different machine won't find
# anything here and falls through to the scp/tar instructions instead.
_known_bundle_path := "/data/IRIS/kw-repo/attendance-supervisor-bundle.tar.gz"

# One-time setup after a fresh clone: checks required tools, initializes the
# frontend submodule, creates .env from the template, and verifies the
# non-git data (uploads/photos/attendance.db) is present -- printing exactly
# what's missing and how to get it, instead of letting `up` fail deep inside
# a docker-compose or Python traceback. See RUNBOOK.md's "Setting up on your
# own machine" section for the full prerequisites list.
bootstrap:
    #!/usr/bin/env bash
    set -uo pipefail
    missing=()
    for tool in {{_required_tools}}; do
        command -v "$tool" >/dev/null 2>&1 || missing+=("$tool")
    done
    docker compose version >/dev/null 2>&1 || missing+=("docker compose plugin")
    if [ ${#missing[@]} -gt 0 ]; then
        echo "ERROR: missing required tools:" >&2
        printf '  - %s\n' "${missing[@]}" >&2
        echo "Install these first, then re-run 'just bootstrap'." >&2
        exit 1
    fi
    echo "tools: ok ({{_required_tools}}, docker compose plugin)"

    # The dashboards are ordinary source in this repo now (2026-09-23): one app
    # under {{frontend}}/, one folder per product inside src/products/. There is
    # no submodule to initialise and no branch to keep in sync -- the product to
    # show is picked in the app itself.
    if [ ! -d {{frontend}}/node_modules ]; then
        echo "{{frontend}}/node_modules missing -- installing..."
        (cd {{frontend}} && npm install --no-audit --no-fund)
    fi
    echo "frontend: ok ({{frontend}}/, products: $(ls {{frontend}}/src/products | tr '\n' ' '))"

    # The attendance service runs as a host process from its own Python environment; a fresh clone has none.
    if [ ! -x products/attendance/.venv/bin/uvicorn ]; then
        if command -v uv >/dev/null 2>&1; then
            echo "products/attendance/.venv missing -- installing..."
            (cd products/attendance && env -u VIRTUAL_ENV uv sync -q --locked)
        else
            echo "WARNING: uv is not installed, so products/attendance/.venv was not created (https://docs.astral.sh/uv/)" >&2
        fi
    fi

    if [ ! -f .env ]; then
        cp .env.example .env
        # The example ships COMPOSE_PROJECT_NAME=<your-name-or-initials>, which Docker Compose rejects. Leave it unset:
        # docker-compose.yml then names the project "iris", which the tools expect. Set your own to run two checkouts.
        sed -i 's/^COMPOSE_PROJECT_NAME=<your-name-or-initials>/# &   (unset = "iris"; set it to run a second checkout)/' .env
        echo ".env created from .env.example -- fill in real values before 'just up':"
    fi
    placeholders=$(grep -v '^[[:space:]]*#' .env 2>/dev/null | grep -E '(change-me|sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx|^RTSP_USER=viewer$|<your-)' || true)
    if [ -n "$placeholders" ]; then
        echo "WARNING: .env still has placeholder values -- fill these in:" >&2
        echo "$placeholders" | sed 's/=.*/=.../' >&2
    fi

    # data/uploads holds recorded clips for the replay cameras and test tools.
    # A live deployment (real cameras over RTSP) never reads it, so it is not
    # required -- only warned about (2026-09-24: production moved without it).
    _data_missing() {
        data_missing=()
        { [ -d data/face_photos ] && [ -n "$(ls -A data/face_photos 2>/dev/null)" ]; } || data_missing+=("data/face_photos/")
        [ -s products/attendance/data/attendance.db ] || data_missing+=("products/attendance/data/attendance.db")
    }
    _data_missing
    # The bundle also sits at data/attendance-supervisor-bundle.tar.gz when it was
    # copied into a checkout. Its paths predate the monorepo layout: every member is
    # prefixed `surveillance-camera-sys/`, the database is under attendance-service/data/,
    # and a stray top-level `justfile` is in it -- extracting it straight into the repo
    # root (as this recipe used to) would overwrite THIS justfile and leave the database
    # where nothing looks for it. So strip the prefix (which also skips that justfile)
    # and move the old service folder's data to products/attendance/data/.
    # Models, golden clips and the supervisor bundle are versioned in DVC, not git (DATA.md).
    if command -v dvc >/dev/null 2>&1 && { [ "$(dvc config core.remote 2>/dev/null)" != onedrive ] || [ -d .data-remote/dvc-store ]; }; then
        echo "dvc: pulling the default data groups..."
        just data-pull || echo "WARNING: dvc pull failed -- models or the bundle may be missing" >&2
    else
        echo "note: no usable DVC remote (dvc missing, or the OneDrive mount is down) -- 'just data-pull' (after 'just data-mount' for OneDrive) fetches models and sample data (DATA.md)" >&2
    fi
    bundle=""
    for candidate in "{{_known_bundle_path}}" data/attendance-supervisor-bundle.tar.gz; do
        [ -f "$candidate" ] && { bundle="$candidate"; break; }
    done
    if [ ${#data_missing[@]} -gt 0 ] && [ -n "$bundle" ]; then
        echo "data missing -- found the supervisor bundle at $bundle, extracting..."
        tar xzf "$bundle" --strip-components=1 -C . --exclude='justfile'
        # The bundle predates the use-case folders in data/uploads: file its CHIRLA clips where the streamer looks.
        mkdir -p data/uploads/attendance
        for clip in data/uploads/camera_*.mp4; do [ -f "$clip" ] && mv -n "$clip" data/uploads/attendance/; done
        if [ -d attendance-service/data ]; then
            mkdir -p products/attendance/data
            cp -a --update=none attendance-service/data/. products/attendance/data/
            rm -rf attendance-service
        fi
        _data_missing
    fi
    if [ ${#data_missing[@]} -gt 0 ]; then
        echo "ERROR: missing required data (not constructible here -- fetch the supervisor bundle):" >&2
        printf '  - %s\n' "${data_missing[@]}" >&2
        echo "  scp <user>@10.2.0.60:/data/IRIS/kw-repo/attendance-supervisor-bundle.tar.gz data/" >&2
        echo "  just bootstrap     # unpacks it into the right folders; do not tar-extract it into the repo root" >&2
        exit 1
    fi
    { [ -d data/uploads ] && [ -n "$(ls -A data/uploads 2>/dev/null)" ]; } \
        || echo "note: data/uploads/ is empty -- fine for live cameras; replay modes need its clips (see the bundle above)"
    echo "data: ok (face photos, attendance.db present)"
    echo
    echo "bootstrap complete -- 'just up' or 'just up-build' next."

# Fast checks only (no cloning/network) so skipping `bootstrap` still fails
# loudly and immediately instead of inside a docker-compose or Python
# traceback several layers in. Depended on by up/up-build/up-roles/up-build-roles.
_preflight:
    #!/usr/bin/env bash
    set -euo pipefail
    missing=()
    for tool in {{_required_tools}}; do
        command -v "$tool" >/dev/null 2>&1 || missing+=("$tool")
    done
    docker compose version >/dev/null 2>&1 || missing+=("docker compose plugin")
    [ -d {{frontend}}/node_modules ] || missing+=("{{frontend}}/node_modules missing (run: just bootstrap)")
    [ -f .env ] || missing+=(".env missing (run: just bootstrap)")
    # Not required: only the replay cameras read data/uploads (see bootstrap).
    { [ -d data/uploads ] && [ -n "$(ls -A data/uploads 2>/dev/null)" ]; } \
        || echo "note: data/uploads/ is empty -- fine for live cameras, replay modes need its clips" >&2
    { [ -d data/face_photos ] && [ -n "$(ls -A data/face_photos 2>/dev/null)" ]; } || missing+=("data/face_photos/ missing (see: just bootstrap)")
    [ -s products/attendance/data/attendance.db ] || missing+=("products/attendance/data/attendance.db missing (see: just bootstrap)")
    [ -x products/attendance/.venv/bin/uvicorn ] || missing+=("products/attendance/.venv missing (run: just bootstrap)")
    tools/data/ensure_data.sh || missing+=("models versioned in DVC are not pulled (see above)")
    if [ ${#missing[@]} -gt 0 ]; then
        echo "ERROR: not ready -- missing:" >&2
        printf '  - %s\n' "${missing[@]}" >&2
        exit 1
    fi

# A preset (`just multi`/`just vlm`) replaces the tracked config.yaml with
# configs/presets/<name>/config.yaml. The attendance/fire stack runs on the tracked
# default, so restore it when a preset is active. (The pipeline side needs no restore:
# a preset only selects an overlay through SIDECAR_CONFIG.)
_use-default-config:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -f .active_profile ]; then
        echo "preset '$(cat .active_profile)' is active -- restoring the default config.yaml"
        git checkout -- config.yaml
        rm -f .active_profile
    fi

# (Removed 2026-09-23.) `just frontend-use` switched the frontend submodule
# between one-app-per-branch checkouts. The dashboards are one app now, and the
# product is chosen inside it -- the switcher at the top, or ?product=<id>.

# Start clean and enrol whoever is IN a video: faces are detected and tracked through the clip, the ones that are the
# same face are merged, and each becomes an employee "Person N" with their best crops as enrolment photos
# (products/attendance/enroll_from_video.py). The current roster (attendance.db and data/face_photos) is copied to
# products/attendance/data/roster-backup-<time>/ first; copy it back to undo. Look at <report>/contact_sheet.jpg:
# one row per person. `just up` then registers them into the face gallery like any other employee.
#   just attendance-enroll-video                                   # the bus-cabin clip
#   just attendance-enroll-video data/uploads/attendance/X.mp4 "Staff"
#   just attendance-enroll-video data/uploads/attendance/X.mp4 Person "--link-sim 0.5 --min-face 16"   # extra flags
attendance-enroll-video video="data/uploads/attendance/Bus/bus-cabin.mp4" prefix="Person" flags="":
    #!/usr/bin/env bash
    set -euo pipefail
    [ -f "{{video}}" ] || { echo "no such video: {{video}} (versioned with DVC? try: just data-pull)" >&2; exit 1; }
    set -a; [ -f .env ] && source .env; set +a
    sidecar_image="$(docker compose config --images 2>/dev/null | grep -m1 -E 'deepstream-sidecar:latest$' || true)"
    sidecar_image="${sidecar_image:-${COMPOSE_PROJECT_NAME:-iris}-deepstream-sidecar:latest}"
    data=products/attendance/data
    backup="$data/roster-backup-$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$backup" data/face_photos
    [ -f "$data/attendance.db" ] && cp -a "$data"/attendance.db* "$backup"/ || true
    cp -a data/face_photos "$backup"/face_photos
    echo "previous roster saved in $backup"
    report="$data/enrolment-report"
    mkdir -p "$report"
    vdir="$(cd "$(dirname "{{video}}")" && pwd)"
    docker run --rm --device nvidia.com/gpu=all \
        -v "$vdir:/video:ro" \
        -v "$(pwd)/data/face_photos:/photos:rw" \
        -v "$(pwd)/$data:/db:rw" \
        -v "$(pwd)/perception:/workspace/iris/perception:ro" \
        -v "$(pwd)/products/attendance:/workspace/iris/products/attendance:ro" \
        -w /workspace/iris --entrypoint sh "$sidecar_image" -c \
        "python3 products/attendance/enroll_from_video.py --video '/video/$(basename "{{video}}")' --db /db/attendance.db \
            --photos-dir /photos --report-dir /db/enrolment-report --prefix '{{prefix}}' --reset --gpu {{flags}} \
            && chown -R $(id -u):$(id -g) /photos /db"

# One command for the bus-cabin demo: attendance on the 6 s bus clip with the people enrolled FROM it, and a bus
# WHITELIST on top: only people with the role `passenger` may be on the bus (a restricted zone over the whole cabin,
# products/attendance/bus_access.py). Everyone enrolled from the clip is allowed except `denied`, who is marked
# "not allowed" so the dashboard shows a role_not_allowed violation for them. A person nobody recognises would show
# as "unidentified". Pass "" for denied to allow everybody.
# Enrols first only when the roster is empty (a clean database made by attendance-enroll-video is reused as is; run
# `just attendance-enroll-video` yourself to redo it). The whitelist is applied once everyone has been recognised
# (the gallery loads ~30 s after start), so nobody is flagged "unidentified" just for being recognised late.
#   just attendance-bus                      # Person 2 is not allowed
#   just attendance-bus "Person 3"           # someone else
#   just attendance-bus ""                   # everybody allowed
attendance-bus denied="Person 2":
    #!/usr/bin/env bash
    set -euo pipefail
    n="$(python3 -c "import sqlite3,os; p='products/attendance/data/attendance.db'; print(sqlite3.connect(p).execute('select count(*) from employee').fetchone()[0] if os.path.exists(p) else 0)" 2>/dev/null || echo 0)"
    if [ "$n" = "0" ]; then
        echo "empty roster -- enrolling from the bus clip first"
        just attendance-enroll-video
    else
        echo "roster has $n employee(s) -- keeping it (just attendance-enroll-video to redo)"
    fi
    python3 products/attendance/bus_access.py prepare
    SIDECAR_CONFIG=tests/configs/attendance-bus.yaml IRIS_PRODUCT=attendance just up cam-22
    python3 products/attendance/bus_access.py setup --wait
    if [ -n "{{denied}}" ]; then
        python3 products/attendance/bus_access.py deny "{{denied}}"
    else
        for who in $(python3 -c "import json,urllib.request as u; print(' '.join(e['id'] for e in json.load(u.urlopen('http://localhost:8001/api/admin/employees'))))"); do
            python3 products/attendance/bus_access.py allow "$who" >/dev/null
        done
    fi
    python3 products/attendance/bus_access.py status
    echo "dashboard: http://localhost:5180/?product=attendance#/attendance  (violations panel)"

# Change who is on the bus whitelist while the bus demo runs (applies on the next cycle, no restart).
#   just attendance-bus-deny "Person 3"     role -> visitor: a role_not_allowed violation opens
#   just attendance-bus-allow "Person 3"    role -> passenger: it closes
#   just attendance-bus-access              who is allowed, and today's violations
attendance-bus-deny who:
    python3 products/attendance/bus_access.py deny "{{who}}"

attendance-bus-allow who:
    python3 products/attendance/bus_access.py allow "{{who}}"

attendance-bus-access:
    python3 products/attendance/bus_access.py status

# Redis has no persistent volume by design (see PIPELINE.md) -- its face
# gallery is wiped on every `down`. Re-registering the 8 already-verified
# people (same photos, same face-ids, confirmed pairwise-distinct) on every
# `up` is safe and idempotent -- it only ever recreates exactly what
# attendance.db's Employee rows already expect.
#
# Display names (2026-08-31): the CHIRLA dataset's subjects are anonymized --
# no real names exist for them -- so these are simulation-only placeholders,
# picked to demo the tool as a real attendance product would look, not to
# represent real people. The name has to agree in TWO places for a given
# face-id: the sidecar's own gallery (set here, at registration -- this is
# what the live overlay draws and what iris:face_rec publishes) and
# products/attendance's own `employee.name` column (updated right after, same
# id order) -- products/attendance's name-based fallback match (storage.
# get_employee_by_name) depends on these agreeing exactly.
_register_people:
    #!/usr/bin/env bash
    set -euo pipefail
    # DB-driven enrollment (2026-09-02, /grilling round 1 Q8 + /mentor
    # design): the employee table + saved photos are now the source of
    # truth -- the admin API can add people at any time, this recipe no
    # longer owns a hardcoded roster. Two steps:
    #   1. seed the original 13 demo people ONCE, only if the table is
    #      empty (a fresh checkout/dev machine) -- plain host python3
    #      (sqlite3 stdlib only, no GPU needed for a row insert);
    #   2. re-register EVERYONE currently in the table (seeded or
    #      admin-added) into the redis gallery, which `down` wipes every
    #      time but the sqlite db and saved photos survive -- via the same
    #      FaceGallery calls the admin upload endpoint uses, minus the
    #      distinctness check (these are already-vetted people, replay not
    #      validation; see reregister_all.py's own docstring). Needs the
    #      GPU container, same as always. The db is mounted rw, not ro --
    #      it's in WAL mode, which needs write access to its -wal/-shm
    #      files even for a read-only query (found live 2026-09-02, a
    #      real bug, not just caution: the ro mount failed outright).
    python3 products/attendance/seed_demo_employees.py --db products/attendance/data/attendance.db
    # Same project-scoped image docker-compose.yml builds for this checkout
    # (${COMPOSE_PROJECT_NAME}-deepstream-sidecar), not another checkout's
    # shared tag -- so .env's COMPOSE_PROJECT_NAME is needed here.
    set -a
    [ -f .env ] && source .env
    set +a
    # The image compose itself uses for the sidecar. Naming it here from COMPOSE_PROJECT_NAME was wrong whenever
    # .env did not set it: compose names images from its `name:` (iris), so this picked an OLDER image
    # (surveillance-camera-sys-deepstream-sidecar, the one with torch's cuDNN) and face registration on the GPU
    # failed with "Unable to load libcudnn_engines_runtime_compiled". tests/test_justfile_images.py pins it.
    sidecar_image="$(docker compose config --images 2>/dev/null | grep -m1 -E 'deepstream-sidecar:latest$' || true)"
    sidecar_image="${sidecar_image:-${COMPOSE_PROJECT_NAME:-iris}-deepstream-sidecar:latest}"
    # Every opening brace pair below is doubled to escape it from just's own
    # interpolation syntax -- this is a literal Go template string meant for
    # `docker inspect --format`, not a just expression. Without the doubling,
    # just fails to even PARSE the justfile (every recipe breaks, not just this
    # one) with "error: unknown start of token '.'" at NetworkSettings (found
    # live 2026-09-08, right after this line was added to resolve the renamed
    # `km` compose project's network dynamically instead of hardcoding "km_default").
    net="$(docker inspect --format '{{{{range $k, $v := .NetworkSettings.Networks}}{{{{$k}}{{{{end}}' "$(docker compose ps -q redis)")"
    docker run --rm --device nvidia.com/gpu=all --network "$net" \
        -v "$(pwd)/data/face_photos:/photos:ro" \
        -v "$(pwd)/products/attendance/data:/db:rw" \
        -v "$(pwd)/perception:/workspace/iris/perception:ro" \
        -v "$(pwd)/products/attendance/reregister_all.py:/workspace/iris/products/attendance/reregister_all.py:ro" \
        -w /workspace/iris --entrypoint python3 \
        "$sidecar_image" \
        products/attendance/reregister_all.py --db /db/attendance.db --photos-dir /photos \
            --redis-url "redis://redis:6379/0" --gallery-key "attendance:face:gallery" --gpu

_start_host_processes cams=cams:
    #!/usr/bin/env bash
    set -euo pipefail
    # Multi-developer isolation (2026-09-10): pick up GO2RTC_RTSP_PORT/
    # REDIS_HOST_PORT from .env (docker-compose.yml already reads these for
    # the container-side port mappings; these plain host processes need them
    # separately since they never go through docker compose's own .env
    # loading). See .env.example.
    set -a
    [ -f .env ] && source .env
    set +a
    mkdir -p {{run_dir}}
    # Each process is started as `setsid bash -c '... echo $$ > pidfile;
    # exec nohup real-command'` rather than the simpler `setsid nohup
    # real-command & echo $! > pidfile`. Reason: setsid(1) only *execs* the
    # given command in place (keeping the same PID) when the calling process
    # isn't already a process-group leader; if it is, it silently forks a
    # child first, so the PID bash's `$!` captures ends up being setsid's own
    # (wrong) PID, not the real process's -- and not its process group either,
    # which `down` relies on for its group kill (needed since e.g. `npm run
    # dev` forks a `vite` child that a plain `kill $pid` would orphan). Found
    # live 2026-08-30: streamer.pid held a PID whose group-kill was a silent
    # no-op. Writing `$$` from *inside* the script, after setsid has already
    # taken effect and right before it execs onward, is unconditionally
    # correct -- no guessing which path setsid took.
    # `just up live` (2026-09-13): real cameras only -- no test streamer at
    # all, so no file is ever pushed into a live camera's go2rtc path.
    if [ "{{cams}}" = "live" ]; then
        echo "live cameras only -- test streamer not started"
        rm -f {{run_dir}}/streamer.pid
        # No test recordings this run: drop the last run's start times so no
        # tile shows a clock for a test camera that isn't streaming.
        rm -f data/stream_starts.json
    else
        echo "starting CHIRLA camera streaming ({{cams}})..."
        cd test_streamer
        setsid bash -c 'echo $$ > ../{{run_dir}}/streamer.pid; exec nohup python3 stream_recordings.py --ids {{cams}} --port '"${GO2RTC_RTSP_PORT:-8554}" \
            > ../{{run_dir}}/streamer.log 2>&1 &
        cd ..
    fi
    # Sidecar watchdog (2026-09-13): restarts the sidecar when a camera has sent
    # no frames for 60s while go2rtc is still receiving its video (the stuck
    # reconnect loop). Runs from the repo root; see tools/sidecar_watchdog.py.
    echo "starting sidecar watchdog..."
    setsid bash -c 'echo $$ > {{run_dir}}/watchdog.pid; exec nohup products/attendance/.venv/bin/python tools/sidecar_watchdog.py' \
        > {{run_dir}}/watchdog.log 2>&1 &
    # {{justfile_directory()}}, not `../`: this product lives two levels down now.
    cd {{justfile_directory()}}/products/attendance
    echo "starting products/attendance..."
    export REDIS_URL="redis://localhost:${REDIS_HOST_PORT:-6379}/0"
    # ATTENDANCE_BACKEND_PORT/ATTENDANCE_FRONTEND_PORT (see .env.example) --
    # an earlier pass at this used ATTENDANCE_PORT/FRONTEND_PORT instead,
    # which silently didn't read a checkout's real values (found live
    # 2026-09-10: the dashboard came up on vite's auto-incremented fallback
    # port instead of the intended one).
    setsid bash -c 'echo $$ > {{justfile_directory()}}/{{run_dir}}/attendance.pid; exec nohup .venv/bin/uvicorn server:app --host 0.0.0.0 --port '"${ATTENDANCE_BACKEND_PORT:-8001}" \
        > {{justfile_directory()}}/{{run_dir}}/attendance.log 2>&1 &
    just _start_frontend
    # Give each pidfile a moment to actually be written before returning --
    # `up`'s next step (waiting on the sidecar, then _register_people) takes
    # long enough in practice that this has never been observed racing, but
    # `down` running immediately after `up` returns depends on these files existing.
    sleep 0.2

# Start the dashboard (vite dev server) as a detached host process; `just up` and `barber-up` both use it.
_start_frontend:
    #!/usr/bin/env bash
    set -euo pipefail
    set -a
    [ -f .env ] && source .env
    set +a
    mkdir -p {{run_dir}}
    cd {{justfile_directory()}}/{{frontend}}
    # --host 0.0.0.0: the user browses from a different machine than this
    # server (found live 2026-08-31: vite's default localhost-only bind made
    # the dashboard unreachable remotely -- "loads forever"). No --port
    # override -- each branch's own vite.config.ts already sets the right
    # one (5173 for feat/attendance-page, 5180 for feat/fire-smoke-tab), now
    # itself env-overridable via the exported ATTENDANCE_BACKEND_PORT/
    # ATTENDANCE_FRONTEND_PORT/IRIS_HOST_PORT below (see .env.example).
    echo "starting frontend ($(git rev-parse --abbrev-ref HEAD))..."
    export ATTENDANCE_BACKEND_PORT="${ATTENDANCE_BACKEND_PORT:-8001}"
    export CONCEALMENT_BACKEND_PORT="${CONCEALMENT_BACKEND_PORT:-8005}"
    export IRIS_HOST_PORT="${IRIS_HOST_PORT:-18000}"
    export ATTENDANCE_FRONTEND_PORT="${ATTENDANCE_FRONTEND_PORT:-}"
    setsid bash -c 'echo $$ > {{justfile_directory()}}/{{run_dir}}/frontend.pid; exec nohup npm run dev -- --host 0.0.0.0' \
        > {{justfile_directory()}}/{{run_dir}}/frontend.log 2>&1 &

# Full rebuild: docker images + Python venv + node_modules, then start everything.
up-build cams=cams: _preflight _use-default-config
    #!/usr/bin/env bash
    set -euo pipefail
    # iris's Dockerfile builds the dashboards from {{frontend}}/, which is plain
    # source in this repo now (2026-09-23) -- nothing to initialise first.
    # iris used to be deliberately excluded from --build (rebuilding its image
    # on every up-build was unnecessary when only deepstream-sidecar's code was
    # actively changing -- decided 2026-08-30, see PIPELINE.md). Reversed
    # 2026-09-04: unlike deepstream-sidecar, iris does NOT bind-mount the repo
    # -- its Python source is COPYed into the image at build time -- so once
    # app/*.py started changing too (fire/smoke alert handling, incident
    # tracking), a stale `iris` image silently kept running old code with no
    # error anywhere (found live: fire/smoke alerts misclassified as plain
    # overcrowding for a full session before this was caught). `up-build`'s own
    # docstring already promised "full rebuild" -- this makes that true.
    #
    # Only BUILD the images here -- do not start the containers yet (`docker
    # compose up -d` would). `just up` (called at the end of this recipe)
    # starts deepstream-sidecar itself at the point in its own sequence where
    # go2rtc and the CHIRLA streamer are already live -- see the comment
    # there for why that ordering matters (found live 2026-08-31: starting
    # deepstream-sidecar this early made every camera's first, and for most
    # cameras only, connection attempt land before anything was streaming).
    docker compose build deepstream-sidecar iris
    docker compose up -d redis iris go2rtc
    cd {{justfile_directory()}}/products/attendance && uv sync --locked
    cd {{justfile_directory()}}/{{frontend}} && npm ci
    cd {{justfile_directory()}}
    just up {{cams}}

# Full rebuild for the purpose-split topology. Mirrors `up-build` except it
# builds/starts deepstream-sidecar-fire + deepstream-sidecar-attendance (the
# `roles` compose profile) instead of the single deepstream-sidecar service,
# and falls through to `up-roles` instead of `up`.
up-build-roles cams=cams: _preflight _use-default-config
    #!/usr/bin/env bash
    set -euo pipefail
    docker compose build iris
    docker compose --profile roles build deepstream-sidecar-fire deepstream-sidecar-attendance
    docker compose up -d redis iris go2rtc
    cd {{justfile_directory()}}/products/attendance && uv sync --locked
    cd {{justfile_directory()}}/{{frontend}} && npm ci
    cd {{justfile_directory()}}
    just up-roles {{cams}}

# Start everything without rebuilding (the common case).
up cams=cams: _preflight _use-default-config
    #!/usr/bin/env bash
    set -euo pipefail
    docker compose up -d redis iris go2rtc
    # The CHIRLA streamer (ffmpeg, a host process) must be actually publishing
    # to go2rtc BEFORE deepstream-sidecar starts, not after. Its pipeline
    # tries to connect to every configured camera once, immediately, as it
    # comes up -- and for most cameras (found live 2026-08-31: 10 of 14,
    # including cam-9), a failed first attempt is never retried, so starting
    # it against a still-silent go2rtc permanently loses that camera until
    # the next restart. Starting the streamer first and giving it a few
    # seconds' head start closes that race.
    just _start_host_processes {{cams}}
    if [ "{{cams}}" != "live" ]; then
    echo "giving the CHIRLA streamer a moment to start publishing to go2rtc..."
    sleep 5
    # The streamer exits immediately on a bad cam id (ids are cam-1..cam-14,
    # no zero padding -- `cam-01` is rejected) or a missing upload file, and
    # setsid detaches it so that error lands only in its log. Fail loudly here
    # instead of letting `up` continue into a silently camera-less stack
    # (found live 2026-08-31: `just up cam-01` looked like it worked but
    # nothing ever streamed).
    if ! ps -p "$(cat {{run_dir}}/streamer.pid 2>/dev/null || echo 0)" > /dev/null 2>&1; then
        echo "ERROR: the CHIRLA streamer is not running -- {{run_dir}}/streamer.log says:" >&2
        tail -n 5 {{run_dir}}/streamer.log >&2
        exit 1
    fi
    fi  # cams != live
    docker compose up -d deepstream-sidecar
    echo "waiting for the sidecar pipeline to come up..."
    for _ in $(seq 1 60); do
        if docker logs "$(docker compose ps -q deepstream-sidecar)" 2>&1 | grep -qE "pipeline PLAYING|ERROR"; then
            break
        fi
        sleep 2
    done
    just _register_people
    echo
    echo -n "up ({{cams}}). "
    just _dashboard_url

# Start everything without rebuilding, using the purpose-split sidecar
# topology (deepstream-sidecar-fire + deepstream-sidecar-attendance) instead
# of the single shared one. Mirrors `up` exactly except for the sidecar
# service(s) started and waited on. Mutually exclusive with `just up` --
# don't run both against the same camera set.
up-roles cams=cams: _preflight _use-default-config
    #!/usr/bin/env bash
    set -euo pipefail
    docker compose up -d redis iris go2rtc
    just _start_host_processes {{cams}}
    if [ "{{cams}}" != "live" ]; then
    echo "giving the CHIRLA streamer a moment to start publishing to go2rtc..."
    sleep 5
    if ! ps -p "$(cat {{run_dir}}/streamer.pid 2>/dev/null || echo 0)" > /dev/null 2>&1; then
        echo "ERROR: the CHIRLA streamer is not running -- {{run_dir}}/streamer.log says:" >&2
        tail -n 5 {{run_dir}}/streamer.log >&2
        exit 1
    fi
    fi  # cams != live
    docker compose --profile roles up -d deepstream-sidecar-fire deepstream-sidecar-attendance
    echo "waiting for both role pipelines to come up..."
    for svc in deepstream-sidecar-fire deepstream-sidecar-attendance; do
        for _ in $(seq 1 60); do
            if docker logs "$(docker compose ps -q "$svc")" 2>&1 | grep -qE "pipeline PLAYING|ERROR"; then
                break
            fi
            sleep 2
        done
    done
    just _register_people
    echo
    echo -n "up-roles ({{cams}}). "
    just _dashboard_url

# 2026-09-10: reads vite's own confirmed bind (the "Local:
# http://localhost:PORT/" line it prints once ready) instead of guessing.
# vite silently auto-increments past an occupied port rather than failing, so
# a guessed URL can be flat wrong -- found live 2026-09-10: `up-roles` printed
# 5180 while another checkout's stack already squatted it and vite actually
# landed on 5183. Falls back to the per-branch literal default (still
# ATTENDANCE_FRONTEND_PORT-overridable) if the log line never shows up within
# the timeout, e.g. the frontend failed to start at all.
_dashboard_url product="":
    #!/usr/bin/env bash
    set -uo pipefail
    # Which product the URL opens. The dashboard shows ?product=... first, then the product the BROWSER last used,
    # then the first one -- so a bare http://localhost:5180 shows whatever was open last (barber run, attendance
    # screen: found live 2026-10-28, not the first time). Say which: the argument, else the first of IRIS_PRODUCT,
    # else the running preset's product (the multi-camera and VLM presets are the "iris" product), else attendance.
    product="{{product}}"
    if [ -z "$product" ]; then
        names="${IRIS_PRODUCT:-}"
        [ -z "$names" ] && [ -f .env ] && names=$(grep -E '^IRIS_PRODUCT=' .env | tail -1 | cut -d= -f2- | tr -d "\"' ")
        product="${names%%,*}"
    fi
    if [ -z "$product" ]; then
        if [ -f .active_profile ]; then product=iris; else product=attendance; fi
    fi
    # One app, one port; the product is picked inside it.
    default_port="${ATTENDANCE_FRONTEND_PORT:-5180}"
    path="/?product=${product}#/${product}"
    port=""
    for _ in $(seq 1 15); do
        port=$(grep -oP 'Local:\s+http://localhost:\K[0-9]+' {{run_dir}}/frontend.log 2>/dev/null | tail -1)
        [ -n "$port" ] && break
        sleep 1
    done
    if [ -z "$port" ]; then
        echo "Dashboard: http://localhost:${default_port}${path}  (vite's readiness line never showed in {{run_dir}}/frontend.log within 15s -- this is a GUESS, check that log before trusting it)"
    else
        echo "Dashboard: http://localhost:${port}${path}"
    fi

# Stop everything (docker compose down -- go2rtc is a compose service now,
# same as the rest of the stack, so no separate container to manage).
down:
    #!/usr/bin/env bash
    set -uo pipefail
    # Multi-developer isolation: see _start_host_processes for why this is needed.
    set -a
    [ -f .env ] && source .env
    set +a
    # The truck-tracking web UI (if present): stop it and clear its tracked trucks, so the
    # next run starts with an empty trucks page. No-op without that checkout.
    bash scripts/truck_ui.sh down
    for name in watchdog streamer attendance frontend barber barber-ui barber-streamer fall concealment; do
        pidfile="{{run_dir}}/$name.pid"
        if [ -f "$pidfile" ]; then
            pid=$(cat "$pidfile")
            kill -TERM "-$pid" 2>/dev/null && echo "stopped $name (pgid $pid)" || echo "$name already stopped"
            rm -f "$pidfile"
        fi
    done
    # A preset run's test streamer (scripts/start_streamer.sh). Its own
    # process, not a group leader -- plain kill; its ffmpeg children are
    # caught by the raw-cam- publisher sweep below.
    if [ -f .streamer.pid ]; then
        pid=$(cat .streamer.pid)
        kill -TERM "$pid" 2>/dev/null && echo "stopped preset streamer (pid $pid)" || echo "preset streamer already stopped"
        rm -f .streamer.pid
    fi
    # Orphan sweep: kill any streamer / RTSP publisher / attendance server /
    # frontend the pidfiles do NOT track (e.g. a stream_recordings.py started
    # manually in another terminal, or a process whose pidfile went stale
    # after a crash). A leftover ffmpeg keeps its go2rtc raw-ingest slot busy,
    # so the next `up`'s publisher gets rejected and that camera never comes
    # up; a leftover uvicorn squats port 8001 the same way (found live
    # 2026-08-31: "sometimes i can't run another video even after just
    # down"); a leftover vite squats 5173/5180 the same way again (found live
    # 2026-09-10: two leftover `vite` processes from an earlier manual run
    # sat on 5180/5181 for hours, silently forcing every later `just up`'s
    # frontend onto an auto-incremented port nothing then reported correctly).
    # Scoping matters on this shared host: an UNRELATED deployment
    # (/data/IRIS/repo or a sibling checkout) runs identically-named
    # processes (found live 2026-08-31, then again 2026-09-10 for the
    # frontend), so cwd-based sweeps below match on this checkout's actual
    # resolved path, not a bare substring like "surveillance-camera-sys" --
    # that string is present in EVERY checkout of this repo on this host, so
    # matching on it alone (the streamer sweep's original check) could kill a
    # sibling checkout's process instead of (or as well as) this one's. The
    # ffmpeg sweep matches on the raw-cam- path segment unique to our go2rtc
    # ingest scheme -- not just the bare port, which (unlike the old mediamtx
    # setup's arbitrary 18554) is go2rtc's common default (8554) and so isn't
    # a safe-enough signature on its own.
    repo_root="$(pwd -P)"
    for pid in $(pgrep -f "stream_recordings.py" 2>/dev/null); do
        if readlink -f "/proc/$pid/cwd" 2>/dev/null | grep -qF "$repo_root/"; then
            kill -TERM "$pid" 2>/dev/null && echo "killed stray streamer (pid $pid)" || true
        fi
    done
    pkill -f "rtsp://localhost:${GO2RTC_RTSP_PORT:-8554}/raw-cam-" 2>/dev/null && echo "killed stray RTSP publisher(s)" || true
    pkill -f "uvicorn server:app --host 0.0.0.0 --port ${ATTENDANCE_BACKEND_PORT:-8001}" 2>/dev/null && echo "killed stray attendance server(s)" || true
    for pid in $(pgrep -f "\.venv/bin/python main\.py" 2>/dev/null); do
        if readlink -f "/proc/$pid/cwd" 2>/dev/null | grep -qE "$repo_root/products/(fall_detection|concealment_detection)"; then
            kill -TERM "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
            echo "killed stray product service (pid $pid)"
        fi
    done
    for pid in $(pgrep -f "node .*node_modules/.bin/vite" 2>/dev/null); do
        if readlink -f "/proc/$pid/cwd" 2>/dev/null | grep -qF "$repo_root/"; then
            kill -TERM "$pid" 2>/dev/null && echo "killed stray frontend (pid $pid)" || true
        fi
    done
    # --profile roles also tears down deepstream-sidecar-fire/-attendance if
    # they were brought up via `up-roles`/`up-build-roles` -- a plain `docker
    # compose down` only removes profile-less services and silently leaves
    # them running (found live 2026-09-04: the network removal failed with
    # "Resource is still in use" because both role containers were still up).
    # --profile barber: the same reason for deepstream-sidecar-barber.
    # --profile fall: the same reason for deepstream-sidecar-fall.
    # --profile concealment: the same reason for deepstream-sidecar-concealment.
    docker compose --profile roles --profile barber --profile multicam --profile fall --profile concealment down
    echo "down."

# ── Barber shop monitoring (its own deployment; see products/barber/README.md) ──
# redis + minio + go2rtc + the barber sidecar replica in docker, products/barber
# as a host process (.run/barber.pid). [clips]: optional comma-separated
# test-clip camera ids for test_streamer (each must be in its CAM_FILES),
# e.g. `just barber-up barber-1`.
# The barber re-ID models, both NVIDIA TAO ReIdentificationNet from NGC and
# gitignored like every .onnx here:
#  * the tracker's ResNet50 (products/barber/tracker.yml),
#    downloaded only when missing or corrupt;
#  * the Swin-T 256-d that perception/modules/appearance/worker.py embeds with in production
#    (sidecar.yaml appearance_reid_spec: tao_swin_t; CLIP-ReID, an offline
#    candidate, is exported by products/barber/dev/appearance/export_clipreid.py).
#    Its first download wrote perception/pipeline/models.reid.sha256; later runs verify it.
barber-reid-model:
    #!/usr/bin/env bash
    set -euo pipefail
    f=models/tracker/resnet50_market1501_aicity156.onnx
    sha=0e21d09278508ec835955f422a9fdd3cd59b2a6ecdef98d705f388f33cebac2b
    mkdir -p models/tracker
    if ! echo "$sha  $f" | sha256sum -c --status 2>/dev/null; then
        echo "downloading the tracker's ReID model (96 MB, NGC)..."
        curl -fL --retry 3 -o "$f.part" \
            "https://api.ngc.nvidia.com/v2/models/nvidia/tao/reidentificationnet/versions/deployable_v1.2/files/resnet50_market1501_aicity156.onnx"
        echo "$sha  $f.part" | sha256sum -c --status || { rm -f "$f.part"; echo "ReID model checksum mismatch" >&2; exit 1; }
        mv "$f.part" "$f"
    fi
    base=https://api.ngc.nvidia.com/v2/models/nvidia/tao/reidentificationnet_transformer/versions/deployable_v1.0/files
    mkdir -p models/reid
    f=swin_tiny_market1501_aicity156_featuredim256.onnx
    out=models/reid/$f
    [ -f "$out" ] || { echo "downloading $f (114 MB, NGC)..."; curl -fL --retry 3 -o "$out.part" "$base/$f"; mv "$out.part" "$out"; }
    if [ -f perception/pipeline/models.reid.sha256 ]; then
        sha256sum -c --quiet perception/pipeline/models.reid.sha256
    else
        sha256sum "$out" > perception/pipeline/models.reid.sha256
        echo "wrote perception/pipeline/models.reid.sha256 -- commit it"
    fi

# The barber staff classifier's zero-shot stage (perception/modules/appearance/worker.py): CLIP
# ViT-B/32 ONNX (Hugging Face Xenova/clip-vit-base-patch32). The image tower
# runs in the sidecar; the text tower and tokenizer are used once, offline, to
# build perception/models/staff/prompts.json. First download writes
# perception/pipeline/models.staff.sha256 (commit it); later runs verify against it.
barber-staff-model:
    #!/usr/bin/env bash
    set -euo pipefail
    base=https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main
    mkdir -p perception/models/staff
    for f in onnx/vision_model.onnx onnx/text_model.onnx tokenizer.json; do
        out=perception/models/staff/$(basename $f)
        [ -f "$out" ] || { echo "downloading $f..."; curl -fL --retry 3 -o "$out.part" "$base/$f"; mv "$out.part" "$out"; }
    done
    if [ -f perception/pipeline/models.staff.sha256 ]; then
        sha256sum -c --quiet perception/pipeline/models.staff.sha256
    else
        sha256sum perception/models/staff/vision_model.onnx perception/models/staff/text_model.onnx perception/models/staff/tokenizer.json > perception/pipeline/models.staff.sha256
        echo "wrote perception/pipeline/models.staff.sha256 -- commit it"
    fi

# The barber sidecar's detector (project 3 chair engine): YOLO26l-seg at a
# rectangular 640x384 input (H 384, W 640), dynamic batch, as the DeepStream
# ONNX perception/models/DeepStream-Yolo-Seg/config_infer_primary_yolo26l_seg_r384.txt loads.
# Weights from the ultralytics assets release (sha-pinned) into models/seg/;
# exported with perception/models/DeepStream-Yolo-Seg/export_yolo26_seg.py (CPU) in a temp dir
# (the exporter names the ONNX after the weights and writes labels.txt to its
# cwd). Skipped when the ONNX exists; never overwrites it. The TensorRT engine
# is built by nvinfer on the sidecar's first start (minutes; the compose
# service's LD_PRELOAD of libgcc_s keeps GStreamer's libunwind from crashing
# that build). Export Python: $SEG_EXPORT_PY when set (needs torch,
# ultralytics, onnx, onnxslim), else the locked env of tools/onnx-export (DEPENDENCIES.md).
barber-seg-model out="perception/models/DeepStream-Yolo-Seg/yolo26l-seg-r384.onnx":
    #!/usr/bin/env bash
    set -euo pipefail
    out="{{out}}"
    if [ -f "$out" ]; then exit 0; fi
    w=models/seg/yolo26l-seg.pt
    sha=636024306410afa1732692322fba57d22ea2b1c2f07613fcee131a93d7dd380c
    mkdir -p models/seg
    if ! echo "$sha  $w" | sha256sum -c --status 2>/dev/null; then
        echo "downloading yolo26l-seg.pt (64 MB, ultralytics assets)..."
        curl -fL --retry 3 -o "$w.part" https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l-seg.pt
        echo "$sha  $w.part" | sha256sum -c --status || { rm -f "$w.part"; echo "yolo26l-seg.pt checksum mismatch" >&2; exit 1; }
        mv "$w.part" "$w"
    fi
    if [ -n "${SEG_EXPORT_PY:-}" ]; then py=("$SEG_EXPORT_PY"); else
        py=(env -u VIRTUAL_ENV uv run --locked --project "{{justfile_directory()}}/tools/onnx-export" python); fi
    tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
    cp "$w" "$tmp/yolo26l-seg-r384.pt"
    script="$PWD/perception/models/DeepStream-Yolo-Seg/export_yolo26_seg.py"
    echo "exporting yolo26l-seg-r384.onnx (640x384, dynamic batch)..."
    (cd "$tmp" && "${py[@]}" "$script" -w yolo26l-seg-r384.pt -s 384 640 --dynamic --simplify)
    mkdir -p "$(dirname "$out")"
    mv "$tmp/yolo26l-seg-r384.onnx" "$out"
    echo "wrote $out"

# The proximity review stack: the proximity pipeline on recorded clips, the app and the dashboard, isolated from any
# compose stack on this host (own containers, Redis and ports); see tools/proximity_review.sh.
# Review stack: up [id=/clip.mp4,...] | down | restart [app|ui|all] | status | logs [pipeline|app|ui] | reset
proximity-review action="status" *args="":
    tools/proximity_review.sh {{action}} {{args}}

# Proximity models (products/proximity): YOLO26l-seg (the detector) and YOLO26m-depth (the depth check), both
# ultralytics v8.4.0 release assets, exported once to ONNX at 1280x736; the sidecar builds the TensorRT engine itself.
# A weight already in models/proximity/ with the right checksum is used as it is.
proximity-models:
    #!/usr/bin/env bash
    set -euo pipefail
    seg=perception/models/DeepStream-Yolo-Seg/yolo26l-seg-736x1280.onnx
    depth=perception/models/depth/yolo26m-depth-736x1280.onnx
    if [ -n "${SEG_EXPORT_PY:-}" ]; then py=("$SEG_EXPORT_PY"); else
        py=(uv run --no-project --python 3.12 --with ultralytics==8.4.172 --with onnx --with onnxslim python); fi
    fetch() {   # fetch <asset> <sha256> -> models/proximity/<asset>
        local w="models/proximity/$1"
        mkdir -p models/proximity
        if ! echo "$2  $w" | sha256sum -c --status 2>/dev/null; then
            echo "downloading $1 (ultralytics assets)..."
            curl -fL --retry 3 -o "$w.part" "https://github.com/ultralytics/assets/releases/download/v8.4.0/$1"
            echo "$2  $w.part" | sha256sum -c --status || { rm -f "$w.part"; echo "$1 checksum mismatch" >&2; exit 1; }
            mv "$w.part" "$w"
        fi
    }
    tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
    if [ ! -f "$seg" ]; then
        fetch yolo26l-seg.pt 636024306410afa1732692322fba57d22ea2b1c2f07613fcee131a93d7dd380c
        cp models/proximity/yolo26l-seg.pt "$tmp/yolo26l-seg-736x1280.pt"
        script="$PWD/perception/models/DeepStream-Yolo-Seg/export_yolo26_seg.py"
        echo "exporting $seg (1280x736, dynamic batch)..."
        (cd "$tmp" && "${py[@]}" "$script" -w yolo26l-seg-736x1280.pt -s 736 1280 --dynamic --simplify)
        mv "$tmp/yolo26l-seg-736x1280.onnx" "$seg"
    fi
    if [ ! -f "$depth" ]; then
        fetch yolo26m-depth.pt c0ff310571f0f159436e95a2a4fa3f8c40e1fbf36fac66a211347bd189990165
        cp models/proximity/yolo26m-depth.pt "$tmp/"
        echo "exporting $depth (1280x736)..."
        (cd "$tmp" && "${py[@]}" -c "from ultralytics import YOLO; YOLO('yolo26m-depth.pt').export(format='onnx', imgsz=[736, 1280], simplify=True, device='cpu')")
        mkdir -p "$(dirname "$depth")"
        mv "$tmp/yolo26m-depth.onnx" "$depth"
    fi
    ls -la "$seg" "$depth"

barber-up clips="":
    #!/usr/bin/env bash
    set -euo pipefail
    set -a
    [ -f .env ] && source .env
    set +a
    # a failing $(...) inside eval doesn't trip set -e (final review I4): check site.sh on its own first
    out="$(products/barber/site.sh '{{barber_site}}' bekky)" || { echo "barber-up: nothing started" >&2; exit 2; }
    eval "$out"
    clips="{{clips}}"; [ -z "$clips" ] && clips="$BARBER_SITE_CLIPS"
    mkdir -p {{run_dir}} && echo "$BARBER_SITE" > {{run_dir}}/barber-site
    echo "barber site: $BARBER_SITE ($BARBER_CONFIG)"
    just barber-reid-model
    just barber-staff-model
    just barber-seg-model
    # The seg parser .so is built into the sidecar image (perception/pipeline/Dockerfile); an image from
    # before that stage lacks it, so rebuild once when its label is missing.
    img=$(docker compose --profile barber config --images | grep -m1 -- '-deepstream-sidecar:')
    if [ "$(docker image inspect -f '{{{{index .Config.Labels "iris.seg-parser"}}' "$img" 2>/dev/null)" != "1" ]; then
        echo "building $img (seg parser)..."
        docker compose --profile barber build deepstream-sidecar-barber
    fi
    docker compose --profile barber up -d redis go2rtc
    # the site's camera feeds before the sidecar connects (products/barber/feeds.py): a live site points go2rtc at
    # its recorder; a clip site needs go2rtc's empty placeholders back if a live run left camera pulls in them
    g2rtc="http://localhost:${GO2RTC_API_PORT:-1984}"
    if [ -n "$BARBER_SITE_FEEDS" ]; then
        python3 products/barber/feeds.py live --api "$g2rtc" --feeds "$BARBER_SITE_FEEDS" --base-env "$BARBER_SITE_FEED_BASE" \
            || { echo "barber-up: go2rtc did not take the camera feeds" >&2; exit 1; }
    elif [ -n "$clips" ] && ! python3 products/barber/feeds.py clips --api "$g2rtc" --ids "$clips"; then
        echo "restarting go2rtc (back to go2rtc.yaml's placeholders)..."; docker compose restart go2rtc
    fi
    docker compose --profile barber up -d deepstream-sidecar-barber
    (cd products/barber && env -u VIRTUAL_ENV uv sync -q --locked)
    if [ -n "$clips" ]; then
        echo "streaming test clips ($clips)..."
        cd test_streamer
        # a site's own recordings for its ids (site.sh BARBER_SITE_CLIP_FILES, e.g. palm-hills-16oct; 2026-10-18)
        setsid bash -c 'echo $$ > ../{{run_dir}}/barber-streamer.pid; exec nohup python3 stream_recordings.py --ids '"$clips"' --port '"${GO2RTC_RTSP_PORT:-8554}"' --files '"'${BARBER_SITE_CLIP_FILES:-}'" \
            > ../{{run_dir}}/barber-streamer.log 2>&1 &
        cd ..
    fi
    # {{justfile_directory()}}, not `../`: this product lives two levels down
    # (products/barber/), so counting `..` back to the repo root breaks the
    # moment a folder moves (2026-09-22, the monorepo restructure).
    # the last run's log is kept, by when it was set aside (2026-10-18: a start truncated it -- two days lost)
    [ -f "{{run_dir}}/barber.log" ] && mv "{{run_dir}}/barber.log" "{{run_dir}}/barber.$(date +%F_%H%M%S).log"
    cd {{justfile_directory()}}/products/barber
    export REDIS_URL="redis://localhost:${REDIS_HOST_PORT:-6379}/0"
    setsid bash -c 'echo $$ > {{justfile_directory()}}/{{run_dir}}/barber.pid; exec nohup .venv/bin/uvicorn server:app --timeout-graceful-shutdown 5 --host 0.0.0.0 --port '"${BARBER_BACKEND_PORT:-8003}" \
        > {{justfile_directory()}}/{{run_dir}}/barber.log 2>&1 &
    cd {{justfile_directory()}}
    # The dashboards are one app (2026-09-23): the same vite server serves every product. `just up` starts it; a
    # barber-only run has no `just up`, so start it here when nothing answers (2026-10-28: barber-up printed
    # "started by just up", nothing was listening, and the page that did open showed attendance).
    if ! curl -sf -o /dev/null "http://localhost:${ATTENDANCE_FRONTEND_PORT:-5180}/"; then
        just _start_frontend
    fi
    just _dashboard_url barber
    echo "waiting for the barber sidecar to reach PLAYING..."
    # `|| true`: grep -q stops reading at the first match, and under pipefail the cut-off `logs` failed the check, so on a
    # sidecar with hours of log this never ended (2026-10-18, after a reboot)
    until { docker compose --profile barber logs deepstream-sidecar-barber 2>&1 || true; } | grep -qE "pipeline PLAYING|ERROR"; do sleep 3; done
    echo "products/barber: http://localhost:${BARBER_BACKEND_PORT:-8003}/docs (log: {{run_dir}}/barber.log)"

# Resync the barber test streamer's synced camera group (barber-c/l/r)
# on demand: sends SIGUSR1 so stream_recordings.py restarts all three
# together from one shared start timestamp, instead of restarting them
# one at a time (which desyncs their clip clock -- see GROUPED_STREAMS in
# test_streamer/stream_recordings.py). The controller runs this after
# deploying, and any time the group's clip clocks drift.
barber-streamer-restart:
    #!/usr/bin/env bash
    set -euo pipefail
    pidfile="{{run_dir}}/barber-streamer.pid"
    [ -f "$pidfile" ] || { echo "barber-streamer isn't running ($pidfile missing)" >&2; exit 1; }
    pid="$(cat "$pidfile")"
    kill -USR1 "$pid"
    echo "sent resync signal to barber-streamer (pid $pid) -- check {{run_dir}}/barber-streamer.log"

# Restart products/barber alone (2026-10-11). soft (default): the data stays (on test clips the
# service still starts clean by itself when the clips are on a new run). hard: a clean slate --
# every customer row (visits, haircuts, waits, alerts, customer cards, door counts, the visit
# baseline) is cleared and tickets restart at #1; staff, roster, zones and settings stay. With the
# test streamer running, hard also restarts the clips from their beginning. The barber sidecar
# (GPU) is never touched. Waits for the old process to exit, so it never runs two copies.
# Restart products/barber only: soft keeps the data, hard clears it (and restarts test clips).
barber-restart mode="soft":
    #!/usr/bin/env bash
    set -euo pipefail
    set -a
    [ -f .env ] && source .env
    set +a
    # a mistyped site must not fall back to Bekky's config (with hard: clear Bekky's day) -- final review I4
    out="$(products/barber/site.sh '{{barber_site}}')" || { echo "barber-restart: nothing restarted" >&2; exit 2; }
    eval "$out"
    running="$(cat {{run_dir}}/barber-site 2>/dev/null || echo bekky)"
    if [ "$BARBER_SITE" != "$running" ]; then
        echo "barber-restart: the running site is $running, not $BARBER_SITE (its sidecar and streams stay) --" \
             "switch sites with just barber-down + just barber_site=$BARBER_SITE barber-up" >&2
        exit 2
    fi
    case "{{mode}}" in soft|hard) ;; *) echo "usage: just barber-restart [soft|hard]" >&2; exit 2 ;; esac
    echo "barber site: $BARBER_SITE ($BARBER_CONFIG)"
    port="${BARBER_BACKEND_PORT:-8003}"
    pidfile="{{run_dir}}/barber.pid"
    if [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
        pid="$(cat "$pidfile")"
        kill -TERM "-$pid" 2>/dev/null || kill -TERM "$pid"
        for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 0.25; done
        # an open dashboard WebSocket holds uvicorn's graceful shutdown (2026-10-15): its second SIGINT forces it
        if kill -0 "$pid" 2>/dev/null; then
            kill -INT "$pid"; sleep 0.5; kill -INT "$pid" 2>/dev/null || true
            for _ in $(seq 1 40); do kill -0 "$pid" 2>/dev/null || break; sleep 0.25; done
        fi
        if kill -0 "$pid" 2>/dev/null; then echo "barber service (pid $pid) did not stop" >&2; exit 1; fi
        echo "stopped the barber service (pid $pid)"
    fi
    if curl -sf -o /dev/null "http://localhost:$port/api/barber/policy"; then
        echo "port $port is still answered by another barber process -- stop it first" >&2; exit 1
    fi
    clear=0
    if [ "{{mode}}" = hard ]; then
        clear=1
        streamer="{{run_dir}}/barber-streamer.pid"
        if [ -f "$streamer" ] && kill -0 "$(cat "$streamer")" 2>/dev/null; then
            before="$(cat data/stream_starts.json 2>/dev/null || true)"
            kill -USR1 "$(cat "$streamer")"
            for _ in $(seq 1 40); do [ "$(cat data/stream_starts.json 2>/dev/null || true)" != "$before" ] && break; sleep 0.5; done
            if [ "$(cat data/stream_starts.json 2>/dev/null || true)" = "$before" ]; then
                echo "warning: the test clips did not restart (see {{run_dir}}/barber-streamer.log); clearing the data anyway" >&2
            else
                echo "test clips restarted from the beginning"
            fi
        fi
    fi
    prev="{{run_dir}}/barber.$(date +%F_%H%M%S).log"     # every run's log is kept (2026-10-18)
    [ -f "{{run_dir}}/barber.log" ] && mv "{{run_dir}}/barber.log" "$prev"
    cd {{justfile_directory()}}/products/barber
    export BARBER_CONFIG
    export REDIS_URL="redis://localhost:${REDIS_HOST_PORT:-6379}/0" BARBER_CLEAR_ON_START="$clear"
    setsid bash -c 'echo $$ > {{justfile_directory()}}/{{run_dir}}/barber.pid; exec nohup .venv/bin/uvicorn server:app --timeout-graceful-shutdown 5 --host 0.0.0.0 --port '"$port" \
        > {{justfile_directory()}}/{{run_dir}}/barber.log 2>&1 < /dev/null &
    for _ in $(seq 1 60); do curl -sf -o /dev/null "http://localhost:$port/api/barber/policy" && break; sleep 0.5; done
    curl -sf -o /dev/null "http://localhost:$port/api/barber/policy" || { echo "the barber service did not come up -- see {{run_dir}}/barber.log" >&2; exit 1; }
    echo "barber service up ({{mode}} restart): http://localhost:$port/docs (log: {{run_dir}}/barber.log, previous: $prev)"
    grep -E "rows cleared|closed rows left open" {{justfile_directory()}}/{{run_dir}}/barber.log || true

# Stop products/barber, its test streamer and the barber sidecar replica.
# Shared containers (redis/minio/go2rtc) keep running -- `just down` stops those.
barber-down:
    #!/usr/bin/env bash
    set -uo pipefail
    set -a
    [ -f .env ] && source .env
    set +a
    for name in barber barber-ui barber-streamer; do
        pidfile="{{run_dir}}/$name.pid"
        if [ -f "$pidfile" ]; then
            kill -TERM "-$(cat "$pidfile")" 2>/dev/null && echo "stopped $name" || echo "$name already stopped"
            rm -f "$pidfile"
        fi
    done
    pkill -f "rtsp://localhost:${GO2RTC_RTSP_PORT:-8554}/raw-(barber|ph)-" 2>/dev/null && echo "killed stray barber RTSP publisher(s)" || true
    docker compose --profile barber rm -sf deepstream-sidecar-barber
    rm -f {{run_dir}}/barber-site
    echo "barber down."

# The fall-detection stack: products/fall_detection (:8004) and the sidecar that feeds it the
# camera's poses (products/fall_detection/sidecar.yaml, profile `fall`, telemetry 9105).
# The camera-exclusivity preflight runs first: the engine's `roles` tag does not stop a second
# process from reading a camera (perception/engine/sources.py:52-74), so a fall camera that a
# live sidecar already holds refuses the start, and the preflight names it.
# The service starts before the sidecar on purpose: its boot gate (a real CUDA op, both
# checkpoints' sha256) refuses in seconds, where a first sidecar start spends minutes building
# TensorRT engines. The gate's own state is what /status reports.
fall-up:
    #!/usr/bin/env bash
    set -euo pipefail
    set -a
    [ -f .env ] && source .env
    set +a
    port="${FALL_BACKEND_PORT:-8004}"
    (cd products/fall_detection && env -u VIRTUAL_ENV uv sync -q --locked)
    env -u VIRTUAL_ENV products/fall_detection/.venv/bin/python products/fall_detection/preflight.py
    docker compose --profile fall up -d redis go2rtc
    mkdir -p {{run_dir}}
    [ -f "{{run_dir}}/fall.log" ] && mv "{{run_dir}}/fall.log" "{{run_dir}}/fall.$(date +%F_%H%M%S).log"
    cd {{justfile_directory()}}/products/fall_detection
    export REDIS_URL="redis://localhost:${REDIS_HOST_PORT:-6379}/0" FALL_BACKEND_PORT="$port"
    setsid bash -c 'echo $$ > {{justfile_directory()}}/{{run_dir}}/fall.pid; exec nohup .venv/bin/python main.py > {{justfile_directory()}}/{{run_dir}}/fall.log 2>&1 < /dev/null &'
    cd {{justfile_directory()}}
    for _ in $(seq 1 120); do curl -sf -o /dev/null "http://localhost:$port/status" && break; sleep 0.5; done
    curl -sf "http://localhost:$port/status" || { echo "the fall service did not come up -- see {{run_dir}}/fall.log" >&2; exit 1; }
    echo
    docker compose --profile fall up -d deepstream-sidecar-fall
    echo "waiting for the fall sidecar to reach PLAYING (first start builds the engines: minutes)..."
    for _ in $(seq 1 100); do
        if { docker compose --profile fall logs deepstream-sidecar-fall 2>&1 || true; } | grep -qE "pipeline PLAYING|ERROR"; then break; fi
        sleep 3
    done
    if ! { docker compose --profile fall logs deepstream-sidecar-fall 2>&1 || true; } | grep -q "pipeline PLAYING"; then
        echo "the fall sidecar is not PLAYING (see: docker compose --profile fall logs deepstream-sidecar-fall)." >&2
        echo "the service is up and idle; stop it with: just fall-down" >&2
        exit 1
    fi
    echo "products/fall_detection: http://localhost:$port/docs (log: {{run_dir}}/fall.log)"

# Stop products/fall_detection and the fall sidecar replica. Shared containers
# (redis/minio/go2rtc) keep running -- `just down` stops those.
fall-down:
    #!/usr/bin/env bash
    set -uo pipefail
    pidfile="{{run_dir}}/fall.pid"
    if [ -f "$pidfile" ]; then
        kill -TERM "-$(cat "$pidfile")" 2>/dev/null && echo "stopped the fall service" || echo "the fall service was already stopped"
        rm -f "$pidfile"
    fi
    docker compose --profile fall rm -sf deepstream-sidecar-fall
    echo "fall detection down."

# Follow the fall service's log. The sidecar's is the compose one:
# docker compose --profile fall logs -f deepstream-sidecar-fall
fall-log:
    tail -n 200 -f {{run_dir}}/fall.log

# Preflight checks for concealment detection: tools, Python env, config + zones validation,
# DVC models (yolo11x + yolo26x-pose), and camera exclusivity against live sidecar containers.
concealment-check:
    #!/usr/bin/env bash
    set -euo pipefail
    set -a
    [ -f .env ] && source .env
    set +a
    for tool in docker uv curl; do
        command -v "$tool" >/dev/null 2>&1 || { echo "concealment-check: missing required tool '$tool'" >&2; exit 1; }
    done
    (cd products/concealment_detection && env -u VIRTUAL_ENV uv sync -q)
    env -u VIRTUAL_ENV products/concealment_detection/.venv/bin/python - <<'EOF'
    from pathlib import Path
    import json, os, subprocess, sys
    sys.path.insert(0, str(Path("products/concealment_detection").resolve()))
    sys.path.insert(0, str(Path(".").resolve()))
    from config import load_config
    from perception.engine.sources import load_sidecar_config, resolve_sources, select_sources

    cfg = load_config()
    wanted = {c.id for c in cfg.cameras}
    print(f"config: ok (port {cfg.port}, cameras: {', '.join(c.id for c in cfg.cameras)})")

    sidecar_path = os.environ.get("CONCEALMENT_SIDECAR_CONFIG") or "products/concealment_detection/sidecar.yaml"
    scfg = load_sidecar_config(sidecar_path)
    scfg_cams = {s["id"] for s in select_sources(resolve_sources(scfg), 0, 1, role="concealment_detection") if s.get("id")}
    missing_sources = wanted - scfg_cams
    if missing_sources:
        raise SystemExit(f"concealment-check: cameras {sorted(missing_sources)} missing from {sidecar_path} (role concealment_detection)")

    names = subprocess.check_output(["docker", "ps", "-q", "--filter", "name=deepstream-sidecar"], text=True).split()
    for cid in names:
        info = json.loads(subprocess.check_output(["docker", "inspect", cid], text=True))[0]
        name = (info.get("Name") or cid).lstrip("/")
        cmd = info["Config"].get("Cmd") or []
        env = dict(item.split("=", 1) for item in (info["Config"].get("Env") or []) if "=" in item)
        cfg_arg = next((a for a in reversed(cmd) if a.endswith((".yaml", ".yml"))), None)
        if not cfg_arg:
            continue
        if cfg_arg.startswith("/workspace/iris/"):
            cfg_arg = cfg_arg[len("/workspace/iris/"):]
        other = load_sidecar_config(cfg_arg)
        held = {s["id"] for s in select_sources(resolve_sources(other), 0, 1, role=env.get("IRIS_SIDECAR_ROLE") or None) if s.get("id")}
        overlap = sorted(wanted & held)
        if overlap:
            raise SystemExit(f"concealment-check: camera conflict -- {name} already holds {', '.join(overlap)} (stop it first)")
    print("exclusivity: ok (no camera conflicts)")
    EOF
    sidecar_cfg="${CONCEALMENT_SIDECAR_CONFIG:-products/concealment_detection/sidecar.yaml}"
    tools/data/ensure_data.sh "$sidecar_cfg"
    echo "models: ok ($sidecar_cfg)"

# The concealment-detection stack: shared infra + iris (/stream frames), host review service
# (:8005), compose GPU sidecar (profile `concealment`, telemetry 9107), and dashboard URL.
concealment-up: concealment-check
    #!/usr/bin/env bash
    set -euo pipefail
    set -a
    [ -f .env ] && source .env
    set +a
    port="${CONCEALMENT_BACKEND_PORT:-8005}"
    telemetry="${SIDECAR_CONCEALMENT_TELEMETRY_PORT:-9107}"
    docker compose --profile concealment up -d redis go2rtc iris
    mkdir -p {{run_dir}}
    [ -f "{{run_dir}}/concealment.log" ] && mv "{{run_dir}}/concealment.log" "{{run_dir}}/concealment.$(date +%F_%H%M%S).log"
    cd {{justfile_directory()}}/products/concealment_detection
    export REDIS_URL="redis://localhost:${REDIS_HOST_PORT:-6379}/0" CONCEALMENT_BACKEND_PORT="$port"
    setsid bash -c 'echo $$ > {{justfile_directory()}}/{{run_dir}}/concealment.pid; exec nohup .venv/bin/python main.py > {{justfile_directory()}}/{{run_dir}}/concealment.log 2>&1 < /dev/null &'
    cd {{justfile_directory()}}
    for _ in $(seq 1 120); do curl -sf -o /dev/null "http://localhost:$port/status" && break; sleep 0.5; done
    curl -sf "http://localhost:$port/status" || { echo "the concealment service did not come up -- see {{run_dir}}/concealment.log" >&2; exit 1; }
    echo
    docker compose --profile concealment up -d deepstream-sidecar-concealment
    echo "waiting for the concealment sidecar to reach PLAYING (first start builds TensorRT engines: minutes)..."
    for _ in $(seq 1 100); do
        if { docker compose --profile concealment logs deepstream-sidecar-concealment 2>&1 || true; } | grep -qE "pipeline PLAYING|ERROR"; then break; fi
        sleep 3
    done
    if ! { docker compose --profile concealment logs deepstream-sidecar-concealment 2>&1 || true; } | grep -q "pipeline PLAYING"; then
        echo "the concealment sidecar is not PLAYING (see: just concealment-sidecar-log)." >&2
        echo "the service is up and idle; stop it with: just concealment-down" >&2
        exit 1
    fi
    if ! curl -sf -o /dev/null "http://localhost:${ATTENDANCE_FRONTEND_PORT:-5180}/"; then
        just _start_frontend
    fi
    just _dashboard_url concealment_detection
    echo "products/concealment_detection: http://localhost:$port/docs (log: {{run_dir}}/concealment.log, metrics: http://localhost:$telemetry/metrics)"

# Stop products/concealment_detection and the concealment sidecar replica.
# Shared containers (redis/minio/go2rtc/iris) keep running -- `just down` stops those.
concealment-down:
    #!/usr/bin/env bash
    set -uo pipefail
    pidfile="{{run_dir}}/concealment.pid"
    if [ -f "$pidfile" ]; then
        kill -TERM "-$(cat "$pidfile")" 2>/dev/null && echo "stopped the concealment service" || echo "the concealment service was already stopped"
        rm -f "$pidfile"
    fi
    repo_root="$(pwd -P)"
    for pid in $(pgrep -f "\.venv/bin/python main\.py" 2>/dev/null); do
        if readlink -f "/proc/$pid/cwd" 2>/dev/null | grep -qF "$repo_root/products/concealment_detection"; then
            kill -TERM "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
            echo "killed stray concealment service (pid $pid)"
        fi
    done
    docker compose --profile concealment rm -sf deepstream-sidecar-concealment
    echo "concealment detection down."

# Follow the concealment service's log.
concealment-log:
    tail -n 200 -f {{run_dir}}/concealment.log

# Follow the concealment DeepStream sidecar container log.
concealment-sidecar-log:
    docker compose --profile concealment logs -f deepstream-sidecar-concealment

# Which staff classifier Palm Hills runs (2026-10-21): r4 = staff_dinov2_s_palm.onnx (round 4, the 16 Oct day added),
# r3 = staff_dinov2_s_palm_r3.onnx (round 3, the one before). Sets sidecar.palm-hills.yaml's model and
# barber.palm-hills.yaml's staff_threshold together -- the threshold from the model's .json, as each model has its
# own calibration -- then restarts a running Palm site so both change at once (its clips play from the start again).
# The change shows in git: commit it to keep it. e.g. just barber-palm-staff r3
barber-palm-staff version:
    #!/usr/bin/env bash
    set -euo pipefail
    case "{{version}}" in
        r4) model=perception/models/staff/staff_dinov2_s_palm.onnx ;;
        r3) model=perception/models/staff/staff_dinov2_s_palm_r3.onnx ;;
        *) echo "usage: just barber-palm-staff r3|r4" >&2; exit 2 ;;
    esac
    python3 - "$model" <<'EOF'
    import json, re, sys
    from pathlib import Path
    model = Path(sys.argv[1])
    thr = json.loads(model.with_suffix(".json").read_text())["threshold"]
    # the sidecar's model is modules.appearance.staff_classifier; both files are checked before either is written,
    # so a miss never leaves the model and the threshold out of step
    edits = []
    for f, pat, new in (("products/barber/sidecar.palm-hills.yaml", r'^    staff_classifier: ".*"$',
                         f'    staff_classifier: "/workspace/iris/{model}"'),
                        ("products/barber/barber.palm-hills.yaml", r"^  staff_threshold: [0-9.]+", f"  staff_threshold: {thr:.2f}")):
        p = Path(f)
        s, n = re.subn(pat, new, p.read_text(), count=1, flags=re.M)
        if n != 1:
            sys.exit(f"{f}: no line to set ({pat})")
        edits.append((p, s))
    for p, s in edits:
        p.write_text(s)
    print(f"Palm staff classifier: {model.name}, staff_threshold {thr:.2f}")
    EOF
    # a staff_threshold set in Setup (the site's policy.json) wins over the yaml
    for d in products/barber/data/palm-hills products/barber/data/palm-hills-16oct; do
        if [ -f "$d/policy.json" ] && grep -q '"staff_threshold"' "$d/policy.json"; then
            echo "note: $d/policy.json sets staff_threshold (Setup) -- it overrides the model's; clear it in Setup" >&2
        fi
    done
    site="$(cat {{run_dir}}/barber-site 2>/dev/null || true)"
    case "$site" in
        palm-hills*) echo "restarting $site..."; just barber-down; just barber_site="$site" barber-up ;;
        *) echo "no Palm site running: it takes effect at the next barber-up" ;;
    esac

# products/barber's own test suite (pure logic + API; no GPU, no redis).
barber-test *args:
    cd products/barber && env -u VIRTUAL_ENV uv run --locked pytest -q {{args}}

# products/fall_detection's own test suite: the product's policy and service tests plus the
# copied vendor's tests (tests/vendor/), in the product's own venv.
fall-test *args:
    cd products/fall_detection && env -u VIRTUAL_ENV uv run --locked pytest -q {{args}}

# products/concealment_detection's own test suite (pure logic, ACL, storage, API; no GPU/redis needed).
concealment-test *args:
    cd products/concealment_detection && env -u VIRTUAL_ENV uv run pytest -q {{args}}

# Move a site's old flat reports/ into the saved-reports layout (2026-10-25), once. Prints the moves; --apply makes
# them. e.g. just barber-reports-migrate data/palm-hills-16oct --laps --label final=final [--apply]
[positional-arguments]
barber-reports-migrate *args:
    cd products/barber && env -u VIRTUAL_ENV uv run --locked python dev/migrate_reports.py "$@"

# down, plus prune this project's docker images. Deliberately does NOT touch
# attendance.db, face photos, or CHIRLA data -- those represent real
# verification work, never wiped as a side effect.
clean: down
    #!/usr/bin/env bash
    set -uo pipefail
    docker compose down --rmi local
    rm -f {{run_dir}}/*.log
    echo "cleaned. (attendance.db, face_photos/, and CHIRLA data were left untouched)"

# Wipe stale simulation state only -- `sighting`, `desk_interval`, and
# `violation` (restricted-zone access), not `employee` (your registered
# people stay enrolled), photos, or CHIRLA data. Useful before a dedicated
# verification run: without this, a fresh run's sightings get merged
# (min/max) with leftover rows from past sessions -- both bucket under the
# video's own recording day (2023-06-08), not real-world "today" -- so old
# first-arrival/last-departure times, or a long-resolved restricted-area
# violation, can leak into what should be a clean table you're correlating
# against a live run (found live 2026-09-06: a single closed violation row
# from an early test sat on the attendance page's ViolationsPanel
# indefinitely -- nothing ever cleared it). Safe to run whether the stack
# is up or down.
clear:
    #!/usr/bin/env bash
    set -euo pipefail
    cd products/attendance
    python3 -c "import sqlite3; conn = sqlite3.connect('data/attendance.db'); counts = {t: (conn.execute(f'DELETE FROM {t}').rowcount) for t in ('sighting', 'desk_interval', 'violation')}; conn.commit(); print(f\"cleared {counts['sighting']} sighting row(s), {counts['desk_interval']} desk interval(s), {counts['violation']} violation row(s)\"); conn.close()"

# Restart the CHIRLA replay from the beginning of the video(s), without
# touching docker or the other host processes (2026-09-01, user request:
# "a quick command to restart the replay of the cam in the web"). With no
# argument it replays whatever is CURRENTLY streaming (read from
# stream_starts.json); pass cams to switch, e.g. `just replay cam-15` or
# `just replay "cam-15,cam-16,cam-17"`. The web dashboard resets itself:
# products/attendance sees every camera's position drop back to 0 and fires
# its loop-wrap reset (clears the sighting table + pushes a fresh snapshot),
# and the sidecar re-attaches to the republished RTSP paths on its own
# reconnection watchdog (~10-20s of frozen tiles is normal).
replay cams="":
    #!/usr/bin/env bash
    set -euo pipefail
    # Multi-developer isolation: see _start_host_processes for why this is needed.
    set -a
    [ -f .env ] && source .env
    set +a
    cams="{{cams}}"
    if [ -z "$cams" ]; then
        cams=$(python3 -c "import json; print(','.join(sorted(json.load(open('data/stream_starts.json')).keys())))" 2>/dev/null || true)
    fi
    if [ -z "$cams" ]; then
        echo "nothing is streaming and no cams given -- e.g.: just replay cam-15" >&2
        exit 1
    fi
    pidfile={{run_dir}}/streamer.pid
    if [ -f "$pidfile" ]; then
        pid=$(cat "$pidfile")
        kill -TERM "-$pid" 2>/dev/null && echo "stopped streamer (pgid $pid)" || true
        rm -f "$pidfile"
        sleep 1
    fi
    echo "replaying from 00:00: $cams"
    cd test_streamer
    setsid bash -c 'echo $$ > ../{{run_dir}}/streamer.pid; exec nohup python3 stream_recordings.py --ids '"$cams"' --port '"${GO2RTC_RTSP_PORT:-8554}" \
        > ../{{run_dir}}/streamer.log 2>&1 &
    sleep 3
    if ! ps -p "$(cat ../{{run_dir}}/streamer.pid 2>/dev/null || echo 0)" > /dev/null 2>&1; then
        echo "ERROR: streamer failed to start -- {{run_dir}}/streamer.log says:" >&2
        tail -n 5 ../{{run_dir}}/streamer.log >&2
        exit 1
    fi
    tail -n +1 ../{{run_dir}}/streamer.log | head -5
    # Verify every replayed cam's preview actually starts advancing again.
    # Killing a publisher mid-RTSP-session sometimes leaves the sidecar's
    # reader WEDGED on the dead socket (found live 2026-09-01: cam-15 looping
    # "gst_rtspsrc_try_send: Could not write to resource" every 6s, forever,
    # while its siblings redialed fine). A wedged session never self-heals and
    # there is no per-source restart control -- the only cure is a sidecar
    # restart, so detect it here and do that automatically, only when needed.
    echo "verifying streams resume (a reconnect can take ~10s)..."
    sleep 6
    wedged=""
    for cam in ${cams//,/ }; do
        ok=""
        deadline=$((SECONDS + 20))
        while [ $SECONDS -lt $deadline ]; do
            h1=$(curl -s --max-time 3 "http://localhost:18000/stream/$cam/frame?t=r$SECONDS" | md5sum | cut -d' ' -f1)
            sleep 1.5
            h2=$(curl -s --max-time 3 "http://localhost:18000/stream/$cam/frame?t=s$SECONDS" | md5sum | cut -d' ' -f1)
            if [ -n "$h1" ] && [ "$h1" != "$h2" ]; then ok=1; break; fi
        done
        if [ -z "$ok" ]; then wedged="$wedged $cam"; else echo "  $cam: streaming"; fi
    done
    if [ -n "$wedged" ]; then
        echo "  stalled:$wedged -- restarting the sidecar to clear the wedged RTSP session(s)"
        docker restart "$(docker compose ps -q deepstream-sidecar)" > /dev/null
        echo "  sidecar restarting; tiles resume once the pipeline is back (~30-60s)"
    else
        echo "replay complete -- all cams streaming."
    fi

# Everything CI runs, run here. Same commands, same order -- when these two
# drift, "works on my machine" comes back. Takes ~2 minutes.
check:
    #!/usr/bin/env bash
    set -uo pipefail
    fail=0
    echo "== the locks are current =="
    python3 -I tools/deps/perception_groups.py --check && python3 -I tools/deps/root_exports.py --check || fail=1
    for d in . perception products/attendance products/barber products/fall_detection tools/onnx-export tools/mtmc-research tools/mlflow-server; do
        (cd {{justfile_directory()}}/$d && env -u VIRTUAL_ENV uv lock --check --quiet) || { echo "stale lock: $d (just deps-lock)" >&2; fail=1; }
    done
    echo "== the repo's own suite =="
    # The locked `test` group CI installs (pyproject.toml), in a throwaway environment
    # uv caches -- NOT the system python, which lacks cv2 and friends, and not a
    # hand-made venv, which is how local and CI drift apart in the first place.
    env -u VIRTUAL_ENV uv run --quiet --locked --isolated --only-group test \
        python -m pytest tests -q || fail=1
    echo "== barber's suite =="
    (cd {{justfile_directory()}}/products/barber && env -u VIRTUAL_ENV uv run --quiet --locked python -m pytest -q) || fail=1
    echo "== the dashboards: type-check + build =="
    (cd {{justfile_directory()}}/{{frontend}} && npm run build) || fail=1
    [ $fail = 0 ] && echo "all checks passed" || { echo "SOMETHING FAILED (see above)" >&2; exit 1; }

# Tail every log at once: the docker-compose stack plus the 3 host processes.
# `just logs <service>` tails only that compose service (e.g. `just logs iris`).
logs service="":
    #!/usr/bin/env bash
    if [ -n "{{service}}" ]; then
        exec docker compose logs -f --tail 100 {{service}}
    fi
    docker compose logs -f --tail=50 &
    compose_pid=$!
    trap 'kill $compose_pid 2>/dev/null' EXIT
    tail -f {{run_dir}}/*.log

# ── Presets: multi-camera tracking + VLM monitoring ────────────────────────
# Each preset is configs/presets/<name>/{config.yaml,sidecar.yaml}, applied over
# the tracked files by scripts/switch_preset.sh. These use the test streamer's
# own daemon (scripts/start_streamer.sh -> .streamer.pid, data/streamer.log),
# not the attendance host processes. `just down` stops either kind of run.

# Run the single-camera VLM Cognitive Monitoring setup (clean start: down -> configure -> up -> streamer)
vlm streamer="true":
    @bash scripts/run_setup.sh vlm {{streamer}}

# Run the 13-camera Multi-Camera Tracking setup (clean start: down -> configure -> up -> streamer)
multicam streamer="true":
    @bash scripts/run_setup.sh multicam {{streamer}}

# Alias for multicam
multi streamer="true":
    @just multicam {{streamer}}

# Clean restart of the currently active preset (vlm by default)
restart streamer="true":
    #!/usr/bin/env bash
    PROFILE="vlm"
    if [[ -f .active_profile ]]; then
        PROFILE="$(cat .active_profile)"
    fi
    echo "Restarting active profile: $PROFILE"
    bash scripts/run_setup.sh "$PROFILE" {{streamer}}

# Switch configuration preset without restarting containers (options: vlm, multicam)
switch profile:
    @bash scripts/switch_preset.sh {{profile}}

# Start the preset test streamer in the background (options: multicam, vlm)
streamer-start profile="multicam" *args="":
    @bash scripts/start_streamer.sh {{profile}} {{args}}

# Stop the background preset test streamer and any orphaned pushers
streamer-stop:
    @bash scripts/stop_streamer.sh

# Check status of the preset test streamer process
streamer-status:
    #!/usr/bin/env bash
    if [[ -f .streamer.pid ]]; then
        PID="$(cat .streamer.pid)"
        if kill -0 "$PID" 2>/dev/null; then
            echo "Test streamer is RUNNING (PID: $PID)"
            echo "--- Recent logs (data/streamer.log) ---"
            tail -n 10 data/streamer.log 2>/dev/null || true
            exit 0
        fi
    fi
    echo "Test streamer is STOPPED"

# Follow the preset test streamer log output
streamer-logs:
    @tail -f data/streamer.log

# Run the test streamer in the foreground (options: multicam, vlm, attendance)
streamer-fg profile="multicam" *args="":
    @python3 test_streamer/stream_recordings.py --profile {{profile}} {{args}}

# ── truck-tracking web UI (../truck-tracking, or $TRUCK_UI_DIR) ──────────
# `just multi` starts it after the stack is ready and `just down` stops it and CLEARS its
# tracked trucks (so every run starts clean). These are for running it on its own.

# Start the truck-tracking UI (database, migrations, cameras, API + web) -> http://localhost:5173
truck-ui:
    @bash scripts/truck_ui.sh up

# Stop the truck-tracking UI, clear its tracked trucks and stop its database
truck-ui-down:
    @bash scripts/truck_ui.sh down

# Clear the truck-tracking UI's tracked trucks and history, leave it running
truck-ui-reset:
    @bash scripts/truck_ui.sh reset

# Is the truck-tracking UI up, and how many trucks does it hold
truck-ui-status:
    @bash scripts/truck_ui.sh status

# Show system status: active preset, containers, streamer
status:
    #!/usr/bin/env bash
    PROFILE="default (attendance/fire)"
    if [[ -f .active_profile ]]; then
        PROFILE="$(cat .active_profile)"
    fi
    echo "Active Profile: $PROFILE"
    echo ""
    echo "Docker Containers:"
    docker compose --profile roles --profile multicam ps
    echo ""
    just streamer-status

# Run test suite
test *args="":
    @pytest tests/ {{args}}


# ──────────────────────────────────────────────────────────────────────────────
# Scale run on THIS machine: 15 cameras of footage from /media/mmtayee/WD_1 -> DeepStream sidecar
# (+ hosted VLM) -> truck-tracking platform, with Grafana health monitoring. Runs from
# ~/URCA/IRIS/scale-local (committed files of HEAD are copied there by `sync`: commit first).
#   just scale-reset      fresh run from footage 11:49 (clears trucks)       <- the usual one
#   just demo-v1          the original historical day (as yesterday) + a LIVE run beside it, always from a clean state
#   just demo-v3          the day the new start rules keep (as yesterday) + a LIVE run beside it, always from a clean state (demo-v2 = the same)
#   just scale-continue   keep the history in the platform (imported day) and start LIVE from <start> beside it
#   just scale-up         start / restart everything, keep the trucks in the database
#   just scale-down       stop it all (containers, footage players, monitoring)
#   just scale-status     what is running, how many cameras publish, where to look
#   just scale-logs [svc] follow a container's log (default: the sidecar)
#   just scale-flight     latest stall evidence recorded by the sidecar's flight recorder
# Platform https://localhost:35444   VLM page /iris/vlm   health dashboard /grafana/
# ──────────────────────────────────────────────────────────────────────────────
scale_env := "LOCAL=1 UI_HTTPS_PORT=35444"
scale_start := "2026-09-13T11:49:00"
# Every command that takes a start also takes just a time of the recorded day: `just demo-v1 14:30`, `just scale-continue 08:15:30`

# Fresh run: stop, clear Grafana's history (metrics, logs, events), start from footage 11:49, clear the platform's trucks, wait for the cameras
scale-reset start=scale_start:
    #!/usr/bin/env bash
    set -euo pipefail
    export LOCAL=1 UI_HTTPS_PORT=35444
    git diff --quiet HEAD -- . || echo "note: uncommitted changes are NOT deployed (sync ships the committed files of HEAD)"
    scripts/scale13_deploy.sh down || true
    scripts/scale13_deploy.sh mon reset      # empty graphs and logs; last run's events/stall dumps are archived
    START={{start}} scripts/scale13_deploy.sh up
    scripts/scale13_deploy.sh ui reset
    docker restart scale-iris-1 >/dev/null
    just scale-wait
    just scale-status

# Keep the platform's history (e.g. the imported historical day) and start LIVE from footage <start> beside it:
# stops the stack, clears only the previous run's Redis streams (id counter and crop index kept), starts the new run.
# No `ui reset`, no `mon reset`: trucks, journeys, sightings and Grafana history stay.
scale-continue start=scale_start:
    #!/usr/bin/env bash
    set -euo pipefail
    export LOCAL=1 UI_HTTPS_PORT=35444
    git diff --quiet HEAD -- . || echo "note: uncommitted changes are NOT deployed (sync ships the committed files of HEAD)"
    START={{start}} scripts/scale13_deploy.sh continue
    just scale-wait
    just scale-status

# DEMO v1: the original historical day (from the read-only archive) as YESTERDAY, then a LIVE run from footage <start> beside it.
# Every call starts from the same clean state: it replaces the platform database, the day's crops/stays and the Redis
# snapshot with the original export (so earlier live trucks are gone), shifts the day's times to yesterday, then runs
# `scale-continue`. Takes about 5 minutes. The archive is only read.
demo-v1 start=scale_start:
    #!/usr/bin/env bash
    set -euo pipefail
    # the live run needs the GPU: fail now, before anything is replaced (an update that leaves the old kernel module loaded breaks it until a reboot)
    nvidia-smi -L >/dev/null 2>&1 || { echo "the NVIDIA driver is not usable here: $(nvidia-smi 2>&1 | head -1)"; echo "after a driver update ('Driver/library version mismatch') reboot the machine, then run this again. Nothing was changed."; exit 1; }
    export LOCAL=1 UI_HTTPS_PORT=35444
    A="$HOME/URCA/IRIS/archive/day-export-2026-09-13-ORIGINAL"
    [ -f "$A/manifest.json" ] || { echo "the archived original day is missing: $A"; exit 1; }
    git diff --quiet HEAD -- . || echo "note: uncommitted changes are NOT deployed (sync ships the committed files of HEAD)"
    DAY=$(date -d yesterday +%F)
    echo "== 1/2 the historical day (original archive) as $DAY"
    SKIP_BACKUP=1 SHIFT_TO="${DAY}T00:00:00" scripts/scale_day_import.sh --local "$A"
    # false-positive trucks removed earlier (scripts/scale_remove_trucks.py) are removed again from the fresh import
    [ -f "$HOME/URCA/IRIS/review/removed_trucks.json" ] && python3 scripts/scale_remove_trucks.py --from-record --apply
    echo "== 2/2 live from footage {{start}} beside it"
    START={{start}} scripts/scale13_deploy.sh continue
    just scale-wait
    just scale-status
    echo "demo-v1 ready: platform https://localhost:35444 (history = $DAY, live = today); Grafana /grafana/"

# demo-v3 (tag demo-v3 in both repos): the day the NEW start rules would keep (182 of the original 365 trucks) as yesterday, + a LIVE run beside it,
# always from a clean state; the web app has the global day picker (top bar) and the Trucks page category filter.
# Built from the archive and the two removal lists (review/removed_trucks.json = your manual false positives,
# review/whatif/removed_trucks_newrules.json = what the new rules would not track). Pick the day in the top bar.
demo-v3 start=scale_start:
    #!/usr/bin/env bash
    set -euo pipefail
    # the live run needs the GPU: fail now, before anything is replaced (an update that leaves the old kernel module loaded breaks it until a reboot)
    nvidia-smi -L >/dev/null 2>&1 || { echo "the NVIDIA driver is not usable here: $(nvidia-smi 2>&1 | head -1)"; echo "after a driver update ('Driver/library version mismatch') reboot the machine, then run this again. Nothing was changed."; exit 1; }
    export LOCAL=1 UI_HTTPS_PORT=35444
    A="$HOME/URCA/IRIS/archive/day-export-2026-09-13-ORIGINAL"
    [ -f "$A/manifest.json" ] || { echo "the archived original day is missing: $A"; exit 1; }
    git diff --quiet HEAD -- . || echo "note: uncommitted changes are NOT deployed (sync ships the committed files of HEAD)"
    DAY=$(date -d yesterday +%F)
    echo "== 1/3 the original export as $DAY (its footage 00:00 = $DAY 00:00)"
    SKIP_BACKUP=1 SHIFT_TO="${DAY}T00:00:00" scripts/scale_day_import.sh --local "$A"
    echo "== 2/3 the trucks the new start rules would not track are removed ($DAY becomes the new-rules day)"
    [ -f "$HOME/URCA/IRIS/review/removed_trucks.json" ] && python3 scripts/scale_remove_trucks.py --from-record --apply
    [ -f "$HOME/URCA/IRIS/review/whatif/removed_trucks_newrules.json" ] && \
        REMOVED_TRUCKS="$HOME/URCA/IRIS/review/whatif/removed_trucks_newrules.json" python3 scripts/scale_remove_trucks.py --from-record --apply
    echo "== 3/3 live from footage {{start}} beside it"
    START={{start}} scripts/scale13_deploy.sh continue
    just scale-wait
    just scale-status
    echo "demo-v3 ready: platform https://localhost:35444 ($DAY = what the new rules keep, today = live); pick the day in the top bar; Grafana /grafana/"

# demo-v4 = demo-v3 + the Hazard Detection tab of the platform UI (choose a stream, Analyse, annotated image). The tab is part of the
# platform repo and talks to the camera system's /api/hazards through the same-origin /iris path, so nothing else is needed.
demo-v4 start=scale_start:
    #!/usr/bin/env bash
    set -euo pipefail
    just demo-v3 {{start}}
    H=$(grep '^PUBLIC_HOST=' "$HOME/URCA/IRIS/scale-local/.env" | cut -d= -f2)
    echo "demo-v4 ready: Hazard Detection  https://$H:35444/hazards   (platform login as usual)"

# demo-v3 on the shared SERVER (ssh iris@100.102.162.114, or SERVER=iris@10.2.0.60 in the office): syncs the code and the removal lists,
# then runs the same steps there (import the export, remove, live run); the server keeps running after this ends.
demo-v3-server start=scale_start:
    scripts/scale_demo_v3_server.sh {{start}}

# demo-v2 is the same thing as demo-v3 now
demo-v2 start=scale_start:
    just demo-v3 {{start}}

# Named versions of the platform's day data: `just day-version save day-v1`, `just day-version list`, `just day-version restore day-v1`
day-version action name="":
    scripts/scale_day_version.sh {{action}} {{name}}

# Start / restart everything without clearing trucks (footage restarts from <start>)
scale-up start=scale_start:
    #!/usr/bin/env bash
    set -euo pipefail
    export LOCAL=1 UI_HTTPS_PORT=35444
    START={{start}} scripts/scale13_deploy.sh up
    just scale-wait
    just scale-status

# Stop the footage players and every container of the scale run
scale-down:
    {{scale_env}} scripts/scale13_deploy.sh down

# Wait (up to 8 min) until all 15 cameras publish
scale-wait:
    #!/usr/bin/env bash
    for _ in $(seq 96); do
        n=$(curl -s -G localhost:39090/api/v1/query --data-urlencode 'query=count(iris_camera_meta_age_seconds<30)' | python3 -c "import sys,json;r=json.load(sys.stdin)['data']['result'];print(r[0]['value'][1] if r else 0)" 2>/dev/null || echo 0)
        [ "$n" = 15 ] && { echo "all 15 cameras publishing"; exit 0; }
        sleep 5
    done
    echo "only ${n:-0}/15 cameras publishing after 8 min -- see: just scale-status, just scale-logs"; exit 1

# What is running and where to look
scale-status:
    #!/usr/bin/env bash
    docker ps --format '{{ "{{.Names}}" }}\t{{ "{{.Status}}" }}' | grep '^scale-' | sort || echo "no scale containers"
    echo; {{scale_env}} scripts/scale13_deploy.sh stream status 2>&1 | head -2
    n=$(curl -s -G localhost:39090/api/v1/query --data-urlencode 'query=count(iris_camera_meta_age_seconds<30)' | python3 -c "import sys,json;r=json.load(sys.stdin)['data']['result'];print(r[0]['value'][1] if r else 0)" 2>/dev/null || echo "?")
    echo "cameras publishing: $n/15"
    echo; echo "platform   https://localhost:35444   (VLM page: /iris/vlm)"
    echo "health     https://localhost:35444/grafana/"

# ── Full-day run on the server (see scripts/scale13_deploy.sh "Full-day run") ─────────────────────
# Arm it: footage 00:00 begins at <begin> (local server time, e.g. 2026-10-06T00:00:00); black until then.
scale-day-arm begin:
    BEGIN_AT={{begin}} scripts/scale13_deploy.sh day-arm

# How the day run is doing (footage clock, cameras, heals, platform counts, disk)
scale-day-status:
    scripts/scale13_deploy.sh day-status

# After the day: collect everything on the server, then copy it here and import it (needs the export folder's name)
scale-day-export:
    ssh -o RequestTTY=no -o RemoteCommand=none iris-work 'cd /data/IRIS/TESTING_PLAYGROUND/surveillance-camera-sys-scale && scripts/scale_day_export.sh'

scale-day-import folder:
    scripts/scale_day_import.sh {{folder}}

# Open an imported day on this machine (no sidecar, no footage players)
scale-browse:
    LOCAL=1 UI_HTTPS_PORT=35444 scripts/scale13_deploy.sh browse

# Follow a container's log (default: the sidecar), e.g. `just scale-logs iris-1`
scale-logs svc="deepstream-sidecar-1":
    docker logs -f --tail 100 scale-{{svc}}

# A/B tests of what the GPU core is spent on (~35 min; restarts the sidecar per test, restores the config). e.g. `just scale-gpu-test T0,T1`
scale-gpu-test tests="T0,T1,T2,T3,T4":
    python3 scripts/scale_gpu_experiment.py --tests {{tests}}

# Recent events (alerts, GPU throttles, stalls) and the latest stall evidence (stacks of every thread)
scale-flight:
    #!/usr/bin/env bash
    d=../../scale-local/data/flight; [ -d "$d" ] || d=$HOME/URCA/IRIS/scale-local/data/flight
    echo "== events (alerts, GPU throttles, stalls, backpressure): $d/events.log"; tail -n 20 "$d/events.log" 2>/dev/null || echo "(none yet)"
    echo; ls -lt "$d" 2>/dev/null | head -8 || echo "no flight records"
    echo; docker logs scale-deepstream-sidecar-1 2>&1 | grep -E "\[flight\] (main loop|native)" | tail -5 | cut -c1-300

# ---- Data (DVC) ------------------------------------------------------------
# Models, datasets, sample videos, golden clips and baselines are versioned with DVC, not git
# (DATA.md). The remote is the team OneDrive folder, reached through an rclone mount at
# .data-remote/ until the S3-compatible store exists; then `dvc remote default minio`.

# Mount the team folder PhyAI/IRIS/repo-data at .data-remote/ (needs rclone remote `iris-onedrive` and fuse3; DATA.md)
data-mount:
    #!/usr/bin/env bash
    set -euo pipefail
    command -v rclone >/dev/null || { echo "rclone is not installed" >&2; exit 1; }
    rclone listremotes | grep -qx 'iris-onedrive:' \
        || { echo "no rclone remote 'iris-onedrive' -- see DATA.md, 'One-time setup'" >&2; exit 1; }
    mkdir -p .data-remote
    if mountpoint -q .data-remote; then echo "already mounted: .data-remote"; exit 0; fi
    # The MLflow server's own folders live beside dvc-store; it alone writes them (tools/mlflow-server), so hide them here.
    rclone mount iris-onedrive:IRIS/repo-data .data-remote --exclude '/mlflow-*/**' \
        --vfs-cache-mode writes --dir-cache-time 30s --transfers 4 --daemon
    for _ in $(seq 1 20); do mountpoint -q .data-remote && break; sleep 0.5; done
    mountpoint -q .data-remote || { echo "mount did not come up (try: rclone mount ... without --daemon)" >&2; exit 1; }
    mkdir -p .data-remote/dvc-store
    echo "mounted: .data-remote -> iris-onedrive:IRIS/repo-data"

data-umount:
    fusermount3 -u .data-remote 2>/dev/null || fusermount -u .data-remote

# Fetch data from the remote. No argument = the default groups (models, golden, bundles);
# `just data-pull datasets samples-multicam` for others; `just data-pull all` for everything.
data-pull *groups:
    #!/usr/bin/env bash
    set -euo pipefail
    command -v dvc >/dev/null || { echo "dvc is not installed (uv tool install 'dvc[s3,ssh]==3.67.1')" >&2; exit 1; }
    if [ "$(dvc config core.remote)" = onedrive ] && [ ! -d .data-remote/dvc-store ]; then
        echo "the data store is not mounted -- run: just data-mount" >&2; exit 1
    fi
    g="{{groups}}"
    if [ "$g" = all ]; then targets=$(python3 -I tools/data/catalog.py targets --all)
    else targets=$(python3 -I tools/data/catalog.py targets $g); fi
    echo "$targets" | xargs dvc pull

# Upload new or changed data to the remote (after `dvc add` / `dvc repro`)
data-push:
    dvc push

# What differs from the remote and from the commit's pointers
data-status:
    dvc status -c
    dvc data status --not-in-remote

# Check every .dvc is in a group (data/catalog.yaml) and regenerate the table in DATA.md
data-catalog:
    python3 -I tools/data/catalog.py check
    python3 -I tools/data/catalog.py docs

# Rebuild the browsable per-use-case view of the store on the server (hardlinks, no extra space; DATA.md)
data-view host="iris-work" root="/data/IRIS/repo-data":
    #!/usr/bin/env bash
    set -euo pipefail
    ssh_opts="-o RemoteCommand=none -o RequestTTY=no"
    tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
    python3 -I tools/data/export_view.py manifest > "$tmp/manifest.json"
    ssh $ssh_opts {{host}} "mkdir -p {{root}}/tools"
    scp -q $ssh_opts tools/data/export_view.py "$tmp/manifest.json" {{host}}:{{root}}/tools/
    ssh $ssh_opts {{host}} "python3 {{root}}/tools/export_view.py apply --store {{root}}/dvc-store --out {{root}}/by-usecase --manifest {{root}}/tools/manifest.json"

# Fill the OneDrive mirror from the local DVC cache (rclone copy, resumable: it skips what is already there). Slow on
# Wi-Fi (about 1 MB/s); a wired machine is much faster. DATA.md, "Remotes".
data-mirror:
    rclone copy .dvc/cache/files iris-onedrive:IRIS/repo-data/dvc-store/files --size-only --transfers 12 --checkers 16 --stats 60s --stats-one-line --log-level INFO

# Is everything in the local cache on the OneDrive mirror? (names and sizes; no download)
data-mirror-check:
    rclone check .dvc/cache/files iris-onedrive:IRIS/repo-data/dvc-store/files --size-only --one-way

# ---- Dependencies (locks) -----------------------------------------------------
# Re-lock after changing a dependency (DEPENDENCIES.md), keeping every version already locked (`uv lock` never upgrades
# unless asked). The sidecar: its groups are regenerated from the requirements.txt files first. The root: the
# requirements*.txt exports are rewritten from its lock after. Commit what changes.
# Re-lock all seven locks after changing a dependency, keeping every locked version
deps-lock:
    python3 -I tools/deps/perception_groups.py
    cd perception && env -u VIRTUAL_ENV uv lock
    env -u VIRTUAL_ENV uv lock
    python3 -I tools/deps/root_exports.py
    cd tools/onnx-export && env -u VIRTUAL_ENV uv lock
    cd tools/mtmc-research && env -u VIRTUAL_ENV uv lock
    cd products/attendance && env -u VIRTUAL_ENV uv lock
    cd products/barber && env -u VIRTUAL_ENV uv lock
    cd products/fall_detection && env -u VIRTUAL_ENV uv lock
    cd tools/mlflow-server && env -u VIRTUAL_ENV uv lock

#   just deps-update fastapi                       the root lock, to its newest version
#   just deps-update insightface perception        the sidecar's lock
#   just deps-update 'pillow<13' perception        newest within a bound (stay in the current major)
#   just deps-update 'pillow==13.0.0' perception   a deliberate major: its own merge request
# `lock` is the lock's directory; then review the printed diff and run the checks that lock needs (DEPENDENCIES.md).
# Update one package in one lock
deps-update package lock=".":
    cd {{lock}} && env -u VIRTUAL_ENV uv lock --upgrade-package '{{package}}'
    python3 -I tools/deps/root_exports.py
    just deps-diff

# Package by package: MAJOR, minor, patch, added, removed, HELD (torch). A MAJOR or HELD line belongs in a deliberate
# merge request of its own.
# What the locks change against a ref (default: the last commit)
deps-diff ref="HEAD":
    python3 -I tools/deps/hold_majors.py --report --base {{ref}}

# ---- Regression harness (benchmarks/SCORECARD.md, issue #174) ------------------------------------------------------------
# Measure this checkout on this machine and write one results file (benchmarks/runs/<machine_id>/): unit suites, build
# sizes, and the golden replays with their GPU cost, fps, drops, boot time and engine rebuilds. Needs the GPU idle:
# it refuses (exit 2) otherwise, or records the run as NOISY with --allow-noisy. A laptop that drives its display sits at
# some background load: REGRESS_IDLE_GPU_UTIL=60 raises the limit, and the load is recorded in the file.
#   just regress                          everything (about 25 minutes)
#   just regress --only unit,build        the part that needs no GPU
#   just regress --only golden --configs attendance,zones
regress *args:
    PYTHONPATH=tools python3 -m regress run {{args}}

# Is this machine idle enough for a trustworthy efficiency measurement? (exit 0 = idle)
regress-idle:
    PYTHONPATH=tools python3 -m regress idle

# What machine and checkout a result would be filed under
regress-fingerprint:
    PYTHONPATH=tools python3 -m regress fingerprint

# Which GPU counter layers does this machine answer? NVML (engines, clocks, power), DCGM profiling (SM, tensor, DRAM),
# tegrastats (Jetson). Run it when a machine is onboarded (about 10 to 45 s).
regress-gpu-probe:
    PYTHONPATH=tools python3 -m regress gpu-probe

# Print a results file as a table: just regress-show benchmarks/runs/<machine_id>/<file>.json
regress-show file:
    PYTHONPATH=tools python3 -m regress show {{file}}

# Record this machine's baseline: N runs (default 3) of the same CLEAN commit, the median of each number, written to
# benchmarks/baselines/<machine_id>/ with the golden outputs. Refuses a dirty tree, a run that was not idle, or a run that
# fails correctness. Run-options go after `--`:  just regress-baseline --runs 3 -- --only unit,golden
regress-baseline *args:
    PYTHONPATH=tools python3 -m regress baseline {{args}}

# PASS / WARN / FAIL of a results file against this machine's baseline (exit 1 on a correctness failure; --strict makes
# efficiency warnings fail too; --all lists passing rows):  just regress-compare benchmarks/runs/<machine_id>/<file>.json
regress-compare file *args:
    PYTHONPATH=tools python3 -m regress compare {{file}} {{args}}

# How much do the numbers move between identical runs? Give it the results files of 5+ runs of one commit:
#   just regress-noise benchmarks/runs/<machine_id>/*.json --tolerances benchmarks/tolerances.json
regress-noise *args:
    PYTHONPATH=tools python3 -m regress noise {{args}}

# Which checks does THIS change need? Reads the changed files (against the merge-base with --base, default origin/main), prints the tier
# (benchmarks/tiers.json), the configs, the exact commands and their minutes:  just regress-plan   or   just regress-plan --base origin/main
regress-plan *args:
    PYTHONPATH=tools python3 -m regress plan {{args}}

# The machine's limits, per use case: un-paced 1-camera ceiling and the cameras kept up with (generated overlay configs, nothing committed):
#   just regress-capacity --configs ppe --cameras 1 [--analysis-fps 5]     just regress-capacity --models all --cameras 1,2     just regress-capacity --engines
regress-capacity *args:
    PYTHONPATH=tools python3 -m regress capacity {{args}}

# Long-run growth of a RUNNING stack (container memory split, disk, Redis, GPU, pipeline), then the coarse gate against the reference soak:
#   just regress-soak --project safety --duration 3h --out <dir> ...   then   just regress-soak-compare <dir>/analysis.json
regress-soak *args:
    PYTHONPATH=tools python3 -m regress soak {{args}}

regress-soak-compare file *args:
    PYTHONPATH=tools python3 -m regress soak-compare {{file}} {{args}}

# What else was the machine doing during a benchmark: memory / CPU / IO pressure, swap, temperatures, throttling, container restarts and logs
#   just regress-hostsense --out <dir> --duration 6h      (run it beside a long benchmark, at low priority: nice -n 19)
regress-hostsense *args:
    PYTHONPATH=tools python3 -m regress hostsense {{args}}

# ── PPE demo: one simulated live camera (configs/presets/ppe/live*.yaml; perception/modules/ppe/README.md) ──
# A test video from data/uploads/ppe (DVC: `just data-pull samples-ppe`) looped at its real speed into go2rtc as raw-<camera id>, read by
# the PPE sidecar as the camera of live-<name>.yaml, with the IRIS app and its dashboard. Own compose project (ppelive)
# and ports, so it runs beside every other stack; its data lives in scratch/ppe-live/data. One video at a time:
# starting another switches to it. Settings (project, ports, image, dirs): the PPE_* lines of .env.example, read by
# scripts/ppe_live_env.sh; the containers run as the user running the recipe (PPE_UID/PPE_GID, set there).
ppe_model := "perception/models/DeepStream-Yolo-PPE/voxdroid_yolov8s_css.onnx"

# Check what the PPE demo needs: the model (SHA-256 from its README) and the two test videos, which are DVC pointers (DATA.md)
ppe-check:
    #!/usr/bin/env bash
    set -uo pipefail
    source scripts/ppe_live_env.sh
    fail=0
    ok() { echo "ok    $*"; }; bad() { echo "FAIL  $*"; fail=1; }
    # the model and the videos are versioned with DVC: a clone has to pull them
    real() { [[ -f "$1" ]] || { bad "$1 is missing: just data-pull $2"; return 1; }; return 0; }
    want="$(grep -A1 "^Expected file" perception/models/DeepStream-Yolo-PPE/README.md | grep -oE '[0-9a-f]{64}' | head -n1)"
    if real {{ppe_model}} models-ppe; then
        got="$(sha256sum {{ppe_model}} | cut -d' ' -f1)"
        [[ "$got" == "$want" ]] && ok "{{ppe_model}} (SHA-256 $got)" || bad "{{ppe_model}}: SHA-256 $got, README says $want"
    fi
    for v in test.mp4 test2.mp4; do
        real "$PPE_LIVE_VIDEOS/$v" samples-ppe && ok "$PPE_LIVE_VIDEOS/$v ($(du -h "$PPE_LIVE_VIDEOS/$v" | cut -f1))"
    done
    # The sidecar, running as you, writes its TensorRT engines beside the models (and the custom-engine ones in .)
    for d in . perception/models/DeepStream-Yolo-PPE perception/models/DeepStream-Yolo; do
        [[ -w "$d" ]] && ok "$d writable (engines)" || bad "$d is not writable by $(id -un): the sidecar runs as you and writes its engines there"
    done
    for c in docker ffmpeg ffprobe; do command -v $c >/dev/null && ok "$c" || bad "$c is not installed"; done
    nvidia-smi -L >/dev/null 2>&1 && ok "GPU: $(nvidia-smi -L | head -n1)" || bad "no NVIDIA GPU (nvidia-smi)"
    [ $fail = 0 ] && echo "PPE demo ready: just ppe-live-sim VIDEO=test.mp4 (or test2.mp4)" || { echo "fix the FAIL lines first" >&2; exit 1; }

# Build the PPE demo's images: the sidecar (PPE_SIDECAR_IMAGE; long, see the message of ppe-live-sim) and the app
ppe-build:
    #!/usr/bin/env bash
    set -euo pipefail
    source scripts/ppe_live_env.sh
    echo "building $PPE_SIDECAR_IMAGE and $PPE_IRIS_IMAGE"
    ppe_compose build deepstream-sidecar iris

# Start the PPE demo on one video (`just ppe-live-sim VIDEO=test2.mp4`): stack + looping streamer
ppe-live-sim VIDEO="test.mp4": ppe-check
    #!/usr/bin/env bash
    set -euo pipefail
    source scripts/ppe_live_env.sh
    d="$PPE_LIVE_DIR"
    video="{{VIDEO}}"; video="${video#VIDEO=}"
    [[ "$video" == */* ]] || video="$PPE_LIVE_VIDEOS/$video"
    [[ -f "$video" ]] || { echo "no video at $video"; exit 1; }
    name="$(basename "${video%.*}")"; cfg="configs/presets/ppe/live-$name.yaml"
    [[ -f "$cfg" ]] || { echo "no $cfg: add one like live-test2.yaml, and raw-<camera id> in go2rtc-live.yaml"; exit 1; }
    # The camera id is the preset's source id (live-test.yaml: cam-ppe-live)
    cam="$(sed -nE 's/.*\{id: *([^,} ]+).*/\1/p' "$cfg" | head -n1)"
    [[ -n "$cam" ]] || { echo "no source id in $cfg"; exit 1; }
    grep -q "raw-$cam:" configs/presets/ppe/go2rtc-live.yaml || { echo "raw-$cam missing in go2rtc-live.yaml"; exit 1; }
    # Images are built only by `just ppe-build`: the sidecar's first build is long, never started unasked
    missing=()
    docker image inspect "$PPE_SIDECAR_IMAGE" >/dev/null 2>&1 || missing+=("$PPE_SIDECAR_IMAGE")
    docker image inspect "$PPE_IRIS_IMAGE" >/dev/null 2>&1 || missing+=("$PPE_IRIS_IMAGE")
    if (( ${#missing[@]} )); then
        base=nvcr.io/nvidia/deepstream:9.0-triton-multiarch
        if docker image inspect "$base" >/dev/null 2>&1; then took="about 10-20 min ($base is already here)"
        else took="about 30-60 min the first time, most of it downloading $base (~26 GB)"; fi
        echo "missing docker image(s): ${missing[*]}"
        echo "build them once with:   just ppe-build"
        echo "  estimated: the sidecar (perception/pipeline/Dockerfile, ~31 GB) $took; the app a few minutes"
        echo "  or set PPE_SIDECAR_IMAGE in .env to a sidecar image built from this repo's perception/pipeline/Dockerfile"
        exit 1
    fi
    # The bind-mounted data dirs, made here so docker does not make them as root; they must be the containers' user's
    mkdir -p "$d/data/segments"
    for p in "$d/data" "$d/data/segments"; do
        [[ "$(stat -c %u:%g "$p")" == "$PPE_UID:$PPE_GID" ]] || {
            echo "$p belongs to $(stat -c %u:%g "$p"), not to $PPE_UID:$PPE_GID, the user the containers run as:"
            echo "chown -R $PPE_UID:$PPE_GID $p (or remove it), then start again"; exit 1; }
    done
    just ppe-live-stop-streamer
    ppe_compose up -d --no-build redis go2rtc
    # go2rtc reads go2rtc-live.yaml only at start; restart it, before the sidecar connects, only if it lacks the stream
    sleep 2
    curl -sf "localhost:$PPE_GO2RTC_API_PORT/api/streams" | grep -q "\"raw-$cam\"" || ppe_compose restart go2rtc
    # Re-encoded, not copied: a copied stream with B-frames made the sidecar decode corrupt frames and stall every
    # 25-45 s. H.264 baseline (no B-frames), no audio, a keyframe every second, the source's own bitrate -- what an
    # IP camera sends. Restarted if it dies; each start is logged with its wall time (the video's zero).
    fps="$(ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate -of csv=p=0 "$video")"
    gop="$(python3 -c "print(round(eval('$fps')))")"
    kbps="$(ffprobe -v error -select_streams v:0 -show_entries stream=bit_rate -of csv=p=0 "$video" | awk '{print int($1/1000)}')"
    setsid bash -c 'while true; do
        echo "START $(date +%s.%N) $2" >> "$1/streamer.log"
        ffmpeg -hide_banner -loglevel warning -re -stream_loop -1 -i "$2" -an \
            -c:v libx264 -preset veryfast -tune zerolatency -profile:v baseline -pix_fmt yuv420p \
            -g "$3" -b:v "${4}k" -maxrate "${4}k" -bufsize "$((2 * $4))k" \
            -max_muxing_queue_size 1024 -flush_packets 1 -rtsp_transport tcp -f rtsp \
            "$5" >> "$1/streamer.log" 2>&1
        echo "EXIT $(date +%s.%N) code=$?" >> "$1/streamer.log"
        sleep 2
    done' _ "$d" "$video" "$gop" "$kbps" "rtsp://localhost:$PPE_GO2RTC_RTSP_PORT/raw-$cam" > /dev/null 2>&1 < /dev/null &
    echo $! > "$d/streamer.pid"
    # The sidecar after the streamer: one that connects to go2rtc before the stream exists waits for a retry (~13 s).
    sleep 1
    SIDECAR_CONFIG="$cfg" ppe_compose up -d --no-build deepstream-sidecar iris
    printf 'VIDEO=%s\nCAM=%s\nCONFIG=%s\n' "$video" "$cam" "$cfg" > "$d/current.env"
    ip="$(hostname -I | awk '{print $1}')"
    echo "streamer started (pid $!): $video -> $cam (${fps} fps, GOP $gop, ${kbps} kbit/s)"
    echo "dashboard (alerts): http://$ip:$PPE_IRIS_PORT/#/iris/dashboard"
    echo "video (MJPEG):      http://$ip:$PPE_IRIS_PORT/stream/$cam"

# Stop the PPE live streamer only (the stack keeps running)
ppe-live-stop-streamer:
    #!/usr/bin/env bash
    source scripts/ppe_live_env.sh
    pidf="$PPE_LIVE_DIR/streamer.pid"
    if [[ -f $pidf ]]; then kill -TERM -- "-$(cat $pidf)" 2>/dev/null && echo "streamer stopped"; rm -f $pidf; fi

# Video, containers, streamer and the sidecar's last fps / alert lines for the PPE demo
ppe-live-status:
    #!/usr/bin/env bash
    source scripts/ppe_live_env.sh
    cat "$PPE_LIVE_DIR/current.env" 2>/dev/null || echo "no video started"
    ppe_compose ps
    pidf="$PPE_LIVE_DIR/streamer.pid"
    if [[ -f $pidf ]] && kill -0 "$(cat $pidf)" 2>/dev/null; then echo "streamer RUNNING (pid $(cat $pidf))"
    else echo "streamer STOPPED"; fi
    ppe_compose logs --no-log-prefix --tail 400 deepstream-sidecar 2>/dev/null \
        | grep -E "\] [0-9.]+ fps|PPE violation|source-stall" | tail -n 6

# Follow the PPE demo's sidecar log (svc=iris for the app)
ppe-live-logs svc="deepstream-sidecar":
    #!/usr/bin/env bash
    source scripts/ppe_live_env.sh
    ppe_compose logs -f --tail 100 {{svc}}

# Stop what ppe-live-sim started: containers, volumes, streamer (keeps scratch/ppe-live/ and the images)
ppe-live-down: ppe-live-stop-streamer
    #!/usr/bin/env bash
    source scripts/ppe_live_env.sh
    ppe_compose down -v
    rm -f "$PPE_LIVE_DIR/current.env"

# ── Shelf demo (empty shelf space, perception/modules/shelf_void/README.md) ──────────────────────────────────────────
# Same shape as the PPE demo above: loops a recorded video at its real speed into go2rtc as a simulated IP camera and
# runs the shelf sidecar as the camera of configs/presets/shelf/live-<name>.yaml, with the IRIS app and its dashboard.
# Own compose project (shelflive) and ports, so it runs beside every other stack; its data lives in
# scratch/shelf-live/data. Settings (project, ports, images, dirs): the SHELF_* lines of .env, read by
# scripts/shelf_live_env.sh; the containers run as the user running the recipe (SHELF_UID/SHELF_GID, set there).
shelf_model := "perception/models/DeepStream-Yolo-Shelf/detect_product_empty_space.onnx"

# Check what the shelf demo needs: the model (SHA-256 from its README), the demo video, docker, ffmpeg, the GPU
shelf-check:
    #!/usr/bin/env bash
    set -uo pipefail
    source scripts/shelf_live_env.sh
    fail=0
    ok() { echo "ok    $*"; }; bad() { echo "FAIL  $*"; fail=1; }
    want="$(grep -A1 "^Expected file" perception/models/DeepStream-Yolo-Shelf/README.md | grep -oE '[0-9a-f]{64}' | head -n1)"
    if [[ -f {{shelf_model}} ]]; then
        got="$(sha256sum {{shelf_model}} | cut -d' ' -f1)"
        [[ "$got" == "$want" ]] && ok "{{shelf_model}} (SHA-256 $got)" || bad "{{shelf_model}}: SHA-256 $got, README says $want"
    else
        bad "{{shelf_model}} is missing: just data-pull models-shelf"
    fi
    v="$SHELF_LIVE_VIDEOS/demo.mp4"
    [[ -f "$v" ]] && ok "$v ($(du -h "$v" | cut -f1))" || bad "$v is missing: just data-pull samples-shelf"
    # The sidecar, running as you, writes its TensorRT engines beside the models
    for d in . perception/models/DeepStream-Yolo-Shelf perception/models/DeepStream-Yolo; do
        [[ -w "$d" ]] && ok "$d writable (engines)" || bad "$d is not writable by $(id -un): the sidecar runs as you and writes its engines there"
    done
    for c in docker ffmpeg ffprobe; do command -v $c >/dev/null && ok "$c" || bad "$c is not installed"; done
    nvidia-smi -L >/dev/null 2>&1 && ok "GPU: $(nvidia-smi -L | head -n1)" || bad "no NVIDIA GPU (nvidia-smi)"
    [ $fail = 0 ] && echo "shelf demo ready: just shelf-live-sim" || { echo "fix the FAIL lines first" >&2; exit 1; }

# Build the shelf demo's docker images (the sidecar's first build is long; see the PPE README, "Run the demo")
shelf-build:
    #!/usr/bin/env bash
    set -euo pipefail
    source scripts/shelf_live_env.sh
    echo "building $SHELF_SIDECAR_IMAGE and $SHELF_IRIS_IMAGE"
    shelf_compose build deepstream-sidecar iris

# Start the shelf demo on one video (`just shelf-live-sim VIDEO=demo.mp4`): stack + looping streamer
shelf-live-sim VIDEO="demo.mp4": shelf-check
    #!/usr/bin/env bash
    set -euo pipefail
    source scripts/shelf_live_env.sh
    d="$SHELF_LIVE_DIR"
    video="{{VIDEO}}"; video="${video#VIDEO=}"
    [[ "$video" == */* ]] || video="$SHELF_LIVE_VIDEOS/$video"
    [[ -f "$video" ]] || { echo "no video at $video"; exit 1; }
    name="$(basename "${video%.*}")"; cfg="configs/presets/shelf/live-$name.yaml"
    [[ -f "$cfg" ]] || { echo "no $cfg: add one like live-demo.yaml, and raw-<camera id> in go2rtc-live.yaml"; exit 1; }
    cam="$(sed -nE 's/.*\{id: *([^,} ]+).*/\1/p' "$cfg" | head -n1)"
    [[ -n "$cam" ]] || { echo "no source id in $cfg"; exit 1; }
    grep -q "raw-$cam:" configs/presets/shelf/go2rtc-live.yaml || { echo "raw-$cam missing in go2rtc-live.yaml"; exit 1; }
    # Images are built only by `just shelf-build`: never started unasked
    missing=()
    docker image inspect "$SHELF_SIDECAR_IMAGE" >/dev/null 2>&1 || missing+=("$SHELF_SIDECAR_IMAGE")
    docker image inspect "$SHELF_IRIS_IMAGE" >/dev/null 2>&1 || missing+=("$SHELF_IRIS_IMAGE")
    if (( ${#missing[@]} )); then
        echo "missing docker image(s): ${missing[*]}"
        echo "build them once with:   just shelf-build"
        echo "  or set SHELF_SIDECAR_IMAGE / SHELF_IRIS_IMAGE in .env to images built from this repo"
        exit 1
    fi
    # The bind-mounted data dirs, made here so docker does not make them as root; they must be the containers' user's
    mkdir -p "$d/data/segments"
    for p in "$d/data" "$d/data/segments"; do
        [[ "$(stat -c %u:%g "$p")" == "$SHELF_UID:$SHELF_GID" ]] || {
            echo "$p belongs to $(stat -c %u:%g "$p"), not to $SHELF_UID:$SHELF_GID, the user the containers run as:"
            echo "chown -R $SHELF_UID:$SHELF_GID $p (or remove it), then start again"; exit 1; }
    done
    just shelf-live-stop-streamer
    shelf_compose up -d --no-build redis go2rtc
    # go2rtc reads go2rtc-live.yaml only at start; restart it, before the sidecar connects, only if it lacks the stream
    sleep 2
    curl -sf "localhost:$SHELF_GO2RTC_API_PORT/api/streams" | grep -q "\"raw-$cam\"" || shelf_compose restart go2rtc
    # Re-encoded, not copied (see ppe-live-sim): H.264 baseline, no audio, a keyframe every second, the source's
    # bitrate. Restarted if it dies; each start is logged with its wall time (the video's zero).
    fps="$(ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate -of csv=p=0 "$video")"
    gop="$(python3 -c "print(round(eval('$fps')))")"
    kbps="$(ffprobe -v error -select_streams v:0 -show_entries stream=bit_rate -of csv=p=0 "$video" | awk '{print int($1/1000)}')"
    setsid bash -c 'while true; do
        echo "START $(date +%s.%N) $2" >> "$1/streamer.log"
        ffmpeg -hide_banner -loglevel warning -re -stream_loop -1 -i "$2" -an \
            -c:v libx264 -preset veryfast -tune zerolatency -profile:v baseline -pix_fmt yuv420p \
            -g "$3" -b:v "${4}k" -maxrate "${4}k" -bufsize "$((2 * $4))k" \
            -max_muxing_queue_size 1024 -flush_packets 1 -rtsp_transport tcp -f rtsp \
            "$5" >> "$1/streamer.log" 2>&1
        echo "EXIT $(date +%s.%N) code=$?" >> "$1/streamer.log"
        sleep 2
    done' _ "$d" "$video" "$gop" "$kbps" "rtsp://localhost:$SHELF_GO2RTC_RTSP_PORT/raw-$cam" > /dev/null 2>&1 < /dev/null &
    echo $! > "$d/streamer.pid"
    sleep 1
    SIDECAR_CONFIG="$cfg" shelf_compose up -d --no-build deepstream-sidecar iris
    printf 'VIDEO=%s\nCAM=%s\nCONFIG=%s\n' "$video" "$cam" "$cfg" > "$d/current.env"
    ip="$(hostname -I | awk '{print $1}')"
    echo "streamer started (pid $!): $video -> $cam (${fps} fps, GOP $gop, ${kbps} kbit/s)"
    echo "dashboard (alerts): http://$ip:$SHELF_IRIS_PORT/#/iris/dashboard"
    echo "video (MJPEG):      http://$ip:$SHELF_IRIS_PORT/stream/$cam"

# Stop the shelf live streamer only (the stack keeps running)
shelf-live-stop-streamer:
    #!/usr/bin/env bash
    source scripts/shelf_live_env.sh
    pidf="$SHELF_LIVE_DIR/streamer.pid"
    if [[ -f $pidf ]]; then kill -TERM -- "-$(cat $pidf)" 2>/dev/null && echo "streamer stopped"; rm -f $pidf; fi

# Video, containers, streamer and the sidecar's last fps / shelf alert lines for the shelf demo
shelf-live-status:
    #!/usr/bin/env bash
    source scripts/shelf_live_env.sh
    cat "$SHELF_LIVE_DIR/current.env" 2>/dev/null || echo "no video started"
    shelf_compose ps
    pidf="$SHELF_LIVE_DIR/streamer.pid"
    if [[ -f $pidf ]] && kill -0 "$(cat $pidf)" 2>/dev/null; then echo "streamer RUNNING (pid $(cat $pidf))"
    else echo "streamer STOPPED"; fi
    shelf_compose logs --no-log-prefix --tail 400 deepstream-sidecar 2>/dev/null \
        | grep -E "\] [0-9.]+ fps|shelf (raised|cleared)|source-stall" | tail -n 6

# Follow the shelf demo's sidecar log (svc=iris for the app)
shelf-live-logs svc="deepstream-sidecar":
    #!/usr/bin/env bash
    source scripts/shelf_live_env.sh
    shelf_compose logs -f --tail 100 {{svc}}

# Stop what shelf-live-sim started: containers, volumes, streamer (keeps scratch/shelf-live/ and the images)
shelf-live-down: shelf-live-stop-streamer
    #!/usr/bin/env bash
    source scripts/shelf_live_env.sh
    shelf_compose down -v
    rm -f "$SHELF_LIVE_DIR/current.env"


# ── Safety: proximity, zones, PPE, fall and fire in one tool (products/safety/README.md) ─────────────────────────
# Its own compose project, ports and data dir (scripts/safety_env.sh, SAFETY_* in .env.example); each feature is a
# role sidecar, several per camera, started/restarted/stopped by tools/safety_controller.py from the cameras and the
# UI's toggles. GPU: one sidecar per feature in use -- check `nvidia-smi` before starting it on a shared GPU.

# Build the safety stack's images once (the sidecar image takes long; or set SAFETY_SIDECAR_IMAGE to an existing one)
[group('safety')]
safety-build:
    #!/usr/bin/env bash
    set -euo pipefail
    source scripts/safety_env.sh
    safety_compose build iris safety-zones

# Start the safety stack: redis, minio, go2rtc, the app, the fall service and the role sidecars the cameras need
[group('safety')]
safety-up:
    #!/usr/bin/env bash
    set -euo pipefail
    source scripts/safety_env.sh
    just safety-models
    missing=()
    docker image inspect "$SAFETY_SIDECAR_IMAGE" >/dev/null 2>&1 || missing+=("$SAFETY_SIDECAR_IMAGE")
    docker image inspect "$SAFETY_IRIS_IMAGE" >/dev/null 2>&1 || missing+=("$SAFETY_IRIS_IMAGE")
    if (( ${#missing[@]} )); then
        echo "missing docker image(s): ${missing[*]} -- build them with: just safety-build"
        echo "  (or set SAFETY_SIDECAR_IMAGE in .env to a sidecar image built from perception/pipeline/Dockerfile)"
        exit 1
    fi
    # The bind-mounted data dir, made here so docker does not make it as root
    mkdir -p "$SAFETY_DIR/data/safety/generated" "$SAFETY_DIR/data/alert_svr" "$SAFETY_DIR/fall"
    safety_controller render
    safety_compose up -d --no-build redis go2rtc iris
    just safety-fall-restart
    safety_controller apply
    just safety-controller-restart
    ip="$(hostname -I | awk '{print $1}')"
    echo "safety app: http://$ip:$SAFETY_IRIS_PORT   (demo clips: just safety-demo)"

# Stop the safety stack (containers, fall service, controller, demo streamers); its data dir is kept
[group('safety')]
safety-down: safety-demo-stop
    #!/usr/bin/env bash
    set -uo pipefail
    source scripts/safety_env.sh
    for name in controller fall; do
        safety_stop "$SAFETY_DIR/$name.pid" && echo "stopped the $name"
    done
    safety_compose down

# Containers, controller, fall service and demo streamers of the safety stack
[group('safety')]
safety-status:
    #!/usr/bin/env bash
    source scripts/safety_env.sh
    safety_compose ps
    safety_controller render --dry-run
    for name in controller fall; do
        if safety_pid "$SAFETY_DIR/$name.pid" > /dev/null; then echo "$name RUNNING"; else echo "$name STOPPED"; fi
    done
    ls "$SAFETY_DIR"/demo-*.pid 2>/dev/null | sed 's#.*/demo-\(.*\)\.pid#streamer \1#'

# Apply the cameras' toggles now (what the UI's Apply button asks the running controller to do)
[group('safety')]
safety-apply:
    #!/usr/bin/env bash
    set -euo pipefail
    source scripts/safety_env.sh
    safety_controller apply

# (Re)start the controller that applies the UI's toggles (in the background; log: $SAFETY_DIR/controller.log)
[group('safety')]
safety-controller-restart:
    #!/usr/bin/env bash
    set -euo pipefail
    source scripts/safety_env.sh
    pidf="$SAFETY_DIR/controller.pid"
    safety_stop "$pidf" || true
    setsid bash -c 'echo $$ > "$1"; exec >> "$2" 2>&1; source scripts/safety_env.sh; safety_controller watch' \
        _ "$pidf" "$SAFETY_DIR/controller.log" < /dev/null &
    echo "controller watching iris:ctrl:safety_apply (log: $SAFETY_DIR/controller.log)"

# (Re)start the fall service with the fall role's cameras, or stop it when no camera has fall on
[group('safety')]
safety-fall-restart:
    #!/usr/bin/env bash
    set -euo pipefail
    source scripts/safety_env.sh
    pidf="$SAFETY_DIR/fall.pid"
    safety_stop "$pidf" || true
    cfg="$SAFETY_DIR/data/safety/generated/fall_detection.yaml"
    if grep -q '^cameras: \[\]' "$cfg"; then echo "no camera has fall on: the fall service stays stopped"; exit 0; fi
    (cd products/fall_detection && env -u VIRTUAL_ENV uv sync -q --locked)
    cd products/fall_detection
    export FALL_DETECTION_CONFIG="$cfg" FALL_DETECTION_DATA_DIR="$SAFETY_DIR/fall" \
        REDIS_URL="redis://localhost:$SAFETY_REDIS_PORT/0" FALL_BACKEND_PORT="$SAFETY_FALL_PORT"
    setsid bash -c 'echo $$ > "$1"; exec nohup .venv/bin/python main.py >> "$2" 2>&1 < /dev/null' \
        _ "$pidf" "$SAFETY_DIR/fall.log" &
    for _ in $(seq 1 120); do curl -sf -o /dev/null "http://localhost:$SAFETY_FALL_PORT/status" && break; sleep 0.5; done
    curl -sf -o /dev/null "http://localhost:$SAFETY_FALL_PORT/status" \
        || { echo "the fall service did not come up -- see $SAFETY_DIR/fall.log" >&2; exit 1; }
    echo "fall service on :$SAFETY_FALL_PORT (log: $SAFETY_DIR/fall.log)"

# Loop each demo camera's clip into go2rtc as if it were a live camera (cameras with `clip:`; SAFETY_DEMO_CLIPS)
[group('safety')]
safety-demo: safety-demo-stop
    #!/usr/bin/env bash
    set -euo pipefail
    source scripts/safety_env.sh
    # Re-encoded like an IP camera stream (H.264 baseline, no B-frames, a keyframe every second, no audio): a copied
    # stream with B-frames made the sidecar decode corrupt frames and stall (see ppe-live-sim).
    safety_controller clips | while IFS=$'\t' read -r cam clip; do
        [[ -f "$clip" ]] || { echo "no clip for $cam at $clip (SAFETY_DEMO_CLIPS)"; continue; }
        setsid bash -c 'echo $$ > "$1"; while true; do
            ffmpeg -hide_banner -loglevel warning -re -stream_loop -1 -i "$2" -an -c:v libx264 -preset veryfast \
                -tune zerolatency -profile:v baseline -pix_fmt yuv420p -g 25 -f rtsp -rtsp_transport tcp "$3" \
                >> "$4" 2>&1
            sleep 2
        done' _ "$SAFETY_DIR/demo-$cam.pid" "$clip" "rtsp://localhost:$SAFETY_GO2RTC_RTSP_PORT/raw-$cam" \
            "$SAFETY_DIR/demo-$cam.log" < /dev/null > /dev/null 2>&1 &
        echo "streaming $clip -> raw-$cam"
    done

# Stop the demo streamers (the stack keeps running)
[group('safety')]
safety-demo-stop:
    #!/usr/bin/env bash
    source scripts/safety_env.sh
    for pidf in "$SAFETY_DIR"/demo-*.pid; do
        [[ -f $pidf ]] || continue
        safety_stop "$pidf"
    done

# Put the models the stack loads (safety.yaml `models`) in this checkout: from SAFETY_DVC_STORE, else dvc pull
[group('safety')]
safety-models:
    #!/usr/bin/env bash
    set -euo pipefail
    source scripts/safety_env.sh
    pull=()
    targets="$(safety_controller models)"          # a failure stops here (set -e), not as "no models"
    [[ -n $targets ]] || { echo "products/safety/safety.yaml lists no models" >&2; exit 1; }
    for t in $targets; do
        out="${t%.dvc}"
        # in place = the size its pointer records: a model left from an older pointer, or cut short, is replaced
        size="$(sed -n 's/^  size: //p' "$t" | head -1)"
        [[ -s $out && ( -z $size || $(stat -c %s "$out") == "$size" ) ]] && continue
        md5="$(sed -n 's/^- md5: //p' "$t" | head -1)"
        obj="$SAFETY_DVC_STORE/files/md5/${md5:0:2}/${md5:2}"
        if [[ -f $obj ]]; then cp --reflink=auto "$obj" "$out"; echo "copied $out"; else pull+=("$t"); fi
    done
    if (( ${#pull[@]} )); then
        command -v dvc >/dev/null || { echo "dvc is not installed (uv tool install 'dvc[s3,ssh]==3.67.1')" >&2; exit 1; }
        dvc pull "${pull[@]}"
    fi
    echo "the safety stack's models are in place"

# Follow the stack's logs: all, or iris | proximity | zones | ppe | fall | fire | controller | fall-service | demo
[group('safety')]
safety-logs what="":
    #!/usr/bin/env bash
    set -uo pipefail
    source scripts/safety_env.sh
    case "{{what}}" in
        "")
            safety_compose logs -f --tail 30 &
            trap 'kill $! 2>/dev/null' EXIT
            tail -F "$SAFETY_DIR/controller.log" "$SAFETY_DIR/fall.log" 2>/dev/null ;;
        controller) tail -F "$SAFETY_DIR/controller.log" ;;
        fall-service) tail -F "$SAFETY_DIR/fall.log" ;;
        demo) tail -F "$SAFETY_DIR"/demo-*.log ;;
        proximity|zones|ppe|fall|fire) safety_compose logs -f --tail 100 "safety-{{what}}" ;;
        *) safety_compose logs -f --tail 100 "{{what}}" ;;
    esac

# GPU memory and SM use of each role sidecar and the fall service (a 5 s sample): the demo's per-feature measure
[group('safety')]
safety-gpu:
    #!/usr/bin/env bash
    set -uo pipefail
    source scripts/safety_env.sh
    declare -A mem sm n who
    while IFS=', ' read -r pid mb; do mem[$pid]=$mb; done \
        < <(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits)
    while read -r _gpu pid _type u _rest; do
        [[ $pid =~ ^[0-9]+$ ]] || continue
        [[ $u =~ ^[0-9]+$ ]] || u=0
        sm[$pid]=$(( ${sm[$pid]:-0} + u )); n[$pid]=$(( ${n[$pid]:-0} + 1 ))
    done < <(nvidia-smi pmon -c 5 -s u)
    for c in $(safety_compose ps -q); do
        name="$(docker inspect -f '{{{{.Name}}' "$c")"
        for pid in $(docker top "$c" -eo pid | tail -n +2); do who[$pid]="${name#/}"; done
    done
    if fall=$(safety_pid "$SAFETY_DIR/fall.pid"); then who[$fall]="fall service (host)"; fi
    printf '%-36s %8s %5s\n' process "GPU MiB" "SM %"
    for pid in "${!mem[@]}"; do
        [[ -n ${who[$pid]:-} ]] || continue
        printf '%-36s %8s %5s\n' "${who[$pid]}" "${mem[$pid]}" "$(( ${n[$pid]:-0} ? ${sm[$pid]:-0} / ${n[$pid]} : 0 ))"
    done | sort
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader \
        | awk -F', ' '{print "whole GPU (every stack on it): " $1 " of " $2 ", " $3 " busy"}'

# Send a test message to the Telegram chat saved in Settings (the stack must be up)
[group('safety')]
safety-telegram-test:
    #!/usr/bin/env bash
    set -uo pipefail
    source scripts/safety_env.sh
    out="$(curl -s -X POST "http://localhost:$SAFETY_IRIS_PORT/api/safety/telegram/test")" \
        || { echo "the safety app does not answer on :$SAFETY_IRIS_PORT (just safety-up)" >&2; exit 1; }
    if grep -q '"ok": *true' <<< "$out"; then echo "test message sent: check the chat"; else echo "not sent: $out" >&2; exit 1; fi

# The safety product's tests: its backend and controller (tests/test_safety_*.py) and its console (vitest)
[group('safety')]
safety-test:
    env -u VIRTUAL_ENV uv run --quiet --locked --isolated --only-group test python -m pytest -q tests/test_safety_*.py
    cd {{frontend}} && npm test -w @iris/safety
