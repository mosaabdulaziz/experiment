"""Attendance service BFF — API + live websocket for the attendance page in
iris-frontend. Not a general-purpose API: exactly what that one page needs.
Throwaway/prototype quality (see grilling session, 2026-08-26) — no auth,
internal use only, until the domain model is proven and the supervisor
answers who actually sees this.
"""
import asyncio
import json
import logging
import os
import shutil
import subprocess
from contextlib import asynccontextmanager
from datetime import date, datetime, time
from pathlib import Path
from uuid import uuid4

import redis
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from cameras import CAM_DURATIONS, CAM_FILENAMES, load_stream_starts
from acl import camera_clock, live_watch
from consumer import ConsumerThread
from domain import DayAttendance, Sighting
from productivity import DEFAULT_POLICY, monitoring_snapshot
from sessions import DEFAULT_ATTENDANCE_POLICY, derive_month
from storage import (clear_violations_for_day, connect, compute_daily_attendance, create_employee,
                     delete_employee, next_ids, recent_violations, update_employee)

FACE_GALLERY_KEY = "attendance:face:gallery"

# Without this, logger.info(...) calls throughout this service (including
# the consumer thread's "consumer started", "flushed N sighting(s)", and
# "resetting the sighting table" lines) are silently dropped -- Python's
# logging module has no configured handler otherwise, so only logger.error/
# exception (via the last-resort WARNING+ handler) ever reached the log file.
# Found live 2026-08-31 while debugging a "table not matching live cam"
# report: the log had zero consumer-thread output at all, even though the
# consumer was demonstrably running (it had written real rows) -- every INFO
# line explaining what it was doing had been invisible the whole time.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

logger = logging.getLogger("attendance.server")

DB_PATH = "data/attendance.db"
# LOCAL-ONLY (2026-09-10): env-overridable so this checkout ("kw") can point
# at its own redis (docker-compose.yml's REDIS_HOST_PORT, not the "repo"
# checkout's hardcoded 6379) without colliding. justfile exports REDIS_URL
# when it starts this process; unset falls back to the old default.
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
# The GPU image photo registration shells out to -- the same project-scoped
# tag docker-compose.yml builds for this checkout
# (${COMPOSE_PROJECT_NAME:-surveillance-camera-sys}-deepstream-sidecar), so
# enrollment never runs in another checkout's image. COMPOSE_PROJECT_NAME
# comes from .env, exported by the justfile's _start_host_processes.
SIDECAR_IMAGE = f"{os.environ.get('COMPOSE_PROJECT_NAME') or 'surveillance-camera-sys'}-deepstream-sidecar:latest"
# 2026-09-12 (issue 1): removed. This used to pin every default "what day is
# it" query to the CHIRLA spike's fake 2023-06-08 recording date instead of
# real today -- correct only as long as every sighting was also dated for
# that same fake day (see consumer.py's own removal of RECORDING_START).
# Every former SPIKE_DAY reference below now calls date.today() directly
# (not a module-level constant) so it stays correct across a real midnight,
# which a fixed value evaluated once at import time would not.

REPO_ROOT = Path(__file__).resolve().parents[2]   # this folder is products/attendance/
FACE_PHOTOS_DIR = REPO_ROOT / "data" / "face_photos"
UPLOADS_DIR = REPO_ROOT / "data" / "uploads"


def _attendance_snapshot(day: date) -> list[dict]:
    conn = connect(DB_PATH)
    employees = conn.execute(
        "SELECT id, name, gallery_rec_id, code, department, display_name_ar, on_leave "
        "FROM employee WHERE active = 1 ORDER BY name"
    ).fetchall()
    result = []
    for emp in employees:
        record = compute_daily_attendance(conn, emp["id"], day)
        # Status precedence (dashboard, 2026-09-01): actually sighted wins even
        # over a leave flag (they showed up); otherwise the leave flag wins
        # over plain absence.
        status = "present" if record.present else ("leave" if emp["on_leave"] else "absent")
        result.append({
            "employee_id": emp["id"],
            "name": emp["name"],
            "photo_url": f"/api/employees/{emp['id']}/photo",
            "day": record.day.isoformat(),
            "first_arrival": record.first_arrival.isoformat() if record.first_arrival else None,
            "last_departure": record.last_departure.isoformat() if record.last_departure else None,
            "present": record.present,
            # Dashboard fields (standalone attendance-frontend) -- additive,
            # the surveillance app's own attendance page ignores them.
            "code": emp["code"],
            "department": emp["department"],
            "display_name_ar": emp["display_name_ar"],
            "on_leave": bool(emp["on_leave"]),
            "status": status,
        })
    return result


class ConnectionManager:
    """Mirrors surveillance-camera-sys's app/server.py ConnectionManager
    (broadcast to every connected client) -- same pattern, much smaller
    surface since this only ever sends one message shape."""

    def __init__(self):
        self._clients: list[WebSocket] = []
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._clients.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self._clients:
            self._clients.remove(ws)

    def broadcast_threadsafe(self, message: dict) -> None:
        """Called from the consumer's background thread -- hands the send
        off to the FastAPI event loop instead of awaiting directly."""
        if self._loop is None:
            return
        asyncio.run_coroutine_threadsafe(self._broadcast(message), self._loop)

    async def _broadcast(self, message: dict) -> None:
        dead = []
        for ws in self._clients:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()


consumer: ConsumerThread | None = None  # set in lifespan; monitoring reads its engine/scene


def _monitoring_payload() -> dict:
    """Productivity monitoring snapshot (2026-09-01): per-employee desk
    states + counters, the latest per-camera scene for the overlays (with
    names resolved), zones and policy -- one message shape for both the
    REST endpoint and the websocket."""
    conn = connect(DB_PATH)
    policy = consumer.policy.load() if consumer else dict(DEFAULT_POLICY)
    zones = consumer.zones.load() if consumer else {}
    engine = consumer.engine if consumer else None
    names = {r["id"]: (r["display_name_ar"] or r["name"])
             for r in conn.execute("SELECT id, name, display_name_ar FROM employee")}
    scene = {}
    for cam_id, entries in (consumer.scene if consumer else {}).items():
        scene[cam_id] = [dict(e, name=names.get(e["employee_id"]),
                              credited_name=names.get(e.get("credited_as"))) for e in entries]
    # Restricted-zone violations (2026-09-02), on the same page as attendance
    # per Q6 (round 2) until the UI person says otherwise -- names resolved
    # the same way as scene above; a NULL employee_id (unidentified person)
    # stays None, the frontend renders that case distinctly.
    violations = [dict(v, name=names.get(v["employee_id"])) for v in recent_violations(conn, limit=50)]
    today = date.today()
    return {
        "type": "monitoring_snapshot",
        "day": today.isoformat(),
        "records": monitoring_snapshot(conn, today, policy, engine),
        "scene": scene,
        "zones": zones,
        "policy": policy,
        "violations": violations,
    }


def _on_flush(flushed: list[Sighting]) -> None:
    """Consumer thread callback: something changed, push a fresh snapshot.
    Full resend rather than incremental patching -- the dataset is tiny
    (a handful of employees), so simplicity wins over a diffing protocol."""
    today = date.today()
    manager.broadcast_threadsafe({
        "type": "attendance_snapshot",
        "day": today.isoformat(),
        "records": _attendance_snapshot(today),
    })
    try:
        manager.broadcast_threadsafe(_monitoring_payload())
    except Exception:
        logger.exception("monitoring snapshot failed")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global consumer
    manager.bind_loop(asyncio.get_event_loop())
    consumer = ConsumerThread(db_path=DB_PATH, redis_url=REDIS_URL, on_flush=_on_flush)
    consumer.start()
    logger.info("consumer thread started")
    yield


app = FastAPI(title="Attendance service (validation)", lifespan=lifespan)


@app.get("/api/employees")
def list_employees():
    conn = connect(DB_PATH)
    rows = conn.execute("SELECT id, name, active FROM employee WHERE active = 1 ORDER BY name").fetchall()
    return [
        {"id": r["id"], "name": r["name"], "active": bool(r["active"]),
         "photo_url": f"/api/employees/{r['id']}/photo"}
        for r in rows
    ]


# ── Admin-driven enrollment (2026-09-02) ────────────────────────────────────
# Separate from list_employees() above (which the existing dashboard already
# depends on, active-only, minimal fields) -- the admin Employees page needs
# everyone, active or not, with role/department, so it gets its own read
# endpoint rather than risking that one's existing contract.

@app.get("/api/admin/employees")
def admin_list_employees():
    conn = connect(DB_PATH)
    rows = conn.execute(
        "SELECT id, name, role, department, email, code, gallery_rec_id, active, on_leave FROM employee ORDER BY name"
    ).fetchall()
    return [
        {"id": r["id"], "name": r["name"], "role": r["role"], "department": r["department"],
         "email": r["email"], "code": r["code"],
         "gallery_rec_id": r["gallery_rec_id"], "active": bool(r["active"]),
         "on_leave": bool(r["on_leave"]),  # 2026-09-14: leave toggle moved to the Employees page
         "photo_url": f"/api/employees/{r['id']}/photo"}
        for r in rows
    ]


@app.post("/api/admin/employees")
def admin_create_employee(payload: dict):
    """Row only -- no face gallery entry yet (needs a photo, see
    /photos below). name is the only required field; role/department can be
    filled in later via PATCH."""
    name = (payload.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")
    conn = connect(DB_PATH)
    employee_id, gallery_rec_id = next_ids(conn)
    create_employee(conn, employee_id, name, gallery_rec_id,
                    role=payload.get("role"), department=payload.get("department"),
                    email=payload.get("email"), code=payload.get("code"))
    _on_flush([])
    return {"id": employee_id, "name": name, "gallery_rec_id": gallery_rec_id,
           "role": payload.get("role"), "department": payload.get("department"),
           "email": payload.get("email"), "code": payload.get("code"), "active": True}


@app.patch("/api/admin/employees/{employee_id}")
def admin_update_employee(employee_id: str, payload: dict):
    conn = connect(DB_PATH)
    fields = {k: payload[k] for k in ("name", "role", "department", "active", "email", "code") if k in payload}
    if not update_employee(conn, employee_id, **fields):
        raise HTTPException(status_code=404, detail="unknown employee")
    _on_flush([])
    row = conn.execute("SELECT id, name, role, department, email, code, active FROM employee WHERE id = ?",
                       (employee_id,)).fetchone()
    return dict(row, active=bool(row["active"]))  # sqlite int -> real JSON boolean


@app.delete("/api/admin/employees/{employee_id}")
def admin_delete_employee(employee_id: str):
    """Permanently remove an employee (2026-09-06, admin Employees page
    "remove employee" button) -- not the same as the existing تعطيل toggle,
    which only flips `active` and keeps history. This purges the DB row and
    everything referencing it (see storage.delete_employee), then best-effort
    cleans up the two things that live outside this database: the enrolled
    face embedding in the live recognition gallery and the photo file(s) on
    disk. Neither of those failing rolls back the DB delete -- the admin
    asked to remove this employee, and stale Redis/disk state can't act like
    a live person once the employee row (and the ACL/attendance logic that
    reads it) is gone."""
    conn = connect(DB_PATH)
    gallery_rec_id = delete_employee(conn, employee_id)
    if gallery_rec_id is None:
        raise HTTPException(status_code=404, detail="unknown employee")

    try:
        redis.Redis.from_url(REDIS_URL).hdel(FACE_GALLERY_KEY, gallery_rec_id)
    except Exception:
        logger.exception("Failed to remove %s from the face gallery", gallery_rec_id)

    (FACE_PHOTOS_DIR / f"{gallery_rec_id}.jpg").unlink(missing_ok=True)
    multi_dir = FACE_PHOTOS_DIR / "multi" / gallery_rec_id
    if multi_dir.is_dir():
        shutil.rmtree(multi_dir, ignore_errors=True)

    _on_flush([])
    return {"id": employee_id, "deleted": True}


def _redis_compose_network() -> str:
    """Resolve the docker network the `redis` compose service is actually on,
    so the registration subprocess (docker run) can reach it by the `redis`
    hostname. Looked up dynamically via the compose service label rather than
    a hardcoded "<project>_default" name -- that already went stale once,
    silently, when the compose project was renamed kw -> km (found live
    2026-09-09: every photo upload failed with "network ... not found" before
    docker run even started the container). justfile's `_register_people`
    hit the identical problem earlier and fixed it the same way (dynamic
    `docker inspect`, see its own comment) -- this mirrors that fix so the
    two entry points into the same GPU image can't drift apart again."""
    cid = subprocess.run(
        ["docker", "ps", "-q", "--filter", "label=com.docker.compose.service=redis"],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    if not cid:
        raise RuntimeError("no running `redis` container found (is the stack up?)")
    fmt = "{{range $k, $v := .NetworkSettings.Networks}}{{$k}}{{end}}"
    net = subprocess.run(
        ["docker", "inspect", "--format", fmt, cid[0]],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if not net:
        raise RuntimeError(f"redis container {cid[0]} is not attached to any network")
    return net


def _run_registration_subprocess(photo_path: Path, face_id: str, name: str) -> dict:
    """Shells out to the GPU sidecar image to validate + register one photo
    -- see register_employee_photo.py's module docstring for why this can't
    happen in this process. `photo_path` must already be somewhere under
    FACE_PHOTOS_DIR (mounted read-only into the container as /photos)."""
    try:
        network = _redis_compose_network()
    except (subprocess.CalledProcessError, RuntimeError) as exc:
        return {"ok": False, "reason": "subprocess_failed", "stderr": str(exc)}
    rel = photo_path.relative_to(FACE_PHOTOS_DIR)
    cmd = [
        "docker", "run", "--rm", "--device", "nvidia.com/gpu=all",
        "--network", network,
        "-v", f"{FACE_PHOTOS_DIR}:/photos:ro",
        # the script imports the pipeline's face code from perception/pipeline/
        "-v", f"{REPO_ROOT / 'perception'}:/workspace/iris/perception:ro",
        "-v", f"{Path(__file__).parent / 'register_employee_photo.py'}:"
              "/workspace/iris/products/attendance/register_employee_photo.py:ro",
        "-w", "/workspace/iris", "--entrypoint", "python3",
        SIDECAR_IMAGE,
        "products/attendance/register_employee_photo.py",
        "--photo", f"/photos/{rel}", "--face-id", face_id, "--name", name,
        "--redis-url", "redis://redis:6379/0", "--gallery-key", "attendance:face:gallery", "--gpu",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return {"ok": False, "reason": "timeout"}
    try:
        # The script prints exactly one JSON line; take the last non-empty
        # line so a stray print from a library upstream can't break parsing.
        return json.loads([l for l in proc.stdout.splitlines() if l.strip()][-1])
    except (IndexError, json.JSONDecodeError):
        return {"ok": False, "reason": "subprocess_failed", "stderr": proc.stderr[-2000:]}


@app.post("/api/admin/employees/{employee_id}/photos")
async def admin_upload_photos(employee_id: str, files: list[UploadFile] = File(...)):
    """One or more photos for an existing employee row. The FIRST one that
    passes validation becomes the primary enrollment photo
    (FACE_PHOTOS_DIR/<gallery_rec_id>.jpg); every one after that is an extra
    angle (FACE_PHOTOS_DIR/multi/<gallery_rec_id>/*.jpg) -- same on-disk
    layout register_extra_embeddings.py already expects, so nothing else
    downstream needs to change. A rejected photo is deleted, never left on
    disk pretending to be enrolled."""
    conn = connect(DB_PATH)
    row = conn.execute("SELECT id, name, gallery_rec_id FROM employee WHERE id = ?",
                       (employee_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="unknown employee")
    gallery_rec_id, name = row["gallery_rec_id"], row["name"]

    results = []
    for upload in files:
        primary_path = FACE_PHOTOS_DIR / f"{gallery_rec_id}.jpg"
        if primary_path.is_file():
            dest = FACE_PHOTOS_DIR / "multi" / gallery_rec_id / f"{uuid4().hex}.jpg"
        else:
            dest = primary_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(await upload.read())

        result = _run_registration_subprocess(dest, gallery_rec_id, name)
        if not result.get("ok"):
            dest.unlink(missing_ok=True)  # never leave a rejected photo on disk
        results.append({"filename": upload.filename, **result})

    _on_flush([])
    return {"employee_id": employee_id, "results": results}


@app.patch("/api/employees/{employee_id}/leave")
def set_leave(employee_id: str, payload: dict):
    """Toggle the manual في-إجازة flag (dashboard, 2026-09-01). Pushes a fresh
    websocket snapshot immediately so every open dashboard reflects it."""
    conn = connect(DB_PATH)
    row = conn.execute("SELECT id FROM employee WHERE id = ?", (employee_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="unknown employee")
    on_leave = 1 if payload.get("on_leave") else 0
    conn.execute("UPDATE employee SET on_leave = ? WHERE id = ?", (on_leave, employee_id))
    conn.commit()
    _on_flush([])
    return {"id": employee_id, "on_leave": bool(on_leave)}


# ── Productivity monitoring (2026-09-01) ────────────────────────────────────

@app.get("/api/monitoring")
def monitoring():
    return _monitoring_payload()


@app.delete("/api/monitoring/violations")
def clear_violations(day: date):
    """Delete one day's restricted-zone violation rows (2026-09-13, the
    dashboard's "clear log" button -- the day shown in its date picker).
    Attendance (sightings), desk intervals and employees are untouched.
    Clearing today also tells the live engine to forget its open
    violations, so one still in progress is logged again as a new row."""
    removed = clear_violations_for_day(connect(DB_PATH), day)
    if day == date.today() and consumer is not None and consumer.violations is not None:
        consumer.violations.request_forget()
    logger.info("cleared %d violation row(s) for %s", removed, day)
    _on_flush([])
    return {"day": day.isoformat(), "removed": removed}


@app.get("/api/monitoring/policy")
def get_policy():
    return consumer.policy.load() if consumer else dict(DEFAULT_POLICY)


@app.put("/api/monitoring/policy")
def put_policy(payload: dict):
    """Update the working/not-working policy knobs (hot-reloaded by the
    engine within a cycle). Only known keys are accepted; numbers only."""
    if consumer is None:
        raise HTTPException(status_code=503, detail="consumer not started")
    current = dict(consumer.policy.load())
    for key in ("debounce_seconds", "away_grace_seconds", "phone_sustained_seconds", "seated_max_aspect"):
        if key in payload:
            try:
                current[key] = max(0.0, float(payload[key]))
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail=f"{key} must be a number")
    consumer.policy.save(current)
    _on_flush([])
    return current


@app.get("/api/monitoring/zones")
def get_zones():
    return consumer.zones.load() if consumer else {}


@app.put("/api/monitoring/zones/{cam_id}")
def put_zones(cam_id: str, payload: list[dict]):
    """Replace one camera's zones: [{id, name, points: [[x,y],...], type,
    allowed_roles}] in normalized 0..1 coordinates. Applied by the engine on
    its next cycle. type defaults to "seat" (2026-09-02: zones now come in
    two kinds -- "seat" feeds desk-monitoring, "restricted" feeds violation
    detection with an allowed_roles list; same zone editor, same file, see
    zones.py)."""
    if consumer is None:
        raise HTTPException(status_code=503, detail="consumer not started")
    cleaned = []
    for z in payload:
        pts = z.get("points") or []
        if len(pts) < 3:
            continue
        zone_type = z.get("type") or "seat"
        if zone_type not in ("seat", "restricted"):
            raise HTTPException(status_code=400, detail="zone type must be 'seat' or 'restricted'")
        entry = {
            "id": str(z.get("id") or f"zone-{len(cleaned) + 1}"),
            "name": str(z.get("name") or f"مكتب {len(cleaned) + 1}"),
            "points": [[min(1.0, max(0.0, float(x))), min(1.0, max(0.0, float(y)))] for x, y in pts],
            "type": zone_type,
        }
        if zone_type == "restricted":
            roles = z.get("allowed_roles") or []
            if not isinstance(roles, list) or not all(isinstance(r, str) for r in roles):
                raise HTTPException(status_code=400, detail="allowed_roles must be a list of role names")
            entry["allowed_roles"] = roles
        cleaned.append(entry)
    zones = dict(consumer.zones.load())
    zones[cam_id] = cleaned
    consumer.zones.save(zones)
    _on_flush([])
    return cleaned


# ── Attendance v2 settings (2026-09-02) ─────────────────────────────────────
# A separate store from /api/monitoring/policy above -- attendance's own
# schedule/gap knobs, not the desk-monitoring feature's. The Settings PAGE in
# the frontend presents both together (Q5, round 2); the backends stay apart.

@app.get("/api/attendance/policy")
def get_attendance_policy():
    return consumer.attendance_policy.load() if consumer else dict(DEFAULT_ATTENDANCE_POLICY)


@app.put("/api/attendance/policy")
def put_attendance_policy(payload: dict):
    if consumer is None:
        raise HTTPException(status_code=503, detail="consumer not started")
    current = dict(consumer.attendance_policy.load())
    for key in ("work_start", "work_end", "departure_reveal_time"):
        if key in payload:
            try:
                time.fromisoformat(str(payload[key]))
            except ValueError:
                raise HTTPException(status_code=400, detail=f"{key} must be HH:MM")
            current[key] = str(payload[key])
    for key in ("late_grace_minutes", "early_leave_grace_minutes", "break_min_seconds",
                "exit_gap_seconds", "violation_debounce_seconds"):
        if key in payload:
            try:
                current[key] = max(0.0, float(payload[key]))
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail=f"{key} must be a number")
    if "camera_types" in payload:
        types = payload["camera_types"]
        if not isinstance(types, dict) or any(v not in ("office", "break_room") for v in types.values()):
            raise HTTPException(status_code=400,
                                detail="camera_types must be {cam_id: 'office' | 'break_room'}")
        current["camera_types"] = {str(k): v for k, v in types.items()}
    if "access_control" in payload:
        ac = payload["access_control"]
        if (not isinstance(ac, dict)
                or any(not isinstance(v, dict) or not isinstance(v.get("allowed_roles", []), list)
                       or not all(isinstance(r, str) for r in v.get("allowed_roles", []))
                       for v in ac.values())):
            raise HTTPException(status_code=400,
                                detail="access_control must be {cam_id: {'allowed_roles': [role, ...]}}")
        current["access_control"] = {str(k): {"allowed_roles": list(v.get("allowed_roles", []))}
                                     for k, v in ac.items()}
    consumer.attendance_policy.save(current)
    return current


def _day_row(d: DayAttendance) -> dict:
    return {
        "day": d.day.isoformat(),
        "sessions": [
            {
                "entry": s.entry.isoformat(),
                "exit": s.exit.isoformat() if s.exit else None,
                "breaks": [{"start": b.start.isoformat(), "end": b.end.isoformat()} for b in s.breaks],
            }
            for s in d.sessions
        ],
        # 2026-09-14: the monthly sheet shows one pair per day (first arrival,
        # last departure; None while the last session is still open) and the
        # span between them as its net hours.
        "first_arrival": d.sessions[0].entry.isoformat() if d.sessions else None,
        "last_departure": (d.sessions[-1].exit.isoformat()
                           if d.sessions and d.sessions[-1].exit else None),
        "span_seconds": round(d.span_seconds),
        "attendance_seconds": round(d.attendance_seconds),
        "break_seconds": round(d.break_seconds),
        "net_seconds": round(d.net_seconds),
        "overtime_seconds": round(d.overtime_seconds),
        "early_arrival": d.early_arrival,
        "late": d.late,
        "early_leave": d.early_leave,
        "absent": d.absent,
    }


@app.get("/api/employees/{employee_id}/history")
def employee_history(employee_id: str, month: str | None = None):
    """Per-employee month view (2026-09-02, /grilling round 2 Q1: its own
    page, not a slide-over): one row per day plus month summary tiles.
    `month` (YYYY-MM) defaults to the real current month (2026-09-12: no
    longer pinned to the old CHIRLA spike's fake date, see issue 1)."""
    conn = connect(DB_PATH)
    row = conn.execute("SELECT id, name, display_name_ar FROM employee WHERE id = ?",
                       (employee_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="unknown employee")
    if month:
        try:
            year_s, month_s = month.split("-")
            year, mon = int(year_s), int(month_s)
            if not 1 <= mon <= 12:
                raise ValueError
        except ValueError:
            raise HTTPException(status_code=400, detail="month must be YYYY-MM")
    else:
        today = date.today()
        year, mon = today.year, today.month

    policy = consumer.attendance_policy.load() if consumer else dict(DEFAULT_ATTENDANCE_POLICY)
    days = derive_month(conn, employee_id, year, mon, policy, datetime.now())
    return {
        "employee_id": employee_id,
        "name": row["name"],
        "display_name_ar": row["display_name_ar"],
        "month": f"{year:04d}-{mon:02d}",
        "days": [_day_row(d) for d in days],
        "summary": {
            "days_present": sum(1 for d in days if not d.absent),
            "times_late": sum(1 for d in days if d.late),
            "times_early_leave": sum(1 for d in days if d.early_leave),
            "total_overtime_seconds": round(sum(d.overtime_seconds for d in days)),
            "total_net_seconds": round(sum(d.net_seconds for d in days)),
        },
    }


@app.get("/api/employees/{employee_id}/photo")
def employee_photo(employee_id: str):
    """The enrollment photo — answers "who is Person-N in the video" directly,
    since it's the exact image whose embedding is in the gallery.

    Both 404s carry no-store too, not just the 200: an employee's avatar is
    routinely requested by the frontend before its photo finishes uploading
    (the row appears, or an upload is in flight) -- a 404 cached without an
    explicit directive risks a browser treating "no photo yet" as a settled
    fact and never re-requesting once the photo actually lands."""
    no_store = {"Cache-Control": "no-store"}
    conn = connect(DB_PATH)
    row = conn.execute("SELECT gallery_rec_id FROM employee WHERE id = ?", (employee_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="unknown employee", headers=no_store)
    path = FACE_PHOTOS_DIR / f"{row['gallery_rec_id']}.jpg"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="no photo on file for this employee", headers=no_store)
    return FileResponse(path, media_type="image/jpeg", headers=no_store)


def _clock_fields(cam_id: str) -> dict:
    """Camera clock (issue 1 Phase 2) for /api/recordings: (camera - server)
    in ms when the camera is sending its own clock over RTCP, else null ->
    the tile shows server time. Same offset the consumer stamps sightings
    with (acl.camera_clock), read once so the two fields can't disagree."""
    off = camera_clock.offset(cam_id)
    return {"clock_offset_ms": None if off is None else round(off * 1000),
            "clock_source": "server" if off is None else "camera"}


@app.get("/api/time")
def server_time():
    """The server's own current time, for the frontend's camera-tile clock.
    2026-09-12: every attendance-table time is written with THIS machine's
    clock (consumer.py's datetime.now()), but the tile clock used to tick on
    the VIEWER's machine clock (browser Date.now()) -- a different computer,
    so any drift between the two showed up as a constant offset between the
    tile and the table even though both were "real current time" (user-
    reported: a steady ~30s shift, while server-side measurements showed
    face-match-to-table-row latency of only ~1-2s). The frontend now uses
    this to correct its clock to the server's, so both read one clock."""
    return {"epoch_ms": datetime.now().timestamp() * 1000}


@app.get("/api/recordings")
def list_recordings():
    """Recording metadata for the live camera-tile overlay -- cam_id, the
    file's real duration, and (if currently streaming) the real time THIS
    camera's ffmpeg stream started.

    2026-09-12 (issue 1, Phase 1): the tile clock (useWallClock, in the
    frontend) now just shows real current time -- it no longer computes
    anything from recording_start/duration_seconds below, only checks that
    stream_started_at is present (so the clock still disappears when a tile
    goes offline, same as before). recording_start used to be a shared fake
    "all these files were recorded starting at this one moment" anchor
    (2023-06-08T12:29:08, the CHIRLA spike's simulated date) that, combined
    with duration_seconds, let the frontend fake "position inside a looping
    recording" -- meaningless for a real live camera with no loop, and the
    cause of the 2026-09-11 cam-20/21 clock-vs-table mismatch (see
    consumer.py's own history on this). Left in the response, still real
    (CAM_DURATIONS is each file's actual measured duration, not faked) in
    case a future seekable-playback feature wants it -- just nothing reads
    recording_start off this endpoint today, so its value stopped mattering
    and was left as-is rather than picking a new fake anchor for no reader.
    """
    # Test recordings (the test streamer's file) plus live cameras that are
    # actually delivering video (acl.live_watch, refreshed by the consumer
    # thread every cycle) -- 2026-09-13, so live tiles get a clock too.
    stream_starts = {**load_stream_starts(), **live_watch.snapshot()}
    return [
        {
            "cam_id": cam_id,
            "recording_start": "2023-06-08T12:29:08",
            # .get(), not [cam_id]: cam-1/cam-2/cam-3 (real live cameras)
            # deliberately have no CAM_DURATIONS entry (see cameras.py) --
            # None here is correct ("no fixed duration, it's live"), not
            # missing data. Frontend doesn't read this field for the tile
            # clock any more anyway (see useWallClock's own 2026-09-12 note).
            "duration_seconds": CAM_DURATIONS.get(cam_id),
            **_clock_fields(cam_id),
            # .astimezone() (no args) attaches the *correct* UTC offset to this
            # naive-local datetime without changing the instant it represents,
            # so the ISO string carries an explicit offset (e.g. "+03:00").
            # Without it, a bare "no-Z" ISO string gets parsed by the browser's
            # Date constructor as browser-LOCAL time, not server-local -- so a
            # viewer in a different timezone than the server got a live-tile
            # clock silently offset from the table by that exact difference
            # (found live 2026-08-31; the table itself was unaffected, since it
            # only ever re-displays a value the server already computed, not
            # arithmetic mixing this string with the browser's own Date.now()).
            "stream_started_at": stream_starts[cam_id].astimezone().isoformat() if cam_id in stream_starts else None,
        }
        for cam_id in CAM_FILENAMES
    ]


@app.get("/media/{cam_id}.mp4")
def camera_recording(cam_id: str):
    """The actual recorded file, served as a plain static file so the browser's
    native <video> element gets real seek/scrub -- deliberately decoupled from
    the live RTSP simulation, which has no seek capability at all."""
    filename = CAM_FILENAMES.get(cam_id)
    if filename is None:
        raise HTTPException(status_code=404, detail="unknown camera")
    h264_name = filename.replace(".avi", "_h264.mp4")
    # the clips live in data/uploads/attendance/ since the use-case layout (DVC); the flat folder is the old place
    path = next((c for c in (UPLOADS_DIR / "attendance" / h264_name, UPLOADS_DIR / h264_name) if c.is_file()), None)
    if path is None:
        raise HTTPException(status_code=404, detail="recording not found on disk")
    return FileResponse(path, media_type="video/mp4")


@app.get("/api/attendance")
def daily_attendance(day: str | None = None):
    """Attendance for every employee on `day` (YYYY-MM-DD). Defaults to the
    real current date (2026-09-12: no longer the old CHIRLA spike's fake
    recording date, see issue 1)."""
    target_day = date.fromisoformat(day) if day else date.today()
    return _attendance_snapshot(target_day)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await manager.connect(ws)
    # Send an immediate snapshot on connect, same shape as a live update --
    # mirrors surveillance's "history" message on the /ws endpoint.
    today = date.today()
    await ws.send_json({
        "type": "attendance_snapshot",
        "day": today.isoformat(),
        "records": _attendance_snapshot(today),
    })
    try:
        while True:
            await ws.receive_text()  # keepalive pings from the client; ignored
    except WebSocketDisconnect:
        manager.disconnect(ws)
