-- Attendance service storage — SQLite (matches this stack's existing
-- convention, see surveillance-camera-sys/app/events.py's data/events.db).
-- Provisional contract, agreed via /mattpocock-skills:grilling on 2026-08-26.

CREATE TABLE IF NOT EXISTS employee (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    gallery_rec_id  TEXT NOT NULL UNIQUE,  -- FK into the forked attendance gallery
    active          INTEGER NOT NULL DEFAULT 1,
    -- Dashboard fields (2026-09-01, standalone attendance-frontend): pure
    -- display/HR metadata -- recognition and the ACL never read these.
    -- display_name_ar is UI-only: the gallery + video overlay keep the Latin
    -- `name` (OpenCV's putText cannot shape Arabic glyphs), and the ACL's
    -- name-fallback matching depends on `name` agreeing with the gallery.
    code            TEXT,               -- الرقم الوظيفي (simulated)
    department      TEXT,               -- القسم (simulated)
    display_name_ar TEXT,               -- Arabic display name for the dashboard
    on_leave        INTEGER NOT NULL DEFAULT 0,  -- في إجازة (manual flag)
    -- Attendance v2 (2026-09-02, /grilling round 1 Q5): separate from
    -- department -- a restricted zone lists allowed roles, not departments.
    -- Freeform text, admin-assigned, same style as department. NULL is
    -- fail-closed: an employee with no role is never allowed into any
    -- restricted zone (storage.connect() migrates this column onto an
    -- already-existing employee table -- a fresh one gets it here directly).
    role            TEXT,
    -- Admin Employees page (2026-09-04, prompt.md UI pass): stored but not
    -- read by anything yet -- no login/notification feature exists to
    -- consume it. Same migrate-on-connect treatment as `role` above.
    email           TEXT
);

-- One-way marker for seed_demo_employees.py (2026-09-06). A plain "is
-- employee empty?" check can't tell a genuinely fresh checkout apart from an
-- admin who just deleted everyone on purpose -- both look like zero rows.
-- Found live: delete every employee, then `just down`/`just up` catches the
-- table empty and silently restores the 13 demo people, undoing the
-- deletion (and anything added before the table was next observed empty).
-- Once this row exists, the demo roster is never (re-)seeded again, no
-- matter how many rows `employee` currently has. storage._migrate()
-- backfills this row for a database that already has employees today, so
-- the very next connect() protects it before anyone deletes anything else.
CREATE TABLE IF NOT EXISTS seed_state (
    key       TEXT PRIMARY KEY,
    seeded_at TEXT NOT NULL
);

-- Persistent high-water mark for next_ids() (2026-09-06). Computing "N" from
-- the highest emp-N/person-N CURRENTLY in the employee table (the original
-- approach) silently reuses a number the moment its row is deleted: delete
-- emp-13 then create a new employee and it becomes emp-13/person-13 again --
-- same id, same face-recognition gallery slot, same history-table foreign
-- keys as whoever used to hold it. This table only ever counts up,
-- regardless of what employee currently contains (storage._migrate()
-- backfills it from today's data on first connect()).
CREATE TABLE IF NOT EXISTS id_sequence (
    key TEXT PRIMARY KEY,
    n   INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS sighting (
    id              TEXT PRIMARY KEY,
    employee_id     TEXT NOT NULL REFERENCES employee(id),
    camera          TEXT NOT NULL,          -- = recognition stream's source_id
    first_seen_at   TEXT NOT NULL,          -- ISO 8601 wall-clock
    last_seen_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sighting_employee_time ON sighting(employee_id, first_seen_at);

-- DailyAttendanceRecord is a query, not a table — nothing to migrate if the
-- aggregation rule ever changes:
--
--   SELECT employee_id,
--          date(first_seen_at)   AS day,
--          MIN(first_seen_at)    AS first_arrival,
--          MAX(last_seen_at)     AS last_departure
--   FROM sighting
--   WHERE employee_id = :employee_id AND date(first_seen_at) = :day
--   GROUP BY employee_id, date(first_seen_at);
--
-- No sighting that day for an employee => absent (left as an application-side
-- check against the full employee roster, not expressible as a single query).

-- Productivity monitoring v1 (2026-09-01, /grilling design): observable desk
-- states as intervals. Only "positive" states are stored -- AT_DESK (and
-- PHONE once a phone signal exists); "away" is DERIVED at read time as
-- presence span minus at-desk time (Q4/Q13a: any time between first and
-- last sighting that isn't at a desk counts as away, visible or not).
-- open=1 rows are still running: ended_at is a heartbeat the engine advances
-- every cycle, so live counters are plain SUM(ended-started).
CREATE TABLE IF NOT EXISTS desk_interval (
    id              TEXT PRIMARY KEY,
    employee_id     TEXT NOT NULL REFERENCES employee(id),
    state           TEXT NOT NULL,          -- 'at_desk' | 'phone'
    started_at      TEXT NOT NULL,          -- ISO 8601 video wall-clock
    ended_at        TEXT NOT NULL,
    open            INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_desk_interval_employee ON desk_interval(employee_id, started_at);

-- Restricted-zone violations (2026-09-02, /grilling round-2 Q4 + /mentor
-- design): same open-interval shape as desk_interval above, debounced by
-- ZoneViolationEngine (violations.py). employee_id is NULL for an
-- "unidentified person in a restricted area" -- the more important alert
-- per the user, so recording one must not require a known identity.
CREATE TABLE IF NOT EXISTS violation (
    id              TEXT PRIMARY KEY,
    employee_id     TEXT REFERENCES employee(id),   -- NULL = unidentified person
    camera          TEXT NOT NULL,
    zone_id         TEXT NOT NULL,
    violation_type  TEXT NOT NULL,          -- 'role_not_allowed' | 'unidentified'
    started_at      TEXT NOT NULL,
    ended_at        TEXT NOT NULL,
    open            INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_violation_started ON violation(started_at);
