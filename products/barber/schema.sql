-- Barber service storage (three cameras, 2026-09-24) -- SQLite, same
-- convention as products/attendance. Times are naive local ISO-8601 wall-clock
-- (seconds precision). File: data/barber-mc.db (the single-camera
-- barber.db is left as it was).

CREATE TABLE IF NOT EXISTS staff (
    id          TEXT PRIMARY KEY,           -- staff-N
    name        TEXT NOT NULL,
    active      INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT NOT NULL
);

-- Only ever counts up, so a deleted staff member's id is never reused.
CREATE TABLE IF NOT EXISTS id_sequence (
    key TEXT PRIMARY KEY,
    n   INTEGER NOT NULL
);

-- Who works which chair, and since when. A haircut is credited to the chair's
-- barber at its start (staff are no longer recognized by face).
CREATE TABLE IF NOT EXISTS roster (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    chair_id    TEXT NOT NULL,              -- the chair's area id (its pair id)
    staff_id    TEXT NOT NULL,
    valid_from  TEXT NOT NULL,
    valid_to    TEXT                        -- NULL = still assigned
);

CREATE INDEX IF NOT EXISTS idx_roster_chair ON roster(chair_id, valid_from);

-- The desk confirms each shop day's chair assignments once (the roster
-- carries over from the day before until someone changes it).
CREATE TABLE IF NOT EXISTS roster_confirmation (
    day           TEXT PRIMARY KEY,         -- YYYY-MM-DD shop day
    confirmed_at  TEXT NOT NULL
);

-- One customer, entry to exit.
CREATE TABLE IF NOT EXISTS visit (
    id                  TEXT PRIMARY KEY,
    person_id           TEXT NOT NULL,          -- the engine's global person id (not stable across restarts)
    entered_at          TEXT NOT NULL,
    entry_observed      INTEGER NOT NULL,       -- 0 = first seen mid-shop
    state               TEXT NOT NULL,          -- waiting | haircut | washing | served | finished
    waiting_started_at  TEXT,
    service_started_at  TEXT,                   -- first haircut's start = end of the wait
    wait_s              REAL,                   -- to the chair (0 = no wait) or, without a haircut, to the exit
    wait_target_s       REAL,
    excessive_wait      INTEGER NOT NULL DEFAULT 0,
    haircuts            INTEGER NOT NULL DEFAULT 0,
    exited_at           TEXT,
    exit_reason         TEXT,                   -- door | lost
    outcome             TEXT,                   -- completed | walked_out | lost | void_staff | interrupted
    customer_id         TEXT,
    ticket              INTEGER,                -- daily #n, assigned on confirm (restarts each shop day)
    open                INTEGER NOT NULL DEFAULT 1,
    updated_at          TEXT NOT NULL,
    wait_unseen         INTEGER NOT NULL DEFAULT 0  -- first seen sitting in a chair: wait unknown, left out of the stats
);

CREATE INDEX IF NOT EXISTS idx_visit_entered ON visit(entered_at);

-- One haircut in one chair; washing trips stay inside it (wash_s).
CREATE TABLE IF NOT EXISTS service_session (
    id                  TEXT PRIMARY KEY,
    visit_id            TEXT NOT NULL REFERENCES visit(id),
    chair_id            TEXT NOT NULL,
    started_at          TEXT NOT NULL,
    ended_at            TEXT,
    service_s           REAL,                   -- NULL while open, or if interrupted by a restart
    wash_s              REAL NOT NULL DEFAULT 0,
    barber_id           TEXT,                   -- the roster's barber for the chair at the start
    service_target_s    REAL,
    excessive_service   INTEGER NOT NULL DEFAULT 0,
    open                INTEGER NOT NULL DEFAULT 1,
    occupancy           REAL,                   -- share of its sittings' frames with the sitter on the seat
    low_occupancy       INTEGER NOT NULL DEFAULT 0   -- occupancy below ce_low_occupancy: possibly not a customer
);

CREATE INDEX IF NOT EXISTS idx_session_ended ON service_session(ended_at);

-- Every persisted state change of a visit: the audit trail.
CREATE TABLE IF NOT EXISTS state_event (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    visit_id    TEXT NOT NULL REFERENCES visit(id) ON DELETE CASCADE,
    from_state  TEXT,
    to_state    TEXT NOT NULL,
    area        TEXT,
    camera      TEXT,
    at          TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_state_event_visit ON state_event(visit_id, id);

-- Confirmed cross-camera handoffs; offset_s tunes sync_window_s.
CREATE TABLE IF NOT EXISTS handoff (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    area        TEXT NOT NULL,
    camera_a    TEXT NOT NULL,
    track_a     INTEGER NOT NULL,
    camera_b    TEXT NOT NULL,
    track_b     INTEGER NOT NULL,
    offset_s    REAL,
    reid_sim    REAL,
    at          TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_handoff_at ON handoff(at);

-- Every track that joined or created a person (merge audit); cleared per lap like handoff.
CREATE TABLE IF NOT EXISTS binding (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    kind            TEXT NOT NULL,
    camera          TEXT NOT NULL,
    track_id        INTEGER NOT NULL,
    person_id       TEXT NOT NULL,
    area            TEXT,
    partner_camera  TEXT,
    partner_track   INTEGER,
    reid_sim        REAL,
    offset_s        REAL,
    at              TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_binding_at ON binding(at);

CREATE TABLE IF NOT EXISTS barber_alert (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,                  -- excessive_wait | excessive_service
    camera      TEXT NOT NULL,                  -- "shop"
    visit_id    TEXT NOT NULL,
    session_id  TEXT,
    started_at  TEXT NOT NULL,
    ended_at    TEXT,
    open        INTEGER NOT NULL DEFAULT 1,
    subject_cam TEXT,
    subject_box TEXT,                -- JSON [x, y, w, h], normalised
    snapshot_path TEXT               -- relative to the crops root
);

CREATE INDEX IF NOT EXISTS idx_alert_started ON barber_alert(started_at);

-- The customer register (project C): one row per recognised customer.
CREATE TABLE IF NOT EXISTS customer (
    id              TEXT PRIMARY KEY,
    first_seen      TEXT NOT NULL,
    last_seen       TEXT NOT NULL,
    visit_count     INTEGER NOT NULL DEFAULT 0,
    last_visit_id   TEXT,
    face_count      INTEGER NOT NULL DEFAULT 0       -- 0 = never had a usable face: never matched
);

CREATE INDEX IF NOT EXISTS idx_customer_last_seen ON customer(last_seen);

-- His best faces (at most customer_max_faces) and bodies (customer_max_bodies).
CREATE TABLE IF NOT EXISTS customer_face (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id TEXT NOT NULL REFERENCES customer(id) ON DELETE CASCADE,
    crop_path   TEXT NOT NULL,                       -- relative to the crops root
    embedding   BLOB NOT NULL,                       -- float32 x 512, unit length
    quality     REAL NOT NULL,
    taken_at    TEXT NOT NULL,
    visit_id    TEXT
);

CREATE TABLE IF NOT EXISTS customer_body (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id TEXT NOT NULL REFERENCES customer(id) ON DELETE CASCADE,
    crop_path   TEXT NOT NULL,
    embedding   BLOB,                                -- float32 re-ID, reference only
    quality     REAL NOT NULL,
    taken_at    TEXT NOT NULL,
    visit_id    TEXT,
    camera      TEXT                                 -- 2026-09-29: body re-ID compares per camera
);

-- Chair engine (project 3, 2026-10-04): door crossings from the entrance line (no identity) and
-- each haircut's sittings (a wash return is a second sitting of the same service_session).
CREATE TABLE IF NOT EXISTS door_crossing (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    at          TEXT NOT NULL,
    direction   TEXT NOT NULL,             -- in | out
    camera      TEXT NOT NULL,
    track_id    INTEGER NOT NULL,
    staff       INTEGER NOT NULL DEFAULT 0,
    entry_id    TEXT,                      -- a non-staff `in`: the door entry (face register)
    customer_id TEXT                       -- the customer the entry's faces registered
);
CREATE INDEX IF NOT EXISTS idx_door_crossing_at ON door_crossing(at);

CREATE TABLE IF NOT EXISTS chair_sitting (
    session_id  TEXT NOT NULL,
    n           INTEGER NOT NULL,
    chair_id    TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    ended_at    TEXT,
    PRIMARY KEY (session_id, n)
);

-- A customer wait (chair engine, project 4 A + B, 2026-10-05): the centre camera's waiting area until a chair
-- row, the door, or a loss. Not paired with haircuts; every wait counts in the average wait.
CREATE TABLE IF NOT EXISTS customer_wait (
    id             TEXT PRIMARY KEY,
    started_at     TEXT NOT NULL,
    ended_at       TEXT,                    -- NULL while open
    wait_s         REAL,                    -- NULL while open
    ended_by       TEXT,                    -- chair | wash | door | lost | left | stall | restart; NULL while open
    target_s       REAL,
    camera         TEXT NOT NULL,
    track_id       INTEGER NOT NULL,        -- the track that started it
    last_track_id  INTEGER NOT NULL,        -- the track holding it last (reconnections)
    ticket         INTEGER,
    last_seen_at   TEXT NOT NULL            -- refreshed while open; a restart closes the wait here
);

CREATE INDEX IF NOT EXISTS idx_customer_wait_ended ON customer_wait(ended_at);

-- Small service facts that must survive a restart (key -> JSON text). 2026-10-11: 'clip_run' = the test clips'
-- run the rows belong to, {camera: [start_ts, clip_duration_s, lap]}; clear_all leaves this table alone.
CREATE TABLE IF NOT EXISTS service_state (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Visits' start-of-day baseline: who was already inside when the shop day's first run settled (one row per shop day).
CREATE TABLE IF NOT EXISTS visit_baseline (
    day       TEXT PRIMARY KEY,
    taken_at  TEXT NOT NULL,
    count     INTEGER NOT NULL,
    haircuts  INTEGER NOT NULL,
    waiting   INTEGER NOT NULL,
    washing   INTEGER NOT NULL
);

-- Every re-ID check's answer (2026-10-18, user: keep every score, to tune the resume thresholds per chair). kind:
-- return = the chair taken again after its customer left; reseat = its seat seen taken again after cover only.
CREATE TABLE IF NOT EXISTS resume_check (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    chair_id         TEXT NOT NULL,
    kind             TEXT NOT NULL,
    sat_at           TEXT NOT NULL,
    decided_at       TEXT NOT NULL,
    gap_s            REAL NOT NULL,
    sim              REAL,                    -- NULL: no re-ID samples on one side, the time rule answered
    threshold        REAL,
    resumed          INTEGER NOT NULL,        -- 1 = the same customer
    haircut_id       TEXT NOT NULL,           -- the haircut the sitter got
    prev_haircut_id  TEXT NOT NULL,           -- the one he was compared with
    ref_samples      INTEGER NOT NULL,
    new_samples      INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_resume_check_decided ON resume_check(decided_at);

-- A haircut's start and end photos (2026-10-22, spec 2026-10-22-barber-haircut-photos-design.md): a close-up of its
-- chair, a file under crops_dir. No foreign key: the start photo is taken before the haircut is confirmed.
CREATE TABLE IF NOT EXISTS haircut_photo (
    session_id  TEXT NOT NULL,              -- the haircut's id (= its service_session id once confirmed)
    kind        TEXT NOT NULL,              -- start | end
    path        TEXT NOT NULL,
    taken_at    TEXT NOT NULL,
    PRIMARY KEY (session_id, kind)
);

CREATE INDEX IF NOT EXISTS idx_haircut_photo_taken ON haircut_photo(taken_at);

-- 2026-11-10 (spec 2026-11-10-barber-pedicure-chairs-design.md): pedicure sessions on pedicure chairs -- never customers
CREATE TABLE IF NOT EXISTS pedicure_session (
    id          TEXT PRIMARY KEY,
    chair_id    TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    ended_at    TEXT,                   -- NULL while open
    service_s   REAL,                   -- sitting time (end gap excluded); NULL while open or if interrupted
    open        INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_pedicure_started ON pedicure_session(started_at);
