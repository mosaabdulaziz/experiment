-- Concealment detection service storage (SQLite).
-- Candidates are ALWAYS suspected incidents requiring human verification, never confirmed theft.

CREATE TABLE IF NOT EXISTS concealment_candidates (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,
    camera                      TEXT NOT NULL,
    location                    TEXT NOT NULL,
    track_id                    INTEGER NOT NULL,
    pts                         REAL NOT NULL,
    display_time                TEXT NOT NULL,
    tier                        TEXT NOT NULL,
    severity                    TEXT NOT NULL,
    category                    TEXT NOT NULL DEFAULT 'security',
    kind                        TEXT NOT NULL DEFAULT 'suspected_concealment',
    score                       REAL NOT NULL,
    zone_id                     TEXT NOT NULL,
    target_kind                 TEXT NOT NULL,
    body_zone_name              TEXT NOT NULL,
    description                 TEXT NOT NULL,
    stage_timeline_json         TEXT NOT NULL,
    clip_window_json            TEXT NOT NULL,
    keyframes_pts_json          TEXT NOT NULL,
    clip_path                   TEXT,
    terminal_stage              TEXT,
    hold_duration_s             REAL,
    dwell_duration_s            REAL,
    thumbnail_path              TEXT,
    requires_human_verification INTEGER NOT NULL DEFAULT 1,
    status                      TEXT NOT NULL DEFAULT 'candidate',
    reviewer                    TEXT,
    reviewed_at                 TEXT,
    note                        TEXT DEFAULT '',
    created_at                  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_concealment_candidates_cam_status
    ON concealment_candidates(camera, status, id DESC);

CREATE TABLE IF NOT EXISTS concealment_assessments (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    camera      TEXT NOT NULL,
    track_id    INTEGER NOT NULL,
    pts         REAL NOT NULL,
    state       TEXT NOT NULL,
    reason      TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_concealment_assessments_cam
    ON concealment_assessments(camera, id DESC);
