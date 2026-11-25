-- Fall detection service storage -- SQLite, the same convention as products/attendance and
-- products/barber. File: data/fall_detection.db. Times are UTC ISO-8601 (seconds precision).

-- One candidate that was too weak to confirm, kept with the evidence that produced it: the
-- evaluation, the combined score, the per-path scores, the reading's top class and the clip's
-- time range. An operator's decision is only useful next to the evidence.
CREATE TABLE IF NOT EXISTS fall_review (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    camera      TEXT NOT NULL,
    pts         REAL,                      -- the sidecar's frame clock, the clip's centre
    track       INTEGER,
    event_id    TEXT,
    kind        TEXT,                      -- fall | immobility | lying
    evaluation  TEXT NOT NULL,             -- falling | immobile | lying | unstable | upright
    combined    REAL NOT NULL,             -- the combined score that was below the threshold
    path_scores TEXT NOT NULL,             -- JSON {action, kinematic, cold}
    top_class   INTEGER,                   -- the reading's top class
    clip_window TEXT,                      -- JSON [t_start, t_end]
    probs       TEXT,                      -- JSON the reading's 8-class vector
    status      TEXT NOT NULL DEFAULT 'candidate',   -- candidate | acknowledged | confirmed | dismissed
    note        TEXT,                      -- the operator's note
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_fall_review_camera ON fall_review(camera, status);

-- What the product saw, and what it decided. This is the product's OWN history: the review
-- queue (fall_review) is the operator's work list, this is the record of every scored
-- candidate. The two are different things and are not joined except by review_id.
CREATE TABLE IF NOT EXISTS fall_event (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    camera      TEXT NOT NULL,
    pts         REAL,                      -- the sidecar's frame clock
    track       INTEGER,
    event_id    TEXT,
    kind        TEXT,                      -- fall | immobility | lying
    evaluation  TEXT NOT NULL,             -- falling | immobile | lying | unstable | upright
    combined    REAL NOT NULL,
    severity    TEXT,                      -- critical | warning | NULL (the closed false alarm)
    path_scores TEXT NOT NULL,             -- JSON {action, kinematic, cold}
    clip_window TEXT,                      -- JSON [t_start, t_end]
    review_id   INTEGER,                   -- the review row, when the candidate was queued
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_fall_event_camera ON fall_event(camera, id);
