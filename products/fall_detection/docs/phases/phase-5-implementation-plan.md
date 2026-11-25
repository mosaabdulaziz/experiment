# Phase 5 — the fall tab in the IRIS house style: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the fall detection product's tab in the IRIS house style (barber's tokens, shell, rail, logo, UI kit), and add the one capability the house does not have — an alarm that shows the operator a fall without ever covering the camera's live view.

**Architecture:** The backend gains one table (`fall_event`), one write (the consumer records every scored candidate) and one endpoint (`GET /events`), so the confirmed fall — which the routing never queues — can reach the tab. The frontend is a rewrite of `frontend/src/products/fall_detection` into the house's shape: `index.tsx` (entry) → `App.tsx` (shell + hash route) → `pages/` (Live, Review) → `alerts/` (the alarm card and the poll). All the decision-making is in pure modules (`alerts/severity.ts`, `alerts/queue.ts`, `alerts/badge.ts`, `api.ts`, `format.ts`) so it is unit-testable without a DOM, exactly as barber's tests are.

**Tech Stack:** React 19, TypeScript 5.4, Vite 5, vitest 2.1 (node environment), CSS custom properties; FastAPI + SQLite (WAL) for the service.

**Spec:** `products/fall_detection/docs/phases/phase-5-ui-redesign.md` (commit `331b75a`)

---

## Global Constraints

Every task below is subject to these. They are not stylistic preferences; the first four are enforced by tests that already exist.

1. **No WebSocket, no Redis contract, no canvas.** AC-F33.1 and AC-F34.1 are frozen requirements, enforced by `frontend/src/products/fall_detection/tab.test.ts:37-47` against every source file in the product. A source file containing the string `WebSocket`, matching `/iris:|XADD/`, or matching `/<canvas|getContext\(/` fails the suite.
2. **Only `live/frameUrl.ts` and `live/FrameLayer.tsx` may contain the string `/frame`.** No other file in the product may mention the frame endpoint; a page composes `FrameLayer` instead.
3. **The house's values are copied, never invented.** Every colour, radius and shadow in the fall product's `app.css` comes from `frontend/src/products/barber/app.css`. The token test in Task 4 fails if one value drifts.
4. **The product is light-only, LTR, English.** No dark-mode block (barber has none), no bilingual provider, no `dir="rtl"` handling — but every horizontal CSS property is logical (`inline-start` / `inline-end`), so RTL stays a later step.
5. **The frontend tests are pure view-model logic, no DOM.** `vitest.config.ts` sets `environment: 'node'`. A test that needs a DOM does not belong in this product.
6. **The Python tests run without the GPU and without the weights.** The corpus replays stay deselected on this dev box. `pytest` in `products/fall_detection` must stay green.
7. **Node.js is not on the PATH.** The portable toolchain is `F:\_tools\node-v24.18.0-win-x64` (node v24.18.0, npm 11.16.0). Every `npm` command in this plan is preceded by `export PATH="/f/_tools/node-v24.18.0-win-x64:$PATH"`. The npm cache lives on `F:` (`npm_config_cache=F:/_tools/npm-cache`).
8. **Nothing is committed without the user's approval.** The user owns every commit and every push in this project. The `git commit` step at the end of each task is to be run only after the user says yes; nothing is ever pushed.

**House CSS, by the line ranges it is copied from** (`frontend/src/products/barber/app.css`):

| What | Lines |
|---|---|
| the `:root` tokens | 10–45 |
| the base (`*`, `html, body`, `#root`, `button/input/select`, `.num`) | 47–59 |
| `.shell`, `.rail`, `.railLogo`, `.railBrandName`, `.railBtn`, `.railActive`, `.railLang`, `.railCount` | 68–131 |
| `.page`, `.pageHead`, `.pageHeadMain`, `.pageHeadTools` | 140–165 |
| `.card`, `.sectionLabel` | 176–195 |
| `.errorBar` | 206–213 |
| `.chip`, `.chipRed`, `.chipAmber`, `.chipGreen`, `.countPill` | 215–242 |
| `.btnPrimary`, `.btnGhost`, `.btnDanger`, `.btnPlain`, `.btnSmall` | 244–260 |
| `.feed`, `.feed img` | 357–368 |
| `.kpiGrid`, `.kpi`, `.kpiLabel`, `.kpiValue`, `.kpiSub`, `.kpiLive` | 470–488 |
| `.tableScroll`, `.dataTable`, `.tableEmpty` | 533–559 |
| `.confirmBackdrop`, `.confirmBox` | 648–653 |
| `.feedTile`, `.feedCaption`, `.feedName`, `.feedClip`, `.feedBadge`, `.feedLive` | 845–857 |

---

## The file structure

**Backend** (`products/fall_detection/`) — created or modified by Tasks 1–3:

| File | Responsibility |
|---|---|
| `schema.sql` | the `fall_event` table and its index (Task 1) |
| `storage.py` | `add_event`, `list_events`, and the two `ReviewQueue` methods that lock them (Task 1) |
| `consumer.py` | `_event_row` and the write in `handle_candidates` (Task 2) |
| `server.py` | `GET /events` (Task 3) |
| `tests/test_storage_event.py` | new — the history persists and reads back (Task 1) |
| `tests/test_worker_event.py` | new — what is recorded and what is not (Task 2) |
| `tests/test_events_api.py` | new — the endpoint's shape (Task 3) |
| `tests/fall_helpers.py` | `FakeStorage.add_event` (Task 2) |

**Frontend** (`frontend/src/products/fall_detection/`) — created or modified by Tasks 4–10:

| File | Responsibility |
|---|---|
| `app.css` | the house tokens and the classes this product uses, plus its own (Task 4) |
| `index.tsx` | the entry: Cairo, the stylesheet, `<App/>` (Task 8) |
| `App.tsx` | the hash route, the shell, the service data, the events feed (Tasks 8–9) |
| `ui/Shell.tsx` | the rail: the IRIS mark, Live, Review, the unseen badge (Task 8) |
| `ui/ui.tsx` | `Icon`, `ICONS`, `Confirm` (Task 8) |
| `pages/LivePage.tsx` | the page head, the KPI row, the camera tile, the side column (Task 8) |
| `pages/ReviewPage.tsx` | the queue table (Task 8) |
| `alerts/severity.ts` | the event → (toast?, card?, badge?) mapping (Task 6) |
| `alerts/queue.ts` | one card at a time, the queued count, the arm delay (Task 6) |
| `alerts/badge.ts` | the unseen counter (Task 6) |
| `alerts/useEvents.ts` | the poll, the cursor, the notices (Task 9) |
| `alerts/AlarmCard.tsx` | the alarm card (Task 10) |
| `api.ts` | the calls, typed (Task 7) |
| `format.ts` | the clock, the score, the chip class (Task 7) |
| `live/FrameLayer.tsx` | one optional `className`; still the only place that knows `/frame` (Task 8) |
| `live/frameUrl.ts` | unchanged |
| `tab.test.ts` | the strengthened frame boundary (Task 5) |
| `app.test.ts` | the tokens are the house's (Task 4) |
| `package.json` | `+@fontsource/cairo` (Task 4) |
| `vitest.config.ts` | new, mirroring barber's (Task 4) |

**Root** (`frontend/`): `package.json` and `package-lock.json` gain `@types/node` (Task 4) — without it the dashboard's `tsc -b` fails on the product's own guard tests.

---

### Task 1: The event history — the table, the write, the read

The confirmed fall reaches the platform over Redis and is then forgotten by the service. The tab needs a record of its own. This task adds the table and the two storage calls; Task 2 wires the write, Task 3 the endpoint.

**Files:**
- Modify: `products/fall_detection/schema.sql`
- Modify: `products/fall_detection/storage.py`
- Create: `products/fall_detection/tests/test_storage_event.py`

**Interfaces:**
- Consumes: nothing (this is the first task).
- Produces: `storage.add_event(conn, row: dict) -> int`, `storage.list_events(conn, camera: str | None = None, since: int = 0, limit: int = 50) -> list[dict]`, and on `storage.ReviewQueue`: `add_event(row: dict) -> int` and `list_events(camera=None, since=0, limit=50) -> list[dict]`. The returned rows carry `review_status` (the joined review row's status, or `None`).

- [ ] **Step 1: Write the failing test**

Create `products/fall_detection/tests/test_storage_event.py`:

```python
"""The product's own history: every scored candidate, kept with its evidence.

The review queue is the operator's work list; this is the record of what the product saw.
A confirmed fall is never queued, so without this table the tab could never show one.
"""
import pytest

import storage

EVENT = {
    "camera": "cam-1",
    "pts": 103.5,
    "track": 3,
    "event_id": "ev-abc123",
    "kind": "fall",
    "evaluation": "falling",
    "combined": 0.81,
    "severity": "critical",
    "path_scores": {"action": 0.9, "kinematic": None, "cold": None},
    "clip_window": [98.0, 105.5],
    "review_id": None,
}


def test_an_event_survives_a_restart_with_all_of_its_evidence(tmp_path):
    db = tmp_path / "fall.db"
    conn = storage.connect(db)
    event_id = storage.add_event(conn, EVENT)
    conn.close()
    conn2 = storage.connect(db)                              # the restart
    (row,) = [r for r in storage.list_events(conn2) if r["id"] == event_id]
    assert row["camera"] == "cam-1"
    assert row["pts"] == pytest.approx(103.5)
    assert row["evaluation"] == "falling"
    assert row["combined"] == pytest.approx(0.81)
    assert row["severity"] == "critical"
    assert row["path_scores"] == {"action": 0.9, "kinematic": None, "cold": None}
    assert row["clip_window"] == [98.0, 105.5]
    assert row["review_status"] is None


def test_the_history_is_newest_first_and_the_cursor_moves_forward(tmp_path):
    conn = storage.connect(tmp_path / "fall.db")
    first = storage.add_event(conn, dict(EVENT, event_id="ev-1"))
    second = storage.add_event(conn, dict(EVENT, event_id="ev-2"))
    assert [r["id"] for r in storage.list_events(conn)] == [second, first]
    assert [r["id"] for r in storage.list_events(conn, since=first)] == [second]


def test_the_history_can_be_narrowed_to_one_camera(tmp_path):
    conn = storage.connect(tmp_path / "fall.db")
    storage.add_event(conn, dict(EVENT, camera="cam-1"))
    storage.add_event(conn, dict(EVENT, camera="cam-2"))
    assert [r["camera"] for r in storage.list_events(conn, camera="cam-2")] == ["cam-2"]


def test_a_queued_candidate_carries_the_review_row_status(tmp_path):
    conn = storage.connect(tmp_path / "fall.db")
    review_id = storage.add_review(conn, {"camera": "cam-1", "evaluation": "falling", "combined": 0.4})
    storage.add_event(conn, dict(EVENT, review_id=review_id))
    (row,) = storage.list_events(conn)
    assert row["review_id"] == review_id
    assert row["review_status"] == "candidate"
    storage.set_review_status(conn, review_id, "acknowledged")
    (row,) = storage.list_events(conn)
    assert row["review_status"] == "acknowledged"


def test_the_queue_serializes_the_history_the_same_way_it_serializes_the_reviews(tmp_path):
    queue = storage.ReviewQueue(storage.connect(tmp_path / "fall.db"))
    assert queue.add_event(dict(EVENT)) == 0
    assert len(queue.list_events()) == 1
```

- [ ] **Step 2: Run the test to verify it fails**

Run from `products/fall_detection`:

```
python -m pytest tests/test_storage_event.py -v
```

Expected: FAIL — `AttributeError: module 'storage' has no attribute 'add_event'`.

- [ ] **Step 3: Add the table**

Append to `products/fall_detection/schema.sql`:

```sql

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
```

- [ ] **Step 4: Add the two storage calls**

In `products/fall_detection/storage.py`, add after `add_review` (line 51):

```python
def add_event(conn: sqlite3.Connection, row: dict) -> int:
    """Store one scored candidate in the product's own history and return its row id."""
    cursor = conn.execute(
        """INSERT INTO fall_event (camera, pts, track, event_id, kind, evaluation, combined,
                                   severity, path_scores, clip_window, review_id, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (row.get("camera"), row.get("pts"), row.get("track"), row.get("event_id"),
         row.get("kind"), row.get("evaluation"), float(row.get("combined") or 0.0),
         row.get("severity"), json.dumps(row.get("path_scores") or {}),
         json.dumps(row.get("clip_window")) if row.get("clip_window") is not None else None,
         row.get("review_id"), datetime.now(timezone.utc).isoformat(timespec="seconds")))
    conn.commit()
    return int(cursor.lastrowid)


def list_events(conn: sqlite3.Connection, camera: str | None = None, since: int = 0,
                limit: int = 50) -> list[dict]:
    """The history, newest first, with the review row's status beside it when it has one."""
    query = ("SELECT e.*, r.status AS review_status "
             "FROM fall_event e LEFT JOIN fall_review r ON r.id = e.review_id "
             "WHERE e.id > ?")
    params: list = [int(since)]
    if camera:
        query += " AND e.camera = ?"
        params.append(camera)
    query += " ORDER BY e.id DESC LIMIT ?"
    params.append(max(1, min(int(limit), 200)))
    return [_as_dict(row) for row in conn.execute(query, params)]
```

Add the two methods to the `ReviewQueue` class, after `add_review` (line 102):

```python
    def add_event(self, row: dict) -> int:
        with self._lock:
            return add_event(self._conn, row)

    def list_events(self, camera: str | None = None, since: int = 0, limit: int = 50) -> list[dict]:
        with self._lock:
            return list_events(self._conn, camera, since, limit)
```

- [ ] **Step 5: Run the test to verify it passes**

```
python -m pytest tests/test_storage_event.py -v
```

Expected: PASS — 5 passed.

- [ ] **Step 6: Run the product's suite to confirm nothing broke**

```
python -m pytest -q
```

Expected: the same count as before this task, plus the 5 new ones. Nothing else changed.

- [ ] **Step 7: Commit** (only after the user approves)

```bash
git add products/fall_detection/schema.sql products/fall_detection/storage.py products/fall_detection/tests/test_storage_event.py
git commit -m "feat(fall): the product's own history -- every scored candidate is kept"
```

---

### Task 2: The consumer writes the history

The worker's `handle_candidates` is where every scored candidate passes. This task records it there — and records *nothing* for a suppressed candidate.

**Files:**
- Modify: `products/fall_detection/consumer.py:185-186`
- Modify: `products/fall_detection/tests/fall_helpers.py` (the `FakeStorage` needs `add_event`)
- Create: `products/fall_detection/tests/test_worker_event.py`

**Interfaces:**
- Consumes: `storage.ReviewQueue.add_event(row: dict) -> int` (Task 1).
- Produces: `CameraWorker._event_row(rec: dict, scored: dict, pts: float, review_id: int | None) -> dict`, and `handle_candidates` now returns `result["review_id"]` as the real row id (it was already the field name).

- [ ] **Step 1: Write the failing test**

Create `products/fall_detection/tests/test_worker_event.py`:

```python
"""What the worker records in the product's history, and what it deliberately does not.

A confirmed fall is never queued for review, so the history is the only place it can live.
The aftermath of an alerted fall is not an event at all (consumer.py:201-210) and must not
become one in the history.
"""
import pytest

from consumer import CameraWorker
from fall_helpers import DECISIVE_FALL, DECISIVE_UPRIGHT, FakeRedis, FakeStorage, make_rec
from scoring import route

from .conftest import POLICY, VENDOR_CFG


def _worker(storage):
    return CameraWorker(camera="cam-1", monitor=None, redis=FakeRedis(), storage=storage,
                        policy=POLICY, vendor_cfg=VENDOR_CFG)


def test_a_confirmed_fall_is_recorded_even_though_it_is_never_queued():
    storage = FakeStorage()
    worker = _worker(storage)
    results = worker.handle_candidates([make_rec(kind="fall", probs=DECISIVE_FALL, track=1)], t=200.0)
    assert results[0]["severity"] == "critical"
    assert results[0]["review_id"] is None            # the routing queues nothing for it
    assert len(storage.events) == 1                   # but the history keeps it
    (event,) = storage.events
    assert event["evaluation"] == "falling"
    assert event["severity"] == "critical"
    assert event["review_id"] is None
    assert event["camera"] == "cam-1"


def test_a_candidate_below_the_threshold_is_recorded_with_its_review_row():
    storage = FakeStorage()
    worker = _worker(storage)
    results = worker.handle_candidates([make_rec(kind="lying", probs=None, track=2)], t=200.0)
    assert results[0]["review_id"] is not None
    (event,) = storage.events
    assert event["review_id"] == results[0]["review_id"]
    assert event["severity"] == "warning"


def test_the_closed_false_alarm_is_recorded_with_no_severity():
    storage = FakeStorage()
    worker = _worker(storage)
    worker.handle_candidates([make_rec(kind="fall", probs=DECISIVE_UPRIGHT, track=3)], t=200.0)
    (event,) = storage.events
    assert event["evaluation"] == "upright"
    assert event["severity"] is None


def test_the_aftermath_of_an_alerted_fall_is_not_recorded():
    storage = FakeStorage()
    worker = _worker(storage)
    worker.handle_candidates([make_rec(kind="fall", probs=DECISIVE_FALL, track=7)], t=200.0)
    assert len(storage.events) == 1
    results = worker.handle_candidates([make_rec(kind="fall", probs=DECISIVE_FALL, track=7)], t=201.0)
    assert results[0]["suppressed"] is True
    assert len(storage.events) == 1                    # the aftermath is not an event
```

> **If `CameraWorker`'s constructor does not take `policy` and `vendor_cfg` by those names**, read `products/fall_detection/consumer.py` for the real signature and use it — the existing `tests/test_worker.py` constructs workers and is the reference for the exact call.

- [ ] **Step 2: Run the test to verify it fails**

```
python -m pytest tests/test_worker_event.py -v
```

Expected: FAIL — `AttributeError: 'FakeStorage' object has no attribute 'events'`.

- [ ] **Step 3: Give the fake storage an event log**

In `products/fall_detection/tests/fall_helpers.py`, replace the `FakeStorage` class (lines 90–96) with:

```python
class FakeStorage:
    def __init__(self):
        self.rows = []
        self.events = []

    def add_review(self, row):
        self.rows.append(dict(row))
        return len(self.rows) - 1

    def set_review_status(self, row_id, status, note=""):
        self.rows[row_id].update(status=status, note=note)

    def add_event(self, row):
        self.events.append(dict(row))
        return len(self.events) - 1
```

- [ ] **Step 4: Write the event row and the write**

In `products/fall_detection/consumer.py`, replace lines 185–186:

```python
            if not suppressed and scored["review"]:
                result["review_id"] = self.storage.add_review(self._review_row(rec, scored, now))
            if not suppressed:
                self.storage.add_event(self._event_row(rec, scored, now, result["review_id"]))
```

and add this method after `_review_row` (line 220):

```python
    def _event_row(self, rec: dict, scored: dict, pts: float, review_id: int | None) -> dict:
        """The scored candidate as the product's own history row.

        The review row is the operator's work list; this row is the record. A confirmed fall
        has no review row and is still kept here -- that is the whole reason the table exists.
        """
        evidence = rec.get("evidence") or {}
        return {"camera": self.camera, "pts": pts, "track": rec.get("track"),
                "event_id": rec.get("event_id"), "kind": rec.get("type"),
                "evaluation": scored["evaluation"], "combined": scored["combined"],
                "severity": scored["severity"], "path_scores": scored["scores"],
                "clip_window": evidence.get("clip_window"), "review_id": review_id}
```

- [ ] **Step 5: Run the test to verify it passes**

```
python -m pytest tests/test_worker_event.py -v
```

Expected: PASS — 4 passed.

- [ ] **Step 6: Run the product's suite**

```
python -m pytest -q
```

Expected: all green, including `tests/test_worker.py` (the worker tests use `FakeStorage`, which now has the extra method).

- [ ] **Step 7: Commit** (only after the user approves)

```bash
git add products/fall_detection/consumer.py products/fall_detection/tests/fall_helpers.py products/fall_detection/tests/test_worker_event.py
git commit -m "feat(fall): the worker records every scored candidate, and no aftermath"
```

---

### Task 3: The `/events` endpoint

The tab polls for what it has not seen. The endpoint is a bare JSON array, exactly like `/cameras` and `/candidates`.

**Files:**
- Modify: `products/fall_detection/server.py:41-44`
- Create: `products/fall_detection/tests/test_events_api.py`

**Interfaces:**
- Consumes: `ReviewQueue.list_events(camera=None, since=0, limit=50) -> list[dict]` (Task 1).
- Produces: `GET /events?camera=<id>&since=<id>&limit=<n>` → a JSON array of event rows, newest first, each carrying `review_status`.

- [ ] **Step 1: Write the failing test**

Create `products/fall_detection/tests/test_events_api.py`:

```python
"""The events endpoint: the tab's feed of what the product has seen since its cursor.

ACs: AC-F22.1 (the evidence beside the operator's decision). Design: phase-5 §4.3.
"""
from fastapi.testclient import TestClient
import pytest

import boot
import storage
from server import create_app
from storage import ReviewQueue

EVENT = {"camera": "cam-1", "pts": 103.5, "track": 3, "kind": "fall",
         "evaluation": "falling", "combined": 0.81, "severity": "critical",
         "path_scores": {"action": 0.9, "kinematic": None, "cold": None},
         "clip_window": [98.0, 105.5], "review_id": None}


@pytest.fixture
def queue(tmp_path):
    q = ReviewQueue(storage.connect(tmp_path / "events.db"))
    yield q
    q.close()


def _client(queue):
    ready = {"device": "cuda:0", "weights_verified": True, "classifier_ready": True}
    return TestClient(create_app(status=boot.status(ready), storage=queue))


def test_the_events_are_a_bare_array_newest_first(queue):
    queue.add_event(dict(EVENT, event_id="ev-1"))
    queue.add_event(dict(EVENT, event_id="ev-2"))
    body = _client(queue).get("/events").json()
    assert isinstance(body, list)
    assert [r["event_id"] for r in body] == ["ev-2", "ev-1"]
    assert body[0]["severity"] == "critical"


def test_the_cursor_returns_only_what_is_newer(queue):
    first = queue.add_event(dict(EVENT, event_id="ev-1"))
    queue.add_event(dict(EVENT, event_id="ev-2"))
    body = _client(queue).get(f"/events?since={first}").json()
    assert [r["event_id"] for r in body] == ["ev-2"]


def test_the_camera_narrows_the_feed(queue):
    queue.add_event(dict(EVENT, camera="cam-1"))
    queue.add_event(dict(EVENT, camera="cam-2"))
    body = _client(queue).get("/events?camera=cam-2").json()
    assert [r["camera"] for r in body] == ["cam-2"]


def test_the_limit_is_capped(queue):
    for i in range(5):
        queue.add_event(dict(EVENT, event_id=f"ev-{i}"))
    body = _client(queue).get("/events?limit=2").json()
    assert len(body) == 2
```

- [ ] **Step 2: Run the test to verify it fails**

```
python -m pytest tests/test_events_api.py -v
```

Expected: FAIL — the route does not exist (404).

- [ ] **Step 3: Add the route**

In `products/fall_detection/server.py`, add after the `/candidates` route (line 44):

```python
    @app.get("/events")
    def events(camera: str | None = None, since: int = 0, limit: int = 50):
        """The product's own history: every scored candidate, newest first.

        The tab polls this with its cursor; `since` is the last id it has seen, so the
        response is only what arrived since. A confirmed fall is never queued for review,
        so this is the only place it can live.
        """
        return storage.list_events(camera, since, limit)
```

> `storage` here is the `ReviewQueue` instance passed to `create_app` — the same name the
> `/candidates` route already uses. `storage.list_events(camera, since, limit)` is the
> `ReviewQueue` method from Task 1, with the same signature as the module-level function.

- [ ] **Step 4: Run the test to verify it passes**

```
python -m pytest tests/test_events_api.py -v
```

Expected: PASS — 4 passed.

- [ ] **Step 5: Run the product's suite**

```
python -m pytest -q
```

Expected: all green.

- [ ] **Step 6: Commit** (only after the user approves)

```bash
git add products/fall_detection/server.py products/fall_detection/tests/test_events_api.py
git commit -m "feat(fall): GET /events -- the tab's feed of what the product has seen"
```

---

### Task 4: The house stylesheet, and the build unblocked

Two things land here because they are one deliverable: the product's stylesheet, and the two build inputs it needs (`@fontsource/cairo` and `@types/node`).

**Why `@types/node` is in this task:** `npm run build` on this branch **fails today**. `frontend/src/products/fall_detection/tab.test.ts` and `frameUrl.test.ts` import `node:fs`, `node:url` and `node:path`, and the dashboard's root `tsconfig.json` has no node types. The error is:

```
src/products/fall_detection/tab.test.ts(5,65): error TS2307: Cannot find module 'node:fs' or its corresponding type declarations.
```

Nothing else in the dashboard fails — this is the fall product's own defect, and it is why the product's frontend has never been built. It is fixed by adding `@types/node` to the dashboard root's `devDependencies`.

**Files:**
- Modify: `frontend/package.json` (add `@types/node`)
- Modify: `frontend/package-lock.json` (by the install)
- Create: `frontend/src/products/fall_detection/app.css`
- Create: `frontend/src/products/fall_detection/vitest.config.ts`
- Modify: `frontend/src/products/fall_detection/package.json` (add `@fontsource/cairo`)
- Create: `frontend/src/products/fall_detection/app.test.ts`

**Interfaces:**
- Consumes: nothing.
- Produces: `app.css` — the house's tokens and the classes this product uses, plus the product's own `.notice`, `.alarmCard*`, `.feedRing*`, `.sideCol`, `.eventList*`. `vitest.config.ts` — `include: ['**/*.test.ts']`, `environment: 'node'`.

- [ ] **Step 1: Add the node types and the Cairo font**

From `frontend/`:

```
export PATH="/f/_tools/node-v24.18.0-win-x64:$PATH"
export npm_config_cache="F:/_tools/npm-cache"
npm install --save-dev @types/node --no-audit --no-fund
npm install -w @iris/fall-detection @fontsource/cairo@^5.3.0 --no-audit --no-fund
```

Expected: both exit 0. `frontend/package.json` gains `"@types/node"` in `devDependencies`; `frontend/src/products/fall_detection/package.json` gains `"@fontsource/cairo": "^5.3.0"` in `dependencies` (barber's exact version).

- [ ] **Step 2: Write the failing token test**

Create `frontend/src/products/fall_detection/app.test.ts`:

```ts
// The fall tab is the house: the tokens are the house's, copied, not re-invented. If this
// drifts, the tab is a different product wearing the same colours.
import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

const ours = readFileSync(new URL('./app.css', import.meta.url), 'utf8')
const house = readFileSync(new URL('../barber/app.css', import.meta.url), 'utf8')

function tokens(css: string): Record<string, string> {
  const block = css.slice(css.indexOf(':root'), css.indexOf('}', css.indexOf(':root')))
  const out: Record<string, string> = {}
  for (const line of block.split('\n')) {
    const m = /^\s*(--[a-z0-9-]+)\s*:\s*(.+);\s*$/.exec(line)
    if (m) out[m[1]] = m[2].trim()
  }
  return out
}

const SHARED = [
  '--green-900', '--green-800', '--green-700', '--green-600', '--green-500', '--green-50',
  '--ground', '--card', '--ink', '--ink-soft', '--ink-mute', '--line', '--line-strong',
  '--red-500', '--red-50', '--amber-600', '--amber-500', '--amber-50',
  '--radius', '--shadow',
]

describe('the fall tab uses the house tokens', () => {
  it('every shared token is the house value, character for character', () => {
    const a = tokens(ours)
    const b = tokens(house)
    for (const name of SHARED) {
      expect(a[name], `${name} must be the house's value`).toBe(b[name])
    }
  })

  it('the stylesheet declares every token the house declares', () => {
    expect(Object.keys(tokens(ours))).toEqual(expect.arrayContaining(Object.keys(tokens(house))))
  })
})
```

- [ ] **Step 3: Run the test to verify it fails**

```
cd frontend/src/products/fall_detection && npx vitest run app.test.ts
```

Expected: FAIL — `ENOENT: no such file or directory ... app.css`.

- [ ] **Step 4: Write the stylesheet**

Create `frontend/src/products/fall_detection/app.css` by copying the house's blocks **verbatim** from `frontend/src/products/barber/app.css` at the line ranges in the Global Constraints table, in this order:

1. The `:root` block (lines 10–45), unchanged.
2. The base (47–59), unchanged.
3. The shell and rail (68–131), unchanged.
4. The page head (140–165), unchanged.
5. The card and section label (176–195), unchanged.
6. The error bar (206–213), unchanged.
7. The chips (215–242), unchanged.
8. The buttons (244–260), unchanged.
9. `.feed` and `.feed img` (357–368) — only those two rules, not the editing variants.
10. The KPI tiles (470–488), unchanged.
11. The table (533–559), unchanged.
12. `.confirmBackdrop`, `.confirmBox` (648–653), unchanged.
13. `.feedTile`, `.feedCaption`, `.feedName`, `.feedClip`, `.feedBadge`, `.feedLive` (845–857), unchanged.

Then add the product's own rules — the fall product's own layout, in house values only:

```css
/* ── the fall product's own ───────────────────────────────────────────────────── */

/* The board: the camera wide, the side column beside it (barber's .liveGrid idiom). */
.sideCol { display: flex; flex-direction: column; gap: 10px; min-width: 0; }
@media (min-width: 901px) {
  .sideCol { position: sticky; top: 10px; max-height: calc(100vh - 20px); }
  .sideCol .eventsCard { flex: 1 1 auto; min-height: 96px; display: flex; flex-direction: column; overflow: hidden; }
  .sideCol .eventList { min-height: 0; overflow-y: auto; overscroll-behavior: contain; }
}

/* The camera tile is marked, never covered: the ring says which camera fell. */
.feedRing { box-shadow: 0 0 0 3px var(--red-500); }
.feedRingWarn { box-shadow: 0 0 0 3px var(--amber-600); }

/* The quiet channel: a notice in the flow, not the house's fixed toast. The fall product
   has no fixed overlay at all -- nothing may cover the camera's live view. */
.notice { display: flex; align-items: center; gap: 8px; padding: 8px 10px; border-radius: 10px;
  background: var(--green-50); color: var(--green-700); font-weight: 700; font-size: 12.5px; }
.noticeCritical { background: var(--red-50); color: var(--red-500); }
.noticeWarning { background: var(--amber-50); color: var(--amber-600); }

/* The alarm card: in the page's flow, so the camera tile can never be covered. */
.alarmCard { background: var(--card); border: 1px solid var(--line); border-radius: var(--radius);
  box-shadow: var(--shadow); overflow: hidden; }
.alarmBar { display: flex; align-items: center; gap: 10px; padding: 8px 12px; color: #fff;
  font-size: 13px; font-weight: 800; }
.alarmCritical { background: var(--red-500); }
.alarmWarning { background: var(--amber-600); }
.alarmQueued { margin-inline-start: auto; background: rgba(0, 0, 0, 0.22); border-radius: 999px;
  padding: 1px 9px; font-size: 11.5px; }
.alarmBody { display: flex; gap: 10px; padding: 10px 12px; }
.alarmFacts { flex: 1 1 auto; min-width: 0; display: flex; flex-direction: column; gap: 3px; }
.alarmEval { font-weight: 800; color: var(--ink); font-size: 14px; }
.alarmMeta { color: var(--ink-soft); font-size: 12px; }
.alarmThumb { width: 96px; flex: none; border-radius: 8px; overflow: hidden; background: #101512; }
.alarmThumb .feed { aspect-ratio: 4 / 3; border-radius: 0; }
.alarmActions { display: flex; gap: 8px; padding: 0 12px 12px; }

/* The event list in the side column. */
.eventList { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 6px; }
.eventRow { display: flex; align-items: baseline; gap: 8px; font-size: 12.5px; }
.eventDot { width: 8px; height: 8px; border-radius: 50%; flex: none; background: var(--line-strong); }
.eventDotCritical { background: var(--red-500); }
.eventDotWarning { background: var(--amber-600); }
.eventEval { font-weight: 700; color: var(--ink); }
.eventScore { color: var(--ink-soft); font-variant-numeric: tabular-nums; margin-inline-start: auto; }
```

> `.notice` replaces the house's fixed `.toast` in this product: the house's toast is
> `position: fixed` at the bottom centre, and this product has no fixed overlay — the
> principle that the alarm never covers the camera applies to the quiet channel too.

- [ ] **Step 5: Run the token test to verify it passes**

```
cd frontend/src/products/fall_detection && npx vitest run app.test.ts
```

Expected: PASS — 2 passed.

- [ ] **Step 6: Add the vitest config**

Create `frontend/src/products/fall_detection/vitest.config.ts`:

```ts
// This product's unit tests (pure view-model logic, no DOM): npm test -w @iris/fall-detection
import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: { include: ['**/*.test.ts'], environment: 'node' },
})
```

- [ ] **Step 7: Build the dashboard**

```
cd frontend && export PATH="/f/_tools/node-v24.18.0-win-x64:$PATH" && npm run build
```

Expected: exit 0, `✓ built in …`. Before Task 4 this command **fails** with the `node:fs` errors quoted above — that is the defect this task removes.

- [ ] **Step 8: Run the product's suite**

```
cd frontend && export PATH="/f/_tools/node-v24.18.0-win-x64:$PATH" && npm run test -w @iris/fall-detection
```

Expected: 11 passed (the 9 that exist, plus the 2 new token tests).

- [ ] **Step 9: Commit** (only after the user approves)

```bash
git add frontend/package.json frontend/package-lock.json frontend/src/products/fall_detection/app.css frontend/src/products/fall_detection/app.test.ts frontend/src/products/fall_detection/vitest.config.ts frontend/src/products/fall_detection/package.json
git commit -m "feat(fall): the house stylesheet -- the tokens are the house's, and the build works"
```

---

### Task 5: Pin the frame boundary

The product's guard test today reads one file (`index.tsx`) and asks it to mention `FrameLayer` but not `/frame`. Task 8 moves that page, which would break the guard for a reason that has nothing to do with the invariant. This task replaces it with a stronger rule that holds before and after the move — so the refactor is protected rather than merely tolerated.

**Files:**
- Modify: `frontend/src/products/fall_detection/tab.test.ts:56-60`

**Interfaces:**
- Consumes: nothing.
- Produces: two passing assertions in `tab.test.ts`: the only files containing `/frame` are `live/frameUrl.ts` and `live/FrameLayer.tsx`; and exactly one file outside `live/` composes the layer.

- [ ] **Step 1: Write the stronger assertions**

In `frontend/src/products/fall_detection/tab.test.ts`, replace the `it('the page composes the layer and never touches the endpoint itself')` block (lines 56–60) with:

```ts
  it('only the frame module knows the endpoint, whatever else the product grows', () => {
    const knowing = texts.filter(t => t.text.includes('/frame')).map(t => t.p).sort()
    expect(knowing).toEqual(['live/FrameLayer.tsx', 'live/frameUrl.ts'])
  })

  it('a page composes the layer', () => {
    const pages = texts.filter(t => !t.p.startsWith('live/') && t.text.includes('FrameLayer'))
    expect(pages.length).toBe(1)
  })
```

- [ ] **Step 2: Run the test to verify it passes now, and after Task 8**

```
cd frontend/src/products/fall_detection && npx vitest run tab.test.ts
```

Expected: PASS — 8 passed. This is a characterisation test: it is green today and must stay green in Task 8, when `index.tsx` becomes `pages/LivePage.tsx`. Run it again at the end of Task 8.

- [ ] **Step 3: Commit** (only after the user approves)

```bash
git add frontend/src/products/fall_detection/tab.test.ts
git commit -m "test(fall): the frame boundary is the whole product, not one file"
```

---

### Task 6: The alarm's rules

Three pure modules decide whether an operator is interrupted, how many alarms are waiting, and what the rail badge counts. All three are pure functions so they are testable without a DOM.

**Files:**
- Create: `frontend/src/products/fall_detection/alerts/severity.ts`
- Create: `frontend/src/products/fall_detection/alerts/severity.test.ts`
- Create: `frontend/src/products/fall_detection/alerts/queue.ts`
- Create: `frontend/src/products/fall_detection/alerts/queue.test.ts`
- Create: `frontend/src/products/fall_detection/alerts/badge.ts`
- Create: `frontend/src/products/fall_detection/alerts/badge.test.ts`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `severity.ts`: `type Severity = 'critical' | 'warning'`; `type FallEvent = { id: number; camera: string; pts: number | null; track: number | null; event_id: string | null; kind: string | null; evaluation: string; combined: number; severity: Severity | null; path_scores: Record<string, number | null>; clip_window: [number, number] | null; review_id: number | null; review_status: string | null; created_at: string }`; `type Channels = { toast: boolean; card: boolean; badge: boolean }`; `channels(event: FallEvent): Channels`.
  - `queue.ts`: `const ARM_DELAY_MS = 1000`; `type AlarmState = { current: FallEvent | null; queued: FallEvent[] }`; `enqueue(state: AlarmState, event: FallEvent): AlarmState`; `acknowledge(state: AlarmState): AlarmState`; `canAcknowledge(now: number, shownAt: number, armDelayMs?: number): boolean`.
  - `badge.ts`: `unseenCount(events: FallEvent[], seenCursor: number): number`.

- [ ] **Step 1: Write the failing tests**

Create `frontend/src/products/fall_detection/alerts/severity.test.ts`:

```ts
// The rule that decides whether an operator is interrupted (phase-5 §3):
//   a confirmed fall            -> toast, card, badge
//   a candidate below the gate  -> toast, card, badge  (it is queued; a human must act)
//   a confirmed lying/immobile  -> toast, badge        (nobody is waiting on it)
//   a closed upright            -> nothing             (the false alarm the product rejected)
import { describe, expect, it } from 'vitest'
import { channels, type FallEvent } from './severity'

function event(over: Partial<FallEvent>): FallEvent {
  return {
    id: 1, camera: 'cam-1', pts: 100, track: 1, event_id: 'ev-1', kind: 'fall',
    evaluation: 'falling', combined: 0.81, severity: 'critical',
    path_scores: {}, clip_window: null, review_id: null, review_status: null,
    created_at: '2026-10-28T10:00:00+00:00', ...over,
  }
}

describe('which channel an event speaks on', () => {
  it('a confirmed fall takes over and counts', () => {
    expect(channels(event({ severity: 'critical', review_id: null })))
      .toEqual({ toast: true, card: true, badge: true })
  })

  it('a candidate below the threshold takes over and counts', () => {
    expect(channels(event({ severity: 'warning', evaluation: 'lying', review_id: 7 })))
      .toEqual({ toast: true, card: true, badge: true })
  })

  it('a confirmed lying only notifies and counts', () => {
    expect(channels(event({ severity: 'warning', evaluation: 'lying', review_id: null })))
      .toEqual({ toast: true, card: false, badge: true })
  })

  it('the closed false alarm says nothing at all', () => {
    expect(channels(event({ severity: null, evaluation: 'upright' })))
      .toEqual({ toast: false, card: false, badge: false })
  })
})
```

Create `frontend/src/products/fall_detection/alerts/queue.test.ts`:

```ts
import { describe, expect, it } from 'vitest'
import { acknowledge, ARM_DELAY_MS, canAcknowledge, enqueue, type AlarmState } from './queue'
import type { FallEvent } from './severity'

const ev = (id: number): FallEvent => ({
  id, camera: 'cam-1', pts: 100, track: 1, event_id: `ev-${id}`, kind: 'fall',
  evaluation: 'falling', combined: 0.8, severity: 'critical', path_scores: {},
  clip_window: null, review_id: null, review_status: null, created_at: '2026-10-28T10:00:00+00:00',
})

describe('the alarm queue', () => {
  it('the first alarm is the current one', () => {
    const state = enqueue({ current: null, queued: [] }, ev(1))
    expect(state.current?.id).toBe(1)
    expect(state.queued).toEqual([])
  })

  it('the ones behind it wait their turn', () => {
    let state: AlarmState = { current: null, queued: [] }
    state = enqueue(state, ev(1))
    state = enqueue(state, ev(2))
    state = enqueue(state, ev(3))
    expect(state.current?.id).toBe(1)
    expect(state.queued.map(e => e.id)).toEqual([2, 3])
  })

  it('acknowledging brings the next one forward', () => {
    let state: AlarmState = { current: null, queued: [] }
    state = enqueue(state, ev(1))
    state = enqueue(state, ev(2))
    state = acknowledge(state)
    expect(state.current?.id).toBe(2)
    expect(state.queued).toEqual([])
  })

  it('acknowledging the last one leaves nothing', () => {
    const state = acknowledge(enqueue({ current: null, queued: [] }, ev(1)))
    expect(state.current).toBeNull()
  })
})

describe('the arm delay', () => {
  it('an alarm cannot be acknowledged before it has been on screen', () => {
    expect(canAcknowledge(1000, 1000)).toBe(false)
    expect(canAcknowledge(1000 + ARM_DELAY_MS - 1, 1000)).toBe(false)
  })

  it('after the delay it can', () => {
    expect(canAcknowledge(1000 + ARM_DELAY_MS, 1000)).toBe(true)
  })
})
```

Create `frontend/src/products/fall_detection/alerts/badge.test.ts`:

```ts
import { describe, expect, it } from 'vitest'
import { unseenCount } from './badge'
import type { FallEvent } from './severity'

const ev = (id: number, over: Partial<FallEvent> = {}): FallEvent => ({
  id, camera: 'cam-1', pts: 100, track: 1, event_id: `ev-${id}`, kind: 'fall',
  evaluation: 'falling', combined: 0.8, severity: 'critical', path_scores: {},
  clip_window: null, review_id: null, review_status: null, created_at: '2026-10-28T10:00:00+00:00',
  ...over,
})

describe('the rail badge', () => {
  it('counts what arrived after the cursor', () => {
    expect(unseenCount([ev(3), ev(2), ev(1)], 1)).toBe(2)
  })

  it('counts nothing when the operator has seen everything', () => {
    expect(unseenCount([ev(3), ev(2), ev(1)], 3)).toBe(0)
  })

  it('does not count the quiet events', () => {
    const events = [ev(2, { severity: null, evaluation: 'upright' }), ev(1, { severity: null, evaluation: 'upright' })]
    expect(unseenCount(events, 0)).toBe(0)
  })
})
```

- [ ] **Step 2: Run the tests to verify they fail**

```
cd frontend/src/products/fall_detection && npx vitest run alerts
```

Expected: FAIL — three "Cannot find module './severity' / './queue' / './badge'" errors.

- [ ] **Step 3: Write `alerts/severity.ts`**

```ts
/** What the product saw, and where each kind of event speaks.
 *
 * The three channels are the house's: a quiet notice, the alarm card, and the rail badge.
 * Which one an event uses is a rule, not a taste -- the rule is what decides whether an
 * operator is interrupted, so it lives here as a pure function (phase-5 §3).
 */

export type Severity = 'critical' | 'warning'

export type FallEvent = {
  id: number
  camera: string
  pts: number | null
  track: number | null
  event_id: string | null
  kind: string | null
  evaluation: string
  combined: number
  severity: Severity | null
  path_scores: Record<string, number | null>
  clip_window: [number, number] | null
  review_id: number | null
  review_status: string | null
  created_at: string
}

export type Channels = { toast: boolean; card: boolean; badge: boolean }

/** Where an event speaks.
 *
 * A confirmed fall takes over: the person is on the floor. A candidate below the confirm
 * threshold takes over too: it was queued, and a queued candidate is a human waiting on it.
 * A confirmed lying or immobile only notifies -- nobody is queued on it. The closed upright
 * reading is the false alarm the product rejected, and it says nothing at all.
 */
export function channels(event: FallEvent): Channels {
  if (!event.severity) return { toast: false, card: false, badge: false }
  const queued = event.review_id != null
  return { toast: true, card: event.severity === 'critical' || queued, badge: true }
}
```

- [ ] **Step 4: Write `alerts/queue.ts`**

```ts
/** The alarm queue: one card at a time, the rest waiting behind it.
 *
 * A fall is not a thing the operator can batch: the first one gets the card, and the ones
 * that arrive while it is up wait their turn with their count on it (phase-5 §3).
 */
import type { FallEvent } from './severity'

/** Attendance's arm delay: an alarm cannot be dismissed before it has been seen. */
export const ARM_DELAY_MS = 1000

export type AlarmState = { current: FallEvent | null; queued: FallEvent[] }

export function enqueue(state: AlarmState, event: FallEvent): AlarmState {
  if (!state.current) return { current: event, queued: state.queued }
  return { current: state.current, queued: [...state.queued, event] }
}

export function acknowledge(state: AlarmState): AlarmState {
  const [next, ...rest] = state.queued
  return { current: next ?? null, queued: rest }
}

/** Whether the card may be acknowledged yet. */
export function canAcknowledge(now: number, shownAt: number, armDelayMs: number = ARM_DELAY_MS): boolean {
  return now - shownAt >= armDelayMs
}
```

- [ ] **Step 5: Write `alerts/badge.ts`**

```ts
/** The rail badge: the alert-worthy events the operator has not looked at yet.
 *
 * A client-side counter over the feed's ids -- the product has no operator session, and a
 * server-side unread flag would be a second source of truth (phase-5 §3).
 */
import { channels, type FallEvent } from './severity'

export function unseenCount(events: FallEvent[], seenCursor: number): number {
  return events.filter(e => e.id > seenCursor && channels(e).badge).length
}
```

- [ ] **Step 6: Run the tests to verify they pass**

```
cd frontend/src/products/fall_detection && npx vitest run alerts
```

Expected: PASS — 3 files, 11 tests.

- [ ] **Step 7: Commit** (only after the user approves)

```bash
git add frontend/src/products/fall_detection/alerts
git commit -m "feat(fall): the alarm's rules -- who is interrupted, who waits, what the badge counts"
```

---

### Task 7: The client and the formatters

**Files:**
- Create: `frontend/src/products/fall_detection/api.ts`
- Create: `frontend/src/products/fall_detection/api.test.ts`
- Create: `frontend/src/products/fall_detection/format.ts`
- Create: `frontend/src/products/fall_detection/format.test.ts`

**Interfaces:**
- Consumes: `FallEvent` from `alerts/severity.ts` (Task 6).
- Produces:
  - `api.ts`: `const API = '/fall_detection/api'`; `type Status = { device: string | null; weights_verified: boolean; classifier_ready: boolean; paths_live: string[] }`; `type Review = { id: number; camera: string; pts: number | null; track: number | null; kind: string | null; evaluation: string; combined: number; path_scores: Record<string, number | null>; clip_window: [number, number] | null; status: string; note: string | null; created_at: string }`; `status(): Promise<Status>`; `cameras(): Promise<string[]>`; `candidates(): Promise<Review[]>`; `events(since: number, camera?: string): Promise<FallEvent[]>`; `review(id: number, verdict: string): Promise<void>`.
  - `format.ts`: `seconds(t: number | null): string`; `pct(v: number | null | undefined): string`; `severityChip(severity: string | null): 'chipRed' | 'chipAmber' | 'chipGreen'`; `evaluationLabel(evaluation: string): string`.

- [ ] **Step 1: Write the failing tests**

Create `frontend/src/products/fall_detection/api.test.ts`:

```ts
import { afterEach, describe, expect, it, vi } from 'vitest'
import { candidates, cameras, events, review, status } from './api'

afterEach(() => { vi.unstubAllGlobals() })

function stubFetch(): string[] {
  const calls: string[] = []
  vi.stubGlobal('fetch', async (url: string) => {
    calls.push(url)
    return { ok: true, status: 200, json: async () => [] } as unknown as Response
  })
  return calls
}

describe('the calls the tab makes', () => {
  it('each one asks the product\'s own service under its own prefix', async () => {
    const calls = stubFetch()
    await status(); await cameras(); await candidates()
    expect(calls).toEqual([
      '/fall_detection/api/status',
      '/fall_detection/api/cameras',
      '/fall_detection/api/candidates',
    ])
  })

  it('the events call carries the cursor', async () => {
    const calls = stubFetch()
    await events(12)
    expect(calls).toEqual(['/fall_detection/api/events?since=12'])
  })

  it('the events call names the camera when there is one', async () => {
    const calls = stubFetch()
    await events(12, 'cam 1')
    expect(calls).toEqual(['/fall_detection/api/events?since=12&camera=cam%201'])
  })

  it('the verdict is posted to the review row', async () => {
    const calls = stubFetch()
    await review(7, 'acknowledged')
    expect(calls).toEqual(['/fall_detection/api/review/7'])
  })
})
```

Create `frontend/src/products/fall_detection/format.test.ts`:

```ts
import { describe, expect, it } from 'vitest'
import { evaluationLabel, pct, seconds, severityChip } from './format'

describe('the formatters', () => {
  it('a clock is the sidecar\'s frame time', () => {
    expect(seconds(0)).toBe('1970-01-01 00:00:00')
    expect(seconds(null)).toBe('—')
  })

  it('a score keeps two decimals', () => {
    expect(pct(0.812)).toBe('0.81')
    expect(pct(null)).toBe('—')
    expect(pct(undefined)).toBe('—')
  })

  it('the severity picks the house chip', () => {
    expect(severityChip('critical')).toBe('chipRed')
    expect(severityChip('warning')).toBe('chipAmber')
    expect(severityChip(null)).toBe('chipGreen')
  })

  it('the evaluation reads as a word', () => {
    expect(evaluationLabel('falling')).toBe('falling')
    expect(evaluationLabel('upright')).toBe('upright')
  })
})
```

- [ ] **Step 2: Run the tests to verify they fail**

```
cd frontend/src/products/fall_detection && npx vitest run api.test.ts format.test.ts
```

Expected: FAIL — "Cannot find module './api'" and "Cannot find module './format'".

- [ ] **Step 3: Write `api.ts`**

```ts
/** The product's own service, under its own prefix.
 *
 * Nothing here reads Redis or opens a socket: the tab asks the product's four REST routes
 * and the platform's published frames (live/FrameLayer). The confirmed fall reaches the
 * platform over `iris:alert:<cam>`; this is the operator's own surface.
 */
import type { FallEvent } from './alerts/severity'

const API = '/fall_detection/api'

export type Status = {
  device: string | null
  weights_verified: boolean
  classifier_ready: boolean
  paths_live: string[]
}

export type Review = {
  id: number
  camera: string
  pts: number | null
  track: number | null
  kind: string | null
  evaluation: string
  combined: number
  path_scores: Record<string, number | null>
  clip_window: [number, number] | null
  status: string
  note: string | null
  created_at: string
}

async function read<T>(path: string): Promise<T> {
  const res = await fetch(`${API}${path}`, { signal: AbortSignal.timeout(4000) })
  if (!res.ok) throw new Error(`${path}: ${res.status}`)
  return res.json() as Promise<T>
}

export const status = () => read<Status>('/status')
export const cameras = () => read<string[]>('/cameras')
export const candidates = () => read<Review[]>('/candidates')

/** The history since the cursor -- only what arrived while the operator was away. */
export const events = (since: number, camera?: string) =>
  read<FallEvent[]>(`/events?since=${since}${camera ? `&camera=${encodeURIComponent(camera)}` : ''}`)

/** The operator's verdict on a queued candidate. */
export async function review(id: number, verdict: string): Promise<void> {
  const res = await fetch(`${API}/review/${id}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ status: verdict }),
    signal: AbortSignal.timeout(4000),
  })
  if (!res.ok) throw new Error(`review ${id}: ${res.status}`)
}
```

- [ ] **Step 4: Write `format.ts`**

```ts
/** The little formats the house shows everywhere. */

/** The sidecar's frame clock, as a readable stamp. */
export const seconds = (t: number | null): string =>
  t == null ? '—' : new Date(t * 1000).toISOString().replace('T', ' ').slice(0, 19)

/** A score, to two decimals. */
export const pct = (v: number | null | undefined): string => (v == null ? '—' : v.toFixed(2))

/** The house chip for a severity. */
export const severityChip = (severity: string | null): 'chipRed' | 'chipAmber' | 'chipGreen' =>
  severity === 'critical' ? 'chipRed' : severity === 'warning' ? 'chipAmber' : 'chipGreen'

/** The evaluation, as the product names it. */
export const evaluationLabel = (evaluation: string): string => evaluation
```

- [ ] **Step 5: Run the tests to verify they pass**

```
cd frontend/src/products/fall_detection && npx vitest run api.test.ts format.test.ts
```

Expected: PASS — 2 files, 9 tests.

- [ ] **Step 6: Commit** (only after the user approves)

```bash
git add frontend/src/products/fall_detection/api.ts frontend/src/products/fall_detection/api.test.ts frontend/src/products/fall_detection/format.ts frontend/src/products/fall_detection/format.test.ts
git commit -m "feat(fall): the client and the formatters -- the calls the tab makes"
```

---

### Task 8: The tab in the house style

The whole tab, rebuilt: the entry, the shell, the rail with the IRIS mark, the two pages, and the camera tile that draws the engine's own frame. The alarm comes in Tasks 9 and 10 — this task delivers the board the alarm will sit in.

**Files:**
- Modify: `frontend/src/products/fall_detection/index.tsx` (becomes the entry)
- Create: `frontend/src/products/fall_detection/App.tsx`
- Create: `frontend/src/products/fall_detection/ui/Shell.tsx`
- Create: `frontend/src/products/fall_detection/ui/ui.tsx`
- Create: `frontend/src/products/fall_detection/pages/LivePage.tsx`
- Create: `frontend/src/products/fall_detection/pages/ReviewPage.tsx`
- Modify: `frontend/src/products/fall_detection/live/FrameLayer.tsx`

**Interfaces:**
- Consumes: `api.ts` (Task 7), `format.ts` (Task 7), `alerts/severity.ts` `FallEvent` (Task 6), `app.css` (Task 4).
- Produces:
  - `ui/ui.tsx`: `Icon({ d, stroke?, size? })`, `ICONS` (`live`, `review`, `fall`), `Confirm({ title, children, confirmLabel, danger?, onConfirm, onCancel })`.
  - `ui/Shell.tsx`: `Shell({ route, unseen, children })` where `route: { path: string }`.
  - `App.tsx`: `App()`, `Route { path: string; params: URLSearchParams }`, `PageProps` (the service data every page takes).
  - `pages/LivePage.tsx`: `LivePage(props: PageProps)`.
  - `pages/ReviewPage.tsx`: `ReviewPage(props: PageProps)`.
  - `live/FrameLayer.tsx`: `FrameLayer({ camera, intervalMs?, className? })` — the `className` is new; the endpoint knowledge does not move.

- [ ] **Step 1: Write `ui/ui.tsx`**

Create `frontend/src/products/fall_detection/ui/ui.tsx`:

```tsx
/** Small shared pieces: the nav icons, the dialogs. The house's own (barber/ui/ui.tsx). */

export function Icon({ d, stroke = false, size = 20 }: { d: string; stroke?: boolean; size?: number }) {
  return (
    <svg viewBox="0 0 24 24" width={size} height={size} aria-hidden="true">
      <path
        d={d}
        fill={stroke ? 'none' : 'currentColor'}
        stroke={stroke ? 'currentColor' : 'none'}
        strokeWidth={stroke ? 1.9 : 0}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}

export const ICONS = {
  live: 'M4 5h16a1 1 0 0 1 1 1v12a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1zm4 4v6l5-3z',
  review: 'M8 4h8a2 2 0 0 1 2 2v14l-6-3-6 3V6a2 2 0 0 1 2-2z',
  fall: 'M12 3 2.5 20h19zM12 10v4.5m0 2.7v.1',
}
```

- [ ] **Step 2: Write `ui/Shell.tsx`**

Create `frontend/src/products/fall_detection/ui/Shell.tsx`:

```tsx
/** The rail and the page: the house's shell (barber/ui/Shell.tsx), with two pages instead of
 * seven -- a fall product has two things to look at, and a third would be a page with no
 * data behind it. */
import type { ReactNode } from 'react'
import { BrandLogo } from '../../../shared/ui/BrandLogo'
import { productHref } from '../../../shared/router'
import { Icon, ICONS } from './ui'

const NAV: { path: string; label: string; icon: string; stroke: boolean }[] = [
  { path: '/', label: 'Live', icon: ICONS.live, stroke: false },
  { path: '/review', label: 'Review', icon: ICONS.review, stroke: true },
]

export function Shell({ route, unseen, children }: {
  route: { path: string }; unseen: number; children: ReactNode
}) {
  return (
    <div className="shell">
      <nav className="rail" aria-label="Fall detection">
        <span className="railLogo" title="IRIS · Fall Detection">
          <BrandLogo size={32} />
          <span className="railBrandName">IRIS</span>
        </span>
        {NAV.map(item => {
          const badge = item.path === '/' && unseen > 0
          return (
            <a
              key={item.path}
              className={route.path === item.path ? 'railBtn railActive' : 'railBtn'}
              href={productHref(item.path)}
              title={item.label}
              aria-label={badge ? `${item.label} (${unseen} new)` : item.label}
              aria-current={route.path === item.path ? 'page' : undefined}
            >
              <Icon d={item.icon} stroke={item.stroke} />
              {badge && <span className="railCount">{unseen}</span>}
            </a>
          )
        })}
      </nav>
      <main className="page">{children}</main>
    </div>
  )
}
```

- [ ] **Step 3: Write `App.tsx`**

Create `frontend/src/products/fall_detection/App.tsx`:

```tsx
/** The tab: the hash route, the shell, and the service's own data.
 *
 * The product's service is polled, not streamed: AC-F33.1 and AC-F34.1 freeze the tab out of
 * WebSockets and Redis, and the platform's own alert path over `iris:alert:<cam>` is what
 * makes the fall urgent. This is the operator's surface.
 */
import { useCallback, useEffect, useState } from 'react'
import { onRouteChange, productPath } from '../../shared/router'
import { cameras as fetchCameras, candidates, review, status, type Review, type Status } from './api'
import { Shell } from './ui/Shell'
import { LivePage } from './pages/LivePage'
import { ReviewPage } from './pages/ReviewPage'

const REFRESH_MS = 5000

export interface Route {
  path: string
  params: URLSearchParams
}

function parseHash(): Route {
  const raw = productPath()
  const [path, q = ''] = raw.split('?')
  return { path: path || '/', params: new URLSearchParams(q) }
}

/** What every page takes: the service's own state. */
export interface PageProps {
  status: Status | null
  cameras: string[]
  camera: string
  setCamera: (id: string) => void
  reviews: Review[]
  error: string
  busy: number | null
  decide: (id: number, verdict: string) => Promise<void>
  reload: () => Promise<void>
}

export function App() {
  const [route, setRoute] = useState(parseHash)
  const [statusState, setStatusState] = useState<Status | null>(null)
  const [cameras, setCameras] = useState<string[]>([])
  const [camera, setCamera] = useState('')
  const [reviews, setReviews] = useState<Review[]>([])
  const [error, setError] = useState('')
  const [busy, setBusy] = useState<number | null>(null)

  useEffect(() => onRouteChange(() => setRoute(parseHash())), [])

  const reload = useCallback(async () => {
    try {
      const [s, c, r] = await Promise.all([status(), fetchCameras(), candidates()])
      setStatusState(s)
      setCameras(c)
      setCamera(prev => (prev && c.includes(prev) ? prev : (c[0] ?? '')))
      setReviews(r)
      setError('')
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc))
    }
  }, [])

  useEffect(() => {
    void reload()
    const id = setInterval(() => void reload(), REFRESH_MS)
    return () => clearInterval(id)
  }, [reload])

  const decide = useCallback(async (id: number, verdict: string) => {
    setBusy(id)
    try {
      await review(id, verdict)
      await reload()
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc))
    } finally {
      setBusy(null)
    }
  }, [reload])

  const props: PageProps = {
    status: statusState, cameras, camera, setCamera, reviews, error, busy, decide, reload,
  }

  return (
    <Shell route={route} unseen={0}>
      {route.path === '/review' ? <ReviewPage {...props} /> : <LivePage {...props} />}
    </Shell>
  )
}
```

> `unseen={0}` is replaced in Task 9, when the events feed exists. The prop is here from the
> start so the rail's badge contract is fixed before anything counts.

- [ ] **Step 4: Write `pages/LivePage.tsx`**

Create `frontend/src/products/fall_detection/pages/LivePage.tsx`:

```tsx
/** The Live page: the board. The camera wide, the side column beside it.
 *
 * The camera tile is never covered -- not by the alarm card, not by anything. The alarm
 * annotates the picture (a ring on the tile that fell) rather than sitting on top of it
 * (phase-5 §3).
 */
import { FrameLayer } from '../live/FrameLayer'
import { pct, seconds } from '../format'
import type { PageProps } from '../App'

export function LivePage({ status, cameras, camera, setCamera, error }: PageProps) {
  const ready = status?.classifier_ready === true && status.weights_verified === true
  return (
    <>
      <div className="pageHead">
        <div className="pageHeadMain">
          <h1>Fall Detection</h1>
          {status && (
            <span className="subLine">
              device {status.device ?? '—'} · weights {status.weights_verified ? 'verified' : 'not verified'} ·
              classifier {status.classifier_ready ? 'ready' : 'NOT ready'} · paths {status.paths_live.join(', ')}
            </span>
          )}
        </div>
        <div className="pageHeadTools">
          <span className={ready ? 'chip chipGreen' : 'chip chipAmber'}>{ready ? 'ready' : 'not ready'}</span>
          {cameras.length > 1 && (
            <select value={camera} onChange={e => setCamera(e.target.value)} aria-label="camera">
              {cameras.map(id => <option key={id} value={id}>{id}</option>)}
            </select>
          )}
        </div>
      </div>

      {error && <div className="errorBar">the fall service: {error}</div>}

      <div className="kpiGrid">
        <div className="kpi"><div className="kpiLabel">falls</div><div className="kpiValue">—</div></div>
        <div className="kpi"><div className="kpiLabel">cameras</div><div className="kpiValue">{cameras.length}</div></div>
        <div className="kpi"><div className="kpiLabel">warnings</div><div className="kpiValue">—</div></div>
        <div className="kpi"><div className="kpiLabel">awaiting review</div><div className="kpiValue">—</div></div>
        <div className="kpi"><div className="kpiLabel">last score</div><div className="kpiValue">—</div></div>
      </div>

      <div className="liveGrid">
        <div className="feedTile">
          {camera
            ? <div className="feed"><FrameLayer camera={camera} /></div>
            : <div className="feed" />}
          <div className="feedCaption">
            <span className="feedName">{camera || 'no camera'}</span>
            <span className="feedClip">the engine's published frame</span>
          </div>
        </div>
        <div className="sideCol">
          <div className="card eventsCard">
            <div className="sectionLabel">the events</div>
            <ul className="eventList"><li className="eventRow"><span className="eventEval">—</span></li></ul>
          </div>
        </div>
      </div>
    </>
  )
}
```

> The KPI values and the event list are placeholders **in this task only**: Task 9 fills them
> from the events feed. This task delivers the board's shape and the house's styling; the
> numbers arrive with their data.

- [ ] **Step 5: Write `pages/ReviewPage.tsx`**

Create `frontend/src/products/fall_detection/pages/ReviewPage.tsx`:

```tsx
/** The Review page: the queue, in the house's table, with the three verdicts. */
import { pct, seconds } from '../format'
import type { PageProps } from '../App'

const VERDICTS = ['acknowledged', 'confirmed', 'dismissed'] as const

export function ReviewPage({ reviews, error, busy, decide }: PageProps) {
  return (
    <>
      <div className="pageHead">
        <div className="pageHeadMain">
          <h1>Review</h1>
          <span className="subLine">the candidates the product could not confirm</span>
        </div>
      </div>

      {error && <div className="errorBar">the fall service: {error}</div>}

      <div className="card">
        <div className="tableScroll">
          <table className="dataTable">
            <thead>
              <tr>
                <th>time</th><th>camera</th><th>track</th><th>kind</th><th>evaluation</th>
                <th>score</th><th>paths</th><th>status</th><th />
              </tr>
            </thead>
            <tbody>
              {reviews.map(row => (
                <tr key={row.id}>
                  <td>{row.pts == null ? row.created_at : seconds(row.pts)}</td>
                  <td>{row.camera}</td>
                  <td>{row.track ?? '—'}</td>
                  <td>{row.kind ?? '—'}</td>
                  <td>{row.evaluation}</td>
                  <td className="num">{pct(row.combined)}</td>
                  <td>
                    action {pct(row.path_scores.action)} · kinematic {pct(row.path_scores.kinematic)} ·
                    cold {pct(row.path_scores.cold)}
                  </td>
                  <td><span className={row.status === 'candidate' ? 'chip chipAmber' : 'chip chipGreen'}>{row.status}</span></td>
                  <td>
                    {row.status === 'candidate' && VERDICTS.map(v => (
                      <button key={v} type="button" className="btnGhost btnSmall"
                              disabled={busy === row.id} onClick={() => void decide(row.id, v)}>
                        {v}
                      </button>
                    ))}
                  </td>
                </tr>
              ))}
              {reviews.length === 0 && (
                <tr><td className="tableEmpty" colSpan={9}>nothing waiting for review</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </>
  )
}
```

- [ ] **Step 6: Replace `index.tsx` with the entry**

Overwrite `frontend/src/products/fall_detection/index.tsx`:

```tsx
/** Fall detection: this product's entry point. See barber/index.tsx. */
import '@fontsource/cairo/400.css'
import '@fontsource/cairo/600.css'
import '@fontsource/cairo/700.css'
import '@fontsource/cairo/800.css'
import './app.css'
import { App } from './App'

export default function FallDetectionProduct() {
  return <App />
}
```

- [ ] **Step 7: Give the frame layer the house's box**

Replace `frontend/src/products/fall_detection/live/FrameLayer.tsx` with:

```tsx
import { useEffect, useRef, useState } from 'react'
import { frameUrl } from './frameUrl'

type Props = {
  camera: string
  /** Gap between a frame finishing and the next request, so a slow frame never stacks requests. */
  intervalMs?: number
  /** The house's own box (.feed) -- the layer draws the picture and nothing else. */
  className?: string
}

/** The camera's latest frame, refreshed only after the previous request finished.
 *
 * Requesting on a blind timer makes the browser abort frames in flight once a few tabs share
 * the origin, and the visible result is stutter; waiting on onLoad/onError keeps exactly one
 * request open per layer.
 *
 * This module is the only place that knows the frame endpoint (tab.test.ts pins that).
 */
export function FrameLayer({ camera, intervalMs = 250, className }: Props) {
  const [tick, setTick] = useState(0)
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    setTick(0)
    return () => {
      if (timer.current !== null) clearTimeout(timer.current)
    }
  }, [camera])

  const scheduleNext = () => {
    if (timer.current !== null) clearTimeout(timer.current)
    timer.current = setTimeout(() => setTick(n => n + 1), intervalMs)
  }

  return (
    <img
      alt={`camera ${camera}`}
      className={className}
      src={`${frameUrl(camera)}?t=${tick}`}
      onLoad={scheduleNext}
      onError={scheduleNext}
    />
  )
}
```

The inline style is gone: the house's `.feed img` (absolute, 16/9 box, `object-fit: contain`,
`#101512`) sizes the picture, which is what makes the tile look like every other camera in
the house.

- [ ] **Step 8: Build and test**

```
cd frontend && export PATH="/f/_tools/node-v24.18.0-win-x64:$PATH" && npm run build && npm run test -w @iris/fall-detection
```

Expected: the build exits 0, and the suite is green — including `tab.test.ts`'s two
strengthened assertions (the only files that know `/frame` are `live/frameUrl.ts` and
`live/FrameLayer.tsx`, and exactly one file outside `live/` composes the layer).

- [ ] **Step 9: Commit** (only after the user approves)

```bash
git add frontend/src/products/fall_detection
git commit -m "feat(fall): the tab in the house style -- the rail, the mark, the board"
```

---

### Task 9: The events feed — the list, the badge, the notices

The tab polls `/events` and shows what arrived. The feed also drives the rail badge and the quiet notices.

**Files:**
- Create: `frontend/src/products/fall_detection/alerts/useEvents.ts`
- Modify: `frontend/src/products/fall_detection/App.tsx`
- Modify: `frontend/src/products/fall_detection/pages/LivePage.tsx`

**Interfaces:**
- Consumes: `api.events(since, camera?)` (Task 7), `channels` (Task 6), `unseenCount` (Task 6).
- Produces: `useEvents({ camera }: { camera?: string }): EventsFeed` where

```ts
export interface EventsFeed {
  events: FallEvent[]          // newest first
  unseen: number               // the rail badge
  notices: FallEvent[]         // the quiet channel, newest first
  markSeen: (id: number) => void
  dismissNotice: (id: number) => void
}
```

- [ ] **Step 1: Write the hook**

Create `frontend/src/products/fall_detection/alerts/useEvents.ts`:

```ts
/** The events feed: what the product has seen, polled.
 *
 * A poll, not a socket: AC-F33.1 and AC-F34.1 freeze the tab out of WebSockets, and the
 * platform's own alert path over `iris:alert:<cam>` is what makes a fall urgent -- this is
 * the operator's surface, two seconds behind is invisible next to the sidecar's cadence.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { events as fetchEvents } from '../api'
import { channels, type FallEvent } from './severity'
import { unseenCount } from './badge'

const POLL_MS = 2000
const KEEP = 100
const NOTICE_MS = 6000

export interface EventsFeed {
  events: FallEvent[]
  unseen: number
  notices: FallEvent[]
  markSeen: (id: number) => void
  dismissNotice: (id: number) => void
}

export function useEvents({ camera }: { camera?: string } = {}): EventsFeed {
  const [events, setEvents] = useState<FallEvent[]>([])
  const [notices, setNotices] = useState<FallEvent[]>([])
  const [seen, setSeen] = useState(0)
  const cursor = useRef(0)

  const poll = useCallback(async () => {
    try {
      const fresh = await fetchEvents(cursor.current, camera)
      if (!fresh.length) return
      cursor.current = Math.max(cursor.current, ...fresh.map(e => e.id))
      setEvents(prev => [...fresh, ...prev].slice(0, KEEP))
      const spoken = fresh.filter(e => channels(e).toast)
      if (spoken.length) setNotices(prev => [...spoken, ...prev].slice(0, 3))
    } catch {
      /* the service is down; the error bar on the page says so */
    }
  }, [camera])

  useEffect(() => {
    void poll()
    const id = setInterval(() => void poll(), POLL_MS)
    return () => clearInterval(id)
  }, [poll])

  const markSeen = useCallback((id: number) => {
    setSeen(prev => Math.max(prev, id))
  }, [])

  const dismissNotice = useCallback((id: number) => {
    setNotices(prev => prev.filter(e => e.id !== id))
  }, [])

  // A notice is a moment, not a queue: it goes quiet on its own.
  useEffect(() => {
    if (!notices.length) return
    const id = setTimeout(() => setNotices(prev => prev.slice(0, -1)), NOTICE_MS)
    return () => clearTimeout(id)
  }, [notices])

  return useMemo(() => ({
    events, notices, unseen: unseenCount(events, seen), markSeen, dismissNotice,
  }), [events, notices, seen, markSeen, dismissNotice])
}
```

- [ ] **Step 2: Wire the feed into `App.tsx`**

In `frontend/src/products/fall_detection/App.tsx`:

- import `useEvents` from `./alerts/useEvents` and `FallEvent` from `./alerts/severity`;
- add `import { useMemo } from 'react'` to the React import line;
- inside `App`, after the `decide` callback, add:

```tsx
  const feed = useEvents({ camera })
  const markSeen = useCallback((id: number) => feed.markSeen(id), [feed])
```

- add to `PageProps`:

```ts
  events: FallEvent[]
  notices: FallEvent[]
  unseen: number
  markSeen: (id: number) => void
  dismissNotice: (id: number) => void
```

- extend the `props` object:

```tsx
  const props: PageProps = {
    status: statusState, cameras, camera, setCamera, reviews, error, busy, decide, reload,
    events: feed.events, notices: feed.notices, unseen: feed.unseen,
    markSeen: feed.markSeen, dismissNotice: feed.dismissNotice,
  }
```

- replace `<Shell route={route} unseen={0}>` with `<Shell route={route} unseen={feed.unseen}>`.

> Remove the `useMemo` import if you did not use it; `tsc` runs with `noUnusedLocals`, and an
> unused import fails the build.

- [ ] **Step 3: Fill the Live page from the feed**

In `frontend/src/products/fall_detection/pages/LivePage.tsx`:

- change the signature to `export function LivePage({ status, cameras, camera, setCamera, error, events, notices, unseen, markSeen, dismissNotice }: PageProps) {`
- import `channels` from `../alerts/severity` and `severityChip` from `../format`;
- compute, above the `return`:

```tsx
  const falls = events.filter(e => e.severity === 'critical').length
  const warnings = events.filter(e => e.severity === 'warning').length
  const awaiting = events.filter(e => e.review_status === 'candidate').length
  const last = events[0]
```

- replace the KPI row's five tiles with:

```tsx
      <div className="kpiGrid">
        <div className="kpi"><div className="kpiLabel">falls</div><div className="kpiValue num">{falls}</div></div>
        <div className="kpi"><div className="kpiLabel">cameras</div><div className="kpiValue num">{cameras.length}</div></div>
        <div className="kpi"><div className="kpiLabel">warnings</div><div className="kpiValue num">{warnings}</div></div>
        <div className="kpi"><div className="kpiLabel">awaiting review</div><div className="kpiValue num">{awaiting}</div></div>
        <div className="kpi"><div className="kpiLabel">last score</div><div className="kpiValue num">{pct(last?.combined)}</div></div>
      </div>
```

- replace the event list with:

```tsx
          <ul className="eventList">
            {events.map(e => (
              <li key={e.id} className="eventRow">
                <span className={e.severity === 'critical' ? 'eventDot eventDotCritical'
                  : e.severity === 'warning' ? 'eventDot eventDotWarning' : 'eventDot'} />
                <span className="eventEval">{e.evaluation}</span>
                <span className="eventScore">{pct(e.combined)}</span>
                <span className={severityChip(e.severity)}>{e.severity ?? 'closed'}</span>
              </li>
            ))}
            {events.length === 0 && <li className="eventRow"><span className="eventEval">nothing seen yet</span></li>}
          </ul>
```

- render the quiet notices above the events card, inside the side column:

```tsx
        <div className="sideCol">
          {notices.map(n => (
            <div key={n.id} className={n.severity === 'critical' ? 'notice noticeCritical' : 'notice noticeWarning'}
                 onClick={() => dismissNotice(n.id)} role="status">
              {n.evaluation} · {pct(n.combined)} · {n.camera}
            </div>
          ))}
          <div className="card eventsCard">
```

- and mark the events as seen when the Live page shows them — add just above the `return`:

```tsx
  // The Live page is where the operator looks; showing them is what clears the badge.
  useEffect(() => {
    if (events.length) markSeen(events[0].id)
  }, [events, markSeen])
```

with `useEffect` added to the React import.

- [ ] **Step 4: Build and test**

```
cd frontend && export PATH="/f/_tools/node-v24.18.0-win-x64:$PATH" && npm run build && npm run test -w @iris/fall-detection
```

Expected: the build exits 0 and the suite is green.

- [ ] **Step 5: Commit** (only after the user approves)

```bash
git add frontend/src/products/fall_detection
git commit -m "feat(fall): the events feed -- the list, the badge and the quiet notices"
```

---

### Task 10: The alarm card — the thing the house does not have

The card is in the page's flow, never over the camera. One at a time, the rest queued behind it, ENTER to acknowledge, and a ring on the camera that fell.

**Files:**
- Create: `frontend/src/products/fall_detection/alerts/AlarmCard.tsx`
- Modify: `frontend/src/products/fall_detection/App.tsx`
- Modify: `frontend/src/products/fall_detection/pages/LivePage.tsx`
- Modify: `frontend/src/products/fall_detection/pages/ReviewPage.tsx`

**Interfaces:**
- Consumes: `alerts/queue.ts` (`enqueue`, `acknowledge`, `canAcknowledge`, `ARM_DELAY_MS`), `alerts/severity.ts` (`channels`), `api.review` (Task 7), `live/FrameLayer` (Task 8).
- Produces: `AlarmCard({ event, queued, shownAt, onAcknowledge, onShowCamera }: {...})` — rendered by both pages in their own slot.

- [ ] **Step 1: Write the card**

Create `frontend/src/products/fall_detection/alerts/AlarmCard.tsx`:

```tsx
/** The alarm card: a fall, in the page's flow.
 *
 * It never covers the camera. The card sits in the side column beside the picture, the
 * camera that fell is ringed, and the operator keeps watching while they decide -- which is
 * the whole point: the fall happened in that frame (phase-5 §3).
 */
import { useEffect, useRef, useState } from 'react'
import { FrameLayer } from '../live/FrameLayer'
import { pct, seconds } from '../format'
import { canAcknowledge } from './queue'
import type { FallEvent } from './severity'

export function AlarmCard({ event, queued, shownAt, onAcknowledge, onShowCamera }: {
  event: FallEvent
  queued: number
  shownAt: number
  onAcknowledge: () => void
  onShowCamera: () => void
}) {
  const [now, setNow] = useState(Date.now())
  const armed = canAcknowledge(now, shownAt)
  const card = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 250)
    return () => clearInterval(id)
  }, [])

  // ENTER acknowledges the card while it holds focus -- and the card does not steal the
  // focus from whatever the operator is doing.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Enter') return
      if (document.activeElement !== card.current) return
      if (!armed) return
      onAcknowledge()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [armed, onAcknowledge])

  const critical = event.severity === 'critical'
  return (
    <div ref={card} className="alarmCard" role="alert" aria-live="assertive"
         tabIndex={0} aria-label={`${event.evaluation} on ${event.camera}`}>
      <div className={critical ? 'alarmBar alarmCritical' : 'alarmBar alarmWarning'}>
        <span>{critical ? 'FALL' : 'ALERT'}</span>
        <span>{event.camera}</span>
        {queued > 0 && <span className="alarmQueued">{queued} waiting</span>}
      </div>
      <div className="alarmBody">
        <div className="alarmFacts">
          <span className="alarmEval">{event.evaluation}</span>
          <span className="alarmMeta">score {pct(event.combined)} · {seconds(event.pts)}</span>
          <span className="alarmMeta">track {event.track ?? '—'}</span>
        </div>
        <div className="alarmThumb">
          <div className="feed"><FrameLayer camera={event.camera} intervalMs={500} /></div>
        </div>
      </div>
      <div className="alarmActions">
        <button type="button" className="btnPrimary btnSmall" disabled={!armed} onClick={onAcknowledge}>
          acknowledge
        </button>
        <button type="button" className="btnGhost btnSmall" onClick={onShowCamera}>
          show camera
        </button>
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Hold the queue in `App.tsx`**

In `frontend/src/products/fall_detection/App.tsx`:

- import `{ acknowledge as popAlarm, enqueue } from './alerts/queue'` and `{ AlarmCard } from './alerts/AlarmCard'`;
- add the state and the handlers inside `App`, after the `feed` (Task 9):

```tsx
  const [alarms, setAlarms] = useState<AlarmState>({ current: null, queued: [] })
  const [shownAt, setShownAt] = useState(0)

  // A card-worthy event joins the queue the moment it arrives.
  useEffect(() => {
    const cardWorthy = feed.events.filter(e => channels(e).card)
    if (!cardWorthy.length) return
    setAlarms(prev => cardWorthy.reverse().reduce(enqueue, prev))
    setShownAt(Date.now())
  }, [feed.events])
```

with `import { useState } from 'react'` already present, plus `import { channels, type FallEvent } from './alerts/severity'` and `import { acknowledge as popAlarm, enqueue, type AlarmState } from './alerts/queue'`.

- add the acknowledge handler:

```tsx
  const ackAlarm = useCallback(async () => {
    const current = alarms.current
    setAlarms(popAlarm)
    if (current?.review_id != null) {
      try { await review(current.review_id, 'acknowledged') } catch { /* the error bar says so */ }
    }
  }, [alarms.current])
```

- add the camera jump:

```tsx
  const showCamera = useCallback((cam: string) => { setCamera(cam) }, [])
```

- add to `PageProps`:

```ts
  alarm: FallEvent | null
  alarmQueued: number
  alarmShownAt: number
  ackAlarm: () => Promise<void>
  showCamera: (cam: string) => void
```

- extend `props`:

```tsx
    alarm: alarms.current, alarmQueued: alarms.queued.length, alarmShownAt: shownAt,
    ackAlarm, showCamera,
```

- [ ] **Step 3: Host the card in the Live page**

In `frontend/src/products/fall_detection/pages/LivePage.tsx`:

- import `AlarmCard` from `../alerts/AlarmCard`;
- add `alarm, alarmQueued, alarmShownAt, ackAlarm, showCamera` to the destructured props;
- render the card at the top of the side column, above the notices:

```tsx
        <div className="sideCol">
          {alarm && (
            <AlarmCard event={alarm} queued={alarmQueued} shownAt={alarmShownAt}
                       onAcknowledge={() => void ackAlarm()}
                       onShowCamera={() => showCamera(alarm.camera)} />
          )}
```

- ring the camera that fell — inside the camera tile:

```tsx
          {camera
            ? <div className={alarm && alarm.camera === camera
                ? (alarm.severity === 'critical' ? 'feed feedRing' : 'feed feedRingWarn')
                : 'feed'}>
                <FrameLayer camera={camera} />
                {alarm && alarm.camera === camera && (
                  <span className={alarm.severity === 'critical' ? 'feedBadge feedLive' : 'feedBadge'}>
                    {alarm.severity === 'critical' ? 'FALL' : 'ALERT'}
                  </span>
                )}
              </div>
            : <div className="feed" />}
```

> The house's `.feedBadge` is positioned top-right; `.feedLive` is the red one with the
> pulsing dot. The ring classes come from Task 4's `app.css`.

- [ ] **Step 4: Host the card in the Review page**

In `frontend/src/products/fall_detection/pages/ReviewPage.tsx`:

- import `AlarmCard` from `../alerts/AlarmCard`;
- add `alarm, alarmQueued, alarmShownAt, ackAlarm, showCamera` to the props;
- render it above the table, in the flow:

```tsx
      {alarm && (
        <AlarmCard event={alarm} queued={alarmQueued} shownAt={alarmShownAt}
                   onAcknowledge={() => void ackAlarm()}
                   onShowCamera={() => showCamera(alarm.camera)} />
      )}
```

- [ ] **Step 5: Build and test**

```
cd frontend && export PATH="/f/_tools/node-v24.18.0-win-x64:$PATH" && npm run build && npm run test -w @iris/fall-detection
```

Expected: the build exits 0 and the suite is green. `tab.test.ts`'s guards still hold — the
card draws no canvas and names no socket.

- [ ] **Step 6: Commit** (only after the user approves)

```bash
git add frontend/src/products/fall_detection
git commit -m "feat(fall): the alarm card -- a fall in the page, never over the camera"
```

---

### Task 11: The proof, and the report

**Files:**
- Modify: `products/fall_detection/docs/phases/phase-5-ui-redesign.md` (the `.toast` → `.notice` correction, noted in Task 4)
- Create: `products/fall_detection/docs/phases/phase-5-delivery-report.md`

**Interfaces:**
- Consumes: everything above.
- Produces: the evidence that the phase is done, in the repo's own format.

- [ ] **Step 1: Run the frontend suite and the build**

```
cd frontend && export PATH="/f/_tools/node-v24.18.0-win-x64:$PATH" && npm run test -w @iris/fall-detection && npm run build
```

Expected: every test passes; the build exits 0.

- [ ] **Step 2: Run the product's Python suite**

```
cd products/fall_detection && python -m pytest -q
```

Expected: all green, including the new `test_storage_event.py`, `test_worker_event.py` and
`test_events_api.py`. The corpus replays stay deselected (Global Constraint 6).

- [ ] **Step 3: Run the root suite's fall group**

```
python -m pytest tests -q -k fall
```

Expected: green.

- [ ] **Step 4: Correct the spec's stylesheet line**

In `products/fall_detection/docs/phases/phase-5-ui-redesign.md`, §5, change `.toast*` to
`.notice` in the list of classes `app.css` carries, and add the reason:

```
`.notice` (the product's own quiet channel -- the house's `.toast` is a fixed overlay, and this
product has no fixed overlay), the alarm card's classes, and the fall product's own
```

- [ ] **Step 5: Write the delivery report**

Create `products/fall_detection/docs/phases/phase-5-delivery-report.md` in the house's format
— the Phase 4 report (`products/fall_detection/docs/phases/phase-4-delivery-report.md`) is the
template. It must carry:

- **What was built** — the house stylesheet, the shell and the two pages, the alarm card, the
  events feed, the `fall_event` table, the `/events` endpoint.
- **The coverage** — every test file that exists for the phase and what it pins, with the
  counts from the run above.
- **The evidence** — the actual command lines and their real output (the build exit code, the
  vitest and pytest counts).
- **What was deliberately not built** — the spec's §7, unchanged.
- **The gap that remains** — the tab has never been rendered in a browser: the build and the
  unit tests are the proof here, and the first real look at it happens on the 5090. Say so
  plainly.

- [ ] **Step 6: Commit** (only after the user approves)

```bash
git add products/fall_detection/docs
git commit -m "docs(fall): the Phase 5 delivery report -- the evidence and what remains"
```

---

## The plan's own self-review

**Spec coverage.** Every section of `phase-5-ui-redesign.md` maps to a task:

| Spec | Tasks |
|---|---|
| §1 the house system | 4 (the stylesheet), 8 (the shell) |
| §2 the shape (the board, two pages) | 8 |
| §3 the alarm (annotate, never cover; the ring; the thumbnail; one at a time; the arm delay; the acknowledge; the badge) | 6, 9, 10 |
| §4.1 the table | 1 |
| §4.2 the write | 2 |
| §4.3 the endpoint | 3 |
| §4.4 the poll, not a socket | 7, 9 |
| §5 the files | 4, 6, 7, 8, 9, 10 |
| §6 the tests | 1, 2, 3, 4, 5, 6, 7 |
| §7 what is not built | 11 (the report restates it) |
| §8 how it is verified | 11 |

**Gaps found and closed while writing this plan:**

1. **`npm run build` fails on this branch** — the product's guard tests import `node:fs` and
   the dashboard has no `@types/node`. Not mentioned in the spec; fixed in Task 4, which is
   the only place the build becomes green.
2. **The house's `.toast` is a fixed overlay** — the spec listed it among the classes to
   copy, which would have put a fixed overlay over the camera. The product gets its own
   `.notice` in the flow instead; the spec's §5 is corrected in Task 11.
3. **The confirmed fall has no review row** — so the acknowledge cannot always post. The
   spec's §3 states this; Task 6's `channels` makes the difference explicit (a queued
   candidate has a `review_id`, a confirmed fall does not), and Task 10's `ackAlarm` writes
   only when there is a row to write.

**Type consistency.** `FallEvent` is defined once, in `alerts/severity.ts` (Task 6), and
imported by `api.ts` (Task 7), `alerts/badge.ts` (Task 6), `alerts/queue.ts` (Task 6),
`alerts/useEvents.ts` (Task 9) and `alerts/AlarmCard.tsx` (Task 10). `AlarmState` is defined
once, in `alerts/queue.ts`, and used by `App.tsx` (Task 10). `PageProps` grows in Task 9
(`events`, `notices`, `unseen`, `markSeen`, `dismissNotice`) and again in Task 10 (`alarm`,
`alarmQueued`, `alarmShownAt`, `ackAlarm`, `showCamera`) — each page destructures only what
it uses, and `tsc`'s `noUnusedLocals` catches a mismatch at build time.

**No placeholders.** Every step carries real code or an exact copy instruction with line
ranges. The two places the plan says "the values arrive with their data" (the KPI tiles and
the event list in Task 8) are explicitly marked as filled by Task 9, and the build is run at
the end of Task 8 so the placeholders are never shipped.
