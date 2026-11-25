# Phase 5 — the fall tab in the IRIS house style

**Date:** 2026-10-28 · **Status:** DRAFT — approved in chat (the sketch, and the alarm's placement), awaiting the written review
**Scope:** the frontend redesign of `products/fall_detection`, and the one backend endpoint it needs.

---

## 0. Why

The fall product's tab was written to the letter of the backend contract and to nothing else: 166
lines of inline styles, no stylesheet, no shell, no logo. Next to the barber tab in the same
dashboard it looks like a different application. The house style already exists — attendance and
barber share it — and a customer pays for one product. The tab must look like IRIS.

The redesign changes **only what the fall product needs to differ** (a fall alarm, a confirmed-fall
feed). Everything else is the house, reused.

---

## 1. The house system — the source of truth

This is not a new design. It is the one attendance and barber already use, cited so the
implementation copies rather than invents.

| Concern | Where it lives | What to take |
|---|---|---|
| Tokens | `attendance/app.css:5-23` and `barber/app.css:10-45` — the values are **byte-identical** for the greens, `--ground`, `--card`, `--ink`, `--ink-soft`, `--line`, `--red-500`, `--red-50`, `--amber-600`, `--amber-50`, `--radius: 14px`, `--shadow` | the `:root` block verbatim; the fall product adds only `--fall-critical` (an alias of `--red-500`) if a name reads better |
| Type | `frontend/index.html:11-14` loads Cairo globally; `barber/package.json` adds `@fontsource/cairo` | `@fontsource/cairo` as a dependency, the barber way |
| The IRIS mark | `frontend/src/shared/ui/BrandLogo.tsx:4-25` — **already shared**, attendance imports it at `attendance/App.tsx:10` | import it; do **not** copy the SVG again (it is already duplicated twice; a third copy is not the answer) |
| The shell | `barber/ui/Shell.tsx:32-71` + `barber/app.css:68-146` | `.shell` → `.rail` (56px, `--green-900`) + `.page` (`--ground`, 20px radius) |
| Page header | `barber/app.css:150-165` | `.pageHead`, `.pageHead h1` (20px/800/`--green-800`), `.subLine`, `.pageHeadTools` |
| Cards | `barber/app.css:176-182` | `.card`, `.sectionLabel` |
| Buttons | `barber/app.css:244-260` | `.btnPrimary`, `.btnGhost`, `.btnDanger`, `.btnPlain`, `.btnSmall` |
| Chips | `barber/app.css:215-242` | `.chip`, `.chipRed`, `.chipAmber`, `.chipGreen`, `.countPill` |
| Table | `barber/app.css:533-559` | `.tableScroll`, `.dataTable` (green-800 header row, hairline rows), `.tableEmpty` |
| KPI tiles | `barber/app.css:470-488` | `.kpiGrid` (5 columns), `.kpi`, `.kpiLabel`, `.kpiValue`, `.kpiSub`, `.kpiLive` (the pulsing dot) |
| The camera tile | `barber/app.css:357-368` (`.feed`, `.feed img`, `.feedCaption`, `.feedName`, `.feedClip`) | the picture's own box: 16/9, `object-fit: contain`, `#101512`, with the caption line under it |
| The live grid | `barber/app.css:759-782` (`.liveGrid`, `.sidePanel`, `.sideBlock`) | the two-column board: the cameras wide, a narrow sticky side column |
| Error bar | `barber/app.css:206-213` | `.errorBar` |
| Notices | this product's own `app.css` (`.notice`, `.noticeCritical`, `.noticeWarning`) | the quiet channel **in the page flow**: the house's `.toast` is a fixed overlay, and nothing here may cover the camera |
| Modal | `barber/app.css:650-651` + `barber/ui/ui.tsx:30-51` | `.confirmBackdrop`, `.confirmBox`, the `Confirm` component |
| **The house's only alarm** | `attendance/FireTakeover.tsx:59-113` + `attendance/app.css:1412-1471` | the *parts* worth taking: the coloured severity bar, `role="alertdialog"`, one at a time with a queued count, ENTER to acknowledge, `ARM_DELAY_MS = 1000`. **Not** the full-screen takeover — see §3 |
| Icons | `barber/ui/ui.tsx:5-28` | the `Icon` wrapper + a local `ICONS` map of 24×24 path strings |
| Routing | `frontend/src/shared/router.ts` | `productPath()`, `productHref()`, `onRouteChange()` — already namespaced per product |
| Unit tests | `barber/vitest.config.ts` | `include: ['**/*.test.ts']`, `environment: 'node'` — pure view-model logic, no DOM |

**Dark mode:** attendance has it (`attendance/app.css:31-42`), barber does not. The fall product is
English and LTR, like barber, so it follows **barber: light only**. The CSS uses logical properties
(`inline-start` / `inline-end`) throughout, exactly as barber's does, so RTL is a later step and not
a rewrite.

---

## 2. The shape

```
              .kpiGrid — five tiles, the full width
┌──┬──────────────────────────────┬──────────────────────────┐
│IR│                              │                          │
│  │   the camera, LIVE           │   ⚠ FALL     cam-2       │  ← the alarm card
│  │   (never covered, ever)      │   falling · 0.81 · crit   │
│  │   ┌────────────────────┐     │   ┌──────────────┐       │
│  │   │  the frame         │     │   │  thumbnail   │       │
│  │   │  ← the red ring    │     │   └──────────────┘       │
│  │   └────────────────────┘     │   [ ACK ]  [ SHOW CAM ]  │
│  │                              │   ─────────────────────   │
│  │                              │   the events              │
│  │                              │    ● falling  0.81 crit   │
│  │                              │    ● lying    0.44 warn   │
│  │                              │    ● upright  —    closed  │
└──┴──────────────────────────────┴──────────────────────────┘
```

* **Live** — the page head (the boot state, the camera picker), the KPI row, then the board: the
  camera tile in the wide column and the side column beside it — the **alarm card** (when one is
  up) above the **event list**. This is barber's `.liveGrid` idiom (`1fr` + a narrow side column),
  so the house's layout rules apply unchanged.
* **Review** — the queue table in `.dataTable`, with the three verdicts as house buttons. The alarm
  card docks at the top of that page when one is up, pushing the table down — never over it.

**No third page.** Barber has seven pages because a barber shop has seven things to look at; the
fall product has two. A "Reports" page with invented charts would be worse than no page.

The camera tile keeps the product's own `live/FrameLayer.tsx` — it draws the **wall's own**
`/stream/<cam>/frame`, the engine-baked JPEG, so a person's box appears identically in the
dashboard wall and in this tab. `frameUrl.test.ts` stays green untouched.

---

## 3. The alarm — visible, and never over the camera

Barber has no popping alert. Attendance's `FireTakeover` is the house's only alarm, and the fall
product's is the same idea **re-skinned** — but not the same behaviour, because a takeover covers
the page and the operator's job is to watch the frame in which the fall happened.

**The rule: the alarm annotates the live view; it never covers it.**

So the fall product does **not** take over the screen. The alarm is a card that sits *in the page's
flow* — in the Live page's side column above the event list, and at the top of the Review page above
the queue table. The camera tile cannot be covered, not even partially, because the grid reserves
the space the card occupies. Nothing dims, nothing is blocked, and the operator keeps working.

| Event | severity | `review_id` | Toast | Alarm card | Rail badge |
|---|---|---|---|---|---|
| a confirmed fall | `critical` | none | yes | **yes — red bar** | yes |
| a candidate below the confirm threshold | `warning` | set | yes | **yes — amber bar** | yes |
| a confirmed lying | `warning` | none | yes | no | yes |
| a confirmed immobile | `critical` | none | yes | **yes — red bar** | yes |
| a closed upright (the false alarm the product rejected) | none | none | no | no | no |

* **The camera that fell is marked.** When the camera on screen is the one that fell, its tile gets
  a red (critical) or amber (warning) ring and a corner tag. The alarm annotates the picture; it
  does not sit on top of it.
* **The card carries that camera's own live frame**, small, in its corner. The product watches
  several cameras and the Live page shows one at a time, so the alarm is often about a camera that
  is *not* the one on screen. `SHOW CAMERA` switches the tile to it; the ring then applies. The
  thumbnail is a `FrameLayer`, so it inherits the guarantee that only `live/` knows the `/frame`
  endpoint (§6).
* **One card at a time.** Further alarms queue behind it and the card shows the count, exactly as
  `FireTakeover` does.
* **The card is not modal.** It does not disable anything and it does not take the keyboard away
  from what the operator is doing. ENTER acknowledges it *while it holds focus*; the Acknowledge
  button arms after 1000 ms — attendance's `ARM_DELAY_MS` — so an alarm cannot be dismissed before
  it has been on screen for a moment.
* **The acknowledge writes only where there is something to write.** A candidate below the confirm
  threshold has a review row, so acknowledging posts `acknowledged` to `POST /review/<id>` — the
  same call the Review page's buttons make. A confirmed fall has **no** review row (the routing
  queues nothing for it, design §5.3), so acknowledging it only dismisses the card; the event row
  is the record.

**The rail badge** counts the alert-worthy events that arrived while the operator was off the Live
page — "3 new since you last looked". It is a client-side counter over the event feed's ids (the
events with `id > seenCursor`), not a server-side unread flag: the product has no notion of an
operator session, and inventing one would be a second source of truth. The Live page advances the
cursor as it renders the list, so returning to it clears the badge.

---

## 4. The backend change — one table, one endpoint

A confirmed fall is **not** in the review queue. `scoring.route` (`scoring.py:118-130`) sends it to
the platform alert and queues nothing; only the below-threshold candidates are queued
(`consumer.py:185-186`). And a confirmed fall is written to Redis and then forgotten — the service
has no record of it.

So "the confirmed fall reaches the tab" needs one addition. It is deliberately the smallest one
that works, and it does **not** touch the routing.

### 4.1 The table

`schema.sql` gains:

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

`schema.sql` is executed on every connect (`storage.py:32`), so the table appears on the next
start with no migration step — the same way `fall_review` arrives today.

### 4.2 The write

`CameraWorker.handle_candidates` (`consumer.py:163-188`) gains one call, after the routing and
after the review row is created:

```python
review_id = None
if not suppressed and scored["review"]:
    review_id = self.storage.add_review(self._review_row(rec, scored, now))
result["review_id"] = review_id
if not suppressed:
    self.storage.add_event(self._event_row(rec, scored, now, review_id))
```

**A suppressed candidate is not recorded.** The aftermath of an alerted fall is deliberately not an
event (`consumer.py:203-212`) and must not become one in the history.

**The closed false alarm *is* recorded**, with a null severity: it is what the product saw and
rejected, and the Live page's event list shows it as "upright — closed". The table is the
*history*; the severity is the *alarm*.

**The routing does not change.** A confirmed fall still writes `iris:alert:<cam>` and still queues
no review row. The event row is a *record*, not a second decision.

### 4.3 The endpoint

```
GET /events?camera=&since=<id>&limit=<n>
```

returns the events with `id > since`, newest first, `limit` default 50 and capped at 200 — a bare
JSON array, exactly like `/cameras` and `/candidates` (`server.py:36-44`). Each row carries the
event's facts plus the review row's **status**, when it has one:

```sql
SELECT e.*, r.status AS review_status
  FROM fall_event e LEFT JOIN fall_review r ON r.id = e.review_id
 WHERE e.id > ? [AND e.camera = ?]
 ORDER BY e.id DESC LIMIT ?
```

so the Live page's event list and the Review page's queue never disagree about what the operator
decided.

### 4.4 Why a poll and not a socket

AC-F33.1 ("no WebSocket, no Redis contract") and AC-F34.1 ("the first release has no WebSocket")
are frozen requirements, and `tab.test.ts:37-39` enforces them against every source file in the
product. A poll keeps **four of the five product guards untouched** and leaves both requirements
with their meaning; a socket would amend a frozen requirement and add a socket server to a service
that today has five REST routes (`GET /events` is the fifth, §4.3).

The confirmed fall still reaches the **platform** instantly over `iris:alert:<cam>` — the urgent
path is unchanged and untouched. The tab is the operator's product surface, not the platform's
alert wall. Two seconds is invisible next to the sidecar's own frame cadence.

If a sub-second push is wanted later, the service can expose its own `/fall_detection/ws` — the
design's B1 — and the tab swaps one hook. The event feed is the same payload either way.

---

## 5. The files

```
frontend/src/products/fall_detection/
  index.tsx            entry — cairo, app.css, <App/>            (like barber/index.tsx)
  App.tsx              the shell, the hash route, the alarm state (the hook's result)
  app.css              the house tokens + the house classes
  ui/
    Shell.tsx          the rail (the mark, Live, Review, the badge)
    ui.tsx             Icon, ICONS, Confirm
  pages/
    LivePage.tsx       the page head, the KPI row, the camera tile, the side column
    ReviewPage.tsx     the queue table
  alerts/
    AlarmCard.tsx      the alarm card
    severity.ts        the event -> (toast?, card?, badge?) mapping
    queue.ts           the one-at-a-time queue, the queued count, the arm delay
    useEvents.ts       the poll, the seen cursor, the toast trigger
  api.ts               the calls, typed
  format.ts            the clock, the score, the chip labels
  live/
    FrameLayer.tsx     + an optional className, so the house's .feed styling applies
    frameUrl.ts        unchanged
  package.json         + @fontsource/cairo
  vitest.config.ts     mirror barber's
```

`app.css` is **not** a copy of barber's 54 KB. It is barber's `:root` block plus the classes the
fall product actually uses (`.shell`, `.rail`, `.railBtn`, `.railActive`, `.railCount`, `.page`,
`.pageHead`, `.card`, `.sectionLabel`, `.btn*`, `.chip*`, `.dataTable`, `.kpi*`, `.feed*`,
`.errorBar`, `.notice` (the product's own quiet channel -- the house's `.toast` is a fixed overlay, and this
product has no fixed overlay), the alarm card's classes, and the fall product's own). Every value is the
house's value.

`live/FrameLayer.tsx` keeps its endpoint knowledge and gains one optional `className` prop: the
house's `.feed img` is absolutely positioned inside a 16/9 box, and the layer's inline styles
(`height: auto`) fight it. The layer is the only place that knows `/frame`, and it stays that way.

**One card component, two hosts.** `App.tsx` owns the alarm state (the hook's result: the current
alarm, the queue behind it, the unseen count). Each page renders the card in **its own slot** — the
Live page's side column, the Review page's top — because the card must be *in the flow*, never an
overlay. There is no fixed-position alarm anywhere in this product.

---

## 6. The tests

| Test | Change | Why |
|---|---|---|
| `tab.test.ts:37-39` (no WebSocket) | **unchanged** | the poll uses no socket |
| `tab.test.ts:41-43` (no Redis contract) | **unchanged** | the tab never names a Redis key |
| `tab.test.ts:45-47` (no canvas) | **unchanged** | the alarm draws no overlay — the camera is marked with a ring, not painted on |
| `tab.test.ts:52-54` (the frame module owns the endpoint) | **unchanged** | `live/frameUrl.ts` and `live/FrameLayer.tsx` stay as they are |
| `tab.test.ts:56-60` (the page composes the layer, never the endpoint) | **re-pointed and strengthened** | see below |

The re-pointed assertion today reads `index.tsx` and requires it to contain `FrameLayer` and not
`/frame`. When the page moves to `pages/LivePage.tsx`, that assertion has to follow it — the intent
is unchanged, the file it reads changes. It is **strengthened** rather than merely moved: instead of
naming one file, it asserts that **`live/frameUrl.ts` and `live/FrameLayer.tsx` are the only two
files in the product that know the `/frame` endpoint**. That survives any future refactor of the
page, and it is what lets the alarm card's thumbnail be a `FrameLayer` without a second copy of the
endpoint.

**New tests** (the product's vitest files are pure view-model logic, no DOM):

* `alerts/severity.test.ts` — the event → (toast?, card?, badge?) mapping above, as a pure
  function. This is the rule that decides whether an operator is interrupted.
* `alerts/queue.test.ts` — the card queue: one at a time, the queued count, and the arm delay
  (`canAcknowledge(now, shownAt)`).
* `alerts/badge.test.ts` — the unseen counter: the events with `id > seenCursor`.
* `api.test.ts` — the URL each call makes.
* `alerts/cursor.test.ts` — the `since` cursor advancing: forward only, never back, unchanged on an
  empty batch.
* `format.test.ts` — the clock, the score, the chip labels.
* The product's **Python** side: the event row is written for an alert-worthy candidate, **not**
  written for a suppressed one, the closed false alarm is written with a null severity, and
  `GET /events` returns them newest-first with `review_status`.

---

## 7. What is deliberately not built

* **No full-screen takeover.** The house has one (`FireTakeover`), and the fall product
  deliberately does not copy it: a takeover covers the frame in which the fall happened, which is
  the one thing the operator must keep watching. The alarm annotates the live view instead (§3).
* **No dark mode.** Barber has none; the fall product is LTR like barber. Attendance's block is
  there if it is ever wanted.
* **No bilingual provider.** The product's registry entry is `lang: 'en'`, `dir: 'ltr'`
  (`shared/products.ts:45-51`), and the strings are English today. The CSS uses logical properties
  so Arabic is a later step, not a rewrite.
* **No "Show clip" button in the alarm card.** The clip is the platform's artefact
  (`iris:svr_request:<cam>` → the platform's clip store); the fall product has no contract with it,
  and inventing one would be exactly the platform coupling the product's design forbids. The card
  shows the clip **window** as a fact, and the camera's live frame.
* **No "Reports" page, no charts.** There is no data behind them yet.
* **No event pruning.** `fall_event` grows like `fall_review` grows today — same house property,
  not a new debt.
* **No websocket, no scene, no chips.** The design's B1 and B2 stay where they are.

---

## 8. How it is verified

1. **The product's vitest suite** — the new pure-logic tests plus the re-pointed guard.
2. **The product's Python suite** — the event write, the suppression rule, the endpoint.
3. **`npm run build`** — `tsc -b && vite build` for the whole dashboard, which compiles the new
   TypeScript.
4. **The root suite** — unchanged by a frontend change, but the product's own Python changes are
   re-run in full.

**The gap, stated plainly:** the frontend is built and unit-tested on this box — a portable Node on
`F:`, `npm run build` green, and the product's vitest files green; the delivery report's §3 quotes
both. What has **never** happened is a render in a browser: `tsc` proves the types, the tests prove
the pure logic and the stylesheet's bytes, and none of them proves what a person sees. The first
real look at the tab is on the 5090.
