# Phase 5: Delivery Report — the fall-detection tab in the house style

**Date:** 2026-10-28 · **Branch:** `feat/fall-detection-module-integration` (base `main`) ·
**Spec:** `products/fall_detection/docs/phases/phase-5-ui-redesign.md`, commit `331b75a` ·
**Plan:** `products/fall_detection/docs/phases/phase-5-implementation-plan.md`, commit `34d69a0`
(eleven tasks, each committed) ·
**Status:** delivered. Two things are owed: the Python suites' re-run on the 5090 (§3), and the
first render of the tab in a browser (§5).

**Scope of this delivery:** the fall product's dashboard tab, rebuilt in the IRIS house style, and
the two backend pieces the tab needs — the `fall_event` history table and `GET /events`. No engine
change, no routing change, no vendor change. The frontend is at
`frontend/src/products/fall_detection/`; the Python is at `products/fall_detection/`.

---

## 1. System summary — what was built

| Layer | What it is | Where |
|---|---|---|
| the house stylesheet | barber's tokens and the house blocks this product uses, copied byte for byte; the product's own `.notice` family and the alarm card's classes at the end | `frontend/src/products/fall_detection/app.css` (`:root` 6-41; `.feed` 242-253; `.feedRing`/`.feedRingWarn` 344-345; `.notice` 350-353; `.alarmCard` 356-370); the copied ranges are pinned at `app.test.ts:28-32` |
| the shell | `.shell` → `.rail` (56 px, `--green-900`) + `.page` (`--ground`, 20 px radius); the shared `BrandLogo`; two routes — Live at `/`, Review at `/review` — and the unseen badge on Live | `ui/Shell.tsx:9-42`; `ui/ui.tsx:4-49` (`Icon`, `ICONS`, `Confirm`); `app.css:62-140` |
| the two pages | Live: the page head (device, weights, classifier, paths), the five KPI tiles, the camera tile, the side column; Review: the queue table with the three verdicts | `pages/LivePage.tsx:13-112`; `pages/ReviewPage.tsx:8-68` |
| the alarm card | one card at a time, the waiting count, the 1000 ms arm delay, ENTER while focused, the camera's live thumbnail, `show camera` | `alerts/AlarmCard.tsx:13-70` |
| the alarm rules | the event → (toast?, card?, badge?) mapping; the queue, the pop and the verdict that retires a card; the unseen counter | `alerts/severity.ts:36-40`; `alerts/queue.ts:9-35`; `alerts/badge.ts:8-10` |
| the events feed | a 2 s poll with the seen cursor; the first response is history, not news; the five calls under `/fall_detection/api`; the clock, the score, the chip | `alerts/useEvents.ts:13-92`; `api.ts:9-80`; `format.ts:4-15` |
| the `fall_event` table | the product's own history: every scored candidate with its severity, its scores and its review id, plus the camera index; created on every connect, so no migration step | `schema.sql:30-46`; `storage.py:32` (`executescript`), `storage.py:53-80` (`add_event`, `list_events`) |
| the write | one `add_event` after the routing and the review row, guarded by `not suppressed`; a confirmed fall is recorded though it is never queued, the closed false alarm with a null severity, the aftermath not at all | `consumer.py:187-188`; `consumer.py:224` (`_event_row`); `consumer.py:203-212` (`_is_aftermath`) |
| the `/events` endpoint | `id > since`, newest first, `limit` default 50 capped at 200, the review row's status joined; the fifth route | `server.py:47-55`; `storage.py:68-80` |

### The shape

The tab is a board with two pages in barber's `.liveGrid` idiom (`app.css:310-314`): the camera wide
in a 16:9 `.feed` box, a 210 px side column beside it. The alarm is a card in that column, above the
events list (`LivePage.tsx:81-108`), and at the top of the Review page above the table
(`ReviewPage.tsx:20-24`). The camera that fell is marked with a ring and a corner tag
(`LivePage.tsx:65-73`) and is never covered. The data arrives on a 2 s poll (`useEvents.ts:13`), not
a socket: the tab names no socket and no channel, and `tab.test.ts:39-49` fails the suite if it ever
does. The frame itself is the wall's own JPEG, drawn by `live/FrameLayer.tsx`, which is one of the
only two files that know its endpoint (`tab.test.ts:58-61`).

---

## 2. The coverage

| File | Count | What it pins |
|---|---|---|
| `app.test.ts` | 21 | the house tokens equal barber's; the copied house blocks are byte-identical |
| `tab.test.ts` | 8 | the frozen guards (no socket, no capture-prefix, no canvas), the `/frame` allowlist, the layer's composition |
| `alerts/severity.test.ts` | 4 | the event → (toast?, card?, badge?) mapping |
| `alerts/queue.test.ts` | 7 | one at a time, the queued count, the arm delay, a verdict retiring its card |
| `alerts/badge.test.ts` | 3 | the unseen counter |
| `api.test.ts` | 4 | the URL each call makes, the `since` cursor |
| `alerts/cursor.test.ts` | 4 | the `since` cursor advancing: forward only, never back, unchanged on an empty batch |
| `format.test.ts` | 4 | the clock, the score, the chip labels |
| `frameUrl.test.ts` | 2 | the frame endpoint's URL |
| `products/fall_detection/tests/test_storage_event.py` | 5 | the row survives a restart; newest first and the cursor moves; the camera filter; the review status join; the same serialization as the queue |
| `products/fall_detection/tests/test_worker_event.py` | 4 | a confirmed fall is recorded though never queued; a candidate carries its review row; the closed false alarm has no severity; the aftermath is not recorded |
| `products/fall_detection/tests/test_events_api.py` | 4 | a bare array newest first; the cursor; the camera filter; the capped limit |

The nine frontend counts are the ones the run in §3 printed, file for file. `app.test.ts`'s 21 are
three token assertions plus eighteen byte-identity assertions, one per house block at
`app.test.ts:28-32`. The three Python counts are `def test_` counts:
`products/fall_detection/tests/test_storage_event.py:25-69`,
`products/fall_detection/tests/test_worker_event.py:17-50` and
`products/fall_detection/tests/test_events_api.py:31-54`. Those three files have not been run in
this task (§3).

---

## 3. The evidence

Both commands were run in this task, after the last code change of the phase.

### The build

```
cd F:/IRIS/surveillance-camera-sys/frontend && export PATH="/f/_tools/node-v24.18.0-win-x64:$PATH" && npm run build
```

The output's tail, verbatim:

```
dist/assets/index-3pdpfxI7.js                         371.16 kB │ gzip: 120.65 kB
dist/assets/index-DWcl7JPE.js                         596.24 kB │ gzip: 168.90 kB

(!) Some chunks are larger than 500 kB after minification. Consider:
- Using dynamic import() to code-split the application
- Use build.rollupOptions.output.manualChunks to improve chunking: https://rollupjs.org/configuration-options/#output-manualchunks
- Adjust chunk size limit for this warning via build.chunkSizeWarningLimit.
✓ built in 4.19s
```

`tsc -b` and `vite build` both completed; the shell's exit code was **0**. The build transformed
1087 modules and wrote the Cairo faces (`dist/assets/cairo-*.woff2` and `.woff`) as separate assets.
The >500 kB warning is the dashboard bundle's (`dist/assets/index-DWcl7JPE.js`, 596.24 kB); the fall
tab's own chunk is `dist/assets/index-CC2JDW27.js` at 13.60 kB, and its Cairo faces are separate
small assets — none of them is over 500 kB.

### The product's suite

```
cd F:/IRIS/surveillance-camera-sys/frontend/src/products/fall_detection && export PATH="/f/_tools/node-v24.18.0-win-x64:$PATH" && npx vitest run
```

The output, verbatim:

```
The CJS build of Vite's Node API is deprecated. See https://vite.dev/guide/troubleshooting.html#vite-cjs-node-api-deprecated for more details.

 RUN  v2.1.9 F:/IRIS/surveillance-camera-sys/frontend/src/products/fall_detection

 ✓ app.test.ts (21 tests) 10ms
 ✓ alerts/queue.test.ts (7 tests) 6ms
 ✓ api.test.ts (4 tests) 10ms
 ✓ alerts/severity.test.ts (4 tests) 5ms
 ✓ format.test.ts (4 tests) 6ms
 ✓ frameUrl.test.ts (2 tests) 3ms
 ✓ alerts/badge.test.ts (3 tests) 5ms
 ✓ alerts/cursor.test.ts (4 tests) 4ms
 ✓ tab.test.ts (8 tests) 6ms

 Test Files  9 passed (9)
      Tests  57 passed (57)
   Start at  23:29:36
   Duration  669ms (transform 327ms, setup 0ms, collect 675ms, tests 55ms, environment 3ms, prepare 1.64s)
```

The shell's exit code was **0**.

### The Python suites — deferred, not run

The two commands the server will run, verbatim:

```
cd products/fall_detection && F:/URCA_PROJECTS/.venv/Scripts/python.exe -m pytest -q \
  -k "not corpus_replay and not onset_to_alert"
```

```
python -m pytest tests -q -k fall
```

The second runs at the repository root. Neither was run here. The user's standing rule defers
GPU/weights runs to the 5090, and the sandbox classifier refuses a local `pytest`; R34 records the
refusal as the expected outcome, not a failure. The deselection is mandatory: the corpus and the
weights are on this box, so the two replay tests in `tests/test_measurements_gated.py` do not take
their "the corpus is absent" skip — they attempt the full GPU replay and the interpreter dies there
(exit 127, no pytest summary). On the 5090 the same two stay deselected: the replays are the measured
experiments, not unit tests. The `-k fall` filter is what Phase 4 ran; `python -m pytest tests/` hangs
on this box at `tests/test_frame_publisher_shed.py`.

The last observed numbers, with their provenance:

| Run | Last observed | Where | When |
|---|---|---|---|
| the product's suite, the vendor copy included, the two replays deselected | `189 passed, 6 skipped, 2 deselected, 1 warning in 53.86s` | Phase 5's Task 3 implementer report — SDD scratch, not tracked in git | after all three Python tasks of this phase |
| the root suite's fall group | `61 passed in 2.84s` | `phase-4-delivery-report.md:139-141` | Phase 4, **before** this phase's Python changes |

**The root fall group has not been re-run since this phase's Python changes** — `schema.sql`,
`storage.py`, `consumer.py`, `server.py` and thirteen new tests under
`products/fall_detection/tests/`. That is the one verification debt this phase carries. This
report does not claim the Python side is green for this phase; the claims above it are the build
and the 57 frontend tests.

---

## 4. What was deliberately not built

The spec's §7, restated unchanged:

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

## 5. The gap that remains

The tab has never been rendered in a browser. Nothing in this task opened one. The build compiles
the tab and the 57 tests pin the pure view-model logic and the stylesheet's bytes — `tsc -b` proves
the types, `app.test.ts` proves the tokens and the copied house blocks, `tab.test.ts` proves the
source-level guards, and the other seven files prove the mapping, the queue, the badge, the calls,
the cursor, the formats and the frame URL. None of them proves what a person sees.

The first real look at the tab is on the 5090 — or any box running `just fall-up` with the dashboard
— against a live camera: the shell (the 56 px rail, the 20 px `.page`), the ring on the camera that
fell, the card's placement in the side column and at the top of Review, and the notice legibility in
the column beside the picture. Until that run, the rendering is unverified code.

What the tab assumes, and a browser must confirm:

* **The Cairo font ships in the bundle.** `@fontsource/cairo` is this product's dependency
  (`frontend/src/products/fall_detection/package.json`), `index.tsx:2-5` imports the 400, 600, 700
  and 800 faces, and the build wrote the `cairo-*.woff2` and `.woff` files into `dist/assets/`
  (§3). `app.css:48` names Cairo first. Whether the glyphs actually load and read at 11.5–14 px is
  the browser's answer.
* **The 16:9 `.feed` box with `object-fit: contain`.** `app.css:242-253` fixes the box's aspect
  ratio at 16/9, `max-height: 60vh`, the picture absolutely positioned and contained. Whether the
  camera's real aspect ratio letterboxes inside the box as intended is visible only on screen.
* **The card's thumbnail at `intervalMs={500}`.** `alerts/AlarmCard.tsx:57` renders the camera's
  `FrameLayer` at half a second per frame; `live/FrameLayer.tsx:31-33` schedules the next request
  only after the previous one finished. Whether that reads as live at 96 px
  (`app.css:368-369`) is a visual judgement.
* **The ring's colours.** `.feedRing` is `--red-500` and `.feedRingWarn` is `--amber-600`
  (`app.css:344-346`), against the picture's `#101512` (`app.css:249`). Contrast and weight on a
  real frame are the browser's question, not the stylesheet's.

**And the suite does not gate a merge.** The dashboard's CI job runs `npm run build` and nothing
else (`.gitlab-ci.yml:112-125`), so `app.test.ts`'s byte-identity guard, `tab.test.ts`'s frozen
guards and the other files do not gate a merge today. Until a vitest step is added to that job,
they are a local and reviewer check.

These are the questions the first render answers. Nothing in this phase answers them.
