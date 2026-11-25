# Running several products on one DeepStream pipeline

**Status:** agreed 2026-09-21 · **Audience:** supervisor + product developers
**Scope:** attendance, fire/smoke, barber-shop monitoring — and every product after them

---

## 1. The problem in one paragraph

We are building several products (attendance, fire/smoke, barber shop, more coming) on
**one** DeepStream perception pipeline. Today each product lives on its own long-lived
git branch, and each branch edits the same shared files — `perception/pipeline/sidecar.yaml`,
`perception/pipeline/sources.py`, `justfile`, `docker-compose.yml`. That works for two products and
stops working at four: branches drift, the same file gets edited three ways, and
whoever merges last pays for everyone.

This document is the agreement on how we stop that: **products are separated by
configuration and folders, never by branch.**

---

## 2. Decisions

| # | Decision | What it means in practice |
|---|---|---|
| 1 | **The config file is the truth** ✅ done | What a product runs is written in its own yaml: the `modules:` list says which capabilities run, in which order, with which settings (`perception/modules/README.md`). A role no longer carries product knowledge in code: what it changes is the `role_overrides:` table in the settings, and which role owns an untagged source's recording is `untagged_recording_owner`. An unknown module, an unknown or missing setting, or a module whose `requires` is not met refuses startup. |
| 2 | **Each product lists its own cameras** ✅ done | A product's yaml names its cameras; the shared file lists none. A process whose settings resolve to no cameras is **refused at startup** rather than inheriting another product's, and a test walks every folder in `products/` to check it. |
| 3 | **One folder per product** | A product is a directory containing its service code, its pipeline profile, its service config and its tests. `IRIS_PRODUCT=<name>` in `.env` says which product this checkout runs. |
| 4 | **Inherit mechanism, declare policy** ✅ done | Shared yaml keeps only what nobody disagrees about (engine paths, driver settings) and lists no module. Anything a product could reasonably want different is stated by each product explicitly, as a module with its settings, and the pipeline refuses to start if a setting is unknown or missing. The engine itself (`perception/engine/`) names no product or use case: `tests/test_engine_boundary.py` holds an empty ratchet over it. |
| 5 | **Duplicated facts are policed by a test** | Camera id, frame size and face-gallery key appear in both the pipeline profile and the service config. A test asserts they agree, so drift fails CI instead of failing silently at runtime. |
| 6 | **One Postgres, one schema per service** | Cheap to run, boundary already drawn, and a later split into separate databases stays a dump-and-restore. **Hard rule: no cross-schema JOIN, no cross-schema foreign key, no transaction spanning two schemas.** |
| 7 | **One config file per service** | Every service reads a yaml of its own. `.env` holds only machine-specific values — ports, passwords, checkout name. |
| 8 | **Product folders are autonomous** | Each product owns everything inside its folder and may structure it however it likes, **on the condition that it changes nothing outside the folder** without review. Accepted cost: some code is duplicated between products and a shared bug may need fixing more than once. |
| 9 | **The Redis payload carries a version** ◐ half done | `iris:meta:<cam>` now has three readers. The engine stamps `"v": "1"` on every entry (`perception/engine/publisher.py`). Not done: consumers do not check it yet, and the shape is still the legacy one (`dets` is 3 fields, or 7 with a vehicle tracker, and each module adds its own field). A fixed `objects` shape under `"v": "2"` means changing the readers in `products/attendance/acl.py`, `products/barber/acl.py`, `app/` and `mtmc/`, which their owners should review (decision 8). |
| 10 | **One frontend, per-product folders** | One shared shell (sidebar, router, components) and a folder per product. Which products appear is decided **at run time** (2026-09-22): one build for every customer, and their enabled products plus permissions decide the menu — the shape the IRIS Studio design uses. See section 5. |
| 11 | **Merge first, restructure immediately after** | The merge request goes up as-is. The restructure is the first work on top of it. |
| 12 | **The IRIS surveillance UI stays a product** | It is kept, not retired. Nothing we merge may remove it from `main`. |
| 13 | **Our look, the IRIS UI's plumbing** | The shared shell keeps the attendance app's visual design, built on the IRIS UI's libraries (react-router + TanStack Query), which our hand-written router and fetch code lack (refined 2026-09-22). Each product sets its own language and text direction (attendance: Arabic, right-to-left; barber: English, left-to-right). |
| 14 | **`IRIS_PRODUCT` accepts a list** | `IRIS_PRODUCT=attendance,fire` gives one sidebar with both products. A single value behaves as in decision 10. |
| 16 | **Fire belongs to attendance; shared defaults are neutral** | Fire is not a product: in `app/` it is one kind of alert beside zone intrusion and overcrowding, so extracting it would hand one product the machinery serving all of them. Its settings are declared by attendance, and the shared file turns **no** detector on — an opt-out default made every future product run a second detector and publish alerts nobody asked for (2026-09-23). |
| 15 | **No MR changes `main`'s frontend pointer until step 4b** | `main` keeps pointing `frontend` at the IRIS UI (decision 12). A product branch that points it at its own app opens its MR from a separate MR branch that restores `main`'s pointer — see step 1 for how. Product apps stay reachable with `just frontend-use` until step 4b ends branch switching. |

---

## 3. What "separated by configuration" actually means

The pipeline is **not** cut into per-product pieces. It is the same program started more
than once with different settings — like opening two documents in one editor.

```
              ┌────────────────────────────────────────┐
              │  perception/pipeline/sidecar.yaml      │  shared MECHANISM:
              │  model paths, driver knobs, timeouts   │  what nobody disagrees
              │  no cameras · every product switch off │  about, and nothing that
              └───────────────────┬────────────────────┘  turns a detector on
                                  │  extends:
              ┌───────────────────┴───────────────────┐
              │                                       │
  products/attendance/sidecar.yaml        products/barber/sidecar.yaml
  cam-1, cam-2, cam-3                     barber-1
  face rule: legacy                       face rule: tracked (highest-top)
  fire + smoke detector ON                no fire, no zone analytics
              │                                       │
  products/attendance/ (service, db)      products/barber/ (service, db)
  fire incidents + clips served by app/   zones evaluated in the service
```

Three levels of separation, and it is worth being precise about which is which:

1. **Settings file** — which cameras, which models, which modules. Per product.
2. **What you start** — one command starts one product's pipeline and its service.
   Running attendance does not start barber.
3. **Where it runs** — your checkout, your container names, your ports, your database.
   Stopping yours cannot stop anyone else's.

**Shared per machine, and therefore assigned deliberately:** the camera link (each
camera is pulled once, through go2rtc), the GPU, and the host ports. These are in
`.env`, one set per checkout.

---

## 4. Target layout

```
surveillance-camera-sys/
├── perception/              # shared. Changing anything here affects every product → review required
│   ├── engine/              #   the camera→detections engine: decode, infer, track, publish, record, recover.
│   │                        #   Knows no use case (python3 -m perception.engine.main <sidecar.yaml>)
│   ├── modules/             #   use-case logic, one folder per capability, picked by `modules:` in a
│   │   ├── zones/           #   product's yaml: zones, fire_alerts, face_detect, face_recognition,
│   │   └── …                #   customer_faces, appearance, plates, mtmc, … (README.md lists them)
│   ├── lib/                 #   helpers more than one module needs (pure functions)
│   ├── pipeline/            #   deployment assets: the Dockerfile, the shared sidecar.yaml
│   │   └── sidecar.yaml     #   MECHANISM ONLY: model paths, driver knobs, timeouts.
│   │                        #   No cameras, no modules.
│   └── models/              #   vendored detectors, upstream names kept so they can be
│       ├── DeepStream-Yolo/ #   compared against and updated from upstream
│       ├── DeepStream-Yolo-Face/
│       └── DeepStream-Yolo-Fire/
├── products/
│   ├── attendance/          # includes fire/smoke: it is attendance's second detector
│   │   ├── sidecar.yaml     #   extends ../../perception/pipeline/sidecar.yaml; declares this
│   │   │                    #   product's cameras and its `modules:` (faces, fire, ...)
│   │   ├── server.py  …     #   service code — the owner decides the structure
│   │   └── data/
│   └── barber/
│       ├── sidecar.yaml
│       ├── tracker.yml
│       ├── barber.yaml      #   service config
│       ├── server.py  …
│       └── tests/
├── app/                     # IRIS backend: camera list, frames, alert incidents and clips.
│                            # Part platform, part product — see "not doing yet" below.
├── tests/configs/           # settings files for recorded-clip test runs
├── frontend/                # one app, one folder per product inside it (steps 4b, 10–12)
├── docker-compose.yml       # shared → review required
├── justfile                 # shared → review required
└── .env                     # never committed: ports, passwords, IRIS_PRODUCT, SIDECAR_CONFIG

`perception`, not `platform`: a top-level `platform/` package shadows Python's standard-library
`platform` module, so `platform.system()` breaks for every dependency in the process (verified
2026-09-22 before the move).
```

### The three kinds of configuration

People keep mixing these up, so plainly:

| File | Question it answers | Read by | Changes when | In git? |
|---|---|---|---|---|
| `perception/pipeline/sidecar.yaml` | *How does a pipeline work?* | the DeepStream process | rarely — it is the engine | yes |
| `<product>/sidecar.yaml` | *What does this product look at?* | the DeepStream process | a product's cameras or models change | yes |
| `<product>/<product>.yaml` | *What do the results mean?* | that product's service | business rules change | yes |
| `.env` | *Where am I running?* | docker compose | per machine, per developer | **no — holds passwords** |

`<product>/sidecar.yaml` is called an **overlay**: its first line is
`extends: ../perception/pipeline/sidecar.yaml`, and it contains only what differs. Two transparent
sheets on a projector — the bottom one is the full drawing, the top one holds the few
lines that are different, and the program sees one merged picture.

The test for where a value belongs: **could two developers running the same product need
different values?** Yes → `.env`. No → a yaml.

---

## 5. Frontend

### Today (verified 2026-09-21)

The `frontend` submodule (`pyai/iris/iris-frontend`) is **three separate apps on three
branches**, and `just frontend-use iris|attendance` switches between them by checking out a
different branch — products separated by branch, which is what this plan exists to end.

| Branch | App | Stack | Notes |
|---|---|---|---|
| `main` | IRIS surveillance UI | React 19, react-router, react-query, zustand, recharts · English | 105 source files |
| `feat/fire-smoke-tab` | attendance + fire | React, no router or state library · Arabic, right-to-left | started 2026-09-01 with **no shared history** with `main`; git refuses a normal merge |
| `feat/barber-ops` | barber | same stack as attendance · English, left-to-right | copied from the attendance app on 2026-09-13; its first commit **deletes** the attendance and fire pages |

Two consequences:

- **Attendance and barber frontends conflict in 8 places**: `App.tsx`, `api.ts`, `app.css`, and
  five attendance pages that barber deleted while attendance kept editing them. In this repo
  all eight show up as the single `frontend` pointer conflict.
- **This repo's `main` points `frontend` at the IRIS UI; `feat/fire-smoke-attendance` points it
  at the attendance app.** Merging that branch as-is would replace the IRIS UI on `main`,
  which decision 12 rules out — so the MR keeps `main`'s pointer where it is (decision 15).

### Target

```
frontend/src/
├── shared/            one shell for every product — review required
│   ├── shell/         sidebar, header, router
│   ├── ui/            buttons, cards, tables, dialogs, icons
│   ├── charts/
│   └── api/           fetch + websocket helpers
├── products/
│   ├── attendance/    Arabic, right-to-left
│   ├── fire/
│   └── barber/        English, left-to-right
└── main.tsx           loads the product(s) named by IRIS_PRODUCT
```

Each product folder describes itself once; the shell reads that description to draw the
sidebar, mount the pages and set the text direction:

```ts
// src/products/barber/index.ts
export const product = {
  id: 'barber', title: 'Barber Ops', lang: 'en', dir: 'ltr',
  nav:    [{ path: '/live', label: 'Live floor' }, { path: '/insights', label: 'Insights' }],
  routes: { '/live': LiveFloorPage, '/insights': InsightsPage },
}
```

The product is chosen **at build time**. `IRIS_PRODUCT` in `.env` becomes `VITE_PRODUCT`; Vite
replaces it with the real value while building and drops the products that weren't named, so
a customer's build contains only the products they bought. `IRIS_PRODUCT=attendance,fire` gives
one sidebar with both (decision 14).

### The IRIS surveillance UI

Kept as a product (decision 12). It uses a different stack from the product apps
(react-router, react-query, zustand), so moving it into the shared shell is a rewrite, not a
move. Until someone decides otherwise it stays a **separate app in the same repo**.

Open for later: whether it eventually moves into the shared shell, and where the combined
product app lives in the frontend repo — its histories are unrelated to `main`, so joining
them is a one-time `--allow-unrelated-histories` merge into a layout such as `apps/iris/` +
`apps/products/`.

### Proposal — for the barber developer to decide

How barber's frontend pages join the shared layout is **his call**. What we propose:

- **Proposed:** start a fresh branch from the new layout and re-apply his pages into
  `src/products/barber/`. He has 6 commits since he copied the app, and pages in their own
  folder cannot conflict with anyone else's.
- **Alternative:** merge `feat/barber-ops` as it is, keeping its commit history, and resolve the
  8 conflicts — most of them by restoring the attendance and fire pages it deleted.

Either way the end state is the same: barber's pages under `src/products/barber/`, nothing
deleted outside it.

---

## 6. How we work together

**Rules of engagement**

1. **Own your folder.** Inside `attendance/`, `fire/` or `barber/`, the owner decides the
   structure, the libraries and the style. No review needed.
2. **Shared files need review.** `perception/pipeline/`, `docker-compose.yml`, `justfile`, `frontend/`
   shared components, and the shared database schema. A change here can break a product
   you cannot test, so it gets a second pair of eyes and a note in the MR describing
   which products it touches.
3. **Never edit another product's folder.** If you need something from it, ask its owner.
4. **Merge to `main` weekly at worst.** Branch age, not branch size, is what makes merges
   painful. A branch that lives a month is a month of other people's changes to
   reconcile at once.
5. **Every product keeps one smoke test** — a test that boots that product's config and
   asserts it is coherent (camera ids, gallery key and frame size agree between its
   pipeline profile and its service config). `tests/test_barber_sidecar_config.py` is the
   model to copy. This is what protects a shared pipeline: it is the only check that
   notices one person's change breaking someone else's product.
6. **Pipeline changes state their blast radius.** Any change to `perception/pipeline/` says in the MR
   which products were run against it.

**Who owns what**

| Area | Owner |
|---|---|
| `perception/pipeline/` — the shared pipeline | shared; review by both product developers |
| `attendance/`, `frontend/` attendance screens | Kareem |
| `barber/`, `frontend/` barber screens | barber developer |
| `fire/` | shared for now |
| `docker-compose.yml`, `justfile`, `.env.example` | shared; review |

To be added as a `CODEOWNERS` file once the folders exist, so review assignment is
automatic rather than remembered.

---

## 7. Sequence

Current state, verified 2026-09-21:

- **The attendance MR is open**, from `mr/fire-smoke-attendance` into `main`: 79 commits,
  0 behind `main`, so it merges without conflicts. `pytest tests`: 754 passed.
- `mr/fire-smoke-attendance` is `feat/fire-smoke-attendance` plus **one commit** that sets the
  `frontend` pointer and `.gitmodules` back to what `main` has (decision 15). The feature branch
  itself keeps pointing at the attendance app, because the live attendance checkout runs from
  it and `just up` refuses to start when the checked-out frontend doesn't match the pointer.
- `feat/barber-shop-monitoring` was branched **from** that branch at `1f5b27b`, so it
  already contains the attendance work and adds 25 commits.
- A dry-run of merging `main` (after the attendance MR) into the barber branch reports
  **3 conflicts**: `justfile`, `.gitmodules` and the `frontend` pointer. Behind the pointer, the
  two frontend apps themselves conflict in 8 places (section 5) — that part is step 4b, not
  this merge.
- 65 test files exist; **no CI pipeline runs them**.

| Step | What | Who | Why this order |
|---|---|---|---|
| 1 | **Open** — MR `mr/fire-smoke-attendance` → `main`. Until it merges, anything committed on `feat/fire-smoke-attendance` is also merged into `mr/fire-smoke-attendance`. | Kareem | Every day it waits, the merge gets larger. |
| 2 | Merge barber on top | barber dev | Step by step below: 3 conflicts (`justfile`, `.gitmodules`, `frontend` pointer). Barber's frontend pages are a separate job (step 4b, section 5). |
| 3 | **Done** — `.gitlab-ci.yml`: the repo's 800 tests, barber's 133 and the dashboard build run on every merge request | shared | Two things the measurement changed: the tests need a 297 MB dependency list (`requirements-test.txt`), not the 5.9 GB `requirements.txt` — no test touches the training tooling; and 35 tests fail in a clean container without `ffmpeg`, which developer machines happen to have. `just check` runs the same three checks locally. **Still needed in GitLab: a runner, and Settings → Merge requests → "Pipelines must succeed".** |
| 4 | **Done** — each product moved into `products/<name>/` with its own settings; the pipeline and models into `perception/` | Kareem | Not as mechanical as written here: ~490 hard-coded paths, and four breakages tests could not catch (a virtualenv hard-codes its own path; recipes counted `../`; scripts counted parent folders; a bind mount to a moved path silently became an empty directory). |
| 4b | Frontend: shared shell + `src/products/<name>/`, product chosen by `IRIS_PRODUCT` | Kareem (shell), each owner (their folder) | Ends switching products by checking out a frontend branch. |
| 5 | One smoke test per product; make it required to merge | each owner | The check that actually protects a shared pipeline. Started: `tests/test_product_declarations.py` walks every product folder, and `tests/configs/` runs a product against a recorded clip. |
| 6 | `CODEOWNERS` | shared | Review routing stops being a memory exercise. |
| 7 | **Done** — role code is data; required keys enforced | shared | A module's settings are validated at startup (unknown or missing keys refuse to start). `apply_role_overrides` is generic and reads the `role_overrides:` table; no product name is left in the engine's code. |
| 8 | `"v": 1` on the Redis payload | shared | The stamp is in; the check in each consumer is not (see decision 9). |
| 9 | Postgres: one instance, schema per service | shared | Starts when the first product outgrows SQLite. |

Steps 1–2 are this week. Steps 3–6 are the first work on top of the merge.

### Merging barber after the attendance MR

For the barber developer, once the attendance MR has merged into `main`. Verified by a
dry-run merge on 2026-09-21.

**1. Bring `main` into the barber branch**

```bash
git fetch
git checkout feat/barber-shop-monitoring
git merge origin/main
```

**2. Resolve the 3 conflicts**

| File | Why it conflicts | Keep |
|---|---|---|
| `justfile` | both branches added commands in the same area | **both** — the barber commands and the attendance ones |
| `.gitmodules` | `main` says `branch = main`, the barber branch says `branch = feat/barber-ops` | the **barber** line |
| `frontend` (pointer) | `main` points at the IRIS UI, the barber branch at the barber app | the **barber** app: `git -C frontend checkout feat/barber-ops && git add frontend` |

The barber branch keeps its own frontend for the same reason the attendance branch does: its
checkout runs the barber app, and `just up` refuses to start when the checked-out frontend
doesn't match the pointer. Then `git commit` to finish the merge.

**3. Check it still works** — `pytest tests`, then run barber end to end. Only the barber
developer can confirm his product still behaves.

**4. Open the barber MR the same way as step 1** — from an MR branch that is the barber branch
plus one commit putting the `frontend` pointer and `.gitmodules` back to what `main` has
(decision 15). `main` gets all the barber backend work and keeps the IRIS UI.

**5. Barber's frontend pages come later** — step 4b. How they join the shared layout is the
barber developer's call (section 5).

---

## 8. Deliberately not doing yet

| Not doing | Why | What would change our mind |
|---|---|---|
| A separate repository per product | Every product uses the shared pipeline, and changes routinely cross the boundary. One repo keeps a pipeline change and its product fix in a single reviewed commit. | A customer freezes on an old version, or a product stops using the pipeline, or another team takes ownership. |
| Extracting a shared service library | The two services have diverged genuinely (`acl.py`: 775 vs 680 lines, 584 differing). Forcing them together produces a shared file full of `if product == …`, which is the problem we are solving. | The same bug has to be fixed in two products twice. |
| One `product.yaml` per product instead of two files | Two processes, two restart cycles. Worth revisiting once the folders exist — it would remove the duplicated facts entirely instead of merely testing them. | After the restructure, if the duplication keeps causing mistakes. |
| Release branch per customer | Nothing is deployed to a customer yet. | First customer install. Then: fix `main` first, cherry-pick down — never the reverse. |
| Splitting the shared yaml by concern | It is a 630-line file; splitting it mid-merge is asking for trouble. | After step 7, incrementally. |

---

## 9. Proposed, not started: a product-aware `just up`

`IRIS_PRODUCT` is supposed to say which product a checkout runs (decision 3).
Today it decides only which products the dashboard offers: nothing in the
justfile or compose reads it, `just up` always starts attendance's service, and
barber has a separate command. `RUNNING.md` documents how it actually works
now; this is what would close the gap.

**Goal:** `just up` starts exactly what `IRIS_PRODUCT` names — one product or
several — and behaves identically to today when it is unset or says
`attendance`, so nobody's daily command changes.

| Option | Shape | Cost |
|---|---|---|
| **A — dispatch** | `just up` reads `IRIS_PRODUCT` and calls today's recipes from a `case` | ~20 lines, no new files. But the shared justfile then names every product, and a fourth product means editing a shared file |
| **B — each product says how to start itself** (recommended) | `products/<name>/product.env` declares its compose service, profile, service directory, port variable, optional registration recipe and test clips; `just up` loops over `IRIS_PRODUCT` and sources them | ~60 lines plus a 6-line file per product. Adding a product edits no shared file. More indirection: what starts is in two files |
| **C — generate containers** | skip compose for pipelines; `just` runs `docker run` per product from the manifest | fully generic, but reimplements what compose already does (lifecycle, ordering, `down`). Not recommended |

**Under either A or B, fix three things found while writing this:**

1. `deepstream-sidecar-barber` publishes telemetry on a **hardcoded 9103**
   instead of an `.env` variable — two checkouts running barber would collide.
2. `deepstream-sidecar-fire` (profile `roles`) now runs **attendance's**
   settings file. It is a leftover from the fire/attendance role split and a
   duplicate waiting to double-process a camera; delete it.
3. Re-check that `just down` stops every product's host processes, not only the
   ones that exist today.

**Risk and verification:** the risk is the daily command. Record what is
running (containers, ports, pid files), make the change, run `just up live`
again and diff the two lists; then `IRIS_PRODUCT=barber just up live` and
confirm barber comes up and attendance does not. One commit, easy to revert.

---

## 10. Glossary

- **Pipeline / sidecar** — the process that reads cameras, runs the models and publishes
  results. One program, started once per product.
- **Base config** — the shared yaml describing how a pipeline works.
- **Overlay / profile** — a product's yaml, holding only what differs from the base.
- **Service config** — a product's yaml describing its business rules.
- **Role** — which product a pipeline process is running as (`IRIS_SIDECAR_ROLE`).
- **Monorepo** — one repository holding several products. About how code is stored and
  released, not about how similar the products are.
- **CI** — a server that runs checks automatically on every merge request.
- **Smoke test** — a fast check that a product still starts and is coherent. Not a
  thorough test; it answers "is it obviously broken?"
- **Blast radius** — how many products a change can affect.
- **Shell** — the part of the frontend every product shares: sidebar, header, navigation.
- **Submodule pointer** — this repo records one exact commit of the `frontend` repo. It can
  point at only one app at a time.
