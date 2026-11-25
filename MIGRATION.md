# Moving work written before the restructure

For anyone holding commits on a branch that predates `integration/monorepo`
(2026-09-23) — in practice, `feat/barber-shop-monitoring` and the frontend's
`feat/barber-ops`. Your work is not lost and nothing of yours was rewritten:
your branches are untouched. What changed is where the files live, so commits
written against the old paths need carrying across rather than merging.

Read `MULTI_PRODUCT_PLAN.md` for *why* the layout is what it is. This file is
only *how* to move.

## Where things went

| Before | After |
|---|---|
| `barber-service/` | `products/barber/` |
| `attendance-service/` | `products/attendance/` |
| `sidecar/` | `perception/pipeline/` |
| `sidecar/sidecar.barber.yaml` | `products/barber/sidecar.yaml` |
| `sidecar/config_tracker_nvdcf_barber.yml` | `products/barber/tracker.yml` |
| `DeepStream-Yolo*` | `perception/models/DeepStream-Yolo*` (names kept) |
| the `frontend` submodule | `frontend/` — ordinary source here, one app, `src/products/<name>/` |

Python imports moved with them: `from sidecar.sources import …` is now
`from perception.engine.sources import …`.

## Moving your commits

**1. Push everything first.** Nothing can be carried across while it only
exists on your machine.

**2. Branch from the new layout.**

```bash
git fetch origin
git switch -c barber/next origin/integration/monorepo
```

**3. Copy your commits across** — the ones you made after `2584f3f`, the last
commit already replayed into the new layout:

```bash
git log --oneline 2584f3f..origin/feat/barber-shop-monitoring   # what is left to move
git cherry-pick 2584f3f..origin/feat/barber-shop-monitoring
```

Git follows the renames on its own: your commit edits `barber-service/visits.py`,
this branch has it at `products/barber/visits.py`, and the three-way merge maps
one onto the other. Where it cannot, it stops and you resolve as usual, then
`git cherry-pick --continue`.

**Turn on `rerere` first** if you expect to redo this — git then remembers each
resolution and replays it next time:

```bash
git config rerere.enabled true
```

**4. Your dashboard work** lives at `frontend/src/products/barber/` now. The
frontend is no longer a submodule and no longer one-app-per-branch, so changes
you made on `feat/barber-ops` after the copy are carried over by hand —
`git log 756184b..origin/feat/barber-ops` in the frontend repo shows what.

**5. Check it, then keep working on the new branch.** Your old branch stays as
it is; it just stops receiving commits.

## Four things that will bite

**1. Your virtualenv is broken, and the error will not say so.**
A virtualenv hard-codes its own absolute path, so moving the folder breaks
every command in it. `uv sync` does *not* notice ("Checked N packages"). The
symptom is `.venv/bin/uvicorn: No such file or directory`.

```bash
rm -rf products/barber/.venv && (cd products/barber && uv sync)
```

**2. Anything that counted `../` is wrong.**
Your product sits two levels down now. Recipes and scripts that walked up to
the repo root by counting `..` land in `products/` instead. In the justfile use
`{{justfile_directory()}}`; in Python, `Path(__file__).resolve().parents[2]`.

**3. A docker mount to a path that moved fails *silently*.**
Docker creates an empty directory instead of refusing, and the container then
reports a missing module. If a container cannot find code that plainly exists,
check the `-v` paths first.

**4. Tests can pass while the product is broken.**
Config paths, mount paths and recipe paths are not covered by any test. After
cherry-picking, run your product, not just the suites:

```bash
just check                    # the same checks CI runs: both suites + the dashboard build
just barber-up barber-1       # your pipeline, your clip
```

## What your product declares now

`products/barber/sidecar.yaml` states what barber runs, instead of inheriting
it: its camera, its face-ownership rule (`tracked`, with the `highest-top`
fallback), and no fire detector. The shared file
(`perception/pipeline/sidecar.yaml`) holds mechanism only — model paths, driver
knobs — and turns no detector on. If a setting of yours seems ignored, check
whether it belongs in your file rather than the shared one; a test walks every
product folder and fails if one declares no cameras.

## What not to do

- **Do not rebase or force-push a branch someone else has a copy of.** That is
  how the two of us end up with duplicated history. Carry commits across with
  `cherry-pick` instead.
- **Do not copy an old folder back in** to make a path work. If something
  cannot find a file, fix the reference; the old paths are gone on purpose.
- **Do not edit another product's folder.** Shared code (`perception/`, the
  justfile, compose) is reviewed by both owners — say in the merge request
  which products you ran.
