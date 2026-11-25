# Provenance — the vendored copy

`products/fall_detection/fall_detection/` is a copy of the vendor's detector package
(`F:\fall_detection_module\src`), taken on **2026-10-25** from the vendor's commit
`e31325a` ("updated readme file"). The vendor's checkout is the baseline, not a dependency:
the product never imports from it, and the product's tests never reach into it except to
prove the copy is faithful (ruling A5, design §4.1).

## What was copied

| From the vendor | Into the product |
|---|---|
| `src/*.py` (7 modules) | `fall_detection/*.py` — byte-identical, except the seam below |
| `tests/*.py` (9 test files + `conftest.py` + `synth.py`) | `tests/vendor/` — byte-identical, except `conftest.py` (the wiring) |
| `config.yaml` | `config.yaml` (the product root) — the vendor's master config, the monitor's thresholds, plus the soft score's span widths (design §4.2, ruling A6; each span is commented where it sits and defaults to the gate it measures) |
| `pytest.ini`, `requirements.txt` | not copied: the product's `pyproject.toml` is the dependency contract |

## What was changed — the seam, and only the seam

`fall_detection/fall_monitor.py` differs from the pristine file in seven edits, the product's
seam (design §4.2), and `fall_detection/tsstg_model.py` in one more. The sealed candidate gains
**exactly two keys**, and no decision changes:

| Added | Where | What it is |
|---|---|---|
| `reading` | the candidate's packaging (`fall_monitor.py:1018-1024`) | `{"probs": [...8 floats...], "top": <class>}` — the smoothed classifier vector at packaging time, or `None` when no window existed |
| `fall_paths` | the candidate's packaging | `{"a": bool, "b": bool, "c": bool}` — which fall paths latched for the event |

The other five edits are the track state that carries the latch: `fall_paths` on the track, set
where the gate fires, carried onto the event, and cleared at the two places the vendor clears
`pending_fall_t0`; the sixth is the candidate's own packaging, which adds the two keys.

The last two edits — one in each of `fall_monitor.py` and `tsstg_model.py` — rewrite the
vendor's `ROOT = Path(__file__).resolve().parent.parent` as
`ROOT = Path(__file__).resolve().parents[1]`. **That is the same path**: `parents[1]` *is*
`parent.parent`, in every layout, so the vendor's behaviour is unchanged. The repo's rule
(`tests/test_product_paths.py`) bans counting parents in a product's sources, because it broke
silently when a product moved one level deeper; the copy lives two levels deep, so the same
expression has to be spelled with an explicit depth here.

`tests/pristine_manifest.json` holds all eight as exact string pairs, and
`tests/test_seam_vendor.py` applies them in reverse to prove the rest of every copied module is
the pristine text — on any machine, with or without the vendor's checkout.

The **path scores are the product's own**. The copy carries the latch as data and nothing else:
`monitor.attach_path_scores` turns it into `{"action": None, "kinematic": …, "cold": …}` with
`scoring.collapse_score`, at the packaging moment, from the candidate's own kinematics. The
formula has one implementation (`scoring.py`) and the copy imports nothing from the product.

The boolean gate — the conditions under which the vendor raises a fall candidate — is
**unchanged**. `tests/test_seam_vendor.py` proves it: the same fixture produces the same
candidates, with the same types, tracks and timings, on the pristine file and on the copy.

## The one thing the environment needed: `mmaction_compat.py`

`mmaction_compat.py` is the product's own module, not a copy of the vendor's. It exists because
**the published `mmaction2` wheel cannot be imported as it stands** (open-mmlab/mmaction2#2714,
open since 2023-10-18; the one-file fix is still an unmerged PR, #2721):

* `mmaction/models/__init__.py` imports every model family eagerly;
* the `localizers` family imports `mmaction.models.localizers.drn`, a package the wheel omits —
  upstream's directory carries no `__init__.py`, so `find_packages()` skipped it. This is the
  unconditional break: the wheel cannot be imported at all;
* the `multimodal` family needs `transformers` (and then `timm`), which the wheel does not
  declare. It is only reached in an environment that happens to have `transformers` installed;
  the product does not depend on it.

The copy needs three submodules of that package (`models.utils.graph`, `models.backbones.stgcn`,
`models.heads.gcn_head`). `mmaction_compat` installs a bare `mmaction.models` package whose
`__path__` is the real directory, so the submodules import and the broken family imports never
run; it also imports `mmaction.models.losses`, which is what registers the `CrossEntropyLoss`
that `GCNHead` builds through the model registry.

Nothing is stubbed out and no third-party code is vendored: the ST-GCN backbone, the graph and
the head are the real upstream classes. The product's dependency contract stays the pinned
`mmaction2` wheel, and the behaviour is the same however that wheel was installed.

`mmaction_compat` must be imported **before** `fall_detection.tsstg_model`; the product's
`boot.py`, `monitor.py` and `tests/conftest.py` do exactly that.

## The rules

* **The pristine vendor checkout is never edited.** `F:\fall_detection_module` is read-only
  here; it is the baseline the copy is diffed against.
* **The copy is diffed against the vendor, not against memory.** A change to the copy shows up
  as a diff against `F:\fall_detection_module\src`; anything beyond the seam is a defect.
* **The copy's `conftest.py` is the only rewritten test file** — it is the wiring: the vendor's
  checkout laid its code out as `<root>/src` and `<root>/tests`, the product's layout is
  `<product>/fall_detection` and `<product>/tests/vendor`. The vendor's nine test files are
  byte-identical to the vendor's (ruling A10).
* **The vendor's checkpoints are not in the repository.** `products/fall_detection/weights/` is
  populated by hand from the vendor's release (the directory is gitignored); the boot gate
  verifies whatever is there against the vendor's own `WEIGHT_SHA256_REGISTRY` and refuses to
  start on a mismatch (design §8).
