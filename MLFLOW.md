# MLflow: model records, registry and manifests

How a model gets from training to a site, and how we can always say which bytes run where and why. DVC keeps the
**bytes** ([DATA.md](DATA.md)); MLflow keeps the **record**: what produced a model, how it scored, and which version each
product runs. The decisions behind this are in the spike, `docs/spikes/mlflow-tracking-registry.md` (branch
`spike/mlflow-tracking-registry`).

- [1. Concepts](#1-concepts)
- [2. Using the server](#2-using-the-server)
- [3. Train, export, evaluate, compare](#3-train-export-evaluate-compare)
- [4. Import an externally trained model](#4-import-an-externally-trained-model)
- [5. Register, promote, roll back](#5-register-promote-roll-back)
- [6. Manifests at startup](#6-manifests-at-startup)
- [7. The server](#7-the-server)
- [8. Known gaps](#8-known-gaps)
- [9. Where the code is](#9-where-the-code-is)

## 1. Concepts

| term | what it is here |
|---|---|
| **run** | one execution of a script: a training, an export, an import, a pipeline evaluation. Holds params, metrics, tags (git commit, DVC data md5, lockfile hash) and artifacts (weights, reports) |
| **experiment** | a folder of runs. Training and export runs go to the experiment named after the model (`vehicle-detector`, `plate-detector`); imports to `imported-models`; pipeline scores to `mtmc-pipeline-eval` |
| **registered model** | one per capability (spike D2): `vehicle-detector`, `plate-ocr`, `face-recognizer`, ... The list is `mtmc/tracking/registry_conventions.yaml` |
| **version** | one model file (plus its companions: labels, dict) registered from one run, identified by its sha256. Versions are never edited; a new file is a new version |
| **alias** | a name pointing at one version: `candidate`, `production-<product>` or `production-<product>-<site>` (e.g. `production-barber-palm-hills`). Promoting = moving an alias |
| **sidecar** | `<model file>.mlflow.json`, committed next to the model's `.dvc` pointer: the run, sha256, registered model and version of that exact file. Evaluation scripts use it to find the run of the model they evaluate |
| **manifest** | `products/<product>/models.manifest[.<site>].json`, committed: every model file a deployment loads, as the registered version and sha256 its production alias points at. Checked at startup |

```
 training script ──run──▶ experiment (vehicle-detector)        DVC: the bytes (.dvc pointer in git)
       │ log_model_file        │ export run = child, source_run_id            │
       ▼                       ▼                                              │
  best.onnx + best.onnx.mlflow.json ──register──▶ registered model vehicle-detector: v1, v2, ...
                                                       │ aliases: candidate, production-multicam ──▶ v2
  eval script ──metrics eval/<set>/*──▶ the model's run      │
                                                       ▼
                     python -m mtmc.tracking.manifest generate ──▶ products/multicam/models.manifest.json (git)
                                                       │
                     sidecar startup: every loaded file listed, present, same sha256 ──▶ start (or refuse)
```

## 2. Using the server

The team server is **`http://10.2.0.50:15050`** (also `http://cmubuntu:15050`), on the LAN only. Every script reads the
standard MLflow variables:

```bash
export MLFLOW_TRACKING_URI=http://10.2.0.50:15050
export MLFLOW_TRACKING_USERNAME=iris
export MLFLOW_TRACKING_PASSWORD=...        # on cmubuntu: set -a; . ~/mlflow-server/secrets/client-iris.env; set +a
```

- **Run everything as the shared `iris` user**: training, exports, imports, evaluations, registering. The user that
  creates an experiment owns it; everyone else can only read it (the server's default permission is READ), and an
  evaluation writes into the model's run. `admin` is for administration only (users, permissions, cleanup), never for
  runs. A run in an experiment someone else created needs, once, an EDIT grant for iris by the admin
  (`mtmc/tracking/eval.py` docstring has the call).
- Other people get their own user: `~/mlflow-server/app/serverctl.sh create-user <name>` on cmubuntu (credentials in
  `~/mlflow-server/secrets/client-<name>.env`).
- **The client** is `mlflow==3.18.0` (the server's version). It is not in a lock yet (section 8): install it into the
  environment you train in.
- **When the server is down, nothing waits and nothing fails**: training, exports and evaluations probe
  `<uri>/health` (3 s) and log to the local offline store `mlruns-offline/mlflow.db` (gitignored), tagged
  `tracking.mode=offline`, with a warning. `MLFLOW_TRACKING_URI` unset does the same. There is no sync tool: offline
  runs stay local (an offline evaluation is an `eval-pending` run there). Re-run with the server up when the record
  matters. The registry and the manifest generator never fall back: they refuse an unreachable server.
- `docker-compose.mlflow.yml` runs a throwaway local server (port 15000) for development.

## 3. Train, export, evaluate, compare

Run from the repo root, in the research environment (with `mlflow==3.18.0`), server variables set. Turn on Ultralytics'
MLflow integration once per machine, or the per-epoch metrics and weights are not logged (the scripts warn):

```bash
yolo settings mlflow=True
```

**Train.** One run per training; the run stays open through the final validation and export.

```bash
python -m mtmc.finetune_vehicle_detector                      # the deployed 1-class recipe (vehicle_detection_1cls)
python -m mtmc.finetune_vehicle_detector --export             # + export child run, writes products/el-mohandes/models/vehicle_1280_1cls/
python -m mtmc.finetune_plate_detector --epochs 100
python -m mtmc.finetune_vehicle_detector --no-track           # record nothing (Ultralytics' callback off too)
```

What the run gets: every CLI arg (`cli.*`), the fixed hyperparameters (`train.*`), Ultralytics' params and per-epoch
metrics, the final validation as `eval/val/*`, tags `git.commit`, `git.dirty`, `data.<dataset>.dvc_md5`,
`env.lockfile(_sha256)`, `base_weights.sha256`, and `weights/best.pt`, `weights/last.pt`, `args.yaml`, `results.csv`,
plots (not the `epoch*.pt` checkpoints). `best.pt.mlflow.json` is written beside `best.pt`. At the end the script prints
the run URL and the exact command to register the result. It never registers by itself.

**Export.** `--export` above, or standalone (its own run; `source_run_id` read from the weights' sidecar, or
`--source-run-id`; `unknown` with a warning when neither):

```bash
python -m mtmc.export_vehicle_onnx --weights products/el-mohandes/runs/vehicle_detector/<name>/weights/best.pt \
    --size 1280 --out-dir products/el-mohandes/models/vehicle_1280_1cls
```

The export run copies the training run's `data.*` tags and its `eval/*` metrics as `source_run/eval/*` (measured on the
.pt), so the ONNX can be registered from it.

**Evaluate on a named set.** The metrics go into the run of the model evaluated (its sidecar, or `--run-id`), as
`eval/<set>/<metric>`, with tags `eval.<set>.dvc_md5` (the data version), `.path`, `.git_commit`, `.evaluated_at`,
`.script`. A model with no run is refused (with the command that gives it one). The same set evaluated again on a
**different** data version is refused (`--force` overwrites and keeps the old md5 in `eval.<set>.previous_dvc_md5`);
data no `.dvc` pointer versions is tagged `unversioned` with a warning.

```bash
python -m mtmc.finetune_vehicle_detector --test-only --model <best.pt>      # set <dataset>-val
python -m mtmc.finetune_plate_detector --test-only --resume <best.pt>       # set <dataset>-test
python -m mtmc.audit_vehicle_detector                                       # deployed vehicle model, set el-mohandes-13clips
python -m mtmc.eval_ocr_quality --crops-dir <dir>                           # deployed plate_rec.onnx, set plate-ocr-golden
python -m mtmc.run_pipeline --scene S01                                     # own run in mtmc-pipeline-eval, set aicity22-S01
```

Flags of the model evaluations (`--test-only`, the audit, the OCR evaluation): `--eval-set <name>`, `--run-id <run>`,
`--force`, `--no-track`; `run_pipeline` takes `--eval-set`, `--manifest`, `--no-track`. The audit reads the vehicle model's
threshold from the nvinfer config that loads it (`--conf` when none does). The OCR evaluation reads crops exactly as the
sidecar does; it has no labels, so its numbers are text rate and confidence, not accuracy, and the per-crop CSV (plate
text) stays local. The pipeline run is tagged with every model it used (`--manifest <file>` adds their registered
versions).

**Compare two versions in the UI.** Open the experiment (e.g. `vehicle-detector`), tick the two runs, press
**Compare**: params, metrics and per-epoch curves side by side. From the registry: **Models** -> the model -> each
version links to the run it came from; its tags hold the sha256, the lineage and a summary of its evaluation metrics.
Compare evaluations only on the same set **and** the same `eval.<set>.dvc_md5`.

## 4. Import an externally trained model

Models trained elsewhere (the Kaggle plate OCR) or third-party models (Re-ID, CLIP, face, pose...) get an import run.
The list is `mtmc/tracking/imported_models.yaml`: per entry the capability, `source` (`imported` or `kaggle`), the
model files (main file first), `expected_sha256` pins, the training records to parse (`records:` kinds
`ultralytics_args`, `ultralytics_results`, `paddle_config`, `ocr_training_report`, `staff_json`), provenance and
evidence. Every value comes from a file in the repo; unknown is written as unknown.

```bash
python -m mtmc.tracking.import_models --dry-run                               # the plan; no MLflow call
python -m mtmc.tracking.import_models --only plate-ocr/v5_arabic_mobile        # import one entry
```

A new Kaggle training: `dvc add` the new files, update the entry's files, pins and records, run the import (a new
model file is a new import key, so a new run), then register and promote it (section 5). `--force` re-imports an
unchanged entry (the older run is tagged `import.superseded_by`). The import writes the sidecars; commit them.

## 5. Register, promote, roll back

```bash
# register: the run must be FINISHED, carry its lineage (git commit, lockfile tag, data.* for a trained run) and,
# for a trained run, evaluation metrics and a real lockfile (env.lockfile=missing/none is refused; an imported or
# Kaggle run may have either); idempotent on the file's sha256
python -m mtmc.tracking.registry register --model products/el-mohandes/models/vehicle_1280_1cls/best.onnx
python -m mtmc.tracking.registry register --run-id <run> --name vehicle-detector --file best.onnx

# promote: point an alias at a version; prints the version it pointed to before
python -m mtmc.tracking.registry promote vehicle-detector 3 --alias candidate
python -m mtmc.tracking.registry promote vehicle-detector 3 --alias production-multicam

# the deployment's manifest, from the production aliases (from a checkout with the registered bytes pulled)
python -m mtmc.tracking.manifest generate --product multicam
python -m mtmc.tracking.manifest generate --product barber --site palm-hills
python -m mtmc.tracking.manifest files --product barber --site bekky          # what it covers (no MLflow)

# all imported models at once, with their production aliases (what main runs; idempotent)
python -m mtmc.tracking.registry baseline --dry-run
```

`register --model` updates the sidecars (the main file's and its companions'). Commit them.

**Shipping a new version** is one commit: the model's `.dvc` pointer (`dvc add`), its sidecars, and the regenerated
manifest. `just data-push` before pushing. On the site: `git pull`, `dvc pull`, restart.

**Rolling back** is the same, backwards, in **one commit**:

1. `python -m mtmc.tracking.registry promote <model> <previous version> --alias production-<product>[-<site>]`
2. put the previous bytes back: `git checkout <commit that shipped it> -- <file>.dvc <file>.mlflow.json` then
   `dvc checkout <file>.dvc`
3. `python -m mtmc.tracking.manifest generate --product <product> [--site <site>]`
4. commit the pointer, the sidecar and the manifest together; on the site `git pull`, `dvc pull`, restart.

The alias lives in MLflow, not git; the manifest's git history is the record of what each site ran.
`just barber-palm-staff r3|r4` switches the Palm Hills staff model in the yamls but not the manifest: also promote
the matching `staff-classifier` version to `production-barber-palm-hills` and regenerate.

## 6. Manifests at startup

The sidecar (`perception/engine/main.py`, after role overrides, before anything loads), the barber service and the
fall detection service check every model file they are about to load against the manifest their yaml names
(`model_manifest:`). Checked: the file is listed, present, and has the listed sha256. Files the manifest lists but this
configuration does not load (a disabled module, another role) are not checked. Nothing here talks to MLflow.

| message | meaning | what to do |
|---|---|---|
| `... loads <file>, which this manifest does not list` | the config loads a model nobody registered/promoted for this deployment | register and promote it, regenerate the manifest |
| `lists <file> (...), which does not exist. Run dvc pull <file>.dvc` | the bytes were not pulled | `dvc pull <file>.dvc` |
| `<file> is not <model> vN (alias): sha256 expected ..., on disk ...` | a different file than the registered version | `dvc checkout <file>.dvc`; if the manifest is what changed, its pointer must change in the same commit |
| `no model manifest for ... model files are NOT verified` (warning) | `model_manifest:` unset or the file missing | generate and commit it |

A manifest that exists is always enforced. **No manifest only warns** unless `MODEL_MANIFEST_REQUIRED=1` (or
`true`/`yes`) is set in the service's environment; then it refuses too. Rollout plan: warn now, while every deployment
gets its manifest committed and sites have pulled the bytes; then set `MODEL_MANIFEST_REQUIRED=1` per deployment once
it has started cleanly with its manifest. Turning it on is a deployment decision; nothing in the repo sets it.

The InsightFace pack (`face_recognition`, `customer_faces`, the app's face registration, the attendance enrollment
CLIs) is loaded from the repo's DVC copy, `perception/models/insightface/models/buffalo_l`, never downloaded: a missing
file refuses startup with the `dvc pull` to run. TensorRT engines record the ONNX sha256 they were built from, so a
same-size model swap rebuilds them.

## 7. The server

One MLflow 3.18 server on cmubuntu, basic auth, Postgres for runs and the registry, artifacts on SharePoint (through an
rclone mount) until the MinIO from #83, daily database backups. Ports, secrets, operations (`serverctl.sh status`
before a shutdown), backups, restore and its known limitations: [`tools/mlflow-server/README.md`](tools/mlflow-server/README.md).

## 8. Known gaps

- **Trained runs before the locks**: runs logged before `tools/mtmc-research/uv.lock` existed are tagged
  `env.lockfile=missing` (with the installed packages as an artifact). The registry refuses to register such a run if
  it is `source=trained`: re-run it (or its export) from the locked environment. mlflow 3.18.0 is in the root lock's
  `test` group (CI runs the MLflow tests) and in `tools/mtmc-research`.
- **One sidecar per file**: a `<file>.mlflow.json` names one run and one version. Identical companions shared across
  versions keep the first registered run: `DeepStream-Yolo-Seg/labels.txt` belongs to person-segmenter v1 (barber's
  r384) and is also loaded by v2 (proximity's 1280). A later import or registration of the same bytes keeps the
  sidecar and warns ("shared by several versions"); that run still has the file as an artifact and its sha256 tag, and
  each deployment's manifest names the version its alias resolves to.
- **OCR evaluation crops missing**: `mtmc/output/Good-plates-OCR/` does not exist yet; once supplied it must be
  `dvc add`-ed, until then its evaluations are `unversioned`. The crops have no labels (quality indicators only).
- **Unknown provenance**: `object-detector` (yolo11x.onnx), `face-detector` (yolov8n-face.onnx) and
  `kids-car-detector` (kids_car.onnx) are registered with `provenance: unknown` in the import catalog.
- **buffalo_l licence**: InsightFace's pretrained models are for non-commercial research only; flagged to the lead.
- **DVC remote `work` lacks the buffalo_l objects**: they were pushed to `onedrive` only. A plain `dvc pull` (default
  remote `work`) reports them missing; `dvc pull -r onedrive` works. The commands to copy them are in DATA.md.
- **Paddle export not instrumented**: `mtmc/export_plate_ocr_onnx.py` needs a locked Paddle environment first.
- **Staff classifier trainer not in the repo**: the staff models are imported versions; their training cannot be
  tracked or reproduced here.
- **Multi-GPU (DDP) training**: Ultralytics re-runs training in a subprocess that `--no-track` and the checkpoint
  filter do not reach (its MLflow callback runs there), and MLflow marks the run FINISHED when that subprocess exits
  (the final status is set correctly at the end). Train on one GPU when the record matters.
- **No offline sync**: runs logged while the server was down stay in `mlruns-offline/`.
- **`just barber-palm-staff` does not switch the manifest** (section 5).

## 9. Where the code is

| path | what |
|---|---|
| `mtmc/tracking/__init__.py`, `_backend.py`, `_lineage.py` | the tracking helper: runs, lineage tags, offline fallback, sidecars |
| `mtmc/tracking/yolo.py`, `eval.py` | training and evaluation logging |
| `mtmc/tracking/import_models.py`, `imported_models.yaml` | imports and their catalog |
| `mtmc/tracking/registry.py`, `registry_conventions.yaml` | register, promote, baseline; model names, products, sites, deployments |
| `mtmc/tracking/manifest.py`, `perception/engine/model_manifest.py` | manifest generator; startup verifier (stdlib only) |
| `perception/insightface_pack.py` | where the InsightFace pack is loaded from |
| `tests/test_mlflow_*.py`, `tests/test_model_manifest*.py` | tests; `tests/test_mlflow_e2e.py` is the whole chain (`IRIS_MLFLOW_E2E=1`, optionally `IRIS_MLFLOW_E2E_URI`) |
| `tools/mlflow-server/` | the server's deployment |
