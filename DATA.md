# Data: models, datasets, sample videos, golden clips

Anything that is not code but is needed to run a use case, train a model or reproduce a test is versioned
with [DVC](https://dvc.org), not git. Git holds a small `<file>.dvc` pointer (a hash and a size); the bytes live in
the team store: the OneDrive folder `PhyAI/IRIS/repo-data` today, an S3-compatible store (MinIO) later.

A checkout plus `just data-pull` gives the exact models and data that commit was built and tested with.

Not versioned: runtime output (`data/segments`, `alert_clips`, `vehicle_crops`, `events.db`, logs ...) and
TensorRT engines (`*.engine` are built per machine by nvinfer from the ONNX files).

## Where the data lives (remotes)

| remote | where | status |
|---|---|---|
| `work` (default) | `ssh://iris@10.2.0.60/data/IRIS/repo-data/dvc-store` on the training server (group `datausers`, setgid) | in use; needs SSH access to the server (LAN, or its Tailscale address 100.102.162.114) |
| `onedrive` | `PhyAI/IRIS/repo-data/dvc-store` on SharePoint, through an rclone mount (below) | a complete mirror of `work` (45,162 objects, 16.6 GiB), filled with `just data-mirror` and verified 2026-10-29: names and sizes, a hash check of 210 objects downloaded back (the 10 largest included) and a real `dvc pull -r onedrive` through the mount. Uploading is slow (about 1.4 MB/s on Wi-Fi) but pulling is fast (about 49 MB/s); `work` stays the default because it is on the LAN |
| `minio` | `s3://iris-dvc` | defined, not created yet (end of this file) |

The OneDrive mirror is filled with `just data-mirror` (an `rclone copy` from the local cache; resumable, it skips what is
already there) and checked with `just data-mirror-check`. Both need a machine that has the objects (the cache) and the
`iris-onedrive` remote; from a laptop on Wi-Fi the full store (18 GB) takes about 5 hours at roughly 1 MB/s.

Any remote holds the same layout, so they are interchangeable: `dvc pull -r <name>`, `dvc push -r <name>`, and
`rclone sync` / `rsync` between stores. `just data-push` pushes to the default; mirror with `dvc push -r onedrive`.

**Objects on `onedrive` that `work` does not have yet.** The InsightFace buffalo_l pack
(`perception/models/insightface/models/buffalo_l/*.onnx.dvc`, 5 files, 341 MB) was pushed to `onedrive` only, from a
machine without SSH access to 10.2.0.60. To make `work` complete again, someone on the training server (or any machine
with SSH to `iris@10.2.0.60` and the `iris-onedrive` rclone remote), in a clone that has these pointers:

```bash
just data-mount
dvc pull -r onedrive perception/models/insightface/models/buffalo_l/*.onnx.dvc
dvc push -r work perception/models/insightface/models/buffalo_l/*.onnx.dvc
dvc status -c -r work perception/models/insightface/models/buffalo_l/*.onnx.dvc   # "Cache and remote 'work' are in sync."
just data-umount
```

Until then a plain `dvc pull` (default remote `work`) reports these 5 objects missing; `dvc pull -r onedrive` gets them.

## Layout: where each use case's data sits

In the repo (the working tree, after `just data-pull`):

```
perception/models/...        shared detector, face, fire, seg, staff, kids-car weights (per product config)
models/                      barber: re-ID, tracker, segmentation
perception/models/DeepStream-Yolo-Pose/   fall: the pose model (ONNX); products/fall_detection/weights/ its checkpoints
perception/models/DeepStream-Yolo-PPE/    ppe: the helmet / hi-vis vest detector (ONNX)
mtmc/models/reid/            multicam (any client): the re-ID backbone
products/el-mohandes/        everything that belongs to the client El-Mohandes Group (the multicam product itself,
                             products/multicam, stays generic and can serve other clients)
  models/                    finetuned vehicle detector (vehicle_1280_1cls), licence-plate detector (finetuned = the deployed
                             one, plate_640 = its 640 export), Arabic plate OCR (v5_arabic_mobile-fine-tuned and its resume checkpoint)
  datasets/                  plate_detection, vehicle_detection_1cls, plate_ocr/{ealpr,t2,egyptian_filtered},
                             plate_ocr_crops/ (OCR finetune crops), vehicle_frame_labels.zip, vehicle_newly_reviewed.zip,
                             plate_reviewed_crops_extra.zip (reviewed batches not inside the training sets)
  videos/                    the client's full-resolution camera footage
  runs/  training/           vehicle detector training runs, the plate OCR notebook
data/uploads/                replay clips; the sidecar's file:// prefix and the UI uploads point here
  attendance/                enter*, exit*, camera_* clips, camera_5 annotations and face crops
  el-mohandes/               the client's lighter re-encodes of products/el-mohandes/videos (same names, different files)
  multicam/                  the public AI City clips
  barber/  fire/  safety/    one folder each (fire/part1..3)
  fall/  ppe/                the fall demo clip; the two PPE demo videos
  misc/                      red-light footage, video11, other one-offs
  golden/                    the golden-replay clips (outputs of the golden_clips stage)
tests/baselines/             golden baselines
```

New clips go in the folder of their use case; a clip uploaded through the UI lands in the `data/uploads` root and is
not versioned until someone moves it into a folder and runs `dvc add` on it.

In the store: the store itself is content-addressed and not meant for browsing. For people there is a second tree
next to it, `by-usecase/<use case>/<kind>/<the file's path in the repo>`, made of hardlinks (no extra disk space):

```
/data/IRIS/repo-data/by-usecase/{attendance,multicam,barber,fire,safety,misc,shared}/...
```

e.g. `el-mohandes/models/products/el-mohandes/models/finetuned/best.pt`, `attendance/samples/data/uploads/attendance/enter.mp4`.
Rebuild it after a push with `just data-view`; read-only reference, DVC never reads it.

## Versions: one path, history in git

DVC works like git. A path holds the **current** version; each `dvc add` changes the pointer, and the commit says which
bytes it was. Older versions are not kept as `v1/`, `v2/` folders: they are history, and the store keeps their objects.

- **Paths and names say what the thing is, never which version** (`finetuned/best.onnx`, `plate_detection`). A new
  model then needs no config change, and the commit pins what was deployed.
- **Retiring a version:** remove its pointer, tag the last commit that has it, list it below.
  Get it back with `git checkout <tag> -- <path>.dvc && dvc checkout <path>` (or in a scratch clone at the tag).
  The store keeps the bytes until someone runs `dvc gc`: never run it without `-A` and the tags in mind.
- **Side by side only when both are used at once** (an A/B, a rollback someone runs, different architectures) --
  e.g. the Palm Hills staff rounds `_palm.onnx` (r4) and `_palm_r3.onnx`, switched by `just barber-palm-staff r3|r4`.
- **Datasets follow the same rule.** One path per dataset, updated in place. Different data is a different folder
  (`plate_detection`, `vehicle_detection_1cls`, `plate_ocr/ealpr` are not versions of each other); a frozen benchmark
  set is named by what it is. Which dataset trained which model belongs in a `dvc.yaml` training stage (dataset in,
  weights out), which is the training-pipeline work (#84).
- A training **run** is an experiment record, not a version of the model; keep the deployed run and the ones worth
  comparing, the rest is history.
- **Models: DVC holds the bytes, MLflow the record** ([MLFLOW.md](MLFLOW.md)): the run that produced a model, its
  evaluations, its registered version, and which version each product runs (aliases). A model's `.dvc` pointer has a
  `<file>.mlflow.json` sidecar beside it, and each deployment commits a model manifest
  (`products/<product>/models.manifest*.json`) that the services check at startup. Shipping or rolling back a model
  is one commit: the pointer, its sidecar and the regenerated manifest.

## Retired versions

Tag `data/retired-2026-10-28` is the last commit with these as pointers.

| what | replaced by | why it is history |
|---|---|---|
| plate detector `finetuned/v1/best.{onnx,pt}` | `finetuned/best.{onnx,pt}` (was `v2`) | superseded; only a `--model v1` baseline in two research scripts |
| run `vehicle_yolo11s_1cls` (mAP50-95 0.780) | run `vehicle_yolo11s_1cls_v3` (0.915, **deployed** as `vehicle_1280_1cls`) | older, lower score, same data |
| run `vehicle_yolo11s_1cls_v2` (0.781) | same | same |
| run `vehicle_yolo11s` (0.443) | same | trained on the older multi-class `vehicle_detection` set, no longer in the repo |
| run `smoke_yolo26_lrg` | -- | a 2-epoch smoke test |
| `train.log`, `train_1cls.log`, `train_1cls_v2.log` | `train_1cls_v3.log` | logs of the runs above |

Kept on purpose: run `vehicle_yolo26s_1cls_v1` (a different architecture, not deployed, 0.838: an experiment record).

## Not versioned on purpose: raw footage extractions on the training server

`/data/IRIS/TESTING_PLAYGROUND/surveillance-camera-sys/mtmc/output/` still holds, as the **only copy**:

| what | size | note |
|---|---|---|
| `vehicle_frames/` | 35 GB | 77 sessions, 31,840 frames; 1,009 are in `vehicle_detection_1cls`. Its labels are `vehicle_frame_labels.zip` (in DVC). |
| `plate_crops.zip` | 26 GB | 131 raw per-session folders (about 23 GB) plus the reviewed batches; the reviewed part that is not in `plate_detection` is `plate_reviewed_crops_extra.zip` (in DVC). |

They were cut from the client's 26 July recordings, which are no longer on that server (none of the 208 sessions has its
source video there), so they cannot be regenerated. They are customer footage and too big for the store, which is why
they are not in DVC: see issue #83, "recorded production video ... needs a legal/PII review first". Do not delete them
before someone decides: keep where they are, move to cold storage, or accept the loss (the reviewed and labelled parts are safe).

Cleaned up from that folder on 2026-10-28, after checking every file by content hash against the store (and the store
objects themselves): the two training datasets and their zip/tar duplicates, the finetuned and third-party model weights,
the base `DeepStream-Yolo` ONNX files, and the epoch checkpoints and plots of the training runs. The disk went from
1.9 GB to 52 GB free.

## One-time setup

1. **DVC**, pinned: `uv tool install 'dvc[s3,ssh]==3.67.1'`
2. **FUSE and rclone**: `sudo apt install fuse3`, rclone 1.6x or newer.
3. **The rclone remote `iris-onedrive`** (the team library `PhyAI`; our folder is `IRIS/repo-data` inside it).
   In a real terminal run `rclone config`: new remote named `iris-onedrive`, type `onedrive`, leave client id/secret
   empty, and at "Type of connection" choose **3, Sharepoint site name or URL** (not the default "OneDrive Personal or
   Business", which only lists your own drives), enter `https://urcaai.sharepoint.com/sites/urcaai`, pick the
   **PhyAI** drive, and sign in in the browser. Check: `rclone lsf iris-onedrive:IRIS/repo-data` lists `dvc-store/`.
   Beside it are `mlflow-artifacts/` and `mlflow-backups/`, written only by the MLflow server
   ([`tools/mlflow-server/README.md`](tools/mlflow-server/README.md)): `just data-mount` hides them, and nothing may
   sync, copy, delete or purge `IRIS/repo-data` itself, only its subfolders (`repo-data/dvc-store`).
   The token stays in `~/.config/rclone/rclone.conf`; it is never committed.
4. `just data-mount` mounts it at `.data-remote/` (gitignored). `just data-umount` unmounts.
5. `just data-pull` fetches the default groups. `just bootstrap` does the same on a fresh clone.

## First run on a fresh clone: what each solution needs

`just bootstrap` does the common part: frontend packages, the attendance Python environment, `.env` (with
`COMPOSE_PROJECT_NAME` left unset, so the project is `iris`), the default data groups and the supervisor bundle.
A solution that needs more than that checks for it before it starts and, at a terminal, offers to pull it.

| Solution | Start it with | Needs beyond `just bootstrap` |
|---|---|---|
| attendance | `SIDECAR_CONFIG=tests/configs/attendance-chirla-cam15.yaml IRIS_PRODUCT=attendance just up cam-15` | nothing: the bundle brings the `cam-15` clip, face photos and `attendance.db` |
| barber | `just barber-up` | the models are in the default groups. A local test clip: `just data-pull samples-barber`, then stream it into `barber-c` (`test_streamer/stream_recordings.py --ids barber-c --files "barber-c=barber/9Hwju1yZBAM.mp4"`); the site clips of `bekky` and `palm-hills` live on the training server |
| multicam | `just multi` | `just data-pull el-mohandes-samples` (the 13 full-resolution clips, 3.5 GB, plus the light replay copies); `just multi` offers it. The truck UI needs the truck-tracking repo next to this one (or `TRUCK_UI_DIR`) |
| fall detection | see `products/fall_detection/README.md` (needs torch and mmaction2, GPU) | the models are in the default groups (`just data-pull models-fall` if you skipped them); the demo clip is `samples-fall` |
| PPE demo | `just ppe-live-sim VIDEO=test.mp4` | `just data-pull models-ppe samples-ppe` (the model and two videos, 0.5 GB); `just ppe-check` verifies them |
| golden replay | `tools/engine_golden_check.sh attendance zones fire multicam barber` | the default groups; the first run builds TensorRT engines |

The golden baselines were recorded with an older engine build. Two builds of the same ONNX differ at the 0.01 px level
and a detection near the 0.5 threshold can flip, so on a fresh checkout `attendance`, `zones`, `multicam` and `barber`
can show small differences from the baseline that unmodified `main` shows too: compare with a run from `main`, or
re-record the baseline (`tools/engine_golden_run.sh`, then `dvc add` and `just data-push`).

If something will not start: `docker compose` rejecting the project name means `.env` still has the
`<your-name-or-initials>` placeholder (bootstrap comments it out; a `.env` from before that needs the same);
`GstFileSrc ... Resource not found` in the sidecar log means a clip was not pulled (`just data-pull <group>`).

Layout of the store: `dvc-store/` is the DVC remote (content-addressed, never edited by hand); `docs/` holds
documents that are not data (plate OCR reports, the Neurocle evaluations).

## Everyday workflow

```bash
just data-pull                      # models, golden clips, bundle
just data-pull datasets samples-multicam   # other groups; `just data-pull all` for everything
just data-status                    # what differs from the remote

# a model or dataset changed
dvc add perception/models/staff/staff_dinov2_s.onnx   # updates the .dvc pointer
git add perception/models/staff/staff_dinov2_s.onnx.dvc && git commit
# (a model a product loads: register and promote it and regenerate the manifest in the same commit, MLFLOW.md)
just data-push                      # BEFORE pushing the branch: CI checks the bytes are on the remote

# a commit from last month
git checkout <sha> && dvc checkout  # models and data match that commit

# a new sample video
dvc add data/uploads/barber/new-shop.mp4    # one pointer per FILE, see below
# then add a glob for it to data/catalog.yaml if no group matches, and run: just data-catalog
```

Rules that bite:

- **In `data/uploads/` and `products/el-mohandes/videos/`, `dvc add` one file at a time, never the folder.** `dvc checkout` on a
  tracked folder deletes files that are not in it, including clips people upload through the UI.
- Every `.dvc` pointer must belong to exactly one group in `data/catalog.yaml` (`just data-catalog` checks; CI runs it).
- `just data-push` before you push the branch.
- A weight the nvinfer `.txt` config sits next to stays a per-file pointer; the `.txt` configs stay in git.
- **A pointer dvc cannot see is silently skipped** (plain `dvc push` / `dvc status -c` then say "in sync"). This happened when
  `.gitignore` used `dir/*` rules although `git check-ignore` showed the pointer as visible. New ignore rules for a data
  folder follow the `dir/**`, `!dir/**/`, `!dir/**/*.dvc` form; `tools/data/check_pushed.py` (CI) is the check that catches it.
- The pulled copy is the reference. `dvc repro golden_clips` re-cuts the golden clips, but the fire clip is
  re-encoded, so another ffmpeg build can produce different bytes.

## Golden replay

`dvc.yaml` has the stage `golden_clips`: `tools/make_golden_clips.sh` cuts `data/uploads/golden/{fire,barber,vehicles}.mp4`
from `data/uploads/fire_samples/part1/bucketfire.mp4`, `data/uploads/barber/9Hwju1yZBAM.mp4` and
`products/el-mohandes/videos/5-CH23-enter.mp4` (all pointers, all in `dvc.lock` with their hashes). `enter-3.mp4` (the attendance and
zones configs) is a plain sample. The baselines `tests/baselines/*.before.jsonl` are pointers too. On a fresh machine:
`just data-pull` and run the replay as in [perception/TESTING.md](perception/TESTING.md).

## Switching the remote to MinIO

`.dvc/config` already defines the `minio` remote (`s3://iris-dvc`, endpoint to be filled in).

1. Create the bucket and credentials (separate from the stack's own buckets).
2. Copy the store as it is: `rclone sync iris-onedrive:IRIS/repo-data/dvc-store minio:iris-dvc` (the layouts are identical).
3. `dvc remote modify minio endpointurl <url>`; credentials in `.dvc/config.local` or `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`.
4. `dvc remote default minio`, commit `.dvc/config`; point the CI job `dvc-pointers` at it.
5. Keep the OneDrive copy read-only as a backup.

## What is versioned

<!-- catalog:start (generated by tools/data/catalog.py docs) -->

| Group | Use case | Artifacts | Size | Pulled by default | What it is |
|---|---|---:|---:|:-:|---|
| `models-fall` | fall detection | 12 (12 files) | 454.4 MB | yes | The pose model (yolo26x-pose ONNX for the sidecar, yolo11n-pose ONNX for the export self-test), the pose checkpoints (.pt) and the skeleton action-recognition checkpoints (mmaction2_*, pyskl_*.pth) the fall product loads. Vendor weights; the product's boot gate checks the two its config loads by SHA-256. |
| `models-ppe` | PPE compliance (helmet and hi-vis vest) | 1 (1 files) | 42.5 MB | yes | The VoxDroid YOLOv8s Construction Site Safety detector (ONNX) of the ppe module. AGPL-3.0 weights, approved for non-commercial use only (perception/models/DeepStream-Yolo-PPE/README.md). |
| `models-shelf` | empty shelf space (supermarket voids) | 3 (3 files) | 25.1 MB | yes | The K-saif YOLO11n empty/product detector of the shelf_void module -- the ONNX at 1280 and at 640 input (detect_product_empty_space_640.onnx) and the .pt checkpoint they were exported from (detect_product_empty_space.pt). NO licence in its source repository and trained with Ultralytics (AGPL-3.0); demo use until the team decides (perception/models/DeepStream-Yolo-Shelf/README.md). |
| `models-perception` | every product (DeepStream sidecar) | 24 (24 files) | 2.1 GB | yes | Detector, face, fire, segmentation, staff-classifier and kids-car weights the nvinfer configs and perception modules load, and the InsightFace buffalo_l pack (perception/models/insightface/models/buffalo_l, insightface v0.7 release) that face_recognition, customer_faces, the app's face registration and the attendance enrollment CLIs load. InsightFace's pretrained models are licensed for non-commercial research only (flagged to the lead). |
| `models-barber` | barber (re-ID, tracker, segmentation) | 3 (3 files) | 261.3 MB | yes | The re-ID and tracker ONNX and the segmentation weights `just barber-up` otherwise downloads and exports (sha-pinned in perception/pipeline/models.*.sha256, which stay the way to regenerate them). |
| `golden` | golden replay (perception/TESTING.md) | 9 (18 files) | 9.8 MB | yes | The clips the golden configs replay and the recorded baselines. enter-3.mp4 is a plain sample (attendance and zones configs); the other clips are the outputs of the golden_clips and golden_proximity_clip stages in dvc.yaml. |
| `benchmark-raw` | regression benchmark (just regress) | 1 (288 files) | 82.3 MB | no | The raw data behind a machine's recorded baseline in benchmarks/baselines/<machine>/ -- run files, the 3 h soaks, the host recorder with the sidecar container logs, capacity and detector-size results, the overnight run. Not needed to run or compare a baseline; needed to rebuild one or to look behind a number. The recorded golden outputs of a baseline (benchmarks/baselines/<machine>/golden) are in the `golden` group. |
| `bundles` | attendance | 1 (1 files) | 322.3 MB | yes | The supervisor bundle `just bootstrap` unpacks (face photos and attendance.db seed). |
| `training-artifacts` | El-Mohandes -- plate OCR and vehicle detector provenance | 15 (25 files) | 440.6 MB | no | Training records behind the client's deployed models -- the Arabic plate OCR logs and resume checkpoint, and the vehicle detector runs (best.pt, results.csv, args). The deployed vehicle_1280_1cls/best.pt is the best.pt of run vehicle_yolo11s_1cls_v3, trained on the vehicle_detection_1cls dataset. |
| `el-mohandes-models` | El-Mohandes -- multicam vehicles, plates, plate OCR | 7 (7 files) | 177.9 MB | yes | The client's finetuned models the multicam sidecar runs -- the vehicle detector (vehicle_1280_1cls), the licence-plate detector (finetuned, and its 640 export) and the Arabic plate OCR. They were finetuned on the client's footage. |
| `models-mtmc` | multicam (any client) | 1 (1 files) | 97.8 MB | yes | The re-ID backbone the multicam tracker uses (mtmc/models/reid). Pretrained, not client-specific. |
| `el-mohandes-datasets` | El-Mohandes -- training (plate detection, plate OCR, vehicle detection) | 12 (43231 files) | 7.4 GB | no | The client's training sets. Plate OCR sets come from the Plate-OCR hand-over; the vehicle and plate detection sets were exported from the training server. |
| `samples-attendance` | attendance replay | 19 (5369 files) | 781.9 MB | no | Entrance/exit and camera_* clips, the camera_5 annotations and the CHIRLA face crops. |
| `el-mohandes-samples` | El-Mohandes -- multicam replay and MTMC scripts | 26 (26 files) | 4.2 GB | no | The client's camera footage -- full-resolution clips (products/el-mohandes/videos) and the lighter re-encodes used for RTSP replay (data/uploads/el-mohandes). Same names, different files. |
| `samples-multicam` | multicam (any client) | 2 (2 files) | 190.7 MB | no | The public AI City benchmark clips. |
| `samples-fall` | fall detection demo | 1 (1 files) | 750.9 KB | no | The fall product's demo clip. |
| `samples-ppe` | PPE demo | 2 (2 files) | 482.0 MB | no | The two PPE demo test videos (test.mp4 17 MB, test2.mp4 489 MB) that `just ppe-live-sim` loops into a simulated camera. |
| `samples-shelf` | empty-shelf demo | 1 (1 files) | 23.6 MB | no | The supermarket-aisle demo clip (demo.mp4, 21 s, 1280x720) that `just shelf-live-sim` loops into a simulated camera. |
| `samples-barber` | barber replay | 1 (1 files) | 204.2 MB | no | The 32-minute barber-shop recording. |
| `samples-fire` | fire/smoke replay | 3 (3 files) | 305.6 MB | no | Fire and smoke sample clips. |
| `samples-safety` | safe-walk / zone violations | 6 (6 files) | 105.0 MB | no | Safe-walk violation clips. |
| `samples-proximity` | pedestrian-vehicle proximity replay | 1 (1 files) | 466.1 MB | no | The Camera189 site recording (forklift placement) the proximity golden clip is cut from. |
| `samples-misc` | manual tests and benchmarks | 6 (6 files) | 466.6 MB | no | Red-light footage, video11 and other one-off clips. |

#### Every artifact

### `models-fall`

- `perception/models/DeepStream-Yolo-Pose/yolo11n-pose.onnx` (11.8 MB)
- `perception/models/DeepStream-Yolo-Pose/yolo26x-pose.onnx` (221.1 MB)
- `products/fall_detection/weights/mmaction2_bm.pth` (11.9 MB)
- `products/fall_detection/weights/mmaction2_bone.pth` (11.9 MB)
- `products/fall_detection/weights/mmaction2_jm.pth` (11.9 MB)
- `products/fall_detection/weights/mmaction2_joint.pth` (11.9 MB)
- `products/fall_detection/weights/pyskl_b.pth` (11.9 MB)
- `products/fall_detection/weights/pyskl_bm.pth` (11.9 MB)
- `products/fall_detection/weights/pyskl_j.pth` (11.9 MB)
- `products/fall_detection/weights/pyskl_jm.pth` (11.9 MB)
- `products/fall_detection/weights/yolo11n-pose.pt` (6.0 MB)
- `products/fall_detection/weights/yolo26x-pose.pt` (120.4 MB)

### `models-ppe`

- `perception/models/DeepStream-Yolo-PPE/voxdroid_yolov8s_css.onnx` (42.5 MB)

### `models-shelf`

- `perception/models/DeepStream-Yolo-Shelf/detect_product_empty_space.onnx` (10.0 MB)
- `perception/models/DeepStream-Yolo-Shelf/detect_product_empty_space.pt` (5.2 MB)
- `perception/models/DeepStream-Yolo-Shelf/detect_product_empty_space_640.onnx` (10.0 MB)

### `models-perception`

- `perception/models/DeepStream-Yolo-Face/yolov8n-face.onnx` (12.4 MB)
- `perception/models/DeepStream-Yolo-Fire/fire_s.onnx` (42.6 MB)
- `perception/models/DeepStream-Yolo-Fire/fire_smoke_box.onnx` (9.2 MB)
- `perception/models/DeepStream-Yolo-Seg/yolo26l-seg-736x1280.onnx` (107.0 MB)
- `perception/models/DeepStream-Yolo-Seg/yolo26l-seg-r384.onnx` (107.0 MB)
- `perception/models/DeepStream-Yolo/damoyolo_m.onnx` (107.7 MB)
- `perception/models/DeepStream-Yolo/damoyolo_s.onnx` (62.2 MB)
- `perception/models/DeepStream-Yolo/yolo11l.onnx` (96.9 MB)
- `perception/models/DeepStream-Yolo/yolo11m.onnx` (76.8 MB)
- `perception/models/DeepStream-Yolo/yolo11s.onnx` (36.2 MB)
- `perception/models/DeepStream-Yolo/yolo11x.onnx` (217.4 MB)
- `perception/models/depth/yolo26m-depth-736x1280.onnx` (84.3 MB)
- `perception/models/insightface/models/buffalo_l/1k3d68.onnx` (137.0 MB)
- `perception/models/insightface/models/buffalo_l/2d106det.onnx` (4.8 MB)
- `perception/models/insightface/models/buffalo_l/det_10g.onnx` (16.1 MB)
- `perception/models/insightface/models/buffalo_l/genderage.onnx` (1.3 MB)
- `perception/models/insightface/models/buffalo_l/w600k_r50.onnx` (166.3 MB)
- `perception/models/kids_car/kids_car.onnx` (9.2 MB)
- `perception/models/staff/staff_dinov2_s.onnx` (82.5 MB)
- `perception/models/staff/staff_dinov2_s_palm.onnx` (82.5 MB)
- `perception/models/staff/staff_dinov2_s_palm_r3.onnx` (82.5 MB)
- `perception/models/staff/text_model.onnx` (242.3 MB)
- `perception/models/staff/tokenizer.json` (2.1 MB)
- `perception/models/staff/vision_model.onnx` (335.4 MB)

### `models-barber`

- `models/reid/swin_tiny_market1501_aicity156_featuredim256.onnx` (108.6 MB)
- `models/seg/yolo26l-seg.pt` (60.7 MB)
- `models/tracker/resnet50_market1501_aicity156.onnx` (91.9 MB)

### `golden`

- `benchmarks/baselines/rtx-5050-laptop/golden` (3.7 MB)
- `data/uploads/attendance/enter-3.mp4` (2.6 MB)
- `tests/baselines/golden-attendance.before.jsonl` (122.3 KB)
- `tests/baselines/golden-barber-faces.before.jsonl` (990.0 KB)
- `tests/baselines/golden-barber.before.jsonl` (568.6 KB)
- `tests/baselines/golden-fire.before.jsonl` (64.4 KB)
- `tests/baselines/golden-multicam.before.jsonl` (13.8 KB)
- `tests/baselines/golden-proximity.before.jsonl` (1.7 MB)
- `tests/baselines/golden-zones.before.jsonl` (123.4 KB)
- stage `golden_clips` (dvc.yaml)
- stage `golden_proximity_clip` (dvc.yaml)

### `benchmark-raw`

- `benchmarks/raw/rtx-5050-laptop` (82.3 MB)

### `bundles`

- `data/attendance-supervisor-bundle.tar.gz` (322.3 MB)

### `training-artifacts`

- `products/el-mohandes/models/v5_arabic_mobile-fine-tuned/epoch-metrics.json` (2.6 MB)
- `products/el-mohandes/models/v5_arabic_mobile-fine-tuned/launcher.log` (1.4 MB)
- `products/el-mohandes/models/v5_arabic_mobile-fine-tuned/training` (33.1 MB)
- `products/el-mohandes/models/v5_arabic_mobile-fine-tuned/training.log` (4.3 MB)
- `products/el-mohandes/models/v5_arabic_mobile-resume-last` (246.2 MB)
- `products/el-mohandes/runs/vehicle_detector/train_1cls_v3.log` (1.1 MB)
- `products/el-mohandes/runs/vehicle_detector/train_yolo26_v1.log` (204.7 KB)
- `products/el-mohandes/runs/vehicle_detector/vehicle_yolo11s_1cls_v3/args.yaml` (1.8 KB)
- `products/el-mohandes/runs/vehicle_detector/vehicle_yolo11s_1cls_v3/results.csv` (17.9 KB)
- `products/el-mohandes/runs/vehicle_detector/vehicle_yolo11s_1cls_v3/weights/best.pt` (18.4 MB)
- `products/el-mohandes/runs/vehicle_detector/vehicle_yolo11s_1cls_v3/weights/last.pt` (18.4 MB)
- `products/el-mohandes/runs/vehicle_detector/vehicle_yolo26s_1cls_v1/args.yaml` (1.8 KB)
- `products/el-mohandes/runs/vehicle_detector/vehicle_yolo26s_1cls_v1/results.csv` (2.2 KB)
- `products/el-mohandes/runs/vehicle_detector/vehicle_yolo26s_1cls_v1/weights/best.pt` (57.5 MB)
- `products/el-mohandes/runs/vehicle_detector/vehicle_yolo26s_1cls_v1/weights/last.pt` (57.5 MB)

### `el-mohandes-models`

- `products/el-mohandes/models/finetuned/best.onnx` (36.0 MB)
- `products/el-mohandes/models/finetuned/best.pt` (36.4 MB)
- `products/el-mohandes/models/plate_640/best.onnx` (36.1 MB)
- `products/el-mohandes/models/v5_arabic_mobile-fine-tuned/inference/inference.pdiparams` (7.2 MB)
- `products/el-mohandes/models/v5_arabic_mobile-fine-tuned/inference/plate_rec.onnx` (7.3 MB)
- `products/el-mohandes/models/vehicle_1280_1cls/best.onnx` (36.4 MB)
- `products/el-mohandes/models/vehicle_1280_1cls/best.pt` (18.4 MB)

### `models-mtmc`

- `mtmc/models/reid/resnet50-11ad3fa6.pth` (97.8 MB)

### `el-mohandes-datasets`

- `products/el-mohandes/datasets/plate_detection` (3.4 GB)
- `products/el-mohandes/datasets/plate_ocr/ealpr` (278.3 MB)
- `products/el-mohandes/datasets/plate_ocr/egyptian_filtered` (42.5 MB)
- `products/el-mohandes/datasets/plate_ocr/t2` (23.9 MB)
- `products/el-mohandes/datasets/plate_ocr_crops/fine_tuned_plate_crops_day.zip` (6.2 MB)
- `products/el-mohandes/datasets/plate_ocr_crops/finetuned_plate_crops` (7.2 MB)
- `products/el-mohandes/datasets/plate_ocr_crops/ocr_finetune_crops` (3.0 MB)
- `products/el-mohandes/datasets/plate_ocr_crops/plate_crops_new` (8.4 MB)
- `products/el-mohandes/datasets/plate_reviewed_crops_extra.zip` (14.8 MB)
- `products/el-mohandes/datasets/vehicle_detection_1cls` (3.3 GB)
- `products/el-mohandes/datasets/vehicle_frame_labels.zip` (8.2 MB)
- `products/el-mohandes/datasets/vehicle_newly_reviewed.zip` (292.7 MB)

### `samples-attendance`

- `data/uploads/attendance/Bus/bus-cabin.mp4` (1.0 MB)
- `data/uploads/attendance/camera_1_2023-06-08-12-29-08_h264.mp4` (28.4 MB)
- `data/uploads/attendance/camera_2_2023-06-08-12-29-08_h264.mp4` (28.3 MB)
- `data/uploads/attendance/camera_3_2023-06-08-12-29-08_h264.mp4` (28.1 MB)
- `data/uploads/attendance/camera_3_2023-07-26-13-21-40_h264.mp4` (190.6 MB)
- `data/uploads/attendance/camera_4_2023-06-08-12-29-08_h264.mp4` (28.3 MB)
- `data/uploads/attendance/camera_4_2023-07-26-13-21-40_h264.mp4` (131.3 MB)
- `data/uploads/attendance/camera_5_2023-06-08-12-29-08.avi` (107.9 MB)
- `data/uploads/attendance/camera_5_2023-06-08-12-29-08.json` (3.3 MB)
- `data/uploads/attendance/camera_5_2023-06-08-12-29-08_h264.mp4` (28.3 MB)
- `data/uploads/attendance/camera_5_faces` (7.1 MB)
- `data/uploads/attendance/camera_6_2023-06-08-12-29-08_h264.mp4` (28.3 MB)
- `data/uploads/attendance/camera_7_2023-06-08-12-29-08_h264.mp4` (28.3 MB)
- `data/uploads/attendance/enter-4.mp4` (3.0 MB)
- `data/uploads/attendance/enter.mp4` (15.6 MB)
- `data/uploads/attendance/exit-1.mp4` (826.3 KB)
- `data/uploads/attendance/exit-3.mp4` (2.3 MB)
- `data/uploads/attendance/exit-4.mp4` (2.7 MB)
- `data/uploads/attendance/exit.mp4` (118.2 MB)

### `el-mohandes-samples`

- `data/uploads/el-mohandes/1-D41-enter.mp4` (143.8 MB)
- `data/uploads/el-mohandes/2-D16_clip.mp4` (52.5 MB)
- `data/uploads/el-mohandes/3-D52-enter.mp4` (83.7 MB)
- `data/uploads/el-mohandes/4-ch05_clip.mp4` (70.0 MB)
- `data/uploads/el-mohandes/5-CH23-enter.mp4` (30.9 MB)
- `data/uploads/el-mohandes/D01-crop.mp4` (62.8 MB)
- `data/uploads/el-mohandes/D05-crop.mp4` (34.0 MB)
- `data/uploads/el-mohandes/D09-crop.mp4` (39.1 MB)
- `data/uploads/el-mohandes/D11-crop.mp4` (43.5 MB)
- `data/uploads/el-mohandes/D15-crop.mp4` (30.2 MB)
- `data/uploads/el-mohandes/D18-crop.mp4` (72.4 MB)
- `data/uploads/el-mohandes/D20-crop.mp4` (64.6 MB)
- `data/uploads/el-mohandes/D21-crop.mp4` (62.6 MB)
- `products/el-mohandes/videos/1-D41-enter.mp4` (357.9 MB)
- `products/el-mohandes/videos/2-D16_clip.mp4` (241.0 MB)
- `products/el-mohandes/videos/3-D52-enter.mp4` (687.7 MB)
- `products/el-mohandes/videos/4-ch05_clip.mp4` (329.2 MB)
- `products/el-mohandes/videos/5-CH23-enter.mp4` (93.1 MB)
- `products/el-mohandes/videos/D01-crop.mp4` (62.8 MB)
- `products/el-mohandes/videos/D05-crop.mp4` (172.5 MB)
- `products/el-mohandes/videos/D09-crop.mp4` (89.0 MB)
- `products/el-mohandes/videos/D11-crop.mp4` (158.1 MB)
- `products/el-mohandes/videos/D15-crop.mp4` (421.6 MB)
- `products/el-mohandes/videos/D18-crop.mp4` (318.5 MB)
- `products/el-mohandes/videos/D20-crop.mp4` (211.9 MB)
- `products/el-mohandes/videos/D21-crop.mp4` (333.7 MB)

### `samples-multicam`

- `data/uploads/multicam/aicity22_c001.mp4` (76.8 MB)
- `data/uploads/multicam/aicity22_c002.mp4` (113.8 MB)

### `samples-fall`

- `data/uploads/fall/video.mp4` (750.9 KB)

### `samples-ppe`

- `data/uploads/ppe/test.mp4` (15.9 MB)
- `data/uploads/ppe/test2.mp4` (466.1 MB)

### `samples-shelf`

- `data/uploads/shelf/demo.mp4` (23.6 MB)

### `samples-barber`

- `data/uploads/barber/9Hwju1yZBAM.mp4` (204.2 MB)

### `samples-fire`

- `data/uploads/fire/part1/bucketfire.mp4` (143.6 MB)
- `data/uploads/fire/part2/roomfire.mp4` (131.8 MB)
- `data/uploads/fire/part3/printer31.mp4` (30.1 MB)

### `samples-safety`

- `data/uploads/safety/0_tr101.mp4` (13.9 MB)
- `data/uploads/safety/0_tr102.mp4` (13.9 MB)
- `data/uploads/safety/0_tr103.mp4` (13.9 MB)
- `data/uploads/safety/0_tr104.mp4` (13.8 MB)
- `data/uploads/safety/5be2b040-6ca3-4940-bd5d-44f75f9182f2.mp4` (21.6 MB)
- `data/uploads/safety/6de7c2a8-2b75-4bb8-a459-d47da96132ca.mp4` (27.8 MB)

### `samples-proximity`

- `data/uploads/proximity/procimity_placement_combined.mp4` (466.1 MB)

### `samples-misc`

- `data/uploads/misc/Red light camera footage [QwueyJr-El4].mp4` (23.3 MB)
- `data/uploads/misc/YTDown.com_Shorts_Two-Cars-Go-Straight-Through-a-Red-Traff_Media_I01ORKkV-C4_001_720p.mp4` (1.8 MB)
- `data/uploads/misc/output-1.mp4` (318.6 KB)
- `data/uploads/misc/output.mp4` (430.3 KB)
- `data/uploads/misc/video11_1.avi` (439.5 MB)
- `data/uploads/misc/video11_1_h264.mp4` (1.4 MB)

<!-- catalog:end -->

## Decision: binaries already in git history

Status: open (decided for now: untrack, do not rewrite). Date: 2026-10-27. Issue: IRIS #83.

### Context

Before DVC, `main` tracked about 1.9 GB of weights, a dataset archive and a 24 MB detection dump; `.git` is about 6 GB
on a developer machine. This change removes them from the working tree (tracked files: 1,884 MB before, 15 MB after)
and puts DVC pointers in their place, but every old commit still carries the bytes, so a new clone is still large.

### Decision for now

Untrack without rewriting. Rewriting `main` forces every developer and every open branch to re-clone or rebase, and the
new storage has not been exercised by a team yet.

### To decide later (a coordinated window)

Run `git filter-repo --invert-paths` on the paths listed below, force-push, and have everyone
re-clone. Before then: confirm every checkout of an old commit that needs the old weights can still get them
(`git checkout <old> && dvc checkout` works only for commits made after this change; older commits keep the bytes in
git until the rewrite, after which they cannot run). If the old commits must stay runnable, keep a tagged bundle of the
pre-rewrite repository instead of rewriting.

Paths to strip: `perception/models/**/*.{onnx,pt,pth}`, `mtmc/models/**/*.{onnx,pt,pth,pdiparams}`, `mtmc/datasets.zip`,
`data/raw_detections.jsonl`, `scratch/**/*.mp4`, `mtmc/models/**/.cache/`, `data/uploads/**` (already removed once).

Record here after the rewrite: clone size before and after.
