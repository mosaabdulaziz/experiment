# Switching the pose model

The pose model is the secondary GIE that runs on every tracked person's crop
(`perception/modules/pose/`). This is the procedure for changing which checkpoint it runs.

Four things name the model — the checkpoint, the ONNX, the config file and the module's
setting — and they have to move together. The module refuses to start when they disagree, so a
half-finished switch fails the boot instead of quietly running the wrong model.

## What a model has to be

- a YOLO **pose** checkpoint (Ultralytics): **17 COCO joints**, three values each (x, y,
  confidence), and **one class** (person). The parser reads exactly 17 x 3 floats per detection and
  the module counts the 17 joints against `min_keypoints`.
- a head that is **not** `end2end`: the export splits the head's own decoded tensor, and an
  end2end head's output is already post-processed.

`yolo11n-pose` and `yolo26x-pose` (the sidecar's shipped model) both satisfy this, and both go
through the same export and the same parser.

## The steps

### 1. The checkpoint

Put it in `products/fall_detection/weights/<name>.pt`. The shipped checkpoints are versioned
with DVC there (`just data-pull models-fall`); a checkpoint you bring is an input, not source:
`dvc add` it, commit the `.dvc` pointer and `just data-push` (DATA.md).

### 2. The export — and its proof

```
cd products/fall_detection
python export_pose.py --weights weights/<name>.pt --check
```

It writes `perception/models/DeepStream-Yolo-Pose/<name>.onnx` and prints the proof: the exported
graph's detection against `model.predict()`'s, in pixels, on a real image. **`ok: True` is the
gate** — it means the two agree within 2 px. An export that does not prove itself is not an export
to ship.

The export needs no GPU: the exported graph is run by onnxruntime's **CPU** provider, and the engine
is built where the GPU is. (The proof's other side — `model.predict()` — is torch's own forward
pass, so on a machine with a card it uses the card. That is an inference, not a build: the same
proof runs on a machine with no GPU at all.)

The ONNX is **one file**: the weights are inside the graph, not in a `<name>.onnx.data` sidecar.
That is the repo's convention (every tracked model in `perception/models/` is self-contained) and
it matters because `*.onnx.data` is ignored — a split export would ship a model whose weights are
missing.

**The ONNX has to reach the deployment.** The engine container carries no torch, so it cannot
export the model itself: the deployed graph (`yolo26x-pose.onnx`) is versioned with DVC for that
reason (`just data-pull models-fall`), and a new model's ONNX is `dvc add`ed beside it, or copied
to the deployment's workspace mount beside the config file.

### 3. The config file

`perception/models/DeepStream-Yolo-Pose/config_sgie_pose.txt` names the model in one place:

```
onnx-file=<name>.onnx
```

It does **not** name the engine. nvinfer serializes the engine beside the ONNX under a derived
name — `<onnx>_b<batch>_gpu<id>_<mode>.engine`, with the batch and the GPU from the same file
(`batch-size=16`, `gpu-id=0`) and the mode from `network-mode` (0 = fp32, 1 = int8) — and the
module derives that name and passes it to the element (`perception/engine/engine_path.py`, the
same way the fire module does). One source of truth, so a name left over from the previous model
cannot survive a switch.

### 4. The module's setting

`products/fall_detection/sidecar.yaml`:

```yaml
modules:
  pose:
    model: <name>
    interval: 2
```

The setting is the module's declaration of which model it means to run; the config file is what
nvinfer actually loads. They must agree — the module refuses to start otherwise.

### 5. The engine cache

The new engine is built on the first boot after the switch (the ONNX changed, so its derived name
changed) and written beside the ONNX. The old `<previous>.onnx_b16_gpu0_fp32.engine` is stale and
can be deleted.

## What is checked, and where

| what | checked by |
|---|---|
| the export equals `model.predict()` | `export_pose.py --check` (2 px, every detection) |
| the model names agree | `modules.pose` startup gate (`ModuleStartupError`) |
| the engine name is the one nvinfer derives | `perception/engine/engine_path.py` (the repo's own derivation) |
| the parser packs 17 x 3 floats | `tests/test_pose_parser.py` (the parser compiled against a stub of DeepStream's header, 40 checks on a host) |
| the config names the parser that is shipped | `tests/test_pose_module.py` |
| the parser exports its symbol | the image build (`build_parsers.sh` runs `nm -D`) |

## The sizes are a lever, not an accident

The GIE runs on **every tracked person's crop**, so the model runs many times per frame. The
pilot's first model was the small `yolo11n-pose` (design §3.5); the shipped one is the far
heavier `yolo26x-pose` — the vendor's monitor's own model, through the same code — so the GPU
budget has to be measured with it (`tools/engine_gpu_budget.sh`).
