# DeepStream-Yolo-Shelf: the empty-shelf (void) detector

The detector of the `shelf_void` module (`perception/modules/shelf_void/README.md`): boxes of empty shelf space,
read by the module through the label `empty`. It runs on the full frame, after the tracker (gie id 5,
`config_infer_shelf.txt`).

> **Licence: none.** The source repository has **no licence file**, and the model was trained with Ultralytics
> (**AGPL-3.0**). The source `.pt` is a **pickle**: load it only from a trusted copy. It is used here for the demo and
> may be replaced; a team decision is needed before any production or customer use.

| file | what |
|---|---|
| `detect_product_empty_space.onnx` | the model, exported for nvinfer (below). Versioned with **DVC** (`detect_product_empty_space.onnx.dvc`, group `models-shelf`): `just data-pull models-shelf` |
| `config_infer_shelf.txt` | nvinfer config: full frame, keeps class 0 (`empty`), floor 0.02, NMS IoU 0.3 |
| `detect_product_empty_space_640.onnx` | the same model exported at **640** (DVC: `detect_product_empty_space_640.onnx.dvc`, group `models-shelf`) |
| `config_infer_shelf_640.txt` | nvinfer config of the shelf preset (`stability: tracker`): the 640 ONNX, NMS IoU 0.7 (the shelf-demo's), otherwise as above |
| `labels_shelf.txt` | `empty`, `product` |

## Provenance

- Source: https://github.com/K-saif/Empty-Shelf-Alert-System, `Models/detect_product_empty_space.pt`, commit
  `6da67fa200359842b34e1e17548cbf5de27eac28` (2026-01-21). Only the weights were taken, none of that repository's code.
- `.pt`: 5,459,290 bytes, SHA-256 `d06fd0ee0a27c38b2c124046b20df5313af7b247286fd7f70e9b208f9dbd0e02`. YOLO11n,
  Ultralytics 8.4.0, classes `{0: empty, 1: product}`; the checkpoint's own metadata says `license: AGPL-3.0`. A scan
  of its pickle opcodes (no unpickling) found only `torch.*`, `ultralytics.*`, `collections` and `builtins.set`.
- First used in the standalone pipeline `~/IRIS/shelf-monitoring` (commit 02538da), where the module's heat map and
  alert logic come from.

## Getting the model

Expected file: `perception/models/DeepStream-Yolo-Shelf/detect_product_empty_space.onnx`, 10,439,971 bytes,
SHA-256 `d9b26c2695350ccb081965cee902f5c79e2af6beb0140f622ce879a1b95461be` (the copy exported 2026-11-02).
`just data-pull models-shelf` fetches it (DATA.md), and `just shelf-check` checks it against this SHA-256. It can also be
rebuilt from the `.pt` (a venv with `ultralytics`, `onnx`, `onnxslim`; CPU is enough):

```bash
python perception/models/DeepStream-Yolo-Fire/export_raw_anchor_onnx.py detect_product_empty_space.pt \
    perception/models/DeepStream-Yolo-Shelf/detect_product_empty_space.onnx --imgsz 1280
```

- **`--imgsz 1280`, not 640**: in the standalone runs, far (small) voids were only found at 1280.
- The raw-anchor recipe, as for PPE and fire (not Ultralytics `nms=True`, see "The batch>1 trap" in the fire README):
  input `images [batch, 3, 1280, 1280]`, output `output [batch, anchors, 6]` (box xyxy, score, class id), dynamic batch.
  Made with Ultralytics 8.4.198, PyTorch 2.11.0, onnx 1.23.2 (opset 17).
- The file's SHA-256 never matches a rebuild (Ultralytics writes the export date into the metadata). Without its
  metadata the model hashes to `14682bec5dc136fb4c5c75b529f7860b2986f10273bb54ae887cae6d84513a08` with those versions
  (the command is in `../DeepStream-Yolo-PPE/README.md`, "Getting the model", step 3).
- Checked at batch 2 against the PyTorch model on two frames of `data/uploads/shelf/demo.mp4`, the same letterboxed
  1280x1280 input: the same `empty` boxes after NMS (all matched at IoU > 0.9), scores within 0.0005.

The TensorRT engine (`detect_product_empty_space.onnx_b<batch>_gpu0_fp16.engine`) is built on first use beside the
ONNX and is never committed (`.gitignore`: `*.engine`).

## 640 input (the shelf preset)

The accepted shelf-demo runs the void model at **640** (its training size) with Ultralytics' default NMS IoU 0.7, so
the shelf preset (`stability: tracker`) uses `config_infer_shelf_640.txt`; the heat map keeps the module default
(1280, NMS 0.3), whose results above it was tuned on.

- `detect_product_empty_space_640.onnx`: 10,439,692 bytes, SHA-256
  `2b4c550c78139842dc9369a85e68002470ea475a84ee558a572e2c375d3b6572` (exported 2026-11-12),
  same recipe with `--imgsz 640` (Ultralytics 8.4.59, onnx 1.22.0, onnxslim 0.1.95, opset 17); input
  `images [batch, 3, 640, 640]`, output `output [batch, anchors (8400), 6]`.
- Checked at batch 2 against the PyTorch model on frames 1177 and 2400 of the shelf-demo's `test.mp4`, the same
  letterboxed 640x640 input: identical class ids, scores within 3e-6, boxes within 0.006 px.
- Engine: `detect_product_empty_space_640.onnx_b<batch>_gpu0_fp16.engine`, built on first use like the 1280 one.

## Swapping the model

The module never reads a class id: any detector whose labels file names a class `empty` (or the module's `label`
setting) works. Export it the same way, point `onnx-file`, `labelfile-path`, `num-detected-classes` and
`filter-out-class-ids` at it, and re-check the module's heat-map settings (`min_area`, `close_kernel`, `show_thresh`)
against a recorded clip: they depend on how the model's scores and boxes behave over time.
