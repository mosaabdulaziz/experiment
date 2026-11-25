# perception/models/DeepStream-Yolo-PPE

The helmet / high-visibility vest detector (`pgie_ppe`) of the `ppe` module (`perception/modules/ppe/`): VoxDroid
YOLOv8s trained on the Construction Site Safety dataset, run on each tracked person's crop and placed after the
tracker.

## Licence — approved for current non-commercial use; revisit before any commercial deployment

Decision of the team lead (2026-10-26): the model may be used **now, while IRIS is not used commercially**. **Before
any commercial use** the company will either buy a licence or switch to a freely licensed model. Re-check this section
before such a deployment.

| part | licence |
|---|---|
| VoxDroid training code ([repo](https://github.com/VoxDroid/Construction-Site-Safety-PPE-Detection)) | MIT, as stated by the repo |
| the weights (`best.pt`, and the ONNX exported from it) | AGPL-3.0: a YOLOv8 (Ultralytics) fine-tune; the exported file's own metadata says `license: AGPL-3.0` |
| the training data: Construction Site Safety (Roboflow; Kaggle copy `snehilsanyal/construction-site-safety-image-dataset-roboflow`) | not checked |

- `perception/models/DeepStream-Yolo-Fire/README.md` ("Licence — deliberate exception") records that this repo
  otherwise avoids AGPL; like the fire detector, this model is a scoped exception, here approved for non-commercial
  use only.
- Switching models: nothing downstream depends on this checkpoint. The module reads only the four PPE labels from
  `labels_ppe.txt` and the `[batch, anchors, 6]` raw-anchor output every detector here uses, so a freely licensed PPE
  detector with the same labels replaces it with a new ONNX and config in this folder.

## Getting the model (DVC)

`voxdroid_yolov8s_css.onnx` is versioned with DVC, not git (DATA.md; it was in Git LFS before 2026-10-29): `just
data-pull models-ppe` fetches it, and `just ppe-check` checks it against the SHA-256 below. The module refuses to
start, with a message pointing here, when the file is missing. It can also be rebuilt from the public weights (below).

Expected file: `perception/models/DeepStream-Yolo-PPE/voxdroid_yolov8s_css.onnx`, 44,598,110 bytes,
SHA-256 `cdd4f0afe9081144dc972eb396594f995386017807aef2ce9dce4a0b41c44656` (the copy in use since 2026-10-26).

To rebuild it (CPU is enough; a throwaway venv with `ultralytics`, which brings PyTorch, and `onnx`):

1. Weights: `best.pt` of run `yolov8s_ppe_css_200_epochs` in the VoxDroid repo, YOLOv8s, 200 epochs, imgsz 640,
   22,544,362 bytes, SHA-256 `470cc1d2f39774ade966488719d20635da56431123a8b189ec87fec041f0bc47`:

   ```bash
   curl -fLo best.pt https://raw.githubusercontent.com/VoxDroid/Construction-Site-Safety-PPE-Detection/main/Model-Training/Outputs/runs/detect/yolov8s_ppe_css_200_epochs/weights/best.pt
   sha256sum best.pt
   ```

2. Export with this repo's raw-anchor recipe, **not** Ultralytics `nms=True` (see "The batch>1 trap" in the fire
   README):

   ```bash
   python perception/models/DeepStream-Yolo-Fire/export_raw_anchor_onnx.py best.pt \
       perception/models/DeepStream-Yolo-PPE/voxdroid_yolov8s_css.onnx --imgsz 640
   ```

   The file in use was made with Ultralytics 8.4.189 and PyTorch 2.14.1 (opset 17): input `images [batch, 3, 640,
   640]`, output `output [batch, anchors, 6]` (box xyxy, score, class id), dynamic batch.

3. Check it. **The file's SHA-256 never matches a rebuild**: Ultralytics writes the export date into the model's
   metadata. Compare the model without its metadata instead; with the same Ultralytics and PyTorch versions it is
   `30d682f4c8657c67e921ca144222e9312b7cb66d109337abd502bcfae58f4740`:

   ```bash
   python -c "import hashlib, onnx, sys; m = onnx.load(sys.argv[1]); del m.metadata_props[:]; \
   print(hashlib.sha256(m.SerializeToString(deterministic=True)).hexdigest())" voxdroid_yolov8s_css.onnx
   ```

   With other versions that hash can differ too; then check the outputs instead: on a few frames at batch 2, boxes,
   scores and class ids must match the PyTorch model (`YOLO("best.pt")`) and the copy in use.

The TensorRT engine (`voxdroid_yolov8s_css.onnx_b<batch>_gpu0_fp16.engine`) is built on first use beside the ONNX
(about a minute) and is never committed (`.gitignore`: `*.engine`).

## Classes

`labels_ppe.txt` is `model.names` in order: `0 Hardhat, 1 Mask, 2 NO-Hardhat, 3 NO-Mask, 4 NO-Safety Vest, 5 Person,
6 Safety Cone, 7 Safety Vest, 8 machinery, 9 vehicle`. `config_infer_ppe.txt` drops 1, 3, 5, 6, 8 and 9
(`filter-out-class-ids`); the module uses the other four. The ids overlap COCO's (Hardhat = 0 = person); a GPU spike
(2026-10-26) showed no effect in the tracker, and the detector now runs after the tracker anyway.

## Settings that must not change casually

- `process-mode=2` on the pgie's persons (`operate-on-gie-id=1`, `operate-on-class-ids=0`), with
  `scaling-filter=1` (bilinear crops; the default, nearest, gave crops the model could not read), while the module
  declares the GIE `role="secondary"` (after the tracker): see the comments in the config and
  `perception/modules/ppe/README.md`. `batch-size=16` is the crops per inference; the module keeps it.
- `gie-unique-id=4` (1 = pgie, 2 = face, 3 = fire and plate).
- `pre-cluster-threshold=0.10`, `nms-iou-threshold=0.6`: the values the rules were evaluated with. "No vest" means no
  Safety Vest box >= 0.10 on the person, so a higher floor changes the rules' evidence.

## Accuracy (ppe-compliance, its own evaluation)

Box level on the dataset's 82-image test split (ppe-compliance, reproduced 2026-10-21, Ultralytics, CPU, imgsz 640):

| class | precision | recall | mAP50 |
|---|---|---|---|
| Person | 0.92 | 0.87 | 0.91 |
| Hardhat | 1.00 | 0.93 | 0.95 |
| NO-Hardhat | 0.84 | 0.63 | 0.64 |
| Safety Vest | 0.96 | 0.92 | 0.93 |
| NO-Safety Vest | 0.94 | 0.79 | 0.84 |

Person-level results (with VoxDroid's own persons) are in ppe-compliance's `docs/EVALUATION.md`. In IRIS the persons
come from YOLO11x and frames pass through the 1080x720 mux, so person-level numbers are re-measured there.
