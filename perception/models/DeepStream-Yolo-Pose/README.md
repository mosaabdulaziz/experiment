# perception/models/DeepStream-Yolo-Pose

The pose model for the fall product: a secondary GIE that runs on each tracked person's crop and
produces 17 COCO keypoints for that person.

| file | what it is |
|---|---|
| `config_sgie_pose.txt` | the nvinfer config (the secondary GIE), the same shape as `DeepStream-Yolo-Face/config_sgie_face.txt` |
| `labels.txt` | the model's single class, `pose` — the name the engine's `relevant_classes` filter uses to keep these items out of the published detections |
| `nvdsinfer_custom_impl_Yolo_pose/` | the output parser: it reads the three tensors the export writes (boxes, scores, keypoints), does the threshold and the NMS, and packs the 17 keypoints (51 floats) and the network's input size into the object's instance-mask field, the way the face parser packs its five landmarks |
| `yolo26x-pose.onnx` | the graph the engine loads: one self-contained file, a DVC pointer (`just data-pull models-fall`) so it reaches the deployment |
| `yolo11n-pose.onnx` | the export tool's default graph (`export_pose.py`'s `DEFAULT_WEIGHTS`), kept for the export's `--check` self-test |
| `MODEL_SWITCH.md` | the procedure for changing which checkpoint this GIE runs: the export, the config file, the module's setting and the engine cache, in that order |

**The ONNX is in git, each as ONE self-contained file.** `yolo26x-pose.onnx` (the deployed graph)
and `yolo11n-pose.onnx` (the export tool's default) are exported by the product's own script (see
`products/fall_detection/export_pose.py`) and tracked here, because the engine image carries no torch and
cannot export the model itself. The export writes no `*.onnx.data` sidecar: the graph and its weights are
one self-contained file, like every other model in `perception/models/`.

**The engine's name is derived, not declared.** nvinfer serializes the engine beside the ONNX under a name
that follows from the config file's own `onnx-file`, `batch-size`, `gpu-id` and `network-mode` — here
`yolo26x-pose.onnx_b16_gpu0_fp32.engine`, written on the workspace volume, so it survives a container
restart. There is no `model-engine-file` line here on purpose: the config file is the one place that
decides the name, and `perception/modules/pose/` derives the same name with the repo's own
`perception/engine/engine_path.py` and hands it to the element. The two must agree, or nvinfer rebuilds
from the ONNX on every boot and the engine cache never fills.

**The export proves itself.** `export_pose.py --check` compares the exported graph against
`model.predict()` on a real image and prints the difference in pixels; `ok: True` means the two
agree within 2 px. Every detection is paired with its counterpart and every box, score and joint
confidence is compared — not just the first detection. A joint the network did not see (its own
confidence below the parser's threshold) carries an arbitrary position, so its *position* is excused
and only its confidence is compared; the count of seen and unseen joints is printed so the excuse is
visible. The
export is model-agnostic — the checkpoint's own head says how many joints it has — so `yolo26x-pose`
goes through the same code (see `MODEL_SWITCH.md`).

**The parser is proved on a plain host.** `tests/test_pose_parser.py` compiles
`nvdsparsepose_Yolo.cpp` against a stub of DeepStream's declarations
(`nvdsinfer_custom_impl_Yolo_pose/test/stub/`) and runs 40 checks over it: the threshold, the NMS, the
packed keypoints, the network's input size, both output layouts (one crop's, and a batch of them),
and the refusals. The stub carries the real declarations the parser uses, under DeepStream's real
names, so a field the parser gets wrong fails on the host instead of in the image build.

**The parser is built by the image.** `perception/models/build_parsers.sh` builds every
`nvdsinfer_custom_impl_*` directory under `perception/models/` and installs the libraries where the
configs expect them (`/opt/nvidia/deepstream/deepstream/custom_parsers`), so adding this directory
needed no Dockerfile change. Each library must export an `NvDsInferParse*` symbol, or the image
build fails.

**The re-inference interval lives in this file, not in the yaml.** nvinfer's element property
`interval` gates the PRIMARY GIE only; a secondary GIE's interval is the config file's
`secondary-reinfer-interval` (in frames), which nvinfer reads when it creates the element and which
has no element property. The module's `interval` setting declares the same number, and the module
refuses to start when the two disagree — so a change here without a change in the yaml (or the
other way round) fails the boot instead of running at an interval nobody asked for. The same holds
for the model: the config file names the ONNX, the module's `model` setting declares it, and the
module refuses to start when the two disagree.

**The geometry is the contract.** The network saw the person's box scaled to fit the network input
keeping its aspect ratio, centred with equal padding (`maintain-aspect-ratio=1`,
`symmetric-padding=1`). The parser leaves the keypoints in the network's INPUT pixels, and
`perception/modules/pose/` undoes that mapping. Changing either half breaks every keypoint's
position — so those two keys are declared numbers, like the interval and the thresholds: the module
refuses to start when the file declares another scaling. See the module's README.
