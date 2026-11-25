# pose

The keypoints of every tracked person (a secondary YOLO pose detector, run on each person's crop), published in capture
pixels. The module docstring in `__init__.py` lists what it publishes and why the coordinates are the whole risk of it;
this page is the product configuration and the notes a product needs.

```yaml
modules:
  pose:
    model: yolo26x-pose             # the model this module runs; the config file names the ONNX it loads
    interval: 2                   # the SGIE's own interval: 2 = 12.5 Hz on a 25 Hz pipeline
    detection_threshold: 0.25      # the detection floor; the config file declares the same number
    keypoint_confidence_threshold: 0.30
    min_keypoints: 6
```

| setting | meaning |
|---|---|
| `config` | the pose detector's nvinfer config. It names the model (`onnx-file`) and the build parameters the engine's name is derived from |
| `engine` | the engine path nvinfer deserializes. Empty: the repo's own derivation from the config file (`perception/engine/engine_path.py`), which is where nvinfer writes the engine anyway |
| `model` | the model this module means to run, by name. It must be the model the config file's `onnx-file` names — the module refuses to start otherwise |
| `detection_threshold` | the confidence a **detection** must reach to be a detection at all. It is the parser's threshold (the config file's `pre-cluster-threshold`), and the module declares the same number so the two cannot drift apart |
| `keypoint_confidence_threshold` | the confidence a **joint** must reach to count as seen. A person whose joints are all below it is not observed: they are omitted from the field, not published with zeroes |
| `min_keypoints` | how many joints a person must be seen through to be published. One confident joint is not an observation: the vendor's kinematics read the two legs (three joints each) and the torso, and below this the monitor would be measuring a body the network mostly did not see |
| `interval` | the SGIE's re-inference interval, in frames (2 = 12.5 Hz on a 25 Hz pipeline). This is the feature's first GPU lever; the camera count is the second and the model size the third |

The two thresholds are different numbers about different things: `detection_threshold` is about the
**detection** (is this a person at all?) and `keypoint_confidence_threshold` is about the **joints**
(did the network see that joint?). Raising the joint floor filters no weak detection — it changes
how many joints count as seen.

## The five gates the module refuses to start through

All five are start-up refusals (`ModuleStartupError`), not warnings: they are cases where the
pipeline would look healthy and run the wrong thing.

| the gate | what it catches |
|---|---|
| the setting's `interval` against the config file's `secondary-reinfer-interval` | the feature's first GPU lever running at an interval nobody asked for |
| the setting's `model` against the config file's `onnx-file` | a config left over from another model — which would also mean the engine belongs to that other model |
| the setting's `detection_threshold` against the config file's `pre-cluster-threshold` | a config edited without the module: the parser would keep a different set of detections, and the module would publish a different set of people than the one that was declared |
| the config file's `onnx-file` against the files beside it | a model that is not there: nvinfer would have nothing to build the engine from, and the failure would surface inside the pipeline rather than at start-up |
| the config file's `maintain-aspect-ratio` and `symmetric-padding` | a different scaling of the crop into the network's input: the module's letterbox inverse is the inverse of **that** scaling, so every joint would land in the wrong place — in every frame, silently |

The engine's **name** is not a gate: it is derived, not declared. nvinfer serializes the engine
beside the ONNX as `<onnx>_b<batch>_gpu<id>_<mode>.engine`, and the module derives that name from
the config file's own `onnx-file`, `batch-size`, `gpu-id` and `network-mode`
(`perception/engine/engine_path.py`, the same helper the fire module uses). One source of truth,
so a name left over from the previous model cannot survive a switch.

Switching models: `perception/models/DeepStream-Yolo-Pose/MODEL_SWITCH.md`.

## The interval: the setting and the config file must agree

nvinfer's element property `interval` gates the **primary** GIE; it does not gate a secondary one.
A secondary GIE's re-inference interval is the **config file's** `secondary-reinfer-interval`, which
nvinfer reads when it creates the element and which has no element property. So the number lives in
two places, and the module checks them at start-up:

* the yaml's `modules.pose.interval` — the declaration,
* `secondary-reinfer-interval` in the model's `config_sgie_pose.txt` — the mechanism.

If they disagree the module raises `ModuleStartupError` and the boot fails, rather than running at
an interval nobody asked for and measuring the wrong thing. Change both, or neither.

The module asks the engine for **no** element interval — the shipped `sgie_plate` is configured
exactly this way — so `perception/engine/**` stays untouched and the pose GIE is built like every
other secondary GIE in the repo.

Whether 2 is the right number is an experiment, not a setting: `products/fall_detection/measure.py`
replays the corpus at the interval and at every frame, and the comparison is written down before
the product ships (requirement N2). The replay models the interval as the detector's frame stride,
which is a pessimistic proxy — the detector's rate drops with it — and every report says so.

## What it publishes

The `pose` field on `iris:meta`, in **capture pixels** (the resolution the preview JPEG is drawn at):

```json
"pose": {"shape": [1920, 1080],
         "items": [{"id": 42, "box": [310.5, 180.0, 455.25, 612.75], "kps": [[x, y, conf], ... 17 ...]}]}
```

* `id` is the DeepStream tracker id — the id the engine draws as `#42` on the person's box, and the join key for
  everything downstream.
* `box` is **xyxy** (the consumer's convention), not the engine's left/top/width/height.
* `kps` are the 17 COCO keypoints, in capture pixels, each with its own confidence.
* `shape` is `[capture_w, capture_h]`, so the consumer never has to guess the frame's size.
* **One entry per person**, not per detection: the network can report the same person twice, and the monitor would
  otherwise see one track twice in one frame (the same `t` twice in its kinematics). The detection whose joints land
  inside its own box is the one that explains it, so it wins.

## The frames the network did not run on

The SGIE's re-inference interval means the network does not run on every frame. On those frames the frame carries
**no `pose` field at all** — and that is different from a field with no people in it:

| the frame | the field | what the consumer does |
|---|---|---|
| the network ran and saw the people | `pose` with one entry per observed person | the monitor gets those people |
| the scene is empty (no tracked people) | `pose` with an empty `items` | the monitor learns the people are gone |
| the network ran and published no item, while people are tracked | **no field** | the frame says nothing: the consumer keeps the tracks it holds |
| the network did not run (the interval skipped it) | **no field** | the frame says nothing: the consumer keeps the tracks it holds |

The third and fourth rows are the same field, and the module cannot tell them apart: a secondary GIE
that ran and found nobody produces no items, exactly like a frame it did not run on. It says nothing
in both cases — the conservative choice, because the vendor's monitor moves every track it cannot
see into the lost ones, and an empty field on a skipped frame would drop every track at the pilot's
own 12.5 Hz. The consumer reads the cases apart (`products/fall_detection/consumer.py`).

## The two questions the module has to answer

1. **Whose keypoints are these?** The secondary GIE's parent pointer arrives unset on this pipeline, so the owner is
   decided by geometry (`owner.py`): the tracked person whose box contains the item's centre.
2. **Which crop did the network see?** The keypoints are in network-input pixels; only the crop rect can put them back
   in the frame. A secondary GIE runs on the **tracked object's crop**, so the crop is the owner's rect — the item's
   own box is the network's *detection* box, not the crop. The owner is found by the stronger rule first (the
   person's rect whose letterbox inverse puts the keypoints inside the item's box), and the containment match is the
   fallback; either way the crop is that person's rect.

Both answers are geometry, and both are checked by `tests/test_pose_module.py` against a known crop.

## Notes

* The class name in the model's `labels.txt` is `pose`, so the engine's `relevant_classes` filter keeps these items out
  of the published detections. The module identifies its items by the GIE's name **or** by that class name (either one
  identifies them, so it depends on neither alone), and it marks them neither drawn nor published itself — the
  person's box is what the operator sees.
* The model and its parser live in `perception/models/DeepStream-Yolo-Pose/`; the image builds every
  `nvdsinfer_custom_impl_*` directory automatically, so a new parser needs no Dockerfile change.
* The parser is proved on a plain host, with no DeepStream and no GPU: `tests/test_pose_parser.py`
  compiles it against a **stub** of DeepStream's header (the declarations the parser uses, under
  DeepStream's real names) and feeds it a **synthetic tensor whose contents are known** — the
  threshold, the NMS, the payload it packs and the output shapes it insists on. A
  field DeepStream does not have is a compile error there, not a container build failure.
* The GPU cost of the feature is `tools/engine_gpu_budget.sh <config>` before and after the module is listed.
