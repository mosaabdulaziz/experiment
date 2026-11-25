"""pose: the keypoints of every tracked person, in capture pixels.

A secondary detector (a YOLO pose model) runs on every person's crop. This module adds it to the
pipeline and turns its raw output into one entry per person:

- the owner: which tracked person a keypoint set belongs to. The secondary GIE's parent pointer
  arrives unset on this pipeline, so the owner is decided by position (see owner.py);
- the crop rect: the rectangle the network actually saw, which is what turns the network-input
  keypoints back into the frame's pixels (the letterbox inverse);
- the `pose` meta field: {"shape": [capture_w, capture_h], "items": [{"id", "box", "kps"}]} in
  CAPTURE pixels -- the space the fall product's service works in (FALL_DETECTION_DESIGN.md §3.2).

`id` is the DeepStream tracker id, the join key between the module, the service and the overlay:
it is the id the engine draws on the person's box. One entry per PERSON, not per detection: the
network can report the same person twice, and the monitor would then see one track twice in one
frame. People seen through fewer than `min_keypoints` joints are omitted rather than published
with zeroes.

A frame the network did not run on (the SGIE's re-inference interval) carries NO `pose` field:
the frame says nothing about the people, which is not the same as saying there are none. Only a
frame with people AND no pose items is such a frame; an empty scene publishes an empty field, so
the consumer can hear that the people are gone.

The coordinates are the whole risk of this module: the keypoints come from the network's input
pixels and the box from the tracker, and the two must end up in ONE space. Everything here is
capture pixels, and `shape` travels with the data so the consumer never has to guess.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from perception.engine.frame import FrameOutput, FrameView, ObjectView
from perception.engine.engine_path import gie_engine_path
from perception.engine.model_manifest import ModelRef
from perception.engine.modules import GieSpec, Module, ModuleConfigError, ModuleStartupError
from perception.engine.nvds import instance_mask
from perception.lib.overlay_boxes import box

from .owner import crop_for, owner_by_containment

logger = logging.getLogger("iris.sidecar")

POSE_LABEL = "pose"                 # the class name in the model's labels.txt
ELEMENT = "sgie_pose"               # the pipeline element's name
KEYPOINT_COUNT = 17                 # COCO's 17 joints, the order the consumer expects
# A keypoint set explains the item's box when most of its points land inside it: the network saw
# one person and put its joints there. Below this the crop rect is not identified, and the OWNER's
# rect is used instead (see owner.py).
MIN_OWNER_HITS = 9
# The config file's key that holds the secondary GIE's re-inference interval, in frames. nvinfer
# reads it when it creates the element; it is not an element property (see config_reinfer_interval).
REINFER_KEY = "secondary-reinfer-interval"
# The config file's key that names the model: the ONNX nvinfer loads. It is relative to the
# config file's own directory.
ONNX_KEY = "onnx-file"
# The config file's section that holds the per-class thresholds, and the key in it that holds the
# DETECTION floor: the confidence a detection must reach to be a detection at all. That threshold
# is the parser's (the parser reads it from the config file, not from this module), but the module
# declares the same number, so the two cannot drift apart -- see on_built.
CLASS_ATTRS_SECTION = "class-attrs-all"
DETECTION_KEY = "pre-cluster-threshold"
# The config file's build parameters, which the engine's DERIVED name depends on: nvinfer
# serializes the engine as <onnx>_b<batch>_gpu<id>_<mode>.engine, and the batch of a SECONDARY
# GIE is the one in its own config file (the element's batch-size property is not set for it).
BATCH_KEY = "batch-size"
GPU_KEY = "gpu-id"
# The config file's keys that say how nvinfer scales the crop into the network's input. The
# module's letterbox inverse is the inverse of EXACTLY this scaling, so these are declared numbers
# like the interval and the thresholds -- and they are gated (see on_built).
ASPECT_KEY = "maintain-aspect-ratio"
PADDING_KEY = "symmetric-padding"
# The scratch keys this module writes: the items it published, and whether the frame carries a
# pose observation at all (see transform).
SCRATCH_ITEMS = "pose"
SCRATCH_OBSERVED = "pose:observed"


def pose_raw(mask):
    """The 17 keypoints of one detection as the parser wrote them, plus the network's input size.

    nvdsparsepose_Yolo.cpp packs them into the object's instance-mask field (an nvds.InstanceMask)
    the way the face parser packs its five landmarks there: 17 * 3 floats (x, y, conf) in the
    network's INPUT pixels, and the mask struct carries the network's input size. None when the
    mask cannot carry them (a malformed mask omits the person; it never crashes the pipeline)."""
    if mask is None:
        return None
    values = mask.values
    if values is None or len(values) < KEYPOINT_COUNT * 3:
        return None
    flat = [float(v) for v in values[:KEYPOINT_COUNT * 3]]
    return ([flat[i * 3:(i + 1) * 3] for i in range(KEYPOINT_COUNT)],
            float(mask.width), float(mask.height))


def keypoints_to_capture(raw, box, sx, sy):
    """Map the network-input keypoints to CAPTURE pixels, or None.

    The pose network ran on the person's box, scaled to fit net_w x net_h keeping its aspect ratio
    and centred with equal padding on both sides (config_sgie_pose.txt: maintain-aspect-ratio=1,
    symmetric-padding=1): undo the padding and the scale, offset by the crop rect (mux coords),
    then scale mux -> capture. The generalisation of face_detect's landmarks_to_capture to 17
    points; `box` is the crop rect, which is NOT necessarily the owner's box."""
    if raw is None or box is None:
        return None
    pts, net_w, net_h = raw
    left, top, width, height = box
    if width <= 0 or height <= 0 or net_w <= 0 or net_h <= 0:
        return None
    scale = min(net_w / width, net_h / height)
    pad_x = (net_w - width * scale) / 2.0
    pad_y = (net_h - height * scale) / 2.0
    return [[(left + (p[0] - pad_x) / scale) * sx,
             (top + (p[1] - pad_y) / scale) * sy] for p in pts]


def keypoint_confidences(raw):
    """The 17 keypoints' own confidences, or None."""
    if raw is None:
        return None
    return [float(p[2]) for p in raw[0]]


def config_sections(path: str) -> dict:
    """The config file's sections, as a dict of section name -> {key: value}.

    nvinfer's config files are INI-like: a `[property]` section, then `[class-attrs-*]`
    sections, `key=value` lines and `#` comments. A file that cannot be read yields no sections,
    and the caller decides what that means."""
    sections: dict = {}
    section = ""
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue
                if line.startswith("[") and line.endswith("]"):
                    section = line[1:-1].strip().lower()
                    sections.setdefault(section, {})
                    continue
                if not section or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                sections[section][key.strip().lower()] = value.strip()
    except OSError:
        return {}
    return sections


def config_properties(path: str) -> dict:
    """The `[property]` section of an nvinfer config file, as a dict.

    nvinfer's config files are INI-like: a `[property]` section, then `[class-attrs-*]`
    sections, `key=value` lines and `#` comments. Only the `[property]` section holds the
    inference settings; the class-attrs sections hold per-class thresholds. A file that cannot
    be read yields no properties, and the caller decides what that means."""
    return config_sections(path).get("property", {})


def config_detection_threshold(path: str) -> float | None:
    """The DETECTION floor the config file declares, or None when it declares none.

    This is the confidence a detection must reach to be a detection at all -- the parser's
    threshold, which nvinfer reads from the config file's class-attrs section. It is a different
    number from the module's JOINT floor (see Settings.keypoint_confidence_threshold), and
    confusing the two would silently change what the module publishes."""
    raw = config_sections(path).get(CLASS_ATTRS_SECTION, {}).get(DETECTION_KEY)
    if raw is None:
        return None
    try:
        return float(str(raw).strip())
    except ValueError:
        return None


def config_reinfer_interval(path: str) -> int:
    """The re-inference interval the config file declares, in frames (0: every frame).

    nvinfer's element property `interval` gates the PRIMARY GIE; a secondary GIE's interval is
    the config file's `secondary-reinfer-interval`, which nvinfer reads when it creates the
    element and which has no element property. The config file is therefore the only place the
    value exists, and the only place it can be verified."""
    raw = config_properties(path).get(REINFER_KEY, "0")
    try:
        return int(str(raw).strip())
    except ValueError:
        return 0


def config_model(path: str) -> str:
    """The model the config file loads: the ONNX file's stem (yolo26x-pose.onnx -> yolo26x-pose).

    The config file is what nvinfer reads, so it is what names the model. The module declares the
    same name in its `model` setting, and the two must agree (see on_built)."""
    onnx = config_properties(path).get(ONNX_KEY, "")
    return Path(onnx).stem if onnx else ""


def config_onnx_path(path: str) -> Path | None:
    """The ONNX the config file names, resolved the way nvinfer resolves it.

    nvinfer reads the config file's paths relative to the config file's own directory -- the
    labels file it names and the engine it writes are there too -- so a relative `onnx-file`
    means the model in that directory. None when the config file names no model at all."""
    onnx = config_properties(path).get(ONNX_KEY, "")
    if not onnx:
        return None
    model = Path(onnx)
    if model.is_absolute():
        return model
    return Path(os.path.dirname(os.path.abspath(path))) / model


def config_int(path: str, key: str, default: int) -> int:
    """One integer from the config file's `[property]` section, or the default.

    The default is nvinfer's own: a secondary GIE's batch is the one in its config file (the
    element's batch-size property is not set for it), and the GPU is 0 unless the file says
    otherwise."""
    raw = config_properties(path).get(key, "")
    try:
        return int(str(raw).strip())
    except ValueError:
        return default


def config_flag(path: str, key: str) -> int | None:
    """One integer from the config file's `[property]` section, or None when it declares none.

    None is not the same as 0: a key the file does not declare is a missing DECLARATION, and a
    declaration is what the module's gates compare against. (The config file's own defaults are
    nvinfer's business, not a declaration this module may rely on.)"""
    raw = config_properties(path).get(key)
    if raw is None:
        return None
    try:
        return int(str(raw).strip())
    except ValueError:
        return None


def config_batch_size(path: str) -> int:
    """The batch the secondary GIE's engine is built at: the config file's own `batch-size`."""
    return config_int(path, BATCH_KEY, 1)


def config_gpu_id(path: str) -> int:
    """The GPU the GIE runs on: the config file's own `gpu-id`."""
    return config_int(path, GPU_KEY, 0)


def _is_pose_item(o: ObjectView) -> bool:
    """Whether this object is one of the pose network's items.

    In the pipeline the item carries the GIE's name; its class name comes from the model's
    labels.txt, which names the single class `pose`. Either identifies it, so the module does not
    depend on the other."""
    return o.gie_name == ELEMENT or o.label == POSE_LABEL


@dataclass
class Settings:
    config: str = "/workspace/iris/perception/models/DeepStream-Yolo-Pose/config_sgie_pose.txt"
    # Where nvinfer will serialize this GIE's engine, and where it looks for it on the next boot.
    # Empty: derive it from the config file (the repo's own derivation,
    # perception/engine/engine_path.py), which is the only place the name is decided -- the
    # config file's onnx-file, batch-size, gpu-id and network-mode.
    engine: str = ""
    # The model this module runs, by name (the checkpoint's stem: yolo26x-pose). The
    # config file says which ONNX it loads; the two must agree, or the module would run a model
    # nobody declared. Switching models is a documented procedure:
    # perception/models/DeepStream-Yolo-Pose/MODEL_SWITCH.md.
    model: str = "yolo26x-pose"
    # The detection floor: the confidence a detection must reach to be a detection at all. It is
    # the config file's `pre-cluster-threshold` (the parser reads it from there), and the two must
    # agree -- see on_built.
    detection_threshold: float = 0.25
    # The JOINT floor: how confident a joint must be to count as seen. This is a different number
    # from the detection floor above -- one is about the detection, the other about the joints --
    # and the name says which is which, because the two used to be called the same thing.
    keypoint_confidence_threshold: float = 0.30
    # How many of the 17 joints a person must be seen through to be published. One confident joint
    # is not an observation: the vendor's kinematics read the two legs (three joints each) and the
    # torso, and below this the monitor would be measuring a body the network mostly did not see.
    min_keypoints: int = 6
    # The SGIE's own interval: 2 = 12.5 Hz on a 25 Hz pipeline. The GPU lever of the feature
    # (FALL_DETECTION_DESIGN.md §3.5); 1 is the experiment the corpus replay measures.
    interval: int = 2
    # Opt-in context boxes remain in mux pixels, independent of pose observations.
    publish_detection_boxes: bool = False

    def __post_init__(self) -> None:
        if int(self.interval) < 1:
            raise ModuleConfigError(f"modules.pose: interval must be at least 1 (the SGIE's own interval), "
                                    f"got {self.interval!r}")
        if not 0.0 < float(self.detection_threshold) <= 1.0:
            raise ModuleConfigError(f"modules.pose: detection_threshold must be in (0, 1], "
                                    f"got {self.detection_threshold!r}")
        if not 0.0 < float(self.keypoint_confidence_threshold) <= 1.0:
            raise ModuleConfigError(f"modules.pose: keypoint_confidence_threshold must be in (0, 1], "
                                    f"got {self.keypoint_confidence_threshold!r}")
        if not 1 <= int(self.min_keypoints) <= KEYPOINT_COUNT:
            raise ModuleConfigError(f"modules.pose: min_keypoints must be between 1 and {KEYPOINT_COUNT}, "
                                    f"got {self.min_keypoints!r}")


class Pose(Module):
    name = "pose"
    Settings = Settings

    @classmethod
    def model_files(cls, settings, every_variant: bool = False) -> list:
        return [ModelRef(settings.config, "config", "nvinfer")]

    def __init__(self, settings: Settings, services) -> None:
        super().__init__(settings, services)
        self.engine = settings.engine

    # build ------------------------------------------------------------------------------------------------
    def gies(self) -> list:
        s = self.settings
        if not self.engine:
            self.engine = self._derive_engine()
        # No `interval` on the spec: nvinfer's element property gates the PRIMARY GIE, and a secondary
        # one's re-inference interval is the config file's `secondary-reinfer-interval` -- the shipped
        # sgie_plate is configured exactly this way. The number lives in the yaml (the declaration)
        # and in the config file (the mechanism); `on_built` compares them and refuses to start when
        # they disagree.
        return [GieSpec(name=ELEMENT, config_path=s.config, role="secondary",
                        engine_path=self.engine, batch_from_mux=False)]

    def _derive_engine(self) -> str:
        """Where nvinfer will serialize this GIE's engine.

        The repo derives that name in one place (perception/engine/engine_path.py), and the fire
        module uses it the same way: the config file names the ONNX and the build parameters
        (batch, GPU, precision), and the derived name follows from them. This GIE is a SECONDARY
        one, so its engine is built at the batch in its OWN config file -- the element's
        batch-size property is not set for it -- so the batch comes from the file, not from the
        pipeline's."""
        s = self.settings
        try:
            return gie_engine_path(s.config, config_batch_size(s.config), config_gpu_id(s.config))
        except ValueError as exc:
            raise ModuleStartupError(f"modules.pose: {exc}") from exc

    def on_built(self, pipeline) -> None:
        """The pipeline exists: check it really runs the declared model at the declared interval.

        A secondary GIE's interval lives in its config file, not in an element property, so the
        setting and the file are two statements of the same thing. When they disagree the GIE
        would run at an interval nobody asked for -- the feature's first GPU lever, silently at
        the wrong setting -- so the module refuses to start rather than measure the wrong thing
        (design §8: a gate, not a warning).

        The same holds for the model: the config file says which ONNX nvinfer loads, and the
        setting says which model the module means to run. A config file left over from another
        model is the same failure -- and worse, its engine is the OTHER model's, so the engine
        would be rebuilt from the ONNX on every boot and the cache would never fill.

        And for the detection floor: the config file's `pre-cluster-threshold` is the parser's
        threshold, and the setting says which floor the module means to run at. They disagree when
        the file was edited without the module, and then the module would publish a different set
        of people than the one that was declared."""
        s = self.settings
        declared = config_reinfer_interval(s.config)
        if declared != int(s.interval):
            raise ModuleStartupError(
                f"modules.pose: interval={s.interval} but {s.config} declares "
                f"{REINFER_KEY}={declared}. nvinfer reads a secondary GIE's re-inference interval "
                f"from its config file (it has no element property), so set "
                f"{REINFER_KEY}={s.interval} in that file, or set modules.pose.interval={declared}.")
        logger.info("%s re-infers every %d frame(s) (%s=%d in %s)", ELEMENT, declared, REINFER_KEY,
                    declared, s.config)

        floor = config_detection_threshold(s.config)
        if floor is None:
            raise ModuleStartupError(
                f"modules.pose: {s.config} declares no {DETECTION_KEY}, so the parser would keep "
                f"every detection. That key is the parser's detection floor (nvinfer reads it from "
                f"the config file), so add {DETECTION_KEY}={s.detection_threshold} to that file.")
        if abs(floor - float(s.detection_threshold)) > 1e-6:
            raise ModuleStartupError(
                f"modules.pose: detection_threshold={s.detection_threshold} but {s.config} declares "
                f"{DETECTION_KEY}={floor}. That key is the parser's detection floor (nvinfer reads "
                f"it from the config file), so set {DETECTION_KEY}={s.detection_threshold} in that "
                f"file, or set modules.pose.detection_threshold={floor}.")
        logger.info("%s keeps a detection at %s=%s", ELEMENT, DETECTION_KEY, floor)

        onnx = config_properties(s.config).get(ONNX_KEY, "")
        if not onnx:
            raise ModuleStartupError(
                f"modules.pose: {s.config} declares no {ONNX_KEY}, so nvinfer would have no model "
                f"to load. The config file names the model; the switch procedure is in "
                f"perception/models/DeepStream-Yolo-Pose/MODEL_SWITCH.md.")
        model = config_model(s.config)
        if model != s.model:
            raise ModuleStartupError(
                f"modules.pose: model={s.model} but {s.config} loads {ONNX_KEY}={onnx}. nvinfer "
                f"loads the ONNX the config file names, so set modules.pose.model={model}, or "
                f"point the config file at {s.model}.onnx. The switch procedure is in "
                f"perception/models/DeepStream-Yolo-Pose/MODEL_SWITCH.md.")
        onnx_path = config_onnx_path(s.config)
        if onnx_path is None or not onnx_path.is_file():
            raise ModuleStartupError(
                f"modules.pose: {s.config} names {ONNX_KEY}={onnx}, but there is no such file "
                f"({onnx_path}). nvinfer would have no model to build the engine from, so the "
                f"module refuses to start. The pose models are versioned with DVC: run "
                f"`just data-pull models-fall` (perception/models/DeepStream-Yolo-Pose/README.md).")

        # The letterbox the module inverts is the one the config file declares. nvinfer scales the
        # crop into the network's input; the module undoes that scaling to put the joints back in
        # the frame. Another scaling would put every joint in the wrong place -- silently, in every
        # frame -- so these two keys are declared numbers like the interval and the thresholds.
        for key in (ASPECT_KEY, PADDING_KEY):
            value = config_flag(s.config, key)
            if value is None:
                raise ModuleStartupError(
                    f"modules.pose: {s.config} declares no {key}, so the module cannot know how "
                    f"nvinfer scaled the crop into the network's input. The module's letterbox "
                    f"inverse assumes {ASPECT_KEY}=1 (the crop keeps its aspect ratio) and "
                    f"{PADDING_KEY}=1 (the padding is centred), so add them to that file.")
            if value != 1:
                raise ModuleStartupError(
                    f"modules.pose: {s.config} declares {key}={value}, but the module's letterbox "
                    f"inverse assumes {ASPECT_KEY}=1 (the crop keeps its aspect ratio) and "
                    f"{PADDING_KEY}=1 (the padding is centred on both sides). Another scaling "
                    f"would put every joint in the wrong place, so the module refuses to start.")
        logger.info("%s runs %s (engine %s)", ELEMENT, model,
                    os.path.basename(self.engine) if self.engine else "(derived at build)")

    # hot path ---------------------------------------------------------------------------------------------
    def extract(self, frame: FrameView):
        """The pose network's raw keypoints of each pose item, as attrs["keypoints_raw"]."""
        def read(o: ObjectView) -> None:
            if _is_pose_item(o):
                o.attrs["keypoints_raw"] = pose_raw(instance_mask(o.raw))
        return read

    def transform(self, frame: FrameView) -> None:
        """The crop rect of every pose item, its owner, and the keypoints in capture pixels."""
        s = self.settings
        if s.publish_detection_boxes:
            frame.dets_with_boxes = True
        sx, sy = frame.capture_scale()
        people = [(o.left, o.top, o.width, o.height, o.id) for o in frame.objects
                  if o.kind == "object" and o.label == "person"]
        # the people's boxes for a fall's clip: the fall service names the falling one's track (app/overlay_track.py)
        self.overlay_boxes.write(frame.stream_id, frame.pts, [box(p[:4], "alert", p[4]) for p in people],
                                 at=frame.arrived)
        claims = []
        for o in frame.objects:
            if not _is_pose_item(o):
                continue
            # the pose items are the network's own: the person's box is what the operator sees, so
            # the item is neither drawn nor published (the face module does the same with its items)
            o.draw = o.publish = False
            claims.append(o)
        # One entry per PERSON: the network can report the same person twice, and the monitor would
        # then see one track twice in one frame. The detection whose joints land inside its own box
        # is the one that explains it, so it wins; the confidence breaks a tie.
        best: dict = {}
        for o in claims:
            raw = o.attrs.get("keypoints_raw")
            if raw is None:
                continue
            confs = keypoint_confidences(raw)
            confident = sum(1 for c in confs if c >= s.keypoint_confidence_threshold)
            if confident < s.min_keypoints:
                continue            # too few joints seen: the person is not observed
            crop, owner, hits = self._crop_and_owner(o, raw, people)
            if owner is None:
                continue
            pts = keypoints_to_capture(raw, crop, sx, sy)
            if pts is None:
                continue
            left, top, width, height, oid = owner
            item = {"id": int(oid),
                    "box": [left * sx, top * sy, (left + width) * sx, (top + height) * sy],
                    "kps": [[p[0], p[1], c] for p, c in zip(pts, confs)]}
            rank = (hits, sum(confs))
            if oid not in best or rank > best[oid][0]:
                best[oid] = (rank, item)
        frame.scratch[SCRATCH_ITEMS] = [entry[1] for entry in best.values()]
        # Whether this frame carries a pose observation at all. The network ran when at least one
        # item survives the gates above; `claims` alone is not enough -- a frame can emit pose
        # items and still publish none (too few confident joints, no owner, an unreadable mask),
        # and that frame says NOTHING about the people, exactly like a skipped one. Publishing an
        # empty list there would make the consumer report an empty scene and drop every track.
        # No people at all is an observation too: the monitor has to learn that they are gone.
        frame.scratch[SCRATCH_OBSERVED] = bool(best) or not people

    def _crop_and_owner(self, item: ObjectView, raw, people: list):
        """The rectangle the network saw, the tracked person the item belongs to, and how well it fits.

        The crop rect is found by geometry: the person's rect whose letterbox inverse puts the
        keypoints inside the item's own box is the one the network saw. When no rect explains the
        item, the OWNER's rect is the fallback -- the network ran on that person's crop, so the
        person's rect is the better guess at it, and the item's own box would place the joints
        wrong (the face module falls back to the owner's rect for the same reason)."""
        item_box = (item.left, item.top, item.width, item.height)
        owner, hits = crop_for(item_box, people,
                               lambda rect: keypoints_to_capture(raw, rect, 1.0, 1.0), MIN_OWNER_HITS)
        if owner is not None:
            return owner[:4], owner, hits
        owner = owner_by_containment(item_box, people)
        if owner is not None:
            return owner[:4], owner, 0
        return item_box, None, 0

    def analyze(self, frame: FrameView, out: FrameOutput) -> None:
        """The `pose` field: the capture size and one entry per observed person."""
        if self.settings.publish_detection_boxes:
            out.fields["dets_shape"] = list(frame.mux_size)
        if not frame.scratch.get(SCRATCH_OBSERVED):
            return              # the network did not run on this frame: it says nothing
        items = frame.scratch.get(SCRATCH_ITEMS) or []
        capture_w, capture_h = frame.capture_size
        out.fields["pose"] = {"shape": [capture_w, capture_h], "items": items}


MODULE = Pose
