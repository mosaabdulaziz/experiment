"""The only place that touches DeepStream's pyds metadata for modules.

The hot path hands modules `ObjectView`s; a module that needs something the engine does not read for everyone (the
analytics region an object is in, its instance mask) reads it through these helpers, from an `extract` hook or a
pad probe. pyds is imported on first use, so a module that imports this stays importable off-GPU.
"""
from collections import namedtuple

InstanceMask = namedtuple("InstanceMask", "values width height threshold")


def analytics_roi(obj_meta) -> frozenset:
    """The nvdsanalytics ROI labels the object's reference point is inside (its NvDsAnalyticsObjInfo.roiStatus)."""
    import pyds
    labels: set = set()
    l_user = obj_meta.obj_user_meta_list
    while l_user is not None:
        user_meta = pyds.NvDsUserMeta.cast(l_user.data)
        # NVDS_OBJ_META_NVDSANALYTICS is nvdsanalytics' built-in enum meta type -- NOT a dynamically-registered
        # custom type, so it is compared directly rather than looked up via nvds_get_user_meta_type(<string>)
        # (that lookup is for custom types and never matches this, silently returning zero detections).
        # Matches NVIDIA's own reference (deepstream_python_apps/apps/deepstream-nvdsanalytics).
        if user_meta.base_meta.meta_type == pyds.NvDsMetaType.NVDS_OBJ_META_NVDSANALYTICS:
            labels.update(pyds.NvDsAnalyticsObjInfo.cast(user_meta.user_meta_data).roiStatus)
        l_user = l_user.next
    return frozenset(labels)


def infer_done(frame_meta) -> bool:
    """Whether the primary detector ran on this frame (NvDsFrameMeta.bInferDone). With nvinfer's `interval` it skips
    frames, and the objects on those are tracker predictions only."""
    return bool(frame_meta.bInferDone)


def instance_mask(obj_meta):
    """The object's instance-mask field as an InstanceMask(values: flat float32 array, width, height, threshold),
    or None when it has none or it cannot be read. A segmentation model fills it with a score grid; a parser may
    also pack other per-object floats into it (the face parser stores five landmarks there)."""
    import numpy as np
    try:
        mp = obj_meta.mask_params
        if mp is None or mp.size <= 0 or mp.width <= 0 or mp.height <= 0:
            return None
        return InstanceMask(np.asarray(mp.get_mask_array(), np.float32).reshape(-1),
                            int(mp.width), int(mp.height), float(mp.threshold))
    except Exception:  # noqa: BLE001
        return None


def batch_objects(info):
    """(frame_meta, [obj_meta]) for each frame of a probe's batch."""
    import pyds
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list if batch_meta else None
    while l_frame is not None:
        frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        objs, l_obj = [], frame_meta.obj_meta_list
        while l_obj is not None:
            objs.append(pyds.NvDsObjectMeta.cast(l_obj.data))
            l_obj = l_obj.next
        yield frame_meta, objs
        l_frame = l_frame.next
