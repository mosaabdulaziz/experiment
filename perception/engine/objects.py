"""Turn one frame's DeepStream object metadata into ObjectViews, in a single pass.

The probe used to walk `frame_meta.obj_meta_list` three separate times per frame (zones, alerts, detections).
Everything now reads the one list this builds, the first time anything asks for it.
"""
import pyds

from .frame import ObjectView


def extract_objects(frame_meta, gie_names=None, extractors=()) -> list:
    """Every object on the frame as an ObjectView, in metadata order.

    gie_names maps nvinfer's unique component id to the inference element's name, so a module can tell which of
    its own GIEs produced an object. Each extractor (a module's `extract` answer for this frame) is called with
    every ObjectView while its `raw` is valid, and fills `attrs`; none is called when no module asked."""
    gie_names = gie_names or {}
    out = []
    l_obj = frame_meta.obj_meta_list
    while l_obj is not None:
        obj = pyds.NvDsObjectMeta.cast(l_obj.data)
        rp = obj.rect_params
        parent_id = int(pyds.NvDsObjectMeta.cast(obj.parent).object_id) if obj.parent else None
        ov = ObjectView(
            id=int(obj.object_id), label=obj.obj_label, conf=float(obj.confidence),
            tracker_conf=float(obj.tracker_confidence),
            left=rp.left, top=rp.top, width=rp.width, height=rp.height,
            gie_name=gie_names.get(obj.unique_component_id, ""), parent_id=parent_id, raw=obj)
        for extract in extractors:
            extract(ov)
        out.append(ov)
        l_obj = l_obj.next
    return out
