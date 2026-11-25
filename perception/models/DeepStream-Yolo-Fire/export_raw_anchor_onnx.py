"""Export a YOLO (v8/11/26/...) checkpoint to the raw-anchor ONNX contract
this repo's pgie_fire parser (NvDsInferParseYolo) actually expects:
per-row [x1,y1,x2,y2,conf,classid], decoded but NOT NMS'd (DeepStream's own
cluster-mode=2 + nms-iou-threshold + pre-cluster-threshold do that), with a
genuinely dynamic batch axis.

Do NOT export with Ultralytics' nms=True for this parser -- see
perception/models/DeepStream-Yolo-Fire/README.md's "The batch>1 trap" for why that silently
breaks (or, for some export combinations, breaks outright) at the
nvstreammux batch sizes this repo actually runs at (>=2 sources sharing one
pgie_fire). This script is the fire_smoke_box.onnx swap's actual recipe,
kept here so the next model swap doesn't have to rediscover it.

Usage:
    python export_raw_anchor_onnx.py <checkpoint.pt> <output.onnx> [--imgsz 640]

Verify the result at the REAL mux batch size (>=2), not just solo-frame --
a solo-batch check would not have caught either failure mode in the README.
"""
import argparse
import tempfile
from pathlib import Path

import onnx
from onnx import helper, TensorProto
from ultralytics import YOLO


def export_raw_anchor_onnx(checkpoint: str, output_path: str, imgsz: int = 640) -> None:
    model = YOLO(checkpoint)
    print("checkpoint names (verify class order against the target labels file):", model.names)

    with tempfile.TemporaryDirectory() as tmp:
        # Step 1: raw export -- dynamic batch, no NMS/argmax baked in.
        # dynamic=True also floats height/width to symbolic dims; pinned back
        # to `imgsz` by hand below, or nvinfer has no fixed resolution to
        # build a TensorRT optimization profile against.
        raw_path = model.export(format="onnx", imgsz=imgsz, simplify=True,
                                 opset=17, dynamic=True, batch=1)
        m = onnx.load(raw_path)

        inp = m.graph.input[0]
        dims = inp.type.tensor_type.shape.dim
        dims[2].Clear(); dims[2].dim_value = imgsz
        dims[3].Clear(); dims[3].dim_value = imgsz
        del m.graph.value_info[:]
        out = m.graph.output[0]
        out.type.tensor_type.shape.dim[2].Clear()
        out.type.tensor_type.shape.dim[2].dim_param = "anchors"
        m = onnx.shape_inference.infer_shapes(m)

        # Step 2: graph surgery -- decode(cxcywh->xyxy) + per-anchor argmax/max
        # over the class-score channels (already sigmoid-activated -- verified
        # against an independent nms=True export at a known frame) + transpose
        # to [batch, anchors, 6], matching fire_s.onnx's existing contract.
        graph = m.graph
        orig_out_name = graph.output[0].name  # shape [batch, 4+nc, anchors]
        nc = graph.output[0].type.tensor_type.shape.dim[1].dim_value - 4
        assert nc >= 1, f"expected >=1 class channel, got nc={nc}"

        nodes = []

        def const(name, value, dtype=TensorProto.INT64):
            nodes.append(helper.make_node(
                "Constant", [], [name],
                value=helper.make_tensor(name + "_v", dtype, [len(value)], value)))

        const("starts_box", [0]); const("ends_box", [4]); const("axes_c", [1])
        nodes.append(helper.make_node("Slice", [orig_out_name, "starts_box", "ends_box", "axes_c"], ["box_cxcywh"]))
        const("starts_cls", [4]); const("ends_cls", [4 + nc])
        nodes.append(helper.make_node("Slice", [orig_out_name, "starts_cls", "ends_cls", "axes_c"], ["cls_scores"]))

        for i, nm in enumerate(["cx", "cy", "w", "h"]):
            const(f"starts_{nm}", [i]); const(f"ends_{nm}", [i + 1])
            nodes.append(helper.make_node("Slice", ["box_cxcywh", f"starts_{nm}", f"ends_{nm}", "axes_c"], [nm]))

        nodes.append(helper.make_node("Constant", [], ["half"], value=helper.make_tensor(
            "half_v", TensorProto.FLOAT, [1], [0.5])))
        nodes.append(helper.make_node("Mul", ["w", "half"], ["half_w"]))
        nodes.append(helper.make_node("Mul", ["h", "half"], ["half_h"]))
        nodes.append(helper.make_node("Sub", ["cx", "half_w"], ["x1"]))
        nodes.append(helper.make_node("Sub", ["cy", "half_h"], ["y1"]))
        nodes.append(helper.make_node("Add", ["cx", "half_w"], ["x2"]))
        nodes.append(helper.make_node("Add", ["cy", "half_h"], ["y2"]))
        nodes.append(helper.make_node("Concat", ["x1", "y1", "x2", "y2"], ["box_xyxy"], axis=1))

        nodes.append(helper.make_node("ArgMax", ["cls_scores"], ["classid_i64"], axis=1, keepdims=1))
        nodes.append(helper.make_node("ReduceMax", ["cls_scores"], ["conf"], axes=[1], keepdims=1))
        nodes.append(helper.make_node("Cast", ["classid_i64"], ["classid_f"], to=TensorProto.FLOAT))
        nodes.append(helper.make_node("Concat", ["box_xyxy", "conf", "classid_f"], ["final_chw"], axis=1))
        nodes.append(helper.make_node("Transpose", ["final_chw"], ["output"], perm=[0, 2, 1]))

        graph.node.extend(nodes)
        new_output = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", "anchors", 6])
        del graph.output[:]
        graph.output.extend([new_output])
        del graph.value_info[:]

        onnx.checker.check_model(m)
        m = onnx.shape_inference.infer_shapes(m)
        onnx.save(m, output_path)

    out_dims = [d.dim_value if d.HasField("dim_value") else d.dim_param
                for d in onnx.load(output_path).graph.output[0].type.tensor_type.shape.dim]
    print(f"saved {output_path} -- output dims {out_dims}")
    print("REMINDER: verify at the real nvstreammux batch size (>=2), not just solo-frame.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("output_onnx")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()
    export_raw_anchor_onnx(args.checkpoint, args.output_onnx, args.imgsz)
