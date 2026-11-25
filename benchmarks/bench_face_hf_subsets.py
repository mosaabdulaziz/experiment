#!/usr/bin/env python3
"""Benchmark baseline (yolov8n-face) vs commercial face detectors on HuggingFace challenge slices.

All downloaded models and dataset slices are stored inside ``--work-dir``
(default: ``scratch/face_eval_tmp``) so everything can be deleted cleanly with
a single directory removal once benchmarking is complete.

Failure-mode slices covered:
  1. WIDER FACE val (full-frame & SGIE person-crop): tiny (<24px), small (24-35px),
     heavy blur, heavy occlusion, extreme pose, crowded (>=10 faces)
  2. DarkFace 384 val: low-light / night surveillance faces
  3. Face Mask Detection: masked vs unmasked occlusion
  4. WFLW test: 5-point landmark NME, yaw error, plausibility, and AuraFace alignment drop
  5. FairFace val (CC BY 4.0): tight enrolment crops across 7 races / gender / age
  6. CelebA test: eyeglasses, hats, beards/mustaches, and blur on tight portraits
  7. COCO val (zero-person scenes): False Positives per Image (FPPI)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import tarfile
import time
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen, urlretrieve

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.bench_face_detectors import landmark_nme, match_detections  # noqa: E402
from benchmarks.face_bench_adapters import (  # noqa: E402
    ArcFaceOnnxEmbedder,
    BaseFaceDetector,
    MediaPipeFaceDetector,
    QualcommFaceDetLiteDetector,
    RfDetrFaceDetector,
    TaoPeopleNetFaceDetector,
    YoloV8FaceDetector,
    YuNetFaceDetector,
    face_yaw_from_5pt,
    fetch_commercial_face_models,
    landmarks_plausible,
    norm_crop_112,
)

logger = logging.getLogger("iris.bench_face_hf")

YOLOV8N_FACE_5PT_URL = "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.onnx"
YOLOV8N_FACE_LINDEVS_URL = "https://github.com/lindevs/yolov8-face/releases/download/1.0.1/yolov8n-face-lindevs.onnx"
TAO_PEOPLENET_ONNX_URL = (
    "https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/peoplenet/"
    "pruned_quantized_decrypted_v2.3.4/files?redirect=true&path=resnet34_peoplenet_int8.onnx"
)
OPENCV_YUNET_ONNX_URL = (
    "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx"
)
LIBREYOLO_RFDETR_FACE_URL = "https://huggingface.co/LibreYOLO/face-rfdetr-nano/resolve/main/face.onnx"
WIDER_VAL_ZIP_URL = "https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/main/data/WIDER_val.zip"
WIDER_SPLIT_ZIP_URL = "https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/main/data/wider_face_split.zip"
DARK_FACE_ANN_URL = "https://huggingface.co/datasets/hieupth/dark_face_384/resolve/main/annotations.zip"
DARK_FACE_IMG_URL = "https://huggingface.co/datasets/hieupth/dark_face_384/resolve/main/images.zip"
WFLW_TEST_URL = "https://huggingface.co/datasets/duongnguy/WFLW_augmented/resolve/main/test_data.tar.gz"
OI_VAL_BBOX_URL = "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv"
OI_HUMAN_FACE_MID = "/m/0dzct"
OI_PERSON_MID = "/m/01g317"


@dataclass
class BoxSample:
    sample_id: str
    image_path: Path
    gt_boxes: list[tuple[float, float, float, float]]
    meta: dict[str, Any]


# ── Pure parsers (unit-tested) ───────────────────────────────────────────────

def parse_wider_val_gt(text: str) -> dict[str, list[dict[str, Any]]]:
    """Parse wider_face_val_bbx_gt.txt -> {rel_path: [{bbox, blur, occlusion, pose, invalid, ...}]}."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    out: dict[str, list[dict[str, Any]]] = {}
    i = 0
    while i < len(lines):
        rel = lines[i]
        i += 1
        if i >= len(lines):
            break
        n = int(lines[i])
        i += 1
        faces: list[dict[str, Any]] = []
        # WIDER format writes 1 dummy zero row when n == 0.
        row_count = max(1, n)
        for _ in range(row_count):
            if i >= len(lines):
                break
            parts = [int(v) for v in lines[i].split()]
            i += 1
            if n == 0 or len(parts) < 10:
                continue
            x1, y1, w, h, blur, expr, illum, invalid, occ, pose = parts[:10]
            if invalid == 1 or w <= 0 or h <= 0:
                continue
            faces.append(
                {
                    "bbox": (float(x1), float(y1), float(x1 + w), float(y1 + h)),
                    "short_side": float(min(w, h)),
                    "blur": blur,
                    "expression": expr,
                    "illumination": illum,
                    "occlusion": occ,
                    "pose": pose,
                }
            )
        out[rel] = faces
    return out


def parse_wflw_csv_row(line: str) -> tuple[str, np.ndarray, list[int], tuple[float, float, float]]:
    """Parse one row of WFLW labels.csv -> (img_basename, kpts_5x2_norm, attrs_6, euler_3)."""
    cols = line.strip().split(",")
    if len(cols) < 206:
        raise ValueError(f"Expected >=206 columns in WFLW CSV, got {len(cols)}")
    pts98 = np.asarray([float(v) for v in cols[:196]], dtype=np.float32).reshape(98, 2)
    # Canonical 98-pt WFLW -> 5 ArcFace points: left eye (96), right eye (97), nose (54), mouth L (76), R (82).
    k5 = np.stack([pts98[96], pts98[97], pts98[54], pts98[76], pts98[82]], axis=0).astype(np.float32)
    if k5[0, 0] > k5[1, 0]:
        k5[[0, 1]] = k5[[1, 0]]
    if k5[3, 0] > k5[4, 0]:
        k5[[3, 4]] = k5[[4, 3]]
    attrs = [int(float(v)) for v in cols[196:202]]  # pose, expression, illumination, makeup, occlusion, blur
    euler = (float(cols[202]), float(cols[203]), float(cols[204]))
    name = Path(cols[205]).name
    return name, k5, attrs, euler


def parse_voc_xml_boxes(xml_text: str) -> list[tuple[str, tuple[float, float, float, float]]]:
    root = ET.fromstring(xml_text)
    out: list[tuple[str, tuple[float, float, float, float]]] = []
    for obj in root.findall("object"):
        label = obj.findtext("name", default="face")
        bb = obj.find("bndbox")
        if bb is None:
            continue
        x1 = float(bb.findtext("xmin", "0"))
        y1 = float(bb.findtext("ymin", "0"))
        x2 = float(bb.findtext("xmax", "0"))
        y2 = float(bb.findtext("ymax", "0"))
        if x2 > x1 and y2 > y1:
            out.append((label, (x1, y1, x2, y2)))
    return out


# ── Download / HF Streaming Helpers (all confined to work_dir) ───────────────

def _ensure_file(url: str, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_file() and dst.stat().st_size > 0:
        return dst
    logger.info("Downloading %s -> %s", url, dst)
    urlretrieve(url, str(dst))
    return dst


def _http_get_bytes(url: str, timeout: int = 25, retries: int = 3) -> bytes:
    for attempt in range(retries):
        try:
            req = Request(url, headers={"User-Agent": "iris-face-bench/1.0"})
            with urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except HTTPError as exc:
            if exc.code in (403, 429, 502, 503) and attempt + 1 < retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise
        except URLError:
            if attempt + 1 < retries:
                time.sleep(1.0)
                continue
            raise
    raise RuntimeError(f"Unreachable retry loop for {url}")


def _stream_hf_rows(dataset: str, config: str, split: str, max_rows: int, offset: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cur = offset
    while len(rows) < max_rows:
        chunk = min(100, max_rows - len(rows))
        url = (
            f"https://datasets-server.huggingface.co/rows?dataset={quote(dataset)}"
            f"&config={quote(config)}&split={quote(split)}&offset={cur}&length={chunk}"
        )
        data = json.loads(_http_get_bytes(url, timeout=30).decode("utf-8"))
        batch = [r["row"] for r in data.get("rows", [])]
        if not batch:
            break
        rows.extend(batch)
        cur += len(batch)
        if len(batch) < chunk:
            break
    return rows


def _save_hf_image(img_field: dict[str, Any], dst: Path) -> Optional[Path]:
    if dst.is_file() and dst.stat().st_size > 0:
        return dst
    src = img_field.get("src")
    if not src:
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        raw = _http_get_bytes(src, timeout=20, retries=2)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Skipping HF image %s: %s", dst.name, exc)
        return None
    dst.write_bytes(raw)
    return dst


def prepare_wider_slices(ds_dir: Path, per_slice: int = 80) -> dict[str, list[BoxSample]]:
    import cv2

    split_zip = _ensure_file(WIDER_SPLIT_ZIP_URL, ds_dir / "wider_face_split.zip")
    val_zip = _ensure_file(WIDER_VAL_ZIP_URL, ds_dir / "WIDER_val.zip")

    with zipfile.ZipFile(split_zip, "r") as zf:
        gt_name = next(n for n in zf.namelist() if n.endswith("wider_face_val_bbx_gt.txt"))
        gt_map = parse_wider_val_gt(zf.read(gt_name).decode("utf-8", "replace"))

    slices: dict[str, list[str]] = {
        "wider_tiny_lt24px": [],
        "wider_small_24_35px": [],
        "wider_heavy_blur": [],
        "wider_heavy_occlusion": [],
        "wider_extreme_pose": [],
        "wider_crowded": [],
    }
    for rel in sorted(gt_map):
        faces = gt_map[rel]
        if not faces:
            continue
        if len(slices["wider_tiny_lt24px"]) < per_slice and any(f["short_side"] < 24.0 for f in faces):
            slices["wider_tiny_lt24px"].append(rel)
        if len(slices["wider_small_24_35px"]) < per_slice and any(24.0 <= f["short_side"] < 35.0 for f in faces):
            slices["wider_small_24_35px"].append(rel)
        if len(slices["wider_heavy_blur"]) < per_slice and any(f["blur"] == 2 for f in faces):
            slices["wider_heavy_blur"].append(rel)
        if len(slices["wider_heavy_occlusion"]) < per_slice and any(f["occlusion"] == 2 for f in faces):
            slices["wider_heavy_occlusion"].append(rel)
        if len(slices["wider_extreme_pose"]) < per_slice and any(f["pose"] == 1 for f in faces):
            slices["wider_extreme_pose"].append(rel)
        if len(slices["wider_crowded"]) < per_slice and len(faces) >= 10:
            slices["wider_crowded"].append(rel)

    needed = {rel for rels in slices.values() for rel in rels}
    img_root = ds_dir / "wider_val_imgs"
    img_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(val_zip, "r") as zf:
        by_suffix = {n.split("images/")[-1]: n for n in zf.namelist() if "images/" in n and not n.endswith("/")}
        for rel in needed:
            out_p = img_root / rel.replace("/", "_")
            if not out_p.is_file() and rel in by_suffix:
                out_p.write_bytes(zf.read(by_suffix[rel]))

    out: dict[str, list[BoxSample]] = {}
    for sname, rels in slices.items():
        samples: list[BoxSample] = []
        for rel in rels:
            p = img_root / rel.replace("/", "_")
            if p.is_file():
                samples.append(BoxSample(rel, p, [f["bbox"] for f in gt_map[rel]], {"faces": gt_map[rel]}))
        out[sname] = samples

    # Also build an SGIE person-crop slice around small/tiny faces (24-35px & <24px)
    # so we test detectors under realistic DeepStream person-crop framing, not only crowd scenes.
    crop_dir = ds_dir / "wider_sgie_crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    sgie_samples: list[BoxSample] = []
    for rel in (slices["wider_small_24_35px"] + slices["wider_tiny_lt24px"])[:per_slice]:
        p = img_root / rel.replace("/", "_")
        if not p.is_file():
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        ih, iw = img.shape[:2]
        cand_faces = [f for f in gt_map[rel] if 14.0 <= f["short_side"] < 40.0]
        if not cand_faces:
            continue
        tf = cand_faces[0]
        fx1, fy1, fx2, fy2 = tf["bbox"]
        fw, fh = fx2 - fx1, fy2 - fy1
        cx1 = max(0, int(round(fx1 - 1.5 * fw)))
        cy1 = max(0, int(round(fy1 - 0.6 * fh)))
        cx2 = min(iw, int(round(fx2 + 1.5 * fw)))
        cy2 = min(ih, int(round(fy2 + 4.0 * fh)))
        if cx2 - cx1 < 40 or cy2 - cy1 < 40:
            continue
        crop_path = crop_dir / f"{len(sgie_samples):04d}_{p.name}"
        if not crop_path.is_file():
            cv2.imwrite(str(crop_path), img[cy1:cy2, cx1:cx2])
        # Keep GT boxes whose center falls inside the person-like crop.
        local_boxes: list[tuple[float, float, float, float]] = []
        for f in gt_map[rel]:
            bx1, by1, bx2, by2 = f["bbox"]
            bcx, bcy = 0.5 * (bx1 + bx2), 0.5 * (by1 + by2)
            if cx1 <= bcx <= cx2 and cy1 <= bcy <= cy2:
                local_boxes.append(
                    (max(0.0, bx1 - cx1), max(0.0, by1 - cy1), min(float(cx2 - cx1), bx2 - cx1), min(float(cy2 - cy1), by2 - cy1))
                )
        if local_boxes:
            sgie_samples.append(BoxSample(f"sgie_{rel}", crop_path, local_boxes, {}))
    out["wider_sgie_person_crop"] = sgie_samples
    return out


def prepare_dark_face_slice(ds_dir: Path, max_imgs: int = 200) -> list[BoxSample]:
    ann_zip = _ensure_file(DARK_FACE_ANN_URL, ds_dir / "dark_ann.zip")
    img_zip = _ensure_file(DARK_FACE_IMG_URL, ds_dir / "dark_imgs.zip")

    with zipfile.ZipFile(ann_zip, "r") as zf:
        coco = json.loads(zf.read("annotations/dark_face_val.json").decode("utf-8"))
    anns_by_img: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    img_by_id = {im["id"]: im for im in coco["images"]}
    for ann in coco["annotations"]:
        im = img_by_id.get(ann["image_id"])
        if not im:
            continue
        bx, by, bw, bh = ann["bbox"]
        iw, ih = float(im["width"]), float(im["height"])
        if bw > 0 and bh > 0:
            anns_by_img[im["id"]].append((bx * iw, by * ih, (bx + bw) * iw, (by + bh) * ih))

    selected = [im for im in coco["images"] if anns_by_img.get(im["id"])][:max_imgs]
    out_dir = ds_dir / "dark_face_imgs"
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(img_zip, "r") as zf:
        names_set = set(zf.namelist())
        by_base = {Path(n).name: n for n in names_set if not n.endswith("/")}
        for im in selected:
            fn = im["file_name"]
            dst = out_dir / Path(fn).name
            if dst.is_file():
                continue
            key = fn if fn in names_set else by_base.get(Path(fn).name)
            if key:
                dst.write_bytes(zf.read(key))

    samples: list[BoxSample] = []
    for im in selected:
        p = out_dir / Path(im["file_name"]).name
        if p.is_file():
            samples.append(BoxSample(p.name, p, anns_by_img[im["id"]], {}))
    return samples


def prepare_face_mask_slice(ds_dir: Path, max_imgs: int = 160) -> list[BoxSample]:
    tree = json.loads(
        urlopen("https://huggingface.co/api/datasets/hmnshudhmn24/face-mask-detection/tree/main/annotations", timeout=20)
        .read()
        .decode("utf-8")
    )
    xml_paths = sorted(x["path"] for x in tree if x["path"].endswith(".xml"))[:max_imgs]
    mask_dir = ds_dir / "face_mask"
    (mask_dir / "annotations").mkdir(parents=True, exist_ok=True)
    (mask_dir / "images").mkdir(parents=True, exist_ok=True)

    samples: list[BoxSample] = []
    for xp in xml_paths:
        stem = Path(xp).stem
        xml_dst = mask_dir / "annotations" / f"{stem}.xml"
        img_dst = mask_dir / "images" / f"{stem}.png"
        if not xml_dst.is_file():
            urlretrieve(f"https://huggingface.co/datasets/hmnshudhmn24/face-mask-detection/resolve/main/{xp}", str(xml_dst))
        if not img_dst.is_file():
            urlretrieve(
                f"https://huggingface.co/datasets/hmnshudhmn24/face-mask-detection/resolve/main/images/{stem}.png",
                str(img_dst),
            )
        parsed = parse_voc_xml_boxes(xml_dst.read_text(encoding="utf-8", errors="replace"))
        if parsed and img_dst.is_file():
            samples.append(
                BoxSample(
                    stem,
                    img_dst,
                    [b for _, b in parsed],
                    {"labels": [lbl for lbl, _ in parsed]},
                )
            )
    return samples


def prepare_lfw_slice(ds_dir: Path, per_class: int = 100) -> list[dict[str, Any]]:
    """Stream official LFW verification pairs (100 same + 100 diff) from logasja/lfw."""
    meta_path = ds_dir / "lfw_pairs_meta.json"
    img_dir = ds_dir / "lfw_pairs_imgs"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return [m for m in meta if Path(m["img0"]).is_file() and Path(m["img1"]).is_file()]

    pos_rows = _stream_hf_rows("logasja/lfw", "pairs", "test", max_rows=per_class, offset=0)
    neg_rows = _stream_hf_rows("logasja/lfw", "pairs", "test", max_rows=per_class, offset=1100)
    meta: list[dict[str, Any]] = []
    for idx, r in enumerate(pos_rows + neg_rows):
        p0 = _save_hf_image(r["img_0"], img_dir / f"pair_{idx:04d}_a.jpg")
        p1 = _save_hf_image(r["img_1"], img_dir / f"pair_{idx:04d}_b.jpg")
        if p0 is not None and p1 is not None:
            meta.append({"img0": str(p0), "img1": str(p1), "same": bool(int(r["pair"]) == 1)})
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    return meta


def prepare_fairface_slice(ds_dir: Path, max_rows: int = 300) -> list[dict[str, Any]]:
    meta_path = ds_dir / "fairface_val_meta.json"
    img_dir = ds_dir / "fairface_imgs"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        raw_rows = _stream_hf_rows("HuggingFaceM4/FairFace", "0.25", "validation", max_rows=max_rows)
        meta = []
        for idx, r in enumerate(raw_rows):
            p = _save_hf_image(r["image"], img_dir / f"ff_{idx:04d}.jpg")
            if p is not None:
                meta.append({"path": str(p), "age": r["age"], "gender": r["gender"], "race": r["race"]})
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
    return [m for m in meta if Path(m["path"]).is_file()]


def prepare_celeba_slice(ds_dir: Path, scan_rows: int = 400, per_attr: int = 50) -> dict[str, list[Path]]:
    meta_path = ds_dir / "celeba_attr_meta.json"
    img_dir = ds_dir / "celeba_imgs"
    if meta_path.is_file():
        return {k: [Path(p) for p in v if Path(p).is_file()] for k, v in json.loads(meta_path.read_text()).items()}

    raw_rows = _stream_hf_rows("flwrlabs/celeba", "img_align+identity+attr", "test", max_rows=scan_rows)
    buckets: dict[str, list[str]] = {"eyeglasses": [], "wearing_hat": [], "beard_mustache": [], "blurry": []}
    for idx, r in enumerate(raw_rows):
        checks = {
            "eyeglasses": bool(r.get("Eyeglasses")),
            "wearing_hat": bool(r.get("Wearing_Hat")),
            "beard_mustache": (not bool(r.get("No_Beard", True))) or bool(r.get("Mustache")),
            "blurry": bool(r.get("Blurry")),
        }
        want = [k for k, ok in checks.items() if ok and len(buckets[k]) < per_attr]
        if not want:
            continue
        p = _save_hf_image(r["image"], img_dir / f"celeba_{idx:04d}.jpg")
        if p is not None:
            for k in want:
                buckets[k].append(str(p))
        if all(len(v) >= per_attr for v in buckets.values()):
            break
    meta_path.write_text(json.dumps(buckets), encoding="utf-8")
    return {k: [Path(p) for p in v] for k, v in buckets.items()}


def prepare_openimages_slices(ds_dir: Path, n_neg: int = 120, n_face: int = 80) -> tuple[list[Path], list[BoxSample]]:
    """Fetch Open Images V7 validation CSV (25 MB) and pull small non-person FP + Human face slices."""
    import csv
    import cv2

    csv_path = _ensure_file(OI_VAL_BBOX_URL, ds_dir / "oi_val_bbox.csv")
    all_ids: list[str] = []
    seen_ids: set[str] = set()
    has_person_or_face: set[str] = set()
    face_boxes_norm: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            iid = row["ImageID"]
            if iid not in seen_ids:
                seen_ids.add(iid)
                all_ids.append(iid)
            lbl = row["LabelName"]
            if lbl in (OI_PERSON_MID, OI_HUMAN_FACE_MID):
                has_person_or_face.add(iid)
            if lbl == OI_HUMAN_FACE_MID and row.get("IsDepiction") != "1" and row.get("IsGroupOf") != "1":
                face_boxes_norm[iid].append(
                    (float(row["XMin"]), float(row["YMin"]), float(row["XMax"]), float(row["YMax"]))
                )

    neg_ids = [iid for iid in all_ids if iid not in has_person_or_face][:n_neg]
    face_ids = [iid for iid in all_ids if iid in face_boxes_norm][:n_face]

    neg_dir = ds_dir / "oi_neg_imgs"
    neg_dir.mkdir(parents=True, exist_ok=True)
    neg_paths: list[Path] = []
    for iid in neg_ids:
        dst = neg_dir / f"{iid}.jpg"
        if not dst.is_file():
            try:
                dst.write_bytes(_http_get_bytes(f"https://open-images-dataset.s3.amazonaws.com/validation/{iid}.jpg"))
            except Exception as exc:  # noqa: BLE001
                logger.debug("Skipping OI neg %s: %s", iid, exc)
                continue
        neg_paths.append(dst)

    face_dir = ds_dir / "oi_face_imgs"
    face_dir.mkdir(parents=True, exist_ok=True)
    face_samples: list[BoxSample] = []
    for iid in face_ids:
        dst = face_dir / f"{iid}.jpg"
        if not dst.is_file():
            try:
                dst.write_bytes(_http_get_bytes(f"https://open-images-dataset.s3.amazonaws.com/validation/{iid}.jpg"))
            except Exception as exc:  # noqa: BLE001
                logger.debug("Skipping OI face %s: %s", iid, exc)
                continue
        img = cv2.imread(str(dst))
        if img is None:
            continue
        ih, iw = img.shape[:2]
        boxes = [(x1 * iw, y1 * ih, x2 * iw, y2 * ih) for x1, y1, x2, y2 in face_boxes_norm[iid]]
        face_samples.append(BoxSample(iid, dst, boxes, {}))
    return neg_paths, face_samples


# ── Evaluation Routines ──────────────────────────────────────────────────────

def eval_box_slice(
    name: str,
    samples: Sequence[BoxSample],
    detectors: Sequence[BaseFaceDetector],
    iou_thresh: float = 0.4,
    label_filter: Optional[str] = None,
    size_range: Optional[tuple[float, float]] = None,
    attr_filter: Optional[tuple[str, int]] = None,
    min_short_side: float = 0.0,
) -> dict[str, Any]:
    import cv2

    out: dict[str, Any] = {}
    for det in detectors:
        gt_total = 0
        hits_op = 0
        hits_low = 0
        cand_total = 0
        plausible = 0
        times_ms: list[float] = []

        for s in samples:
            img = cv2.imread(str(s.image_path))
            if img is None:
                continue
            target_boxes = s.gt_boxes
            if attr_filter is not None and "faces" in s.meta:
                ak, av = attr_filter
                target_boxes = [f["bbox"] for f in s.meta["faces"] if f.get(ak) == av]
            if label_filter and "labels" in s.meta:
                target_boxes = [b for b, lbl in zip(s.gt_boxes, s.meta["labels"]) if lbl == label_filter]
            if size_range is not None:
                lo, hi = size_range
                target_boxes = [b for b in target_boxes if lo <= min(b[2] - b[0], b[3] - b[1]) < hi]
            if min_short_side > 0.0:
                target_boxes = [b for b in target_boxes if min(b[2] - b[0], b[3] - b[1]) >= min_short_side]
            if not target_boxes:
                continue

            t0 = time.perf_counter()
            preds_op = det.detect(img)
            times_ms.append((time.perf_counter() - t0) * 1000.0)
            preds_low = det.detect(img, conf_thresh=0.15)

            gt_total += len(target_boxes)
            cand_total += len(preds_op)
            plausible += sum(1 for p in preds_op if landmarks_plausible(p.landmarks, p.bbox))

            m_op, _, _ = match_detections(target_boxes, [p.bbox for p in preds_op], iou_thresh=iou_thresh)
            m_low, _, _ = match_detections(target_boxes, [p.bbox for p in preds_low], iou_thresh=iou_thresh)
            hits_op += len(m_op)
            hits_low += len(m_low)

        out[det.name] = {
            "images": len(samples),
            "gt_faces": gt_total,
            "recall_op": round(hits_op / max(1, gt_total), 4),
            "recall_0.15": round(hits_low / max(1, gt_total), 4),
            "plausible_kpt_rate": round(plausible / max(1, cand_total), 4) if cand_total else 0.0,
            "mean_ms": round(float(np.mean(times_ms)), 2) if times_ms else None,
        }
    logger.info("[%s] evaluated %d images", name, len(samples))
    return out


def eval_lfw_end_to_end(
    rows: Sequence[dict[str, Any]],
    detectors: Sequence[BaseFaceDetector],
    aurafaces: Optional[ArcFaceOnnxEmbedder],
) -> dict[str, Any]:
    """End-to-end LFW pair verification (logasja/lfw same/diff pairs) with AuraFace."""
    import cv2

    out: dict[str, Any] = {}
    for det in detectors:
        det_hits = 0
        plausible = 0
        total_imgs = 0
        same_sims: list[float] = []
        diff_sims: list[float] = []
        missed_same = 0
        missed_diff = 0
        emb_cache: dict[str, Optional[np.ndarray]] = {}

        def _detect_and_embed(path_str: str) -> Optional[np.ndarray]:
            nonlocal total_imgs, det_hits, plausible
            if path_str in emb_cache:
                return emb_cache[path_str]
            img = cv2.imread(path_str)
            if img is None:
                emb_cache[path_str] = None
                return None
            total_imgs += 1
            preds = det.detect(img)
            if not preds:
                emb_cache[path_str] = None
                return None
            det_hits += 1
            preds.sort(key=lambda f: f.width * f.height, reverse=True)
            top = preds[0]
            if landmarks_plausible(top.landmarks, top.bbox):
                plausible += 1
            vec = None if aurafaces is None else aurafaces.embed_face(img, top)
            emb_cache[path_str] = vec
            return vec

        for r in rows:
            va = _detect_and_embed(str(r["img0"]))
            vb = _detect_and_embed(str(r["img1"]))
            is_same = bool(r["same"])
            if va is None or vb is None:
                if is_same:
                    missed_same += 1
                else:
                    missed_diff += 1
                continue
            sim = float(np.dot(va, vb))
            if is_same:
                same_sims.append(sim)
            else:
                diff_sims.append(sim)

        entry: dict[str, Any] = {
            "images": total_imgs,
            "detection_rate": round(det_hits / max(1, total_imgs), 4),
            "plausible_kpt_rate": round(plausible / max(1, det_hits), 4),
        }
        if aurafaces is not None and (same_sims or diff_sims):
            g_arr = np.asarray(same_sims, dtype=np.float32)
            i_arr = np.asarray(diff_sims, dtype=np.float32)
            op_pts = []
            for thr in (0.15, 0.20, 0.25, 0.30, 0.35):
                far = float(np.mean(i_arr >= thr)) if i_arr.size else None
                frr = float(np.mean(g_arr < thr)) if g_arr.size else None
                op_pts.append({
                    "threshold": thr,
                    "far": round(far, 4) if far is not None else None,
                    "frr": round(frr, 4) if frr is not None else None,
                })
            entry["verification"] = {
                "n_same_pairs": int(g_arr.size),
                "n_diff_pairs": int(i_arr.size),
                "missed_same_pairs": missed_same,
                "missed_diff_pairs": missed_diff,
                "genuine_mean": round(float(np.mean(g_arr)), 4) if g_arr.size else None,
                "genuine_p10": round(float(np.percentile(g_arr, 10)), 4) if g_arr.size else None,
                "impostor_mean": round(float(np.mean(i_arr)), 4) if i_arr.size else None,
                "impostor_max": round(float(np.max(i_arr)), 4) if i_arr.size else None,
                "operating_points": op_pts,
            }
        out[det.name] = entry
    return out


def eval_fairface_consistency(
    rows: Sequence[dict[str, Any]],
    detectors: Sequence[BaseFaceDetector],
    baseline_det: BaseFaceDetector,
    aurafaces: Optional[ArcFaceOnnxEmbedder],
) -> dict[str, Any]:
    """Compare landmark NME, yaw error, and AuraFace embedding cosine vs baseline 5pt on FairFace."""
    import cv2

    base_cache: dict[Path, Any] = {}
    for r in rows:
        p = Path(r["path"])
        img = cv2.imread(str(p))
        if img is None:
            continue
        preds = baseline_det.detect(img)
        if not preds:
            continue
        preds.sort(key=lambda f: f.width * f.height, reverse=True)
        top = preds[0]
        if top.landmarks is not None and landmarks_plausible(top.landmarks, top.bbox):
            base_emb = (
                aurafaces.embed_aligned(norm_crop_112(img, top.landmarks)) if aurafaces is not None else None
            )
            base_cache[p] = (img, top, base_emb)

    out: dict[str, Any] = {}
    for det in detectors:
        nmes: list[float] = []
        yaw_errs: list[float] = []
        aura_sims: list[float] = []
        for p, (img, b_face, b_emb) in base_cache.items():
            preds = det.detect(img)
            if not preds:
                continue
            preds.sort(key=lambda f: f.width * f.height, reverse=True)
            top = preds[0]
            if top.landmarks is None:
                continue
            nme = landmark_nme(b_face.landmarks, top.landmarks)
            if not np.isnan(nme):
                nmes.append(nme)
            yaw_errs.append(abs(face_yaw_from_5pt(top.landmarks) - face_yaw_from_5pt(b_face.landmarks)))
            if aurafaces is not None and b_emb is not None and landmarks_plausible(top.landmarks, top.bbox):
                c_emb = aurafaces.embed_aligned(norm_crop_112(img, top.landmarks))
                aura_sims.append(float(np.dot(b_emb, c_emb)))
        out[det.name] = {
            "compared_faces": len(nmes),
            "mean_nme_vs_yolo5pt": round(float(np.mean(nmes)), 4) if nmes else None,
            "p90_nme_vs_yolo5pt": round(float(np.percentile(nmes, 90)), 4) if nmes else None,
            "mean_yaw_err": round(float(np.mean(yaw_errs)), 4) if yaw_errs else None,
            "aurafaces_cos_vs_yolo_align": round(float(np.mean(aura_sims)), 4) if aura_sims else None,
            "aurafaces_p10_vs_yolo_align": round(float(np.percentile(aura_sims, 10)), 4) if aura_sims else None,
        }
    return out


def eval_tight_crops(
    label: str,
    groups: dict[str, list[Path]],
    detectors: Sequence[BaseFaceDetector],
) -> dict[str, Any]:
    import cv2

    out: dict[str, Any] = {}
    for det in detectors:
        by_group: dict[str, Any] = {}
        all_det = 0
        all_single = 0
        all_plaus = 0
        all_tot = 0
        for gname, paths in groups.items():
            det_ok = 0
            single_ok = 0
            plaus_ok = 0
            for p in paths:
                img = cv2.imread(str(p))
                if img is None:
                    continue
                all_tot += 1
                preds = det.detect(img)
                if len(preds) == 1:
                    single_ok += 1
                    all_single += 1
                if preds:
                    det_ok += 1
                    all_det += 1
                    preds.sort(key=lambda f: f.width * f.height, reverse=True)
                    if landmarks_plausible(preds[0].landmarks, preds[0].bbox):
                        plaus_ok += 1
                        all_plaus += 1
            n = max(1, len(paths))
            by_group[gname] = {
                "n": len(paths),
                "det_rate": round(det_ok / n, 4),
                "single_face_rate": round(single_ok / n, 4),
                "plausible_kpt_rate": round(plaus_ok / max(1, det_ok), 4),
            }
        out[det.name] = {
            "total": all_tot,
            "overall_det_rate": round(all_det / max(1, all_tot), 4),
            "overall_single_face_rate": round(all_single / max(1, all_tot), 4),
            "overall_plausible_kpt_rate": round(all_plaus / max(1, all_det), 4),
            "groups": by_group,
        }
    logger.info("[%s] tight-crop eval complete", label)
    return out


def eval_coco_false_positives(paths: Sequence[Path], detectors: Sequence[BaseFaceDetector]) -> dict[str, Any]:
    import cv2

    out: dict[str, Any] = {}
    for det in detectors:
        fp_op = 0
        imgs_op = 0
        fp_18 = 0
        imgs_18 = 0
        for p in paths:
            img = cv2.imread(str(p))
            if img is None:
                continue
            preds_op = det.detect(img)
            if preds_op:
                imgs_op += 1
                fp_op += len(preds_op)
            preds_18 = det.detect(img, conf_thresh=0.18)
            if preds_18:
                imgs_18 += 1
                fp_18 += len(preds_18)
        n = max(1, len(paths))
        out[det.name] = {
            "images": len(paths),
            "total_fp": fp_op,
            "fppi": round(fp_op / n, 4),
            "pct_images_with_fp": round(imgs_op / n, 4),
            "fppi_at_0.18": round(fp_18 / n, 4),
            "pct_images_with_fp_at_0.18": round(imgs_18 / n, 4),
        }
    return out


def format_report_md(report: dict[str, Any]) -> str:
    lines = ["# Face Detector Challenge Benchmark (Baseline vs Commercial Candidates)", ""]
    det_names = report["detectors"]

    if box_slices := report.get("box_slices"):
        lines += ["## 1. Detection Recall Across Failure-Mode Slices (IoU >= 0.40)", ""]
        header = "| Slice | GT Faces | " + " | ".join(f"`{d}` (op / @0.15)" for d in det_names) + " |"
        sep = "|---|---:|" + "|".join("---:" for _ in det_names) + "|"
        lines += [header, sep]
        for sname, res in box_slices.items():
            gt = next(iter(res.values()))["gt_faces"] if res else 0
            cells = [f"{res[d]['recall_op']:.1%} / {res[d]['recall_0.15']:.1%}" for d in det_names]
            lines.append(f"| `{sname}` | {gt} | " + " | ".join(cells) + " |")
        lines.append("")

    if lfw := report.get("lfw_verification"):
        lines += ["## 2. End-to-End LFW Verification with AuraFace (`glintr100.onnx`)", ""]
        lines += [
            "| Detector | Det Rate | Plausible Kpts | Genuine Mean / p10 | Impostor Max | FAR@0.20 | FRR@0.20 | FAR@0.25 | FRR@0.25 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for d in det_names:
            m = lfw[d]
            v = m.get("verification", {})
            pts = {p["threshold"]: p for p in v.get("operating_points", [])}
            p20, p25 = pts.get(0.20, {}), pts.get(0.25, {})
            lines.append(
                f"| `{d}` | {m['detection_rate']:.1%} | {m['plausible_kpt_rate']:.1%} "
                f"| {v.get('genuine_mean')} / {v.get('genuine_p10')} | {v.get('impostor_max')} "
                f"| {p20.get('far')} | {p20.get('frr')} | {p25.get('far')} | {p25.get('frr')} |"
            )
        lines.append("")

    if ff_cons := report.get("fairface_alignment_consistency"):
        lines += ["## 3. FairFace Landmark & AuraFace Alignment Consistency (vs `baseline_yolov8n_face_5pt`)", ""]
        lines += [
            "| Detector | Compared | Mean NME | p90 NME | Mean Yaw Err | AuraFace Cos (mean / p10) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for d in det_names:
            m = ff_cons[d]
            lines.append(
                f"| `{d}` | {m['compared_faces']} | {m['mean_nme_vs_yolo5pt']} | {m['p90_nme_vs_yolo5pt']} "
                f"| {m['mean_yaw_err']} | {m['aurafaces_cos_vs_yolo_align']} / {m['aurafaces_p10_vs_yolo_align']} |"
            )
        lines.append("")

    if sec := report.get("fairface"):
        lines += ["## 4. Tight Enrolment Crops & Demographics (FairFace)", ""]
        lines += ["| Detector | Total | Det Rate | Single-Face (`==1`) | Plausible Kpts |", "|---|---:|---:|---:|---:|"]
        for d in det_names:
            m = sec[d]
            lines.append(
                f"| `{d}` | {m['total']} | {m['overall_det_rate']:.1%} | {m['overall_single_face_rate']:.1%} | {m['overall_plausible_kpt_rate']:.1%} |"
            )
        lines.append("")

    if fp := report.get("nonface_false_positives"):
        lines += ["## 5. False Positives on Zero-Person Scenes (Open Images V7)", ""]
        lines += ["| Detector | Images | FP / Image (op) | Images w/ FP (op) | FP / Image (@0.18) | Images w/ FP (@0.18) |", "|---|---:|---:|---:|---:|---:|"]
        for d in det_names:
            m = fp[d]
            lines.append(
                f"| `{d}` | {m['images']} | {m['fppi']} | {m['pct_images_with_fp']:.1%} | {m['fppi_at_0.18']} | {m['pct_images_with_fp_at_0.18']:.1%} |"
            )
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Benchmark yolov8n-face baseline vs commercial candidates on HF subsets")
    ap.add_argument("--work-dir", type=Path, default=Path("scratch/face_eval_tmp"),
                    help="Isolated folder for ALL downloaded models and datasets (safe to delete afterwards)")
    ap.add_argument("--out-dir", type=Path, default=Path("benchmarks/results/face_hf_eval"))
    ap.add_argument("--per-slice", type=int, default=60, help="Images per WIDER failure slice")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    gpu = not args.cpu
    models_dir = args.work_dir / "models"
    ds_dir = args.work_dir / "datasets"
    models_dir.mkdir(parents=True, exist_ok=True)
    ds_dir.mkdir(parents=True, exist_ok=True)

    yolo_5pt_path = _ensure_file(YOLOV8N_FACE_5PT_URL, models_dir / "akanametov_yolov8n_face.onnx")
    yolo_box_path = _ensure_file(YOLOV8N_FACE_LINDEVS_URL, models_dir / "lindevs_yolov8n_face.onnx")
    peoplenet_path = _ensure_file(TAO_PEOPLENET_ONNX_URL, models_dir / "tao_peoplenet_v2.3.4.onnx")
    yunet_path = _ensure_file(OPENCV_YUNET_ONNX_URL, models_dir / "opencv_yunet_2023mar.onnx")
    rfdetr_face_path = _ensure_file(LIBREYOLO_RFDETR_FACE_URL, models_dir / "libreyolo_face_rfdetr_nano.onnx")
    commercial = fetch_commercial_face_models(models_dir)
    mp_landmark = commercial.get("mediapipe_landmark")

    detectors: list[BaseFaceDetector] = [
        YoloV8FaceDetector(yolo_5pt_path, gpu=gpu, default_conf=0.18),
    ]
    detectors[0].name = "baseline_yolov8n_face_5pt"

    yolo_box_det = YoloV8FaceDetector(yolo_box_path, gpu=gpu, default_conf=0.18)
    yolo_box_det.name = "baseline_yolov8n_face_lindevs"
    detectors.append(yolo_box_det)

    if "face_det_lite" in commercial:
        detectors.append(QualcommFaceDetLiteDetector(commercial["face_det_lite"], gpu=gpu, default_conf=0.20))
    if peoplenet_path.is_file():
        detectors.append(TaoPeopleNetFaceDetector(peoplenet_path, mp_landmark, gpu=gpu, default_conf=0.20))
    if "mediapipe_detector" in commercial:
        detectors.append(
            MediaPipeFaceDetector(commercial["mediapipe_detector"], mp_landmark, gpu=gpu, default_conf=0.50)
        )
    if yunet_path.is_file():
        detectors.append(YuNetFaceDetector(yunet_path, gpu=gpu, default_conf=0.35))
    if rfdetr_face_path.is_file():
        detectors.append(RfDetrFaceDetector(rfdetr_face_path, mp_landmark, gpu=gpu, default_conf=0.30))

    aurafaces = (
        ArcFaceOnnxEmbedder("aurafaces_glintr100", commercial["aurafaces_glintr100"], gpu=gpu)
        if "aurafaces_glintr100" in commercial
        else None
    )

    report: dict[str, Any] = {"detectors": [d.name for d in detectors], "work_dir": str(args.work_dir)}

    wider_slices = prepare_wider_slices(ds_dir, per_slice=args.per_slice)
    box_slices: dict[str, Any] = {}
    box_slices["wider_tiny_lt24px"] = eval_box_slice("wider_tiny_lt24px", wider_slices["wider_tiny_lt24px"], detectors, size_range=(0.0, 24.0))
    box_slices["wider_small_24_35px"] = eval_box_slice("wider_small_24_35px", wider_slices["wider_small_24_35px"], detectors, size_range=(24.0, 35.0))
    box_slices["wider_sgie_crop_lt24px"] = eval_box_slice("wider_sgie_crop_lt24px", wider_slices["wider_sgie_person_crop"], detectors, size_range=(0.0, 24.0))
    box_slices["wider_sgie_crop_24_35px"] = eval_box_slice("wider_sgie_crop_24_35px", wider_slices["wider_sgie_person_crop"], detectors, size_range=(24.0, 35.0))
    box_slices["wider_heavy_blur_ge16px"] = eval_box_slice(
        "wider_heavy_blur_ge16px", wider_slices["wider_heavy_blur"], detectors, attr_filter=("blur", 2), min_short_side=16.0
    )
    box_slices["wider_heavy_occlusion_ge16px"] = eval_box_slice(
        "wider_heavy_occlusion_ge16px", wider_slices["wider_heavy_occlusion"], detectors, attr_filter=("occlusion", 2), min_short_side=16.0
    )
    box_slices["wider_extreme_pose_ge16px"] = eval_box_slice(
        "wider_extreme_pose_ge16px", wider_slices["wider_extreme_pose"], detectors, attr_filter=("pose", 1), min_short_side=16.0
    )
    box_slices["wider_crowded_ge16px"] = eval_box_slice(
        "wider_crowded_ge16px", wider_slices["wider_crowded"], detectors, min_short_side=16.0
    )

    dark_samples = prepare_dark_face_slice(ds_dir, max_imgs=min(150, args.per_slice * 2))
    box_slices["dark_face_384"] = eval_box_slice("dark_face_384", dark_samples, detectors)

    mask_samples = prepare_face_mask_slice(ds_dir, max_imgs=min(150, args.per_slice * 2))
    box_slices["face_mask_all"] = eval_box_slice("face_mask_all", mask_samples, detectors)
    box_slices["face_mask_with_mask"] = eval_box_slice("face_mask_with_mask", mask_samples, detectors, label_filter="with_mask")

    oi_neg_paths, oi_face_samples = prepare_openimages_slices(ds_dir, n_neg=120, n_face=80)
    if oi_face_samples:
        box_slices["openimages_v7_human_face"] = eval_box_slice("openimages_v7_human_face", oi_face_samples, detectors)
    report["box_slices"] = box_slices

    lfw_rows = prepare_lfw_slice(ds_dir, per_class=100)
    if lfw_rows:
        report["lfw_verification"] = eval_lfw_end_to_end(lfw_rows, detectors, aurafaces)

    fairface_rows = prepare_fairface_slice(ds_dir, max_rows=280)
    ff_groups: dict[str, list[Path]] = defaultdict(list)
    for r in fairface_rows:
        ff_groups[f"race_{r['race']}"].append(Path(r["path"]))
    report["fairface"] = eval_tight_crops("fairface", ff_groups, detectors)
    report["fairface_alignment_consistency"] = eval_fairface_consistency(
        fairface_rows, detectors, detectors[0], aurafaces
    )

    if oi_neg_paths:
        report["nonface_false_positives"] = eval_coco_false_positives(oi_neg_paths, detectors)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = format_report_md(report)
    (args.out_dir / "report.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
