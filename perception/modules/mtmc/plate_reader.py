"""Reads the text of a licence-plate crop (PP-OCRv5 Arabic mobile, fine-tuned on Egyptian plates).

The model (products/el-mohandes/models/v5_arabic_mobile-fine-tuned) was exported to ONNX with
mtmc/export_plate_ocr_onnx.py. It takes a 48 px high strip and answers per column with a
probability over [blank] + the 27 plate characters (digits and Arabic letters); the text is the
greedy CTC read in the order the characters stand on the plate, left to right.

`PlateBook` keeps, per global id, the best plate crop of EACH camera with what was read from it,
and writes one picture per vehicle (crop | camera | text | confidence per camera) so the cameras'
readings can be compared side by side.
"""
from __future__ import annotations

import json
import logging
import math
import os
import queue
import threading
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_H, _W = 48, 320
_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf",
)


_LUMA = np.array([0.114, 0.587, 0.299], np.float32)       # B, G, R -> gray
_BINOMIAL = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32) / 16.0


def _char_metrics(crop_bgr: np.ndarray) -> tuple[float, float]:
    """(sharpness, edge strength) of the region where the characters are (8-92% across, 34-92%
    down): variance of the Laplacian and mean horizontal Sobel response, on 0-255 grey."""
    h, w = crop_bgr.shape[:2]
    x1, x2 = min(round(.08 * w), w - 1), min(max(round(.92 * w), 1), w)
    y1, y2 = min(round(.34 * h), h - 1), min(max(round(.92 * h), 1), h)
    region = (crop_bgr[y1:y2, x1:x2].astype(np.float32) @ _LUMA)
    if region.shape[0] < 3 or region.shape[1] < 3:
        return 0.0, 0.0
    lap = cv2.filter2D(region, -1, np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32),
                       borderType=cv2.BORDER_REPLICATE)
    sobel = cv2.filter2D(region, -1, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32) / 8.0,
                         borderType=cv2.BORDER_REPLICATE)
    return float(lap.var()), float(np.abs(sobel).mean())


def enhance(crop_bgr: np.ndarray, upscale: int = 3) -> np.ndarray:
    """The enhancement the OCR was fine-tuned on (products/el-mohandes/models/v5_arabic_mobile-fine-tuned/
    plate_enhancement.py, here on the CPU): a 3x bicubic enlargement, and a gentle local-contrast
    and sharpening pass ONLY when the character region is already sharp (sharpness >= 55 and edge
    strength >= 11); a blurry crop is enlarged and left alone."""
    sharp, edges = _char_metrics(crop_bgr)
    im = crop_bgr
    if upscale > 1:
        im = cv2.resize(im, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    if sharp >= 55 and edges >= 11:
        f = im.astype(np.float32) / 255.0
        gray = f @ _LUMA
        local = cv2.blur(gray, (9, 9), borderType=cv2.BORDER_REPLICATE)
        f = np.clip(f + 0.07 * (gray - local)[:, :, None], 0, 1)
        blur = cv2.filter2D(f, -1, _BINOMIAL, borderType=cv2.BORDER_REPLICATE)
        f = np.clip(f + 0.12 * (f - blur), 0, 1)
        im = np.round(f * 255.0).astype(np.uint8)
    return im


class PlateReader:
    def __init__(self, model_path: str, dict_path: str, enhance_crops: bool = True):
        self._enhance = enhance_crops
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1      # a tiny model; do not fight the pipeline for cores
        self._sess = ort.InferenceSession(model_path, sess_options=opts,
                                          providers=["CPUExecutionProvider"])
        with open(dict_path, encoding="utf-8") as fh:
            self._chars = ["blank"] + [ln.strip() for ln in fh.read().split("\n") if ln.strip()]

    def read(self, crop_bgr: np.ndarray) -> tuple[str, float]:
        """(text, confidence) of one plate crop; confidence is the mean probability of the kept
        characters (0 when nothing was read)."""
        if crop_bgr is None or crop_bgr.size == 0:
            return "", 0.0
        if self._enhance:
            crop_bgr = enhance(crop_bgr)
        h, w = crop_bgr.shape[:2]
        nw = max(1, min(math.ceil(w * _H / h), _W))
        # The model takes the BGR crop, scaled to 48 px high, normalised, on a 320 px wide canvas
        # whose unused part is ZERO AFTER NORMALISING (mid-grey), as in PaddleOCR's RecResizeImg.
        # Padding with black pixels instead makes it read only the digits and none of the letters.
        x = np.zeros((3, _H, _W), np.float32)
        strip = cv2.resize(crop_bgr, (nw, _H), interpolation=cv2.INTER_LINEAR)
        x[:, :, :nw] = ((strip.astype(np.float32) / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)
        x = x[None]
        probs = self._sess.run(None, {"x": x})[0][0]      # (columns, 28), already softmaxed
        return self.decode(probs)

    def decode(self, probs: np.ndarray) -> tuple[str, float]:
        idx = probs.argmax(axis=1)
        out, conf, prev = [], [], -1
        for t, i in enumerate(idx):
            if i != prev and i != 0 and i < len(self._chars):
                out.append(self._chars[i])
                conf.append(float(probs[t, i]))
            prev = i
        return "".join(out), (float(np.mean(conf)) if conf else 0.0)


class PlateBook:
    """Reads plates on a background thread and writes <dir>/<gid>.png and <dir>/<gid>.json.

    submit() never blocks the pipeline; a full queue drops the oldest-style way (the new request
    is skipped, a better plate will come again).
    """

    def __init__(self, reader: PlateReader, out_dir: str, on_read=None, maxsize: int = 64,
                 min_size: tuple = (0, 0), save_crops: bool = False):
        self._reader, self._dir, self._on_read = reader, out_dir, on_read
        # With save_crops every crop handed to the reader is also kept as it was (lossless, before the reader's
        # enhancement) in <dir>/crops, with its reading in crops/index.jsonl: the material for offline experiments
        # (scripts/plate_enhance_lab.py). Unlike the per-vehicle files these are never cleared.
        self._save_crops = save_crops
        self._started = int(time.time())       # which start of the sidecar a crop belongs to (global ids restart every run)
        self._min_size = min_size          # (width, height) in px: a smaller crop is never read (see plate_size_ok)
        self.too_small = 0
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        self._rows: dict[int, dict[str, dict]] = {}      # gid -> cam -> row
        self._reads: dict[int, list] = {}                # gid -> every reading so far (for the vote)
        self._font = None
        self._clear_old()
        threading.Thread(target=self._run, name="plate-book", daemon=True).start()

    def _clear_old(self) -> None:
        """Global ids start again at 0 with every run: what an earlier run left here is another
        vehicle's, and would be taken for this one's."""
        import re
        if not os.path.isdir(self._dir) or os.environ.get("SCALE_KEEP_RUN_FILES") == "1":
            return        # (full-day run: the sidecar restarts under the healer, ids continue from Redis, keep the day's plates)
        for name in os.listdir(self._dir):
            if re.fullmatch(r"\d+(_[\w-]+)?\.(json|png|jpg)", name):
                try:
                    os.remove(os.path.join(self._dir, name))
                except OSError:
                    pass

    def submit(self, gid: int, cam: str, crop: np.ndarray, conf: float, local_id: int,
               at: float | None = None) -> bool:
        h, w = crop.shape[:2]
        if w < self._min_size[0] or h < self._min_size[1]:
            self.too_small += 1
            return False
        try:
            self._q.put_nowait((gid, cam, crop, conf, local_id, at or time.time()))
            return True
        except queue.Full:
            return False

    def _run(self) -> None:
        while True:
            gid, cam, crop, conf, local_id, at = self._q.get()
            try:
                self._restore(gid)
                text, tconf = self._reader.read(crop)
                if self._save_crops:
                    self._save_crop(gid, cam, local_id, at, crop, text, tconf, conf)
                row = {"cam": cam, "local_id": local_id, "text": text, "ocr_conf": round(tconf, 3),
                       "det_conf": round(conf, 3), "at": at, "crop": crop}
                # a camera's row (the crop and text shown for it) is its surest reading, not its latest: a later,
                # shakier read of the same plate must not replace a better one
                kept = self._rows.setdefault(gid, {}).get(cam)
                if kept is None or row["ocr_conf"] >= kept["ocr_conf"]:
                    self._rows[gid][cam] = row
                hist = self._reads.setdefault(gid, [])
                hist.append({"cam": cam, "text": text, "ocr_conf": row["ocr_conf"]})
                del hist[:-60]
                self._write(gid)
                logger.info("[plate] global %d %s #%d read %r (ocr %.2f, det %.2f)",
                            gid, cam, local_id, text, tconf, conf)
                if self._on_read is not None:
                    self._on_read(gid, row)
            except Exception as exc:                      # the pipeline must never feel this
                logger.warning("plate read failed: %s", exc)

    def _restore(self, gid: int) -> None:
        """First read of a vehicle after the sidecar was restarted with its files kept (SCALE_KEEP_RUN_FILES=1): load
        what was read before, so the new reading adds to it instead of replacing it when the files are rewritten."""
        if os.environ.get("SCALE_KEEP_RUN_FILES") != "1" or gid in self._reads or gid in self._rows:
            return
        try:
            with open(os.path.join(self._dir, f"{gid}_reads.json"), encoding="utf-8") as fh:
                self._reads[gid] = json.load(fh)
            with open(os.path.join(self._dir, f"{gid}.json"), encoding="utf-8") as fh:
                saved = json.load(fh)
            rows = {}
            for r in saved:
                im = cv2.imread(os.path.join(self._dir, f"{gid}_{r['cam']}.jpg"))
                if im is not None:
                    rows[r["cam"]] = {**r, "crop": im}
            self._rows[gid] = rows
        except (OSError, ValueError):
            pass

    def _save_crop(self, gid, cam, local_id, at, crop, text, tconf, det_conf) -> None:
        try:
            d = os.path.join(self._dir, "crops")
            os.makedirs(d, exist_ok=True)
            name = f"{int(at * 1000)}_g{gid}_{cam}_t{local_id}.png"
            cv2.imwrite(os.path.join(d, name), crop)
            with open(os.path.join(d, "index.jsonl"), "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"file": name, "gid": gid, "cam": cam, "local_id": local_id, "at": at, "run": self._started,
                                     "text": text, "ocr_conf": round(tconf, 3), "det_conf": round(det_conf, 3)},
                                    ensure_ascii=False) + "\n")
        except Exception as exc:                          # saving evidence must never stop the reading
            logger.warning("plate crop save failed: %s", exc)

    def _write(self, gid: int) -> None:
        os.makedirs(self._dir, exist_ok=True)
        rows = sorted(self._rows[gid].values(), key=lambda r: -r["ocr_conf"])
        cv2.imwrite(os.path.join(self._dir, f"{gid}.png"), self._sheet(gid, rows))
        for r in rows:            # each camera's crop on its own, for the truck's profile
            cv2.imwrite(os.path.join(self._dir, f"{gid}_{r['cam']}.jpg"), r["crop"],
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
        with open(os.path.join(self._dir, f"{gid}_reads.json"), "w", encoding="utf-8") as fh:
            json.dump(self._reads.get(gid, []), fh, ensure_ascii=False)
        with open(os.path.join(self._dir, f"{gid}.json"), "w", encoding="utf-8") as fh:
            json.dump([{k: v for k, v in r.items() if k != "crop"} for r in rows], fh,
                      ensure_ascii=False, indent=1)

    def _sheet(self, gid: int, rows: list[dict]) -> np.ndarray:
        """One picture for a vehicle: a row per camera, the crop on the left and what was read
        beside it (text is drawn with PIL so the Arabic letters show)."""
        from PIL import Image, ImageDraw, ImageFont
        if self._font is None:
            path = next((p for p in _FONT_CANDIDATES if os.path.exists(p)), None)
            self._font = (ImageFont.truetype(path, 30), ImageFont.truetype(path, 16)) if path else \
                         (ImageFont.load_default(), ImageFont.load_default())
        big, small = self._font
        row_h, crop_w, width = 72, 240, 760
        img = Image.new("RGB", (width, 34 + row_h * len(rows)), (24, 24, 24))
        d = ImageDraw.Draw(img)
        d.text((8, 8), f"Global ID {gid}", font=small, fill=(180, 180, 180))
        for i, r in enumerate(rows):
            y = 34 + i * row_h
            c = r["crop"]
            s = min((row_h - 8) / c.shape[0], crop_w / c.shape[1])
            thumb = cv2.resize(c, (max(1, int(c.shape[1] * s)), max(1, int(c.shape[0] * s))),
                               interpolation=cv2.INTER_CUBIC)
            img.paste(Image.fromarray(thumb[:, :, ::-1]), (8, y + 4))
            d.text((crop_w + 24, y + 6), r["text"] or "-", font=big, fill=(240, 240, 240))
            d.text((crop_w + 24, y + 46),
                   f"{r['cam']} #{r['local_id']}   read {r['ocr_conf']:.2f}   plate {r['det_conf']:.2f}",
                   font=small, fill=(150, 150, 150))
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
