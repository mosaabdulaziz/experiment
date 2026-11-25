import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _fmt_offset(s: float) -> str:
    hh, rem = divmod(int(s), 3600)
    mm, ss = divmod(rem, 60)
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


@dataclass
class Detection:
    """One detected object in one frame.

    track_id and bbox are populated by the DeepStream backend (the sidecar
    publishes them on iris:meta); the local YOLO backend leaves them None.
    bbox is (left, top, width, height) in mux pixel coordinates.
    """
    cls: str
    conf: float
    track_id: int | None = None
    bbox: tuple[float, float, float, float] | None = None


@dataclass
class FrameMeta:
    offset: float
    detections: list[Detection] = field(default_factory=list)


@dataclass
class CVBatchContext:
    frames: list[bytes] = field(default_factory=list)
    frame_metas: list[FrameMeta] = field(default_factory=list)

    def has_detections(self) -> bool:
        return any(m.detections for m in self.frame_metas)

    def format_metadata(self) -> str:
        lines = ["## CV OBSERVATIONS (local detector — use to guide your analysis)"]
        for meta in self.frame_metas:
            ts = _fmt_offset(meta.offset)
            if not meta.detections:
                lines.append(f"  [{ts}]  (no detections)")
                continue
            counts: Counter = Counter(d.cls for d in meta.detections)
            tl_confs = [d.conf for d in meta.detections if d.cls == "traffic light"]
            parts = []
            for cls, n in sorted(counts.items()):
                if cls == "traffic light" and tl_confs:
                    parts.append(f"traffic_light×{n}(conf:{max(tl_confs):.2f})")
                else:
                    parts.append(f"{cls}×{n}" if n > 1 else cls)
            lines.append(f"  [{ts}]  " + "  ".join(parts))
        return "\n".join(lines)


class CVEnricher:
    def __init__(self, config: dict):
        cv_cfg = config.get("cv", {})
        self._enabled = bool(cv_cfg.get("enabled", True))
        self._annotate = bool(cv_cfg.get("annotate_frames", False))
        self._conf = float(cv_cfg.get("confidence", 0.40))
        self._infer_width = int(cv_cfg.get("inference_width", 640))
        self._jpeg_quality = config.get("sampling", {}).get("jpeg_quality", 82)

        raw_classes = cv_cfg.get("relevant_classes")
        self._relevant: set[str] | None = set(raw_classes) if raw_classes else None
        self._debug_log_path: str = cv_cfg.get("debug_log_path", "data/cv_debug.json")
        self._debug_max_entries: int = int(cv_cfg.get("debug_max_entries", 200))
        self._debug_frame_path: str = cv_cfg.get("debug_frame_path", "data/debug_frame.jpg")

        self._model = None
        if self._enabled:
            try:
                from ultralytics import YOLO
                model_path = cv_cfg.get("yolo_model", "yolo11n.pt")
                self._model = YOLO(model_path)
                logger.info("CVEnricher: loaded %s", model_path)
            except ImportError:
                logger.warning("ultralytics not installed — CV enrichment disabled")
                self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled and self._model is not None

    def annotate_single(self, frame: bytes) -> tuple[bytes, list[Detection]]:
        """Annotate one JPEG frame with YOLO boxes. Returns (annotated_jpeg, detections).
        No logging side-effects — safe to call at stream fps."""
        if not self.enabled:
            return frame, []
        img = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return frame, []
        if self._infer_width > 0 and img.shape[1] > self._infer_width:
            scale = self._infer_width / img.shape[1]
            img = cv2.resize(img, (self._infer_width, int(img.shape[0] * scale)))
        results = self._model(img, conf=self._conf, verbose=False)[0]
        detections = [
            Detection(results.names[int(box.cls[0])], float(box.conf[0]))
            for box in results.boxes
            if self._relevant is None or results.names[int(box.cls[0])] in self._relevant
        ]
        annotated = results.plot(line_width=1, font_size=0.1, conf=False)
        ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
        return (buf.tobytes() if ok else frame), detections

    def process_batch(self, frames: list[bytes], offsets: list[float]) -> CVBatchContext:
        """Synchronous — must be called via run_in_executor from async code."""
        if not self.enabled:
            return CVBatchContext(frames=frames)

        out_frames: list[bytes] = []
        metas: list[FrameMeta] = []
        all_results = []

        for jpeg, offset in zip(frames, offsets):
            img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            if self._infer_width > 0 and img.shape[1] > self._infer_width:
                scale = self._infer_width / img.shape[1]
                img = cv2.resize(img, (self._infer_width, int(img.shape[0] * scale)))

            results = self._model(img, conf=self._conf, verbose=False)[0]
            all_results.append(results)

            detections = [
                Detection(results.names[int(box.cls[0])], float(box.conf[0]))
                for box in results.boxes
                if self._relevant is None or results.names[int(box.cls[0])] in self._relevant
            ]
            metas.append(FrameMeta(offset=offset, detections=detections))

            if self._annotate:
                annotated = results.plot(line_width=1, font_size=0.1, conf=False)
                ok, buf = cv2.imencode(
                    ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
                )
                out_frames.append(buf.tobytes() if ok else jpeg)
            else:
                out_frames.append(jpeg)

        # Always save the middle frame annotated for visual debugging
        self._save_debug_frame(all_results, metas)

        logger.debug("CVEnricher: processed %d frame(s)", len(frames))
        return CVBatchContext(frames=out_frames, frame_metas=metas)

    def _save_debug_frame(self, all_results: list, metas: list[FrameMeta]) -> None:
        if not all_results:
            return
        mid = len(all_results) // 2
        try:
            annotated = all_results[mid].plot(line_width=2, font_size=0.15, conf=False)
            # Burn the stream offset into the top-left corner for context
            offset_str = _fmt_offset(metas[mid].offset)
            cv2.putText(annotated, offset_str, (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imwrite(self._debug_frame_path, annotated)
        except OSError as exc:
            logger.warning("Debug frame save failed: %s", exc)

    def log_cycle(self, cv_ctx: CVBatchContext) -> None:
        """Append one analysis cycle to the debug JSON file."""
        if not self.enabled or not cv_ctx.frame_metas:
            return
        entry = {
            "t": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "frames": [
                {
                    "offset": _fmt_offset(m.offset),
                    "detections": [{"cls": d.cls, "conf": round(d.conf, 3)} for d in m.detections],
                }
                for m in cv_ctx.frame_metas
            ],
        }
        try:
            try:
                with open(self._debug_log_path, "r", encoding="utf-8") as f:
                    entries: list = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                entries = []
            entries.append(entry)
            if len(entries) > self._debug_max_entries:
                entries = entries[-self._debug_max_entries:]
            with open(self._debug_log_path, "w", encoding="utf-8") as f:
                json.dump(entries, f, indent=2)
        except OSError as exc:
            logger.warning("CV debug log write failed: %s", exc)

    def clear_log(self) -> None:
        """Truncate the debug JSON file back to an empty array."""
        try:
            with open(self._debug_log_path, "w", encoding="utf-8") as f:
                json.dump([], f)
        except OSError as exc:
            logger.warning("CV debug log clear failed: %s", exc)

    def save_debug_jpeg(self, jpeg: bytes) -> None:
        """Write a pre-annotated JPEG as the debug frame (used when skipping process_batch)."""
        try:
            with open(self._debug_frame_path, "wb") as f:
                f.write(jpeg)
        except OSError as exc:
            logger.warning("Debug frame save failed: %s", exc)
