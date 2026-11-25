"""The measurement harnesses: the corpus replay, the latency, the classifier's cost.

Three reports, one shape: the numbers the requirement asks for, or `skipped` with the reason the
run could not happen here. A number is never reported that was not measured, and a number that
could not be measured is `null`, not zero (design §N2, §N4, §N5).

The heavy imports (torch, ultralytics, the vendor's copy) happen inside the harnesses, so
`import measure` works on a box that has none of them: a measurement is gated by its
prerequisites, not by an import.

The replay is the vendor's own harness (`fall_detection/evaluate_benchmark.py`) with the
product's policy applied where the service applies it (design §5.2): the harness counts the
product's alerts, not the vendor's candidates. The interval is the harness's frame stride --
the SGIE's re-inference interval is the pipeline's frame rate for that stage, and the replay
models it by processing every Nth frame. The proxy's limit (the detector's rate drops with it,
so the tracking is pessimistic) is stated in every report.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from pathlib import Path

PRODUCT = Path(__file__).resolve().parent
DEFAULT_CORPUS = Path(os.environ.get("FALL_CORPUS") or r"F:\fall_detection\data")
MANIFEST_NAME = "manifest.csv"
# the vendor's own matching rules -- the replay uses them, so the numbers are comparable
TOLERANCE_S = 2.0
MIN_IOU = 0.3
# the pose model the replay detects with: the vendor's own default, overridable by path
DEFAULT_MODEL = "yolo26x-pose.pt"
# the clip the classifier takes: (batch, 2, 100, 17, 3) -- the vendor's own window shape
CLIP_SHAPE = (1, 2, 100, 17, 3)


# ── the public harnesses ─────────────────────────────────────────────────────────────────────────────────
def action_path_eval(corpus_root=None, interval: int = 2, device: str | None = None,
                      manifest: str = MANIFEST_NAME, model_name: str = DEFAULT_MODEL,
                      **kwargs) -> dict:
    """The action path's contribution on the corpus, at the given interval.

    The report carries the product's precision, recall and false alarms per hour, the share of
    the fall decisions the action path carried, and the attribution count the requirement's
    "the candidate is on the right camera and track" clause asks for (design §N2, §N4).
    """
    return _replay(corpus_root, interval=interval, device=device, manifest=manifest,
                   model_name=model_name, **kwargs)


def latency(corpus_root=None, device: str | None = None, manifest: str = MANIFEST_NAME,
            model_name: str = DEFAULT_MODEL, **kwargs) -> dict:
    """The onset-to-alert latency, end to end, on the replayed footage.

    The onset is the ground truth's fall onset, the alert the product's alert. The maximum is
    the requirement's number -- every fall's onset-to-alert is under 5 s -- and the mean and
    the 95th percentile are beside it, because a mean hides a slow tail (design §N3).

    The latency is `null` when no fall was matched: a report that says "0.0 s" because nothing
    was measured is a false green, and the requirement's number is about the falls that were
    detected.
    """
    report = _replay(corpus_root, interval=kwargs.pop("interval", 2), device=device,
                     manifest=manifest, model_name=model_name, **kwargs)
    if report.get("skipped"):
        return report
    return {"skipped": False,
            "onset_to_alert_s": report["onset_to_alert_s"],
            "onset_to_alert_mean_s": report["onset_to_alert_mean_s"],
            "onset_to_alert_p95_s": report["onset_to_alert_p95_s"],
            "matched_falls": report["true_positives"],
            "videos": report["videos"],
            "interval": report["interval"]}


def classifier_cost(device: str = "cuda:0", concurrent_tracks: tuple[int, ...] = (1, 4, 8)) -> dict:
    """The classifier's cost on the card it shares with the pose GIE.

    ms per forward at each concurrency level, and the card's utilisation WHILE the forwards
    run (a snapshot taken afterwards reads an idle card). The card is the deployment's RTX 5090
    (design §N5): the number is meaningless on another one, so the harness says which card it
    measured.
    """
    import torch

    try:
        cfg = _config()
    except Exception as exc:  # noqa: BLE001 -- a broken config is a skip, not a crash
        return _skipped(f"the product's config could not be read: {exc}")
    device = device or str(cfg.classifier["device"])
    if not torch.cuda.is_available():
        return _skipped(f"{device} is not available on this machine")
    classifier, reason = _classifier(device)
    if classifier is None:
        return _skipped(reason)

    card = _card_name()
    streams = tuple(getattr(classifier, "streams", ("j", "jm")))
    results = []
    for tracks in concurrent_tracks:
        measured = _concurrent_forwards(classifier, streams, int(tracks), device)
        if measured.get("skipped"):
            return _skipped(measured["reason"])
        results.append({"concurrent_tracks": int(tracks), **measured})

    return {"skipped": False,
            "device": device,
            "card": card,
            "ms_per_forward": round(sum(r["ms_per_forward"] for r in results) / len(results), 3),
            "concurrent_tracks": [r["concurrent_tracks"] for r in results],
            "gpu_util": max((r["gpu_util"] for r in results), default=0.0),
            "per_level": results}


# ── the replay ─────────────────────────────────────────────────────────────────────────────────────────
def _replay(corpus_root, interval: int, device: str | None = None, manifest: str = MANIFEST_NAME,
            model_name: str = DEFAULT_MODEL, **kwargs) -> dict:
    """The one replay: the corpus through the vendor's monitor and the product's policy."""
    root = Path(corpus_root) if corpus_root is not None else DEFAULT_CORPUS
    manifest_path = root / manifest
    if not manifest_path.is_file():
        return _skipped(f"the corpus manifest {manifest_path} is not on this machine")

    harness, reason = _vendor_harness()
    if harness is None:
        return _skipped(reason)

    try:
        cfg = _config()
    except Exception as exc:  # noqa: BLE001 -- a broken config is a skip, not a crash
        return _skipped(f"the product's config could not be read: {exc}")
    device = device or str(cfg.classifier["device"])
    classifier, reason = _classifier(device)
    if classifier is None:
        return _skipped(reason)

    policy, vendor_cfg = _policy(cfg)
    monitor_cls = _product_monitor(policy, vendor_cfg)

    with tempfile.TemporaryDirectory(prefix="fall-measure-") as tmp:
        metrics = harness.evaluate_video_manifest(
            manifest_path=manifest_path,
            monitor_cls=monitor_cls,
            classifier=classifier,
            device=device,
            model_name=model_name,
            stride=max(1, int(interval)),
            output_dir=tmp,
            **kwargs,
        )
        summary = json.loads((Path(tmp) / "benchmark_summary.json").read_text(encoding="utf-8"))
        records = summary.get("records") or []

    truth = _ground_truth(manifest_path, records, harness)
    return _decide(records, truth, metrics, interval, manifest_path, policy, harness)


def _decide(records: list, truth: dict, metrics: dict, interval: int, manifest_path: Path,
            policy: dict, harness) -> dict:
    """The product's fall alerts against the ground truth.

    A confirmed alert is a candidate the product confirmed on its own; a delegated one went to a
    human (design §5.3). The precision and the recall are the confirmed ones -- they are the
    product's own claims -- and the delegated ones are counted beside them, with their own false
    alarms, because they reach the platform as warnings.

    A confirmed alert that does not match the ground truth's fall is a false alarm WHETHER the
    video has a fall or not: a fall the product missed while alarming on something else costs
    both a false positive and a false negative, and counting only the second would flatter the
    precision.
    """
    scoring = policy["scoring"]
    threshold = float(scoring["confirm_threshold"])
    action_trust = float(scoring["trust"]["action"])

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    delegated = 0
    delegated_false = 0
    alerts = 0
    confirmed_total = 0
    suppressed = 0
    with_reading = 0
    alone = 0
    attributed = 0
    attribution_checked = 0
    attribution_unavailable = 0
    latencies: list[float] = []

    for record in records:
        gt = truth.get(str(record.get("video")), {})
        fall_alerts = [e for e in record.get("events") or [] if e.get("type") == "fall"]
        # the product's own alarms: the aftermath of an alerted fall is suppressed on its track
        # (design §13.3), so those candidates never reach the platform
        product_alerts = [e for e in fall_alerts if not e.get("suppressed")]
        suppressed += len(fall_alerts) - len(product_alerts)
        confirmed = [e for e in product_alerts if not (e.get("decision") or {}).get("review")]
        delegated_here = [e for e in product_alerts if (e.get("decision") or {}).get("review")]
        alerts += len(product_alerts)
        confirmed_total += len(confirmed)
        delegated += len(delegated_here)
        with_reading += sum(1 for e in product_alerts
                            if (e.get("decision") or {}).get("scores", {}).get("action") is not None)
        alone += sum(1 for e in confirmed if _action_alone(e, action_trust, threshold))

        matched = None
        for event in confirmed:
            if harness.match_fall_event_to_gt(event, gt, tolerance_s=TOLERANCE_S, min_iou=MIN_IOU):
                matched = event
                break

        if gt.get("has_fall"):
            if matched is None:
                false_negatives += 1
                false_positives += len(confirmed)     # every confirmed alert here is a false alarm
            else:
                true_positives += 1
                false_positives += len(confirmed) - 1
                verdict = _attribution(matched, gt.get("boxes") or {}, harness)
                if verdict is None:
                    attribution_unavailable += 1
                else:
                    attribution_checked += 1
                    attributed += 1 if verdict else 0
                onset = float(gt.get("fall_start_s") or 0.0)
                latencies.append(max(0.0, float(matched.get("t_alert",
                                                             matched.get("t_candidate", 0.0))) - onset))
        else:
            false_positives += len(confirmed)

        for event in delegated_here:
            if not harness.match_fall_event_to_gt(event, gt, tolerance_s=TOLERANCE_S, min_iou=MIN_IOU):
                delegated_false += 1

    hours = float(metrics.get("total_duration_hours") or 0.0)
    precision = _ratio(true_positives, true_positives + false_positives)
    recall = _ratio(true_positives, true_positives + false_negatives)
    return {
        "skipped": False,
        "interval": int(interval),
        "manifest": str(manifest_path),
        "videos": len(records),
        "hours": round(hours, 4),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "false_alarms_per_hour": _ratio(false_positives, hours) if hours > 0 else None,
        "action_path_share": _ratio(with_reading, alerts),
        "action_path_alone": _ratio(alone, confirmed_total),
        "attributed": attributed,
        "attribution_checked": attribution_checked,
        "attribution_unavailable": attribution_unavailable,
        "delegated": delegated,
        "delegated_false_alarms": delegated_false,
        "platform_false_alarms_per_hour": _ratio(false_positives + delegated_false, hours)
        if hours > 0 else None,
        "alerts": alerts,
        "confirmed": confirmed_total,
        "suppressed": suppressed,
        "onset_to_alert_s": round(max(latencies), 4) if latencies else None,
        "onset_to_alert_mean_s": round(sum(latencies) / len(latencies), 4) if latencies else None,
        "onset_to_alert_p95_s": round(_percentile(latencies, 95), 4) if latencies else None,
        "note": "the interval is the replay's frame stride: the detector's rate drops with it, so "
                "the tracking is a pessimistic proxy for the SGIE's own interval; the precision "
                "and the recall are over the product's confirmed alerts, the delegated ones are "
                "counted beside them (they reach the platform as warnings)",
    }


def _action_alone(event: dict, action_trust: float, threshold: float) -> bool:
    """Whether the action path alone would have confirmed this candidate.

    The combined score is 1 - Π(1 - trust[path] * score[path]); with only the action path in
    it, that path alone reaches `trust.action * score.action`. A candidate with no reading has
    no action score at all, so the action path cannot have carried it."""
    score = (event.get("decision") or {}).get("scores", {}).get("action")
    if score is None:
        return False
    return (1.0 - (1.0 - action_trust * float(score))) >= threshold


def _attribution(event: dict, gt_boxes: dict, harness):
    """Whether the alert's box clears the ground truth's spatial gate, or None when it cannot be checked.

    The gate is the vendor's own: the alert's box against the ground truth's boxes, the best
    IoU over all of them. It can only be checked when both sides carry a box -- a corpus
    without per-frame boxes (the omnifall manifest) cannot answer the requirement's "the
    candidate is on the right camera and the right track" clause, and saying so is better than
    counting a check that never ran."""
    box = event.get("box")
    if not gt_boxes or not box:
        return None
    best = max(harness.compute_box_iou(box, b) for b in gt_boxes.values())
    return best >= MIN_IOU


def _ground_truth(manifest_path: Path, records: list, harness) -> dict:
    """The ground truth of each replayed video: the harness's times, the annotation's boxes.

    The harness parses the times (it knows the video's frame rate); the boxes are what the
    spatial gate needs and they live in the annotation, so they are parsed here.
    """
    boxes = {}
    for entry in harness.load_manifest(manifest_path):
        annotation = entry.get("annotation_path")
        if not annotation:
            continue
        path = _resolve(manifest_path.parent, annotation)
        if not path.is_file():
            continue
        video = _resolve(manifest_path.parent, entry["video_path"])
        boxes[str(video)] = harness.parse_le2i_annotation(path).get("boxes") or {}

    truth = {}
    for record in records:
        truth[str(record.get("video"))] = {
            "has_fall": bool(record.get("gt_fall")),
            "fall_start_s": record.get("fall_start_s"),
            "fall_end_s": record.get("fall_end_s"),
            "boxes": boxes.get(str(record.get("video")), {}),
        }
    return truth


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


# ── the product's own pieces ─────────────────────────────────────────────────────────────────────────
def _config():
    from config import load_config
    return load_config()


def _policy(cfg):
    """The product's policy and the vendor's blocks it scores against."""
    from monitor import vendor_blocks
    return ({"scoring": cfg.scoring, "suppress_after_fall_s": cfg.suppress_after_fall_s},
            vendor_blocks())


def _vendor_harness():
    """The vendor's benchmark harness, or the reason it cannot be used here.

    The copy is a package, so importing one of its modules pulls the whole package in (torch,
    mmaction); the import shim has to be installed first. Every failure is a skip, never a
    crash: a measurement that cannot run says why."""
    try:
        import mmaction_compat  # noqa: F401  (the import shim, before the copy is loaded)
        import fall_detection.evaluate_benchmark as harness
    except Exception as exc:  # noqa: BLE001 -- a missing dependency must skip, not fail the run
        return None, f"the vendor's benchmark harness could not be imported: {exc}"
    return harness, None


def _classifier(device: str):
    """The product's classifier, or the reason it cannot be built here."""
    try:
        import mmaction_compat  # noqa: F401  (the import shim, before the copy is loaded)
        from fall_detection.tsstg_model import STREAM_FILE_MAP, TSSTGClassifier
    except Exception as exc:  # noqa: BLE001 -- a missing dependency must skip, not fail the run
        return None, f"the classifier could not be imported: {exc}"

    cfg = _config()
    weights_dir = Path(cfg.classifier["weights_dir"])
    streams = tuple(cfg.classifier.get("streams") or ("j", "jm"))
    file_map = STREAM_FILE_MAP["mmaction2"]
    for stream in streams:
        path = weights_dir / file_map.get(stream, "")
        if not path.is_file():
            return None, f"the checkpoint {path.name} is missing ({path})"
    try:
        model = TSSTGClassifier(weight_dir=str(weights_dir), variant="mmaction2",
                                streams=streams, device=device)
    except Exception as exc:
        return None, f"the classifier could not be built: {exc}"
    return model, None


def _product_monitor(policy: dict, vendor_cfg: dict):
    """The vendor's monitor, with the product's decision applied where the service applies it.

    The service scores the candidates and routes them (design §5.2, §5.3); the replay does the
    same here, so the harness measures the product's alerts and not the vendor's candidates.
    Three things the service does are reproduced here:
      - the candidate carries the track's box -- the vendor's event record has none, and the
        ground truth's spatial gate only fires when the event has one;
      - the aftermath of an alerted fall is suppressed on that track (design §13.3), so the
        replay counts the alarms the platform would really receive;
      - only the candidates the policy gives a severity are returned: a closed false alarm
        (a confirmed upright reading) raises nothing.
    """
    try:
        import mmaction_compat  # noqa: F401  (the import shim, before the copy is loaded)
        from fall_detection.fall_monitor import FallMonitor
    except Exception as exc:  # noqa: BLE001 -- a missing dependency must skip, not fail the run
        raise RuntimeError(f"the vendor's monitor could not be imported: {exc}") from exc
    import scoring

    class _ProductMonitor(FallMonitor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.policy = policy
            self.vendor_cfg = vendor_cfg
            self.alerted_falls: dict = {}     # track -> the time of the last fall alerted on it

        def update(self, t, ids, boxes, kps, frame=None):
            events = super().update(t, ids, boxes, kps, frame=frame)
            alerts = []
            for event in events:
                if event.get("type") == "fall_recovered":
                    continue                       # the product ignores it (design §5.2)
                event["box"] = self._track_box(event.get("track"))
                event["decision"] = scoring.score_candidate(event, self.policy, self.vendor_cfg)
                if event["decision"]["severity"] is None:
                    continue                       # a closed false alarm: no severity, no alert
                now = float(event.get("t_alert", t))
                event["suppressed"] = self._is_aftermath(event.get("track"), now)
                if not event["suppressed"] and event.get("type") == "fall":
                    self.alerted_falls[event.get("track")] = now
                alerts.append(event)
            return alerts

        def _is_aftermath(self, track, now: float) -> bool:
            last = self.alerted_falls.get(track)
            if last is None:
                return False
            return (now - last) <= float(self.policy["suppress_after_fall_s"])

        def _track_box(self, track):
            data = self.tracks.get(int(track)) if track is not None else None
            if not data or not data.get("box"):
                return None
            return [float(v) for v in data["box"][-1]]

    return _ProductMonitor


# ── the small helpers ───────────────────────────────────────────────────────────────────────────────
def _skipped(reason: str) -> dict:
    return {"skipped": True, "reason": reason}


def _ratio(numerator: int, denominator: float):
    """The share, or None when there is nothing to take the share of.

    A zero here would be a claim: "the action path carried none of the decisions" reads very
    differently from "there were no decisions to carry"."""
    return round(numerator / denominator, 4) if denominator else None


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((percentile / 100.0) * (len(ordered) - 1)))))
    return float(ordered[index])


def _concurrent_forwards(classifier, streams, tracks: int, device: str) -> dict:
    """Time `tracks` forwards that run at the same time, and sample the card while they run.

    The service classifies one track at a time per camera worker, so N tracks in flight means N
    forwards queued on the same card: the number that matters is the wall time for all of them
    (the batch's latency) and the card's utilisation during it -- a snapshot taken after the run
    reads an idle card."""
    import torch

    tracks = max(1, int(tracks))
    clips = {stream: torch.zeros(CLIP_SHAPE, device=device) for stream in streams}
    for stream, clip in clips.items():                    # a warm-up forward, not measured
        classifier({stream: clip})
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)

    sampler = _GpuSampler()
    sampler.start()
    failures: list = []

    def work() -> None:
        try:
            classifier(clips)
        except Exception as exc:  # noqa: BLE001 -- reported below, not raised into the harness
            failures.append(exc)

    started = time.perf_counter()
    threads = [threading.Thread(target=work) for _ in range(tracks)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    sampler.stop()
    if failures:
        return {"skipped": True, "reason": f"a forward failed: {failures[0]}"}
    return {"ms_per_forward": round(elapsed_ms / tracks, 3),
            "ms_for_batch": round(elapsed_ms, 3),
            "gpu_util": sampler.peak}


class _GpuSampler(threading.Thread):
    """The card's utilisation while the forwards run.

    Polled, not sampled once: a snapshot taken after the run reads an idle card, and a report
    that says "the card was 0% busy" while the classifier was running is a false green."""

    def __init__(self, interval: float = 0.05):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak = 0.0
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, _gpu_utilisation())
            self._stop.wait(self.interval)

    def stop(self) -> None:
        self._stop.set()
        self.join(timeout=2.0)


def _card_name() -> str:
    import shutil
    import subprocess
    if shutil.which("nvidia-smi") is None:
        return ""
    out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                         capture_output=True, text=True)
    return out.stdout.strip()


def _gpu_utilisation() -> float:
    import shutil
    import subprocess
    if shutil.which("nvidia-smi") is None:
        return 0.0
    out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True)
    try:
        return float(out.stdout.strip().splitlines()[0])
    except (IndexError, ValueError):
        return 0.0
