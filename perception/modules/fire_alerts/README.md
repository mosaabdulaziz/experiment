# fire_alerts

Full-frame fire and smoke detection with sustained-incident alerting. Turned on by listing it under `modules:` in a
product's sidecar yaml (a product that does not list it runs no fire detector and raises no fire alerts):

```yaml
modules:
  fire_alerts:
    interval: 0
    fire_sustain_seconds: 2.0
    smoke_sustain_seconds: 4.0
    release_seconds: 10.0
    min_confidence: 0.45
    dropout_grace_seconds: 1.0
```

What it does: adds a second full-frame primary GIE (a D-Fire-trained YOLOv8 beside the COCO detector), keeps one
sustain latch per class per stream, writes `iris:overcrowd_alert:<cam>` with `kind=fire_smoke` /
`fire_smoke_resolved`, asks the engine for a clip when an incident starts and about once a second while it stays
open, and draws fire and smoke boxes in the hazard colour. It needs no zone and no analytics element.

**Fire zones** (`zones_path`, default `data/zones_config.txt`): a camera with zones whose `fire` block is on counts
only fire and smoke whose box centre is inside one, and the `fire_smoke` entry's `zone` names the zone of the class's
most confident detection; a camera with no such zone covers the whole frame. They reload on
`iris:ctrl:zones_version` (acknowledged on `iris:analytics:fire_zones_applied`); points are in mux pixels
(perception/lib/feature_zones.py).

## Why the numbers are what they are

(Moved here from the shared sidecar yaml, where this history used to sit beside the keys.)

**`interval` must stay 0.** The detector runs BEFORE the tracker, so on a batch where `interval` skips inference no
fire metadata is produced; the tracker keeps the object alive as a shadow track (confidence -0.1), which is excluded
from `dets` but still drawn in the preview. A nonzero interval therefore draws fire boxes continuously while starving
the VLM prompt: fire only reaches `dets` on real inference frames, roughly one published frame in five at
`interval: 4`, which silently defeats the feature. The cost of every-frame inference (~80 extra small-model
inferences/s at 8 streams x analysis_fps 10) is affordable; raising it is a deliberate trade against VLM prompt
fidelity, not a free saving. The same gap also keeps the sustain window from ever accumulating once it outlasts
`dropout_grace_seconds`.

**Sustain windows** are the primary false-positive control: the class must be present continuously for this long
before an alert fires (published results put temporal confirmation at 52% -> 4% false alarms on negative clips).
Fire is visually unambiguous so it confirms fast; smoke shares its appearance with steam, vapour, exhaust and dust,
so it waits longer. A 2026-09 tuning pass cut smoke to 2.0 s, which removed its debounce margin above fire's and made
false positives (and the stuck "ongoing" incidents they cause) easier to trigger; it was restored to 4.0. Reasoned,
not measured: tune against real site footage.

**`release_seconds`** is how long the condition must stay clear before the incident resolves. It was 30.0, reasoned.
Measured 2026-09-04 against `bucketfire.mp4` (raw per-frame detections, not the ~12 fps-sampled `iris:meta` view): the
clip's real, exactly-repeating fire-absence window is ~27.26 s (27.262 / 27.263 / 27.263 s over three loops), 2.7 s
SHORTER than 30.0, so the release countdown was reset by the next loop's fire reappearing just before it completed and
the incident could never resolve. It went to 15.0 (a real margin under 27.26 s), then to 10.0 on request: the
frontend's "ongoing" duration is a live wall-clock counter with no signal about the stream itself, so it keeps
counting for the whole release window after the fire is really gone, and a shorter window tightens that lag. It cannot
reach zero by design: some debounce is the point, or a single missed frame would flap resolved/started.

**`min_confidence`** gates which detections count. It was lowered to 0.35 in the same tuning pass, but the fire
model's own pre-cluster threshold (`DeepStream-Yolo-Fire/config_infer_fire.txt`) is 0.4, so 0.35 filtered nothing but
DeepStream's -0.1 shadow-track sentinel and every marginal model output counted. It is restored above the model's own
floor so it is an independent gate again. It must be positive for the same sentinel reason.

**`dropout_grace_seconds`**: detectors flicker, smoke especially, and with zero tolerance an 8 s window at ~10 fps
would demand 80 consecutive hits and might never be met. 1.0 s tolerates ~10 dropped frames at that rate. Raise it if
alerts never fire on real footage; lower it if brief glints start confirming.

## Testing

`tests/test_fire_module.py` (the module), `tests/test_fire_alerts.py` (the state machine), and the golden replay
`tests/configs/golden-fire.yaml` (a 61 s recorded fire clip; the alert fires at pts ~20.3 s).
