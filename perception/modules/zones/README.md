# zones

Restricted/valid regions drawn in the UI, with intrusion and overcrowding alerts, plus per-camera filter regions.

```yaml
modules:
  zones:
    config_path: data/zones_config.txt
    analytics_config_path: data/nvdsanalytics_config.txt
```

Every setting has a default, so `zones: {}` is enough for the standard layout.

## What it does

- **Intrusion alerts** (`iris:zone_alert:<cam>`): an `nvdsanalytics` element, which this module adds to the pipeline,
  flags which ROIs each tracked object is inside. One alert per tracked object entering a restricted zone (or leaving
  a valid one). `ZoneReentryGrace` swallows tracker re-IDs: a new id appearing near where an alerted id vanished less
  than `grace_seconds` ago inherits its alerted status.
- **Overcrowding alerts** (`iris:overcrowd_alert:<cam>`): the count of the chosen classes inside a zone crossing the
  zone's threshold, with a debounced release (`overcrowding_release_seconds`). Both alert kinds ask for a smart-record
  clip.
- **End markers** (`end_markers: true`, off by default): the end of an incident on the same streams, for an app that
  keeps incidents with an end (the safety product). `kind: overcrowding_resolved` (zone, count, duration) on
  `iris:overcrowd_alert:<cam>` when a zone's overcrowding latch releases; `kind: intrusion_cleared` (zone, duration)
  on `iris:zone_alert:<cam>` once a zone that alerted has had nobody violating it for `intrusion_clear_seconds` (5).
  The app's readers skip both kinds, so they never read as new alerts. A person who leaves and comes back under the
  same tracker id does not alert again (the once-per-id rule), so an intrusion ended by a clear is not re-opened by them.
- **Filter zones** (`data/filter_zones.json`, found beside the zones file by name): detections whose box centre is
  outside every filter polygon of a camera are dropped before anything later in the module order sees them, so MTMC
  and the preview agree. A camera with no filter polygons is unfiltered.
- **Box colours**: while some zone has intrusion detection on, a violating box is red, another tracked class green
  and the rest gray. Without it the engine's default applies. A hazard colour (`fire_alerts`) wins over these.

## Hot reload

The app writes the zone files and bumps `iris:ctrl:zones_version`; every zones process sees every bump. Each renders
its own `nvdsanalytics` config from the zones file and its own streammux slots (`<analytics_config_path
base>_worker_<role>.txt`; `analytics_config_path` itself is only the fallback when that cannot be written), so no
global stream numbering is involved. It reloads the element only if its render changed, and rebuilds a stream's zones
with fresh dedup state only if that stream's zones (settings or ROI geometry) changed: someone inside a re-drawn zone
re-alerts under the new geometry, while an edit on another camera leaves this camera's alert state and open incidents
alone. It then writes the applied version to `iris:analytics:zones_applied` and, for each camera it runs, to
`iris:analytics:zones_applied:<camera>`; the zone editor waits on the camera it saved ("applying" vs "active"). A
failed reload sends no acknowledgement and is retried on the next tick.

`Zones.slots_changed({slot: camera})` is the hook for attaching/detaching streams at runtime (#173): it re-renders,
ends a detached camera's open incidents, and gives a camera that gained, lost or changed its slot fresh zone state.

## Order

`transform` (the filter) runs where `zones` is listed relative to the other modules that change detections: list it
after `box_stabilizer` and before `plates`/`mtmc`, so stabilizing happens on every box and MTMC sees only filtered ones.

## Notes carried over from the old base yaml

- The `nvdsanalytics` element drew a spurious solid vertical line into the preview in a 2026-09-10 on/off test, despite
  `osd-mode=0` and no ROIs. The root cause was not found inside the plugin; revisit before relying on zones for a
  camera whose preview people watch.
- `analytics_config_path` lives under `data/` so the app container's shared `./data` mount carries the write out to
  the host without a new bind mount.
- Settings take effect on the next sidecar restart; zone edits themselves hot-reload.
