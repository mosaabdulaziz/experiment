# Pipeline modules

The whole design, with diagrams, is in [`perception/ARCHITECTURE.md`](../ARCHITECTURE.md); what the next use cases
(fall, PPE, proximity) need from the core is in [`perception/NEXT_MODULES.md`](../NEXT_MODULES.md).

The DeepStream engine (`perception/engine/`) decodes, infers, tracks, records, recovers and publishes. Everything that
is specific to one use case lives here, in a **module**, and a product switches modules on by listing them in its
sidecar yaml:

```yaml
modules:                      # ordered: name -> settings
  fire_alerts:
    interval: 0
    fire_sustain_seconds: 2.0
```

Nothing is switched on by inheritance. The shared `perception/pipeline/sidecar.yaml` has `modules: {}`, so a product
runs exactly the modules it names. An unknown module, an unknown setting, a missing required setting, or a `requires`
that is not satisfied refuses startup.

## Available modules

| module | what it does | docs |
|---|---|---|
| `fire_alerts` | a second full-frame detector plus sustained fire/smoke incidents, clip requests and hazard-coloured boxes | [README](fire_alerts/README.md) |
| `ppe` | a helmet/vest detector on each tracked person's crop, placed after the tracker, per-person PPE rules with smoothing per track, candidate violation alerts with clips, PPE boxes and violators drawn | [README](ppe/README.md) |
| `shelf_void` | a full-frame empty-shelf detector placed after the tracker, a heat map over time per camera, `shelf_empty` / `shelf_restocked` alerts with clips, open voids drawn | [README](shelf_void/README.md) |
| `zones` | restricted/valid regions with intrusion and overcrowding alerts, filter regions, zone box colours, hot reload | [README](zones/README.md) |
| `box_stabilizer` | steady whole-object boxes: drops nested part-boxes, holds a box across detector flicker | docstring in `box_stabilizer/__init__.py` |
| `person_boxes` | the tracked people's boxes as the `person_boxes` meta field | docstring in `person_boxes/__init__.py` |
| `person_masks` | each person's segmentation mask as an outline, the `person_masks` meta field | docstring in `person_masks/__init__.py` |
| `pose` | a secondary pose GIE on every person: the 17 keypoints in capture pixels, the `pose` meta field | [README](pose/README.md) |
| `face_detect` | a face detector on each person, whose face is whose, the `faces` meta field, a sampling gate and an idle switch | [README](face_detect/README.md) |
| `face_recognition` | names the faces against a Redis gallery with per-track voting; optional CCTV capture and enhancement | [README](face_recognition/README.md) |
| `customer_faces` | quality-gated face pictures with embeddings for a customer register | [README](customer_faces/README.md) |
| `appearance` | a re-ID embedding and a staff probability per sampled person; optional body crops | docstring in `appearance/__init__.py` |
| `plates` | a plate detector on each vehicle, one steady plate per vehicle, the `plates` meta field | docstring in `plates/__init__.py` |
| `mtmc` | one identity per vehicle across cameras (Re-ID, camera layout, a vision-language arbiter) and a reading of each vehicle's best plate | docstring in `mtmc/__init__.py` |
| `proximity` | a person too close to a moving vehicle inside a zone: per-camera person-scale map, hidden-feet recovery, parked vehicles excluded, one safety alert per episode, distance lines; optional depth check | [README](proximity/README.md) |

Nothing is left in the engine: `tests/test_engine_boundary.py` holds an empty ratchet over `perception/engine/`.

## Writing a module

Copy [`_template/`](_template/__init__.py), rename it, and keep only the hooks you need. The contract is in
[`perception/engine/modules.py`](../engine/modules.py); a module sees a frame only through `FrameView`/`ObjectView`
and changes what is published only through `FrameOutput` ([`frame.py`](../engine/frame.py)).

Hooks, in the order they run for one frame (streaming thread unless noted; never block in one). `FrameView` also
carries `scratch` (per-frame notes, keyed by module name) and `dets_with_boxes` (ask for 7-field detections):

1. `sample` -- decide whether this frame is a meta sample (first module with an opinion wins)
2. `extract` -- return a function the engine runs on every object, while its metadata is valid, to fill `obj.attrs`
   (the analytics region it is in, an instance mask); asked when the frame's objects are first read
3. `on_frame` -- every frame, before any gating: alerting that must see every frame
4. `transform` -- frames that will be published: change `frame.objects` (drop, replace or add objects, set their
   `draw`/`publish`/`kind`/`layer`), in the order the modules are listed
5. `wants_pixels` -- ask for the CPU copy of the frame
6. `on_pixels` -- given the CPU copy: crops, samplers
7. `analyze` -- objects final: events, annotations, extra meta through `FrameOutput`; `out.after_publish(fn)` hands
   the clean frame (as captured, before boxes were drawn on it) to `fn` on the publisher thread; `out.draw(fn)` has
   the publisher call `fn(img, sx, sy)` after the boxes, to paint a shape of your own (an outline, a distance line)

Plus `gies()` (inference elements the module needs), `elements()` (other pipeline elements, added after the inference
chain), `on_built(PipelineHandle)`, `start`/`stop`, `control()` (Redis keys the engine polls once a second and tells
you when they change) and `gauges()`. `on_built` and `start` failing refuses startup; a per-frame hook failing is
logged and counted and never stops the pipeline.

Things a module reaches through the engine, so none of them needs a core change:

- **Alerts**: `self.services.alert(stream_id, "proximity_warning", severity="critical", category="safety", pts=...)`
  writes `iris:alert:<camera>`; the app turns every entry into an event (saved, broadcast, clip queued) with no code of
  its own for your module. `message=` overrides the generated text.
- **Object attributes**: `extract` fills `obj.attrs[...]` from DeepStream metadata through `perception.engine.nvds`
  (`analytics_roi`, `instance_mask`) -- the only place pyds is touched for a module; modules read each other's
  attributes by key, so document yours in the module's docstring.
- **Running a model on fewer objects/frames**: `perception.engine.gie_gate.GieGate.install(element, ...)` keeps a
  secondary GIE off unsampled frames and off objects `object_allowed(camera, box)` refuses (what `face_detect`'s
  `sample_only`/`regions_key` use).
- **Settings the app edits live**: `perception.lib.versioned.VersionedConfig` gives the version-bump, reload and
  acknowledge protocol zones use (`control()` returns `cfg.control(self._reload)`).
- **Another model**: its `config_infer_*.txt` under `perception/models/<Name>/`; a custom output parser is a
  `nvdsinfer_custom_impl_*/` directory there, built by the image automatically (`perception/models/parser_flags` for `make` variables).
- **Another dependency**: a `requirements.txt` in the module's folder; `tests/test_module_requirements.py` fails when a
  module imports something no requirements file declares. Add the module to the Dockerfile's `BASE_MODULES` (every image)
  or `MTMC_MODULES` (the heavy image), then `just deps-lock` (DEPENDENCIES.md; a `--build-arg` that differs from the
  lock stops the build).
- **What a model costs on the GPU**: `tools/engine_gpu_budget.sh <config>` prints each GPU stage's busy%; run it before
  and after adding a module.
- **Order**: `requires = (...)` must be configured and listed before; `after = (...)` only constrains the order when
  that module is configured (use it for a module that reads what another leaves when present).

Order is part of a product's configuration: hooks run in the order modules are listed, so a module that changes
detections (the stabilizer, a filter) acts on what the modules above it left, and a module that reports on detections
sees them as of where it is listed.

Shared vocabulary -- modules agree on these between themselves, the engine never compares one:

| name | meaning | set by | read by |
|---|---|---|---|
| kind `object` | an ordinary detection from the primary detector | engine | everyone |
| kind `face` | a face box on its owner's track id (`parent_id` = the owner), `publish=False`, layer 0 | `face_detect` | `face_recognition`, `customer_faces` |
| kind `plate` / `plate_shown` | a vehicle's plate box / the steadied one that is drawn | `plates` | `mtmc` |
| label `person` | the detector's person class | the model | `person_boxes`, `person_masks`, `face_detect`, `appearance`, `zones`, `ppe` |
| label `pose` | the pose model's own class name (`labels.txt`), so `relevant_classes` keeps its items out of the dets | the model | `pose` |
| `attrs["roi"]` | analytics regions the object is in (frozenset of labels) | `zones` | `zones` |
| `attrs["landmarks"]` | five facial landmarks in capture pixels | `face_detect` | `face_recognition`, `customer_faces` |
| `attrs["keypoints_raw"]` | the 17 keypoints as the parser packed them, with the network's input size | `pose` | `pose` |
| `attrs["mask"]` | the instance mask (`nvds.InstanceMask`) | `person_masks`, `proximity` | `person_masks`, `proximity` |
| `attrs["global_id"]` | the cross-camera vehicle id | `mtmc` | modules listed after it |
| kind `ppe_item` | a PPE detector box (Hardhat, NO-Hardhat, Safety Vest, NO-Safety Vest): untracked (no meaningful `id`), drawn, `publish=False`, layer 2 | `ppe` | `ppe` |
| kind `shelf_void_raw` | a void detector box (label `empty`): untracked, never drawn, never published; evidence for the heat map | `shelf_void` | `shelf_void` |

A new module that introduces a kind or an attribute adds a row here.

An object's `kind` says what it is: `"object"` is an ordinary detection, a module uses its own kind (`"face"`) for
the objects it takes over so another module can leave them alone. Annotations (colour, text) belong to one object, so
a face box and the person box beside it on the same track id are styled apart.

Rules that keep the boundary real:

- The engine never imports a module by name, and a module never calls another module. Data passes through
  `ObjectView` fields and `FrameOutput` only; say so in `requires` when you rely on an earlier module.
- Every tunable is a field of the module's `Settings` dataclass; unknown yaml keys are errors.
- No pyds in a module's hot path: use `FrameView`/`ObjectView`; what has to come from metadata goes through
  `perception.engine.nvds`.
- `tests/test_engine_boundary.py` is a ratchet over `perception/engine/`: it may not gain imports of use-case code or
  mentions of a product, and the recorded counts only go down.
- Add the module's own tests, and if it changes what the sidecar publishes, a golden config that exercises it
  (`tests/configs/golden-*.yaml`).
