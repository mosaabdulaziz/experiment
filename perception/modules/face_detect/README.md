# face_detect

Finds faces on the people the primary detector tracks (a secondary YOLO face detector, run on each person's crop) and
decides whose face each one is. The module docstring in `__init__.py` lists what it publishes and the settings;
this page is the product configuration and the notes the shared yaml used to carry.

```yaml
modules:
  face_detect:
    confidence_threshold: 0.18
    min_pixels: 24
    owner_mode: legacy            # legacy | tracked
    landmarks: true               # recognition and customer_faces need the five landmarks
    idle_when_unused: {gallery_key: attendance:face:gallery, capture: true}
```

| setting | meaning |
|---|---|
| `config`, `engine` | the face detector's nvinfer config, and the engine path nvinfer deserializes (empty: beside the config) |
| `confidence_threshold`, `min_pixels` | a face below the threshold is not a face; one whose shorter side is under `min_pixels` capture pixels is dropped before it is drawn or recognised |
| `owner_mode`, `owner_fallback` | which person box a face belongs to; see `owner.py` for the rules. Per product: it applies to every camera of the process and changes what recognition sees. Measure on the product's recorded clip before switching |
| `landmarks` | attach five landmarks (capture pixels) to every face |
| `sample_only`, `regions_key` | run the detector only on published frames, and only on people inside the door areas a service publishes to `regions_key` (`regions.py`). Needs `pipeline_preview: false` and no `process_fps` throttle, else the module refuses to start |
| `idle_when_unused` | switch the detector off while the gallery is empty and nothing captures faces (`idle.py`) |

## Notes from the shared yaml

These notes predate the modules, when the keys below sat at the top level of the shared yaml. They are now settings of
this module (or of the product's module list): `sgie_face_enabled` is "list `face_detect` under `modules:`",
`sgie_auto_interval` / `sgie_auto_poll_seconds` are `idle_when_unused`, `sgie_face_sample_only` is `sample_only`
(the gate itself is `perception/engine/gie_gate.py`, shared by any secondary GIE), and `sgie_face_config` is `config`.

```yaml
# Face detection (secondary GIE, runs on person detections from pgie)
# NOTE: disabled by default because the face parser .so was never compiled
# into the DeepStream image.  To enable, build the custom parser first:
#   cd nvdsinfer_custom_impl_Yolo_face && make CUDA_VER=13.1
sgie_face_enabled: true
# Auto-toggle the face SGIE off when no gallery is registered: nvinfer
# `interval` skips every batch (zero GPU, buffers pass through) and restores
# to 0 the moment a face is registered or face capture is enabled. Set false
# to keep the SGIE always-on (pre-auto-toggle behavior).
sgie_auto_interval: true
sgie_auto_poll_seconds: 10
# Run sgie_face only on the frames whose meta is published (one per
# sample_fps interval per source) instead of every frame -- faces are only
# read on those. Needs pipeline_preview: false (preview frames draw faces) and
# process_fps: 0; ignored with a warning otherwise. See
# perception/engine/gie_gate.py.
sgie_face_sample_only: false
sgie_face_config: "/workspace/iris/perception/models/DeepStream-Yolo-Face/config_sgie_face.txt"
```
