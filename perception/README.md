# perception/ -- the DeepStream sidecar

The GPU side of IRIS: it decodes camera streams, runs the detectors, tracks, records, and publishes what it sees to
Redis. It is built as a **use-case-free engine** plus **modules** (one per capability), so a product is a yaml file that
lists the modules it wants. Read the documents in this order:

| # | Read | To learn |
|---|---|---|
| 0 | [`PIPELINE_FLOW.md`](PIPELINE_FLOW.md) | **start here if you want the AI flow as a story**: from `just ...` to the last Redis key, DeepStream configuration, how modules tell it what they need and get data back, and attendance, barber and multicam end to end |
| 1 | [`ARCHITECTURE.md`](ARCHITECTURE.md) | the idea, the system from above, one frame step by step, what a module sees, the hooks, configuration layers, the Redis keys, what changed from the old single-file sidecar (with diagrams) |
| 2 | [`modules/README.md`](modules/README.md) | the module catalogue, how to write one, what the engine offers a module (alerts, draw, attributes, GIE gate, versioned reload, models, dependencies), the vocabulary modules share |
| 3 | [`modules/<name>/README.md`](modules/) | one module's settings (zones, fire_alerts, face_detect, face_recognition, customer_faces; the others document themselves in their `__init__.py`) |
| 4 | [`TESTING.md`](TESTING.md) | unit tests, the golden replay that proves a change did not alter behaviour, the GPU budget tool, and how to bring each product up and what to check |
| 5 | [`MIGRATING.md`](MIGRATING.md) | porting a branch written against the old `sidecar.py`: where each file and key went |
| 6 | [`NEXT_MODULES.md`](NEXT_MODULES.md) | how fall detection, PPE and vehicle-to-human proximity fit, and what in the core was changed for them |

## Where things are

```
perception/
  engine/      the engine: pipeline, per-frame probe, publisher, recording, recovery, telemetry   (no use case in here)
  modules/     one folder per capability + _template/ to copy                                      (all the use-case logic)
  lib/         shared helpers with no GStreamer (versioned reload, dedup)
  models/      one folder per model family: nvinfer configs, ONNX, custom parsers (nvdsinfer_custom_impl_*/)
  pipeline/    sidecar.yaml (shared engine settings), the Dockerfile, diagnostic tools
products/<name>/sidecar.yaml    a product = its cameras + its `modules:` list (+ roles it runs as)
configs/presets/<name>/         app + pipeline presets (`just multicam`)
tests/configs/                  recorded-clip configs: golden-*.yaml (replay checks), demo and CHIRLA configs
tools/                          engine_golden*.sh/.py (replay), engine_gpu_budget.sh/.py (GPU cost per model)
```

The engine is started as `python3 -m perception.engine.main <yaml>`; compose picks the yaml with `SIDECAR_CONFIG`
(or the product's own service). A product yaml says `extends: ../../perception/pipeline/sidecar.yaml`, then adds
`sources:` and an ordered `modules:` mapping; order is hook order.

## The five rules that keep it this way

1. The engine imports no module and no product, and names no use case (`tests/test_engine_boundary.py` enforces it).
2. A module never calls another module: data flows through `ObjectView` fields (`kind`, `attrs`) and `FrameOutput`.
3. A product differs from another only by yaml: the modules it lists and their settings.
4. pyds is touched only through `perception/engine/nvds.py`.
5. Anything that changes what is published is proven by the golden replay, not by eye (see `TESTING.md`).

## Adding a capability, in one paragraph

Copy `modules/_template/`, name it by capability, write its `Settings` and the hooks it needs, list it in a product
yaml, add unit tests. A model: a `perception/models/<Name>/` folder and a `GieSpec` from `gies()`; a custom parser is a
`nvdsinfer_custom_impl_*/` directory there (the image builds it). A dependency: the module's `requirements.txt`. An
alert the app should show: `services.alert(...)`. Details and the checklist are in `modules/README.md` and
`ARCHITECTURE.md` section 12.
