"""The DeepStream engine: decode, infer, track, publish, record, recover.

It knows no use case. What a product does with the frames lives in perception/modules/, selected and tuned by the
`modules:` section of the product's sidecar yaml, and reaches the engine only through the hook API in modules.py.
Run it with `python3 -m perception.engine.main <sidecar.yaml>` from the repo root (the image's working directory).
Importing this package touches nothing: no sys.path edits, no GStreamer.
"""
