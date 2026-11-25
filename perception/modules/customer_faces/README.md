# customer_faces

Quality-gated face pictures with embeddings, for a register of visiting customers (the barber product). The module
docstring in `__init__.py` has the gates and what is published.

```yaml
modules:
  face_detect: {landmarks: true}
  customer_faces:
    crops_dir: /workspace/iris/data/barber/crops
    min_pixels: 20
    min_sharpness: 5.0
    max_yaw: 0.9
    interval_s: 0.25
```

`face_detect` must be listed before it, with `landmarks: true`. Settings it replaces: `customer_faces_enabled`,
`customer_crops_dir`, `face_min_pixels_customer`, `face_min_sharpness`, `face_max_yaw`, `customer_face_interval_s`.

It loads its own ArcFace model: before, the embedding ran on the recognition thread with the recognition model, so
customer faces only existed while face recognition did. Now the two are independent.
