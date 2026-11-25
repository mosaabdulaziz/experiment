"""The shared perception layer: cameras in, detections and faces out.

Everything here is used by every product, so a change belongs to nobody in
particular and needs review by both product owners (MULTI_PRODUCT_PLAN.md).
Named `perception`, not `platform`: a top-level `platform` package shadows
Python's standard-library module of that name and breaks `platform.system()`
for every dependency in the process (verified 2026-09-22).
"""
