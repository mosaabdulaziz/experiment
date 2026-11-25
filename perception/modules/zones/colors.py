"""Detection-box color selection for the live feed.

Pure Python (no gi/pyds imports) so it stays unit-testable off-GPU.
"""

# BGR (OpenCV) box/label colors.
BOX_GREEN = (0, 255, 0)     # compliant person (and everything, when analytics is off)
BOX_RED = (94, 111, 255)    # person violating the zone — warm coral (#ff6f5e), matches
                            # the tool's light-orange theme rather than a harsh pure red
BOX_GRAY = (160, 160, 160)  # non-person detection (car, etc.)


def box_color(object_id, label, violating_ids, analytics_enabled, person_classes):
    """Pick a detection box's color. When analytics is off there is no zone concept,
    so everything is green. Otherwise: violating person -> red, other person -> green,
    non-person -> gray.

    A person "violates" when they break ANY of the stream's intrusion zones this frame --
    inside a restricted zone or outside a valid one. The probe computes that per-zone from
    each object's roiStatus (ROIs are non-inverted; see sidecar._frame_objects) and passes
    the union of violating ids here.
    """
    if not analytics_enabled:
        return BOX_GREEN
    if object_id in violating_ids:
        return BOX_RED
    if label in person_classes:
        return BOX_GREEN
    return BOX_GRAY
