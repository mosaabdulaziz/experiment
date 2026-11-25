"""The scoring policy: the evaluation, the reading's margin, the combination, the routes.

Pure functions -- no Redis, no torch, no GPU, no I/O (design §6.1), so the maths is covered by
the repo's own test suite. The class sets live in fall_detection.yaml, never here.
"""
from __future__ import annotations

# The detector decides first, the classifier second (design §6.2). The immobility detector owns
# its own group, so a fall candidate never takes it.
_IMMOBILITY_GROUP = "immobile"

# The route table (design §5.3): a confirmed evaluation's severity. The upright reading is the
# closed false alarm and has no severity.
_CONFIRMED_SEVERITY = {
    "falling": "critical",
    "immobile": "critical",
    "lying": "warning",
    "unstable": "warning",
}
_CLOSED_EVALUATION = "upright"

# The policy's vocabulary: every evaluation the product knows. The classifier's class count is
# the vendor's action_classes.names -- eight.
EVALUATIONS = (*_CONFIRMED_SEVERITY, _CLOSED_EVALUATION)
CLASS_COUNT = 8

# Each span measures one quantity and defaults to the gate's own threshold (ruling A6).
_SPAN_SOURCES = (
    ("drop_span", "drop"),
    ("vdown_span", "vdown"),
    ("hip_span", "low_hip"),
    ("angle_span", "angle"),
    ("aspect_span", "aspect"),
    ("motion_span", "max_motion"),
    ("disp_span", "disp_thresh"),
)


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))


def evaluate(kind: str, top_class: int | None, evaluations: dict) -> str:
    """The evaluation of a candidate.

    kind -- the detector that raised it: "fall", "immobility" or "lying".
    top_class -- the reading's top class, or None when there is no reading yet.

    The detector decides first and the classifier second (design §6.2): the immobility detector
    owns the "immobile" group, so the fall candidate's lookup skips that group and takes the
    first other group holding the reading's top class. A fall candidate with no reading yet
    keeps its detector's own verdict.
    """
    if kind == "lying":
        return "lying"
    if kind == "immobility":
        # a person who is falling is not immobile (design §6.2)
        return "falling" if top_class in evaluations["falling"] else _IMMOBILITY_GROUP
    if top_class is None:
        return "falling"
    for name, classes in evaluations.items():
        if name == _IMMOBILITY_GROUP:
            continue
        if top_class in classes:
            return name
    raise ValueError(f"no evaluation group holds class {top_class}")


def validate_evaluations(evaluations: dict) -> None:
    """The policy must name every evaluation the routes know and cover every class.

    Checked at start-up: a policy that drops a group, invents one, or swallows every class
    would otherwise fail on the first candidate that needs it, in the worker.
    """
    missing = [name for name in EVALUATIONS if name not in evaluations]
    if missing:
        raise ValueError(f"scoring.evaluations is missing {', '.join(missing)}")
    unknown = [name for name in evaluations if name not in EVALUATIONS]
    if unknown:
        raise ValueError(f"scoring.evaluations has no route for {', '.join(unknown)}")
    for name, classes in evaluations.items():
        if not isinstance(classes, list) or not classes or not all(isinstance(c, int) for c in classes):
            raise ValueError(f"scoring.evaluations.{name} must be a non-empty list of class indices")
        # a group that holds every class leaves the margin with no opposing class to measure against
        if len(set(classes)) == CLASS_COUNT:
            raise ValueError(f"scoring.evaluations.{name} cannot hold every class")
    covered = {c for classes in evaluations.values() for c in classes}
    if covered != set(range(CLASS_COUNT)):
        raise ValueError(f"scoring.evaluations must cover the classes 0..{CLASS_COUNT - 1}, "
                         f"not {sorted(covered)}")


def reading_score(probs: list[float], evaluation: str, scoring_cfg: dict) -> float:
    """How decisively the classifier reached this evaluation (design §6.2).

    A classifier that cannot tell Fall from Lying has a margin near zero, and the candidate
    goes to a human instead of being confirmed.
    """
    classes = scoring_cfg["evaluations"][evaluation]
    support = max(probs[c] for c in classes)
    oppose = max(p for c, p in enumerate(probs) if c not in classes)
    return _clip((support - oppose) / scoring_cfg["margin_span"])


def combine(path_scores: dict, trust: dict) -> float:
    """The combination (design §6.3).

    A path that did not speak leaves its trust unclaimed, and nothing is renormalized over the
    paths that did: the trust weights are the paths' ceilings.
    """
    product = 1.0
    for path, score in path_scores.items():
        if score is not None:
            product *= 1.0 - trust[path] * score
    return 1.0 - product


# What each evaluation means, in the words an alert carries (the incident list, Telegram): as a fact when the
# product is sure, hedged when it routed the candidate to a human review.
_STATEMENTS = {
    "falling": ("Person fell", "a person may have fallen"),
    "immobile": ("Person down and not moving", "a person may be down and not moving"),
    "lying": ("Person lying on the floor", "a person may be lying on the floor"),
    "unstable": ("Person unsteady, at risk of falling", "a person may be unsteady"),
    _CLOSED_EVALUATION: ("Person upright again", "a person may have fallen and got up again"),
}


def alert_statement(evaluation: str, combined: float, review: bool) -> str:
    """The alert's message: what happened and how sure the product is. One routed to a human review
    (below the confirm threshold) says the fall is only possible."""
    sure, hedged = _STATEMENTS.get(evaluation, (f"Person {evaluation}", f"a person may be {evaluation}"))
    confidence = f"(confidence {round(combined * 100)}%)"
    # what happened first: a narrow card cuts the end off
    return f"{hedged[0].upper()}{hedged[1:]}: needs review {confidence}" if review else f"{sure} {confidence}"


def route(evaluation: str, combined: float, confirm_threshold: float) -> tuple[str | None, bool]:
    """The route table (design §5.3): the alert's severity and whether a human is queued.

    A confirmed upright reading is a closed false alarm: no alert, nobody queued.
    """
    if combined < confirm_threshold:
        return "warning", True
    if evaluation == _CLOSED_EVALUATION:
        return None, False
    try:
        return _CONFIRMED_SEVERITY[evaluation], False
    except KeyError:
        raise ValueError(f"unknown evaluation {evaluation!r}") from None


def default_spans(gate: dict) -> dict:
    """Each span defaults to the threshold it measures, until the spans are benchmarked."""
    spans = {span: float(gate[source]) for span, source in _SPAN_SOURCES if source in gate}
    # the lying block's gates are the "in" ones: the posture that starts a lying candidate
    if "angle_span" not in spans and "angle_in" in gate:
        spans["angle_span"] = float(gate["angle_in"])
    if "aspect_span" not in spans and "aspect_in" in gate:
        spans["aspect_span"] = float(gate["aspect_in"])
    return spans


def spans_for(block: dict) -> dict:
    """The block's own spans when the config carries them, else the gate's own thresholds."""
    spans = default_spans(block)
    spans.update({name: float(value) for name, value in block.items() if name.endswith("_span")})
    return spans


def collapse_score(kinematics: dict, gate: dict, spans: dict) -> float:
    """The collapse margin over the path's own thresholds (design §4.2).

    The geometric mean is the one aggregate where a single collapsed condition cannot be
    averaged away, which is what the gates' conjunction means.
    """
    s_drop = _clip((kinematics["drop"] - gate["drop"]) / spans["drop_span"])
    s_vdown = _clip((kinematics["vdown"] - gate["vdown"]) / spans["vdown_span"])
    # hip_rel is the hip's height over the box's bottom, so a collapsed person reads low
    s_hip = _clip((gate["low_hip"] - kinematics["hip_rel"]) / spans["hip_span"])
    s_posture = max(_clip((kinematics["angle"] - gate["angle"]) / spans["angle_span"]),
                     _clip((kinematics["aspect"] - gate["aspect"]) / spans["aspect_span"]))
    return (s_drop * s_vdown * s_hip * s_posture) ** 0.25


def fall_path_scores(latch: dict | None, kinematics: dict, vendor_cfg: dict) -> dict:
    """The soft score of each fall path that spoke (design §4.2).

    `latch` is the copy's record of which paths latched (a, b, c); the margin is measured at
    the packaging moment, whose kinematics travel on the candidate. The action slot stays
    None: the action score is the reading's margin, and the margin needs the product's own
    class sets, so the service fills it.
    """
    scores = {"action": None, "kinematic": None, "cold": None}
    if not latch:
        return scores
    for path, slot in (("b", "kinematic"), ("c", "cold")):
        if latch.get(path):
            gate = vendor_cfg["path_b" if path == "b" else "path_c"]
            scores[slot] = collapse_score(kinematics, gate, spans_for(gate))
    return scores


def lying_score(kinematics: dict, cfg: dict, spans: dict) -> float:
    """The posture and the stillness against the vendor's own lying thresholds (design §13.3)."""
    s_posture = _clip(min((kinematics["angle"] - cfg["angle_in"]) / spans["angle_span"],
                          (kinematics["aspect"] - cfg["aspect_in"]) / spans["aspect_span"]))
    s_still = _clip((cfg["max_motion"] - kinematics["motion"]) / spans["motion_span"])
    return (s_posture * s_still) ** 0.5


def stillness_score(disp_win: float, cfg: dict, spans: dict) -> float:
    """The stillness margin against the immobility detector's own threshold (design §4.2)."""
    return _clip((cfg["disp_thresh"] - disp_win) / spans["disp_span"])


def score_candidate(rec: dict, policy: dict, vendor_cfg: dict) -> dict:
    """The candidate's evaluation, its three path scores, the combined score and its route.

    rec -- the candidate as the monitor hands it over: the copy's seam keys (`reading`, the
    smoothed vector; `fall_paths`, which paths latched) and the `path_scores` the monitor
    already computed from the latch (design §4.2).
    """
    scoring_cfg = policy["scoring"]
    reading = rec.get("reading") or {}
    probs = reading.get("probs")
    has_reading = bool(probs)
    evaluation = evaluate(rec.get("type", "fall"), reading.get("top"), scoring_cfg["evaluations"])
    scores = {"action": None, "kinematic": None, "cold": None}
    if has_reading:
        # the action score is the reading's margin, over the product's own class sets (design §6.2)
        scores["action"] = reading_score(probs, evaluation, scoring_cfg)
    physical = _physical_score(rec, vendor_cfg)
    if physical is not None:
        scores[_physical_slot(rec, has_reading)] = physical
    combined = combine(scores, scoring_cfg["trust"])
    severity, review = route(evaluation, combined, scoring_cfg["confirm_threshold"])
    return {"evaluation": evaluation, "scores": scores, "combined": combined,
            "severity": severity, "review": review}


def _physical_slot(rec: dict, has_reading: bool) -> str:
    """Which trust ceiling claims the physical score (design §6.3, §6.4).

    The cold ceiling is the fallback: a candidate with no reading has nothing to corroborate
    it, so it can never reach the threshold. A fall latched by the cold path stays cold even
    when a window appears by packaging time -- the latch is what named the path.
    """
    if not has_reading:
        return "cold"
    if rec.get("type", "fall") == "fall":
        path_scores = rec.get("path_scores") or {}
        return "kinematic" if path_scores.get("kinematic") is not None else "cold"
    # the product-computed candidates (lying, immobility): the window is the only rule
    return "kinematic"


def _physical_score(rec: dict, vendor_cfg: dict) -> float | None:
    """The candidate's physical score.

    The fall candidate carries the vendor's soft margin over the path's own gate, measured at
    the packaging (design §4.2). The vendor never scores its lying and immobility candidates,
    so the product measures them from the kinematics of the emission moment (ruling A7,
    design §13.3) -- the same moment, the same quantities.
    """
    kind = rec.get("type", "fall")
    if kind == "lying":
        cfg = vendor_cfg["lying"]
        return lying_score(rec["evidence"]["kinematics"], cfg, spans_for(cfg))
    if kind == "immobility":
        cfg = vendor_cfg["immobility"]
        return stillness_score(rec["evidence"]["kinematics"]["disp_win"], cfg, spans_for(cfg))
    path_scores = rec.get("path_scores") or {}
    return path_scores.get("kinematic") if path_scores.get("kinematic") is not None \
        else path_scores.get("cold")
