"""Same person back in the chair? (2026-10-07, ce_resume_check; dev/chairrun/same_person/FINDINGS.md).

A chair taken again within ce_resume_max_s of its haircut's end compares the new sitter's re-ID (every
sample from the moment he sat down -- the start chain included -- until the decision, >= ce_resume_new_s)
with the previous customer's (every sample of his last ce_resume_ref_s in the chair before he left): the
cosine of the two mean embeddings. Two levels, on the gap since the previous sitting ended:
  gap <= ce_resume_window_s: the same customer back from the wash, unless sim < ce_resume_sim_low;
  gap <= ce_resume_max_s:    the same customer only if sim >= ce_resume_sim_high;
  longer:                    a new customer.
No sim (no samples on either side) falls back to the time rule.
2026-10-07 (evening, review finding 1): the mean of 3 picked samples swung 0.60-0.76 on one real return
over four live laps (0.60 split it). Offline, bekky-50min pairs rebuilt with every crop: last 60 s +
all samples -> AUC 0.986 (3 picked, last 300 s: 0.971), 7/7 real for low 0.575-0.70 x high 0.70-0.95,
lowest real return 0.71, highest new sitter within 6 min 0.55; built pairs at 0.62 / 0.82: no same
pair below low, 1.6 % of different pairs above high. One shop, two recordings."""
from __future__ import annotations

import numpy as np

Sample = tuple[float, tuple[float, ...]]     # (ts, unit re-ID embedding)
WAIT_EXTRA_S = 120.0                         # fewer than 2 new samples: wait up to this past ce_resume_new_s
KEEP_S = 300.0                               # how much of a sitter's re-ID is buffered (ref window <= this)


def mean_unit(samples: list[Sample]) -> np.ndarray | None:
    if not samples:
        return None
    m = np.mean(np.asarray([r for _, r in samples], np.float32), axis=0)
    n = float(np.linalg.norm(m))
    return None if n == 0.0 else m / n


def similarity(ref: list[Sample], new: list[Sample]) -> float | None:
    a, b = mean_unit(ref), mean_unit(new)
    return None if a is None or b is None else float(a @ b)


def ref_window(reid: list[Sample], start: float, end: float, ref_s: float) -> list[Sample]:
    """His samples in [max(start, end - ref_s), end]; fewer than 2 there: his last 2 before `end`."""
    a = max(start, end - ref_s)
    win = [r for r in reid if a <= r[0] <= end]
    return win if len(win) >= 2 else [r for r in reid if r[0] <= end][-2:]


def threshold(gap_s: float, policy: dict) -> float | None:
    """The similarity decide() holds the sitter to after this gap; None past ce_resume_max_s (no check)."""
    if gap_s <= policy["ce_resume_window_s"]:
        return policy["ce_resume_sim_low"]
    if gap_s <= policy["ce_resume_max_s"]:
        return policy["ce_resume_sim_high"]
    return None


def decide(gap_s: float, sim: float | None, policy: dict) -> bool:
    """True: the previous customer (resume his haircut); False: a new customer."""
    if gap_s <= policy["ce_resume_window_s"]:
        return sim is None or sim >= policy["ce_resume_sim_low"]
    if gap_s <= policy["ce_resume_max_s"]:
        return sim is not None and sim >= policy["ce_resume_sim_high"]
    return False
