"""Linear sum assignment (Hungarian / Jonker-Volgenant) with automatic pure-NumPy fallback."""

from __future__ import annotations

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment as _scipy_lsa
except ImportError:  # pragma: no cover
    _scipy_lsa = None


def _hungarian_numpy(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact O(n^3) shortest-augmenting-path Hungarian solver for rectangular cost matrices."""
    cost = np.asarray(cost, dtype=np.float64)
    n_rows, n_cols = cost.shape
    if n_rows == 0 or n_cols == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    transposed = False
    if n_rows > n_cols:
        cost = cost.T
        n_rows, n_cols = n_cols, n_rows
        transposed = True

    u = np.zeros(n_rows + 1, dtype=np.float64)
    v = np.zeros(n_cols + 1, dtype=np.float64)
    p = np.zeros(n_cols + 1, dtype=np.int64)
    way = np.zeros(n_cols + 1, dtype=np.int64)

    inf = 1e18
    for i in range(1, n_rows + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n_cols + 1, inf, dtype=np.float64)
        used = np.zeros(n_cols + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = inf
            j1 = 0
            row_cost = cost[i0 - 1]
            for j in range(1, n_cols + 1):
                if not used[j]:
                    cur = row_cost[j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n_cols + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    row_ind = np.empty(n_rows, dtype=np.int64)
    col_ind = np.empty(n_rows, dtype=np.int64)
    k = 0
    for j in range(1, n_cols + 1):
        if p[j] > 0:
            row_ind[k] = p[j] - 1
            col_ind[k] = j - 1
            k += 1

    if transposed:
        row_ind, col_ind = col_ind, row_ind
    order = np.argsort(row_ind)
    return row_ind[order], col_ind[order]


def linear_assignment(
    cost_matrix: np.ndarray,
    cost_limit: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Solve min-cost assignment and reject pairs with cost > cost_limit."""
    cost = np.asarray(cost_matrix, dtype=np.float64)
    n_rows, n_cols = cost.shape
    if n_rows == 0 or n_cols == 0:
        return [], list(range(n_rows)), list(range(n_cols))

    sentinel = max(float(cost_limit) * 100.0 + 1e4, 1e6)
    safe_cost = np.where(np.isfinite(cost), cost, sentinel)

    if _scipy_lsa is not None:
        row_ind, col_ind = _scipy_lsa(safe_cost)
    else:
        row_ind, col_ind = _hungarian_numpy(safe_cost)

    matches: list[tuple[int, int]] = []
    matched_rows: set[int] = set()
    matched_cols: set[int] = set()
    for r, c in zip(row_ind.tolist(), col_ind.tolist()):
        if np.isfinite(cost[r, c]) and cost[r, c] <= cost_limit:
            matches.append((r, c))
            matched_rows.add(r)
            matched_cols.add(c)

    unmatched_rows = [r for r in range(n_rows) if r not in matched_rows]
    unmatched_cols = [c for c in range(n_cols) if c not in matched_cols]
    return matches, unmatched_rows, unmatched_cols
