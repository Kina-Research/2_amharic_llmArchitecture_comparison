"""Significance testing and confidence intervals (spec §4).

  * Percentile bootstrap CIs for any metric (seeded).
  * McNemar's test (exact binomial if discordant < 25, else χ² with continuity correction)
    for paired classifier accuracy.
  * Paired bootstrap test for differences in a corpus metric (e.g. macro-F1).
  * Wilcoxon signed-rank on per-document NLG scores.
  * Holm–Bonferroni correction across the family of pairwise comparisons.
"""
from __future__ import annotations

from itertools import combinations
from typing import Callable

import numpy as np
from scipy import stats


def bootstrap_ci(metric: Callable[[np.ndarray], float], n: int, n_boot: int = 1000,
                 level: float = 0.95, seed: int = 42) -> dict:
    """metric(idx) -> float, evaluated on resampled index arrays."""
    rng = np.random.default_rng(seed)
    point = metric(np.arange(n))
    boots = np.array([metric(rng.integers(0, n, n)) for _ in range(n_boot)])
    a = (1 - level) / 2
    return {"point": float(point), "lo": float(np.quantile(boots, a)),
            "hi": float(np.quantile(boots, 1 - a)), "std": float(boots.std(ddof=1))}


def mcnemar(y_true, pred_a, pred_b) -> dict:
    y, a, b = map(np.asarray, (y_true, pred_a, pred_b))
    ca, cb = a == y, b == y
    n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    n = n01 + n10
    if n == 0:
        return {"n01": n01, "n10": n10, "statistic": 0.0, "p_value": 1.0, "method": "degenerate"}
    if n < 25:
        p = float(stats.binomtest(n01, n, 0.5).pvalue)
        return {"n01": n01, "n10": n10, "statistic": float(n01), "p_value": p, "method": "exact"}
    chi2 = (abs(n01 - n10) - 1) ** 2 / n
    return {"n01": n01, "n10": n10, "statistic": float(chi2),
            "p_value": float(stats.chi2.sf(chi2, 1)), "method": "chi2_cc"}


def paired_bootstrap(metric: Callable[[np.ndarray, str], float], n: int, n_boot: int = 1000,
                     seed: int = 42) -> dict:
    """metric(idx, which) with which in {"a","b"}. Two-sided p for H0: Δ = 0."""
    rng = np.random.default_rng(seed)
    full = np.arange(n)
    delta = metric(full, "a") - metric(full, "b")
    ds = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        ds[i] = metric(idx, "a") - metric(idx, "b")
    # Shift-to-null method (Koehn 2004 / Berg-Kirkpatrick et al. 2012 style).
    p = float((np.abs(ds - delta) >= abs(delta)).mean())
    return {"delta": float(delta), "lo": float(np.quantile(ds, 0.025)),
            "hi": float(np.quantile(ds, 0.975)), "p_value": max(p, 1.0 / n_boot)}


def wilcoxon(scores_a, scores_b) -> dict:
    a, b = np.asarray(scores_a), np.asarray(scores_b)
    if np.allclose(a, b):
        return {"statistic": 0.0, "p_value": 1.0}
    r = stats.wilcoxon(a, b, zero_method="wilcox")
    return {"statistic": float(r.statistic), "p_value": float(r.pvalue)}


def holm(pvals: dict[str, float]) -> dict[str, float]:
    keys = sorted(pvals, key=pvals.get)
    m, adj, running = len(keys), {}, 0.0
    for i, k in enumerate(keys):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        adj[k] = running
    return adj


def pairwise(keys: list[str]) -> list[tuple[str, str]]:
    return list(combinations(keys, 2))

