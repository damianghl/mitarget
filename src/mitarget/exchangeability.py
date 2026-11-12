"""
Tests of exchangeability: do levels of different sizes share one prior for ``q``?

Under exchangeable conditionals (``q_x`` independent of the popularity), the
law of ``q`` among levels seen ``n`` times is the same for every ``n``. On a
level seen ``n`` times with ``k`` ones, ``k / n`` is unbiased for ``m_1 = E q``
and ``k (k - 1) / (n (n - 1))`` for ``m_2 = E q^2``, so under the null every
stratum of level sizes estimates the same ``(m_1, m_2)``.
:func:`stratum_moment_test` compares the strata with a Wald (GLS) test; the
frequency-aware prior's likelihood-ratio test (:func:`mitarget.fit_freq_prior`)
is a parametric alternative with power against a mean that drifts with
``log n``.

Both concern the level-weighted laws ``pi^lev_n`` of the paper, which is what
the mutual-information intervals assume equal across ``n``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import chi2

#: default strata of level sizes (inclusive bounds)
STRATA = ((1, 1), (2, 2), (3, 3), (4, 4), (5, 6), (7, 10), (11, 20), (21, 50), (51, 100), (101, np.inf))


@dataclass
class ExchangeabilityTest:
    statistic: float
    df: int
    p_value: float
    pooled: np.ndarray          # GLS estimate of (m1, m2) under the null
    strata: pd.DataFrame        # per stratum: size range, levels, m1, se_m1, m2, se_m2
    max_dev_m1: float           # largest |m1 stratum - pooled m1| (effect size)


def stratum_moment_test(n: np.ndarray, strata=STRATA, min_levels: int = 30,
                        moments: int = 2) -> ExchangeabilityTest:
    """
    Wald test that every stratum of level sizes has the same first ``moments``
    (1 or 2) moments of ``q``, for a binary ``(kx, 2)`` count matrix.

    Strata with fewer than ``min_levels`` levels are dropped. The second moment
    enters only for strata of levels seen at least twice. Within a stratum the
    per-level estimates are i.i.d. under the null, so their sample covariance
    divided by the number of levels is the stratum's covariance; strata are
    independent. The statistic is the GLS residual quadratic form, chi-square
    with ``len(theta) - moments`` degrees of freedom.
    """
    n = np.asarray(n, dtype=float)
    if n.ndim != 2 or n.shape[1] != 2:
        raise ValueError("stratum_moment_test needs a binary target: a (kx, 2) count matrix")
    if moments not in (1, 2):
        raise ValueError("moments must be 1 or 2")
    k, nx = n[:, 1], n.sum(axis=1)
    theta, blocks, design, rows = [], [], [], []
    for lo, hi in strata:
        sel = (nx >= lo) & (nx <= hi)
        L = int(sel.sum())
        if L < min_levels:
            continue
        a = k[sel] / nx[sel]
        stats = [a]
        if moments == 2 and lo >= 2:
            stats.append(k[sel] * (k[sel] - 1) / (nx[sel] * (nx[sel] - 1)))
        X = np.array(stats).T
        mean = X.mean(axis=0)
        cov = np.atleast_2d(np.cov(X, rowvar=False, ddof=1)) / L
        cov += np.eye(len(mean)) * 1e-12
        theta.append(mean)
        blocks.append(cov)
        design.append(np.eye(moments)[: len(mean)])
        r = dict(n_lo=lo, n_hi=hi, levels=L, m1=mean[0], se_m1=float(np.sqrt(cov[0, 0])))
        if len(mean) == 2:
            r.update(m2=mean[1], se_m2=float(np.sqrt(cov[1, 1])))
        rows.append(r)
    table = pd.DataFrame(rows)
    if len(theta) < 2:
        nan = float("nan")
        return ExchangeabilityTest(nan, 0, nan, np.full(moments, np.nan), table, nan)
    t = np.concatenate(theta)
    X = np.vstack(design)
    W = np.zeros((len(t), len(t)))
    i = 0
    for b in blocks:
        j = i + len(b)
        W[i:j, i:j] = np.linalg.inv(b)
        i = j
    est_cols = np.flatnonzero(X.any(axis=0))       # m2 is absent when only singleton-heavy strata qualify
    Xr = X[:, est_cols]
    beta = np.linalg.solve(Xr.T @ W @ Xr, Xr.T @ W @ t)
    res = t - Xr @ beta
    stat = float(res @ W @ res)
    df = len(t) - len(est_cols)
    pooled = np.full(moments, np.nan)
    pooled[est_cols] = beta
    return ExchangeabilityTest(stat, df, float(chi2.sf(stat, df)), pooled, table,
                               float(np.max(np.abs(table.m1 - pooled[0]))))
