"""
Moment-identified interval for ``I(X;Y)`` with a binary target.

Under exchangeable conditionals (``q_x`` i.i.d. from a prior ``pi``, independent
of popularity) and many levels,

    I(X;Y) = h(mu) - E_pi[h(q)],      h = binary entropy (nats), mu = E_pi[q].

A level seen ``n`` times reveals moments of ``pi`` only up to order ``n``, so
levels seen at most ``L`` times identify ``m_1..m_L`` and ``I`` only up to an
interval: the range of ``h(m_1) - E h`` over all priors with those moments.
Singletons alone give ``[0, h(mu)]``; the worst-case half-width is the best
uniform degree-``L`` polynomial approximation error of ``h``, about ``0.22 / L^2``.

- :func:`moment_estimates`: pooled U-statistics ``m_j = mean_x C(k_x, j) / C(n_x, j)``
  over levels with ``n_x >= j``, with standard errors.
- :func:`identified_interval`: the population interval given exact moments
  (two linear programmes over priors on a grid).
- :func:`mi_interval`: the interval from data, with the moments allowed to vary
  within ``slack`` standard errors (a conservative box, not a joint region).
- :func:`worst_case_halfwidth`: ``E_L(h)``.
- :func:`credible_interval`: the posterior mean and standard deviation of ``I`` under the
  beta prior of Hernandez & Samengo (2019, Appendix A), for comparison.

The beta / Dirichlet estimators return one point inside this interval; reporting
both shows how much of the point estimate comes from the beta assumption.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.stats import norm
from scipy.special import digamma, entr, logsumexp, polygamma

from .core import fit_beta_star, per_state_log_likelihood, qhat_y

#: Default grid of support points for the prior of q.
GRID_SIZE = 2001


def binary_entropy(q) -> np.ndarray:
    q = np.clip(np.asarray(q, dtype=float), 0.0, 1.0)
    return entr(q) + entr(1 - q)


def _binary_kn(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = np.asarray(n, dtype=float)
    if n.ndim != 2 or n.shape[1] != 2:
        raise ValueError("the MI interval needs a binary target: a (kx, 2) count matrix")
    return n[:, 1], n.sum(axis=1)


def moment_estimates(n: np.ndarray, L: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Unbiased estimates of ``m_j = E_pi[q^j]``, ``j = 1..L``, from a ``(kx, 2)`` count matrix.

    Returns ``(m, se, n_levels)``: the estimates, their standard errors and the number of
    levels each uses (those with ``n_x >= j``). ``C(k, j) / C(n, j)`` is computed as
    ``prod_{i<j} (k - i) / (n - i)``.
    """
    k, nx = _binary_kn(n)
    m, se, used = np.full(L, np.nan), np.full(L, np.nan), np.zeros(L, dtype=int)
    for j in range(1, L + 1):
        sel = nx >= j
        used[j - 1] = int(sel.sum())
        if used[j - 1] == 0:
            continue
        u = np.ones(used[j - 1])
        for i in range(j):
            u *= np.clip(k[sel] - i, 0, None) / (nx[sel] - i)
        m[j - 1] = u.mean()
        se[j - 1] = u.std(ddof=1) / np.sqrt(used[j - 1]) if used[j - 1] > 1 else np.inf
    return m, se, used


def _eh_range(A_eq, b_eq, A_ub, b_ub, grid):
    """min and max of ``sum_i w_i h(grid_i)`` over histograms ``w >= 0`` with the given constraints."""
    hg = binary_entropy(grid)
    out = []
    for sign in (1.0, -1.0):
        r = linprog(sign * hg, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0, None),
                    method="highs")
        if r.status != 0:
            return np.nan, np.nan
        out.append(sign * r.fun)
    return out[0], out[1]


def _symmetry_rows(G):
    S = np.zeros((G // 2, G))
    i = np.arange(G // 2)
    S[i, i], S[i, G - 1 - i] = 1.0, -1.0
    return S


def identified_interval(moments, symmetric: bool = False,
                        grid_size: int = GRID_SIZE) -> tuple[float, float]:
    """
    ``[I_lo, I_hi]`` over all priors on ``[0, 1]`` whose first ``L`` moments equal
    ``moments``; ``symmetric=True`` also imposes ``pi(q) = pi(1 - q)``.
    """
    moments = np.atleast_1d(np.asarray(moments, dtype=float))
    grid = np.linspace(0.0, 1.0, grid_size)
    A = np.vstack([np.ones_like(grid)] + [grid ** j for j in range(1, len(moments) + 1)])
    b = np.concatenate([[1.0], moments])
    if symmetric:
        S = _symmetry_rows(grid_size)
        A, b = np.vstack([A, S]), np.concatenate([b, np.zeros(len(S))])
    lo, hi = _eh_range(A, b, None, None, grid)
    hmu = float(binary_entropy(moments[0]))
    return hmu - hi, hmu - lo


@dataclass
class MIInterval:
    lower: float
    upper: float
    L: int                  # number of moments used
    moments: np.ndarray     # (L,) estimated m_1..m_L
    se: np.ndarray          # (L,) standard errors
    n_levels: np.ndarray    # (L,) levels with n_x >= j

    @property
    def midpoint(self) -> float:
        """The minimax point estimate (centre of the interval)."""
        return 0.5 * (self.lower + self.upper)

    @property
    def width(self) -> float:
        return self.upper - self.lower


def mi_interval(n: np.ndarray, L: int | None = None, slack: float = 2.0, max_L: int = 4,
                min_levels: int = 100, grid_size: int = GRID_SIZE, n_mu: int = 21) -> MIInterval:
    """
    Interval for ``I(X;Y)`` from a ``(kx, 2)`` count matrix.

    The moments ``m_1..m_L`` may vary within ``slack`` standard errors of their estimates
    (``slack=0`` gives the identified interval at the plug-in moments). For each of
    ``n_mu`` values of ``m_1`` in its box, ``E h`` is minimized and maximized by linear
    programming, and the interval is the range of ``h(m_1) - E h`` over them.

    ``L=None`` picks the largest ``L <= max_L`` with at least ``min_levels`` levels seen
    ``L`` or more times (at least 1). Higher moments rest on fewer levels, so their noise
    soon outweighs what they add.
    """
    k, nx = _binary_kn(n)
    if L is None:
        L = max([1] + [j for j in range(1, max_L + 1) if np.sum(nx >= j) >= min_levels])
    m, se, used = moment_estimates(n, L)
    if np.isnan(m).any():
        raise ValueError(f"no level is seen {L} times; use a smaller L")
    se = np.where(np.isfinite(se), se, 0.0)
    half = slack * se
    grid = np.linspace(0.0, 1.0, grid_size)
    powers = np.vstack([grid ** j for j in range(1, L + 1)])
    A_eq = np.vstack([np.ones_like(grid), grid])
    A_ub = np.vstack([powers[1:], -powers[1:]]) if L > 1 else None
    b_ub = np.concatenate([m[1:] + half[1:], -(m[1:] - half[1:])]) if L > 1 else None
    mu_lo, mu_hi = max(m[0] - half[0], 0.0), min(m[0] + half[0], 1.0)
    lows, highs = [], []
    for mu in np.unique(np.linspace(mu_lo, mu_hi, n_mu if half[0] > 0 else 1)):
        lo, hi = _eh_range(A_eq, np.array([1.0, mu]), A_ub, b_ub, grid)
        if np.isnan(lo):
            continue
        hmu = float(binary_entropy(mu))
        lows.append(hmu - hi)
        highs.append(hmu - lo)
    if not lows:
        raise ValueError("the estimated moments admit no prior; increase slack")
    return MIInterval(max(min(lows), 0.0), max(highs), L, m, se, used)


def worst_case_halfwidth(L: int, grid_size: int = GRID_SIZE) -> float:
    """
    ``E_L(h)``: best uniform degree-``L`` polynomial approximation error of the binary
    entropy on ``[0, 1]``, the worst-case half-width of the identified interval given
    ``L`` moments.
    """
    grid = np.linspace(0.0, 1.0, grid_size)
    V = np.polynomial.chebyshev.chebvander(2 * grid - 1, L)
    hv = binary_entropy(grid)
    ones = np.ones((grid_size, 1))
    c = np.zeros(L + 2)
    c[-1] = 1.0
    A = np.vstack([np.hstack([V, -ones]), np.hstack([-V, -ones])])
    r = linprog(c, A_ub=A, b_ub=np.concatenate([hv, -hv]),
                bounds=[(None, None)] * (L + 1) + [(0, None)], method="highs")
    return float(r.x[-1])


@dataclass
class CredibleInterval:
    mean: float
    sd: float
    level: float
    beta_star: float
    sd_beta: float          # part of the variance from the posterior of beta
    sd_levels: float        # part from the conditionals at fixed beta

    @property
    def lower(self) -> float:
        return self.mean - norm.ppf(0.5 + self.level / 2) * self.sd

    @property
    def upper(self) -> float:
        return self.mean + norm.ppf(0.5 + self.level / 2) * self.sd


def _beta_h_moments(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``E h(q)`` and ``E h(q)^2`` for ``q ~ Beta(a, b)`` (Hernandez & Samengo 2019, Eqs. A5, A7-A8)."""
    s = a + b
    eh = digamma(s + 1) - (a * digamma(a + 1) + b * digamma(b + 1)) / s
    d2 = digamma(s + 2)
    t1 = polygamma(1, s + 2)
    c = 1.0 / (s * (s + 1))

    def g(z):
        return (digamma(z + 2) - d2) ** 2 + polygamma(1, z + 2) - t1

    cross = (digamma(a + 1) - d2) * (digamma(b + 1) - d2) - t1
    eh2 = c * (a * (a + 1) * g(a) + b * (b + 1) * g(b) + 2 * a * b * cross)
    return eh, eh2


def credible_interval(n: np.ndarray, level: float = 0.95, n_grid: int = 41,
                      width_sd: float = 6.0) -> CredibleInterval:
    """
    Posterior mean and standard deviation of ``I(X;Y)`` under ``q_x ~ Beta(beta * mu)``,
    following Appendix A of Hernandez & Samengo (2019), and the normal interval
    ``mean +- z sd``.

    The estimand is ``h(mu_hat) - sum_x (n_x / N) h(q_x)`` over the seen levels, with
    ``mu`` fixed at the empirical label rate. The posterior of ``log beta`` (flat prior)
    is evaluated on ``n_grid`` points within ``width_sd`` Laplace standard deviations of
    ``beta*``; the variance is the variance of ``I(beta)`` over that posterior plus the
    mean of ``sum_x (n_x / N)^2 Var[h(q_x) | n_x, beta]``.
    """
    k, nx = _binary_kn(n)
    n = np.asarray(n, dtype=float)[nx > 0]
    k, nx = k[nx > 0], nx[nx > 0]
    qy = qhat_y(n)
    N = nx.sum()
    pairs, inv, mult = np.unique(np.column_stack([nx - k, k]), axis=0, return_inverse=True,
                                 return_counts=True)
    w = pairs.sum(axis=1) / N
    beta_star = fit_beta_star(n, qy)

    def loglik(lb):
        return float(mult @ per_state_log_likelihood(np.exp(lb), pairs, qy))

    lb0, eps = np.log(beta_star), 1e-3
    curv = -(loglik(lb0 + eps) - 2 * loglik(lb0) + loglik(lb0 - eps)) / eps ** 2
    half = width_sd / np.sqrt(curv) if curv > 0 else 3.0
    lbs = np.linspace(lb0 - half, lb0 + half, n_grid)
    logpost = np.array([loglik(lb) for lb in lbs])
    post = np.exp(logpost - logsumexp(logpost))
    hy = float(binary_entropy(qy[1]))
    I_b, var_b = np.empty(n_grid), np.empty(n_grid)
    for i, lb in enumerate(lbs):
        a0, a1 = np.exp(lb) * qy[0] + pairs[:, 0], np.exp(lb) * qy[1] + pairs[:, 1]
        eh, eh2 = _beta_h_moments(a0, a1)
        I_b[i] = hy - float(mult @ (w * eh))
        var_b[i] = float(mult @ (w ** 2 * (eh2 - eh ** 2)))
    mean = float(post @ I_b)
    v_beta = float(post @ (I_b - mean) ** 2)
    v_levels = float(post @ var_b)
    return CredibleInterval(mean, float(np.sqrt(v_beta + v_levels)), level, beta_star,
                            float(np.sqrt(v_beta)), float(np.sqrt(v_levels)))
