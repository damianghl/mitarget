"""
Competing encoders from the target-encoding literature, on the same count
matrix interface as the rest of the package.

- :func:`fit_glmm`: random-intercept logistic GLMM, ``logit q_x = b0 + u_x``
  with ``u_x ~ N(0, sigma^2)``, fit by marginal likelihood (adaptive
  Gauss-Hermite quadrature, or lme4's default Laplace approximation); the
  encoding is ``expit(b0 + u_x)`` at the conditional modes. This is the
  encoder Pargent et al. (2022) found best overall.
- :func:`fit_npmle`: Kiefer-Wolfowitz nonparametric maximum-likelihood prior
  for ``q_x`` on a grid (binomial mixture, fit by EM); the encoding is the
  posterior mean. It estimates the whole prior instead of a beta's two moments.
- :func:`catboost_ts_encoding` and :func:`ordered_target_statistics`: CatBoost
  target statistics (Prokhorenkova et al. 2018). At test time a category is
  encoded with all training rows, ``(n_xy + a p_y) / (n_x + a)``, i.e. fixed-m
  smoothing with ``m = a = 1``; the ordered statistic only changes how
  *training* rows are encoded for a downstream model.
- :func:`catboost_classifier_encoding`: the full CatBoost classifier on the
  single feature (optional ``catboost`` dependency).

GLMM and NPMLE are binary models; for ``ky > 2`` classes they are fit
one-vs-rest and the class probabilities renormalized, as Pargent et al. do.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import entr, expit, logit, logsumexp
from scipy.stats import binom

from .core import entropy, posterior_mean_conditional, qhat_y

#: Search interval for log(sigma) of the GLMM random intercept.
LOG_SIGMA_BOUNDS: tuple[float, float] = (-7.0, 4.0)


def _binary_groups(k1: np.ndarray, m: np.ndarray):
    """Unique ``(k1, m)`` pairs among seen categories, their multiplicities and the inverse map."""
    seen = m > 0
    pairs, inverse, mult = np.unique(np.stack([k1[seen], m[seen]], axis=1), axis=0,
                                     return_inverse=True, return_counts=True)
    return seen, pairs[:, 0], pairs[:, 1], mult.astype(float), inverse.ravel()


def _one_vs_rest(n: np.ndarray, fit_binary):
    """Apply a binary fit to each class against the rest; return rows renormalized and the fits."""
    nx = n.sum(axis=1)
    fits = [fit_binary(n[:, c], nx) for c in range(n.shape[1])]
    enc = np.stack([f[0] for f in fits], axis=1)
    unseen = np.array([f[1] for f in fits])
    return enc / enc.sum(axis=1, keepdims=True), unseen / unseen.sum(), [f[2] for f in fits]


# --- GLMM -------------------------------------------------------------------


@dataclass
class GLMMFit:
    intercept: float | np.ndarray   # b0 (one per class for ky > 2)
    sigma: float | np.ndarray       # random-intercept standard deviation
    qy: np.ndarray                  # prediction for unseen categories
    target_encoding: np.ndarray     # (kx, ky)


def _conditional_modes(b0, sigma, k1, m, n_iter=200):
    """
    Newton for the mode of each level's random effect. The log-posterior is
    concave, but a full step can overshoot into the flat logistic tail, so
    steps are capped at 1 on the logit scale.
    """
    s2 = sigma ** 2
    u = np.zeros_like(k1, dtype=float)
    for _ in range(n_iter):
        p = expit(b0 + u)
        hess = m * p * (1 - p) + 1 / s2
        step = np.clip((k1 - m * p - u / s2) / hess, -1.0, 1.0)
        u += step
        if np.max(np.abs(step)) < 1e-10:
            break
    p = expit(b0 + u)
    return u, m * p * (1 - p) + 1 / s2


#: Standardized grid for levels whose posterior is wide on the prior scale.
_Z_GRID = np.linspace(-8.0, 8.0, 1601)


def _glmm_loglik(b0, sigma, k1, m, mult, n_agq):
    """
    Marginal log-likelihood of the random-intercept model. Levels whose
    posterior is narrow compared with the prior (informative data) use
    ``n_agq`` adaptive Gauss-Hermite nodes around their mode (``n_agq = 1``
    is the Laplace approximation); the rest, typically small all-0 or all-1
    levels at large ``sigma``, where the posterior is one-sided, are
    integrated on a dense grid of the prior.
    """
    u, hess = _conditional_modes(b0, sigma, k1, m)
    z, w = np.polynomial.hermite.hermgauss(n_agq)
    scale = np.sqrt(2 / hess)[:, None]
    uu = u[:, None] + scale * z[None, :]
    eta = b0 + uu
    g = (k1[:, None] * eta - m[:, None] * np.logaddexp(0, eta) - uu ** 2 / (2 * sigma ** 2)
         - 0.5 * np.log(2 * np.pi * sigma ** 2) + z[None, :] ** 2)
    ll = logsumexp(g, b=w[None, :], axis=1) + np.log(scale[:, 0])
    wide = scale[:, 0] > 0.15 * sigma
    if n_agq > 1 and wide.any():
        eta = b0 + sigma * _Z_GRID[None, :]
        g = (k1[wide, None] * eta - m[wide, None] * np.logaddexp(0, eta) - _Z_GRID[None, :] ** 2 / 2
             - 0.5 * np.log(2 * np.pi))
        ll[wide] = logsumexp(g, axis=1) + np.log(_Z_GRID[1] - _Z_GRID[0])
    return float(np.sum(mult * ll))


def _fit_glmm_binary(k1: np.ndarray, m: np.ndarray, n_agq: int = 1):
    seen, gk1, gm, mult, inverse = _binary_groups(k1, m)
    rate = np.clip(k1.sum() / m.sum(), 1e-6, 1 - 1e-6)

    def objective(theta):
        return -_glmm_loglik(theta[0], np.exp(theta[1]), gk1, gm, mult, n_agq)

    res = minimize(objective, x0=[float(logit(rate)), 0.0], method="L-BFGS-B",
                   bounds=[(-30, 30), LOG_SIGMA_BOUNDS])
    b0, sigma = float(res.x[0]), float(np.exp(res.x[1]))
    u, _ = _conditional_modes(b0, sigma, gk1, gm)
    p = np.full(len(k1), expit(b0))
    p[seen] = expit(b0 + u[inverse])
    return p, float(expit(b0)), (b0, sigma)


def fit_glmm(n: np.ndarray, n_agq: int = 15) -> GLMMFit:
    """
    Random-intercept logistic GLMM on a ``(kx, ky)`` count matrix (one-vs-rest
    for ``ky > 2``). The marginal likelihood is integrated numerically (see
    :func:`_glmm_loglik`); ``n_agq=1`` is lme4's default Laplace
    approximation instead, which with mostly single-observation levels can
    pick a spuriously large ``sigma``.
    """
    n = np.asarray(n, dtype=float)

    def fit_binary(k1, m):
        return _fit_glmm_binary(k1, m, n_agq=n_agq)

    if n.shape[1] == 2:
        p1, unseen, (b0, sigma) = fit_binary(n[:, 1], n.sum(axis=1))
        return GLMMFit(b0, sigma, np.array([1 - unseen, unseen]), np.stack([1 - p1, p1], axis=1))
    enc, unseen, params = _one_vs_rest(n, fit_binary)
    return GLMMFit(np.array([b for b, _ in params]), np.array([s for _, s in params]), unseen, enc)


# --- NPMLE ------------------------------------------------------------------


@dataclass
class NPMLEFit:
    grid: np.ndarray                # support points for q (binary: P(y = 1 | x))
    weights: np.ndarray | list      # prior mass on the grid (one array per class for ky > 2)
    qy: np.ndarray                  # prediction for unseen categories (prior mean)
    target_encoding: np.ndarray     # (kx, ky)
    I_hat: float                    # posterior-mean MI (NaN for ky > 2)
    log_likelihood: float | None


def _fit_npmle_binary(k1, m, grid, max_iter=5000, tol=1e-10):
    seen, gk1, gm, mult, inverse = _binary_groups(k1, m)
    loglik = binom.logpmf(gk1[:, None], gm[:, None], grid[None, :])
    shift = loglik.max(axis=1, keepdims=True)
    lik = np.exp(loglik - shift)
    w = np.full(len(grid), 1.0 / len(grid))
    total = mult.sum()
    prev = -np.inf
    for _ in range(max_iter):
        mix = lik @ w
        ll = float(np.sum(mult * np.log(mix)))
        w = w * (lik.T @ (mult / mix)) / total
        if ll - prev < tol * max(1.0, abs(ll)):
            break
        prev = ll
    post = lik * w[None, :]
    post /= post.sum(axis=1, keepdims=True)
    prior_mean = float(w @ grid)
    p = np.full(len(k1), prior_mean)
    p[seen] = (post @ grid)[inverse]
    h_grid = entr(grid) + entr(1 - grid)
    h_post = np.zeros(len(k1))
    h_post[seen] = (post @ h_grid)[inverse]
    return p, prior_mean, (w, h_post, ll + float(np.sum(mult * shift[:, 0])))


def fit_npmle(n: np.ndarray, grid_size: int = 400, max_iter: int = 5000) -> NPMLEFit:
    """
    Kiefer-Wolfowitz NPMLE of the prior of ``q_x`` on ``grid_size`` equispaced
    points in [0, 1], and the posterior-mean encoding. The log-likelihood
    omits the binomial coefficients. For ``ky > 2`` it is fit one-vs-rest and
    ``I_hat`` is NaN.
    """
    n = np.asarray(n, dtype=float)
    grid = np.linspace(0.0, 1.0, grid_size)

    def fit_binary(k1, m):
        return _fit_npmle_binary(k1, m, grid, max_iter=max_iter)

    if n.shape[1] == 2:
        nx = n.sum(axis=1)
        p1, unseen, (w, h_post, ll) = fit_binary(n[:, 1], nx)
        I_hat = entropy(qhat_y(n)) - float(np.sum(nx / nx.sum() * h_post))
        return NPMLEFit(grid, w, np.array([1 - unseen, unseen]), np.stack([1 - p1, p1], axis=1),
                        I_hat, ll)
    enc, unseen, params = _one_vs_rest(n, fit_binary)
    return NPMLEFit(grid, [w for w, _, _ in params], unseen, enc, float("nan"), None)


# --- CatBoost ---------------------------------------------------------------


def catboost_ts_encoding(n: np.ndarray, a: float = 1.0, prior: np.ndarray | None = None) -> np.ndarray:
    """
    CatBoost target statistic of a category computed with all training rows,
    ``(n_xy + a * prior_y) / (n_x + a)``: what CatBoost applies at prediction
    time. ``prior`` defaults to the empirical marginal of Y.
    """
    n = np.asarray(n, dtype=float)
    return posterior_mean_conditional(n, a, qhat_y(n) if prior is None else np.asarray(prior))


def ordered_target_statistics(x: np.ndarray, y: np.ndarray, ky: int | None = None, a: float = 1.0,
                              prior: np.ndarray | None = None, seed: int | None = None) -> np.ndarray:
    """
    Ordered target statistics for training rows (Prokhorenkova et al. 2018):
    in a random permutation, each row is encoded only with the rows of its
    category that come before it, ``(n_xy^< + a * prior_y) / (n_x^< + a)``.
    Returns an ``(len(x), ky)`` array in the original row order.
    """
    x = np.asarray(x, dtype=int)
    y = np.asarray(y, dtype=int)
    ky = int(y.max()) + 1 if ky is None else ky
    prior = np.bincount(y, minlength=ky) / len(y) if prior is None else np.asarray(prior, dtype=float)
    perm = np.random.default_rng(seed).permutation(len(x))
    xs, ys = x[perm], y[perm]
    order = np.argsort(xs, kind="stable")
    xo, yo = xs[order], ys[order]
    onehot = np.eye(ky)[yo]
    start = np.r_[True, xo[1:] != xo[:-1]]
    group_start = np.maximum.accumulate(np.where(start, np.arange(len(xo)), 0))
    cum = np.cumsum(onehot, axis=0) - onehot
    before_y = cum - cum[group_start]
    before_n = (np.arange(len(xo)) - group_start)[:, None]
    stats_sorted = (before_y + a * prior[None, :]) / (before_n + a)
    out = np.empty_like(stats_sorted)
    out[perm[order]] = stats_sorted
    return out


def catboost_classifier_encoding(x: np.ndarray, y: np.ndarray, kx: int, ky: int,
                                 iterations: int = 300, seed: int = 0, **params) -> np.ndarray:
    """
    Fit ``catboost.CatBoostClassifier`` on the single categorical feature and
    return its predicted ``(kx, ky)`` class probabilities for every category
    (categories absent from training get CatBoost's own default). Requires the
    optional ``catboost`` package.
    """
    from catboost import CatBoostClassifier

    model = CatBoostClassifier(iterations=iterations, random_seed=seed, verbose=0,
                               loss_function="Logloss" if ky == 2 else "MultiClass",
                               allow_writing_files=False, **params)
    model.fit(np.asarray(x, dtype=int).reshape(-1, 1), np.asarray(y, dtype=int), cat_features=[0])
    proba = model.predict_proba(np.arange(kx).reshape(-1, 1))
    enc = np.zeros((kx, ky))
    enc[:, np.asarray(model.classes_, dtype=int)] = proba
    return enc
