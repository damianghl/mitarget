"""
Shared Dirichlet-multinomial building blocks.

Every estimator in this package works on a joint count matrix ``n`` of shape
``(kx, ky)``: ``n[x, y]`` is the number of samples with category ``x`` and
target class ``y``. ``X`` is the large, under-sampled categorical feature and
``Y`` the few-state, well-sampled target.

The prior is the one of Hernandez & Samengo (2019), Entropy 21(6):623, Eq. (13),
written for an arbitrary number of target classes ``ky``:

    q_{y|x} ~ Dirichlet(beta * q_y)    independently for each x,

with ``q_y`` fixed to the empirical marginal of ``Y``. Nothing here assumes
``ky == 2``.

All information quantities are in nats.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import digamma, gammaln

#: Default search interval for log(beta) in empirical-Bayes fits.
LOG_BETA_BOUNDS: tuple[float, float] = (float(np.log(1e-3)), float(np.log(1e4)))


def counts_from_labels(x: np.ndarray, y: np.ndarray, kx: int | None = None,
                       ky: int | None = None) -> np.ndarray:
    """Joint count matrix ``n[x, y]`` from integer-coded label arrays."""
    x = np.asarray(x, dtype=int)
    y = np.asarray(y, dtype=int)
    kx = int(x.max()) + 1 if kx is None else kx
    ky = int(y.max()) + 1 if ky is None else ky
    n = np.zeros((kx, ky), dtype=float)
    np.add.at(n, (x, y), 1.0)
    return n


def labels_from_counts(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`counts_from_labels`: one ``(x, y)`` pair per sample."""
    x_idx, y_idx = np.nonzero(n)
    reps = np.asarray(n[x_idx, y_idx]).astype(int)
    return np.repeat(x_idx, reps), np.repeat(y_idx, reps)


def qhat_y(n: np.ndarray) -> np.ndarray:
    """Plug-in marginal of Y (assumed reliable: Y is well sampled)."""
    ny = n.sum(axis=0)
    return ny / ny.sum()


def entropy(p: np.ndarray) -> float:
    """Shannon entropy of a probability vector, in nats."""
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def per_state_log_likelihood(beta: float, n: np.ndarray, qy: np.ndarray) -> np.ndarray:
    """
    ``log p(n_x | beta)`` for each state ``x`` (length-``kx`` array), with
    ``q_{y|x}`` integrated out under ``Dirichlet(beta * qy)``.

    Sums to the log marginal likelihood (paper Eq. 21, generalized to any ky).
    A singleton state (``n_x = 1``, all mass on ``y0``) evaluates to exactly
    ``log qy[y0]`` independently of beta: only coincidences carry information
    about beta (paper Eq. 23).
    """
    nx = n.sum(axis=1)
    alpha = beta * qy
    term_x = gammaln(beta) - gammaln(nx + beta)
    term_xy = gammaln(n + alpha[None, :]) - gammaln(alpha[None, :])
    return term_x + term_xy.sum(axis=1)


def log_marginal_likelihood(beta: float, n: np.ndarray, qy: np.ndarray) -> float:
    """``log p(n | beta)``, summed over the observed states."""
    nx = n.sum(axis=1)
    if not np.any(nx > 0):
        return 0.0
    return float(per_state_log_likelihood(beta, n[nx > 0], qy).sum())


def fit_beta_star(n: np.ndarray, qy: np.ndarray,
                  log_beta_bounds: tuple[float, float] = LOG_BETA_BOUNDS,
                  weights: np.ndarray | None = None) -> float:
    """
    Empirical-Bayes ``beta*`` maximizing ``log p(n | beta)`` (paper Section 5).

    With ``weights`` (length ``kx``), maximizes the weighted sum of per-state
    log-likelihoods instead: the M-step for one cluster's beta in the
    finite-mixture and CRP extensions.
    """
    if weights is None:
        def objective(lb):
            return -log_marginal_likelihood(np.exp(lb), n, qy)
    else:
        def objective(lb):
            return -float(np.sum(weights * per_state_log_likelihood(np.exp(lb), n, qy)))

    res = minimize_scalar(objective, bounds=log_beta_bounds, method="bounded",
                          options={"xatol": 1e-6})
    return float(np.exp(res.x))


def per_state_conditional_entropy(beta: float, n: np.ndarray, qy: np.ndarray) -> np.ndarray:
    """
    Posterior mean entropy ``<H(Y|x)>`` for each state under the
    ``Dirichlet(beta * qy)`` posterior (general-ky form of paper Eq. 16/20).
    """
    nx = n.sum(axis=1)
    alpha = beta * qy
    w = (alpha[None, :] + n) / (beta + nx)[:, None]
    inner = digamma(alpha[None, :] + n + 1.0)
    return digamma(beta + nx + 1.0) - np.sum(w * inner, axis=1)


def posterior_mean_conditional(n: np.ndarray, beta: float, qy: np.ndarray) -> np.ndarray:
    """
    The smoothed target encoding: posterior mean of ``q_{y|x}`` given beta,

        qhat_{y|x} = (n_xy + beta * qy) / (n_x + beta),

    i.e. classic target encoding with pseudo-count ``m = beta``.
    """
    nx = n.sum(axis=1, keepdims=True)
    return (n + beta * qy[None, :]) / (nx + beta)


def information_from_conditional_entropy(n: np.ndarray, qy: np.ndarray,
                                         h_y_given_x: np.ndarray) -> float:
    """``Hhat(Y) - sum_x (n_x / N) <H(Y|x)>``."""
    nx = n.sum(axis=1)
    return float(entropy(qy) - np.sum((nx / n.sum()) * h_y_given_x))


def posterior_mean_information(n: np.ndarray, beta: float, qy: np.ndarray) -> float:
    """Posterior mean mutual information given beta (paper Eq. 16/20)."""
    return information_from_conditional_entropy(
        n, qy, per_state_conditional_entropy(beta, n, qy))


def plugin_information(n: np.ndarray) -> float:
    """Naive frequency-substitution MI estimate (for comparison)."""
    nx = n.sum(axis=1)
    mask = nx > 0
    p = n[mask] / nx[mask, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(p > 0, np.log(p), 0.0)
    h_y_given_x = -np.sum(p * logp, axis=1)
    return float(entropy(qhat_y(n)) - np.sum((nx[mask] / n.sum()) * h_y_given_x))


def plugin_target_encoding(n: np.ndarray) -> np.ndarray:
    """
    Per-state frequencies ``n_xy / n_x``. Unobserved states (``n_x = 0``) get the
    global marginal so every method yields a complete ``(kx, ky)`` matrix.
    """
    nx = n.sum(axis=1, keepdims=True)
    qy = qhat_y(n)
    return np.where(nx > 0, n / np.maximum(nx, 1), qy[None, :])


def fixed_m_encoding(n: np.ndarray, m: float = 20.0, qy: np.ndarray | None = None) -> np.ndarray:
    """Additive smoothing with a hand-picked pseudo-count ``m`` (``category_encoders`` default 20)."""
    qy = qhat_y(n) if qy is None else qy
    return posterior_mean_conditional(n, m, qy)
