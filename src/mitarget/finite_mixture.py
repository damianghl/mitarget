"""
Finite K-component mixture extension of the baseline.

Each state belongs to one of K latent clusters with its own concentration
beta_c (shared center qy, so clusters differ only in spread):

    z_x ~ Categorical(pi)
    q_{y|x} | z_x = c ~ Dirichlet(beta_c * qy)
    n_x | q_{y|x}     ~ Multinomial(n_x, q_{y|x})

Fit by EM with {pi_c, beta_c} as parameters and z_x as missing data:

    E-step: r[x, c] = P(z_x = c | n_x, pi, beta) from the same per-state
            Dirichlet-multinomial likelihood as the baseline.
    M-step: pi_c <- mean_x r[x, c];  beta_c <- argmax_beta sum_x r[x, c] ll_x(beta).

Singleton states have a beta-independent likelihood, so they carry no signal
for the E-step either. EM is restarted from several random initializations
and the highest-likelihood solution is kept.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from .core import (
    LOG_BETA_BOUNDS, fit_beta_star, information_from_conditional_entropy,
    per_state_conditional_entropy, per_state_log_likelihood, posterior_mean_conditional,
    qhat_y,
)


@dataclass
class FiniteMixtureFit:
    K: int
    pi: np.ndarray                  # (K,) mixing weights
    beta: np.ndarray                # (K,) per-cluster concentration, ascending
    qy: np.ndarray                  # (ky,) shared center
    responsibilities: np.ndarray    # (kx, K) P(z_x = c | data)
    target_encoding: np.ndarray     # (kx, ky) mixture posterior mean of q_{y|x}
    I_hat: float
    log_likelihood: float           # observed-data log-likelihood at convergence
    n_iter: int


def _e_step(n, qy, pi, beta):
    log_r = np.stack([np.log(pi[c]) + per_state_log_likelihood(beta[c], n, qy)
                      for c in range(len(pi))], axis=1)
    log_norm = logsumexp(log_r, axis=1)
    return np.exp(log_r - log_norm[:, None]), float(log_norm.sum())


def _m_step(n, qy, r, log_beta_bounds):
    pi = np.clip(r.mean(axis=0), 1e-8, None)
    pi = pi / pi.sum()
    beta = np.array([fit_beta_star(n, qy, log_beta_bounds=log_beta_bounds, weights=r[:, c])
                     for c in range(r.shape[1])])
    return pi, beta


def _fit_single_init(n, qy, K, rng, log_beta_bounds, max_iter, tol):
    beta = np.sort(np.exp(rng.uniform(log_beta_bounds[0], log_beta_bounds[1], size=K)))
    pi = np.full(K, 1.0 / K)
    prev_ll = -np.inf
    n_iter = 0
    while n_iter < max_iter:
        n_iter += 1
        r, ll = _e_step(n, qy, pi, beta)
        pi, beta = _m_step(n, qy, r, log_beta_bounds)
        converged = abs(ll - prev_ll) < tol * (1 + abs(prev_ll))
        prev_ll = ll
        if converged:
            break
    r, ll = _e_step(n, qy, pi, beta)
    return pi, beta, r, ll, n_iter


def fit_finite_mixture(n: np.ndarray, K: int, qy: np.ndarray | None = None,
                       n_init: int = 4, max_iter: int = 60, tol: float = 1e-6,
                       log_beta_bounds: tuple[float, float] = LOG_BETA_BOUNDS,
                       seed: int | None = None) -> FiniteMixtureFit:
    """Fit a K-component mixture by EM with ``n_init`` random restarts."""
    rng = np.random.default_rng(seed)
    qy = qhat_y(n) if qy is None else qy

    best = None
    for _ in range(n_init):
        cand = _fit_single_init(n, qy, K, rng, log_beta_bounds, max_iter, tol)
        if best is None or cand[3] > best[3]:
            best = cand
    pi, beta, r, ll, n_iter = best

    # Sort clusters by beta; labels are otherwise arbitrary.
    order = np.argsort(beta)
    pi, beta, r = pi[order], beta[order], r[:, order]

    target_encoding = np.zeros(n.shape, dtype=float)
    h_y_given_x = np.zeros(n.shape[0])
    for c in range(K):
        target_encoding += r[:, [c]] * posterior_mean_conditional(n, beta[c], qy)
        h_y_given_x += r[:, c] * per_state_conditional_entropy(beta[c], n, qy)

    return FiniteMixtureFit(
        K=K, pi=pi, beta=beta, qy=qy, responsibilities=r,
        target_encoding=target_encoding,
        I_hat=information_from_conditional_entropy(n, qy, h_y_given_x),
        log_likelihood=ll, n_iter=n_iter,
    )


def bic(fit: FiniteMixtureFit, kx: int) -> float:
    """BIC with (K - 1) weights + K betas as free parameters. Lower is better."""
    n_params = (fit.K - 1) + fit.K
    return -2 * fit.log_likelihood + n_params * np.log(kx)
