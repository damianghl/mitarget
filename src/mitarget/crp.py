"""
Nonparametric (CRP) extension: states are partitioned by a Chinese Restaurant
Process instead of a fixed K.

    z_x ~ CRP(alpha_crp)
    beta_{z_x} ~ p(beta)                    (uniform on a log-spaced grid)
    q_{y|x} | z_x, beta ~ Dirichlet(beta * qy)
    n_x | q_{y|x}       ~ Multinomial(n_x, q_{y|x})

Cluster assignments are sampled with a collapsed Gibbs sampler (Neal 2000,
Algorithm 2 for conjugate DPMs), with beta integrated over the grid. For each
retained sweep, every cluster's beta is refit by empirical-Bayes maximization
(matching the paper's finding that the marginal-likelihood peak is as good as
integrating over beta), and the per-state target encoding and MI estimate are
averaged across sweeps. Averaging per-state quantities sidesteps label
switching entirely.

Self-tuning alpha_crp: with ``alpha_prior=(a0, b0)`` a Gamma(shape=a0, rate=b0)
hyperprior is placed on alpha_crp and it is resampled every sweep with the
Escobar & West (1995) auxiliary-variable update, so the data decide how much
splitting is warranted. ``Gamma(0.5, 2)`` (mean 0.25) was the conservative
choice calibrated in earlier experiments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from .core import (
    LOG_BETA_BOUNDS, entropy, fit_beta_star, per_state_conditional_entropy,
    per_state_log_likelihood, posterior_mean_conditional, qhat_y,
)

#: Recommended self-tuning configuration from the original experiments.
DEFAULT_ALPHA_INIT = 0.25
DEFAULT_ALPHA_PRIOR = (0.5, 2.0)


def make_beta_grid(beta_bounds: tuple[float, float] = (0.03, 30.0), G: int = 20) -> np.ndarray:
    """Log-spaced beta grid; uniform weights on it approximate a log-uniform prior."""
    return np.geomspace(beta_bounds[0], beta_bounds[1], G)


@dataclass
class CRPFit:
    target_encoding: np.ndarray     # (kx, ky) MC-averaged posterior mean
    I_hat: float                    # MC-averaged MI estimate
    I_hat_samples: np.ndarray       # per-sample MI (for an MC error bar)
    n_clusters_trace: np.ndarray    # number of clusters per sweep
    alpha_trace: np.ndarray         # alpha_crp used at each sweep
    qy: np.ndarray
    beta_grid: np.ndarray


def _update_alpha(alpha: float, K: int, kx: int, a0: float, b0: float,
                  rng: np.random.Generator) -> float:
    """
    Escobar & West (1995) update of the CRP concentration under a
    Gamma(shape=a0, rate=b0) prior, given K clusters over kx items:

        eta | alpha        ~ Beta(alpha + 1, kx)
        pi / (1 - pi)      = (a0 + K - 1) / (kx (b0 - log eta))
        alpha | eta, K     ~ pi Gamma(a0 + K, b0 - log eta)
                             + (1 - pi) Gamma(a0 + K - 1, b0 - log eta)
    """
    eta = rng.beta(alpha + 1.0, kx)
    rate = b0 - np.log(eta)
    r = (a0 + K - 1.0) / (kx * rate)
    shape = (a0 + K) if rng.random() < r / (1.0 + r) else (a0 + K - 1.0)
    return float(rng.gamma(shape, 1.0 / rate))


def _readout_at_ml_beta(n, qy, clusters, log_beta_bounds):
    """Per-state encoding and <H(Y|x)> for one partition, with beta_c refit per cluster."""
    kx, ky = n.shape
    target = np.zeros((kx, ky))
    h_y_given_x = np.zeros(kx)
    for info in clusters.values():
        members = np.array(sorted(info["members"]))
        weights = np.zeros(kx)
        weights[members] = 1.0
        beta_c = fit_beta_star(n, qy, log_beta_bounds=log_beta_bounds, weights=weights)
        target[members] = posterior_mean_conditional(n[members], beta_c, qy)
        h_y_given_x[members] = per_state_conditional_entropy(beta_c, n[members], qy)
    return target, h_y_given_x


def fit_crp_mixture(n: np.ndarray, alpha_crp: float = DEFAULT_ALPHA_INIT,
                    alpha_prior: tuple[float, float] | None = DEFAULT_ALPHA_PRIOR,
                    beta_grid: np.ndarray | None = None,
                    qy: np.ndarray | None = None,
                    n_sweeps: int = 150, burn_in: int = 50, thin: int = 5,
                    log_beta_bounds: tuple[float, float] = LOG_BETA_BOUNDS,
                    seed: int | None = None, verbose: bool = False) -> CRPFit:
    """
    Collapsed Gibbs fit of the CRP mixture.

    alpha_crp: starting value of the CRP concentration; with ``alpha_prior=None``
        it stays fixed for every sweep.
    alpha_prior: ``(a0, b0)`` of a Gamma hyperprior on alpha_crp, which is then
        resampled once per sweep (self-tuning, the default).
    """
    if n_sweeps <= burn_in:
        raise ValueError("n_sweeps must exceed burn_in to retain any samples")
    rng = np.random.default_rng(seed)
    kx, _ = n.shape
    qy = qhat_y(n) if qy is None else qy
    beta_grid = make_beta_grid() if beta_grid is None else beta_grid
    G = len(beta_grid)
    # Uniform over the grid; only drives the assignment step, never the readout.
    log_prior_beta = np.full(G, -np.log(G))

    ll_grid = np.stack([per_state_log_likelihood(b, n, qy) for b in beta_grid], axis=1)  # (kx, G)
    nx = n.sum(axis=1)
    N = n.sum()
    h_y = entropy(qy)

    # Start from a single cluster (the K = 1 baseline).
    z = np.zeros(kx, dtype=int)
    next_id = 1
    clusters = {0: {"members": set(range(kx)), "LL": ll_grid.sum(axis=0).copy()}}

    order = np.arange(kx)
    acc_target = np.zeros(n.shape)
    acc_I = []
    n_clusters_trace = []
    alpha_trace = []

    for sweep in range(1, n_sweeps + 1):
        rng.shuffle(order)
        for x in order:
            c_old = z[x]
            clusters[c_old]["members"].discard(x)
            clusters[c_old]["LL"] -= ll_grid[x]
            if not clusters[c_old]["members"]:
                del clusters[c_old]

            ids = list(clusters)
            sizes = np.array([len(clusters[c]["members"]) for c in ids], dtype=float)
            ll_others = np.stack([clusters[c]["LL"] for c in ids], axis=0)  # (C, G)

            # Existing clusters: predictive p(n_x | cluster without x), grid-marginalized.
            num = logsumexp(log_prior_beta[None, :] + ll_others + ll_grid[x][None, :], axis=1)
            den = logsumexp(log_prior_beta[None, :] + ll_others, axis=1)
            log_w = np.append(np.log(sizes) + (num - den),
                              np.log(alpha_crp) + logsumexp(log_prior_beta + ll_grid[x]))
            probs = np.exp(log_w - logsumexp(log_w))
            probs /= probs.sum()

            choice = rng.choice(len(probs), p=probs)
            if choice == len(ids):
                c_new = next_id
                next_id += 1
                clusters[c_new] = {"members": {x}, "LL": ll_grid[x].copy()}
            else:
                c_new = ids[choice]
                clusters[c_new]["members"].add(x)
                clusters[c_new]["LL"] += ll_grid[x]
            z[x] = c_new

        n_clusters_trace.append(len(clusters))
        alpha_trace.append(alpha_crp)
        if alpha_prior is not None:
            alpha_crp = _update_alpha(alpha_crp, len(clusters), kx, *alpha_prior, rng)

        if sweep > burn_in and (sweep - burn_in) % thin == 0:
            target, h_y_given_x = _readout_at_ml_beta(n, qy, clusters, log_beta_bounds)
            acc_target += target
            acc_I.append(float(h_y - np.sum((nx / N) * h_y_given_x)))

        if verbose and sweep % 25 == 0:
            last_I = acc_I[-1] if acc_I else float("nan")
            print(f"  sweep {sweep:4d}: clusters={len(clusters):3d}  "
                  f"alpha={alpha_trace[-1]:.3f}  I_sample={last_I:.4f}")

    if not acc_I:
        raise ValueError("no samples retained; check n_sweeps, burn_in and thin")
    return CRPFit(target_encoding=acc_target / len(acc_I),
                  I_hat=float(np.mean(acc_I)),
                  I_hat_samples=np.array(acc_I),
                  n_clusters_trace=np.array(n_clusters_trace),
                  alpha_trace=np.array(alpha_trace),
                  qy=qy, beta_grid=beta_grid)
