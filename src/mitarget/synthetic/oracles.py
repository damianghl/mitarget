"""
Bayes oracles: learners that know the true generative family (and possibly its
parameters) but, like any estimator, must infer each state's branch or
cluster from that state's own counts. Their excess cross-entropy is the floor
a realistic estimator can be compared against.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from ..core import per_state_log_likelihood, posterior_mean_conditional

#: (pi_junk, pi_q0, pi_1-q0) for the standard Scenario 4.
PARITY_PRIORS = (0.5, 0.25, 0.25)


def oracle_known_clusters_encoding(n: np.ndarray, qy: np.ndarray, betas, pis) -> np.ndarray:
    """
    Bayes-optimal encoding knowing the cluster betas and mixing weights but not
    each state's cluster. With one cluster it equals the baseline formula.
    """
    log_l = np.stack([np.log(pis[c]) + per_state_log_likelihood(betas[c], n, qy)
                      for c in range(len(betas))], axis=1)
    w = np.exp(log_l - logsumexp(log_l, axis=1, keepdims=True))
    return sum(w[:, [c]] * posterior_mean_conditional(n, betas[c], qy)
               for c in range(len(betas)))


def _three_point_log_joint(n: np.ndarray, q0: float, priors) -> np.ndarray:
    """(kx, 3) log p(branch, n_x) for branches (junk, q0, 1 - q0); binary y only."""
    pi_e, pi_l, pi_h = priors
    n0, n1 = n[:, 0], n[:, 1]
    return np.column_stack([
        np.log(pi_e) + (n0 + n1) * np.log(0.5),
        np.log(pi_l) + n1 * np.log(q0) + n0 * np.log(1 - q0),
        np.log(pi_h) + n1 * np.log(1 - q0) + n0 * np.log(q0),
    ])


def oracle_three_point_encoding(n: np.ndarray, q0: float, priors=PARITY_PRIORS) -> np.ndarray:
    """Bayes-optimal encoding for the 3-point-mass model (Scenario 4) given q0 and priors."""
    log_j = _three_point_log_joint(n, q0, priors)
    post = np.exp(log_j - logsumexp(log_j, axis=1, keepdims=True))
    q1 = post @ np.array([0.5, q0, 1 - q0])
    return np.column_stack([1 - q1, q1])


def three_point_log_marginal_likelihood(n: np.ndarray, q0: float, priors=PARITY_PRIORS) -> float:
    """``log p(n | q0, priors)`` under the 3-point-mass model."""
    return float(logsumexp(_three_point_log_joint(n, q0, priors), axis=1).sum())


def fit_q0_star(n: np.ndarray, priors=PARITY_PRIORS) -> float:
    """Empirical-Bayes q0 under the correctly specified 3-point-mass model (q0 < 1/2 by symmetry)."""
    res = minimize_scalar(lambda q0: -three_point_log_marginal_likelihood(n, q0, priors),
                          bounds=(1e-4, 0.5 - 1e-4), method="bounded",
                          options={"xatol": 1e-6})
    return float(res.x)
