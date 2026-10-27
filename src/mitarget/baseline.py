"""
K = 1 empirical-Bayes estimator of Hernandez & Samengo (2019).

A single global concentration beta is fit by maximizing the marginal
likelihood; the posterior mean of ``q_{y|x}`` at ``beta*`` is the target
encoding, and the posterior mean MI at ``beta*`` is the information estimate.

:func:`singleton_rate` gives the "baseline + unseen correction": categories
never seen in training get the label distribution of the categories seen
exactly once, instead of the prior mean ``qhat_y``. Under popularity
dependence the right prior for categories seen ``n`` times is the law of ``q``
among categories seen ``n + 1`` times, so for ``n = 0`` it is the singletons'
rate; when ``q`` does not depend on popularity the two agree up to noise.
Seen categories keep the baseline's encoding.

:func:`doubleton_rates` extends this one size up, the "baseline + Robbins
correction (n <= 1)": a category seen once with label ``c`` gets the label
distribution of the other row of the doubletons containing a ``c``, which is
the oracle's prediction for once-seen categories read off the doubleton table
(Robbins form). Categories seen twice or more keep the baseline's encoding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scipy.stats import chi2

from .core import (
    LOG_BETA_BOUNDS, fit_beta_star, plugin_information, plugin_target_encoding,
    posterior_mean_conditional, posterior_mean_information, qhat_y,
)


@dataclass
class BaselineFit:
    beta_star: float
    qy: np.ndarray
    target_encoding: np.ndarray   # (kx, ky) posterior mean q_{y|x}
    I_hat: float
    I_plugin: float
    target_encoding_plugin: np.ndarray


def fit_baseline(n: np.ndarray, qy: np.ndarray | None = None,
                 log_beta_bounds: tuple[float, float] = LOG_BETA_BOUNDS) -> BaselineFit:
    """Empirical-Bayes fit, target encoding, MI estimate and the plug-in reference."""
    qy = qhat_y(n) if qy is None else qy
    beta_star = fit_beta_star(n, qy, log_beta_bounds=log_beta_bounds)
    return BaselineFit(
        beta_star=beta_star,
        qy=qy,
        target_encoding=posterior_mean_conditional(n, beta_star, qy),
        I_hat=posterior_mean_information(n, beta_star, qy),
        I_plugin=plugin_information(n),
        target_encoding_plugin=plugin_target_encoding(n),
    )


def singleton_rate(n: np.ndarray, qy: np.ndarray | None = None, pseudo: float = 1.0) -> np.ndarray:
    """
    Label distribution of the categories seen exactly once, shrunk toward
    ``qy`` by ``pseudo`` pseudo-observations (so a handful of singletons
    cannot give a zero probability). ``qy`` if there are no singletons.
    """
    n = np.asarray(n, dtype=float)
    qy = qhat_y(n) if qy is None else np.asarray(qy, dtype=float)
    single = n[n.sum(axis=1) == 1]
    return (single.sum(axis=0) + pseudo * qy) / (len(single) + pseudo)


def doubleton_rates(n: np.ndarray, prior: np.ndarray | None = None, pseudo: float = 1.0) -> np.ndarray:
    """
    Robbins-form prediction for categories seen exactly once, read off the
    doubletons (categories seen exactly twice).

    Row ``c`` of the returned ``(ky, ky)`` matrix is the predicted label
    distribution for a category seen once with label ``c``: among ordered
    pairs of rows from the same doubleton whose first row has label ``c``, the
    distribution of the second row's label. For a binary target this is
    ``P(1 | 1) = 2 F_22 / (2 F_22 + F_21)`` and ``P(1 | 0) = F_21 / (F_21 + 2 F_20)``.
    Each row is shrunk toward ``prior[c]`` (default: the marginal ``qy``) by
    ``pseudo`` pseudo-pairs, so a handful of doubletons cannot give a zero
    probability; with no doubletons it is ``prior``.
    """
    n = np.asarray(n, dtype=float)
    ky = n.shape[1]
    if prior is None:
        prior = np.tile(qhat_y(n), (ky, 1))
    d = n[n.sum(axis=1) == 2]
    pairs = d.T @ d - np.diag(d.sum(axis=0))      # pairs[c, y]: ordered pairs (c then y) within a doubleton
    return (pairs + pseudo * prior) / (pairs.sum(axis=1, keepdims=True) + pseudo)


def doubleton_test(n: np.ndarray, beta: float, qy: np.ndarray) -> float:
    """
    p-value of the likelihood-ratio (G) test that the doubletons' label pairs
    follow the fitted single beta (Dirichlet) with concentration ``beta`` and
    mean ``qy``, against a free multinomial over the ``ky (ky + 1) / 2``
    unordered pairs. A small p-value says the categories seen twice, and hence
    the oracle for categories seen once, differ from the pooled fit. The fitted
    parameters are treated as fixed, so the test is slightly conservative.
    """
    n = np.asarray(n, dtype=float)
    d = n[n.sum(axis=1) == 2]
    if len(d) == 0:
        return 1.0
    ky = n.shape[1]
    iu = np.triu_indices(ky)
    pairs = d.T @ d - np.diag(d.sum(axis=0))       # ordered pairs; diagonal counts each {c, c} twice
    obs = np.where(iu[0] == iu[1], pairs[iu] / 2, pairs[iu])
    expd = qy[:, None] * (beta * qy[None, :] + np.eye(ky)) / (beta + 1)   # P(first c, second y)
    expd = np.where(iu[0] == iu[1], expd[iu], 2 * expd[iu]) * len(d)
    pos = obs > 0
    g = 2 * float((obs[pos] * np.log(obs[pos] / expd[pos])).sum())
    return float(chi2.sf(g, df=len(obs) - 1))
