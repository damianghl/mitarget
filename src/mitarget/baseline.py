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
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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
