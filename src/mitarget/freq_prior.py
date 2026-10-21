"""
Frequency-aware prior for a binary target: beta-binomial regression of the
prior mean on the level's count.

    q_x ~ Beta(mu_x s_x, (1 - mu_x) s_x),
    logit mu_x = a + b z_x,        z_x = log(1 + n_x),
    log s_x    = c                 (spread="shared")   or   c + d z_x   (spread="linear").

All parameters are fit by type-II ML on the levels seen in training. With
``b = d = 0`` this is the K = 1 baseline with a free prior mean.

When ``q`` depends on popularity, the right prior for a level seen ``n`` times
is the law of ``q`` among levels seen ``n + 1`` times (shift identity: the loss
weights a level by ``p_x``, and ``p Bin(n | N, p)`` is proportional to
``Bin(n + 1 | N + 1, p)``). So the *encoding* evaluates the fitted curve at
``n_x + shift`` with ``shift = 1``; unseen levels get the curve at ``n = 1``,
which tracks the singletons' label rate. The *MI estimate* sums over training
levels with weights ``n_x / N``, which already estimate ``p_x``, so it uses the
unshifted curve.

With ``lr_test=True`` (default) the regression is used only when the
likelihood-ratio test of ``b = d = 0`` rejects at level ``alpha``; otherwise the
fit falls back to the baseline. On data without a frequency-target link this
makes the method match the baseline.

Binary targets only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln, digamma, expit, logit
from scipy.stats import chi2

from .baseline import BaselineFit, fit_baseline
from .core import LOG_BETA_BOUNDS, entropy, qhat_y

SPREADS = ("shared", "linear")


@dataclass
class FreqPriorFit:
    params: np.ndarray              # (a, b, c[, d]) on z = log(1 + n)
    spread: str                     # "shared" or "linear"
    shift: int                      # strata shift used for the encoding
    selected: bool                  # regression used (LR test passed, or lr_test=False)
    lr_statistic: float             # 2 (loglik - loglik_null)
    p_value: float
    loglik: float
    loglik_null: float              # b = d = 0 (one free mean, one spread)
    qy: np.ndarray                  # (2,) prediction for unseen categories
    target_encoding: np.ndarray     # (kx, 2) posterior mean of q(y | x)
    I_hat: float                    # MI estimate in nats
    baseline: BaselineFit           # the fallback

    def prior(self, n, shift: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Prior mean and concentration for levels seen ``n`` times, read at ``n + shift``."""
        shift = self.shift if shift is None else shift
        return _prior(self.params, np.log1p(np.asarray(n, dtype=float) + shift), self.spread)

    def predict(self, k, n, shift: int | None = None) -> np.ndarray:
        """``P(y = 1 | x)`` for a level with ``k`` ones out of ``n``, using the regression prior."""
        k, n = np.asarray(k, dtype=float), np.asarray(n, dtype=float)
        mu, s = self.prior(n, shift)
        return (k + mu * s) / (n + s)


def _prior(theta, z, spread):
    mu = expit(theta[0] + theta[1] * z)
    log_s = theta[2] + (theta[3] * z if spread == "linear" else 0.0)
    return mu, np.exp(np.clip(log_s, *LOG_BETA_BOUNDS))


def _negll(theta, k, n, z, mult, spread):
    mu, s = _prior(theta, z, spread)
    mu = np.clip(mu, 1e-9, 1 - 1e-9)
    a, b = mu * s, (1 - mu) * s
    return -float(np.sum(mult * (betaln(k + a, n - k + b) - betaln(a, b))))


def _beta_entropy(a, b):
    """``E[h(q)]`` in nats for ``q ~ Beta(a, b)``."""
    s = a + b
    return digamma(s + 1) - (a / s) * digamma(a + 1) - (b / s) * digamma(b + 1)


def fit_freq_prior(n: np.ndarray, spread: str = "shared", shift: int = 1, lr_test: bool = True,
                   alpha: float = 0.05) -> FreqPriorFit:
    """
    Fit the frequency-aware prior to a ``(kx, 2)`` count matrix.

    Parameters
    ----------
    spread : "shared" (one concentration) or "linear" (log concentration linear in z).
        "linear" helps only with plenty of data per level.
    shift : strata shift for the encoding (1 is the oracle-consistent choice, 0 evaluates
        the curve at each level's own count).
    lr_test : fall back to the baseline unless the regression is significant.
    alpha : level of the likelihood-ratio test (chi-square with 1 or 2 degrees of freedom).
    """
    n = np.asarray(n, dtype=float)
    if n.ndim != 2 or n.shape[1] != 2:
        raise ValueError("fit_freq_prior needs a binary target: a (kx, 2) count matrix")
    if spread not in SPREADS:
        raise ValueError(f"unknown spread {spread!r}; expected one of {SPREADS}")
    k_all, n_all = n[:, 1], n.sum(axis=1)
    qy = qhat_y(n)
    base = fit_baseline(n, qy=qy)

    # The likelihood depends on a level only through (k, n): fit on the unique pairs.
    seen = n_all > 0
    pairs, mult = np.unique(np.stack([k_all[seen], n_all[seen]], axis=1), axis=0, return_counts=True)
    k, m, mult = pairs[:, 0], pairs[:, 1], mult.astype(float)
    z = np.log1p(m)
    zc = float(np.sum(mult * z) / mult.sum())   # centre z for conditioning; undone below
    rate = float(np.clip(qy[1], 1e-6, 1 - 1e-6))
    s0 = base.beta_star if base.beta_star < 1e3 else 5.0
    x0 = [logit(rate), 0.0, np.log(s0)] + ([0.0] if spread == "linear" else [])
    res = minimize(_negll, x0, args=(k, m, z - zc, mult, spread), method="L-BFGS-B")
    theta = res.x.copy()
    theta[0] -= theta[1] * zc
    if spread == "linear":
        theta[2] -= theta[3] * zc
    null = minimize(_negll, [logit(rate), 0.0, np.log(s0)], args=(k, m, 0 * z, mult, "shared"),
                    method="L-BFGS-B")
    loglik, loglik_null = -float(res.fun), -float(null.fun)
    lr = max(0.0, 2 * (loglik - loglik_null))
    p_value = float(chi2.sf(lr, df=len(theta) - 2))
    selected = (p_value < alpha) if lr_test else True

    if not selected:
        return FreqPriorFit(theta, spread, shift, False, lr, p_value, loglik, loglik_null, qy,
                            base.target_encoding, base.I_hat, base)

    mu, s = _prior(theta, np.log1p(n_all + shift), spread)
    p1 = (k_all + mu * s) / (n_all + s)
    mu0, _ = _prior(theta, np.log1p(np.array([float(shift)])), spread)
    unseen = float(mu0[0])

    mu, s = _prior(theta, np.log1p(n_all[seen]), spread)
    a = k_all[seen] + mu * s
    b = n_all[seen] - k_all[seen] + (1 - mu) * s
    I_hat = entropy(qy) - float(np.sum(n_all[seen] / n_all.sum() * _beta_entropy(a, b)))
    return FreqPriorFit(theta, spread, shift, True, lr, p_value, loglik, loglik_null,
                        np.array([1 - unseen, unseen]), np.stack([1 - p1, p1], axis=1), I_hat, base)
