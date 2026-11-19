"""
Classical bias-corrected estimators of I(X;Y) from a ``(kx, ky)`` count matrix, for comparison.

Each estimator is applied to the three entropies of ``I = H(X) + H(Y) - H(X, Y)``:

- :func:`mi_miller_madow`: maximum-likelihood entropies plus the Miller (1955) correction
  ``(K_obs - 1) / (2N)``; for the information this is the Panzeri-Treves (1996) first-order
  correction of the plug-in.
- :func:`mi_chao_shen`: the coverage-adjusted Horvitz-Thompson entropy of Chao & Shen (2003).
- :func:`mi_nsb`: the NSB posterior-mean entropy (Nemenman, Shafee & Bialek 2002) for an alphabet
  of known size; it needs the number of possible levels ``kx`` (counting unseen ones), so the
  comparison gives it information the other estimators do not have.

These estimators treat ``X`` as an unstructured alphabet: they do not model the conditionals
``q_x``, which is the point of comparing them with the estimators of :mod:`mitarget.mi_interval`.
"""

from __future__ import annotations

import numpy as np
from scipy.special import digamma, gammaln, polygamma

__all__ = ["entropy_miller_madow", "entropy_chao_shen", "entropy_nsb",
           "mi_miller_madow", "mi_chao_shen", "mi_nsb"]


def _counts(c) -> np.ndarray:
    c = np.asarray(c, dtype=float).ravel()
    return c[c > 0]


def _entropy_ml(c: np.ndarray) -> float:
    p = c / c.sum()
    return float(-(p * np.log(p)).sum())


def entropy_miller_madow(c) -> float:
    """Plug-in entropy (nats) plus ``(K_obs - 1) / (2N)``."""
    c = _counts(c)
    return _entropy_ml(c) + (len(c) - 1) / (2 * c.sum())


def entropy_chao_shen(c) -> float:
    """Chao-Shen coverage-adjusted entropy (nats)."""
    c = _counts(c)
    N = c.sum()
    f1 = float((c == 1).sum())
    if f1 == N:
        f1 = N - 1
    pa = (1 - f1 / N) * c / N
    return float(-(pa * np.log(pa) / (1 - (1 - pa) ** N)).sum())


def entropy_nsb(c, k: int, n_grid: int = 400) -> float:
    """
    NSB posterior mean of the entropy (nats) of a ``k``-symbol distribution, given the
    counts of the observed symbols (the other ``k - K_obs`` symbols have count 0). The
    mixture over the Dirichlet concentration ``b`` uses the NSB prior ``d E[H | b] / d b``
    and is integrated on a log grid of ``b``.
    """
    c = _counts(c)
    N, K_obs = c.sum(), len(c)
    if k < K_obs:
        raise ValueError("k must be at least the number of observed symbols")
    # symbols with the same count contribute identical terms: sum over distinct counts
    cu, mult = np.unique(c, return_counts=True)
    b = np.logspace(-6, 4, n_grid)
    A = k * b
    cb = cu[None, :] + b[:, None]
    # log evidence p(counts | b) up to a constant; zero-count symbols contribute nothing
    log_ev = gammaln(A) - gammaln(N + A) + ((gammaln(cb) - gammaln(b[:, None])) * mult).sum(axis=1)
    # NSB prior on b: derivative of the prior mean entropy, k psi1(k b + 1) - psi1(b + 1)
    prior = k * polygamma(1, A + 1) - polygamma(1, b + 1)
    # posterior mean entropy given b
    s = (cb * digamma(cb + 1) * mult).sum(axis=1)
    s += (k - K_obs) * b * digamma(b + 1)
    eh = digamma(N + A + 1) - s / (N + A)
    lw = log_ev + np.log(np.maximum(prior, 1e-300)) + np.log(b)  # integrate in log b
    w = np.exp(lw - lw.max())
    return float((w * eh).sum() / w.sum())


def _mi(n, ent) -> float:
    n = np.asarray(n, dtype=float)
    return ent(n.sum(axis=1)) + ent(n.sum(axis=0)) - ent(n)


def mi_miller_madow(n) -> float:
    """``I(X;Y)`` from Miller-Madow entropies of X, Y and (X, Y)."""
    return _mi(n, entropy_miller_madow)


def mi_chao_shen(n) -> float:
    """``I(X;Y)`` from Chao-Shen entropies of X, Y and (X, Y)."""
    return _mi(n, entropy_chao_shen)


def mi_nsb(n, kx: int | None = None) -> float:
    """``I(X;Y)`` from NSB entropies, with ``kx`` possible levels (default: the rows of ``n``)."""
    n = np.asarray(n, dtype=float)
    kx = n.shape[0] if kx is None else kx
    ky = n.shape[1]
    return (entropy_nsb(n.sum(axis=1), kx) + entropy_nsb(n.sum(axis=0), ky)
            - entropy_nsb(n, kx * ky))
