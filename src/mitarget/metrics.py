"""Evaluation metrics against a known truth or held-out data. All in nats."""

from __future__ import annotations

import numpy as np


def exact_mutual_information(qx: np.ndarray, q_ygivenx: np.ndarray) -> float:
    """``I(X;Y) = sum_x q_x KL(q_{y|x} || q_y)`` with ``q_y = sum_x q_x q_{y|x}``."""
    qy = qx @ q_ygivenx
    with np.errstate(divide="ignore", invalid="ignore"):
        logratio = np.where(q_ygivenx > 0, np.log(q_ygivenx / qy), 0.0)
    return float(np.sum(qx * np.sum(q_ygivenx * logratio, axis=1)))


def excess_cross_entropy(qx: np.ndarray, q_ygivenx_true: np.ndarray,
                         q_ygivenx_hat: np.ndarray) -> float:
    """
    ``E_{x ~ q_x}[ KL(q_{y|x}^true || qhat_{y|x}) ]``: the extra log-loss a
    classifier pays for predicting with ``qhat`` instead of the true conditional.

    Unlike MI bias, it does not let per-state errors cancel in aggregate.
    Always >= 0, and 0 iff ``qhat == q_true`` wherever ``q_x > 0``.
    """
    mask = q_ygivenx_true > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.where(
            mask, np.log(q_ygivenx_true / np.clip(q_ygivenx_hat, 1e-300, None)), 0.0)
    return float(np.sum(qx * np.sum(q_ygivenx_true * log_ratio, axis=1)))


def heldout_log_loss(encoding: np.ndarray, x: np.ndarray, y: np.ndarray,
                     eps: float = 1e-12) -> float:
    """Mean ``-log qhat(y | x)`` over held-out pairs; ranks methods like excess cross-entropy."""
    p = np.clip(encoding[np.asarray(x, dtype=int), np.asarray(y, dtype=int)], eps, 1.0)
    return float(np.mean(-np.log(p)))
