"""
scikit-learn-style API on top of the estimators.

- :class:`BayesianTargetEncoder` turns each categorical column into the fitted
  posterior mean ``qhat(y | x)``, with cross-fitting in ``fit_transform``.
- :class:`BayesianCategoricalClassifier` predicts ``Y`` from a single
  categorical feature with ``predict_proba = qhat(y | x)``.
- :func:`fit_counts` and :func:`mutual_information` expose the same estimators
  for a count matrix or a pair of label arrays.

Every method handles any number of target classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted

from .baseline import fit_baseline, singleton_rate
from .competitors import _one_vs_rest, catboost_ts_encoding, fit_glmm, fit_npmle
from .core import counts_from_labels, fixed_m_encoding, plugin_information, plugin_target_encoding, qhat_y
from .crp import fit_crp_mixture
from .finite_mixture import fit_finite_mixture
from .freq_prior import fit_freq_prior

METHODS = ("baseline", "baseline_unseen", "mixture", "crp", "fixed_m", "plugin", "glmm", "npmle",
           "catboost_ts", "freq_prior")


@dataclass
class EncodingFit:
    encoding: np.ndarray    # (kx, ky) qhat(y | x)
    qy: np.ndarray          # (ky,) prior used for unseen categories
    I_hat: float            # MI estimate in nats (NaN for fixed_m)
    details: Any            # the method's own fit object, if any


def fit_counts(n: np.ndarray, method: str = "baseline", *, n_components: int = 2,
               m: float = 20.0, n_init: int = 4, crp_kwargs: dict | None = None,
               random_state: int | None = None) -> EncodingFit:
    """Fit one of :data:`METHODS` to a ``(kx, ky)`` count matrix."""
    n = np.asarray(n, dtype=float)
    qy = qhat_y(n)
    if method == "baseline":
        f = fit_baseline(n, qy=qy)
        return EncodingFit(f.target_encoding, qy, f.I_hat, f)
    if method == "baseline_unseen":
        f = fit_baseline(n, qy=qy)
        unseen = singleton_rate(n, qy)
        enc = f.target_encoding.copy()
        enc[n.sum(axis=1) == 0] = unseen
        return EncodingFit(enc, unseen, f.I_hat, f)
    if method == "mixture":
        f = fit_finite_mixture(n, K=n_components, qy=qy, n_init=n_init, seed=random_state)
        return EncodingFit(f.target_encoding, qy, f.I_hat, f)
    if method == "crp":
        f = fit_crp_mixture(n, qy=qy, seed=random_state, **(crp_kwargs or {}))
        return EncodingFit(f.target_encoding, qy, f.I_hat, f)
    if method == "fixed_m":
        return EncodingFit(fixed_m_encoding(n, m=m, qy=qy), qy, float("nan"), None)
    if method == "plugin":
        return EncodingFit(plugin_target_encoding(n), qy, plugin_information(n), None)
    if method == "glmm":
        f = fit_glmm(n)
        return EncodingFit(f.target_encoding, f.qy, float("nan"), f)
    if method == "npmle":
        f = fit_npmle(n)
        return EncodingFit(f.target_encoding, f.qy, f.I_hat, f)
    if method == "catboost_ts":
        return EncodingFit(catboost_ts_encoding(n), qy, float("nan"), None)
    if method == "freq_prior":
        if n.shape[1] == 2:
            f = fit_freq_prior(n)
            return EncodingFit(f.target_encoding, f.qy, f.I_hat, f)
        enc, unseen, fits = _one_vs_rest(n, _freq_prior_binary)
        return EncodingFit(enc, unseen, float("nan"), fits)
    raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")


def _freq_prior_binary(k1, m):
    f = fit_freq_prior(np.stack([m - k1, k1], axis=1))
    return f.target_encoding[:, 1], f.qy[1], f


def mutual_information(x, y, method: str = "baseline", **kwargs) -> float:
    """Estimate ``I(X;Y)`` in nats from paired categorical samples."""
    x_codes, _ = pd.factorize(pd.Series(np.asarray(x, dtype=object)), use_na_sentinel=False)
    y_codes = LabelEncoder().fit_transform(np.asarray(y))
    return fit_counts(counts_from_labels(x_codes, y_codes), method, **kwargs).I_hat


class _EstimatorParams:
    """Shared constructor parameters and per-column fitting."""

    def _fit_kwargs(self):
        return dict(n_components=self.n_components, m=self.m, n_init=self.n_init,
                    crp_kwargs=self.crp_kwargs, random_state=self.random_state)

    def _fit_column(self, col: np.ndarray, y_codes: np.ndarray, ky: int):
        codes, categories = pd.factorize(pd.Series(col, dtype=object), use_na_sentinel=False)
        n = counts_from_labels(codes, y_codes, kx=len(categories), ky=ky)
        return pd.Index(categories), fit_counts(n, self.method, **self._fit_kwargs())

    @staticmethod
    def _lookup(col: np.ndarray, categories: pd.Index, fit: EncodingFit) -> np.ndarray:
        idx = categories.get_indexer(pd.Series(col, dtype=object))
        out = np.tile(fit.qy, (len(col), 1))
        seen = idx >= 0
        out[seen] = fit.encoding[idx[seen]]
        return out


def _as_2d_object(X) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        return X.to_numpy(dtype=object)
    arr = np.asarray(X, dtype=object)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError("X must be 2-dimensional")
    return arr


class BayesianTargetEncoder(_EstimatorParams, TransformerMixin, BaseEstimator):
    """
    Empirical-Bayes target encoder for high-cardinality categorical features.

    Each column is encoded independently by ``qhat(y | x)``. For a binary
    target the output has one column per feature (``P(y = classes_[1] | x)``);
    for ``ky > 2`` classes it has ``ky`` columns per feature. Unseen categories
    get the training marginal of ``y``. Missing values are their own category.

    ``fit_transform`` returns out-of-fold encodings (``cv`` folds) to avoid
    target leakage into a downstream model, as sklearn's ``TargetEncoder`` does;
    ``fit(X, y).transform(X)`` returns in-sample encodings.

    Parameters
    ----------
    method : {"baseline", "mixture", "crp", "fixed_m", "plugin"}
        "baseline" is the single-beta estimator of Hernandez & Samengo (2019);
        "baseline_unseen" is the same with the unseen correction: categories
        not seen in training get the label rate of the categories seen once;
        "mixture" a finite mixture with ``n_components`` betas; "crp" the
        self-tuning CRP mixture (slowest). "fixed_m" and "plugin" are the
        classic smoothed and unsmoothed encoders, for comparison.
        "glmm", "npmle" and "catboost_ts" are the competitors in
        :mod:`mitarget.competitors` (random-intercept GLMM, nonparametric
        prior, CatBoost's test-time target statistic). "freq_prior" lets the
        prior mean depend on the category's count (:mod:`mitarget.freq_prior`;
        binary targets only), and unseen categories get its prediction at
        count 0 instead of the marginal. It is a binary model, fit
        one-vs-rest for ``ky > 2`` classes.
    n_components : int, number of mixture components for ``method="mixture"``.
    m : float, pseudo-count for ``method="fixed_m"``.
    n_init : int, EM restarts for ``method="mixture"``.
    crp_kwargs : dict, extra arguments to :func:`mitarget.crp.fit_crp_mixture`.
    cv : int, cross-fitting folds in ``fit_transform``.
    shuffle : bool, shuffle before splitting folds.
    random_state : int or None.
    """

    def __init__(self, method: str = "baseline", n_components: int = 2, m: float = 20.0,
                 n_init: int = 4, crp_kwargs: dict | None = None, cv: int = 5,
                 shuffle: bool = True, random_state: int | None = None):
        self.method = method
        self.n_components = n_components
        self.m = m
        self.n_init = n_init
        self.crp_kwargs = crp_kwargs
        self.cv = cv
        self.shuffle = shuffle
        self.random_state = random_state

    def _fit_all(self, X, y):
        X = _as_2d_object(X)
        le = LabelEncoder().fit(np.asarray(y))
        y_codes = le.transform(np.asarray(y))
        ky = len(le.classes_)
        cols = [self._fit_column(X[:, j], y_codes, ky) for j in range(X.shape[1])]
        return X, le, y_codes, cols

    def fit(self, X, y):
        X, le, _, cols = self._fit_all(X, y)
        self._store(X, le, cols)
        return self

    def _store(self, X, le, cols):
        self.classes_ = le.classes_
        self.n_features_in_ = X.shape[1]
        self.categories_ = [c for c, _ in cols]
        self.fits_ = [f for _, f in cols]
        self.encodings_ = [f.encoding for f in self.fits_]
        self.target_mean_ = self.fits_[0].qy if self.fits_ else None

    def _encode(self, X, categories, fits):
        blocks = []
        for j, (cats, f) in enumerate(zip(categories, fits, strict=True)):
            enc = self._lookup(X[:, j], cats, f)
            blocks.append(enc[:, 1:] if enc.shape[1] == 2 else enc)
        return np.hstack(blocks)

    def transform(self, X):
        check_is_fitted(self, "fits_")
        X = _as_2d_object(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, expected {self.n_features_in_}")
        return self._encode(X, self.categories_, self.fits_)

    def fit_transform(self, X, y, **fit_params):
        X, le, y_codes, cols = self._fit_all(X, y)
        self._store(X, le, cols)
        splitter_cls = StratifiedKFold if np.bincount(y_codes).min() >= self.cv else KFold
        splitter = splitter_cls(n_splits=self.cv, shuffle=self.shuffle,
                                random_state=self.random_state if self.shuffle else None)
        ky = len(le.classes_)
        out = None
        for tr, te in splitter.split(X, y_codes):
            fold_cols = [self._fit_column(X[tr, j], y_codes[tr], ky) for j in range(X.shape[1])]
            enc = self._encode(X[te], [c for c, _ in fold_cols], [f for _, f in fold_cols])
            if out is None:
                out = np.empty((X.shape[0], enc.shape[1]))
            out[te] = enc
        return out

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "fits_")
        if input_features is None:
            input_features = [f"x{j}" for j in range(self.n_features_in_)]
        if len(self.classes_) == 2:
            return np.asarray(list(input_features), dtype=object)
        return np.asarray([f"{f}_{c}" for f in input_features for c in self.classes_], dtype=object)


class BayesianCategoricalClassifier(_EstimatorParams, ClassifierMixin, BaseEstimator):
    """
    Classifier on a single high-cardinality categorical feature:
    ``predict_proba(x) = qhat(y | x)``. Accepts a 1-D array or a one-column 2-D input.
    After fitting, ``mutual_information_`` holds the MI estimate in nats.
    Parameters are as in :class:`BayesianTargetEncoder`.
    """

    def __init__(self, method: str = "baseline", n_components: int = 2, m: float = 20.0,
                 n_init: int = 4, crp_kwargs: dict | None = None,
                 random_state: int | None = None):
        self.method = method
        self.n_components = n_components
        self.m = m
        self.n_init = n_init
        self.crp_kwargs = crp_kwargs
        self.random_state = random_state

    @staticmethod
    def _column(X) -> np.ndarray:
        X = _as_2d_object(X)
        if X.shape[1] != 1:
            raise ValueError("BayesianCategoricalClassifier takes exactly one categorical feature")
        return X[:, 0]

    def fit(self, X, y):
        le = LabelEncoder().fit(np.asarray(y))
        self.classes_ = le.classes_
        self.n_features_in_ = 1
        self.categories_, self.fit_ = self._fit_column(
            self._column(X), le.transform(np.asarray(y)), len(self.classes_))
        self.mutual_information_ = self.fit_.I_hat
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "fit_")
        return self._lookup(self._column(X), self.categories_, self.fit_)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
