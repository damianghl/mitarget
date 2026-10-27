"""
Evaluation harness shared by the benchmark scripts.

- :func:`evaluate_synthetic`: repeated draws of N samples from a fixed ground
  truth, scored by MI error and excess cross-entropy. Every method, oracles
  included, is scored on the same draw in each repetition: the gaps between
  methods are often smaller than draw-to-draw noise.
- :func:`evaluate_cv`: k-fold held-out log-loss (and AUC for binary targets)
  on a real categorical feature.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

from .core import counts_from_labels, labels_from_counts, qhat_y
from .crp import DEFAULT_ALPHA_INIT, DEFAULT_ALPHA_PRIOR
from .competitors import catboost_classifier_encoding
from .encoders import fit_counts
from .metrics import excess_cross_entropy, heldout_log_loss
from .synthetic.generative import GroundTruth, observe

#: A method maps a training count matrix (plus the training label arrays, for
#: encoders that need raw samples) to ``(encoding, I_hat)``.
Method = Callable[[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, float]]

CRP_FAST = dict(alpha_crp=DEFAULT_ALPHA_INIT, alpha_prior=DEFAULT_ALPHA_PRIOR,
                n_sweeps=30, burn_in=10, thin=3)


def sklearn_target_encoding(x: np.ndarray, y: np.ndarray, kx: int, ky: int) -> np.ndarray:
    """
    sklearn ``TargetEncoder(smooth="auto")`` expanded to a full ``(kx, ky)``
    matrix; categories absent from training get the empirical marginal.
    """
    te = TargetEncoder(target_type="binary" if ky == 2 else "multiclass", smooth="auto")
    te.fit(np.asarray(x).reshape(-1, 1), y)
    enc = np.tile(qhat_y(counts_from_labels(x, y, kx, ky)), (kx, 1))
    cats = np.asarray(te.categories_[0], dtype=int)
    if ky == 2:
        p1 = np.asarray(te.encodings_[0])
        enc[cats, 1] = p1
        enc[cats, 0] = 1 - p1
    else:
        for cls in range(ky):
            enc[cats, cls] = np.asarray(te.encodings_[cls])
        enc /= enc.sum(axis=1, keepdims=True)
    return enc


def _from_counts(method: str, **kwargs) -> Method:
    def run(n, x, y):
        f = fit_counts(n, method, **kwargs)
        return f.encoding, f.I_hat
    return run


def catboost_available() -> bool:
    try:
        import catboost  # noqa: F401
    except ImportError:
        return False
    return True


def standard_methods(mixture_ks=(2, 3), crp: bool = True, crp_kwargs: dict | None = None,
                     fixed_m: float = 20.0, sklearn: bool = True, competitors: bool = True,
                     catboost: bool | None = None, freq_prior: bool = False,
                     unseen_correction: bool = False) -> dict[str, Method]:
    """
    The method line-up of the original experiments, in display order.

    ``competitors`` adds the GLMM, NPMLE and CatBoost target-statistic
    encoders; ``catboost`` the full CatBoost classifier (default: when the
    ``catboost`` package is installed). ``freq_prior`` adds the
    frequency-aware prior (binary targets only); ``unseen_correction`` the
    baseline with the singleton rate for unseen categories, and the same with
    the doubleton (Robbins) rates for categories seen once.
    """
    methods: dict[str, Method] = {
        "Plug-in": _from_counts("plugin"),
        f"Fixed m={fixed_m:g}": _from_counts("fixed_m", m=fixed_m),
    }
    if sklearn:
        methods["sklearn TargetEncoder (auto)"] = (
            lambda n, x, y: (sklearn_target_encoding(x, y, *n.shape), float("nan")))
    methods["K=1 baseline"] = _from_counts("baseline")
    if unseen_correction:
        methods["Baseline + unseen correction"] = _from_counts("baseline_unseen")
        methods["Baseline + Robbins (n<=1)"] = _from_counts("baseline_robbins")
    if freq_prior:
        methods["Frequency prior"] = _from_counts("freq_prior")
    for K in mixture_ks:
        methods[f"K={K} mixture"] = _from_counts("mixture", n_components=K, random_state=K)
    if crp:
        methods["CRP (self-tuned)"] = _from_counts("crp", crp_kwargs=crp_kwargs or CRP_FAST)
    if competitors:
        methods["GLMM"] = _from_counts("glmm")
        methods["NPMLE"] = _from_counts("npmle")
        methods["CatBoost TS (a=1)"] = _from_counts("catboost_ts")
    if catboost if catboost is not None else (competitors and catboost_available()):
        methods["CatBoost classifier"] = (
            lambda n, x, y: (catboost_classifier_encoding(x, y, *n.shape), float("nan")))
    return methods


def evaluate_synthetic(gt: GroundTruth, N: int, n_reps: int,
                       methods: Mapping[str, Method],
                       oracles: Mapping[str, Callable[[np.ndarray], np.ndarray]] | None = None,
                       reps_per_method: Mapping[str, int] | None = None,
                       seed: int = 0) -> pd.DataFrame:
    """
    One row per (method, repetition) with ``I_hat``, ``MI_error``, ``excess_CE``
    and ``seconds``. ``oracles`` map a count matrix to an encoding (no MI).
    ``reps_per_method`` caps slow methods (e.g. CRP) to fewer repetitions; they
    run on the first draws, so they stay paired with the other methods.
    """
    rng = np.random.default_rng(seed)
    reps_per_method = reps_per_method or {}
    rows = []
    for r in range(n_reps):
        n = observe(gt, N, seed=int(rng.integers(0, 2**31 - 1)))
        x, y = labels_from_counts(n)
        for name, fn in methods.items():
            if r >= reps_per_method.get(name, n_reps):
                continue
            t0 = time.perf_counter()
            enc, I_hat = fn(n, x, y)
            rows.append(dict(method=name, rep=r, N=N, I_hat=I_hat, MI_error=I_hat - gt.I_true,
                             excess_CE=excess_cross_entropy(gt.qx, gt.q_ygivenx, enc),
                             seconds=time.perf_counter() - t0))
        for name, fn in (oracles or {}).items():
            rows.append(dict(method=name, rep=r, N=N, I_hat=np.nan, MI_error=np.nan,
                             excess_CE=excess_cross_entropy(gt.qx, gt.q_ygivenx, fn(n)),
                             seconds=np.nan))
    return pd.DataFrame(rows)


def evaluate_cv(x: np.ndarray, y: np.ndarray, methods: Mapping[str, Method],
                n_splits: int = 5, seed: int = 0, kx: int | None = None,
                ky: int | None = None) -> pd.DataFrame:
    """One row per (method, fold) with held-out ``logloss``, ``auc`` (binary) and ``seconds``."""
    x = np.asarray(x, dtype=int)
    y = np.asarray(y, dtype=int)
    kx = int(x.max()) + 1 if kx is None else kx
    ky = int(y.max()) + 1 if ky is None else ky
    rows = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr, te) in enumerate(skf.split(x, y)):
        n_tr = counts_from_labels(x[tr], y[tr], kx, ky)
        for name, fn in methods.items():
            t0 = time.perf_counter()
            enc, _ = fn(n_tr, x[tr], y[tr])
            row = dict(method=name, fold=fold,
                       logloss=heldout_log_loss(enc, x[te], y[te]),
                       seconds=time.perf_counter() - t0)
            if ky == 2:
                try:
                    row["auc"] = float(roc_auc_score(y[te], enc[x[te], 1]))
                except ValueError:
                    row["auc"] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, metrics=("MI_error", "excess_CE"),
              by=("method",)) -> pd.DataFrame:
    """Mean and standard error per group for each metric present in ``df``."""
    metrics = [m for m in metrics if m in df.columns]
    def sem(v):
        v = np.asarray(v, dtype=float)
        v = v[~np.isnan(v)]
        return v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
    agg = {}
    for m in metrics:
        agg[f"{m}_mean"] = (m, "mean")
        agg[f"{m}_sem"] = (m, sem)
    agg["n"] = (metrics[0], "size")
    return df.groupby(list(by), sort=False).agg(**agg)
