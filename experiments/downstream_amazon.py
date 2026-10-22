"""
Downstream multi-feature test on the Amazon employee access data (Kaggle
"Amazon.com - Employee Access Challenge", train.csv: 32,769 rows, y = ACTION).

Each encoder in turn replaces every categorical column by its out-of-fold
target encoding, and a downstream model (logistic regression on the logit of
the encodings, or sklearn's histogram gradient boosting) is trained on them.

Leakage control is nested:

- outer repeated stratified K-fold splits the rows into train / test;
- on the outer train part, ``BayesianTargetEncoder.fit_transform`` gives
  out-of-fold encodings (``--inner-cv`` folds) for the downstream fit;
- the outer test part is encoded by the encoder fit on the whole outer train.

Every encoder sees the same outer and inner splits, so the per-fold metric
differences are paired. Differences against the baseline are tested with the
Nadeau-Bengio corrected resampled t-test, since repeated-CV folds overlap.

Feature sets: ``raw`` (the 8 columns, ROLE_CODE dropped since it is a
relabelling of ROLE_TITLE) and ``pairs`` (raw plus all 28 pairwise
concatenations, the usual winning trick on this data; most pairs are large
and under-sampled). Optional references that do not use a target encoder:
one-hot logistic regression and CatBoost on the raw categoricals.

Usage::

    python experiments/downstream_amazon.py --data path/to/train.csv --out results/
    python experiments/downstream_amazon.py --data train.csv --fast   # smoke run
"""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import special, stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mitarget import BayesianTargetEncoder

ENCODERS = {
    "baseline": dict(method="baseline"),
    "npmle": dict(method="npmle"),
    "freq_prior": dict(method="freq_prior"),
    "glmm": dict(method="glmm"),
    "fixed_m20": dict(method="fixed_m", m=20.0),
    "catboost_ts": dict(method="catboost_ts"),
}
RAW_COLUMNS = ["RESOURCE", "MGR_ID", "ROLE_ROLLUP_1", "ROLE_ROLLUP_2", "ROLE_DEPTNAME", "ROLE_TITLE",
               "ROLE_FAMILY_DESC", "ROLE_FAMILY"]
EPS = 1e-6


def load(path: Path, feature_set: str) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(path)
    y = df["ACTION"].to_numpy()
    X = df[RAW_COLUMNS].astype(str)
    if feature_set == "pairs":
        pairs = {f"{a}*{b}": X[a] + "_" + X[b] for a, b in itertools.combinations(RAW_COLUMNS, 2)}
        X = pd.concat([X, pd.DataFrame(pairs)], axis=1)
    return X, y


def downstream_models(seed: int):
    return {
        "logreg": lambda: LogisticRegression(C=1.0, max_iter=2000),
        "gbm": lambda: HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=31,
                                                      l2_regularization=1.0, early_stopping=False,
                                                      random_state=seed),
    }


def features(enc: np.ndarray, model: str) -> np.ndarray:
    # Linear model works on the logit scale (the natural scale of a log-odds sum).
    return special.logit(np.clip(enc, EPS, 1 - EPS)) if model == "logreg" else enc


def score(y, p) -> dict:
    p = np.clip(p, EPS, 1 - EPS)
    return {"log_loss": log_loss(y, p, labels=[0, 1]), "auc": roc_auc_score(y, p)}


def run_fold(X, y, tr, te, encoders, inner_cv, seed, models):
    rows = []
    Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y[tr], y[te]
    for name, params in encoders.items():
        t0 = time.time()
        enc = BayesianTargetEncoder(cv=inner_cv, random_state=seed, **params)
        ftr = enc.fit_transform(Xtr, ytr)   # out-of-fold on the outer train part
        fte = enc.transform(Xte)            # fit on the whole outer train part
        t_enc = time.time() - t0
        for mname, make in models.items():
            scaler = StandardScaler().fit(features(ftr, mname))
            model = make().fit(scaler.transform(features(ftr, mname)), ytr)
            p = model.predict_proba(scaler.transform(features(fte, mname)))[:, 1]
            rows.append({"encoder": name, "model": mname, "enc_seconds": t_enc, **score(yte, p)})
    return rows


def run_references(X_raw, y, tr, te, seed, use_catboost):
    rows = []
    oh = OneHotEncoder(handle_unknown="ignore").fit(X_raw.iloc[tr])
    lr = LogisticRegression(C=1.0, max_iter=5000).fit(oh.transform(X_raw.iloc[tr]), y[tr])
    rows.append({"encoder": "onehot", "model": "logreg",
                 **score(y[te], lr.predict_proba(oh.transform(X_raw.iloc[te]))[:, 1])})
    if use_catboost:
        from catboost import CatBoostClassifier
        cb = CatBoostClassifier(iterations=500, random_seed=seed, verbose=0, allow_writing_files=False,
                                thread_count=-1)
        cb.fit(X_raw.iloc[tr], y[tr], cat_features=list(range(X_raw.shape[1])))
        rows.append({"encoder": "catboost_native", "model": "catboost",
                     **score(y[te], cb.predict_proba(X_raw.iloc[te])[:, 1])})
    return rows


def one_fold(X, X_raw, y, fs, i, tr, te, encoders, args, use_catboost, n_splits):
    t0 = time.time()
    rows = run_fold(X, y, tr, te, encoders, args.inner_cv, args.seed + i, downstream_models(args.seed + i))
    if fs == "raw" and not args.no_references:
        rows += run_references(X_raw, y, tr, te, args.seed + i, use_catboost)
    for r in rows:
        r.update(feature_set=fs, fold=i)
    print(f"  [{fs}] fold {i + 1}/{n_splits} done in {time.time() - t0:.0f}s", flush=True)
    return rows


def corrected_ttest(d: np.ndarray, n_train: int, n_test: int) -> tuple[float, float]:
    """Nadeau & Bengio (2003) corrected resampled t-test on per-fold differences ``d``."""
    k = len(d)
    var = np.var(d, ddof=1)
    if var == 0:
        return float("nan"), float("nan")
    t = d.mean() / np.sqrt(var * (1.0 / k + n_test / n_train))
    return float(t), float(2 * stats.t.sf(abs(t), df=k - 1))


def paired_summary(raw: pd.DataFrame, n_train: int, n_test: int, reference: str = "baseline") -> pd.DataFrame:
    out = []
    for (fs, model), g in raw.groupby(["feature_set", "model"]):
        ref = g[g.encoder == reference].set_index("fold")
        for name, h in g.groupby("encoder", sort=False):
            h = h.set_index("fold")
            row = {"feature_set": fs, "model": model, "encoder": name, "folds": len(h),
                   "log_loss": h.log_loss.mean(), "log_loss_sd": h.log_loss.std(),
                   "auc": h.auc.mean(), "auc_sd": h.auc.std()}
            common = h.index.intersection(ref.index)
            if name != reference and len(common) > 1:
                for metric, better in (("log_loss", -1), ("auc", 1)):
                    d = (h.loc[common, metric] - ref.loc[common, metric]).to_numpy()
                    t, p = corrected_ttest(d, n_train, n_test)
                    row[f"d_{metric}"] = d.mean()
                    row[f"p_{metric}"] = p
                    row[f"wins_{metric}"] = f"{int(np.sum(better * d > 0))}/{len(d)}"
            out.append(row)
    return pd.DataFrame(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="Amazon employee access train.csv")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--feature-sets", nargs="+", default=["raw", "pairs"], choices=["raw", "pairs"])
    ap.add_argument("--encoders", nargs="+", default=list(ENCODERS), choices=list(ENCODERS))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--inner-cv", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=1, help="outer folds run in parallel")
    ap.add_argument("--no-references", action="store_true", help="skip one-hot LR and native CatBoost")
    ap.add_argument("--fast", action="store_true", help="3 folds x 1 repeat, raw features only")
    args = ap.parse_args(argv)
    if args.fast:
        args.folds, args.repeats, args.feature_sets = 3, 1, ["raw"]

    try:
        import catboost  # noqa: F401
        use_catboost = True
    except ImportError:
        use_catboost = False

    encoders = {k: ENCODERS[k] for k in args.encoders}
    X_raw, y = load(args.data, "raw")
    splitter = RepeatedStratifiedKFold(n_splits=args.folds, n_repeats=args.repeats, random_state=args.seed)
    splits = list(splitter.split(X_raw, y))
    rows = []
    for fs in args.feature_sets:
        X, _ = load(args.data, fs)
        print(f"[{fs}] {X.shape[1]} columns, levels per column: "
              f"{', '.join(str(X[c].nunique()) for c in X.columns[:8])}{' ...' if X.shape[1] > 8 else ''}")
        jobs = (delayed(one_fold)(X, X_raw, y, fs, i, tr, te, encoders, args, use_catboost, len(splits))
                for i, (tr, te) in enumerate(splits))
        results = Parallel(n_jobs=args.jobs)(jobs)
        rows += [r for fold_rows in results for r in fold_rows]
    raw = pd.DataFrame(rows)
    n_test = len(y) // args.folds
    summary = paired_summary(raw, len(y) - n_test, n_test)
    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.precision", 4):
        print(summary.drop(columns=["log_loss_sd", "auc_sd"]).to_string(index=False))
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        raw.to_csv(args.out / "downstream_amazon_raw.csv", index=False)
        summary.to_csv(args.out / "downstream_amazon_summary.csv", index=False)


if __name__ == "__main__":
    main()
