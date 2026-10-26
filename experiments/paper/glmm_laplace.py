"""
GLMM encoder fitted with the Laplace approximation (n_agq = 1, lme4's default) against adaptive Gauss-Hermite
quadrature with 15 nodes (the paper's GLMM), and the beta baseline (Appendix, GLMM fitting). Held-out log-loss
over stratified 5-fold CV, with the fitted intercept and random-intercept sd, on singleton-heavy real features
and on synthetic beta data, where the exact excess cross-entropy is also reported.

    python glmm_laplace.py --arff-dir code/data [--avazu avazu_train_4M.csv.gz]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold

from mitarget import counts_from_labels, datasets, fit_baseline, fit_glmm, qhat_y

FITS = {
    "baseline": lambda n: (fit_baseline(n, qy=qhat_y(n)).target_encoding[:, 1], None),
    "GLMM AGQ-15": lambda n: _glmm(n, 15),
    "GLMM Laplace": lambda n: _glmm(n, 1),
}


def _glmm(n, n_agq):
    f = fit_glmm(n, n_agq=n_agq)
    return f.target_encoding[:, 1], (f.intercept, f.sigma)


def logloss(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def run_cv(label, x, y, kx, folds=5, seed=0):
    rows = []
    for i, (tr, te) in enumerate(StratifiedKFold(folds, shuffle=True, random_state=seed).split(x, y)):
        n = counts_from_labels(x[tr], y[tr], kx, 2)
        nx = n.sum(axis=1)
        for name, fit in FITS.items():
            enc, par = fit(n)
            rows.append(dict(task=label, fold=i, method=name, logloss=logloss(enc[x[te]], y[te]),
                             b0=par[0] if par else np.nan, sigma=par[1] if par else np.nan,
                             levels_seen=int((nx > 0).sum()),
                             singleton_share=float((nx == 1).sum() / (nx > 0).sum())))
        marginal = np.full(len(te), y[tr].mean())
        rows.append(dict(task=label, fold=i, method="marginal", logloss=logloss(marginal, y[te])))
    return rows


def synthetic(K=20000, nbar=1.0, mu=0.3, s=5.0, reps=10, seed=0):
    """Beta(mu s, (1 - mu) s) conditionals, equal popularity, Poisson(nbar) level sizes: exact eCE."""
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(reps):
        q = rng.beta(mu * s, (1 - mu) * s, K)
        m = rng.poisson(nbar, K)
        k = rng.binomial(m, q)
        n = np.stack([m - k, k], axis=1).astype(float)
        for name, fit in FITS.items():
            enc, par = fit(n)
            enc = np.clip(enc, 1e-12, 1 - 1e-12)
            ece = np.mean(q * np.log(q / enc) + (1 - q) * np.log((1 - q) / (1 - enc)))
            rows.append(dict(task=f"synthetic beta nbar={nbar}", fold=r, method=name, eCE=ece,
                             b0=par[0] if par else np.nan, sigma=par[1] if par else np.nan))
    return rows


def profile(x, y, kx, sigmas):
    """Profile log-likelihood in sigma (maximised over b0) under AGQ-15 and Laplace, on all rows."""
    from scipy.optimize import minimize_scalar

    from mitarget.competitors import _binary_groups, _glmm_loglik

    n = counts_from_labels(x, y, kx, 2)
    _, gk1, gm, mult, _ = _binary_groups(n[:, 1], n.sum(axis=1))
    out = []
    for sg in sigmas:
        for q in (15, 1):
            r = minimize_scalar(lambda b0, sg=sg, q=q: -_glmm_loglik(b0, sg, gk1, gm, mult, q),
                                bounds=(-10, 10), method="bounded")
            out.append(dict(sigma=sg, n_agq=q, b0=r.x, loglik=-r.fun, mean_q=float(expit(r.x))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arff-dir", type=Path, required=True)
    ap.add_argument("--avazu", type=Path, default=None, help="raw Avazu train (first rows used)")
    ap.add_argument("--avazu-rows", type=int, default=500_000)
    ap.add_argument("--out", type=Path, default=Path("glmm_laplace"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    f = datasets.DATA_FILES
    tasks = [datasets.load_churn_composite_id(a.arff_dir / f["churn"]),
             datasets.load_click_prediction(a.arff_dir / f["click"], cat_col="user_id"),
             datasets.load_employee_salaries(a.arff_dir / f["employee_salaries"])]
    rows = []
    for t in tasks:
        if t.ky != 2:
            continue
        rows += run_cv(t.label, t.x, t.y, t.kx)
    if a.avazu is not None:
        d = pd.read_csv(a.avazu, dtype=str, usecols=["click", "device_ip"], nrows=a.avazu_rows)
        x, u = pd.factorize(d.device_ip)
        label = f"avazu/device_ip ({a.avazu_rows // 1000}k rows)"
        rows += run_cv(label, x, d.click.astype(int).values, len(u))
    cv = pd.DataFrame(rows)
    cv.to_csv(a.out / "cv_raw.csv", index=False)
    print(cv.groupby(["task", "method"])[["logloss", "b0", "sigma", "singleton_share"]].mean().to_string())
    syn = pd.DataFrame(synthetic(nbar=1.0) + synthetic(nbar=2.0))
    syn.to_csv(a.out / "synthetic_raw.csv", index=False)
    print(syn.groupby(["task", "method"])[["eCE", "b0", "sigma"]].agg(["mean", "std"]).to_string())
    t = tasks[0]
    prof = pd.DataFrame(profile(t.x, t.y, t.kx, np.exp(np.linspace(np.log(0.05), np.log(10), 25))))
    prof.to_csv(a.out / "profile_churn_pseudo_id.csv", index=False)
    print(prof.pivot(index="sigma", columns="n_agq", values="loglik").to_string())


if __name__ == "__main__":
    main()
