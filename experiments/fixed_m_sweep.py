"""
Fixed-smoothing sweep: how much does the m-estimate lose as its pseudo-count m moves away from the
marginal-likelihood value s-hat?

The m-estimate ``(k_x + m mu) / (n_x + m)`` with ``m = s-hat`` is the baseline, so every encoder here
differs from the baseline only by its m. Two designs:

``tasks``  the 24 real features of ``realdata_paired.py``, same repeated k-fold splits (seeds 0..reps-1).
           Per fold: s-hat, the leave-one-out CV choice of m, and the held-out log-loss of the baseline, the
           CV m-estimate, a common grid of fixed m and the relative grid m = c * s-hat.
``pools``  the real-data subsampling oracles of ``realdata_oracle.py`` (same seeds and draws), where the
           excess cross-entropy is exact. Per training set: s-hat, CV m and the eCE of the same encoders,
           next to the marginal and full-oracle eCE.

    python experiments/fixed_m_sweep.py tasks --data-root DATA --out sweep_tasks/
    python experiments/fixed_m_sweep.py pools --data movies.tsv --feature movie_id \
        --nbar 1 2 5 --out sweep_pools/
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from mitarget.baseline import fit_baseline, fit_cv_m
from mitarget.core import counts_from_labels, posterior_mean_conditional, qhat_y

sys.path.insert(0, str(Path(__file__).resolve().parent))
from realdata_tasks import GROUPS  # noqa: E402

#: common fixed pseudo-counts (20 is the category_encoders default, 1 is CatBoost's prior weight)
M_GRID = (0.1, 0.3, 1.0, 3.0, 10.0, 20.0, 30.0, 100.0, 300.0, 1000.0)
#: multiples of the marginal-likelihood value s-hat
REL_GRID = (1 / 16, 1 / 4, 1 / 2, 2.0, 4.0, 16.0)


def encoders(n: np.ndarray) -> tuple[dict, float, float]:
    """All m-estimates of one training count matrix: name -> (m, (kx, ky) encoding), with s-hat and CV m."""
    qy = qhat_y(n)
    s = float(fit_baseline(n, qy=qy).beta_star)
    m_cv, enc_cv = fit_cv_m(n, qy=qy)
    out = {"baseline": (s, posterior_mean_conditional(n, s, qy)), "cv_m": (m_cv, enc_cv)}
    for m in M_GRID:
        out[f"fixed_m{m:g}"] = (m, posterior_mean_conditional(n, m, qy))
    for c in REL_GRID:
        out[f"rel_{c:g}"] = (c * s, posterior_mean_conditional(n, c * s, qy))
    return out, s, m_cv


def run_tasks(args):
    from realdata_paired import task_tag
    args.out.mkdir(parents=True, exist_ok=True)
    for group in args.groups:
        loader, sub = GROUPS[group]
        for task in loader(args.data_root / sub):
            if args.tasks and task.label not in args.tasks:
                continue
            t0 = time.time()
            x, y, kx, ky = task.x, task.y, task.kx, task.ky
            rows = []
            for rep in range(args.reps):
                skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=rep)
                for fold, (tr, te) in enumerate(skf.split(x, y)):
                    n_tr = counts_from_labels(x[tr], y[tr], kx, ky)
                    encs, s, m_cv = encoders(n_tr)
                    for name, (m, enc) in encs.items():
                        ll = -np.log(np.clip(enc[x[te], y[te]], 1e-12, 1.0)).mean()
                        rows.append(dict(task=task.label, rep=rep, fold=fold, method=name, m=m, s_hat=s,
                                         m_cv=m_cv, logloss=ll))
            pd.DataFrame(rows).to_csv(args.out / f"sweep_{task_tag(task.label)}.csv", index=False)
            print(f"{task.label}: {time.time() - t0:.0f}s", flush=True)


def run_pools(args):
    from realdata_oracle import load_population, run_cell
    args.out.mkdir(parents=True, exist_ok=True)
    # same population, permutation and draw order as realdata_oracle.py
    p, q, _ = load_population(args.data, args.feature, args.min_rows, tail=args.tail, seed=args.seed + 2)
    if args.shuffle:
        q = np.random.default_rng(args.seed + 1).permutation(q)
    rng = np.random.default_rng(args.seed)
    name = args.name or args.feature
    rows = []
    for nbar in args.nbar:
        N = int(round(nbar * len(p)))
        for rep in range(args.reps):
            store = {}

            def grab(n, key, store=store):
                if "encs" not in store:
                    store["encs"] = encoders(n)
                return store["encs"][0][key][1][:, 1]

            keys = ["baseline", "cv_m"] + [f"fixed_m{m:g}" for m in M_GRID] + [f"rel_{c:g}" for c in REL_GRID]
            fitted = {k: (lambda n, k=k: grab(n, k)) for k in keys}
            for r in run_cell(p, q, N, rng, fitted=fitted):
                if r["method"] not in fitted and r["method"] not in ("marginal", "oracle"):
                    continue
                encs, s, m_cv = store["encs"]
                r.update(pool=name, shuffled=args.shuffle, nbar=nbar, rep=rep, s_hat=s, m_cv=m_cv,
                         m=encs[r["method"]][0] if r["method"] in encs else np.nan)
                rows.append(r)
        print(f"{name} nbar={nbar} done", flush=True)
    tag = "shuffled" if args.shuffle else "real"
    pd.DataFrame(rows).to_csv(args.out / f"sweep_{name}_{tag}{args.suffix}.csv", index=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tasks")
    t.add_argument("--data-root", type=Path, required=True)
    t.add_argument("--groups", nargs="+", default=list(GROUPS), choices=list(GROUPS))
    t.add_argument("--tasks", nargs="*", default=None)
    t.add_argument("--reps", type=int, default=5)
    t.add_argument("--folds", type=int, default=5)
    t.add_argument("--out", type=Path, required=True)
    o = sub.add_parser("pools")
    o.add_argument("--data", type=Path, required=True)
    o.add_argument("--feature", required=True)
    o.add_argument("--name", default=None)
    o.add_argument("--min-rows", type=int, default=100)
    o.add_argument("--tail", choices=["shifted-head"], default=None)
    o.add_argument("--reps", type=int, default=20)
    o.add_argument("--seed", type=int, default=2026)
    o.add_argument("--nbar", type=float, nargs="*", default=[1.0, 2.0, 5.0])
    o.add_argument("--shuffle", action="store_true")
    o.add_argument("--suffix", default="")
    o.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    (run_tasks if a.cmd == "tasks" else run_pools)(a)


if __name__ == "__main__":
    main()
