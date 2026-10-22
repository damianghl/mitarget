"""
Repeated k-fold benchmark on real high-cardinality features with paired tests.

    python experiments/realdata_paired.py --data-root DATA --groups openml movielens --out results/
    python experiments/realdata_paired.py --data-root DATA --groups criteo --reps 2 --out results/

Every method is fit on the same training folds, so per-fold and per-row losses
are paired. For each task the script writes to ``--out``:

- ``meta_<task>.csv``: size, cardinality, singleton/doubleton shares and ``H(Y)``.
- ``folds_<task>.csv``: one row per (rep, fold, method) with held-out log-loss,
  AUC (binary) and fit time.
- ``rows_<task>.csv``: one row per (rep, method A, method B) with the
  row-level paired difference ``mean(loss_A - loss_B)`` over all out-of-fold
  rows of that repetition and its standard error with levels as independent
  clusters (rows of one level share an encoding, so they are not independent).
- ``bins_<task>.csv``: one row per (rep, method, training-count bin) with the
  out-of-fold loss of test rows whose level was seen ``n`` times in training.

``experiments/realdata_tables.py`` turns these into the paper tables.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from mitarget.benchmark import sklearn_target_encoding
from mitarget.core import counts_from_labels, qhat_y
from mitarget.encoders import fit_counts

sys.path.insert(0, str(Path(__file__).resolve().parent))
from realdata_tasks import GROUPS  # noqa: E402

BASELINE = "K=1 baseline"
UNSEEN = "Baseline + unseen correction"

#: name -> function of the training count matrix and training labels returning a (kx, ky) encoding
METHODS = {
    "Prior only": lambda n, x, y: np.tile(qhat_y(n), (n.shape[0], 1)),
    "Fixed m=20": lambda n, x, y: fit_counts(n, "fixed_m", m=20.0).encoding,
    BASELINE: lambda n, x, y: fit_counts(n, "baseline").encoding,
    UNSEEN: lambda n, x, y: fit_counts(n, "baseline_unseen").encoding,
    "Frequency prior": lambda n, x, y: fit_counts(n, "freq_prior").encoding,
    "GLMM": lambda n, x, y: fit_counts(n, "glmm").encoding,
    "NPMLE": lambda n, x, y: fit_counts(n, "npmle").encoding,
    "CatBoost TS (a=1)": lambda n, x, y: fit_counts(n, "catboost_ts").encoding,
    "sklearn TargetEncoder (auto)": lambda n, x, y: sklearn_target_encoding(x, y, *n.shape),
}

#: training-count bins [lo, hi] for the per-n breakdown
BINS = [(0, 0), (1, 1), (2, 2), (3, 5), (6, 10), (11, 30), (31, 100), (101, np.inf)]


def bin_label(lo, hi) -> str:
    if lo == hi:
        return str(lo)
    return f">{lo - 1}" if np.isinf(hi) else f"{lo}-{hi}"


def task_tag(label: str) -> str:
    return "".join(c if c.isalnum() or c in "-_=" else "_" for c in label.replace("/", "__"))


def clustered_diff(a: np.ndarray, b: np.ndarray, x: np.ndarray, kx: int) -> tuple[float, float]:
    """
    Mean of ``a - b`` and its standard error with levels ``x`` as independent clusters.

    The variance is ``sum_c D_c^2 / N^2`` over the level totals ``D_c`` of the
    difference, without centering. Centering by the row mean would assume that
    a level's expected difference scales with its size; on click logs one
    level can hold a quarter of the rows while the methods differ only on the
    small levels, and the centered estimator then reports ``z ~ N / n_max``.
    """
    d = (a - b).astype(float)
    N = len(d)
    D = np.bincount(x, weights=d, minlength=kx)
    if np.count_nonzero(np.bincount(x, minlength=kx)) < 2:
        return float(d.mean()), float("nan")
    return float(d.mean()), float(np.sqrt(np.sum(D ** 2))) / N


def task_meta(task) -> dict:
    vc = np.bincount(task.x, minlength=task.kx)
    F1, F2 = np.sum(vc == 1), np.sum(vc == 2)
    qy = np.bincount(task.y, minlength=task.ky) / task.n
    return dict(task=task.label, N=task.n, kx=task.kx, ky=task.ky, N_per_level=task.n / task.kx,
                F1_share=F1 / task.n, F1_2F2_share=(F1 + 2 * F2) / task.n,
                H_Y=float(-np.sum(qy[qy > 0] * np.log(qy[qy > 0]))))


def run_task(task, methods, reps: int, folds: int, out: Path, eps: float = 1e-12):
    x, y, kx, ky = task.x, task.y, task.kx, task.ky
    N = len(y)
    tag = task_tag(task.label)
    pd.DataFrame([task_meta(task)]).to_csv(out / f"meta_{tag}.csv", index=False)
    fold_rows, pair_rows, bin_rows = [], [], []
    for rep in range(reps):
        loss = {m: np.empty(N, dtype=np.float64) for m in methods}
        n_seen = np.empty(N, dtype=np.int64)
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=rep)
        for fold, (tr, te) in enumerate(skf.split(x, y)):
            n_tr = counts_from_labels(x[tr], y[tr], kx, ky)
            n_seen[te] = n_tr.sum(axis=1)[x[te]]
            for name, fn in methods.items():
                t0 = time.perf_counter()
                enc = fn(n_tr, x[tr], y[tr])
                sec = time.perf_counter() - t0
                lt = -np.log(np.clip(enc[x[te], y[te]], eps, 1.0))
                loss[name][te] = lt
                row = dict(task=task.label, rep=rep, fold=fold, method=name, logloss=lt.mean(),
                           seconds=sec, n_train=len(tr), n_test=len(te))
                if ky == 2:
                    try:
                        row["auc"] = float(roc_auc_score(y[te], enc[x[te], 1]))
                    except ValueError:
                        row["auc"] = np.nan
                fold_rows.append(row)
        for a, b in itertools.combinations(methods, 2):
            mean, se = clustered_diff(loss[a], loss[b], x, kx)
            pair_rows.append(dict(task=task.label, rep=rep, method_a=a, method_b=b,
                                  mean_diff=mean, se_clustered=se, N=N))
        for lo, hi in BINS:
            mask = (n_seen >= lo) & (n_seen <= hi)
            if not mask.any():
                continue
            for name in methods:
                r = dict(task=task.label, rep=rep, method=name, bin=bin_label(lo, hi), n_lo=lo,
                         rows=int(mask.sum()), logloss=float(loss[name][mask].mean()))
                if name != BASELINE:
                    r["diff_vs_baseline"], r["se_clustered"] = clustered_diff(
                        loss[name][mask], loss[BASELINE][mask], x[mask], kx)
                bin_rows.append(r)
        print(f"   rep {rep}: " + "  ".join(f"{m} {loss[m].mean():.5f}" for m in methods), flush=True)
    pd.DataFrame(fold_rows).to_csv(out / f"folds_{tag}.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(out / f"rows_{tag}.csv", index=False)
    pd.DataFrame(bin_rows).to_csv(out / f"bins_{tag}.csv", index=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, required=True,
                    help="folder holding one sub-path per group (see realdata_tasks.GROUPS)")
    ap.add_argument("--groups", nargs="+", default=list(GROUPS), choices=list(GROUPS))
    ap.add_argument("--tasks", nargs="*", default=None, help="only these task labels (e.g. bx/user)")
    ap.add_argument("--reps", type=int, default=5, help="repetitions of k-fold (seeds 0..reps-1)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--methods", nargs="*", default=list(METHODS), choices=list(METHODS))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--meta-only", action="store_true", help="only write meta_<task>.csv")
    args = ap.parse_args(argv)
    if BASELINE not in args.methods:
        ap.error(f"{BASELINE!r} is the reference method and must be included")

    args.out.mkdir(parents=True, exist_ok=True)
    methods = {m: METHODS[m] for m in args.methods}
    for group in args.groups:
        loader, sub = GROUPS[group]
        for task in loader(args.data_root / sub):
            if args.tasks and task.label not in args.tasks:
                continue
            if args.meta_only:
                pd.DataFrame([task_meta(task)]).to_csv(args.out / f"meta_{task_tag(task.label)}.csv",
                                                       index=False)
                continue
            t0 = time.time()
            print(f"== {task.label}: N={task.n} kx={task.kx} ky={task.ky}", flush=True)
            run_task(task, methods, args.reps, args.folds, args.out)
            print(f"   done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
