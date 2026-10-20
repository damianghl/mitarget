"""
Real-data bake-off: k-fold held-out log-loss (and AUC for binary targets) of
every encoder on high-cardinality features from four OpenML datasets.

    python benchmarks/realdata_bakeoff.py --data-dir path/to/arff/
    python benchmarks/realdata_bakeoff.py --data-dir data/ --fast --tasks employee_salaries

Held-out log-loss equals excess cross-entropy plus the constant H(Y|X), so
rankings within a dataset match the synthetic metric.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mitarget import datasets
from mitarget.benchmark import CRP_FAST, evaluate_cv, standard_methods, summarize

#: key -> (loader, file key, extra loader kwargs, folds in full mode)
TASKS = {
    "employee_salaries": (datasets.load_employee_salaries, "employee_salaries", {}, 5),
    "churn_state": (datasets.load_churn, "churn", {}, 5),
    "churn_pseudo_id": (datasets.load_churn_composite_id, "churn", {}, 5),
    "airlines_airport": (datasets.load_airlines, "airlines", {"cat_col": "AirportFrom"}, 5),
    "airlines_flight": (datasets.load_airlines, "airlines", {"cat_col": "Flight"}, 3),
    "click_advertiser": (datasets.load_click_prediction, "click", {"cat_col": "advertiser_id"}, 3),
    "click_user": (datasets.load_click_prediction, "click", {"cat_col": "user_id"}, 3),
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True, help="folder with the ARFF files")
    ap.add_argument("--tasks", nargs="*", default=list(TASKS), choices=list(TASKS))
    ap.add_argument("--fast", action="store_true",
                    help="2 folds, shorter CRP chains, CRP only up to kx=500")
    ap.add_argument("--crp-max-kx", type=int, default=None,
                    help="skip CRP above this cardinality (default 8000, fast 500)")
    ap.add_argument("--no-competitors", action="store_true",
                    help="skip the GLMM, NPMLE and CatBoost competitors")
    ap.add_argument("--no-catboost", action="store_true",
                    help="skip the full CatBoost classifier (used by default when installed)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None, help="directory for raw and summary CSVs")
    args = ap.parse_args(argv)

    crp_max_kx = args.crp_max_kx or (500 if args.fast else 8000)
    crp_kwargs = dict(CRP_FAST, n_sweeps=15, burn_in=5) if args.fast else CRP_FAST

    frames = []
    for key in args.tasks:
        loader, file_key, kwargs, folds = TASKS[key]
        task = loader(args.data_dir / datasets.DATA_FILES[file_key], **kwargs)
        use_crp = task.kx <= crp_max_kx
        print(f"\n== {task.label}: kx={task.kx}, ky={task.ky}, n={task.n}"
              f"{'' if use_crp else '  (CRP skipped)'}")
        methods = standard_methods(crp=use_crp, crp_kwargs=crp_kwargs,
                                   competitors=not args.no_competitors,
                                   catboost=False if args.no_catboost else None)
        df = evaluate_cv(task.x, task.y, methods, n_splits=2 if args.fast else folds,
                         seed=args.seed, kx=task.kx, ky=task.ky)
        df.insert(0, "task", task.label)
        summ = summarize(df, metrics=("logloss", "auc")).sort_values("logloss_mean")
        print(summ.to_string(float_format=lambda v: f"{v:.5f}"))
        frames.append(df)

    if args.out and frames:
        args.out.mkdir(parents=True, exist_ok=True)
        raw = pd.concat(frames, ignore_index=True)
        raw.to_csv(args.out / "realdata_raw.csv", index=False)
        summarize(raw, metrics=("logloss", "auc"), by=("task", "method")).to_csv(
            args.out / "realdata_summary.csv")
        print(f"\nwrote {args.out}/realdata_raw.csv and realdata_summary.csv")


if __name__ == "__main__":
    main()
