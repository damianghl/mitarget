"""
Summary statistics of the bakeoff that do not depend on which tasks reach significance.

    python experiments/bakeoff_summary.py --raw raw_a/ raw_b/ ... --out tables/ [--min-gain 0.001]

Reads the ``folds_*.csv`` files of ``realdata_paired.py`` (merged as in ``realdata_tables.py``) and writes:

- ``table_wtl.csv``: for every ordered pair of fitted methods, the number of tasks on which the first is
  significantly better (win), not significantly different (tie) or significantly worse (loss) than the
  second. The test is the Nadeau-Bengio corrected resampled t-test over the 25 splits. Holm's correction
  is applied within each task over *all* pairwise comparisons among the fitted methods, so the family is
  fixed in advance and the same for every comparison. Significance: Holm-adjusted p < ``--alpha``.
- ``table_medians.csv``: per method, the median and the interquartile range of its log-loss difference to
  the baseline as a share of the baseline's gain, over a fixed task set: every task on which the baseline
  gains at least ``--min-rel-gain`` times H(Y) over the prior-only predictor (default 0.5%: shares of a
  smaller gain are dominated by noise). No significance selection.
- ``table_unseen_share.csv``: per method, over the tasks of the fixed set where it improves on the
  baseline, the share of its improvement per test row, summed over those tasks, that comes from test rows
  on levels unseen in training (pooled share), and the median of the per-task shares.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from realdata_tables import BASELINE, SHORT, _read, holm, logloss_table, nb_test  # noqa: E402

#: methods compared in the win/tie/loss table (all fitted encoders of the paper)
FITTED = ["Fixed m=20", "CV m (LOO)", BASELINE, "Baseline + unseen correction", "Baseline + Robbins (n<=1)",
          "Frequency prior", "NPMLE", "NPMLE + Robbins (n<=1)", "GLMM", "GLMM (posterior mean)",
          "CatBoost TS (a=1)", "sklearn TargetEncoder (auto)"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--min-rel-gain", type=float, default=0.005,
                    help="tasks where the baseline gains less than this fraction of H(Y) are left out")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)

    folds, bins, meta = _read(a.raw, "folds"), _read(a.raw, "bins"), _read(a.raw, "meta")
    for df in (folds, bins, meta):
        df["task"] = df.task.replace(SHORT)
    methods = [m for m in FITTED if m in set(folds.method)]
    ll = logloss_table(folds)

    # --- win / tie / loss with Holm over all pairs within a task ---
    rows = []
    for task, ft in folds.groupby("task", sort=False):
        have = [m for m in methods if m in set(ft.method)]
        pairs = list(itertools.combinations(have, 2))
        res = [nb_test(ft, x, y) for x, y in pairs]
        p_holm = holm(np.array([r[3] for r in res]))
        for (x, y), r, ph in zip(pairs, res, p_holm, strict=True):
            rows.append(dict(task=task, a=x, b=y, diff=r[0], p=r[3], p_holm=ph))
            rows.append(dict(task=task, a=y, b=x, diff=-r[0], p=r[3], p_holm=ph))
    pt = pd.DataFrame(rows)
    pt["outcome"] = np.where(pt.p_holm >= a.alpha, "tie", np.where(pt["diff"] < 0, "win", "loss"))
    pt.to_csv(a.out / "table_pairwise_tasks.csv", index=False)
    wtl = (pt.groupby(["a", "b"]).outcome.value_counts().unstack(fill_value=0)
           .reindex(columns=["win", "tie", "loss"], fill_value=0).reset_index())
    wtl.to_csv(a.out / "table_wtl.csv", index=False)

    # --- medians over a fixed task set ---
    base = ll[ll.method == BASELINE].set_index("task")
    h_y = meta.set_index("task").H_Y.reindex(base.index)
    keep = base.index[base.gain >= a.min_rel_gain * h_y]
    dropped = sorted(set(base.index) - set(keep))
    sub = ll[ll.task.isin(keep)].merge(base[["logloss", "gain"]], left_on="task", right_index=True,
                                       suffixes=("", "_base"))
    sub["share"] = (sub.logloss - sub.logloss_base) / sub.gain_base
    med = (sub[sub.method.isin(methods)].groupby("method").share
           .agg(tasks="size", median="median", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75),
                worse_point=lambda s: int((s > 0).sum()))
           .reindex(methods).dropna(how="all").reset_index())
    med.to_csv(a.out / "table_medians.csv", index=False)

    # --- pooled share of the improvement that comes from unseen test rows ---
    b = bins[bins.task.isin(keep)]
    rows_per = b.groupby(["task", "method", "bin"]).agg(rows=("rows", "mean"), logloss=("logloss", "mean"))
    rows_per = rows_per.reset_index()
    # per-row loss of each bin, so every task counts by its improvement per test row, not by its size
    n_task = rows_per[rows_per.method == BASELINE].groupby("task").rows.sum()
    tot = rows_per.assign(loss=rows_per.rows * rows_per.logloss / rows_per.task.map(n_task))
    by = tot.groupby(["task", "method"]).loss.sum().rename("all").to_frame()
    by["unseen"] = tot[tot["bin"].astype(str) == "0"].groupby(["task", "method"]).loss.sum()
    by = by.reset_index()
    bb = by[by.method == BASELINE].set_index("task")
    by = by.join(bb[["all", "unseen"]], on="task", rsuffix="_base")
    by["impr_all"] = by.all_base - by["all"]
    by["impr_unseen"] = by.unseen_base - by.unseen
    out = []
    for m in methods:
        g = by[(by.method == m) & (by.impr_all > 0)]
        if m == BASELINE or g.empty:
            continue
        out.append(dict(method=m, tasks_improving=len(g),
                        pooled_unseen_share=g.impr_unseen.sum() / g.impr_all.sum(),
                        median_task_share=float((g.impr_unseen / g.impr_all).median())))
    pd.DataFrame(out).to_csv(a.out / "table_unseen_share.csv", index=False)

    print(f"fixed task set: {len(keep)} tasks (gain >= {a.min_rel_gain:g} H(Y)); left out: {dropped}")
    print(med.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    w = wtl.pivot(index="a", columns="b", values="win").reindex(index=methods, columns=methods)
    lo = wtl.pivot(index="a", columns="b", values="loss").reindex(index=methods, columns=methods)
    print("\nwins/losses of row vs column (Holm over all pairs within task)")
    print((w.fillna(0).astype(int).astype(str) + "/" + lo.fillna(0).astype(int).astype(str)).to_string())
    print(pd.DataFrame(out).to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
