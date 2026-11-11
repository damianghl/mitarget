"""
Paper tables from the output of ``realdata_paired.py``.

    python experiments/realdata_tables.py --raw results/ --out tables/
    python experiments/realdata_tables.py --raw old_results/ new_methods/ --out tables/

With several ``--raw`` folders run on the same splits, the files are merged:
a (task, rep, fold, method) row, a (task, rep, method pair) row or a
(task, rep, method, bin) row is taken from the first folder that has it, so
a later run that adds methods (and repeats the baseline) extends the earlier
one. Holm's correction is then over the merged family of methods.

Writes:

- ``table_logloss.csv``: per (task, method) mean held-out log-loss over all
  splits, its standard error, AUC, the gain over the prior-only predictor and
  that gain as a share of the baseline's gain.
- ``table_paired.csv``: per (task, method, reference) two paired tests of the
  log-loss difference ``method - reference``:
  the Nadeau-Bengio corrected resampled t-test over the R x K split means
  (accounts for split-to-split variation, conservative), and the row-level
  difference with levels as independent clusters (conditional on the
  data set, much more powerful). ``p_*_holm`` are Holm-adjusted within each
  (task, reference) family.
- ``table_by_n.csv``: per (task, method, training-count bin) the share of test
  rows, their mean log-loss and the difference to the baseline.
- ``table_tasks.csv``: per task N, kx, singleton shares and H(Y).
- ``table_main.csv``: one row per task for the paper: task statistics, the
  baseline's gain over the prior, each method's log-loss difference to the
  baseline as a share of that gain, the Holm-adjusted split-test p-value, and
  the share of the difference that comes from test rows of levels unseen in
  training (``n0share``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASELINE = "K=1 baseline"
ROBBINS = "Baseline + Robbins (n<=1)"
REFERENCES = (BASELINE, "NPMLE", ROBBINS)
#: compact task names for tables (the OpenML loaders use long labels)
SHORT = {
    "employee_salaries / employee_position_title (salary 4-quantile)": "employee_salaries/position_title",
    "churn / state": "churn/state",
    "churn / state x area_code x account_length": "churn/pseudo_id",
    "airlines / AirportFrom": "airlines/AirportFrom",
    "airlines / Flight": "airlines/Flight",
    "click_prediction_small / advertiser_id": "click_small/advertiser_id",
    "click_prediction_small / user_id": "click_small/user_id",
}
METHOD_ORDER = ["Prior only", "Fixed m=20", "CV m (LOO)", BASELINE, "Baseline + unseen correction",
                ROBBINS, "Baseline + Robbins (n<=1) raw", "Frequency prior",
                "GLMM", "NPMLE", "NPMLE + Robbins (n<=1)", "CatBoost TS (a=1)",
                "sklearn TargetEncoder (auto)"]


#: columns that identify a row of each raw file, for merging several runs
KEYS = {"folds": ["task", "rep", "fold", "method"], "rows": ["task", "rep", "method_a", "method_b"],
        "bins": ["task", "rep", "method", "bin"], "meta": ["task"]}


def _read(raws, prefix: str) -> pd.DataFrame:
    frames = []
    for raw in raws if isinstance(raws, (list, tuple)) else [raws]:
        files = sorted(Path(raw).glob(f"{prefix}_*.csv"))
        frames += [pd.read_csv(f) for f in files]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if prefix == "rows":   # a pair may appear as (a, b) in one run and (b, a) in another
        key = df[["method_a", "method_b"]].apply(lambda r: tuple(sorted(r)), axis=1)
        return df[~pd.concat([df[["task", "rep"]], key.rename("pair")], axis=1).duplicated()].reset_index(
            drop=True)
    return df.drop_duplicates(KEYS[prefix], keep="first").reset_index(drop=True)


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    q = p[ok]
    order = np.argsort(q)
    adj = np.maximum.accumulate((len(q) - np.arange(len(q))) * q[order])
    res = np.empty_like(q)
    res[order] = np.minimum(adj, 1.0)
    out[ok] = res
    return out


def logloss_table(folds: pd.DataFrame) -> pd.DataFrame:
    def sem(v):
        return v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
    g = folds.groupby(["task", "method"], sort=False)
    t = g.agg(logloss=("logloss", "mean"), logloss_sem=("logloss", sem), splits=("logloss", "size"),
              seconds=("seconds", "mean"))
    if "auc" in folds:
        t["auc"] = g.auc.mean()
    t = t.reset_index()
    prior = t[t.method == "Prior only"].set_index("task").logloss
    base = t[t.method == BASELINE].set_index("task").logloss
    t["gain"] = t.task.map(prior) - t.logloss
    t["share_of_baseline_gain"] = t.gain / (t.task.map(prior) - t.task.map(base))
    return t


def nb_test(folds: pd.DataFrame, a: str, b: str):
    """Nadeau-Bengio corrected resampled t-test on paired split means of ``a - b``."""
    fa = folds[folds.method == a].set_index(["rep", "fold"])
    fb = folds[folds.method == b].set_index(["rep", "fold"])
    d = (fa.logloss - fb.logloss).dropna()
    J = len(d)
    ratio = (fa.n_test / fa.n_train).mean()
    var = (1 / J + ratio) * d.var(ddof=1)
    se = np.sqrt(var)
    t = d.mean() / se if se > 0 else (0.0 if d.abs().max() == 0 else np.nan)
    p = 2 * stats.t.sf(abs(t), J - 1) if np.isfinite(t) else np.nan
    return d.mean(), se, t, p, float((d < 0).mean()), J


def paired_table(folds: pd.DataFrame, rows: pd.DataFrame, ll: pd.DataFrame) -> pd.DataFrame:
    out = []
    for task, ft in folds.groupby("task", sort=False):
        rt = rows[rows.task == task]
        gain = ll[(ll.task == task) & (ll.method == BASELINE)].gain.iloc[0]
        for ref in REFERENCES:
            for m in ft.method.unique():
                if m == ref or m == "Prior only":
                    continue
                mean, se, t, p, wins, J = nb_test(ft, m, ref)
                ab = rt[(rt.method_a == m) & (rt.method_b == ref)]
                ba = rt[(rt.method_a == ref) & (rt.method_b == m)]
                r = pd.concat([ab.assign(sign=1.0), ba.assign(sign=-1.0)])
                rmean = float((r.sign * r.mean_diff).mean())
                rse = float(r.se_clustered.mean())
                z = rmean / rse if rse > 0 else (0.0 if rmean == 0 else np.nan)
                out.append(dict(task=task, method=m, reference=ref, splits=J,
                                diff=mean, diff_share_of_baseline_gain=mean / gain,
                                se_nb=se, t_nb=t, p_nb=p, win_rate=wins,
                                diff_rows=rmean, se_clustered=rse, z_clustered=z,
                                p_clustered=2 * stats.norm.sf(abs(z)) if np.isfinite(z) else np.nan))
    df = pd.DataFrame(out)
    for col in ("p_nb", "p_clustered"):
        df[col + "_holm"] = df.groupby(["task", "reference"])[col].transform(lambda s: holm(s.values))
    return df


def by_n_table(bins: pd.DataFrame) -> pd.DataFrame:
    g = bins.groupby(["task", "method", "bin", "n_lo"], sort=False)
    t = g.agg(rows=("rows", "mean"), logloss=("logloss", "mean"),
              diff_vs_baseline=("diff_vs_baseline", "mean"),
              se_clustered=("se_clustered", "mean")).reset_index()
    total = t[t.method == BASELINE].groupby("task").rows.sum()
    t["row_share"] = t.rows / t.task.map(total)
    return t.sort_values(["task", "n_lo", "method"], kind="stable")


def main_table(ll, paired, by_n, meta) -> pd.DataFrame:
    base = ll[ll.method == BASELINE].set_index("task")
    t = meta.set_index("task")[["N", "kx", "F1_2F2_share", "H_Y"]] if not meta.empty else pd.DataFrame(
        index=base.index)
    t["baseline_logloss"] = base.logloss
    t["baseline_gain"] = base.gain
    vs = paired[paired.reference == BASELINE].set_index(["task", "method"])
    contrib = by_n.assign(c=by_n.diff_vs_baseline * by_n.row_share)
    total = contrib.groupby(["task", "method"]).c.sum()
    unseen = contrib[contrib.n_lo == 0].groupby(["task", "method"]).c.sum()
    for m in METHOD_ORDER:
        if m in ("Prior only", BASELINE) or m not in set(vs.index.get_level_values(1)):
            continue
        sub = vs.xs(m, level="method")
        t[f"{m}: share"] = sub.diff_share_of_baseline_gain
        t[f"{m}: p_holm"] = sub.p_nb_holm
        t[f"{m}: n0share"] = (unseen.xs(m, level="method") / total.xs(m, level="method")).where(
            total.xs(m, level="method") != 0)
    return t.reset_index()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, nargs="+", required=True,
                    help="output folder(s) of realdata_paired.py, run on the same splits")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    folds, rows, bins, meta = (_read(args.raw, p) for p in ("folds", "rows", "bins", "meta"))
    for df in (folds, rows, bins, meta):
        if not df.empty:
            df["task"] = df.task.replace(SHORT)
    ll = logloss_table(folds)
    ll["method"] = pd.Categorical(ll.method, [m for m in METHOD_ORDER if m in set(ll.method)])
    ll = ll.sort_values(["task", "method"], kind="stable")
    ll["method"] = ll.method.astype(str)
    paired = paired_table(folds, rows, ll)
    ll.to_csv(args.out / "table_logloss.csv", index=False)
    paired.to_csv(args.out / "table_paired.csv", index=False)
    by_n = by_n_table(bins)
    by_n.to_csv(args.out / "table_by_n.csv", index=False)
    if not meta.empty:
        meta.to_csv(args.out / "table_tasks.csv", index=False)
    main_table(ll, paired, by_n, meta).to_csv(args.out / "table_main.csv", index=False)

    wide = ll.pivot(index="task", columns="method", values="logloss")
    print(wide[[m for m in METHOD_ORDER if m in wide]].to_string(float_format=lambda v: f"{v:.5f}"))
    vs = paired[paired.reference == BASELINE].pivot(index="task", columns="method", values="z_clustered")
    print("\nclustered z vs baseline (negative = better than baseline)")
    print(vs.to_string(float_format=lambda v: f"{v:+.1f}"))


if __name__ == "__main__":
    main()
