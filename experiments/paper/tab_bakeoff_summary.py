"""
Main-text bakeoff table (M4 of the second simulated review): every method against the baseline over all 24
tasks, in absolute and relative terms, with counts by task and by data source.

Reads ``table_paired.csv`` and ``table_logloss.csv`` from ``experiments/realdata_tables.py`` and writes
``tab_bakeoff_summary.tex``. Columns: median over the 24 tasks of the held-out log-loss difference to the
baseline (milli-nats per test row) and of the relative log-loss difference (% of the baseline's log-loss);
median over the 9 data sources of each source's median difference; median share of the baseline's gain over
the fixed set of 20 tasks; number of tasks and of sources (out of 9) with a significant improvement /
degradation (corrected resampled t-test, Holm over the 12 comparisons with the baseline, p < 0.01).

    python tab_bakeoff_summary.py table_paired.csv table_logloss.csv
"""

import argparse

import numpy as np
import pandas as pd
from tab_bakeoff import METHODS

BASE = "K=1 baseline"
LOW_GAIN = {"churn/state", "churn/pseudo_id", "airlines/Flight", "click_small/user_id"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paired")
    ap.add_argument("logloss")
    ap.add_argument("--tab", default="tab_bakeoff_summary.tex")
    a = ap.parse_args()
    p = pd.read_csv(a.paired)
    p = p[p.reference == BASE].copy()
    ll = pd.read_csv(a.logloss)
    base = ll[ll.method == BASE].set_index("task").logloss
    p["rel"] = 100 * p["diff"] / p.task.map(base)
    p["src"] = p.task.str.split("/").str[0].replace({"bx_explicit": "bx"})
    p["sig"] = np.where(p.p_nb_holm < 0.01, np.sign(p["diff"]), 0)
    lines = [r"\begin{tabular}{l r r r r r r}", r"\toprule",
             r" & \multicolumn{2}{c}{median over 24 tasks} & median over & median \% of & tasks & sources\\",
             r"method & mnats & rel.\ \% & 9 sources (mnats) & gain (20 tasks) & better/worse & better/worse\\",
             r"\midrule"]
    for m, lab in METHODS:
        d = p[p.method == m]
        if d.empty:
            continue
        src = d.groupby("src")["diff"].median().median()
        share = d[~d.task.isin(LOW_GAIN)].diff_share_of_baseline_gain.median()
        cells = [lab, f"{1e3 * d['diff'].median():+.2f}", f"{d.rel.median():+.3f}", f"{1e3 * src:+.2f}",
                 f"{100 * share:+.1f}", f"{(d.sig < 0).sum()}/{(d.sig > 0).sum()}",
                 f"{d[d.sig < 0].src.nunique()}/{d[d.sig > 0].src.nunique()}"]
        lines.append(" & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(a.tab, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
