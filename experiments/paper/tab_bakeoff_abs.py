"""
Appendix table: bakeoff differences to the baseline in absolute milli-nats, with corrected 95% intervals.

Reads ``table_paired.csv`` from ``experiments/realdata_tables.py`` (Nadeau-Bengio standard errors over the
25 splits) and writes ``tab_bakeoff_abs.tex``: per task and method, the mean held-out log-loss difference
to the baseline in milli-nats per test row, +- the half-width of a Bonferroni-corrected 95% interval over
the 11 comparisons with the baseline on the task (t quantile with J - 1 degrees of freedom).

    python tab_bakeoff_abs.py tables/table_paired.csv
"""

import argparse

import pandas as pd
from scipy import stats
from tab_bakeoff import METHODS, tex_name

BASE = "K=1 baseline"
N_COMP = 11   # comparisons with the baseline per task (scikit-learn's encoder is left out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paired")
    ap.add_argument("--tab", default="tab_bakeoff_abs.tex")
    a = ap.parse_args()
    d = pd.read_csv(a.paired)
    d = d[d.reference == BASE]
    methods = [(m, lab) for m, lab in METHODS if m in set(d.method)]
    lines = [r"\begin{tabular}{l " + "r" * len(methods) + "}", r"\toprule",
             " & ".join(["feature"] + [lab for _, lab in methods]) + r"\\", r"\midrule"]
    for task in sorted(d.task.unique()):
        t = d[d.task == task].set_index("method")
        cells = [tex_name(task)]
        for m, _ in methods:
            if m not in t.index:
                cells.append("--")
                continue
            r = t.loc[m]
            q = stats.t.ppf(1 - 0.05 / (2 * N_COMP), int(r.splits) - 1)
            mean, hw = 1e3 * r["diff"], 1e3 * q * r.se_nb
            fmt = ".1f" if max(abs(mean), hw) >= 9.95 else ".2f"
            cells.append(f"${mean:+{fmt}}$\\,{{\\scriptsize$\\pm${hw:{fmt}}}}")
        lines.append(" & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(a.tab, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
