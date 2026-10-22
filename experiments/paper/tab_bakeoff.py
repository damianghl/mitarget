"""
Table for Section 7.3: held-out log-loss of single-feature encoders on the real data.

Reads the bakeoff raw CSVs (task, method, fold, logloss, ...) and writes a LaTeX table with the
baseline's gain over the prior-only predictor and, for every other method, its loss relative to
the baseline as a share of that gain (negative = better than the baseline), +/- the SE of the
paired fold differences.
"""

import argparse

import numpy as np
import pandas as pd

TASKS = [
    ("avazu/user=ip+model", "Avazu user (ip+model)"),
    ("avazu/device_ip", "Avazu device\\_ip"),
    ("criteo/C1", "Criteo C1"),
    ("bx/user", "BX user, $r\\geq8$"),
    ("bx_implicit/user", "BX user, explicit"),
    ("amz/MGR_ID", "Amazon MGR\\_ID"),
    ("amz/RESOURCE", "Amazon RESOURCE"),
    ("employee_salaries / employee_position_title (salary 4-quantile)", "employee\\_salaries$^\\dagger$"),
]
METHODS = [("NPMLE", "NPMLE"), ("GLMM", "GLMM"), ("K=2 mixture", "$K{=}2$"),
           ("CatBoost TS (a=1)", "CB TS"), ("Fixed m=20", "$m{=}20$"),
           ("sklearn TargetEncoder (auto)", "sklearn")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_csvs", nargs="+")
    ap.add_argument("--out", default="tab_bakeoff.tex")
    a = ap.parse_args()
    df = pd.concat([pd.read_csv(p) for p in a.raw_csvs], ignore_index=True)
    lines = []
    for task, label in TASKS:
        t = df[df.task == task].pivot_table(index="fold", columns="method", values="logloss")
        base = t["K=1 baseline"]
        if "Prior only" in t:
            prior = t["Prior only"]
        else:  # balanced 4-class target: the prior-only log-loss is log 4
            prior = pd.Series(np.log(4.0), index=t.index)
        gain = float((prior - base).mean())
        cells = [label, f"{gain:.4f}"]
        vals = {}
        for m, _ in METHODS:
            if m not in t or t[m].isna().all():
                cells.append("--")
                continue
            d = (t[m] - base) / gain * 100
            vals[m] = d.mean()
            se = d.std(ddof=1) / np.sqrt(d.count())
            se_txt = "$<$0.1" if se < 0.05 else f"{se:.1f}"
            cells.append(f"{d.mean():+.0f}" if abs(d.mean()) >= 100 else f"{d.mean():+.1f} ({se_txt})")
        best = min(vals, key=vals.get)
        if vals[best] < 0:
            i = 2 + [m for m, _ in METHODS].index(best)
            cells[i] = "\\textbf{" + cells[i] + "}"
        lines.append(" & ".join(cells) + " \\\\")
        print(label, gain, {k: round(v, 1) for k, v in vals.items()})
    head = " & ".join(["feature", "gain"] + [lab for _, lab in METHODS]) + " \\\\"
    tex = "\n".join([
        "\\begin{tabular}{l r " + "r" * len(METHODS) + "}",
        "\\toprule", head, "\\midrule", *lines, "\\bottomrule", "\\end{tabular}", ""])
    with open(a.out, "w") as fh:
        fh.write(tex)


if __name__ == "__main__":
    main()
