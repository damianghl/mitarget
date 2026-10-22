"""
Tables for Section 7.3: held-out log-loss of single-feature encoders on the real data.

Reads ``table_paired.csv`` and ``table_logloss.csv`` written by ``experiments/realdata_tables.py``
(5 x 5 repeated folds) and writes a LaTeX table with the baseline's gain over the prior-only
predictor and, for every other method, its log-loss minus the baseline's as a share of that gain
(negative = better than the baseline), with the Nadeau-Bengio corrected standard error in brackets.
Cells whose Holm-adjusted corrected p-value is below 0.05 carry a star.
"""

import argparse
import os

import pandas as pd

TASKS = [
    ("avazu/user", "Avazu user (ip+model)"),
    ("avazu/device_ip", "Avazu device\\_ip"),
    ("criteo/C1", "Criteo C1"),
    ("bx/user", "BX user, $r\\geq8$"),
    ("bx_explicit/user", "BX user, explicit"),
    ("amazon/MGR_ID", "Amazon MGR\\_ID"),
    ("amazon/RESOURCE", "Amazon RESOURCE"),
    ("employee_salaries/position_title", "employee\\_salaries$^\\dagger$"),
]
METHODS = [("Baseline + unseen correction", "+unseen"), ("Frequency prior", "freq."), ("NPMLE", "NPMLE"),
           ("GLMM", "GLMM"), ("CatBoost TS (a=1)", "CB TS"), ("Fixed m=20", "$m{=}20$"),
           ("sklearn TargetEncoder (auto)", "sklearn")]


def cell(v, se, star):
    if abs(v) >= 100:
        return f"{v:+.0f}"
    se_txt = "$<$0.1" if se < 0.05 else f"{se:.1f}"
    return f"{v:+.1f}{'$^*$' if star else ''} ({se_txt})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tables_dir", help="output folder of experiments/realdata_tables.py")
    ap.add_argument("--out", default="tab_bakeoff.tex")
    a = ap.parse_args()
    paired = pd.read_csv(os.path.join(a.tables_dir, "table_paired.csv"))
    ll = pd.read_csv(os.path.join(a.tables_dir, "table_logloss.csv"))
    paired = paired[paired.reference == "K=1 baseline"].set_index(["task", "method"])
    gains = ll[ll.method == "K=1 baseline"].set_index("task").gain
    lines = []
    for task, label in TASKS:
        gain = gains[task]
        cells, vals = [label, f"{gain:.4f}"], {}
        for m, _ in METHODS:
            if (task, m) not in paired.index:
                cells.append("--")
                continue
            r = paired.loc[(task, m)]
            v, se = 100 * r["diff"] / gain, 100 * r["se_nb"] / gain
            vals[m] = v
            cells.append(cell(v, se, r.p_nb_holm < 0.05))
        best = min(vals, key=vals.get)
        if vals[best] < 0:
            i = 2 + [m for m, _ in METHODS].index(best)
            cells[i] = "\\textbf{" + cells[i] + "}"
        lines.append(" & ".join(cells) + " \\\\")
        print(label, f"{gain:.4f}", {k: round(v, 1) for k, v in vals.items()})
    head = " & ".join(["feature", "gain"] + [lab for _, lab in METHODS]) + " \\\\"
    tex = "\n".join([
        "\\begin{tabular}{l r " + "r" * len(METHODS) + "}",
        "\\toprule", head, "\\midrule", *lines, "\\bottomrule", "\\end{tabular}", ""])
    with open(a.out, "w") as fh:
        fh.write(tex)


if __name__ == "__main__":
    main()
