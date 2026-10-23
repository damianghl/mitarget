"""
Tables for Section 7.3: held-out log-loss of single-feature encoders on the real data.

Reads ``table_main.csv`` from the paired real-data rerun (experiments/realdata_tables.py): one row per
task with the baseline's gain over the marginal predictor and, per method, its log-loss minus the
baseline's as a share of that gain (negative = better), the Holm-adjusted p of the corrected
resampled t-test, and the share of the difference that comes from test rows on unseen levels.
Writes the main-text table (representative sparse features) and the full appendix table.
"""

import argparse
import os

import numpy as np
import pandas as pd

MAIN = [
    ("avazu/user", "Avazu user (ip+model)"),
    ("avazu/device_ip", "Avazu device\\_ip"),
    ("criteo/C1", "Criteo C1"),
    ("bx/user", "BX user, $r\\geq8$"),
    ("bx_explicit/user", "BX user, rated"),
    ("bx/isbn", "BX ISBN, $r\\geq8$"),
    ("amazon/MGR_ID", "Amazon MGR\\_ID"),
    ("amazon/RESOURCE", "Amazon RESOURCE"),
    ("employee_salaries/position_title", "employee\\_salaries$^\\dagger$"),
]
METHODS = [("Baseline + unseen correction", "+unseen"), ("NPMLE", "NPMLE"),
           ("Frequency prior", "freq.\\ prior"), ("GLMM", "GLMM"), ("Fixed m=20", "$m{=}20$"),
           ("CatBoost TS (a=1)", "CatBoost TS")]
ALPHA = 0.01


def tex_name(task):
    # the explicit-vs-implicit Book-Crossing target is called "rated" in the paper
    return task.replace("bx_explicit/", "bx_rated/").replace("_", "\\_")


def cell(row, m):
    v, p = row[f"{m}: share"], row[f"{m}: p_holm"]
    if not np.isfinite(v):
        return "--"
    txt = f"{100 * v:+.0f}" if abs(v) >= 1 else f"{100 * v:+.1f}"
    return txt + ("$^{*}$" if p < ALPHA else "")


def n0(row, m):
    v = row[f"{m}: n0share"]
    return f"{100 * v:.0f}\\%" if np.isfinite(v) and row[f"{m}: share"] < 0 and row[f"{m}: p_holm"] < ALPHA \
        else "--"


def table(df, tasks, with_n0):
    head = ["feature", "$N$", "levels", "$\\frac{F_1+2F_2}{N}$", "gain"] + [lab for _, lab in METHODS]
    if with_n0:
        head += ["NPMLE at $n_x{=}0$", "freq.\\ at $n_x{=}0$"]
    lines = []
    for task, label in tasks:
        r = df.loc[task]
        cells = [label, f"{r.N / 1e3:,.0f}k" if r.N < 1e6 else f"{r.N / 1e6:.1f}M",
                 f"{r.kx:,.0f}", f"{r.F1_2F2_share:.2f}", f"{r.baseline_gain:.4f}"]
        cells += [cell(r, m) for m, _ in METHODS]
        if with_n0:
            cells += [n0(r, "NPMLE"), n0(r, "Frequency prior")]
        best = min((r[f"{m}: share"], i) for i, (m, _) in enumerate(METHODS) if np.isfinite(r[f"{m}: share"]))
        # no bold where the baseline gains nothing over the marginal: shares are meaningless there
        if best[0] < 0 and r[f"{METHODS[best[1]][0]}: p_holm"] < ALPHA and r.baseline_gain > 0:
            cells[5 + best[1]] = "\\textbf{" + cells[5 + best[1]] + "}"
        lines.append(" & ".join(cells) + " \\\\")
    return "\n".join(["\\begin{tabular}{l rrrr " + "r" * (len(head) - 5) + "}", "\\toprule",
                      " & ".join(head) + " \\\\", "\\midrule", *lines, "\\bottomrule", "\\end{tabular}", ""])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table_main", help="table_main.csv from experiments/realdata_tables.py")
    ap.add_argument("--out-dir", default=".")
    a = ap.parse_args()
    df = pd.read_csv(a.table_main).set_index("task")
    with open(os.path.join(a.out_dir, "tab_bakeoff.tex"), "w") as fh:
        fh.write(table(df, MAIN, with_n0=True))
    with open(os.path.join(a.out_dir, "tab_bakeoff_full.tex"), "w") as fh:
        fh.write(table(df, [(t, tex_name(t)) for t in df.index], with_n0=False))
    for task, _ in MAIN:
        r = df.loc[task]
        parts = [f"{m}={100 * r[f'{m}: share']:+.1f} p={r[f'{m}: p_holm']:.1e} n0={r[f'{m}: n0share']:.2f}"
                 for m, _ in METHODS]
        print(task, " | ".join(parts))


if __name__ == "__main__":
    main()
