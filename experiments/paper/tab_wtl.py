"""
Appendix table: paired win/loss counts between fitted encoders over the 24 bakeoff tasks.

Reads ``table_wtl.csv`` from ``experiments/bakeoff_summary.py`` (Nadeau-Bengio test over 25 splits, Holm over
all pairs of fitted encoders within a task, significance at p_Holm < 0.01) and writes ``tab_wtl.tex``: the
cell in row A, column B is "wins/losses" of A against B; the remaining tasks are ties.

    python tab_wtl.py tables/table_wtl.csv
"""

import argparse

import pandas as pd

METHODS = [("K=1 baseline", "baseline"),
           ("Baseline + unseen correction", "+unseen"),
           ("Baseline + Robbins (n<=1)", "+Robbins"), ("Frequency prior", "freq."), ("NPMLE", "NPMLE"),
           ("NPMLE + Robbins (n<=1)", "NPMLE+R"), ("GLMM", "GLMM mode"),
           ("GLMM (posterior mean)", "GLMM mean"),
           ("CV m (LOO)", "CV $m$"), ("Fixed m=20", "$m{=}20$")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wtl")
    ap.add_argument("--tab", default="tab_wtl.tex")
    a = ap.parse_args()
    w = pd.read_csv(a.wtl).set_index(["a", "b"])
    lines = [r"\begin{tabular}{l" + "c" * len(METHODS) + "}", r"\toprule",
             " & ".join([""] + [lab for _, lab in METHODS]) + r"\\", r"\midrule"]
    for m, lab in METHODS:
        cells = [lab]
        for o, _ in METHODS:
            cells.append("--" if m == o else f"{int(w.loc[(m, o), 'win'])}/{int(w.loc[(m, o), 'loss'])}")
        lines.append(" & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(a.tab, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
