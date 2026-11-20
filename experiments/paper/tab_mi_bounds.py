"""
Appendix table: what can be said about I(X;Y) on the full real features without exchangeability (K5).

Reads the CSV of ``experiments/mi_realdata_bounds.py`` and writes ``tab_mi_bounds.tex``: per binary task, the
singletons' row share, the predictive lower bound (best held-out encoder), the stratified identified interval
at the plug-in moments, the interval that combines them, the singletons' part of its width,
the baseline and NPMLE estimates, and the likelihood interval under exchangeability where it was computed.

    python tab_mi_bounds.py bounds.csv
"""

import argparse

import numpy as np
import pandas as pd
from tab_bakeoff import tex_name


def f3(v):
    return "--" if pd.isna(v) else f"{v:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bounds")
    ap.add_argument("--tab", default="tab_mi_bounds.tex")
    a = ap.parse_args()
    d = pd.read_csv(a.bounds).sort_values("task")
    lines = [r"\begin{tabular}{l r r r r r r r r}", r"\toprule",
             r"feature & $F_1/N$ & pred.\ bound & stratified & combined & singletons"
             r" & baseline & NPMLE"
             r" & lik.\ (exch.)\\", r"\midrule"]
    for _, r in d.iterrows():
        lo = np.nanmax([r.strat_lo, r.pred_lb, 0.0])
        lik = "--" if pd.isna(r.get("lik_lo")) else f"[{r.lik_lo:.3f}, {r.lik_hi:.3f}]"
        lines.append(" & ".join([
            tex_name(r.task), f"{r.F1_share:.2f}", f3(r.pred_lb),
            f"[{max(r.strat_lo, 0):.3f}, {r.strat_hi:.3f}]",
            f"[{lo:.3f}, {r.strat_hi:.3f}]",
            f3(0.0 if pd.isna(r.get("single_w")) else r.get("single_w")),
            f3(r.I_baseline), f3(r.I_npmle), lik]) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(a.tab, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
