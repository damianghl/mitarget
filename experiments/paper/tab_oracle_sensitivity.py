"""
Appendix table: sensitivity of the real-data oracle to how the population is built (K4 of the review).

Reads the CSV of ``experiments/oracle_sensitivity.py`` (designs ``head100``, ``head50`` and ``tail``) and
writes ``tab_oracle_sensitivity.tex``: for every real pool at nbar = 2, the number of levels, corr(log p, q),
the baseline's gap to the full oracle (milli-nats), the share of that gap on levels seen at most once, and the
share of the gap closed by each method, with its paired standard error.

    python tab_oracle_sensitivity.py sens.csv
"""

import argparse

import numpy as np
import pandas as pd

POOLS = [("movie_id", "MovieLens movies"), ("user_id", "MovieLens users"),
         ("bx_user_implicit", "Book-Crossing users, rated"), ("bx_user", r"Book-Crossing users, $r\geq8$"),
         ("avazu_device_ip", r"Avazu device\_ip"), ("criteo_C1", "Criteo C1")]
DESIGNS = [("head100", r"head $\geq100$"), ("head50", r"head $\geq50$"), ("tail", "head + tail")]
CLOSERS = [("baseline_unseen", "+unseen"), ("baseline_robbins", "+Robbins"), ("freq_prior", r"freq.\ prior"),
           ("npmle", "NPMLE"), ("npmle_robbins", "NPMLE +R"), ("glmm_mean", "GLMM mean")]


def pct(r, m):
    v = r.get(f"closed_{m}", np.nan)
    if pd.isna(v):
        return "--"
    se = r.get(f"closed_{m}_se", np.nan)
    return f"{100 * v:.0f}" + ("" if pd.isna(se) else f" $\\pm$ {100 * se:.0f}") + r"\%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sens")
    ap.add_argument("--nbar", type=float, default=2.0)
    ap.add_argument("--shuffled", action="store_true")
    ap.add_argument("--tab", default="tab_oracle_sensitivity.tex")
    a = ap.parse_args()
    d = pd.read_csv(a.sens)
    d = d[(d.nbar == a.nbar) & (d.shuffled == a.shuffled)]
    head = ["pool", "population", "levels", "corr", "baseline gap", r"at $n_x\le1$"]
    head += [f"closed: {lab}" if i == 0 else lab for i, (_, lab) in enumerate(CLOSERS)]
    lines = [r"\begin{tabular}{ll r r r r " + "r" * len(CLOSERS) + "}", r"\toprule",
             " & ".join(head) + r"\\", r"\midrule"]
    for feat, plab in POOLS:
        first = True
        for des, dlab in DESIGNS:
            r = d[(d.feature == feat) & (d.design == des)]
            if r.empty:
                continue
            r = r.iloc[0].to_dict()
            cells = [plab if first else "", dlab, f"{int(r['L']):,}", f"{r['corr_logp_q']:+.2f}",
                     f"{r['baseline_gap_mnats']:.1f}", f"{100 * r['gap_share_n01']:.0f}" + r"\%"]
            cells += [pct(r, m) for m, _ in CLOSERS]
            lines.append(" & ".join(cells) + r"\\")
            first = False
        lines.append(r"\addlinespace")
    lines[-1:] = [r"\bottomrule", r"\end{tabular}"]
    with open(a.tab, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
