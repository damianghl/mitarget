"""
Appendix table: bias (RMSE) in milli-nats of classical estimators of I(X;Y) on the data sets of the coverage
study (Section 8): Miller-Madow, Chao-Shen and NSB (given the true number of levels), each applied to
I = H(X) + H(Y) - H(X, Y), at nbar = 1 and 5.

Inputs: the ``base_coverage_*.csv`` files of ``experiments/mi_coverage.py --baselines-only``.

    python tab_mi_classical.py base_raw/
"""

import argparse
from pathlib import Path

import pandas as pd

from fig_mi_coverage import POOLS, PRIORS, SYNTH

ESTIMATES = [("I_mm", "M--M"), ("I_cs", "C--S"), ("I_nsb", "NSB")]
NBARS = (1.0, 5.0)


def cells(g):
    out = []
    for nbar in NBARS:
        r = g[g.nbar == nbar]
        for col, _ in ESTIMATES:
            e = r[col] - r.I_true
            b, s = 1e3 * e.mean(), 1e3 * (e ** 2).mean() ** 0.5
            fmt = ".0f" if abs(b) >= 99.5 else ".1f"
            out.append(f"${b:+{fmt}}$ \\scriptsize({s:{fmt}})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", type=Path)
    ap.add_argument("--tab", default="tab_mi_classical.tex")
    a = ap.parse_args()
    d = pd.concat([pd.read_csv(f) for f in sorted(a.raw.glob("base_coverage_*.csv"))], ignore_index=True)
    k = len(ESTIMATES)
    lines = [rf"\begin{{tabular}}{{l {'r' * k} {'r' * k}}}", r"\toprule",
             rf" & \multicolumn{{{k}}}{{c}}{{$\bar n=1$}} & \multicolumn{{{k}}}{{c}}{{$\bar n=5$}}\\",
             rf"\cmidrule(lr){{2-{1 + k}}}\cmidrule(lr){{{2 + k}-{1 + 2 * k}}}",
             " & ".join(["population"] + [lab for _, lab in ESTIMATES] * 2) + r"\\", r"\midrule",
             rf"\multicolumn{{{1 + 2 * k}}}{{l}}{{\emph{{synthetic, $10^5$ equally popular levels}}}}\\"]
    for prior, label in PRIORS:
        lines.append(" & ".join([label] + cells(d[d.pool == f"{prior}|{SYNTH}"])) + r"\\")
    for shuffled, head in [(False, "oracle pools, real"), (True, "oracle pools, shuffled $q$")]:
        lines += [r"\midrule", rf"\multicolumn{{{1 + 2 * k}}}{{l}}{{\emph{{{head}}}}}\\"]
        for pool, label in POOLS:
            lines.append(" & ".join([label] + cells(d[(d.pool == pool) & (d.shuffled == shuffled)])) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    Path(a.tab).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
