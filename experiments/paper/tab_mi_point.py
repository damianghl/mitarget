"""
Appendix table: bias and RMSE of the point estimates of I(X;Y) in the coverage study (Section 8),
on the same populations as ``tab_mi_coverage.tex``.

tab_mi_point.tex   bias (RMSE) in milli-nats of the plug-in, baseline, NPMLE, frequency-prior and
                   moment-interval-midpoint estimates at nbar = 1 and 5

Input: the ``summary.csv`` written by ``experiments/mi_coverage_tables.py``.

    python tab_mi_point.py mi_tables/summary.csv
"""

import argparse

import numpy as np
import pandas as pd

from fig_mi_coverage import NBARS, POOLS, PRIORS, SYNTH

ESTIMATES = [("plugin", "plug-in"), ("baseline", "baseline"), ("npmle", "NPMLE"), ("freq", "freq."),
             ("mom_mid", "midpoint")]


def _cells(g):
    out = []
    for nbar in NBARS:
        r = g[g.nbar == nbar]
        for name, _ in ESTIMATES:
            if r.empty or not np.isfinite(r.iloc[0][f"bias_{name}"]):
                out.append("--")
                continue
            b, e = 1e3 * r.iloc[0][f"bias_{name}"], 1e3 * r.iloc[0][f"rmse_{name}"]
            # the plug-in is off by tens to hundreds of milli-nats: no decimals there
            fmt = ".0f" if abs(b) >= 99.5 else ".1f"
            out.append(f"${b:+{fmt}}$ \\scriptsize({e:{fmt}})")
    return out


def table(s, out):
    k = len(ESTIMATES)
    lines = [rf"\begin{{tabular}}{{lr {'r' * k} {'r' * k}}}", r"\toprule",
             rf" & & \multicolumn{{{k}}}{{c}}{{$\bar n=1$}} & \multicolumn{{{k}}}{{c}}{{$\bar n=5$}}\\",
             rf"\cmidrule(lr){{3-{2 + k}}}\cmidrule(lr){{{3 + k}-{2 + 2 * k}}}",
             " & ".join(["population", "$I$"] + [lab for _, lab in ESTIMATES] * 2) + r"\\",
             r"\midrule",
             rf"\multicolumn{{{2 + 2 * k}}}{{l}}{{\emph{{synthetic, $10^5$ equally popular levels}}}}\\"]
    for prior, label in PRIORS:
        g = s[s.pool == f"{prior}|{SYNTH}"]
        lines.append(" & ".join([label, f"{g.I.mean():.3f}"] + _cells(g)) + r"\\")
    for shuffled, head in [(False, "oracle pools, real"), (True, "oracle pools, shuffled $q$")]:
        lines += [r"\midrule", rf"\multicolumn{{{2 + 2 * k}}}{{l}}{{\emph{{{head}}}}}\\"]
        for pool, label in POOLS:
            g = s[(s.pool == pool) & (s.shuffled == shuffled)]
            lines.append(" & ".join([label, f"{g.I.mean():.3f}"] + _cells(g)) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summary")
    ap.add_argument("--tab", default="tab_mi_point.tex")
    a = ap.parse_args()
    table(pd.read_csv(a.summary), a.tab)


if __name__ == "__main__":
    main()
