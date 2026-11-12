"""
Appendix tables: exchangeability tests (Section 6.4) on the oracle pools and on the bakeoff tasks.

tab_exch_pools.tex   rejection rate at level 0.05 of the stratum moment test and of the frequency
                     prior's LR test over the 50 training sets, and the likelihood interval's coverage
                     of I, at nbar = 1, 5, 20
tab_exch_tasks.tex   both tests on the full data of the 24 bakeoff tasks, with the largest deviation
                     of a stratum's mean label rate from the pooled one

Inputs: ``exch_pools_summary.csv`` and ``exch_tasks.csv`` written by ``experiments/exchangeability.py``
(``summary --coverage`` and ``tasks``).

    python tab_exch.py exch_tables/exch_pools_summary.csv exch_raw/exch_tasks.csv --out-dir .
"""

import argparse
import math
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from realdata_tables import SHORT  # noqa: E402

from fig_mi_coverage import POOLS  # noqa: E402

NBARS = (1.0, 5.0, 20.0)


def _rate(v):
    return "--" if pd.isna(v) else f"{v:.2f}"


def pools_table(s, out):
    k = len(NBARS)
    lines = [rf"\begin{{tabular}}{{l {'ccc ' * k}}}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{$\bar n={n:g}$}}" for n in NBARS) + r"\\",
             "".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(k)),
             " & ".join(["pool"] + ["stratum", "LR", "lik.\\ cov."] * k) + r"\\", r"\midrule"]
    for shuffled, head in [(False, "real"), (True, "shuffled $q$")]:
        lines.append(rf"\multicolumn{{{1 + 3 * k}}}{{l}}{{\emph{{oracle pools, {head}}}}}\\")
        for pool, label in POOLS:
            g = s[(s.feature == pool) & (s.shuffled == shuffled)].set_index("nbar")
            cells = [label]
            for n in NBARS:
                r = g.loc[n] if n in g.index else None
                cells += ["--"] * 3 if r is None else [_rate(r.reject_stratum), _rate(r.reject_lr),
                                                       _rate(r.cov_lik)]
            lines.append(" & ".join(cells) + r"\\")
        if not shuffled:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")


def _p(p):
    if pd.isna(p):
        return "--"
    if p < 1e-300:
        return r"$<10^{-300}$"
    if p >= 1e-3:
        return f"{p:.3f}"
    e = math.floor(math.log10(p)) + 1
    return rf"$<10^{{{e}}}$"


def tasks_table(t, out):
    t = t.assign(name=t.task.map(lambda x: SHORT.get(x, x).replace("bx_explicit", "bx_rated")))
    t = t.sort_values("name")
    lines = [r"\begin{tabular}{l rr r r r}", r"\toprule",
             r"feature & $N$ & levels & $p$ stratum & $p$ LR & max.\ dev.\ $m_1$\\", r"\midrule"]
    for _, r in t.iterrows():
        n = f"{r.N / 1e6:.1f}M" if r.N >= 1e6 else f"{r.N / 1e3:.0f}k"
        dev = "--" if pd.isna(r.max_dev_m1) else f"{r.max_dev_m1:.3f}"
        lines.append(" & ".join([r["name"].replace("_", r"\_"), n, f"{int(r.kx):,}", _p(r.p_stratum),
                                 _p(r.p_lr), dev]) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    ok = t.p_stratum.notna()
    print(f"stratum test: {int((t.p_stratum[ok] < 1e-3).sum())}/{int(ok.sum())} tasks at p<1e-3; "
          f"LR test: {int((t.p_lr < 1e-3).sum())}/{len(t)} at p<1e-3, {int((t.p_lr < 0.05).sum())} at p<0.05")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pools_summary")
    ap.add_argument("tasks")
    ap.add_argument("--out-dir", default=".")
    a = ap.parse_args()
    pools_table(pd.read_csv(a.pools_summary), os.path.join(a.out_dir, "tab_exch_pools.tex"))
    tasks_table(pd.read_csv(a.tasks), os.path.join(a.out_dir, "tab_exch_tasks.tex"))


if __name__ == "__main__":
    main()
