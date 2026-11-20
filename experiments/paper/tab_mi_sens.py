"""
Appendix table: implementation checks of the likelihood interval.

Reads the ``lik_coverage_*.csv`` files that ``experiments/mi_coverage.py --lik-only`` writes with the
tags below and writes ``tab_mi_sens.tex``:

- grid of priors, 101 against 401 points (tags g101, g401s; same data sets);
- the three-atom prior with atoms on and off the 101-point grid (atoms3, atoms3_off);
- thinning cap, 200 against 1000 rows per level, on shuffled Avazu (cap200B/cap1000B; same data sets);
- optionally, lognormal-popularity cells (tag lognormalB; too slow to run at nbar >= 2 with 1e5 levels).

    python tab_mi_sens.py RUNS_DIR
"""

import argparse
from pathlib import Path

import pandas as pd

PRIOR_TEX = {"atoms3": "three atoms", "atoms3_off": "three atoms, off grid",
             "beta_1_3": r"$\mathrm{Beta}(1,3)$",
             "beta_0.5_0.5": r"$\mathrm{Beta}(\tfrac12,\tfrac12)$", "junk_f0.5": "junk, $f=0.5$",
             "junk_f0.8": "junk, $f=0.8$"}


def load(path):
    d = pd.read_csv(path)
    d["cov"] = (d.lik_lo <= d.I_true) & (d.I_true <= d.lik_hi)
    d["w"] = d.lik_hi - d.lik_lo
    return d


def paired(a, b, keys):
    m = a.merge(b, on=keys + ["rep"], suffixes=("_a", "_b"))
    assert (m.I_true_a - m.I_true_b).abs().max() < 1e-12, "paired runs must use the same data sets"
    m["dend"] = pd.concat([(m.lik_lo_a - m.lik_lo_b).abs(),
                           (m.lik_hi_a - m.lik_hi_b).abs()], axis=1).max(axis=1)
    return m.groupby(keys).agg(ca=("cov_a", "mean"), cb=("cov_b", "mean"), wa=("w_a", "mean"),
                               wb=("w_b", "mean"), d=("dend", "max"), n=("rep", "size")).reset_index()


def row(check, cell, nbar, ca, cb, wa, wb, d, n):
    d = "--" if d is None else f"{d:.1e}"
    return f"{check} & {cell} & {nbar:g} & {int(n)} & {ca:.2f} / {cb:.2f} & {wa:.3f} / {wb:.3f} & {d}\\\\"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=Path)
    ap.add_argument("--tab", default="tab_mi_sens.tex")
    a = ap.parse_args()
    r = a.runs
    lines = [r"\begin{tabular}{l l r r c c r}", r"\toprule",
             r"check (A / B) & cell & $\bar n$ & sets & coverage & width (nats) & max.\ $|\Delta|$ end\\",
             r"\midrule"]
    g = paired(load(r / "lik_coverage_synthetic_g101.csv"), load(r / "lik_coverage_synthetic_g401s.csv"),
               ["prior", "nbar"])
    for _, x in g.iterrows():
        cell = PRIOR_TEX.get(x.prior, x.prior)
        lines.append(row("grid 101 / 401", cell, x.nbar, x.ca, x.cb, x.wa, x.wb, x.d, x.n))
    lines.append(r"\addlinespace")
    s = load(r / "lik_coverage_synthetic_g101.csv")
    s = s.groupby(["prior", "nbar"]).agg(c=("cov", "mean"), w=("w", "mean"), n=("rep", "size"))
    for nbar in (1.0, 5.0):
        on, off = s.loc[("atoms3", nbar)], s.loc[("atoms3_off", nbar)]
        lines.append(row("atoms on / off grid", "three atoms", nbar, on.c, off.c, on.w, off.w, None, on.n))
    lines.append(r"\addlinespace")
    # (on MovieLens movies no level exceeds 200 rows at these nbar, so the cap never acts there)
    for name, tex, ta, tb in (("avazu_device_ip", "Avazu device IP, shuffled", "cap200B", "cap1000B"),):
        c = paired(load(r / f"lik_coverage_{name}_shuffled_{ta}.csv"),
                   load(r / f"lik_coverage_{name}_shuffled_{tb}.csv"), ["nbar"])
        for _, x in c.iterrows():
            lines.append(row("thinning cap 200 / 1000", tex, x.nbar, x.ca, x.cb, x.wa, x.wb, x.d, x.n))
    ln = r / "lik_coverage_synthetic_lognormalB.csv"
    if ln.exists():
        lines.append(r"\midrule")
        lines.append(r"\multicolumn{7}{l}{\emph{lognormal popularity, $10^5$ levels (coverage, width)}}\\")
        d = load(ln).groupby(["prior", "nbar"]).agg(c=("cov", "mean"), w=("w", "mean"), n=("rep", "size"))
        for (prior, nbar), x in d.iterrows():
            lines.append(f"lognormal & {PRIOR_TEX.get(prior, prior)} & {nbar:g} & {int(x.n)} & {x.c:.2f} & "
                         f"{x.w:.3f} & --\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    Path(a.tab).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
