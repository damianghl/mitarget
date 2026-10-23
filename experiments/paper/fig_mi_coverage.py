"""
Section 7.4 figure and table: coverage of the intervals for I(X;Y).

fig_mi_coverage.pdf   true I, the baseline estimate and the median 95% credible and moment intervals
                      vs nbar, for three populations
tab_mi_coverage.tex   coverage and mean width of both intervals at nbar = 1 and 5, synthetic priors
                      (10^5 equally popular levels) and the oracle pools (real and shuffled)

Input: the ``summary.csv`` written by ``experiments/mi_coverage_tables.py``.

    python fig_mi_coverage.py mi_tables/summary.csv
"""

import argparse

import numpy as np
import pandas as pd

from _style import COLORS, plt

PRIORS = [
    ("junk_f0.5", r"junk $f{=}0.5$"),
    ("junk_f0.8", r"junk $f{=}0.8$"),
    ("atoms3", r"atoms $\{.2,.3,.85\}$"),
    ("beta_0.5_0.5", r"$\mathrm{Beta}(0.5,0.5)$"),
    ("beta_1_3", r"$\mathrm{Beta}(1,3)$"),
]
POOLS = [
    ("movie_id", "MovieLens movies"),
    ("user_id", "MovieLens users"),
    ("bx_user_implicit", "BX users, rated"),
    ("bx_user", r"BX users, $r\geq8$"),
    ("avazu_device_ip", "Avazu device\\_ip"),
    ("criteo_C1", "Criteo C1"),
]
SYNTH = "K=100000|uniform"
PANELS = [
    (f"atoms3|{SYNTH}", False, r"synthetic: atoms $\{.2,.3,.85\}$"),
    ("avazu_device_ip", False, "Avazu device_ip (real)"),
    ("movie_id", False, "MovieLens movies (real)"),
]
NBARS = (1.0, 5.0)


def fig(s, out):
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.3))
    for ax, (pool, shuffled, title) in zip(axes, PANELS, strict=True):
        g = s[(s.pool == pool) & (s.shuffled == shuffled)].sort_values("nbar")
        x = g.nbar.to_numpy()
        ax.fill_between(x, g.mom_lo, g.mom_hi, color=COLORS["ref"], alpha=0.18, lw=0,
                        label="moment interval")
        ax.fill_between(x, g.cred_lo, g.cred_hi, color=COLORS["baseline"], alpha=0.35, lw=0,
                        label="credible interval")
        ax.plot(x, g.I, color="black", lw=1.2, label="true $I$")
        ax.plot(x, g.I + g.bias_baseline, color=COLORS["baseline"], marker="s", label="baseline")
        ax.plot(x, g.I + g.bias_npmle, color=COLORS["npmle"], marker="v", ls="--", label="NPMLE")
        ax.set_xscale("log")
        ax.set_xticks(x, [f"{v:g}" for v in x])
        ax.minorticks_off()
        ax.set_xlabel(r"mean level size $\bar n$")
        ax.set_title(title)
        ax.set_ylim(0, 1.05 * float(np.nanmax(g.mom_hi)))
    axes[0].set_ylabel("$I(X;Y)$ (nats)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    fig.savefig(out)


def _cells(g):
    out = []
    for nbar in NBARS:
        r = g[g.nbar == nbar]
        if r.empty:
            out += ["--"] * 4
            continue
        r = r.iloc[0]
        out += [f"{r.cov_cred:.2f}", f"{r.width_cred:.3f}", f"{r.cov_mom:.2f}", f"{r.width_mom:.3f}"]
    return out


def table(s, out):
    lines = [r"\begin{tabular}{lr cccc cccc}", r"\toprule",
             r" & & \multicolumn{4}{c}{$\bar n=1$} & \multicolumn{4}{c}{$\bar n=5$}\\",
             r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
             r"population & $I$ & cred. & width & mom. & width & cred. & width & mom. & width\\",
             r"\midrule", r"\multicolumn{10}{l}{\emph{synthetic, $10^5$ equally popular levels}}\\"]
    for prior, label in PRIORS:
        g = s[s.pool == f"{prior}|{SYNTH}"]
        lines.append(" & ".join([label, f"{g.I.mean():.3f}"] + _cells(g)) + r"\\")
    for shuffled, head in [(False, "oracle pools, real"), (True, "oracle pools, shuffled $q$")]:
        lines += [r"\midrule", rf"\multicolumn{{10}}{{l}}{{\emph{{{head}}}}}\\"]
        for pool, label in POOLS:
            g = s[(s.pool == pool) & (s.shuffled == shuffled)]
            lines.append(" & ".join([label, f"{g.I.mean():.3f}"] + _cells(g)) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summary")
    ap.add_argument("--fig", default="fig_mi_coverage.pdf")
    ap.add_argument("--tab", default="tab_mi_coverage.tex")
    a = ap.parse_args()
    s = pd.read_csv(a.summary)
    fig(s, a.fig)
    table(s, a.tab)


if __name__ == "__main__":
    main()
