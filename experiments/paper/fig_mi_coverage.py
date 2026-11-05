"""
Section 7.4 figure and table: coverage of the intervals for I(X;Y).

fig_mi_coverage.pdf   true I, the baseline estimate, the median 95% credible and moment intervals and
                      the identified interval at the population moments vs nbar, for three populations
tab_mi_coverage.tex   coverage and mean width of both intervals, and the width of the identified
                      interval, at nbar = 1 and 5, synthetic priors (10^5 equally popular levels) and
                      the oracle pools (real and shuffled)

``--with-lik`` adds the likelihood-region interval (``mi_coverage.py --lik-only``) as a band in the
figure; ``--intervals`` picks the table's intervals. The paper uses

    python fig_mi_coverage.py summary.csv --with-lik --intervals cred lik
    python fig_mi_coverage.py summary.csv --fig full.pdf --tab tab_mi_coverage_full.tex \
        --intervals cred mom lik

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


def fig(s, out, with_lik=False):
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.3))
    for ax, (pool, shuffled, title) in zip(axes, PANELS, strict=True):
        g = s[(s.pool == pool) & (s.shuffled == shuffled)].sort_values("nbar")
        x = g.nbar.to_numpy()
        ax.fill_between(x, g.mom_lo, g.mom_hi, color=COLORS["ref"], alpha=0.18, lw=0,
                        label="moment interval")
        ax.fill_between(x, g.cred_lo, g.cred_hi, color=COLORS["baseline"], alpha=0.35, lw=0,
                        label="credible interval")
        if with_lik:
            ax.fill_between(x, g.lik_lo, g.lik_hi, color=COLORS["npmle"], alpha=0.25, lw=0,
                            label="likelihood interval")
        ax.plot(x, g.id_lo, color=COLORS["ref"], lw=0.8, ls=":", label="identified")
        ax.plot(x, g.id_hi, color=COLORS["ref"], lw=0.8, ls=":")
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
    fig.legend(handles, labels, loc="lower center", ncol=7 if with_lik else 6, frameon=False,
               bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    fig.savefig(out)


COLS = {"cred": "cred.", "mom": "mom.", "lik": "lik."}


def _cells(g, intervals):
    out = []
    for nbar in NBARS:
        r = g[g.nbar == nbar]
        if r.empty:
            out += ["--"] * (2 * len(intervals) + 1)
            continue
        r = r.iloc[0]
        for k in intervals:
            out += [f"{r[f'cov_{k}']:.2f}", f"{r[f'width_{k}']:.3f}"]
        out.append(f"{r.width_id:.3f}")
    return out


def table(s, out, intervals=("cred", "mom")):
    m = 2 * len(intervals) + 1
    heads = [h for k in intervals for h in (COLS[k], "width")] + ["ident."]
    lines = [rf"\begin{{tabular}}{{lr {'c' * m} {'c' * m}}}", r"\toprule",
             rf" & & \multicolumn{{{m}}}{{c}}{{$\bar n=1$}} & \multicolumn{{{m}}}{{c}}{{$\bar n=5$}}\\",
             rf"\cmidrule(lr){{3-{2 + m}}}\cmidrule(lr){{{3 + m}-{2 + 2 * m}}}",
             " & ".join(["population", "$I$"] + heads * 2) + r"\\",
             r"\midrule",
             rf"\multicolumn{{{2 + 2 * m}}}{{l}}{{\emph{{synthetic, $10^5$ equally popular levels}}}}\\"]
    for prior, label in PRIORS:
        g = s[s.pool == f"{prior}|{SYNTH}"]
        lines.append(" & ".join([label, f"{g.I.mean():.3f}"] + _cells(g, intervals)) + r"\\")
    for shuffled, head in [(False, "oracle pools, real"), (True, "oracle pools, shuffled $q$")]:
        lines += [r"\midrule", rf"\multicolumn{{{2 + 2 * m}}}{{l}}{{\emph{{{head}}}}}\\"]
        for pool, label in POOLS:
            g = s[(s.pool == pool) & (s.shuffled == shuffled)]
            lines.append(" & ".join([label, f"{g.I.mean():.3f}"] + _cells(g, intervals)) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summary")
    ap.add_argument("--fig", default="fig_mi_coverage.pdf")
    ap.add_argument("--tab", default="tab_mi_coverage.tex")
    ap.add_argument("--with-lik", action="store_true",
                    help="add the likelihood-region interval to the figure")
    ap.add_argument("--intervals", nargs="+", default=["cred", "mom"], choices=list(COLS),
                    help="intervals in the table, in order (default: cred mom)")
    a = ap.parse_args()
    s = pd.read_csv(a.summary)
    fig(s, a.fig, a.with_lik)
    table(s, a.tab, a.intervals)


if __name__ == "__main__":
    main()
