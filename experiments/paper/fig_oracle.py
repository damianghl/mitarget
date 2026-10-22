"""
Section 7.2 figures from the real-data subsampling oracle (experiment A and its four new pools).

fig_oracle.pdf     share of the achievable gain lost vs mean level size nbar, one panel per pool
fig_gap_by_n.pdf   where the baseline's gap lives: split by training level size at nbar = 2
tab_oracle.tex     gap to the oracle at nbar = 2, by method, with the gap closed by the frequency prior

Inputs are the ``summary.csv`` and ``baseline_gap_by_n.csv`` files written by the experiment-A
scripts (one results directory per run).
"""

import argparse
import os

import numpy as np
import pandas as pd

from _style import COLORS, plt

POOLS = [
    ("movie_id", "MovieLens movies"),
    ("user_id", "MovieLens users"),
    ("bx_user_implicit", "Book-Crossing users, explicit"),
    ("bx_user", r"Book-Crossing users, $r\geq8$"),
    ("avazu_device_ip", "Avazu device_ip"),
    ("criteo_C1", "Criteo C1"),
]
LINES = [
    ("baseline", False, dict(color=COLORS["baseline"], marker="s", label="baseline")),
    ("freq_mean_shift_lr", False, dict(color=COLORS["freq"], marker="o", label="freq. prior (shift, LR)")),
    ("fixed_m20", False, dict(color=COLORS["fixed_m"], marker="^", label="fixed $m=20$")),
    ("beta_mm", False, dict(color=COLORS["ref"], ls="--", label="moment-matched beta")),
    ("oracle_blind", False, dict(color=COLORS["ref"], ls=":", label="popularity-blind oracle")),
    ("baseline", True, dict(color=COLORS["baseline"], marker="s", mfc="white", ls="--", alpha=0.6,
                            label="baseline, shuffled $q$")),
]
STRATA = [("gap_n0", "0"), ("gap_n1", "1"), ("gap_n2", "2"), ("gap_n3_5", "3-5"),
          ("gap_n6_10", "6-10"), ("gap_n11_1000000000", "11+")]


def load(dirs, name):
    return pd.concat([pd.read_csv(os.path.join(d, name)) for d in dirs], ignore_index=True)


def fig_oracle(summary, out, corr):
    fig, axes = plt.subplots(2, 3, figsize=(6.8, 4.2), sharex=True, sharey=True)
    for ax, (feat, title) in zip(axes.flat, POOLS, strict=False):
        s = summary[summary.feature == feat]
        for method, shuffled, style in LINES:
            g = s[(s.method == method) & (s.shuffled == shuffled)].sort_values("nbar")
            ax.plot(g.nbar, g.share_of_gain_lost.clip(upper=1.05), **style)
        ax.set_xscale("log")
        ax.set_ylim(-0.02, 1.05)
        ax.set_title(f"{title}\ncorr$(\\log p,q)={corr[feat]:+.2f}$")
    for ax in axes[1]:
        ax.set_xlabel(r"mean level size $\bar n$")
    for ax in axes[:, 0]:
        ax.set_ylabel("share of gain lost")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    fig.savefig(out)


def fig_gap_by_n(gap, out, nbar=2.0):
    g = gap[(gap.nbar == nbar) & (~gap.shuffled)].set_index("feature")
    feats = [f for f, _ in POOLS if f in g.index]
    shares = np.array([[g.loc[f, c] / g.loc[f, "gap"] for c, _ in STRATA] for f in feats])
    blues = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    left = np.zeros(len(feats))
    for j, (_, lab) in enumerate(STRATA):
        ax.barh(range(len(feats)), shares[:, j], left=left, color=blues[j], edgecolor="white",
                linewidth=1, label=lab, height=0.7)
        left += np.clip(shares[:, j], 0, None)
    ax.set_yticks(range(len(feats)), [dict(POOLS)[f] for f in feats])
    ax.invert_yaxis()
    ax.set_xlabel(rf"share of the baseline's gap at $\bar n={nbar:g}$")
    ax.grid(axis="y", visible=False)
    ax.legend(title="training $n_x$", ncol=6, frameon=False, loc="lower center",
              bbox_to_anchor=(0.4, 1.0), columnspacing=0.8, handlelength=1.2)
    fig.savefig(out)
    for f, row in zip(feats, shares, strict=False):
        print(f"{f:18s} " + " ".join(f"{lab}:{v:.2f}" for (_, lab), v in zip(STRATA, row, strict=False)))


def tab_oracle(summary, gap, corr, out, nbar=2.0):
    """Gap to the oracle at one nbar (milli-nats, share of the achievable gain in brackets)."""
    s = summary[(summary.nbar == nbar) & (~summary.shuffled)]
    g = gap[(gap.nbar == nbar) & (~gap.shuffled)].set_index("feature")
    cols = ["baseline", "freq_mean_shift_lr", "fixed_m20", "beta_mm", "oracle_blind"]
    lines = []
    for feat, title in POOLS:
        r = s[s.feature == feat].set_index("method")
        achievable = r.loc["marginal", "eCE"] - r.loc["oracle", "eCE"]
        cells = [title.replace("_", "\\_"), f"{corr[feat]:+.2f}", f"{1e3 * achievable:.1f}"]
        for m in cols:
            v = r.loc[m, "gap_to_oracle"]
            cells.append(f"{1e3 * v:.1f} ({100 * v / achievable:.0f}\\%)")
        cells.append(f"{100 * g.loc[feat, 'share_gap_at_n<=1']:.0f}\\%")
        cells.append(f"{100 * g.loc[feat, 'gap_closed_freq_mean_shift_lr']:.0f}\\%")
        lines.append(" & ".join(cells) + " \\\\")
    head = ("pool & corr & gain & baseline & freq.\\ prior & $m{=}20$ & matched beta & blind oracle"
            " & gap at $n_x\\le1$ & closed by freq. \\\\")
    with open(out, "w") as fh:
        fh.write("\n".join(["\\begin{tabular}{l r r r r r r r r r}", "\\toprule", head, "\\midrule",
                             *lines, "\\bottomrule", "\\end{tabular}", ""]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dirs", nargs="+")
    ap.add_argument("--out-dir", default=".")
    a = ap.parse_args()
    summary = load(a.results_dirs, "summary.csv")
    gap = load(a.results_dirs, "baseline_gap_by_n.csv")
    # corr(log p, q) over each pool's levels, from the experiment READMEs / conclusions doc.
    corr = {"movie_id": 0.41, "user_id": -0.27, "bx_user_implicit": -0.24, "bx_user": -0.03,
            "avazu_device_ip": 0.02, "criteo_C1": -0.02}
    fig_oracle(summary, os.path.join(a.out_dir, "fig_oracle.pdf"), corr)
    fig_gap_by_n(gap, os.path.join(a.out_dir, "fig_gap_by_n.pdf"))
    tab_oracle(summary, gap, corr, os.path.join(a.out_dir, "tab_oracle.tex"))


if __name__ == "__main__":
    main()
