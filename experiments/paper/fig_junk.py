"""
Figure for Section 7.1 (fig:junk): share of the oracle's gain lost on the junk-fraction family,
as heatmaps over (f, nbar), one panel per method. Reads the summary written by run_junk_fraction.py.
"""

import argparse

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from _style import plt

PANELS = [("baseline", "baseline"), ("npmle", "NPMLE"), ("fixed_m20", "fixed $m=20$")]
# One-hue sequential ramp, light (nothing lost) to dark (everything lost).
CMAP = LinearSegmentedColormap.from_list("blue", ["#f4f8fe", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summary")
    ap.add_argument("--kx", type=int, default=2500)
    ap.add_argument("--min-gain", type=float, default=1e-3,
                    help="blank cells where the oracle gains less than this over the marginal (nats)")
    ap.add_argument("--out", default="fig_junk.pdf")
    a = ap.parse_args()
    s = pd.read_csv(a.summary)
    fs = sorted(s.f.unique())
    Ns = sorted(s.N.unique())
    fig, axes = plt.subplots(1, len(PANELS), figsize=(6.8, 2.3), sharey=True)
    for ax, (m, title) in zip(axes, PANELS, strict=False):
        g = s[s.method == m]
        share = g.pivot(index="f", columns="N", values="share_of_gain_lost").loc[fs, Ns].to_numpy()
        gain = g.pivot(index="f", columns="N", values="gain").loc[fs, Ns].to_numpy()
        share = np.where(gain > a.min_gain, share, np.nan)
        ax.imshow(np.clip(share, 0, 1), cmap=CMAP, vmin=0, vmax=1, aspect="auto", origin="lower")
        for i in range(len(fs)):
            for j in range(len(Ns)):
                v = share[i, j]
                if np.isfinite(v):
                    txt = f"{100 * v:.0f}" if v >= 0.095 else f"{100 * v:.1f}"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=6.5,
                            color="white" if min(v, 1) > 0.5 else "#1a1a19")
        ax.set_xticks(range(len(Ns)), [f"{N / a.kx:g}" for N in Ns])
        ax.set_xlabel(r"$\bar n = N/k_x$")
        ax.set_title(title + " (% of gain lost)")
        ax.grid(False)
    axes[0].set_yticks(range(len(fs)), [f"{f:g}" for f in fs])
    axes[0].set_ylabel("junk fraction $f$")
    fig.tight_layout()
    fig.savefig(a.out)
    piv = s.pivot_table(index=["method", "f"], columns="N", values="share_of_gain_lost")
    print((100 * piv).round(1).to_string())


if __name__ == "__main__":
    main()
