"""
Figure 1 (teaser): prediction is easy, information is hard.

(a) Where the test rows of real high-cardinality features fall: share of out-of-fold test rows by the
    training count n_x of their level (bakeoff, 5-fold CV). On levels seen at most once (dark) the fitted beta
    is exactly the Bayes oracle when q does not depend on popularity (Prop. 1).
(b) On the true distribution of q of three oracle pools (row-weighted, infinitely many levels): regret of the
    moment-matched beta on a level seen n times, and half-width of the identified interval for I(X;Y) when
    levels are seen at most L = n times, both as a percentage of I(X;Y).

    python fig_teaser.py --by-n table_by_n.csv --pools movie_id=ml1m_ratings.tsv:movie_id \
        bx_user_implicit=bx_user_implicit.tsv avazu_device_ip=avazu_device_ip.tsv --out fig_teaser.pdf

``table_by_n.csv`` is written by ``experiments/realdata_tables.py``; the pool files are the prepared ``.tsv``
files of ``experiments/realdata_oracle.py``.
"""

import argparse
from math import comb

import numpy as np
import pandas as pd

from _style import COLORS, plt
from mitarget import identified_interval
from real_pool_regret import kl

FEATURES = [
    ("avazu/user", "Avazu user (1.5M levels)"),
    ("bx/isbn", r"Book-Crossing ISBN, $r\geq8$"),
    ("avazu/device_ip", "Avazu device_ip"),
    ("bx_explicit/isbn", "Book-Crossing ISBN, rated"),
    ("amazon/RESOURCE", "Amazon RESOURCE"),
    ("criteo/C1", "Criteo C1"),
    ("bx/user", r"Book-Crossing user, $r\geq8$"),
    ("movielens/movie_id", "MovieLens movies"),
]
#: (label, training-count bins as n_lo values of table_by_n.csv, colour)
SEGMENTS = [
    ("$n=0$", [0], "#1d4f91"),
    ("$n=1$", [1], "#4f8fdc"),
    ("$n=2$", [2], "#c9c9c9"),
    ("3-10", [3, 6], "#a9a9a9"),
    ("11-100", [11, 31], "#868686"),
    (">100", [101], "#5e5e5e"),
]
POOLS = {
    "movie_id": "MovieLens movies",
    "bx_user_implicit": "Book-Crossing users, rated",
    "avazu_device_ip": "Avazu device_ip",
}
POOL_COLORS = {"movie_id": COLORS["baseline"], "bx_user_implicit": COLORS["freq"],
               "avazu_device_ip": COLORS["npmle"]}
SIZES = range(1, 7)


def pool_curves(path, feature, min_rows=100):
    """Regret of the moment-matched beta at n and identified half-width of I at L = n, as shares of I."""
    d = pd.read_csv(path, sep="\t", usecols=[feature, "rating"])
    g = d.assign(y=(d.rating >= 4).astype(float)).groupby(feature).y.agg(["size", "mean"])
    g = g[g["size"] >= min_rows]
    w = g["size"].to_numpy(float) / g["size"].sum()
    q = np.clip(g["mean"].to_numpy(), 1e-6, 1 - 1e-6)
    mu = (w * q).sum()
    s = mu * (1 - mu) / (w * (q - mu) ** 2).sum() - 1
    info = (w * kl(q, mu)).sum()
    regret, half = [], []
    for n in SIZES:
        r = 0.0
        for k in range(n + 1):
            lik = comb(n, k) * q ** k * (1 - q) ** (n - k)
            pk = (w * lik).sum()
            r += pk * kl((w * lik * q).sum() / pk, (k + mu * s) / (n + s))
        regret.append(max(r, 0.0) / info)
        lo, hi = identified_interval([(w * q ** j).sum() for j in range(1, n + 1)])
        half.append((hi - lo) / 2 / info)
    return np.array(regret), np.array(half)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--by-n", required=True)
    ap.add_argument("--pools", nargs="+", required=True, help="name=path[:feature]")
    ap.add_argument("--out", default="fig_teaser.pdf")
    a = ap.parse_args()

    t = pd.read_csv(a.by_n)
    t = t[t.method == "K=1 baseline"].pivot(index="task", columns="n_lo", values="row_share").fillna(0.0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.75, 2.5), gridspec_kw=dict(width_ratios=[1.25, 1]))

    labels = [lab for _, lab in FEATURES]
    y = np.arange(len(FEATURES))[::-1]
    left = np.zeros(len(FEATURES))
    for seg, bins, col in SEGMENTS:
        share = np.array([t.loc[task, bins].sum() for task, _ in FEATURES])
        ax1.barh(y, share, left=left, color=col, height=0.7, label=seg, edgecolor="white", linewidth=0.4)
        left += share
    low = np.array([t.loc[task, [0, 1]].sum() for task, _ in FEATURES])
    for yi, v in zip(y, low, strict=True):
        if v >= 0.06:
            ax1.text(v / 2, yi, f"{100 * v:.0f}%", color="white", ha="center", va="center", fontsize=6.5)
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels)
    ax1.set_xlim(0, 1)
    ax1.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax1.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    ax1.set_xlabel("share of test rows, by training count $n_x$ of their level")
    ax1.grid(axis="y", visible=False)
    ax1.legend(ncol=6, loc="lower left", bbox_to_anchor=(-0.02, 1.0), frameon=False, handlelength=1.0,
               columnspacing=0.8, fontsize=6.5)
    ax1.set_title("(a) where the test rows are", loc="left", pad=24)

    floor = 1e-5
    for spec in a.pools:
        name, rest = spec.split("=", 1)
        path, _, feature = rest.partition(":")
        regret, half = pool_curves(path, feature or name)
        col = POOL_COLORS.get(name, COLORS["ref"])
        x = np.array(list(SIZES))
        ax2.plot(x, np.maximum(regret, floor), color=col, marker="o", label=POOLS.get(name, name))
        ax2.plot(x, half, color=col, ls="--", marker="s", mfc="white")
    ax2.set_yscale("log")
    ax2.set_ylim(floor * 0.7, 10)
    ax2.set_yticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10])
    ax2.set_yticklabels(["0 (floor)", "0.01%", "0.1%", "1%", "10%", "100%", "1000%"])
    ax2.set_xticks(list(SIZES))
    ax2.set_xlabel("$n$ (regret, solid) or $L$ (half-width, dashed)")
    ax2.set_ylabel(r"share of $I(X;Y)$")
    pools = ax2.legend(loc="lower right", frameon=False, fontsize=6.5, handlelength=1.5)
    ax2.add_artist(pools)
    style = [plt.Line2D([], [], color="#444444", marker="o", label="prediction: regret of matched beta"),
             plt.Line2D([], [], color="#444444", ls="--", marker="s", mfc="white",
                        label="information: identified half-width")]
    ax2.legend(handles=style, ncol=1, loc="lower left", bbox_to_anchor=(-0.02, 0.99), frameon=False,
               fontsize=6.5, handlelength=2.0, borderpad=0.1, labelspacing=0.2)
    ax2.set_title("(b) prediction is easy, information is hard", loc="left", pad=24)
    fig.subplots_adjust(wspace=0.55)
    fig.savefig(a.out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
