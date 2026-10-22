"""
Figure 3 (fig:interval): the identified interval for I(X;Y) as a function of the largest
usable level size L, for three priors on q, with the true I and the moment-matched beta point.
"""

import argparse

import numpy as np
from scipy import integrate, stats

from mitarget import identified_interval, worst_case_halfwidth
from mitarget.mi_interval import binary_entropy
from _style import COLORS, plt


def beta_mi(a, b):
    mu = a / (a + b)
    eh = integrate.quad(lambda q: binary_entropy(q) * stats.beta.pdf(q, a, b), 0, 1, limit=200)[0]
    return float(binary_entropy(mu) - eh)


def beta_prior(a, b):
    def moments(L):
        return np.array([stats.beta.moment(j, a, b) for j in range(1, L + 1)])
    return moments, beta_mi(a, b)


def atomic_prior(atoms, weights):
    atoms, weights = np.asarray(atoms, float), np.asarray(weights, float)

    def moments(L):
        return np.array([np.sum(weights * atoms ** j) for j in range(1, L + 1)])
    mu = np.sum(weights * atoms)
    return moments, float(binary_entropy(mu) - np.sum(weights * binary_entropy(atoms)))


def matched_beta_mi(m1, m2):
    s = (m1 - m2) / (m2 - m1 ** 2)          # concentration a + b from the first two moments
    return beta_mi(m1 * s, (1 - m1) * s)


PRIORS = [
    (r"$\mathrm{Beta}(0.5,0.5)$", beta_prior(0.5, 0.5), True),
    (r"atoms $\{.2,.3,.85\}$, weights $\{.3,.4,.3\}$", atomic_prior([0.2, 0.3, 0.85], [0.3, 0.4, 0.3]),
     False),
    (r"junk $f{=}0.5$, $q_0{=}0.1$", atomic_prior([0.1, 0.5, 0.9], [0.25, 0.5, 0.25]), True),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L-max", type=int, default=8)
    ap.add_argument("--out", default="fig_interval.pdf")
    args = ap.parse_args()
    # L = 1 gives [0, h(mu)] for every prior; start at L = 2 so the panels stay readable.
    Ls = np.arange(2, args.L_max + 1)
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.2), sharey=False)
    for ax, (name, (moments, I_true), sym) in zip(axes, PRIORS, strict=False):
        m = moments(args.L_max)
        lo, hi = np.array([identified_interval(m[:L], symmetric=sym) for L in Ls]).T
        I_beta = matched_beta_mi(m[0], m[1])
        print(f"{name}: I={I_true:.4f} beta={I_beta:.4f} "
              + " ".join(f"L{L}:[{a:.3f},{b:.3f}]" for L, a, b in zip(Ls, lo, hi, strict=False)))
        ax.fill_between(Ls, lo, hi, step=None, color=COLORS["baseline"], alpha=0.18, lw=0,
                        label="identified interval")
        ax.plot(Ls, lo, color=COLORS["baseline"], lw=1)
        ax.plot(Ls, hi, color=COLORS["baseline"], lw=1)
        ax.axhline(I_true, color="black", lw=1.2, label="true $I$")
        ax.axhline(I_beta, color=COLORS["freq"], lw=1.2, ls="--", label="moment-matched beta")
        ax.set_title(name + (" (symmetric)" if sym else ""))
        ax.set_xlabel("largest level size $L$")
        ax.set_xticks(Ls)
    axes[0].set_ylabel("$I(X;Y)$ (nats)")
    axes[1].legend(frameon=False, loc="upper right")
    print("E_L(h):", " ".join(f"{worst_case_halfwidth(L):.4f}" for L in Ls))
    fig.tight_layout()
    fig.savefig(args.out)


if __name__ == "__main__":
    main()
