"""
Figure 2 (fig:shift): the n+1 shift on a synthetic popularity-linked prior.

200k levels, log p_x ~ 1.5 z, logit q_x = -0.3 + 0.8 z + noise. The frequency prior
(log-linear spread, no LR test) is read at log(1+n) and at log(2+n); the baseline and the
marginal predictor are references. eCE is exact because q_x and p_x are known.
"""

import argparse

import numpy as np
from scipy.special import expit

from mitarget import fit_baseline, fit_freq_prior
from _style import COLORS, plt


def kl(a, b):
    b = np.clip(b, 1e-12, 1 - 1e-12)
    return a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))


def one_draw(N, rng, K=200_000):
    z = rng.normal(size=K)
    p = np.exp(1.5 * z)
    p /= p.sum()
    q = np.clip(expit(-0.3 + 0.8 * z + rng.normal(size=K)), 1e-6, 1 - 1e-6)
    n = rng.binomial(N, p)
    k = rng.binomial(n, q)
    counts = np.stack([n - k, k], axis=1).astype(float)
    fit = fit_freq_prior(counts, spread="linear", lr_test=False)
    base = fit_baseline(counts)
    enc = {
        "at log(1+n)": fit.predict(k, n, shift=0),
        "at log(2+n)": fit.predict(k, n, shift=1),
        "baseline": base.target_encoding[:, 1],
        "marginal": np.full(K, k.sum() / n.sum()),
    }
    return {name: float((p * kl(q, e)).sum()) for name, e in enc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--out", default="fig_shift.pdf")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    Ns = [10_000, 20_000, 50_000, 100_000, 200_000, 400_000]
    res = {N: [one_draw(N, rng) for _ in range(args.reps)] for N in Ns}
    names = ["at log(1+n)", "at log(2+n)", "baseline", "marginal"]
    mean = {m: np.array([np.mean([r[m] for r in res[N]]) for N in Ns]) for m in names}
    sem = {m: np.array([np.std([r[m] for r in res[N]], ddof=1) / np.sqrt(args.reps) for N in Ns])
           for m in names}
    for i, N in enumerate(Ns):
        print(f"N={N:7d} " + "  ".join(f"{m}: {mean[m][i]:.4f}±{sem[m][i]:.4f}" for m in names))

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    style = {
        "at log(2+n)": dict(color=COLORS["freq"], marker="o", label=r"freq. prior at $\log(2+n)$ (shifted)"),
        "at log(1+n)": dict(color=COLORS["freq"], marker="o", ls="--", mfc="white",
                            label=r"freq. prior at $\log(1+n)$"),
        "baseline": dict(color=COLORS["baseline"], marker="s", label="baseline (single beta)"),
        "marginal": dict(color=COLORS["ref"], ls=":", label="marginal predictor"),
    }
    for m in names:
        ax.errorbar(Ns, mean[m], yerr=sem[m], capsize=0, **style[m])
    ax.set_xscale("log")
    ax.set_xlabel("training rows $N$")
    ax.set_ylabel("eCE (nats)")
    ax.legend(frameon=False)
    fig.savefig(args.out)


if __name__ == "__main__":
    main()
