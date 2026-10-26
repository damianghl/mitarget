"""
Regret of the moment-matched beta at large level size n (Appendix, second-order gap and large-n behaviour):
exact sums over binomial outcomes for mixtures of betas and for atomic priors, no Monte Carlo. Prints
Delta_n and Delta_n divided by the rate of the appendix's propositions (n^-2, n^-2 log n,
n^-(1 + alpha), 1/n), which should level off.

    python large_n_regret.py
"""

import numpy as np
from scipy.special import betaln, gammaln, logsumexp

#: name -> ("beta", [(w, a, b), ...]) or ("atoms", (atoms, weights)); the boundary exponents in the comments.
PRIORS = {
    "vanishing": ("beta", [(0.5, 2.0, 4.0), (0.5, 4.0, 2.0)]),      # alpha = beta = 2
    "flat": ("beta", [(2 / 3, 1.0, 1.0), (1 / 3, 2.0, 1.0)]),       # density 2(1 + q)/3, alpha = beta = 1
    "u_shaped": ("beta", [(0.5, 0.3, 2.0), (0.5, 2.0, 0.3)]),       # alpha = beta = 0.3
    "atoms3": ("atoms", ([0.2, 0.3, 0.85], [0.3, 0.4, 0.3])),
    "junk_f0.5": ("atoms", ([0.1, 0.5, 0.9], [0.25, 0.5, 0.25])),
}


def _moments(kind, par):
    if kind == "beta":
        m1 = sum(w * a / (a + b) for w, a, b in par)
        m2 = sum(w * a * (a + 1) / ((a + b) * (a + b + 1)) for w, a, b in par)
    else:
        q, w = map(np.asarray, par)
        m1, m2 = (w * q).sum(), (w * q ** 2).sum()
    return m1, m2


def _log_marginals(kind, par, n):
    """log P(K = k | n) and log E[q 1{K = k} | n] for k = 0..n."""
    k = np.arange(n + 1)
    lc = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
    if kind == "beta":
        lp = [np.log(w) + lc + betaln(k + a, n - k + b) - betaln(a, b) for w, a, b in par]
        lq = [np.log(w) + lc + betaln(k + a + 1, n - k + b) - betaln(a, b) for w, a, b in par]
    else:
        q, w = map(np.asarray, par)
        lp = [np.log(wj) + lc + k * np.log(qj) + (n - k) * np.log1p(-qj) for qj, wj in zip(q, w, strict=True)]
        lq = [x + np.log(qj) for x, qj in zip(lp, q, strict=True)]
    return logsumexp(lp, axis=0), logsumexp(lq, axis=0)


def regret(kind, par, n):
    """Delta_n of the moment-matched beta: sum_k P(k|n) kl(p_pi(k,n) || (k + mu s)/(n + s))."""
    m1, m2 = _moments(kind, par)
    s = m1 * (1 - m1) / (m2 - m1 ** 2) - 1
    lp, lq = _log_marginals(kind, par, n)
    p = np.exp(lq - lp)
    r = (np.arange(n + 1) + m1 * s) / (n + s)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl = (np.where(p > 0, p * np.log(p / r), 0.0)
              + np.where(p < 1, (1 - p) * np.log((1 - p) / (1 - r)), 0.0))
    return float(np.exp(lp) @ kl), m1 * s, (1 - m1) * s


#: rate predicted for each prior
RATES = {
    "vanishing": ("n^-2", lambda n: n ** -2.0),
    "flat": ("n^-2 log n", lambda n: np.log(n) / n ** 2),
    "u_shaped": ("n^-1.3", lambda n: n ** -1.3),
    "atoms3": ("1/n", lambda n: 1.0 / n),
    "junk_f0.5": ("1/n", lambda n: 1.0 / n),
}


def main():
    sizes = [4, 16, 64, 256, 1024, 4096]
    print("prior        a      b     rate        " + "".join(f"{n:>18d}" for n in sizes))
    for name, (kind, par) in PRIORS.items():
        _, a, b = regret(kind, par, 1)
        label, rate = RATES[name]
        cells = []
        for n in sizes:
            d = regret(kind, par, n)[0]
            cells.append(f"{d:9.2e} ({d / rate(n):5.3f})")
        print(f"{name:10s} {a:6.3f} {b:6.3f}  {label:10s}" + "".join(f"{c:>18s}" for c in cells))


if __name__ == "__main__":
    main()
