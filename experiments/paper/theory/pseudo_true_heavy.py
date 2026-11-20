"""
Type-II ML beta fit under heavy-tailed level sizes.

The pseudo-true concentration maximises the expected beta-binomial log-likelihood over the level-size
distribution. With equal popularities (``pseudo_true.py``) level sizes are Poisson and the fit stays
within 5e-5 nats of the oracle on levels seen at most once. Here large levels dominate the likelihood:

- design "1/2/200": 60% / 30% / 10% of the levels seen 1 / 2 / 200 times (per-level regret only);
- design "lognormal": popularities lognormal (sigma = 1.5), Poisson counts, nbar = N / K in {1, 2, 5}.

For each prior: s_pi (moment-matched), pseudo-true s, the per-level regret Delta_0, Delta_1 of the
pseudo-true fit (zero for the moment-matched beta), and, for the lognormal design, the test-weighted
loss on n <= 1, w_0 Delta_0 + w_1 Delta_1, in nats and as a share of I.

    python pseudo_true_heavy.py
"""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import betaln, gammaln

PRIORS = [(np.array([.1, .5, .9]), np.array([.25, .5, .25])), (np.array([.1, .5, .9]), np.array([.4, .2, .4])),
          (np.array([.05, .95]), np.array([.5, .5])), (np.array([.2, .3, .85]), np.array([.3, .4, .3]))]


def h(q):
    q = np.clip(q, 1e-300, 1 - 1e-16)
    return -q * np.log(q) - (1 - q) * np.log(1 - q)


def loglik_table(at, w, nmax):
    """P(k | n) under the prior, for n = 0..nmax, as a list of arrays."""
    out = []
    for n in range(nmax + 1):
        k = np.arange(n + 1)
        lc = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
        la = np.log(at)[:, None] * k + np.log1p(-at)[:, None] * (n - k)
        out.append((w[:, None] * np.exp(lc + la)).sum(0))
    return out


def bb_log(n, mu, s):
    k = np.arange(n + 1)
    lc = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
    return lc + betaln(mu * s + k, (1 - mu) * s + n - k) - betaln(mu * s, (1 - mu) * s)


def pseudo_true(pk, fn, mu):
    ns = [n for n in range(len(fn)) if fn[n] > 1e-12 and n >= 2]

    def nll(ls):
        s = np.exp(ls)
        return -sum(fn[n] * (pk[n] * bb_log(n, mu, s)).sum() for n in ns)
    return np.exp(minimize_scalar(nll, bounds=(-6, 8), method="bounded").x)


def delta(at, w, n, mu, s):
    k = np.arange(n + 1)
    lik = at[:, None] ** k * (1 - at)[:, None] ** (n - k)
    post = (w[:, None] * lik)
    pk = post.sum(0)
    p_or = (post * at[:, None]).sum(0) / pk
    r = (mu * s + k) / (s + n)
    kl = p_or * np.log(p_or / r) + (1 - p_or) * np.log((1 - p_or) / (1 - r))
    from math import comb
    return float((np.array([comb(n, j) for j in k]) * pk * kl).sum())


def main():
    rng = np.random.default_rng(0)
    K = 200_000
    p = rng.lognormal(0, 1.5, K)
    p /= p.sum()
    print("prior | s_pi | design | pseudo-true s | Delta_0 | Delta_1 | w0+w1 | loss n<=1 | share of I")
    for at, w in PRIORS:
        mu = (w * at).sum()
        v = (w * at ** 2).sum() - mu ** 2
        s_pi = mu * (1 - mu) / v - 1
        I_val = h(mu) - (w * h(at)).sum()
        pk = loglik_table(at, w, 200)
        fn = np.zeros(201)
        fn[[1, 2, 200]] = [.6, .3, .1]
        s = pseudo_true(pk, fn, mu)
        print(f"{at}/{w} | {s_pi:.3f} | 1/2/200 | {s:.3f} | {delta(at, w, 0, mu, s):.2e} | "
              f"{delta(at, w, 1, mu, s):.2e} | - | - | -", flush=True)
        for nbar in (1, 2, 5):
            lam = nbar * K * p
            nmax = int(min(np.quantile(lam, 0.9999) * 2 + 50, 3000))
            ns = np.arange(nmax + 1)
            logpois = ns[None, :] * np.log(lam)[:, None] - lam[:, None] - gammaln(ns + 1)[None, :]
            pois = np.exp(logpois)
            fn = pois.mean(0)
            wn = (p[:, None] * pois).sum(0)
            pk = loglik_table(at, w, nmax)
            s = pseudo_true(pk, fn, mu)
            d0, d1 = delta(at, w, 0, mu, s), delta(at, w, 1, mu, s)
            loss = wn[0] * d0 + wn[1] * d1
            print(f"{at}/{w} | {s_pi:.3f} | lognormal nbar={nbar} | {s:.3f} | {d0:.2e} | {d1:.2e} | "
                  f"{wn[0] + wn[1]:.3f} | {loss:.2e} | {loss / I_val:.2%}", flush=True)


if __name__ == "__main__":
    main()
