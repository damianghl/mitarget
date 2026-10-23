"""
Regret of the moment-matched beta on the true distribution of q in each real-data oracle pool (Sec. 4,
numerical check): exact sums over binomial outcomes at n = 1, 2, 3, 5, in milli-nats, with the third central
moment of q and of its moment-matched beta. Pools as in realdata_oracle.py: levels with >= 100 rows,
y = rating >= 4 in the prepared .tsv files, q weighted by rows (the popularity-blind law of q).

    python real_pool_regret.py movie_id=ml1m_ratings.tsv:movie_id bx_user=bx_user.tsv ...
"""

import argparse
from math import comb

import numpy as np
import pandas as pd


def kl(a, b):
    return a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))


def pool_regret(path, feature, min_rows=100, sizes=(1, 2, 3, 5)):
    d = pd.read_csv(path, sep="\t", usecols=[feature, "rating"])
    g = d.assign(y=(d.rating >= 4).astype(float)).groupby(feature).y.agg(["size", "mean"])
    g = g[g["size"] >= min_rows]
    w = g["size"].to_numpy(float) / g["size"].sum()
    q = np.clip(g["mean"].to_numpy(), 1e-6, 1 - 1e-6)
    mu = (w * q).sum()
    v = (w * (q - mu) ** 2).sum()
    s = mu * (1 - mu) / v - 1
    k3 = (w * (q - mu) ** 3).sum()
    k3_beta = 2 * v * (1 - 2 * mu) / (s + 2)
    regret = {}
    for n in sizes:
        r = 0.0
        for k in range(n + 1):
            lik = comb(n, k) * q ** k * (1 - q) ** (n - k)
            pk = (w * lik).sum()
            r += pk * kl((w * lik * q).sum() / pk, (k + mu * s) / (n + s))
        regret[n] = r
    return dict(mu=mu, t=1 / (1 + s), k3=k3, k3_beta=k3_beta, I=(w * kl(q, mu)).sum(),
                **{f"D{n}": r for n, r in regret.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pools", nargs="+", help="name=path[:feature]; feature defaults to name")
    a = ap.parse_args()
    for spec in a.pools:
        name, rest = spec.split("=", 1)
        path, _, feature = rest.partition(":")
        r = pool_regret(path, feature or name)
        print(f"{name:18s} mu={r['mu']:.3f} t={r['t']:.3f} k3={r['k3']:+.4f} k3_beta={r['k3_beta']:+.4f} "
              f"I={1e3 * r['I']:.1f} mnats | regret n=1,2,3,5 (mnats): "
              + " ".join(f"{1e3 * r[f'D{n}']:.3f}" for n in (1, 2, 3, 5)))


if __name__ == "__main__":
    main()
