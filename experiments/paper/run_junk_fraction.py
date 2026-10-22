"""
Synthetic sweep for Section 7.1: scenario 4 with an unknown junk fraction f.

kx states with uniform popularity; a fraction f has q = 1/2, the rest is split between q0 and
1 - q0, with q0 ~ U[0, 1] per draw. The reference oracle knows the three-atom family and fits
(f, q0) by marginal likelihood. Every method is scored on the same draw (paired eCE).
Ported from the project's scenario_junk_fraction/ run, with NPMLE as the richer prior.
"""

import argparse
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

from mitarget import fit_baseline
from mitarget.competitors import fit_npmle
from mitarget.core import fixed_m_encoding, qhat_y
from mitarget.finite_mixture import fit_finite_mixture
from mitarget.metrics import exact_mutual_information, excess_cross_entropy


def ground_truth(kx, f, q0, rng):
    n_junk = int(round(f * kx))
    n_inf = kx - n_junk
    labels = np.concatenate([np.zeros(n_junk, int), np.ones(n_inf - n_inf // 2, int), np.full(n_inf // 2, 2)])
    rng.shuffle(labels)
    q1 = np.choose(labels, [0.5, q0, 1.0 - q0])
    return np.full(kx, 1.0 / kx), np.column_stack([1.0 - q1, q1])


def _component_loglik(n, f, q0):
    n0, n1 = n[:, 0], n[:, 1]
    q0 = np.clip(q0, 1e-12, 1 - 1e-12)
    pi_inf = np.clip((1.0 - f) / 2.0, 1e-300, None)
    return np.stack([np.log(np.clip(f, 1e-300, None)) + (n0 + n1) * np.log(0.5),
                     np.log(pi_inf) + n1 * np.log(q0) + n0 * np.log1p(-q0),
                     np.log(pi_inf) + n1 * np.log1p(-q0) + n0 * np.log(q0)], axis=1)


def oracle_encoding(n, f, q0):
    L = _component_loglik(n, f, q0)
    w = np.exp(L - L.max(axis=1, keepdims=True))
    w /= w.sum(axis=1, keepdims=True)
    q1 = 0.5 * w[:, 0] + q0 * w[:, 1] + (1 - q0) * w[:, 2]
    return np.column_stack([1 - q1, q1])


def _neg_marglik(theta, n):
    L = _component_loglik(n, expit(theta[0]), 0.5 * expit(theta[1]))
    m = L.max(axis=1)
    return -float(np.sum(m + np.log(np.exp(L - m[:, None]).sum(axis=1))))


def fit_f_q0(n):
    n = n[n.sum(axis=1) > 0]
    best = None
    for f0 in (0.2, 0.5, 0.8):
        for s in (0.05, 0.15, 0.3, 0.45):
            r = minimize(_neg_marglik, [logit(f0), logit(2 * s)], args=(n,), method="L-BFGS-B",
                         bounds=[(-12, 12), (-12, 12)])
            if best is None or r.fun < best[0]:
                best = (r.fun, expit(r.x[0]), 0.5 * expit(r.x[1]))
    return best[1], best[2]


def run_one(task):
    f, N, rep, kx, seed = task
    rng = np.random.default_rng(seed)
    q0 = rng.uniform(0, 1)
    qx, qyx = ground_truth(kx, f, q0, rng)
    nx = rng.multinomial(N, qx)
    n1 = rng.binomial(nx, qyx[:, 1])
    n = np.column_stack([nx - n1, n1]).astype(float)
    qy = qhat_y(n)
    f_hat, q0_hat = fit_f_q0(n)
    encs = {
        "marginal": np.tile(qy, (kx, 1)),
        "fixed_m20": fixed_m_encoding(n, 20.0, qy),
        "baseline": fit_baseline(n, qy=qy).target_encoding,
        "mix_K2": fit_finite_mixture(n, K=2, n_init=4, seed=2).target_encoding,
        "npmle": fit_npmle(n).target_encoding,
        "oracle": oracle_encoding(n, f_hat, q0_hat),
    }
    I_true = exact_mutual_information(qx, qyx)
    return [dict(f=f, N=N, rep=rep, q0=q0, I_true=I_true, method=m, eCE=excess_cross_entropy(qx, qyx, e))
            for m, e in encs.items()]


def summarize(df):
    wide = df.pivot_table(index=["f", "N", "rep"], columns="method", values="eCE")
    rows = []
    for (f, N), g in wide.groupby(level=["f", "N"]):
        gain = (g["marginal"] - g["oracle"]).mean()
        for m in g.columns:
            gap = g[m] - g["oracle"]
            rows.append(dict(f=f, N=N, method=m, reps=len(g), eCE=g[m].mean(), gap=gap.mean(),
                             gap_sem=gap.std(ddof=1) / np.sqrt(len(g)), gain=gain,
                             share_of_gain_lost=gap.mean() / gain if gain > 0 else np.nan))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kx", type=int, default=2500)
    ap.add_argument("--fs", type=float, nargs="+", default=[0.1, 0.25, 0.5, 0.75, 0.9])
    ap.add_argument("--Ns", type=int, nargs="+", default=[500, 1000, 2500, 5000, 50000])
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ss = np.random.SeedSequence(a.seed)
    tasks = [(f, N, r, a.kx, int(s.generate_state(1)[0]))
             for (f, N, r), s in zip([(f, N, r) for f in a.fs for N in a.Ns for r in range(a.reps)],
                                     ss.spawn(len(a.fs) * len(a.Ns) * a.reps), strict=False)]
    t0 = time.time()
    rows = []
    with Pool(a.procs) as pool:
        for i, r in enumerate(pool.imap_unordered(run_one, tasks, chunksize=2)):
            rows += r
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{len(tasks)} ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows).sort_values(["f", "N", "rep", "method"])
    df.to_csv(a.out, index=False)
    summarize(df).to_csv(a.out.replace(".csv", "_summary.csv"), index=False)
    print(f"wrote {a.out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
