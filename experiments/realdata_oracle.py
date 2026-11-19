"""
Real-data subsampling oracle (Section 7.2): populations with known q_x built from real data.

    python experiments/realdata_oracle.py --data ml1m_ratings.tsv --feature movie_id --reps 40 --out raw/
    python experiments/realdata_oracle.py --data ml1m_ratings.tsv --feature movie_id --shuffle --out raw/

Population. Rows of a tab-separated file with a categorical column ``--feature``
and a ``rating`` column, ``y = 1[rating >= 4]``. Only levels with at least
``--min-rows`` rows are kept. The population is the empirical distribution of
those rows: level j is drawn with probability ``p_j = M_j / M`` and
``y | j ~ Bern(q_j)``, with ``q_j`` the level's rate over all its rows. Training
sets of ``N = nbar * L`` rows are drawn i.i.d., so the excess cross-entropy
``sum_j p_j KL(Bern(q_j) || Bern(r_j))`` is exact, with no test noise.

Methods, all fit on the same training set (so every comparison is paired):
marginal, fixed m = 20, m chosen by leave-one-out cross-validated log-loss
(``cv_m``), the baseline, the baseline with the unseen correction,
the baseline with the Robbins correction for n <= 1 (``baseline_robbins``, with
its doubleton test; ``baseline_robbins_raw`` without),
the frequency prior (``freq_prior``: n + 1 shift, LR fallback), NPMLE,
NPMLE with the same Robbins correction (``npmle_robbins``), GLMM;
and three references that use the truth: the beta moment-matched to the q of
a random row (``beta_mm``), the popularity-blind oracle (Bayes with the true
law of q over the pool, ignoring ``n_x``) and the full oracle (Bayes with the
true joint law of ``(p_j, q_j)``: posterior over pool levels proportional to
``p_j Poi(n; N p_j) q_j^k (1 - q_j)^(n - k)``).

``--tail shifted-head`` keeps every level of the feature instead (the click
features' popularity dependence sits on levels with one to five rows, which
the ``--min-rows`` pool drops). Levels with at least ``--min-rows`` rows keep
their observed rate. A level with fewer rows gets a known ``q_j`` drawn from
the head's law of ``q`` (levels weighted equally), shifted on the logit scale
so that its mean equals the observed mean rate of the levels in the same size
bucket (1, 2, 3-5, 6-10, 11-20, 21-50, 51 to min-rows - 1). The tail's shape
is not identified from the data (Section 6), so this is one model of it; what
it keeps is the observed size histogram and the observed step of the label
rate with size.

``--shuffle`` permutes ``q_j`` across levels (control without popularity
dependence). One output row per (nbar, rep, method) with the eCE, its split by
the training count of the test row's level, and the (F1 + 2 F2) / N diagnostic
next to the true mass of levels seen at most once.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logsumexp

from mitarget.encoders import fit_counts

NBAR_GRID = (0.25, 0.5, 1, 2, 3, 5, 10, 20)
N_BUCKETS = ((0, 0), (1, 1), (2, 2), (3, 5), (6, 10), (11, 10**9))
EPS = 1e-12

#: name -> function of the (L, 2) training count matrix returning P(y = 1 | level)
FITTED = {
    "fixed_m20": lambda n: fit_counts(n, "fixed_m", m=20.0).encoding[:, 1],
    "cv_m": lambda n: fit_counts(n, "cv_m").encoding[:, 1],
    "baseline": lambda n: fit_counts(n, "baseline").encoding[:, 1],
    "baseline_unseen": lambda n: fit_counts(n, "baseline_unseen").encoding[:, 1],
    "baseline_robbins": lambda n: fit_counts(n, "baseline_robbins").encoding[:, 1],
    "baseline_robbins_raw": lambda n: fit_counts(n, "baseline_robbins_raw").encoding[:, 1],
    "freq_prior": lambda n: fit_counts(n, "freq_prior").encoding[:, 1],
    "npmle": lambda n: fit_counts(n, "npmle").encoding[:, 1],
    "npmle_robbins": lambda n: fit_counts(n, "npmle_robbins").encoding[:, 1],
    "glmm": lambda n: fit_counts(n, "glmm").encoding[:, 1],
    "glmm_mean": lambda n: fit_counts(n, "glmm_mean").encoding[:, 1],
}


def kl_bern(q, r):
    q = np.clip(q, EPS, 1 - EPS)
    r = np.clip(r, EPS, 1 - EPS)
    return q * np.log(q / r) + (1 - q) * np.log((1 - q) / (1 - r))


TAIL_BUCKETS = (1, 2, 3, 6, 11, 21, 51)


def load_population(path, feature, min_rows, tail=None, seed=0):
    d = pd.read_csv(path, sep="\t", usecols=[feature, "rating"])
    g = d.assign(y=(d.rating >= 4).astype(float)).groupby(feature).y.agg(["size", "mean"])
    if tail is None:
        g = g[g["size"] >= min_rows]
        M = g["size"].to_numpy(float)
        return M / M.sum(), g["mean"].to_numpy(), int(M.sum())
    M = g["size"].to_numpy(float)
    q = g["mean"].to_numpy().copy()
    head = M >= min_rows
    qh = np.clip(q[head], 0.5 / M[head], 1 - 0.5 / M[head])
    lh = np.log(qh / (1 - qh))
    rng = np.random.default_rng(seed)
    edges = [b for b in TAIL_BUCKETS if b < min_rows] + [min_rows]
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (M >= lo) & (M < hi)
        if not m.any():
            continue
        target = q[m].mean()
        c = brentq(lambda c, t=target: expit(lh + c).mean() - t, -20, 20)
        q[m] = expit(lh[rng.integers(len(lh), size=m.sum())] + c)
    return M / M.sum(), q, int(M.sum())


def bayes_predict(n1, n0, q, log_prior):
    """Posterior mean of q given counts, with atoms at the pool's q and log prior weights (rows, atoms)."""
    lq, l1q = np.log(np.clip(q, EPS, 1)), np.log(np.clip(1 - q, EPS, 1))
    ll = log_prior + n1[:, None] * lq[None, :] + n0[:, None] * l1q[None, :]
    post = np.exp(ll - logsumexp(ll, axis=1, keepdims=True))
    return post @ q


def run_cell(p, q, N, rng, fitted=FITTED):
    L = len(p)
    level = rng.choice(L, size=N, p=p)
    y = rng.random(N) < q[level]
    n1 = np.bincount(level, weights=y, minlength=L)
    nx = np.bincount(level, minlength=L).astype(float)
    n0 = nx - n1
    counts = np.column_stack([n0, n1])

    preds, seconds = {"marginal": np.full(L, n1.sum() / N)}, {}
    for name, fn in fitted.items():
        t0 = time.perf_counter()
        preds[name] = fn(counts)
        seconds[name] = time.perf_counter() - t0

    mu = (p * q).sum()
    s_mm = mu * (1 - mu) / (p * (q - mu) ** 2).sum() - 1
    preds["beta_mm"] = (n1 + mu * s_mm) / (nx + s_mm)
    # Both oracles are posteriors over the pool's levels; levels sharing (p_j, q_j) are one atom.
    pairs, inv = np.unique(np.column_stack([n1, n0]), axis=0, return_inverse=True)
    inv = inv.ravel()
    atoms, mult = np.unique(np.column_stack([p, q]), axis=0, return_counts=True)
    pa, qa, lm = atoms[:, 0], atoms[:, 1], np.log(mult)
    # popularity-blind: law of q over levels, weighted by p as in the paper's definition
    blind_w = pd.Series(pa * mult).groupby(qa).sum()
    qb, lb = blind_w.index.to_numpy(), np.log(blind_w.to_numpy())
    preds["oracle_blind"] = bayes_predict(pairs[:, 0], pairs[:, 1], qb, lb[None, :])[inv]
    lam = N * pa
    lp = (np.log(pa) + lm)[None, :] + pairs.sum(1)[:, None] * np.log(lam)[None, :] - lam[None, :]
    preds["oracle"] = bayes_predict(pairs[:, 0], pairs[:, 1], qa, lp)[inv]

    F1, F2 = int((nx == 1).sum()), int((nx == 2).sum())
    common = dict(N=N, GT_w01=(F1 + 2 * F2) / N, true_w01=float(p[nx <= 1].sum()))
    rows = []
    for name, r in preds.items():
        per = p * kl_bern(q, r)
        row = dict(method=name, eCE=per.sum(), seconds=seconds.get(name, np.nan))
        for lo, hi in N_BUCKETS:
            m = (nx >= lo) & (nx <= hi)
            row[f"eCE_n{lo}" if lo == hi else f"eCE_n{lo}_{hi}"] = per[m].sum()
        rows.append({**common, **row})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--feature", required=True)
    ap.add_argument("--name", default=None, help="pool name in the output (default: the feature)")
    ap.add_argument("--min-rows", type=int, default=100)
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--nbar", type=float, nargs="*", default=NBAR_GRID)
    ap.add_argument("--shuffle", action="store_true", help="permute q_j across levels")
    ap.add_argument("--methods", nargs="*", default=list(FITTED), choices=list(FITTED),
                    help="fitted methods to run (training sets do not depend on this choice)")
    ap.add_argument("--suffix", default="", help="appended to the output file name")
    ap.add_argument("--tail", choices=["shifted-head"], default=None,
                    help="keep all levels; levels below --min-rows get a modelled q (see the docstring)")
    ap.add_argument("--out", type=Path, required=True, help="output folder")
    a = ap.parse_args(argv)
    fitted = {m: FITTED[m] for m in a.methods}

    name = a.name or a.feature
    p, q, M = load_population(a.data, a.feature, a.min_rows, tail=a.tail, seed=a.seed + 2)
    corr = float(np.corrcoef(np.log(p), q)[0, 1])
    if a.shuffle:
        q = np.random.default_rng(a.seed + 1).permutation(q)
    L = len(p)
    I_pop = float((p * kl_bern(q, (p * q).sum())).sum())
    print(f"{name}{' (shuffled)' if a.shuffle else ''}: L={L} levels, M={M} rows, I(X;Y)={I_pop:.4f} nats, "
          f"corr(log p, q)={corr:+.2f}", flush=True)

    a.out.mkdir(parents=True, exist_ok=True)
    path = a.out / f"oracle_{name}_{'shuffled' if a.shuffle else 'real'}{a.suffix}.csv"
    rng = np.random.default_rng(a.seed)
    out = []
    for nbar in a.nbar:
        N = int(round(nbar * L))
        t0 = time.time()
        for rep in range(a.reps):
            for r in run_cell(p, q, N, rng, fitted):
                out.append(dict(feature=name, shuffled=a.shuffle, L=L, I_XY=I_pop, corr_logp_q=corr,
                                nbar=nbar, rep=rep, **r))
        print(f"  nbar={nbar:5.2f} N={N:6d} done in {time.time() - t0:.1f}s", flush=True)
        pd.DataFrame(out).to_csv(path, index=False)


if __name__ == "__main__":
    main()
