"""
Coverage of the intervals for I(X;Y) (Section 7.4): the moment interval of Section 6
(``mitarget.mi_interval``) and the credible interval of Hernandez & Samengo (2019)
(``mitarget.credible_interval``), with the bias and RMSE of the point estimates.

    python experiments/mi_coverage.py synthetic --reps 100 --jobs 4 --out raw/
    python experiments/mi_coverage.py oracle --data ml1m_ratings.tsv --feature movie_id --reps 50 --out raw/
    python experiments/mi_coverage.py oracle --data ml1m_ratings.tsv --feature movie_id --shuffle --out raw/

Populations.

- ``synthetic``: ``K`` levels with ``q_x`` i.i.d. from one of the priors of Table ``tab:id``
  (redrawn for every training set) and uniform or lognormal (sigma = 1.5) popularity.
- ``oracle``: the subsampling-oracle pools of ``realdata_oracle.py``: the levels of a real
  feature with at least ``--min-rows`` rows, ``p_j`` their share of the rows and ``q_j`` their
  rate. ``--shuffle`` permutes ``q_j`` across levels, which makes Assumption 1 hold.

Each training set has ``N = nbar * K`` rows drawn i.i.d. from the population. The target is
the population value ``I = h(sum_j p_j q_j) - sum_j p_j h(q_j)``. The credible interval is
built for the count-weighted value ``h(mu) - sum_x (n_x / N) h(q_x)`` over the seen levels,
whose mean is ``I`` (Lemma ``lem:rep``); its coverage of that value is reported too.

One output row per training set: the intervals (95%), the point estimates (plug-in, baseline,
credible mean, NPMLE, frequency prior, moment midpoint) and the level-size profile.
``mi_coverage_tables.py`` summarises them. The moment interval uses the largest
``L <= --max-L`` with at least 100 levels seen ``L`` times, and is computed twice: with the
default joint region of ``mi_interval`` (a Sidak box in the principal axes of the moment
covariance; ``mom_*``) and with the per-moment box of ``--slack`` standard errors
(``box_*``). ``id_*`` is the identified interval at the same ``L`` from the population's own
row-weighted moments ``sum_j p_j q_j^l``: the width no amount of data on levels seen at most
``L`` times can remove, which always contains ``I``. ``lik_*`` is the likelihood-region interval
(``mi_interval(region="likelihood")``): every prior whose likelihood for the table of
``(n_x, k_x)`` is within ``chi2_1(0.95) / 2`` of the NPMLE.

``--lik-only`` redraws the same training sets (same seeds) and computes only the likelihood
interval, to add it to an existing run; it writes ``lik_coverage_*.csv`` next to the
``mi_coverage_*.csv`` files, and ``mi_coverage_tables.py`` merges them by data set.
``--baselines-only`` does the same for the classical estimators of ``mitarget.mi_baselines``
(Miller-Madow, Chao-Shen, NSB with the true number of levels) and writes ``base_coverage_*.csv``.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from mitarget import credible_interval, fit_counts, identified_interval, mi_interval, plugin_information
from mitarget.mi_baselines import mi_chao_shen, mi_miller_madow, mi_nsb
from mitarget.mi_interval import binary_entropy

NBAR_SYNTH = (0.5, 1, 2, 5)
NBAR_ORACLE = (0.5, 1, 2, 5, 10, 20)

#: name -> (sampler of K conditionals, atoms and weights or beta parameters for the exact moments)
PRIORS = {
    "junk_f0.5": ("atoms", ([0.1, 0.5, 0.9], [0.25, 0.5, 0.25])),
    "junk_f0.8": ("atoms", ([0.1, 0.5, 0.9], [0.1, 0.8, 0.1])),
    "atoms3": ("atoms", ([0.2, 0.3, 0.85], [0.3, 0.4, 0.3])),
    # the same prior with atoms off the 101-point grid of the likelihood interval (sensitivity check)
    "atoms3_off": ("atoms", ([0.2037, 0.3061, 0.8529], [0.3, 0.4, 0.3])),
    "beta_0.5_0.5": ("beta", (0.5, 0.5)),
    "beta_1_3": ("beta", (1.0, 3.0)),
}
#: (K, popularity)
SYNTH_POOLS = ((10_000, "uniform"), (100_000, "uniform"), (100_000, "lognormal"))


def draw_q(prior, K, rng):
    kind, par = PRIORS[prior]
    if kind == "beta":
        return rng.beta(*par, size=K)
    atoms, w = par
    return np.asarray(atoms)[rng.choice(len(atoms), size=K, p=w)]


def draw_p(popularity, K, rng):
    if popularity == "uniform":
        return np.full(K, 1.0 / K)
    w = np.exp(rng.normal(0.0, 1.5, K))
    return w / w.sum()


#: grid size and thinning cap of the likelihood interval (``--lik-grid``, ``--n-cap``)
LIK_OPTS = {}


def likelihood_columns(n):
    t0 = time.perf_counter()
    try:
        lk = mi_interval(n, region="likelihood", **LIK_OPTS)
        row = dict(lik_lo=lk.lower, lik_hi=lk.upper)
    except ValueError:
        row = dict(lik_lo=np.nan, lik_hi=np.nan)
    row["lik_seconds"] = time.perf_counter() - t0
    return row


def evaluate(p, q, N, rng, slack, max_L, lik_only=False):
    """``lik_only``: False (everything), True (likelihood interval only) or "baselines"."""
    """Draw one training set of N rows from the population (p, q) and fit everything."""
    K = len(p)
    level = rng.choice(K, size=N, p=p)
    y = rng.random(N) < q[level]
    k = np.bincount(level, weights=y, minlength=K)
    nx = np.bincount(level, minlength=K).astype(float)
    n = np.column_stack([nx - k, k])

    mu = float(p @ q)
    I_true = float(binary_entropy(mu) - p @ binary_entropy(q))
    I_count = float(binary_entropy(mu) - (nx / N) @ binary_entropy(q))
    row = dict(N=N, I_true=I_true, I_count=I_count, F1=int((nx == 1).sum()), F2=int((nx == 2).sum()),
               K2=int((nx >= 2).sum()), K3=int((nx >= 3).sum()), K4=int((nx >= 4).sum()),
               nmax=int(nx.max()))
    if lik_only == "baselines":
        row.update(I_mm=mi_miller_madow(n), I_cs=mi_chao_shen(n), I_nsb=mi_nsb(n, K))
        return row
    row.update(likelihood_columns(n))
    if lik_only:
        return row

    t0 = time.perf_counter()
    try:
        mi = mi_interval(n, max_L=max_L)
        row.update(mom_L=mi.L, mom_lo=mi.lower, mom_hi=mi.upper)
        row["id_lo"], row["id_hi"] = identified_interval([p @ q ** j for j in range(1, mi.L + 1)])
    except ValueError:
        row.update(mom_L=np.nan, mom_lo=np.nan, mom_hi=np.nan, id_lo=np.nan, id_hi=np.nan)
    row["mom_seconds"] = time.perf_counter() - t0
    try:
        bx = mi_interval(n, slack=slack, max_L=max_L, region="box")
        row.update(box_lo=bx.lower, box_hi=bx.upper)
    except ValueError:
        row.update(box_lo=np.nan, box_hi=np.nan)
    cr = credible_interval(n)
    row.update(cred_mean=cr.mean, cred_sd=cr.sd, cred_lo=cr.lower, cred_hi=cr.upper,
               cred_sd_beta=cr.sd_beta, beta_star=cr.beta_star)
    row.update(I_plugin=plugin_information(n), I_baseline=fit_counts(n, "baseline").I_hat,
               I_npmle=fit_counts(n, "npmle").I_hat, I_freq=fit_counts(n, "freq_prior").I_hat)
    return row


def _synthetic_task(args):
    prior, K, popularity, nbar, rep, seed, slack, max_L, lik_only = args
    rng = np.random.default_rng([seed, rep, int(nbar * 100), K, list(PRIORS).index(prior),
                                 int(popularity == "lognormal")])
    q, p = draw_q(prior, K, rng), draw_p(popularity, K, rng)
    row = evaluate(p, q, int(round(nbar * K)), rng, slack, max_L, lik_only)
    return dict(pool=f"{prior}|K={K}|{popularity}", prior=prior, K=K, popularity=popularity,
                shuffled=False, nbar=nbar, rep=rep, **row)


def _oracle_task(args):
    name, shuffled, p, q, nbar, rep, seed, slack, max_L, lik_only = args
    rng = np.random.default_rng([seed, rep, int(nbar * 100)])
    row = evaluate(p, q, int(round(nbar * len(p))), rng, slack, max_L, lik_only)
    return dict(pool=name, prior="real", K=len(p), popularity="real", shuffled=shuffled, nbar=nbar,
                rep=rep, **row)


def load_population(path, feature, min_rows):
    d = pd.read_csv(path, sep="\t", usecols=[feature, "rating"])
    g = d.assign(y=(d.rating >= 4).astype(float)).groupby(feature).y.agg(["size", "mean"])
    g = g[g["size"] >= min_rows]
    M = g["size"].to_numpy(float)
    return M / M.sum(), g["mean"].to_numpy()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["synthetic", "oracle"])
    ap.add_argument("--data", type=Path, help="oracle: tab-separated file with the feature and `rating`")
    ap.add_argument("--feature", help="oracle: the categorical column")
    ap.add_argument("--name", default=None, help="oracle: pool name in the output (default: the feature)")
    ap.add_argument("--min-rows", type=int, default=100)
    ap.add_argument("--shuffle", action="store_true", help="oracle: permute q_j across levels")
    ap.add_argument("--priors", nargs="*", default=list(PRIORS), help="synthetic: priors to run")
    ap.add_argument("--nbar", type=float, nargs="*", default=None)
    ap.add_argument("--pools", nargs="*", default=None,
                    help="synthetic: pools to run as K|popularity, e.g. 100000|uniform (default: all)")
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--slack", type=float, default=2.0, help="half-width of the per-moment box, in SEs")
    ap.add_argument("--max-L", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--lik-only", action="store_true",
                    help="only the likelihood interval, on the same data sets (lik_coverage_*.csv)")
    ap.add_argument("--baselines-only", action="store_true",
                    help="redraw the same data sets and compute only the classical MI estimators")
    ap.add_argument("--lik-grid", type=int, default=101, help="grid size of the likelihood interval")
    ap.add_argument("--n-cap", type=int, default=200, help="levels above this size are thinned to it")
    ap.add_argument("--tag", default=None, help="suffix of the output file (default: none)")
    ap.add_argument("--out", type=Path, required=True, help="output folder")
    a = ap.parse_args(argv)
    LIK_OPTS.update(lik_grid_size=a.lik_grid, n_cap=a.n_cap)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.baselines_only:
        a.lik_only = "baselines"

    if a.mode == "synthetic":
        nbars = a.nbar or NBAR_SYNTH
        tasks = [(prior, K, pop, nbar, rep, a.seed, a.slack, a.max_L, a.lik_only) for prior in a.priors
                 for K, pop in SYNTH_POOLS if a.pools is None or f"{K}|{pop}" in a.pools
                 for nbar in nbars for rep in range(a.reps)]
        fn, path = _synthetic_task, a.out / f"mi_coverage_synthetic{'_' + a.tag if a.tag else ''}.csv"
    else:
        name = a.name or a.feature
        p, q = load_population(a.data, a.feature, a.min_rows)
        if a.shuffle:
            q = np.random.default_rng(a.seed + 1).permutation(q)
        nbars = a.nbar or NBAR_ORACLE
        tasks = [(name, a.shuffle, p, q, nbar, rep, a.seed, a.slack, a.max_L, a.lik_only) for nbar in nbars
                 for rep in range(a.reps)]
        fn = _oracle_task
        kind = "shuffled" if a.shuffle else "real"
        suffix = "_" + a.tag if a.tag else ""
        path = a.out / f"mi_coverage_{name}_{kind}{suffix}.csv"
        print(f"{name}{' (shuffled)' if a.shuffle else ''}: K={len(p)} levels", flush=True)
    if a.lik_only:
        prefix = "base_coverage_" if a.lik_only == "baselines" else "lik_coverage_"
        path = path.with_name(path.name.replace("mi_coverage_", prefix))

    t0, rows = time.time(), []
    with ProcessPoolExecutor(a.jobs) as ex:
        for i, row in enumerate(ex.map(fn, tasks, chunksize=4)):
            rows.append(row)
            if (i + 1) % 200 == 0 or i + 1 == len(tasks):
                print(f"  {i + 1}/{len(tasks)} fits, {time.time() - t0:.0f}s", flush=True)
                pd.DataFrame(rows).to_csv(path, index=False)


if __name__ == "__main__":
    main()
