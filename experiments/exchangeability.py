"""
Exchangeability tests on the real-data oracle pools and on the bakeoff tasks.

    python experiments/exchangeability.py pools --data ml1m_ratings.tsv --feature movie_id --out raw/
    python experiments/exchangeability.py pools --data ml1m_ratings.tsv --feature movie_id --shuffle \
        --out raw/
    python experiments/exchangeability.py tasks --data-root DATA --out raw/
    python experiments/exchangeability.py summary --raw raw/ --out tables/ \
        [--coverage mi_coverage_v3/summary.csv]

Two tests of the null that levels of every size share one prior for ``q``
(Assumption 1 of the paper; what the MI intervals rest on):

- ``stratum``: :func:`mitarget.stratum_moment_test`, a Wald test that the
  unbiased estimates of ``m1`` and ``m2`` agree across strata of level sizes.
- ``lr``: the likelihood-ratio test of the frequency-aware prior
  (:func:`mitarget.fit_freq_prior`), logit mean linear in ``log(1 + n)``
  against a constant mean.

``pools`` draws training sets exactly as ``realdata_oracle.py`` does with the
same ``--seed`` (levels, then labels, from one stream), so the tests are run on
the training sets behind the oracle tables. ``tasks`` runs both tests on the
full data of each bakeoff task (one-vs-rest with Bonferroni for more than two
classes). ``summary`` writes rejection rates per pool and ``nbar`` and, with
``--coverage``, joins the likelihood-interval coverage of ``mi_coverage.py``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from mitarget import fit_freq_prior, stratum_moment_test
from mitarget.core import counts_from_labels

sys.path.insert(0, str(Path(__file__).resolve().parent))
from realdata_oracle import NBAR_GRID, load_population  # noqa: E402

ALPHA = 0.05


def tests(n: np.ndarray) -> dict:
    """Both tests on a binary (kx, 2) count matrix."""
    n = n[n.sum(axis=1) > 0]
    st = stratum_moment_test(n)
    fp = fit_freq_prior(n)
    nx = n.sum(axis=1)
    return dict(p_stratum=st.p_value, stat_stratum=st.statistic, df_stratum=st.df,
                max_dev_m1=st.max_dev_m1, strata=len(st.strata),
                p_lr=fp.p_value, lr=fp.lr_statistic, slope_logit_mean=float(fp.params[1]),
                F1_2F2_share=float(((nx == 1).sum() + 2 * (nx == 2).sum()) / nx.sum()))


def cmd_pools(a):
    name = a.name or a.feature
    p, q, _ = load_population(a.data, a.feature, a.min_rows)
    if a.shuffle:
        q = np.random.default_rng(a.seed + 1).permutation(q)
    L = len(p)
    rng = np.random.default_rng(a.seed)
    out = []
    for nbar in a.nbar:
        N = int(round(nbar * L))
        t0 = time.time()
        for rep in range(a.reps):
            level = rng.choice(L, size=N, p=p)          # the same draws as realdata_oracle.run_cell
            y = rng.random(N) < q[level]
            n1 = np.bincount(level, weights=y, minlength=L)
            nx = np.bincount(level, minlength=L).astype(float)
            out.append(dict(feature=name, shuffled=a.shuffle, nbar=nbar, rep=rep, N=N,
                            **tests(np.column_stack([nx - n1, n1]))))
        print(f"{name}{' (shuffled)' if a.shuffle else ''} nbar={nbar}: {time.time() - t0:.1f}s", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    tag = "shuffled" if a.shuffle else "real"
    pd.DataFrame(out).to_csv(a.out / f"exch_pool_{name}_{tag}.csv", index=False)


def cmd_tasks(a):
    from realdata_tasks import GROUPS
    rows = []
    for group in a.groups:
        loader, sub = GROUPS[group]
        for task in loader(a.data_root / sub):
            t0 = time.time()
            n = counts_from_labels(task.x, task.y, task.kx, task.ky)
            if task.ky == 2:
                r = tests(n)
            else:   # one-vs-rest, Bonferroni over classes
                per = [tests(np.column_stack([n.sum(axis=1) - n[:, c], n[:, c]])) for c in range(task.ky)]
                r = dict(per[0])
                for key in ("p_stratum", "p_lr"):
                    r[key] = min(1.0, task.ky * min(d[key] for d in per))
                r["max_dev_m1"] = max(d["max_dev_m1"] for d in per)
            rows.append(dict(task=task.label, N=task.n, kx=task.kx, ky=task.ky, **r))
            print(f"{task.label}: p_stratum={r['p_stratum']:.3g} p_lr={r['p_lr']:.3g} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out / "exch_tasks.csv", index=False)


def cmd_summary(a):
    d = pd.concat([pd.read_csv(f) for f in sorted(a.raw.glob("exch_pool_*.csv"))], ignore_index=True)
    d = d.assign(reject_stratum=d.p_stratum < ALPHA, reject_lr=d.p_lr < ALPHA,
                 reject_either=(d.p_stratum < ALPHA) | (d.p_lr < ALPHA))
    g = d.groupby(["feature", "shuffled", "nbar"])
    s = g[["reject_stratum", "reject_lr", "reject_either"]].mean()
    s["reps"] = g.size()
    s["max_dev_m1"] = g.max_dev_m1.median()
    s = s.reset_index()
    if a.coverage is not None:
        c = pd.read_csv(a.coverage)
        c = c[c.prior == "real"][["pool", "shuffled", "nbar", "cov_lik", "width_lik", "cov_cred"]]
        s = s.merge(c.rename(columns={"pool": "feature"}), on=["feature", "shuffled", "nbar"], how="left")
    a.out.mkdir(parents=True, exist_ok=True)
    s.to_csv(a.out / "exch_pools_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(s.pivot_table(index=["feature", "shuffled"], columns="nbar", values="reject_either").round(2)
          .to_string())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pools")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--feature", required=True)
    p.add_argument("--name", default=None)
    p.add_argument("--min-rows", type=int, default=100)
    p.add_argument("--reps", type=int, default=50)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--nbar", type=float, nargs="*", default=NBAR_GRID)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    t = sub.add_parser("tasks")
    t.add_argument("--data-root", type=Path, required=True)
    t.add_argument("--groups", nargs="+", default=None)
    t.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("summary")
    s.add_argument("--raw", type=Path, required=True)
    s.add_argument("--coverage", type=Path, default=None, help="summary.csv of mi_coverage.py (optional)")
    s.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "tasks" and a.groups is None:
        from realdata_tasks import GROUPS
        a.groups = list(GROUPS)
    {"pools": cmd_pools, "tasks": cmd_tasks, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    main()
