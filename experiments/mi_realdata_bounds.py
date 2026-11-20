"""
Statements about I(X;Y) on the full real features that do not assume exchangeability across level sizes.

For every binary bakeoff task this writes one row with

- ``pred_lb``: the predictive lower bound of Remark ``rem:mi-lower``, ``H(Y) - CE_test`` for the best held-out
  encoder of the bakeoff (any encoder's held-out cross-entropy is at least ``H(Y|X)``, with no assumption);
- ``strat_lo``, ``strat_hi``: the stratified identified interval of Section ``sec:mi-pop`` at the plug-in
  moments. Levels are grouped by training size (1, 2, 3, 4, 5-6, 7-10, 11-20, 21-50, 51-100, >100); each
  group gets its own prior, identified through its first ``min(smallest size, MAX_L)`` moments, and enters
  ``H(Y|X)`` with its share of the rows. Exchangeability is assumed only within a group, so the singletons'
  term is the trivial ``[0, h(mu_1)]`` of Proposition ``prop:nonid``;
- ``single_w``: that term's width, the singletons' row share times ``h`` of their label rate;
- the baseline, NPMLE and plug-in point estimates, and the likelihood interval under exchangeability.

    python experiments/mi_realdata_bounds.py --data-root DATA \
        --logloss bake_tables/table_logloss.csv --out bounds.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import comb

from mitarget import fit_counts, identified_interval, mi_interval
from mitarget.core import counts_from_labels

sys.path.insert(0, str(Path(__file__).resolve().parent))
from realdata_tables import SHORT  # noqa: E402
from realdata_tasks import GROUPS  # noqa: E402

BINS = ((1, 1), (2, 2), (3, 3), (4, 4), (5, 6), (7, 10), (11, 20), (21, 50), (51, 100), (101, np.inf))
MAX_L = 4


def h(q):
    q = np.clip(q, 1e-15, 1 - 1e-15)
    return float(-q * np.log(q) - (1 - q) * np.log(1 - q))


def eh_range(nx, kx, L):
    """Range of E h(q) over priors whose first L moments equal the U-statistics of these levels."""
    for ell in range(L, 0, -1):
        m = np.array([np.mean(comb(kx, j) / comb(nx, j)) for j in range(1, ell + 1)])
        if ell == 1:
            return 0.0, h(m[0]), 1
        try:
            lo, hi = identified_interval(m)
        except Exception:  # noqa: BLE001 - estimates outside the moment space: use fewer moments
            continue
        if np.isfinite(lo) and np.isfinite(hi) and hi >= lo - 1e-9:
            return h(m[0]) - hi, h(m[0]) - lo, ell
    return 0.0, h(np.mean(kx / nx)), 1


def task_bounds(task, best_ce, lik=True, max_l=MAX_L):
    n = counts_from_labels(task.x, task.y, task.kx, 2).astype(float)
    nx, kx = n.sum(1), n[:, 1]
    N = nx.sum()
    mu = kx.sum() / N
    row = dict(task=SHORT.get(task.label, task.label), N=int(N), H_Y=h(mu),
               F1_share=float((nx == 1).sum() / N))
    lo_eh = hi_eh = 0.0
    for a, b in BINS:
        s = (nx >= a) & (nx <= b)
        if not s.any():
            continue
        w = nx[s].sum() / N
        elo, ehi, ell = eh_range(nx[s], kx[s], min(a, max_l))
        lo_eh += w * elo
        hi_eh += w * ehi
        if a == 1:
            row["single_w"] = w * (ehi - elo)
    row["strat_lo"], row["strat_hi"] = h(mu) - hi_eh, h(mu) - lo_eh
    row["pred_lb"] = h(mu) - best_ce if best_ce is not None else np.nan
    for meth in ("baseline", "npmle", "plugin"):
        row[f"I_{meth}"] = float(fit_counts(n, meth).I_hat)
    if lik:
        t0 = time.time()
        try:
            r = mi_interval(n, region="likelihood")
            row["lik_lo"], row["lik_hi"] = r.lower, r.upper
        except Exception as e:  # noqa: BLE001
            print(f"   likelihood interval failed: {e}", flush=True)
        row["lik_seconds"] = time.time() - t0
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--groups", nargs="+", default=list(GROUPS), choices=list(GROUPS))
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--logloss", type=Path, required=True, help="table_logloss.csv of realdata_tables.py")
    ap.add_argument("--no-lik", action="store_true")
    ap.add_argument("--max-l", type=int, default=MAX_L, help="moments per size group")
    ap.add_argument("--lik-max-rows", type=int, default=500_000,
                    help="likelihood interval only below this size")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    ll = pd.read_csv(a.logloss)
    ll = ll[~ll.method.isin(["Prior only"])]
    best = ll.groupby("task").logloss.min()
    rows = []
    for group in a.groups:
        loader, _ = GROUPS[group]
        for task in loader(a.data_root / GROUPS[group][1]):
            if task.ky != 2 or (a.tasks and task.label not in a.tasks):
                continue
            t0 = time.time()
            label = SHORT.get(task.label, task.label)
            lik = not a.no_lik and task.n <= a.lik_max_rows
            rows.append(task_bounds(task, best.get(label), lik=lik, max_l=a.max_l))
            print(f"{task.label}: {time.time() - t0:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
