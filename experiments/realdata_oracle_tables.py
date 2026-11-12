"""
Summaries of the real-data subsampling oracle (``realdata_oracle.py``).

    python experiments/realdata_oracle_tables.py --raw raw/ --out tables/

Writes:

- ``summary.csv``: per (pool, shuffled, nbar, method) mean eCE, gap to the full
  oracle, its standard error over training sets, and the share of the
  oracle's gain lost, ``gap / (I(X;Y) - eCE_oracle)``.
- ``paired.csv``: per (pool, shuffled, nbar, method) the paired difference to
  the baseline over the same training sets, ``eCE_method - eCE_baseline``: mean,
  standard error, t, two-sided p (training sets are independent draws, so a
  one-sample t-test on the differences is exact up to normality), the share
  of the baseline's gap the method closes and its standard error, and the
  share of training sets on which it beats the baseline.
- ``baseline_gap_by_n.csv``: the baseline's gap split by the training count of
  the test row's level, the share at ``n_x <= 1``, the ``(F1 + 2 F2) / N``
  diagnostic next to the true mass, and the gap closed by each method.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

KEYS = ["feature", "shuffled", "nbar"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)

    d = pd.concat([pd.read_csv(f) for f in sorted(a.raw.glob("oracle_*.csv"))], ignore_index=True)
    # runs with --suffix that add methods repeat the references on the same training sets: keep one copy
    d = d.drop_duplicates(KEYS + ["rep", "method"], keep="first").reset_index(drop=True)
    bcols = [c for c in d.columns if c.startswith("eCE_n")]
    orc = d[d.method == "oracle"].set_index(KEYS + ["rep"])[["eCE"] + bcols].add_suffix("_orc")
    d = d.join(orc, on=KEYS + ["rep"])
    d["gap"] = d.eCE - d.eCE_orc
    d["achievable"] = d.I_XY - d.eCE_orc
    for c in bcols:
        d["gap_" + c[4:]] = d[c] - d[c + "_orc"]

    g = d.groupby(KEYS + ["method"])
    summ = pd.DataFrame({
        "N": g.N.first(), "reps": g.rep.nunique(), "eCE": g.eCE.mean(), "gap_to_oracle": g.gap.mean(),
        "gap_sem": g.gap.std() / np.sqrt(g.gap.size()),
        "share_of_gain_lost": g.gap.mean() / g.achievable.mean(), "I_XY": g.I_XY.first(),
        "corr_logp_q": g.corr_logp_q.first(),
    }).reset_index()
    summ.to_csv(a.out / "summary.csv", index=False)

    base = d[d.method == "baseline"].set_index(KEYS + ["rep"])
    rows = []
    for (feat, sh, nbar, m), t in d[~d.method.isin(["baseline", "oracle", "marginal"])].groupby(
            KEYS + ["method"]):
        t = t.set_index(KEYS + ["rep"])
        b = base.loc[t.index]
        diff = (t.eCE - b.eCE).to_numpy()
        gap_b = b.gap.to_numpy()
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        tstat = diff.mean() / se if se > 0 else 0.0
        # share of the baseline's gap closed (ratio of means), delta-method SE
        closed = -diff.mean() / gap_b.mean()
        cov = np.cov(-diff, gap_b, ddof=1) / len(diff)
        r = closed
        se_closed = np.sqrt(max(cov[0, 0] - 2 * r * cov[0, 1] + r ** 2 * cov[1, 1], 0.0)) / abs(gap_b.mean())
        rows.append(dict(feature=feat, shuffled=sh, nbar=nbar, method=m, reps=len(diff),
                         diff_vs_baseline=diff.mean(), diff_sem=se, t=tstat,
                         p=2 * stats.t.sf(abs(tstat), len(diff) - 1) if se > 0 else 1.0,
                         gap_closed=closed, gap_closed_sem=se_closed, win_rate=float((diff < 0).mean())))
    paired = pd.DataFrame(rows)
    paired.to_csv(a.out / "paired.csv", index=False)

    b = d[d.method == "baseline"]
    gb = b.groupby(KEYS)
    gapcols = ["gap_" + c[4:] for c in bcols]
    by_n = gb[["gap"] + gapcols].mean()
    by_n["share_gap_at_n<=1"] = (by_n["gap_n0"] + by_n["gap_n1"]) / by_n["gap"]
    by_n = by_n.join(gb[["GT_w01", "true_w01", "I_XY", "corr_logp_q"]].mean())
    closed = paired.pivot_table(index=KEYS, columns="method", values="gap_closed").add_prefix("gap_closed_")
    by_n = by_n.join(closed).reset_index()
    by_n.to_csv(a.out / "baseline_gap_by_n.csv", index=False)

    pd.set_option("display.width", 250)
    show = ["fixed_m20", "cv_m", "baseline", "baseline_unseen", "baseline_robbins", "freq_prior", "npmle",
            "npmle_robbins", "glmm", "beta_mm",
            "oracle_blind"]
    piv = summ[summ.nbar == 2].pivot_table(index=["feature", "shuffled"], columns="method",
                                           values="share_of_gain_lost")
    print("share of the oracle's gain lost at nbar = 2")
    print(piv[[m for m in show if m in piv]].round(3).to_string())
    pc = paired[paired.nbar == 2].pivot_table(index=["feature", "shuffled"], columns="method",
                                              values="gap_closed")
    print("\nshare of the baseline's gap closed at nbar = 2")
    print(pc.round(2).to_string())


if __name__ == "__main__":
    main()
