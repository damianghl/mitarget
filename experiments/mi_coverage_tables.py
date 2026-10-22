"""
Summary of ``mi_coverage.py``: coverage and width of the intervals, bias and RMSE of the
point estimates, per (pool, shuffled, nbar).

    python experiments/mi_coverage_tables.py --raw raw/ --out tables/

``summary.csv`` columns: ``cov_mom`` (moment interval covers the population ``I``),
``cov_cred`` (credible interval covers ``I``), ``cov_cred_count`` (credible interval covers
the count-weighted value it is built for), ``width_*`` (mean width), ``L_med`` (median ``L``
of the moment interval), and ``bias_*`` / ``rmse_*`` of each point estimate against ``I``.
``cov_se`` is the binomial standard error of a coverage near 0.95.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ESTIMATES = {"plugin": "I_plugin", "baseline": "I_baseline", "npmle": "I_npmle", "freq": "I_freq",
             "mom_mid": "mom_mid", "cred": "cred_mean"}
KEYS = ["pool", "prior", "K", "popularity", "shuffled", "nbar"]


def summarise(d: pd.DataFrame) -> pd.DataFrame:
    d = d.assign(mom_mid=0.5 * (d.mom_lo + d.mom_hi),
                 hit_mom=(d.mom_lo <= d.I_true) & (d.I_true <= d.mom_hi),
                 hit_cred=(d.cred_lo <= d.I_true) & (d.I_true <= d.cred_hi),
                 hit_cred_count=(d.cred_lo <= d.I_count) & (d.I_count <= d.cred_hi),
                 width_mom=d.mom_hi - d.mom_lo, width_cred=d.cred_hi - d.cred_lo)
    agg = dict(reps=("rep", "size"), N=("N", "first"), I=("I_true", "mean"), F1_N=("F1", "mean"),
               K2=("K2", "mean"), L_med=("mom_L", "median"),
               cov_mom=("hit_mom", "mean"), cov_cred=("hit_cred", "mean"),
               cov_cred_count=("hit_cred_count", "mean"),
               width_mom=("width_mom", "mean"), width_cred=("width_cred", "mean"),
               cred_sd=("cred_sd", "mean"), cred_sd_beta=("cred_sd_beta", "mean"),
               mom_lo=("mom_lo", "median"), mom_hi=("mom_hi", "median"),
               cred_lo=("cred_lo", "median"), cred_hi=("cred_hi", "median"))
    out = d.groupby(KEYS, dropna=False).agg(**agg).reset_index()
    out["F1_N"] = out.F1_N / out.N
    out["cov_se"] = np.sqrt(0.95 * 0.05 / out.reps)
    g = d.groupby(KEYS, dropna=False)
    for name, col in ESTIMATES.items():
        err = d[col] - d.I_true
        out[f"bias_{name}"] = err.groupby([d[k] for k in KEYS], dropna=False).mean().to_numpy()
        sq = (err ** 2).groupby([d[k] for k in KEYS], dropna=False).mean()
        out[f"rmse_{name}"] = np.sqrt(sq).to_numpy()
    out["sd_est_baseline"] = g.I_baseline.std().to_numpy()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    d = pd.concat([pd.read_csv(f) for f in sorted(a.raw.rglob("mi_coverage_*.csv"))], ignore_index=True)
    a.out.mkdir(parents=True, exist_ok=True)
    s = summarise(d)
    s.to_csv(a.out / "summary.csv", index=False)
    cols = ["pool", "shuffled", "nbar", "I", "F1_N", "L_med", "cov_mom", "width_mom", "cov_cred",
            "cov_cred_count", "width_cred", "bias_baseline", "bias_npmle", "bias_mom_mid", "rmse_baseline"]
    with pd.option_context("display.width", 250, "display.max_rows", 500):
        print(s[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
