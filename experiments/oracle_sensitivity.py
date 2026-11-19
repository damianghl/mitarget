"""
Sensitivity of the real-data oracle to how the population is built (Section 7.2, Appendix).

    python experiments/oracle_sensitivity.py --design "head100=raw100/" \
        --design "head50=thresh/*_min50.csv" --design "tail=tail/" --nbar 2 --out sensitivity.csv

Each ``--design LABEL=PATH`` names a folder (all ``oracle_*.csv`` in it) or a glob of raw files of
``realdata_oracle.py``. Several designs may share a label (their files are merged, as when a later run added
methods on the same training sets). For every design, pool, real or shuffled, and ``nbar``, the output has the
number of levels, corr(log p, q), the full oracle's gain over the marginal, the baseline's gap to the oracle
(milli-nats and share of the gain), the share of that gap on levels seen at most once, and for each method
the share of the baseline's gap it closes, with its paired standard error over training sets.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["design", "feature", "shuffled", "nbar"]
CLOSERS = ["baseline_unseen", "baseline_robbins", "freq_prior", "npmle", "npmle_robbins", "glmm", "glmm_mean",
           "cv_m", "fixed_m20"]


def read_design(spec: str) -> pd.DataFrame:
    label, path = spec.split("=", 1)
    p = Path(path)
    files = sorted(p.glob("oracle_*.csv")) if p.is_dir() else sorted(Path(f) for f in glob.glob(path))
    if not files:
        raise SystemExit(f"no files for {spec}")
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return d.assign(design=label)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", action="append", required=True)
    ap.add_argument("--nbar", type=float, nargs="*", default=None)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    d = pd.concat([read_design(s) for s in a.design], ignore_index=True)
    d = d.drop_duplicates(KEYS + ["rep", "method"], keep="first")
    if a.nbar:
        d = d[d.nbar.isin(a.nbar)]
    d = d.assign(e01=d.eCE_n0 + d.eCE_n1)
    w = d.pivot_table(index=KEYS + ["rep"], columns="method", values=["eCE", "e01"])
    meta = d.groupby(KEYS)[["L", "corr_logp_q"]].first()
    gap = w["eCE"].sub(w["eCE"]["oracle"], axis=0)
    gap01_base = w["e01"]["baseline"] - w["e01"]["oracle"]
    gain = w["eCE"]["marginal"] - w["eCE"]["oracle"]

    rows = []
    for key, g in gap.groupby(level=KEYS):
        gb = g["baseline"]
        r = dict(zip(KEYS, key, strict=True), L=int(meta.loc[key, "L"]),
                 corr_logp_q=meta.loc[key, "corr_logp_q"], reps=len(g),
                 gain_mnats=1e3 * gain.loc[g.index].mean(), baseline_gap_mnats=1e3 * gb.mean(),
                 baseline_share_lost=gb.mean() / gain.loc[g.index].mean(),
                 gap_share_n01=gap01_base.loc[g.index].mean() / gb.mean())
        for m in CLOSERS:
            if m not in g or g[m].isna().all():
                continue
            diff = (gb - g[m]).dropna()
            r[f"closed_{m}"] = diff.mean() / gb.mean()
            r[f"closed_{m}_se"] = diff.std(ddof=1) / np.sqrt(len(diff)) / gb.mean()
        rows.append(r)
    out = pd.DataFrame(rows).sort_values(["feature", "shuffled", "design", "nbar"])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    cols = ["design", "feature", "shuffled", "nbar", "L", "corr_logp_q"]
    cols += ["baseline_gap_mnats", "gap_share_n01"]
    cols += [c for c in out.columns if c.startswith("closed_") and not c.endswith("_se")]
    print(out[cols].to_string(index=False, float_format=lambda v: f"{v:+.3f}"))


if __name__ == "__main__":
    main()
