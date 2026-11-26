"""
Synthetic scenarios 1-4: MI error and excess cross-entropy of every method,
against Bayes oracles where the true generative family is known.

    python benchmarks/synthetic_scenarios.py            # default settings
    python benchmarks/synthetic_scenarios.py --fast     # quick smoke run
    python benchmarks/synthetic_scenarios.py --scenarios 4 --out results/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mitarget.benchmark import evaluate_synthetic, standard_methods, summarize
from mitarget.synthetic import (
    fit_q0_star, oracle_known_clusters_encoding, oracle_three_point_encoding,
    scenario_parity, scenario_single_beta, scenario_sphere, scenario_two_clusters,
)


def scenarios(fast: bool):
    """(key, label, ground truth, N, oracles) for each scenario."""
    scale = 0.2 if fast else 1.0

    gt1 = scenario_single_beta(kx=int(1000 * scale))
    yield "1", "1. Matched (single beta)", gt1, int(800 * scale), {
        "Oracle (true params)": lambda n: oracle_known_clusters_encoding(
            n, n.sum(0) / n.sum(), betas=[2.3], pis=[1.0]),
    }

    gt2 = scenario_two_clusters(kx=int(1500 * scale))
    yield "2", "2. Matched (2 true clusters)", gt2, int(3000 * scale), {
        "Oracle (true params)": lambda n: oracle_known_clusters_encoding(
            n, n.sum(0) / n.sum(), betas=[0.2, 10.0], pis=[0.5, 0.5]),
    }

    gt3 = scenario_sphere()
    yield "3", "3. Mismatch (sphere symmetry)", gt3, int(2000 * scale), {}

    gt4 = scenario_parity(kx=int(2500 * scale), q0=0.1)
    yield "4", "4. Mismatch (3-delta parity)", gt4, int(2000 * scale), {
        "Oracle (true params)": lambda n: oracle_three_point_encoding(n, q0=0.1),
        "Oracle (est. q0)": lambda n: oracle_three_point_encoding(n, q0=fit_q0_star(n)),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fast", action="store_true", help="smaller problems and fewer repetitions")
    ap.add_argument("--scenarios", nargs="*", default=["1", "2", "3", "4"])
    ap.add_argument("--reps", type=int, default=None, help="repetitions per scenario (default 10, fast 3)")
    ap.add_argument("--crp-reps", type=int, default=None, help="repetitions for CRP (default 2, fast 1)")
    ap.add_argument("--no-crp", action="store_true")
    ap.add_argument("--no-competitors", action="store_true",
                    help="skip the GLMM, NPMLE and CatBoost competitors")
    ap.add_argument("--no-catboost", action="store_true",
                    help="skip the full CatBoost classifier (used by default when installed)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None, help="directory for raw and summary CSVs")
    args = ap.parse_args(argv)

    reps = args.reps or (3 if args.fast else 10)
    crp_reps = args.crp_reps or (1 if args.fast else 2)
    methods = standard_methods(crp=not args.no_crp, sklearn=False,
                               competitors=not args.no_competitors,
                               catboost=False if args.no_catboost else None)

    frames = []
    for key, label, gt, N, oracles in scenarios(args.fast):
        if key not in args.scenarios:
            continue
        print(f"\n== {label}: kx={gt.kx}, N={N}, I_true={gt.I_true:.4f} nats")
        df = evaluate_synthetic(gt, N, reps, methods, oracles=oracles,
                                reps_per_method={"CRP (self-tuned)": crp_reps},
                                seed=args.seed + int(key))
        df.insert(0, "scenario", label)
        print(summarize(df).to_string(float_format=lambda v: f"{v:.5f}"))
        frames.append(df)

    if args.out and frames:
        args.out.mkdir(parents=True, exist_ok=True)
        raw = pd.concat(frames, ignore_index=True)
        raw.to_csv(args.out / "synthetic_raw.csv", index=False)
        summarize(raw, by=("scenario", "method")).to_csv(args.out / "synthetic_summary.csv")
        print(f"\nwrote {args.out}/synthetic_raw.csv and synthetic_summary.csv")


if __name__ == "__main__":
    main()
