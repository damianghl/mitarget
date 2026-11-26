"""
Rebuild the paper's tables and figures from the released result files, without rerunning experiments.

    python experiments/paper/make_all.py --results results/ --out paper_out/
    python experiments/paper/make_all.py --results results/ --out paper_out/ --check PAPER/tables

``--results`` is the unpacked ``results.zip`` of the tagged release (layout in its ``README.md``).
``--check`` compares every regenerated ``.tex`` table with the copy in the manuscript's ``tables/``.
``--with-synthetic`` also redraws the two figures that simulate their own data (``fig_shift``,
``fig_interval``; a few minutes). ``fig_teaser`` needs the prepared pool files of
``experiments/realdata_oracle.py`` and is built only with ``--pools-dir``.
"""

import argparse
import filecmp
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def jobs(R: Path, out: Path, a):
    b = R / "bakeoff" / "tables"
    mi = R / "mi_coverage" / "summary.csv"
    yield "fig_junk.py", [R / "synthetic_junk" / "junk_fraction_summary.csv", "--out", out / "fig_junk.pdf"]
    yield "fig_oracle.py", [R / "oracle" / "tables", "--out-dir", out]
    yield "fig_mi_coverage.py", [mi, "--with-lik", "--intervals", "cred", "lik",
                                 "--fig", out / "fig_mi_coverage.pdf", "--tab", out / "tab_mi_coverage.tex"]
    yield "fig_mi_coverage.py", [mi, "--intervals", "cred", "mom", "lik",
                                 "--fig", out / "fig_mi_coverage_full.pdf",
                                 "--tab", out / "tab_mi_coverage_full.tex"]
    yield "tab_mi_point.py", [mi, "--tab", out / "tab_mi_point.tex"]
    yield "tab_mi_classical.py", [R / "mi_coverage" / "classical", "--tab", out / "tab_mi_classical.tex"]
    yield "tab_mi_sens.py", [R / "mi_coverage" / "sensitivity", "--tab", out / "tab_mi_sens.tex"]
    yield "tab_mi_bounds.py", [R / "mi_bounds" / "bounds_final.csv", "--tab", out / "tab_mi_bounds.tex"]
    yield "tab_bakeoff.py", [b / "table_main.csv", "--out-dir", out]
    yield "tab_bakeoff_abs.py", [b / "table_paired.csv", "--tab", out / "tab_bakeoff_abs.tex"]
    yield "tab_bakeoff_summary.py", [b / "table_paired.csv", b / "table_logloss.csv",
                                     "--tab", out / "tab_bakeoff_summary.tex"]
    yield "tab_wtl.py", [b / "table_wtl.csv", "--tab", out / "tab_wtl.tex"]
    yield "tab_exch.py", [R / "exchangeability" / "tables" / "exch_pools_summary.csv",
                          R / "exchangeability" / "raw" / "exch_tasks.csv", "--out-dir", out]
    yield "tab_oracle_sensitivity.py", [R / "oracle" / "sensitivity" / "oracle_sensitivity.csv",
                                        "--tab", out / "tab_oracle_sensitivity.tex"]
    yield "fig_fixed_m.py", ["--pools", R / "fixed_m" / "pools", "--tasks", R / "fixed_m" / "tasks",
                             "--meta", b / "table_tasks.csv", "--logloss", b / "table_logloss.csv",
                             "--fig", out / "fig_fixed_m.pdf", "--tab", out / "tab_fixed_m.tex"]
    yield "tab_downstream.py", [R / "downstream_amazon", "--tab", out / "tab_downstream.tex"]
    if a.with_synthetic:
        yield "fig_interval.py", ["--out", out / "fig_interval.pdf"]
        yield "fig_shift.py", ["--reps", "20", "--out", out / "fig_shift.pdf"]
    if a.pools_dir:
        P = Path(a.pools_dir)
        yield "fig_teaser.py", ["--by-n", b / "table_by_n.csv", "--pools",
                                f"movie_id={P / 'ml1m_ratings.tsv'}:movie_id",
                                f"bx_user_implicit={P / 'bx_user_implicit.tsv'}",
                                f"avazu_device_ip={P / 'avazu_device_ip.tsv'}",
                                "--out", out / "fig_teaser.pdf"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--check", type=Path, help="manuscript tables/ directory to compare against")
    ap.add_argument("--with-synthetic", action="store_true")
    ap.add_argument("--pools-dir",
                    help="folder with ml1m_ratings.tsv, bx_user_implicit.tsv and avazu_device_ip.tsv")
    a = ap.parse_args(argv)
    R, out = a.results.resolve(), a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    failed = []
    for script, args in jobs(R, out, a):
        cmd = [sys.executable, str(HERE / script)] + [str(x) for x in args]
        print("+", script, " ".join(str(x) for x in args[:1]), flush=True)
        r = subprocess.run(cmd, cwd=HERE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if r.returncode:
            failed.append(script)
            print(r.stderr[-2000:], file=sys.stderr)
    (out / "fig_mi_coverage_full.pdf").unlink(missing_ok=True)   # by-product of the full coverage table
    if a.check:
        same, diff = [], []
        for t in sorted(out.glob("*.tex")):
            ref = a.check / t.name
            if ref.exists():
                (same if filecmp.cmp(t, ref, shallow=False) else diff).append(t.name)
        print(f"identical to the manuscript: {len(same)} tables")
        if diff:
            print("different:", " ".join(diff))
    if failed:
        sys.exit(f"failed: {' '.join(failed)}")


if __name__ == "__main__":
    main()
