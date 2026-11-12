# Paper figures and tables

Scripts behind the figures and tables of the manuscript (damianghl/mitarget-paper). Outputs are copied
into the paper repo's `figures/` and `tables/`. Data are not committed (licences); paths are arguments.

| output | paper | script | input |
|---|---|---|---|
| `fig_shift.pdf` | Fig. `fig:shift` (Sec. 5) | `python fig_shift.py --reps 20` | synthetic |
| `fig_interval.pdf` | Fig. `fig:interval` (Sec. 6) | `python fig_interval.py` | none (moment LPs) |
| `junk_fraction.csv`, `_summary.csv` | Sec. 7.1 | `python run_junk_fraction.py --reps 40 --out junk_fraction.csv` | synthetic |
| `fig_junk.pdf` | Fig. `fig:junk` (Sec. 7.1) | `python fig_junk.py junk_fraction_summary.csv` | sweep above |
| `fig_oracle.pdf`, `fig_gap_by_n.pdf`, `tab_oracle.tex`, `tab_oracle_gaps.tex` | Sec. 7.2 | `python fig_oracle.py <tables dir>` | `summary.csv`, `paired.csv`, `baseline_gap_by_n.csv` from `../realdata_oracle_tables.py` |
| `fig_mi_coverage.pdf`, `tab_mi_coverage.tex` | Sec. 8 | `python fig_mi_coverage.py <tables dir>/summary.csv` | `summary.csv` from `../mi_coverage_tables.py` |
| `tab_mi_point.tex` | Tab. `tab:mi-point` (App. `app:mi-point`) | `python tab_mi_point.py <tables dir>/summary.csv` | `summary.csv` from `../mi_coverage_tables.py` |
| real-pool regret numbers | Sec. 4, numerical check | `python real_pool_regret.py name=path[:feature] ...` | prepared pool `.tsv` files of `../realdata_oracle.py` |
| `tab_bakeoff.tex`, `tab_bakeoff_full.tex` | Tab. `tab:bakeoff` (Sec. 7.3) | `python tab_bakeoff.py <tables dir>/table_main.csv` | `table_main.csv` from `../realdata_tables.py` |
| `tab_exch_pools.tex`, `tab_exch_tasks.tex` | Tabs. `tab:exch-pools`, `tab:exch-tasks` (App. `app:exch`) | `python tab_exch.py <exch tables>/exch_pools_summary.csv <exch raw>/exch_tasks.csv` | `exch_pools_summary.csv` from `../exchangeability.py summary --coverage`, `exch_tasks.csv` from `../exchangeability.py tasks` |
| `fig_teaser.pdf` | Fig. `fig:teaser` (Sec. 1) | `python fig_teaser.py --by-n <tables dir>/table_by_n.csv --pools movie_id=ml1m_ratings.tsv:movie_id bx_user_implicit=bx_user_implicit.tsv avazu_device_ip=avazu_device_ip.tsv` | `table_by_n.csv` from `../realdata_tables.py`, prepared pool `.tsv` files |
| large-n regret table | Tab. `tab:largen` (App. `app:gap`) | `python large_n_regret.py` | none (exact sums) |
| GLMM Laplace vs AGQ | Tab. `tab:glmm` (App. `app:glmm`) | `python glmm_laplace.py --arff-dir code/data --avazu avazu_train_4M.csv.gz` | OpenML ARFF exports, raw Avazu |
| theory checks | Tabs. `tab:regret`, `tab:avg`, `tab:id`; MC check of Prop. `prop:ml` | `python theory/verify.py`, `python theory/pseudo_true.py`, `python theory/unseen_mi_bounds.py` | none (`verify.py` needs `sympy`) |

The MI coverage input comes from `experiments/mi_coverage.py` (see `experiments/README.md`). The oracle and bakeoff inputs come from `experiments/realdata_oracle.py` and
`experiments/realdata_paired.py` (see `experiments/README.md`). The Robbins columns (`baseline_robbins`) were run
later on the same training sets and splits (`realdata_oracle.py --methods baseline_robbins --suffix _robbins`,
`realdata_paired.py --methods "K=1 baseline" "Baseline + Robbins (n<=1)" ...`) and merged into the earlier raw
files before running the table scripts. Training sets and splits do not depend on the method list, so the
merged files are paired.
Run from this directory (`_style.py` holds the shared matplotlib style).
