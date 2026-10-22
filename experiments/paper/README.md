# Paper figures and tables

Scripts behind the figures and tables of the manuscript (damianghl/mitarget-paper). Outputs are copied
into the paper repo's `figures/` and `tables/`. Data are not committed (licences); paths are arguments.

| output | paper | script | input |
|---|---|---|---|
| `fig_shift.pdf` | Fig. `fig:shift` (Sec. 5) | `python fig_shift.py --reps 20` | synthetic |
| `fig_interval.pdf` | Fig. `fig:interval` (Sec. 6) | `python fig_interval.py` | none (moment LPs) |
| `junk_fraction.csv`, `_summary.csv` | Sec. 7.1 | `python run_junk_fraction.py --reps 40 --out junk_fraction.csv` | synthetic |
| `fig_junk.pdf` | Fig. `fig:junk` (Sec. 7.1) | `python fig_junk.py junk_fraction_summary.csv` | sweep above |
| `fig_oracle.pdf`, `fig_gap_by_n.pdf`, `tab_oracle.tex` | Sec. 7.2 | `python fig_oracle.py <results dirs>` | experiment-A `summary.csv`, `baseline_gap_by_n.csv` |
| `tab_bakeoff.tex`, `tab_bakeoff_full.tex` | Tab. `tab:bakeoff` (Sec. 7.3), Tab. `tab:bakeoff-full` (App. E) | `python tab_bakeoff.py table_main.csv` | `table_main.csv` from `experiments/realdata_tables.py` (paired rerun) |

The experiment-A results directories and the bakeoff raw CSVs are produced by the oracle and bakeoff
scripts, which still live outside this package; porting them here is on the manuscript to-do list.
Run from this directory (`_style.py` holds the shared matplotlib style).
