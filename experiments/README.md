# Experiments

Scripts behind the manuscript's real-data results. Unlike `benchmarks/`, which
reproduces the original notebooks, these run the paper's settings.

## Real-data benchmark with paired tests

```bash
python experiments/realdata_paired.py --data-root DATA --reps 5 --folds 5 --out raw/   # about 1.5 h on 3 cores
python experiments/realdata_paired.py --data-root DATA --groups bx --tasks bx/user --out raw/
python experiments/realdata_tables.py --raw raw/ --out tables/
```

Methods: prior only (reference for the gain), fixed m = 20, the K = 1
baseline, the frequency-aware prior (`freq_prior`), the random-intercept GLMM
(adaptive quadrature) and the binomial NPMLE. All are fit on the same training
folds of a repeated stratified k-fold (seed = repetition index), so every
comparison is paired.

Two paired tests per comparison, in `table_paired.csv`:

- **Nadeau-Bengio corrected resampled t-test** on the R x K split means. It
  inflates the split variance by `1/J + n_test/n_train` to account for
  overlapping training sets. It covers split-to-split variation and is
  conservative.
- **Row-level paired difference, clustered by level.** Each repetition gives
  one out-of-fold loss per row and method; the mean difference is tested with
  a standard error that treats levels as independent clusters, because rows
  of one level share an encoding: `SE = sqrt(sum_c D_c^2) / N` over the level
  totals `D_c` of the difference. The totals are not centered at the row
  mean: on click logs one level can hold a quarter of the rows while the
  methods differ only on small levels, and the usual centered estimator then
  reports `z ~ N / n_max` for every comparison. It is conditional on the data
  set and much more powerful than the split test. The table averages the mean
  and the standard error over repetitions.

p-values are Holm-adjusted within each (task, reference) family.
`table_by_n.csv` splits the out-of-fold loss by the number of training rows
of the test row's level.

### Data layout under `--data-root`

| group | path | source |
|---|---|---|
| `openml` | `arff/` | the four ARFF exports in `code_new.zip:code/data` |
| `movielens` | `movielens/ml1m_ratings.tsv` | MovieLens 1M ratings, y = rating >= 4 |
| `amazon` | `amazon_employee_access/train.csv` | Kaggle Amazon Employee Access |
| `bx` | `book_crossing/BX-*.csv` | Book-Crossing CSV dump; y = rating >= 8 (explicit) or explicit vs implicit |
| `criteo` | `criteo_1tb_day0/*.parquet` | raw parts of Criteo 1TB day 0 (huggingface `criteo/CriteoClickLogs`) |
| `avazu` | `avazu/avazu_train_4M.csv.gz` | first 4M rows of the raw Kaggle Avazu train |

`criteo` needs `pyarrow`.
