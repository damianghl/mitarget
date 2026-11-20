# Experiments

Scripts behind the paper's experiments that are not package benchmarks. Data
files are not committed; pass their paths on the command line.

## Downstream test on Amazon employee access

`downstream_amazon.py` plugs each encoder (baseline, NPMLE, freq_prior, GLMM,
fixed m = 20, CatBoost's test-time target statistic) into a downstream logistic
regression and a histogram gradient-boosting model on the Kaggle Amazon
employee access data (`train.csv`, 32,769 rows). Encodings for the downstream
fit are out-of-fold inside each outer training split; the outer test split is
encoded by the encoder fit on the whole outer training split. Outer folds are
shared across encoders, and differences against the baseline are tested with
the Nadeau-Bengio corrected resampled t-test.

```bash
python experiments/downstream_amazon.py --data train.csv --fast          # ~1 minute
python experiments/downstream_amazon.py --data train.csv --jobs 4 --out results/
```

Feature sets: `raw` (8 columns; ROLE_CODE duplicates ROLE_TITLE) and `pairs`
(raw plus the 28 pairwise concatenations). One-hot logistic regression and
native CatBoost (if installed) are run on the raw columns as references.

## Real-data benchmark with paired tests

```bash
python experiments/realdata_paired.py --data-root DATA --reps 5 --folds 5 --out raw/   # about 1.5 h on 3 cores
python experiments/realdata_paired.py --data-root DATA --groups bx --tasks bx/user --out raw/
python experiments/realdata_tables.py --raw raw/ --out tables/
```

Methods: prior only (reference for the gain), fixed m = 20, the K = 1
baseline, the baseline with the unseen correction (`baseline_unseen`), the
baseline with the Robbins correction for levels seen at most once
(`baseline_robbins`, and `baseline_robbins_raw` without its doubleton test), the
frequency-aware prior (`freq_prior`), the random-intercept GLMM (adaptive
quadrature), the binomial NPMLE, CatBoost's prediction-time target statistic
and scikit-learn's `TargetEncoder(smooth="auto")`. All are fit on the same training
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

## Real-data subsampling oracle

```bash
python experiments/realdata_oracle.py --data ml1m_ratings.tsv --feature movie_id --reps 50 --out oracle_raw/
python experiments/realdata_oracle.py --data ml1m_ratings.tsv --feature movie_id --reps 50 --shuffle --out oracle_raw/
python experiments/realdata_oracle_tables.py --raw oracle_raw/ --out oracle_tables/
```

Populations with known `q_x`: the levels of a real feature with at least 100
rows, with `q_x` their observed rate. Training sets of `N = nbar * L` rows are
drawn from that population, and the excess cross-entropy is exact. Every
method is fit on the same training sets, so `paired.csv` tests each method's
eCE against the baseline's with a paired t-test over training sets and
reports the share of the baseline's gap to the full oracle it closes.
`--shuffle` permutes `q_x` across levels (control without popularity
dependence). Input files are tab-separated with the feature column and a
`rating` column (`y = rating >= 4`).

## Exchangeability tests

```bash
python experiments/exchangeability.py pools --data ml1m_ratings.tsv --feature movie_id --reps 50 --out exch_raw/
python experiments/exchangeability.py pools --data ml1m_ratings.tsv --feature movie_id --shuffle --out exch_raw/
python experiments/exchangeability.py tasks --data-root DATA --out exch_raw/
python experiments/exchangeability.py summary --raw exch_raw/ --coverage mi_tables/summary.csv --out exch_tables/
```

Two tests of whether levels of every size share one prior for `q` (the assumption behind the MI
intervals): `mitarget.stratum_moment_test`, a Wald test that the unbiased estimates of `m1` and `m2`
agree across strata of level sizes, and the likelihood-ratio test of the frequency-aware prior.
`pools` uses the same seed and draw order as `realdata_oracle.py`, so it tests the oracle's training
sets; `summary` gives rejection rates per pool and `nbar` next to the likelihood interval's coverage.

## Coverage of the MI intervals

```bash
export OMP_NUM_THREADS=1   # one BLAS thread per worker; about 5x faster with --jobs 4
python experiments/mi_coverage.py synthetic --reps 100 --jobs 4 --out mi_raw/        # about 1.5 h on 4 cores
python experiments/mi_coverage.py oracle --data ml1m_ratings.tsv --feature movie_id --reps 50 --jobs 4 --out mi_raw/
python experiments/mi_coverage.py oracle --data ml1m_ratings.tsv --feature movie_id --shuffle --reps 50 --jobs 4 --out mi_raw/
python experiments/mi_coverage_tables.py --raw mi_raw/ --out mi_tables/
```

How often the 95% moment interval (`mitarget.mi_interval`, largest `L <= 4` with at least 100
levels seen `L` times; computed with the default joint region in the principal axes of the moment
covariance, and with the older per-moment box of 2 standard errors) and the 95% credible interval of Hernandez &
Samengo (2019, Appendix A; `mitarget.credible_interval`) cover the population `I(X;Y)`, with
their widths and the bias and RMSE of the point estimates (plug-in, baseline, NPMLE,
frequency prior, moment midpoint). Synthetic populations draw `q_x` from the priors of the
paper's Table `tab:id` (`K = 10^4, 10^5` levels, uniform or lognormal popularity); the oracle
pools are those of `realdata_oracle.py`, real and shuffled. Each row also has the identified
interval at the population's own moments (`id_lo`, `id_hi`), the width that remains with exact
moments up to the same `L`.

## Second simulated review (2026-11-20)

```bash
# fixed-smoothing sweep: s-hat, CV m and m-estimates on a grid of fixed and relative m (K6)
python experiments/fixed_m_sweep.py tasks --data-root DATA --out sweep_tasks/
python experiments/fixed_m_sweep.py pools --data ml1m_ratings.tsv --feature movie_id --nbar 0.5 1 2 5 --out sweep_pools/
python experiments/paper/fig_fixed_m.py --pools sweep_pools/ --tasks sweep_tasks/ --meta table_tasks.csv --logloss table_logloss.csv
# assumption-light statements about I(X;Y) on the full real features (K5)
python experiments/mi_realdata_bounds.py --data-root DATA --logloss table_logloss.csv --lik-max-rows 100000 --out bounds.csv
python experiments/paper/tab_mi_bounds.py bounds.csv
# bakeoff summary by task and by source (M4)
python experiments/paper/tab_bakeoff_summary.py table_paired.csv table_logloss.csv
# likelihood-interval sensitivity: grid size, thinning cap, off-grid atoms (M3)
python experiments/mi_coverage.py synthetic --lik-only --priors atoms3 atoms3_off beta_1_3 junk_f0.5 --pools "10000|uniform" --nbar 1 5 --reps 50 --lik-grid 401 --tag g401 --out mi_sens/
python experiments/mi_coverage.py oracle --lik-only --data ml1m_ratings.tsv --feature movie_id --shuffle --nbar 5 20 --reps 30 --n-cap 1000 --tag cap1000 --out mi_sens/
```
