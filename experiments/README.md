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
