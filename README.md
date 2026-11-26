# mitarget

Empirical-Bayes target encoding and mutual-information estimation for **large, under-sampled
categorical features**: user, device or product identifiers with many levels, most of them seen
zero, one or two times, and a binary (or few-class) target.

Each level's target rate `q_x` gets a prior fitted to the whole feature, and the encoding is its
posterior mean. With a beta prior this is the classic smoothed encoding
`(k_x + m * mu) / (n_x + m)` (Micci-Barreca 2001), with `m` **fitted by marginal likelihood instead of
fixed**. The package also has richer priors (nonparametric NPMLE, random-intercept GLMM, a
frequency-aware prior), corrections for levels whose rate depends on their popularity, and point
estimates and intervals for the information `I(X;Y)` the feature carries.

<!-- public -->
The methods and their theory are in the paper *When is simple target encoding optimal? Moment
exactness for sparse categorical features, and the limits of estimating their information*
(Damián G. Hernández, 2026; see [Citation](#citation)). `experiments/` reproduces every table and
figure in it.
<!-- /public -->

## Install

Python 3.10 or later.

<!-- public -->
```bash
pip install "mitarget @ git+https://github.com/damianghl/mitarget@v0.1.0"
```

or from a clone:
<!-- /public -->

```bash
pip install -e .                 # numpy, scipy, pandas, scikit-learn
pip install -e ".[lik]"          # + cvxpy, for the likelihood-region MI interval
pip install -e ".[plots]"        # + matplotlib, for the paper's figures
pip install -e ".[catboost]"     # + catboost, for the CatBoost comparisons
pip install -e ".[dev]"          # + pytest, ruff, cvxpy
```

## Quick start

```python
import numpy as np, pandas as pd
from mitarget import BayesianTargetEncoder, counts_from_labels, fit_counts, mi_interval

# Encode categorical columns for a downstream model.
enc = BayesianTargetEncoder(method="baseline")     # or "npmle", "freq_prior", "glmm_mean", ...
X_enc = enc.fit_transform(X_cat, y)                # out-of-fold encodings for the training rows
X_test_enc = enc.transform(X_cat_test)             # unseen levels get the prior mean

# Work on one feature's level-by-class count matrix.
n = counts_from_labels(x_codes, y)                 # (levels, classes)
fit = fit_counts(n, method="npmle")
fit.encoding                                       # (levels, classes) posterior-mean encoding
fit.I_hat                                          # point estimate of I(X;Y), nats
r = mi_interval(n)                                 # r.lower, r.upper: interval for I(X;Y)
```

`BayesianCategoricalClassifier` predicts `y` from a single feature, and `mutual_information(x, y,
method=...)` returns the MI estimate directly.

## Encoders

`fit_counts(n, method)` and the `method=` argument of the scikit-learn wrappers:

| `method` | Prior or rule | Fit |
|---|---|---|
| `"baseline"` | one beta (Dirichlet) prior, mean at the label rate | `m` by type-II maximum likelihood |
| `"baseline_unseen"` | baseline; unseen levels get the label rate of levels seen once | as baseline |
| `"baseline_robbins"` | baseline; levels seen at most once read off the singleton and doubleton counts (Robbins), when a test on the doubletons rejects the baseline (`"baseline_robbins_raw"`: always) | as baseline |
| `"npmle"` | nonparametric prior on a grid (Kiefer-Wolfowitz) | EM; `"npmle_robbins"` adds the Robbins step |
| `"freq_prior"` | beta prior whose mean depends on the level's count | type-II ML, kept only if a likelihood-ratio test rejects a constant mean |
| `"glmm"`, `"glmm_mean"` | random-intercept logistic GLMM (Pargent et al. 2022), predicted at the conditional mode or the posterior mean | adaptive Gauss-Hermite quadrature |
| `"cv_m"` | smoothed encoding with `m` chosen by cross-validated log-loss | grid search |
| `"fixed_m"`, `"plugin"` | fixed `m` (default 20); raw frequencies | none, for comparison |
| `"catboost_ts"` | CatBoost's prediction-time target statistic, `m = 1` | none, for comparison |
| `"mixture"`, `"crp"` | `K` clusters of levels, each with its own beta; Chinese-restaurant-process mixture | EM; collapsed Gibbs (experimental, not in the paper) |

The beta-based estimators handle any number of classes; `"glmm"`, `"npmle"` and `"freq_prior"` are
binary models fitted one-vs-rest for more classes. `stratum_moment_test` tests whether levels of
different sizes share one prior (exchangeability).

## Mutual information

Categories seen at most `L` times identify only the first `L` moments of the
prior of `q(y=1 | x)`, so `I(X;Y)` is only partially identified: singletons
alone give `[0, H(Y)]`, and the worst-case half-width is about `0.22 / L^2`
nats. `mi_interval` returns that interval from a binary count matrix (two
linear programmes per value of the label rate, with the estimated moments
allowed to vary within a joint 95% region: a box in the principal axes of their
covariance, or with `region="box"` the older per-moment box of `slack` standard
errors); the beta estimators return one point inside it.

```python
from mitarget import counts_from_labels, fit_baseline, mi_interval

n = counts_from_labels(x_codes, y_codes)
fit_baseline(n).I_hat          # point estimate under the beta prior
r = mi_interval(n)             # r.lower, r.upper, r.midpoint, r.L
```

`mi_interval(n, region="likelihood")` skips the moments: it takes every prior
whose likelihood for the table of level sizes and positives is within
`chi2_1(0.95) / 2` of the NPMLE and returns the range of `I` over that convex
set. Each cell then counts with its own precision and no `L` is chosen, which
narrows the sampling part of the width; the identified part stays (needs
`cvxpy`: `pip install -e ".[lik]"`).

`identified_interval(moments)` gives the population interval for known
moments, and `worst_case_halfwidth(L)` the bound `E_L(h)`.

## Reproducing the paper

`experiments/` has every experiment script, and `experiments/paper/` the scripts that turn their
outputs into the paper's tables and figures. The result files of the paper's runs are in `results.zip`
<!-- public -->
on the [v0.1.0 release](https://github.com/damianghl/mitarget/releases/tag/v0.1.0)
<!-- /public -->
(85 MB unpacked). To rebuild every table and figure from them, about two minutes:

```bash
pip install -e ".[plots]"
unzip results.zip
python experiments/paper/make_all.py --results results/ --out paper_out/ --with-synthetic
```

To rerun the experiments themselves, see `experiments/README.md`: the commands, run times and the
expected layout of the data. The data sets (MovieLens 1M, Book-Crossing, Criteo 1TB, Avazu, Amazon
employee access, four OpenML data sets) are not redistributed; the paper's appendix lists their sources
and terms. Large sweeps run fastest with `OMP_NUM_THREADS=1` and `--jobs`.

## Layout

```
src/mitarget/
  core.py            Dirichlet-multinomial likelihood, posterior means, entropies
  baseline.py        beta-prior estimator, unseen and Robbins corrections, CV m
  competitors.py     NPMLE, GLMM and CatBoost encoders
  freq_prior.py      frequency-aware prior
  exchangeability.py stratum moment test
  mi_interval.py     identified, moment and likelihood-region intervals for I(X;Y); credible interval
  encoders.py        fit_counts, scikit-learn encoder and classifier
  finite_mixture.py, crp.py   K-mixture and CRP-mixture priors (experimental)
  metrics.py         exact MI, excess cross-entropy, held-out log-loss
  datasets.py, benchmark.py   data loaders and evaluation harness
  synthetic/         generative models and Bayes oracles
experiments/         the paper's experiments (README.md) and, in paper/, its tables and figures
benchmarks/          earlier synthetic and real-data benchmarks
tests/
```

## Tests

```bash
pip install -e ".[dev]"
pytest -q
ruff check .
```

## License

BSD-3-Clause, see [LICENSE](LICENSE).
<!-- public -->

## Citation

If you use mitarget, please cite the paper:

```bibtex
@misc{hernandez2026targetencoding,
  title  = {When is simple target encoding optimal? Moment exactness for sparse categorical
            features, and the limits of estimating their information},
  author = {Hern{\'a}ndez, Dami{\'a}n G.},
  year   = {2026},
  note   = {Preprint}
}
```

The estimator builds on D. G. Hernández and I. Samengo, [Estimating the mutual information between two
discrete, asymmetric variables with limited samples](https://doi.org/10.3390/e21060623), *Entropy*
21(6):623, 2019.
<!-- /public -->
