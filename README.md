# mitarget

Empirical-Bayes target encoding, classification and mutual-information
estimation for **large, under-sampled categorical features**: many categories,
most seen only a handful of times, and a target with few, well-sampled classes.

The core estimator is the one of Hernandez & Samengo (2019),
[*Estimating the Mutual Information between Two Discrete, Asymmetric Variables
with Limited Samples*](https://doi.org/10.3390/e21060623), Entropy 21(6):623.
Each category's conditional distribution gets the prior

    q(y | x) ~ Dirichlet(beta * q(y))

and the concentration `beta` is fit by maximizing the marginal likelihood. The
posterior mean

    qhat(y | x) = (n_xy + beta * q(y)) / (n_x + beta)

is classic smoothed target encoding (Micci-Barreca 2001) with the pseudo-count
learned from coincidences in the data instead of hand-picked.

## Estimators

| `method` | Model | Fit |
|---|---|---|
| `"baseline"` | one shared `beta` | empirical Bayes (1-D optimization) |
| `"baseline_unseen"` | the baseline, but unseen categories get the label rate of categories seen once (`singleton_rate`) | as baseline |
| `"baseline_robbins"` | `"baseline_unseen"`, and categories seen once get the label rate of the other row in doubletons sharing their label (`doubleton_rates`), when a likelihood-ratio test on the doubletons rejects the baseline (`doubleton_test`; `"baseline_robbins_raw"` always) | as baseline |
| `"mixture"` | `K` clusters of categories, each with its own `beta_c` | EM with restarts |
| `"crp"` | Chinese Restaurant Process over clusters, self-tuning concentration | collapsed Gibbs |
| `"fixed_m"` | additive smoothing with fixed `m` (default 20) | none, for comparison |
| `"plugin"` | raw frequencies | none, for comparison |
| `"glmm"` | random-intercept logistic GLMM, `logit q_x = b0 + u_x` (Pargent et al. 2022) | Laplace marginal likelihood, as lme4 |
| `"npmle"` | nonparametric prior for `q_x` on a grid (Kiefer-Wolfowitz) | EM |
| `"catboost_ts"` | CatBoost target statistic at prediction time, `(n_xy + q_y) / (n_x + 1)` | none, for comparison |
| `"freq_prior"` | beta prior whose mean depends on the category's count, `logit mu_x = a + b log(1 + n_x)` (binary `y`) | type-II ML, used only if a likelihood-ratio test rejects `b = 0` |

Every estimator handles any number of target classes; `"glmm"`, `"npmle"` and
`"freq_prior"` are binary models fit one-vs-rest for `ky > 2`. `ordered_target_statistics`
gives CatBoost's ordered encoding of training rows, and
`catboost_classifier_encoding` the full CatBoost classifier on one feature
(`pip install -e ".[catboost]"`).

When the target rate depends on how popular a category is, the frequency prior
recovers most of the gap to the oracle. The encoding reads the fitted curve at
`n_x + 1`, since the right prior for a category seen `n` times is the law of
`q` among categories seen `n + 1` times; unseen categories therefore get
roughly the label rate of singletons. The MI estimate uses the curve at `n_x`.

## Mutual information: point estimate and identified interval

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

## Install

```bash
pip install -e ".[dev]"
```

## Usage

```python
from mitarget import BayesianTargetEncoder, BayesianCategoricalClassifier, mutual_information

# Encode categorical columns for a downstream model (out-of-fold in fit_transform).
enc = BayesianTargetEncoder(method="baseline")
X_enc = enc.fit_transform(X_cat, y)          # binary y: one column per feature
X_test_enc = enc.transform(X_cat_test)       # unseen categories get the prior q(y)

# Predict y from one categorical feature directly.
clf = BayesianCategoricalClassifier(method="mixture", n_components=2).fit(x, y)
proba = clf.predict_proba(x_test)
clf.mutual_information_                      # MI estimate in nats

# Or estimate I(X;Y) on its own.
mutual_information(x, y, method="baseline")
```

Lower-level functions work on a `(kx, ky)` count matrix: `fit_baseline`,
`fit_finite_mixture`, `fit_crp_mixture`, or `fit_counts(n, method)`.

## Layout

```
src/mitarget/
  core.py            Dirichlet-multinomial likelihood, posterior means, entropies
  baseline.py        single-beta empirical-Bayes estimator
  finite_mixture.py  K-component mixture (EM)
  crp.py             CRP mixture (collapsed Gibbs, Escobar-West alpha updates)
  competitors.py     GLMM, NPMLE and CatBoost encoders from the literature
  freq_prior.py      frequency-aware prior (prior mean depends on the category's count)
  mi_interval.py     moment-identified interval for I(X;Y)
  encoders.py        scikit-learn encoder and classifier
  metrics.py         exact MI, excess cross-entropy, held-out log-loss
  datasets.py        ARFF reader and real-data loaders
  benchmark.py       evaluation harness shared by the scripts
  synthetic/         generative model, scenarios 1-4, Bayes oracles
benchmarks/          synthetic and real-data benchmark scripts (see benchmarks/README.md)
tests/
```

## Tests

```bash
pytest -q
ruff check .
```

## License

GPL-3.0, see [LICENSE](LICENSE).
