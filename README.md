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
| `"mixture"` | `K` clusters of categories, each with its own `beta_c` | EM with restarts |
| `"crp"` | Chinese Restaurant Process over clusters, self-tuning concentration | collapsed Gibbs |
| `"fixed_m"` | additive smoothing with fixed `m` (default 20) | none, for comparison |
| `"plugin"` | raw frequencies | none, for comparison |

Every estimator handles any number of target classes.

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
