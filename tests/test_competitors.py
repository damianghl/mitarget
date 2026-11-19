import numpy as np
import pytest
from scipy.special import expit

from mitarget import (
    BayesianTargetEncoder, catboost_ts_encoding, fit_counts, fit_glmm, fit_npmle,
    ordered_target_statistics,
)
from mitarget.benchmark import catboost_available, evaluate_cv, standard_methods
from mitarget.core import fixed_m_encoding


def _binary_counts(k1, m):
    return np.stack([m - k1, k1], axis=1).astype(float)


def test_glmm_recovers_parameters():
    rng = np.random.default_rng(0)
    u = rng.normal(0, 1.0, 4000)
    m = rng.poisson(15, 4000)
    k1 = rng.binomial(m, expit(-0.5 + u))
    f = fit_glmm(_binary_counts(k1, m))
    assert abs(f.intercept + 0.5) < 0.1
    assert abs(f.sigma - 1.0) < 0.1
    assert np.allclose(f.target_encoding.sum(axis=1), 1)


def test_glmm_large_level_far_from_intercept():
    # 228 of 319 against a rare-class intercept: a full Newton step from u = 0 overshoots
    n = _binary_counts(np.r_[228, np.zeros(400, int), np.ones(40, int)], np.r_[319, np.full(440, 3)])
    f = fit_glmm(n)
    assert abs(f.target_encoding[0, 1] - 228 / 319) < 0.03


def test_glmm_unseen_and_shrinkage():
    n = _binary_counts(np.array([0, 3, 0, 10]), np.array([0, 3, 2, 20]))
    f = fit_glmm(n)
    assert np.isclose(f.target_encoding[0, 1], f.qy[1])
    assert f.target_encoding[1, 1] < 1   # shrunk towards the intercept


def test_npmle_recovers_atoms_and_beats_beta_on_dense_levels():
    rng = np.random.default_rng(1)
    q = np.where(rng.uniform(size=3000) < 0.3, 0.1, 0.8)
    m = rng.poisson(20, 3000)
    n = _binary_counts(rng.binomial(m, q), m)
    f = fit_npmle(n)
    near = lambda c: f.weights[np.abs(f.grid - c) < 0.05].sum()
    assert near(0.1) > 0.25 and near(0.8) > 0.65
    ce = lambda enc: -np.mean(q * np.log(enc[:, 1]) + (1 - q) * np.log(enc[:, 0]))
    assert ce(f.target_encoding) < ce(fit_counts(n, "baseline").encoding)
    assert 0 < f.I_hat < np.log(2)


def test_multiclass_one_vs_rest():
    rng = np.random.default_rng(2)
    n = rng.poisson(1.5, size=(200, 3)).astype(float)
    for method in ("glmm", "npmle"):
        f = fit_counts(n, method)
        assert f.encoding.shape == (200, 3)
        assert np.allclose(f.encoding.sum(axis=1), 1)
        assert np.isclose(f.qy.sum(), 1)
    assert np.isnan(fit_counts(n, "npmle").I_hat)


def test_catboost_ts_is_fixed_m_one():
    rng = np.random.default_rng(3)
    n = rng.poisson(2, size=(50, 2)).astype(float)
    assert np.allclose(catboost_ts_encoding(n), fixed_m_encoding(n, m=1.0))


def test_ordered_target_statistics_matches_brute_force():
    rng = np.random.default_rng(4)
    x = rng.integers(0, 6, 80)
    y = rng.integers(0, 3, 80)
    out = ordered_target_statistics(x, y, ky=3, seed=7)
    perm = np.random.default_rng(7).permutation(80)
    prior = np.bincount(y, minlength=3) / 80
    for pos, i in enumerate(perm):
        prev = [j for j in perm[:pos] if x[j] == x[i]]
        expected = (np.bincount(y[prev], minlength=3) + prior) / (len(prev) + 1)
        assert np.allclose(out[i], expected)


def test_glmm_posterior_mean_matches_quadrature_and_prior_mean():
    rng = np.random.default_rng(3)
    u = rng.normal(0, 1.6, 6000)
    m = rng.poisson(1.0, 6000)
    k1 = rng.binomial(m, expit(-1.5 + u))
    n = _binary_counts(k1, m)
    mode, mean = fit_glmm(n), fit_glmm(n, predict="mean")
    assert np.isclose(mode.intercept, mean.intercept) and np.isclose(mode.sigma, mean.sigma)
    b0, s = mean.intercept, mean.sigma
    z, w = np.polynomial.hermite_e.hermegauss(80)
    w = w / w.sum()
    # unseen levels: the prior mean of q, not expit(b0) (its median)
    assert np.isclose(mean.qy[1], (w * expit(b0 + s * z)).sum(), atol=1e-6)
    assert np.isclose(mode.qy[1], expit(b0))
    # a level with 1 of 2: posterior mean by Gauss-Hermite on the prior
    i = np.flatnonzero((m == 2) & (k1 == 1))[0]
    lik = expit(b0 + s * z) * (1 - expit(b0 + s * z))
    expected = (w * lik * expit(b0 + s * z)).sum() / (w * lik).sum()
    assert np.isclose(mean.target_encoding[i, 1], expected, atol=1e-6)
    with pytest.raises(ValueError):
        fit_glmm(n, predict="median")


@pytest.mark.parametrize("method", ["glmm", "glmm_mean", "npmle", "catboost_ts"])
def test_encoder_accepts_competitor_methods(method):
    rng = np.random.default_rng(5)
    x = rng.integers(0, 30, 300).astype(str)
    y = rng.integers(0, 2, 300)
    out = BayesianTargetEncoder(method=method, cv=3, random_state=0).fit_transform(x.reshape(-1, 1), y)
    assert out.shape == (300, 1) and np.all((out > 0) & (out < 1))


def test_standard_methods_include_competitors():
    methods = standard_methods(mixture_ks=(), crp=False, catboost=False)
    assert {"GLMM", "NPMLE", "CatBoost TS (a=1)"} <= set(methods)
    assert "CatBoost classifier" not in methods
    assert "GLMM" not in standard_methods(crp=False, competitors=False)


@pytest.mark.skipif(not catboost_available(), reason="catboost not installed")
def test_catboost_classifier_in_cv():
    rng = np.random.default_rng(6)
    x = rng.integers(0, 40, 400)
    y = (rng.uniform(size=400) < expit(x / 20 - 1)).astype(int)
    df = evaluate_cv(x, y, {"cb": standard_methods(mixture_ks=(), crp=False, catboost=True)
                            ["CatBoost classifier"]}, n_splits=2)
    assert np.isfinite(df.logloss).all()


def test_glmm_quadrature_avoids_laplace_blowup_on_singletons():
    rng = np.random.default_rng(0)
    m = np.repeat([1, 2, 3], [3800, 500, 60])
    n = _binary_counts(rng.binomial(m, 0.15), m)
    laplace, agq = fit_glmm(n, n_agq=1), fit_glmm(n)
    assert laplace.sigma > 2 and abs(laplace.qy[1] - 0.15) > 0.08
    assert agq.sigma < 1 and abs(agq.qy[1] - 0.15) < 0.02
