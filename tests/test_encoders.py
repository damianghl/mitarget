import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from mitarget import (
    METHODS, BayesianCategoricalClassifier, BayesianTargetEncoder, fit_baseline, fit_counts,
    mutual_information,
)
from mitarget.core import counts_from_labels


@pytest.fixture
def binary_data():
    rng = np.random.default_rng(0)
    cats = np.array([f"c{i}" for i in range(80)])
    p = rng.uniform(size=80)
    idx = rng.integers(0, 80, size=600)
    y = np.where(rng.uniform(size=600) < p[idx], "yes", "no")
    return cats[idx], y


def test_encoder_binary_matches_baseline(binary_data):
    x, y = binary_data
    enc = BayesianTargetEncoder().fit(x.reshape(-1, 1), y)
    out = enc.transform(x.reshape(-1, 1))
    assert out.shape == (600, 1)
    codes, cats = pd.factorize(x)
    n = counts_from_labels(codes, (y == "yes").astype(int))
    expected = fit_baseline(n).target_encoding[codes, 1]
    assert np.allclose(out[:, 0], expected)
    assert list(enc.classes_) == ["no", "yes"]


def test_encoder_unseen_and_missing(binary_data):
    x, y = binary_data
    X = pd.DataFrame({"a": x, "b": np.where(np.arange(600) % 7 == 0, None, x)})
    enc = BayesianTargetEncoder().fit(X, y)
    out = enc.transform(pd.DataFrame({"a": ["never-seen"], "b": [None]}))
    assert out[0, 0] == pytest.approx(np.mean(y == "yes"))
    assert 0 < out[0, 1] < 1
    assert list(enc.get_feature_names_out(["a", "b"])) == ["a", "b"]


def test_encoder_multiclass_shape():
    rng = np.random.default_rng(1)
    x = rng.integers(0, 50, size=500)
    y = rng.integers(0, 3, size=500)
    enc = BayesianTargetEncoder(method="mixture", n_components=2, random_state=0)
    out = enc.fit(x.reshape(-1, 1), y).transform(x.reshape(-1, 1))
    assert out.shape == (500, 3)
    assert np.allclose(out.sum(axis=1), 1)
    assert len(enc.get_feature_names_out()) == 3


def test_fit_transform_is_out_of_fold(binary_data):
    x, y = binary_data
    X = x.reshape(-1, 1)
    enc = BayesianTargetEncoder(method="plugin", cv=5, random_state=0)
    oof = enc.fit_transform(X, y)
    in_sample = enc.transform(X)
    assert oof.shape == in_sample.shape
    assert not np.allclose(oof, in_sample)


def test_encoder_in_pipeline(binary_data):
    x, y = binary_data
    model = make_pipeline(BayesianTargetEncoder(random_state=0), LogisticRegression())
    model.fit(x.reshape(-1, 1), y)
    assert model.score(x.reshape(-1, 1), y) > 0.6


@pytest.mark.parametrize("method", METHODS)
def test_all_methods_produce_distributions(method):
    rng = np.random.default_rng(2)
    n = rng.integers(0, 4, size=(60, 3)).astype(float)
    kw = {"crp_kwargs": dict(n_sweeps=4, burn_in=1, thin=1)} if method == "crp" else {}
    f = fit_counts(n, method, random_state=0, **kw)
    assert f.encoding.shape == (60, 3)
    assert np.allclose(f.encoding.sum(axis=1), 1)


def test_unknown_method():
    with pytest.raises(ValueError):
        fit_counts(np.ones((3, 2)), "nope")


def test_classifier(binary_data):
    x, y = binary_data
    clf = BayesianCategoricalClassifier().fit(x, y)
    proba = clf.predict_proba(x)
    assert proba.shape == (600, 2)
    assert set(clf.predict(x)) <= {"no", "yes"}
    assert clf.mutual_information_ == pytest.approx(mutual_information(x, y))
    with pytest.raises(ValueError):
        clf.fit(np.column_stack([x, x]), y)


def test_baseline_unseen_correction():
    from mitarget import fit_baseline, singleton_rate
    # rows: unseen, three singletons (two with y=1), and two larger levels
    n = np.array([[0, 0], [0, 1], [0, 1], [1, 0], [5, 5], [8, 2]], dtype=float)
    f = fit_counts(n, "baseline_unseen")
    base = fit_baseline(n)
    qy = n.sum(0) / n.sum()
    expected = (np.array([1.0, 2.0]) + qy) / 4
    assert np.allclose(singleton_rate(n), expected)
    assert np.allclose(f.encoding[0], expected)
    assert np.allclose(f.qy, expected)
    assert np.allclose(f.encoding[1:], base.target_encoding[1:])
    assert f.I_hat == base.I_hat
    # no singletons: falls back to the marginal
    n2 = np.array([[0, 0], [2, 3], [4, 1]], dtype=float)
    assert np.allclose(fit_counts(n2, "baseline_unseen").encoding[0], n2.sum(0) / n2.sum())


def test_baseline_robbins_correction():
    from mitarget import doubleton_rates, fit_baseline, singleton_rate
    # unseen, two singletons, doubletons (y counts) 2x(0,2), 1x(1,1), 1x(2,0), one larger level
    n = np.array([[0, 0], [0, 1], [1, 0], [0, 2], [0, 2], [1, 1], [2, 0], [6, 3]], dtype=float)
    f = fit_counts(n, "baseline_robbins_raw")
    base = fit_baseline(n)
    # binary Robbins form, with one pseudo-pair toward the prior row
    F20, F21, F22 = 1, 1, 2
    prior = np.array([[0.3, 0.7], [0.6, 0.4]])
    R = doubleton_rates(n, prior)
    assert R[1, 1] == pytest.approx((2 * F22 + 0.4) / (2 * F22 + F21 + 1))
    assert R[0, 1] == pytest.approx((F21 + 0.7) / (F21 + 2 * F20 + 1))
    assert np.allclose(R.sum(axis=1), 1)
    s = base.beta_star
    qy = base.qy
    pri = (np.eye(2) + s * qy) / (1 + s)
    R = doubleton_rates(n, pri)
    assert np.allclose(f.encoding[0], singleton_rate(n))
    assert np.allclose(f.encoding[1], R[1]) and np.allclose(f.encoding[2], R[0])
    assert np.allclose(f.encoding[3:], base.target_encoding[3:])
    assert f.I_hat == base.I_hat
    # no doubletons: once-seen levels keep the baseline encoding
    n2 = np.array([[0, 1], [1, 0], [4, 1], [2, 5]], dtype=float)
    assert np.allclose(fit_counts(n2, "baseline_robbins_raw").encoding[:2],
                       fit_baseline(n2).target_encoding[:2])
    # multi-class: rows of the pair table are distributions
    n3 = np.array([[1, 1, 0], [0, 0, 2], [0, 1, 1], [1, 0, 0], [3, 2, 4]], dtype=float)
    assert np.allclose(doubleton_rates(n3).sum(axis=1), 1)


def test_doubleton_test_rejects_popularity_shift():
    from mitarget import doubleton_test
    rng = np.random.default_rng(0)
    # singletons and big levels around 0.5; doubletons mostly all-ones (high mean, strong agreement)
    single = np.array([[1, 0], [0, 1]] * 500, dtype=float)
    k = rng.binomial(20, 0.5, 300)
    big = np.column_stack([20 - k, k]).astype(float)
    dbl = np.array([[0, 2]] * 150 + [[2, 0]] * 50 + [[1, 1]] * 20, dtype=float)
    n = np.vstack([single, big, dbl])
    f = fit_counts(n, "baseline")
    assert doubleton_test(n, f.details.beta_star, f.qy) < 1e-6
    g = fit_counts(n, "baseline_robbins")
    assert not np.allclose(g.encoding[:2], f.encoding[:2])
    # doubletons drawn from the fitted beta itself: not rejected
    m = np.vstack([single, big])
    fm = fit_counts(m, "baseline")
    s, q1 = fm.details.beta_star, fm.qy[1]
    k = rng.binomial(2, rng.beta(s * q1, s * (1 - q1), 400))
    m = np.vstack([m, np.column_stack([2 - k, k])])
    fm = fit_counts(m, "baseline")
    assert doubleton_test(m, fm.details.beta_star, fm.qy) > 0.01
