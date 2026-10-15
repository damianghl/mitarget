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
