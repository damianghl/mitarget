import numpy as np
import pytest
from scipy.special import expit

from mitarget import (
    BayesianTargetEncoder, fit_counts, fit_freq_prior, identified_interval, mi_interval,
    moment_covariance, moment_estimates, worst_case_halfwidth,
)
from mitarget.metrics import exact_mutual_information, excess_cross_entropy


def _popularity_sample(N, link, seed, K=50_000):
    """Levels with lognormal popularity; logit q rises with log popularity when ``link`` > 0."""
    rng = np.random.default_rng(seed)
    logp = rng.normal(0, 1.5, K)
    p = np.exp(logp) / np.exp(logp).sum()
    q = expit(-0.5 + link * (logp - logp.mean()) + rng.normal(0, 1, K))
    level = rng.choice(K, size=N, p=p)
    y = rng.random(N) < q[level]
    k = np.bincount(level, weights=y, minlength=K)
    nx = np.bincount(level, minlength=K)
    return np.stack([nx - k, k], axis=1).astype(float), p, np.stack([1 - q, q], axis=1)


def test_freq_prior_beats_baseline_with_popularity_link():
    n, p, q = _popularity_sample(25_000, link=0.6, seed=0)
    f = fit_freq_prior(n)
    assert f.selected and f.p_value < 1e-6
    base = fit_counts(n, "baseline").encoding
    assert excess_cross_entropy(p, q, f.target_encoding) < excess_cross_entropy(p, q, base) - 0.005
    assert np.allclose(f.target_encoding.sum(axis=1), 1)
    # unseen levels get the shifted curve at n = 0, close to the singletons' label rate
    single = n.sum(axis=1) == 1
    assert abs(f.qy[1] - n[single, 1].mean()) < 0.03
    assert np.allclose(f.target_encoding[n.sum(axis=1) == 0], f.qy)


def test_freq_prior_falls_back_without_link():
    n, _, _ = _popularity_sample(25_000, link=0.0, seed=1)
    f = fit_freq_prior(n)
    base = fit_counts(n, "baseline")
    assert f.p_value > 0.01
    if not f.selected:
        assert np.allclose(f.target_encoding, base.encoding)
        assert f.I_hat == base.I_hat
    assert fit_freq_prior(n, lr_test=False).selected


def test_freq_prior_shift_and_linear_spread():
    n, _, _ = _popularity_sample(25_000, link=0.6, seed=2)
    f = fit_freq_prior(n, spread="linear")
    assert len(f.params) == 4 and f.selected
    k, nx = n[:, 1], n.sum(axis=1)
    assert np.allclose(f.predict(k, nx), f.target_encoding[:, 1])
    assert not np.allclose(f.predict(k, nx, shift=0), f.predict(k, nx))
    mu1, _ = f.prior(np.array([0.0]), shift=1)
    mu0, _ = f.prior(np.array([1.0]), shift=0)
    assert np.isclose(mu1[0], mu0[0])


def test_freq_prior_rejects_multiclass_and_works_in_encoder():
    with pytest.raises(ValueError):
        fit_freq_prior(np.ones((5, 3)))
    rng = np.random.default_rng(3)
    x = rng.integers(0, 300, 3000)
    y = rng.random(3000) < expit(0.1 * (x % 7))
    out = BayesianTargetEncoder(method="freq_prior", cv=3, random_state=0).fit_transform(x[:, None], y)
    assert out.shape == (3000, 1) and np.all((out > 0) & (out < 1))


def _atoms_moments(atoms, w, L):
    return np.array([np.sum(np.asarray(w) * np.asarray(atoms) ** j) for j in range(1, L + 1)])


def test_identified_interval_known_values():
    atoms, w = [0.1, 0.5, 0.9], [0.25, 0.5, 0.25]
    m = _atoms_moments(atoms, w, 4)
    lo, hi = identified_interval(m[:1])
    assert lo == pytest.approx(0, abs=1e-9) and hi == pytest.approx(np.log(2))
    lo, hi = identified_interval(m[:2])           # population interval in unseen_mi_limits
    assert lo == pytest.approx(0.170, abs=1e-3) and hi == pytest.approx(0.222, abs=1e-3)
    q = np.array(atoms)
    true_I = exact_mutual_information(np.array(w), np.stack([1 - q, q], axis=1))
    for L in (2, 3, 4):
        lo, hi = identified_interval(m[:L])
        assert lo - 1e-6 <= true_I <= hi + 1e-6


def test_worst_case_halfwidth():
    e = [worst_case_halfwidth(L) for L in (1, 2, 4, 8)]
    assert e[0] == pytest.approx(np.log(2) / 2, abs=1e-6)
    assert all(a > b for a, b in zip(e[:-1], e[1:], strict=True))
    assert 64 * e[3] == pytest.approx(0.22, abs=0.02)


def test_mi_interval_from_data_covers_truth():
    rng = np.random.default_rng(4)
    atoms, w = np.array([0.1, 0.5, 0.9]), np.array([0.25, 0.5, 0.25])
    q = rng.choice(atoms, size=200_000, p=w)
    nx = rng.poisson(1.0, q.size)
    k = rng.binomial(nx, q)
    n = np.stack([nx - k, k], axis=1).astype(float)
    m, se, used = moment_estimates(n, 3)
    assert np.allclose(m, _atoms_moments(atoms, w, 3), atol=4 * se)
    assert used[0] == np.sum(nx >= 1)
    r = mi_interval(n, L=2)
    true_I = exact_mutual_information(w, np.stack([1 - atoms, atoms], axis=1))
    assert r.lower <= true_I <= r.upper
    assert r.width < 0.1 and r.lower <= r.midpoint <= r.upper
    r0 = mi_interval(n, L=2, slack=0)
    assert r.lower <= r0.lower <= r0.upper <= r.upper
    assert mi_interval(n, max_L=3).L == 3


def test_mi_interval_principal_axes_region():
    rng = np.random.default_rng(5)
    atoms, w = np.array([0.2, 0.3, 0.85]), np.array([0.3, 0.4, 0.3])
    q = rng.choice(atoms, size=100_000, p=w)
    nx = rng.poisson(5.0, q.size)
    k = rng.binomial(nx, q)
    n = np.stack([nx - k, k], axis=1).astype(float)
    m, se, _ = moment_estimates(n, 4)
    C = moment_covariance(n, 4)
    assert np.allclose(np.sqrt(np.diag(C)), se, rtol=0.01)
    assert np.all(np.linalg.eigvalsh(C) > 0) and C[0, 3] / np.sqrt(C[0, 0] * C[3, 3]) > 0.5
    true_I = exact_mutual_information(w, np.stack([1 - atoms, atoms], axis=1))
    pca, box = mi_interval(n), mi_interval(n, region="box")
    assert pca.lower <= true_I <= pca.upper and box.lower <= true_I <= box.upper
    # the moments are strongly correlated at nbar = 5, so the rotated region is much narrower
    assert pca.width < 0.6 * box.width
    ident = identified_interval(_atoms_moments(atoms, w, 4))
    assert pca.lower <= ident[0] and ident[1] <= pca.upper
    with pytest.raises(ValueError):
        mi_interval(n, region="ellipse")


def test_mi_interval_likelihood_region():
    pytest.importorskip("cvxpy")
    rng = np.random.default_rng(6)
    atoms, w = np.array([0.2, 0.3, 0.85]), np.array([0.3, 0.4, 0.3])
    q = rng.choice(atoms, size=10_000, p=w)
    nx = rng.poisson(2.0, q.size)
    k = rng.binomial(nx, q)
    n = np.stack([nx - k, k], axis=1).astype(float)
    true_I = exact_mutual_information(w, np.stack([1 - atoms, atoms], axis=1))
    lik, pca = mi_interval(n, region="likelihood"), mi_interval(n)
    assert lik.lower <= true_I <= lik.upper
    assert lik.width < 0.7 * pca.width
    # slack=0 is the range over the NPMLEs, inside the 95% region
    r0 = mi_interval(n, region="likelihood", slack=0)
    assert lik.lower - 1e-4 <= r0.lower
    assert r0.upper <= lik.upper + 1e-4 and r0.width < 0.5 * lik.width
    # thinning large levels keeps the counts consistent
    from mitarget.mi_interval import _thin
    kk, nn = _thin(np.array([3.0, 400.0]), np.array([5.0, 1000.0]), 200)
    assert nn.tolist() == [5.0, 200.0] and kk[0] == 3 and 0 <= kk[1] <= 200


def test_credible_interval_beta_moments_and_coverage():
    from scipy import integrate, stats

    from mitarget import credible_interval
    from mitarget.mi_interval import _beta_h_moments, binary_entropy

    for a, b in [(0.3, 2.0), (4.0, 1.5)]:
        eh, eh2 = _beta_h_moments(np.array(a), np.array(b))
        pdf = stats.beta(a, b).pdf
        assert np.isclose(eh, integrate.quad(lambda q: binary_entropy(q) * pdf(q), 0, 1)[0])  # noqa: B023
        assert np.isclose(eh2, integrate.quad(lambda q: binary_entropy(q) ** 2 * pdf(q), 0, 1)[0])  # noqa: B023
    rng = np.random.default_rng(3)
    K = N = 20_000
    q = rng.beta(0.5, 1.5, K)
    level = rng.integers(0, K, N)
    k = np.bincount(level, weights=rng.random(N) < q[level], minlength=K)
    nx = np.bincount(level, minlength=K)
    n = np.stack([nx - k, k], axis=1).astype(float)
    c = credible_interval(n)
    assert abs(c.mean - fit_counts(n, "baseline").I_hat) < 0.002
    assert c.lower < binary_entropy(q.mean()) - binary_entropy(q).mean() < c.upper
    assert c.sd_beta > c.sd_levels > 0
