import numpy as np
import pytest

from mitarget import (
    doubleton_test, doubleton_test_pairs, fit_counts, fit_cv_m, fit_npmle, loo_log_loss, singleton_rate,
    stratum_moment_test,
)


def _beta_counts(rng, levels=20000, mean_n=1.5, a=1.0, b=3.0, dep=0.0):
    z = rng.normal(size=levels)
    p = np.exp(1.5 * z)
    p /= p.sum()
    nx = rng.multinomial(int(mean_n * levels), p)
    q = rng.beta(a, b, levels) if dep == 0 else 1 / (1 + np.exp(-(-0.5 + dep * z + rng.normal(size=levels))))
    k = rng.binomial(nx, q)
    return np.column_stack([nx - k, k]).astype(float)


def test_loo_log_loss_matches_brute_force():
    n = np.array([[0, 1], [2, 1], [3, 0], [1, 4], [0, 0]], dtype=float)
    qy = n.sum(0) / n.sum()
    m = 2.5
    total, rows = 0.0, 0
    for c in n:
        for y in (0, 1):
            for _ in range(int(c[y])):
                rest = c.copy()
                rest[y] -= 1
                total -= np.log((rest[y] + m * qy[y]) / (rest.sum() + m))
                rows += 1
    assert loo_log_loss(n, m, qy) == pytest.approx(total / rows)


def test_cv_m_close_to_marginal_likelihood_on_beta_data():
    n = _beta_counts(np.random.default_rng(0))
    m, enc = fit_cv_m(n)
    base = fit_counts(n, "baseline")
    assert m == pytest.approx(base.details.beta_star, rel=0.2)       # true s = a + b = 4
    assert np.allclose(enc.sum(axis=1), 1)
    f = fit_counts(n, "cv_m")
    assert f.details == pytest.approx(m) and np.allclose(f.encoding, enc)


def test_doubleton_test_pairs_matches_beta_version():
    n = _beta_counts(np.random.default_rng(1), mean_n=1.0)
    qy, s = n.sum(0) / n.sum(), 3.0
    pairs = qy[:, None] * (s * qy[None, :] + np.eye(2)) / (s + 1)
    assert doubleton_test_pairs(n, pairs) == pytest.approx(doubleton_test(n, s, qy))


def test_npmle_robbins():
    rng = np.random.default_rng(2)
    n = _beta_counts(rng)
    f, g = fit_npmle(n), fit_counts(n, "npmle_robbins_raw")
    nx = n.sum(axis=1)
    assert np.allclose(g.encoding[nx == 0], singleton_rate(n, n.sum(0) / n.sum()))
    assert np.allclose(g.encoding[nx >= 2], f.target_encoding[nx >= 2])
    # exchangeable beta data: the gated version keeps the NPMLE on once-seen levels
    h = fit_counts(n, "npmle_robbins")
    assert np.allclose(h.encoding[nx == 1], f.target_encoding[nx == 1])
    # popularity dependence: doubletons reject the NPMLE fit and the once-seen encoding changes
    n2 = _beta_counts(rng, dep=0.8)
    h2, f2 = fit_counts(n2, "npmle_robbins"), fit_npmle(n2)
    once = n2.sum(axis=1) == 1
    assert not np.allclose(h2.encoding[once], f2.target_encoding[once])


def test_stratum_moment_test_size_and_power():
    rng = np.random.default_rng(3)
    p_null = [stratum_moment_test(_beta_counts(rng, mean_n=2.0)).p_value for _ in range(40)]
    assert np.mean(np.array(p_null) < 0.05) <= 0.2
    t = stratum_moment_test(_beta_counts(rng, mean_n=2.0, dep=0.6))
    assert t.p_value < 1e-6 and t.df == len(t.strata) * 2 - 1 - 2 and t.max_dev_m1 > 0.02
    with pytest.raises(ValueError):
        stratum_moment_test(np.ones((5, 3)))
