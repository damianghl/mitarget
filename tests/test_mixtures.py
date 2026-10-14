import numpy as np
import pytest

from mitarget import excess_cross_entropy, fit_baseline, fit_crp_mixture, fit_finite_mixture
from mitarget.synthetic import observe, oracle_known_clusters_encoding, scenario_two_clusters


@pytest.fixture(scope="module")
def two_clusters():
    gt = scenario_two_clusters(kx=600, seed=11)
    return gt, observe(gt, 3000, seed=5).astype(float)


def test_mixture_recovers_two_betas(two_clusters):
    gt, n = two_clusters
    fit = fit_finite_mixture(n, K=2, seed=0)
    assert fit.beta[0] < 1.0 < 3.0 < fit.beta[1]   # true betas 0.2 and 10
    assert np.allclose(fit.pi.sum(), 1)
    assert np.allclose(fit.responsibilities.sum(axis=1), 1)
    assert np.allclose(fit.target_encoding.sum(axis=1), 1)


def test_mixture_close_to_oracle(two_clusters):
    gt, n = two_clusters
    qy = n.sum(0) / n.sum()
    oracle = oracle_known_clusters_encoding(n, qy, betas=[0.2, 10.0], pis=[0.5, 0.5])
    ce = lambda enc: excess_cross_entropy(gt.qx, gt.q_ygivenx, enc)
    ce_mix = ce(fit_finite_mixture(n, K=2, seed=0).target_encoding)
    assert ce(oracle) <= ce_mix + 1e-3
    assert ce_mix < ce(fit_baseline(n).target_encoding)


def test_one_component_mixture_equals_baseline(two_clusters):
    _, n = two_clusters
    mix = fit_finite_mixture(n, K=1, n_init=1, seed=0)
    base = fit_baseline(n)
    assert mix.beta[0] == pytest.approx(base.beta_star, rel=1e-4)
    assert mix.I_hat == pytest.approx(base.I_hat, rel=1e-4)


def test_crp_runs_and_is_reproducible(two_clusters):
    _, n = two_clusters
    kw = dict(n_sweeps=8, burn_in=2, thin=3, seed=1)
    a = fit_crp_mixture(n, **kw)
    b = fit_crp_mixture(n, **kw)
    assert np.array_equal(a.target_encoding, b.target_encoding)
    assert len(a.I_hat_samples) == 2
    assert len(a.n_clusters_trace) == 8
    assert np.allclose(a.target_encoding.sum(axis=1), 1)


def test_crp_fixed_alpha_keeps_alpha_constant(two_clusters):
    _, n = two_clusters
    fit = fit_crp_mixture(n, alpha_crp=0.5, alpha_prior=None, n_sweeps=4, burn_in=1, thin=1, seed=0)
    assert np.all(fit.alpha_trace == 0.5)


def test_crp_rejects_empty_chain(two_clusters):
    _, n = two_clusters
    with pytest.raises(ValueError):
        fit_crp_mixture(n, n_sweeps=5, burn_in=5)


def test_mixture_multiclass():
    from mitarget.synthetic import generate_ground_truth_fixed_groups
    gt = generate_ground_truth_fixed_groups(kx=300, ky=4, proportions=[0.5, 0.5],
                                            betas=[0.3, 20.0], qx_kind="uniform", seed=2)
    n = observe(gt, 2000, seed=3).astype(float)
    fit = fit_finite_mixture(n, K=2, seed=0)
    assert fit.target_encoding.shape == (300, 4)
    assert fit.beta[0] < fit.beta[1]
