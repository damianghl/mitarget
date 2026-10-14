import numpy as np
import pytest

from mitarget.synthetic import (
    fit_q0_star, generate_ground_truth, observe, oracle_three_point_encoding, scenario_junk_mixture,
    scenario_parity, scenario_single_beta, scenario_sphere,
)


def test_observe_is_reproducible():
    gt = generate_ground_truth(kx=50, ky=3, seed=0)
    assert np.array_equal(observe(gt, 100, seed=1), observe(gt, 100, seed=1))
    assert observe(gt, 100, seed=1).sum() == 100


def test_ground_truth_rows_are_distributions():
    gt = scenario_single_beta(kx=200)
    assert np.allclose(gt.qx.sum(), 1)
    assert np.allclose(gt.q_ygivenx.sum(axis=1), 1)
    assert gt.I_true > 0


def test_sphere_scenario_matches_paper():
    gt = scenario_sphere()
    assert gt.kx == 4096
    assert gt.I_true == pytest.approx(0.5, abs=0.01)


def test_parity_scenario_three_levels():
    gt = scenario_parity(kx=500, q0=0.1, seed=0)
    assert set(np.round(gt.q_ygivenx[:, 1], 6)) <= {0.1, 0.5, 0.9}


@pytest.mark.parametrize("junk_fraction", [0.2, 0.5, 0.8])
def test_junk_mixture_fraction(junk_fraction):
    gt = scenario_junk_mixture(kx=20000, junk_fraction=junk_fraction, q0=0.2, seed=1)
    assert np.mean(gt.z == 0) == pytest.approx(junk_fraction, abs=0.02)
    assert np.all(gt.q_ygivenx[gt.z == 0, 1] == 0.5)
    assert np.allclose(gt.q_ygivenx[gt.z == 1, 1], 0.2)


def test_junk_mixture_draws_q0():
    gt = scenario_junk_mixture(kx=100, q0=None, seed=3)
    assert len(np.unique(gt.q_ygivenx[:, 1])) == 3


def test_q0_estimate_and_oracle():
    gt = scenario_junk_mixture(kx=3000, q0=0.15, seed=4)
    n = observe(gt, 20000, seed=5)
    assert fit_q0_star(n) == pytest.approx(0.15, abs=0.03)
    enc = oracle_three_point_encoding(n, q0=0.15)
    assert np.allclose(enc.sum(axis=1), 1)
    assert np.all((enc[:, 1] >= 0.15 - 1e-12) & (enc[:, 1] <= 0.85 + 1e-12))
