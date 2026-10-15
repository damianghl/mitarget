import numpy as np
import pytest
from scipy import integrate
from scipy.special import betaln

from mitarget import fit_baseline
from mitarget.core import (
    counts_from_labels, fit_beta_star, labels_from_counts, log_marginal_likelihood,
    per_state_conditional_entropy, per_state_log_likelihood, plugin_information,
    plugin_target_encoding, posterior_mean_conditional, qhat_y,
)
from mitarget.synthetic import generate_ground_truth, observe


def test_counts_roundtrip():
    x = np.array([0, 2, 2, 1, 0, 2])
    y = np.array([1, 0, 1, 1, 1, 0])
    n = counts_from_labels(x, y)
    assert n.shape == (3, 2)
    assert n.sum() == len(x)
    x2, y2 = labels_from_counts(n)
    assert np.array_equal(counts_from_labels(x2, y2, 3, 2), n)


def test_singleton_likelihood_is_beta_independent():
    qy = np.array([0.2, 0.5, 0.3])
    n = np.array([[0, 1, 0], [0, 0, 1]], dtype=float)
    for beta in (0.01, 1.0, 100.0):
        assert np.allclose(per_state_log_likelihood(beta, n, qy), np.log([0.5, 0.3]))


def test_binary_likelihood_matches_beta_binomial():
    # p(n_x | beta) = B(n1 + b q1, n0 + b q0) / B(b q1, b q0) for an ordered sequence.
    qy = np.array([0.7, 0.3])
    n = np.array([[3.0, 2.0], [0.0, 4.0]])
    beta = 1.7
    expected = [betaln(n0 + beta * qy[0], n1 + beta * qy[1]) - betaln(beta * qy[0], beta * qy[1])
                for n0, n1 in n]
    assert np.allclose(per_state_log_likelihood(beta, n, qy), expected)


def test_conditional_entropy_matches_numerical_integral():
    qy = np.array([0.4, 0.6])
    beta = 2.0
    n = np.array([[3.0, 1.0]])
    a, b = n[0, 0] + beta * qy[0], n[0, 1] + beta * qy[1]

    def integrand(q):
        h = -(q * np.log(q) + (1 - q) * np.log(1 - q))
        return h * q ** (a - 1) * (1 - q) ** (b - 1) / np.exp(betaln(a, b))

    expected, _ = integrate.quad(integrand, 0, 1)
    assert per_state_conditional_entropy(beta, n, qy)[0] == pytest.approx(expected, rel=1e-6)


def test_posterior_mean_is_smoothed_frequency():
    n = np.array([[2.0, 3.0], [0.0, 0.0]])
    qy = np.array([0.5, 0.5])
    enc = posterior_mean_conditional(n, 4.0, qy)
    assert np.allclose(enc[0], [(2 + 2) / 9, (3 + 2) / 9])
    assert np.allclose(enc[1], qy)
    assert np.allclose(enc.sum(axis=1), 1)


def test_plugin_encoding_fills_unseen_with_marginal():
    n = np.array([[1.0, 0.0], [0.0, 0.0], [1.0, 2.0]])
    enc = plugin_target_encoding(n)
    assert np.allclose(enc[0], [1, 0])
    assert np.allclose(enc[1], qhat_y(n))


def test_fit_beta_star_maximizes_likelihood():
    gt = generate_ground_truth(kx=400, ky=2, qx_kind="uniform", beta_range=(3.0, 3.0), seed=1)
    n = observe(gt, 3000, seed=2).astype(float)
    qy = qhat_y(n)
    b = fit_beta_star(n, qy)
    ll = log_marginal_likelihood(b, n, qy)
    assert ll >= log_marginal_likelihood(b * 1.1, n, qy)
    assert ll >= log_marginal_likelihood(b / 1.1, n, qy)
    assert 1.0 < b < 9.0  # the true beta is 3


@pytest.mark.parametrize("ky", [2, 3, 5])
def test_baseline_beats_plugin_on_mi(ky):
    gt = generate_ground_truth(kx=1000, ky=ky, qx_kind="uniform", beta_range=(1.0, 1.0), seed=3)
    n = observe(gt, 600, seed=4).astype(float)
    fit = fit_baseline(n)
    assert abs(fit.I_hat - gt.I_true) < abs(plugin_information(n) - gt.I_true)
    assert fit.target_encoding.shape == (1000, ky)
    assert np.allclose(fit.target_encoding.sum(axis=1), 1)
