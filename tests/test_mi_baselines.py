import numpy as np
from scipy.special import xlogy

from mitarget.mi_baselines import (
    entropy_chao_shen, entropy_miller_madow, entropy_nsb, mi_chao_shen, mi_miller_madow, mi_nsb,
)


def _h(p):
    return float(-xlogy(p, p).sum())


def test_entropies_on_well_sampled_data_agree_with_plugin():
    c = np.array([4000, 3000, 2000, 1000])
    h = _h(c / c.sum())
    for est in (entropy_miller_madow, entropy_chao_shen):
        assert abs(est(c) - h) < 1e-3
    assert abs(entropy_nsb(c, 4) - h) < 1e-3


def test_undersampled_uniform_bias_order():
    rng = np.random.default_rng(1)
    c = np.bincount(rng.integers(1000, size=500), minlength=1000)
    true = np.log(1000)
    plug = _h(c[c > 0] / c.sum())
    assert plug < entropy_miller_madow(c) < true + 0.3
    assert abs(entropy_nsb(c, 1000) - true) < 0.1


def test_mi_independent_table_is_near_zero():
    rng = np.random.default_rng(2)
    x = rng.integers(50, size=20000)
    y = rng.random(20000) < 0.3
    n = np.zeros((50, 2))
    np.add.at(n, (x, y.astype(int)), 1)
    for v in (mi_miller_madow(n), mi_chao_shen(n), mi_nsb(n)):
        assert abs(v) < 0.005
