"""
The synthetic scenarios used to compare estimators.

1. Matched, single beta (the paper's Fig. 2a/2d setup).
2. Matched, two true clusters with different beta.
3. Mismatch: the paper's spherical-harmonic example (289 distinct true values).
4. Mismatch: "reliable vs. junk" states. Junk states have q(y=1|x) = 1/2; the
   informative ones have q(y=1|x) = q0 or 1 - q0. Available both as the paper's
   parity construction and as a direct mixture with a tunable junk fraction.
"""

from __future__ import annotations

from importlib import resources

import numpy as np

from ..metrics import exact_mutual_information
from .generative import (
    ClusterParams, GroundTruth, generate_ground_truth, generate_ground_truth_fixed_groups,
)


def scenario_single_beta(kx: int = 1000, beta: float = 2.3, seed: int = 21) -> GroundTruth:
    """Scenario 1: Pitman-Yor(50, 0.55) q_x and one shared beta for every state."""
    return generate_ground_truth(kx=kx, ky=2, qx_kind="pitman_yor",
                                 qx_kwargs=dict(alpha=50, d=0.55), alpha_crp=1e-6,
                                 beta_range=(beta, beta), center_mode="shared", seed=seed)


def scenario_two_clusters(kx: int = 1500, betas: tuple[float, float] = (0.2, 10.0),
                          seed: int = 11) -> GroundTruth:
    """Scenario 2: two equal-sized clusters, strongly (0.2) and weakly (10) dependent."""
    return generate_ground_truth_fixed_groups(kx=kx, ky=2, proportions=[0.5, 0.5],
                                              betas=list(betas), qx_kind="uniform", seed=seed)


def scenario_sphere() -> GroundTruth:
    """Scenario 3: the paper's spherical-symmetry example (kx = 4096, uniform q_x)."""
    text = resources.files(__package__).joinpath("data/prob10.dat").read_text()
    p1 = np.loadtxt(text.splitlines())
    kx = len(p1)
    qx = np.full(kx, 1.0 / kx)
    q_ygivenx = np.column_stack([1 - p1, p1])
    return GroundTruth(qx=qx, q_ygivenx=q_ygivenx, z=np.zeros(kx, dtype=int),
                       cluster_params=ClusterParams(beta=np.array([1.0]), mu=np.array([[0.5, 0.5]])),
                       I_true=exact_mutual_information(qx, q_ygivenx), kx=kx, ky=2)


def _three_level_truth(q1: np.ndarray, z: np.ndarray) -> GroundTruth:
    kx = len(q1)
    qx = np.full(kx, 1.0 / kx)
    q_ygivenx = np.column_stack([1 - q1, q1])
    return GroundTruth(qx=qx, q_ygivenx=q_ygivenx, z=z,
                       cluster_params=ClusterParams(beta=np.array([1.0]), mu=np.array([[0.5, 0.5]])),
                       I_true=exact_mutual_information(qx, q_ygivenx), kx=kx, ky=2)


def scenario_parity(kx: int = 2500, D: int = 40, p: float = 0.05, q0: float = 0.1,
                    seed: int | None = 0) -> GroundTruth:
    """
    Scenario 4, parity construction (the paper's third example): X is a D-bit
    Bernoulli(p) vector; q(y=1|x) = 1/2 for even bit-sum, else q0 or 1 - q0 by
    a fair coin fixed per state. ``z`` is the parity (0 = junk, 1 = informative).
    """
    rng = np.random.default_rng(seed)
    parity = rng.binomial(1, p, size=(kx, D)).sum(axis=1) % 2
    coin = rng.integers(0, 2, size=kx)
    q1 = np.where(parity == 0, 0.5, np.where(coin == 0, q0, 1 - q0))
    return _three_level_truth(q1.astype(float), parity.astype(int))


def scenario_junk_mixture(kx: int = 2500, junk_fraction: float = 0.5,
                          q0: float | None = None, seed: int | None = 0) -> GroundTruth:
    """
    Scenario 4, direct mixture form: a fraction ``junk_fraction`` of states has
    q(y=1|x) = 1/2, and the rest split evenly between q0 and 1 - q0. With
    ``q0=None`` it is drawn from U[0, 1]. ``z`` labels the branch:
    0 = junk, 1 = q0, 2 = 1 - q0.
    """
    rng = np.random.default_rng(seed)
    if q0 is None:
        q0 = float(rng.uniform())
    pi_inf = (1.0 - junk_fraction) / 2
    z = rng.choice(3, size=kx, p=[junk_fraction, pi_inf, pi_inf])
    q1 = np.choose(z, [0.5, q0, 1 - q0]).astype(float)
    return _three_level_truth(q1, z)
