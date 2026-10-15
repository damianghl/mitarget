"""
Hierarchical, CRP-clustered Dirichlet-Multinomial generative model for a pair
(X, Y), where X has many effective states and is severely undersampled, and Y
has few states and is well sampled.

Model
-----
    q_x                       : marginal over X            (kx,)  user-specified
    z_x  ~ CRP(alpha_crp)     : cluster label of state x,   x = 1..kx
    (beta_c, mu_c)            : per-cluster concentration & center, c = 1..K
    q_{y|x} | z_x = c  ~ Dirichlet(beta_c * mu_c)                  (ky,)
    n_x        ~ Multinomial(N, q_x)
    n_{x,:} | n_x, q_{y|x}  ~ Multinomial(n_x, q_{y|x})

K = 1 (e.g. alpha_crp -> 0) recovers the single-hyperparameter model of
Hernandez & Samengo (2019), Entropy 21(6):623, Eq. (13).

All randomness goes through an explicit numpy Generator for reproducibility.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Literal, Optional

from ..metrics import exact_mutual_information


# --------------------------------------------------------------------------
# Marginal q_x
# --------------------------------------------------------------------------

def sample_qx_dirichlet(kx: int, rng: np.random.Generator,
                        alpha: Optional[float] = None) -> np.ndarray:
    """
    q_x ~ Dirichlet(alpha/kx, ..., alpha/kx). Larger alpha -> more uniform qx.
    Defaults to alpha = kx, i.e. q_x uniform on the simplex.
    """
    alpha = float(kx) if alpha is None else alpha
    return rng.dirichlet(np.full(kx, alpha / kx))


def sample_qx_pitman_yor(kx: int, alpha: float, d: float, rng: np.random.Generator) -> np.ndarray:
    """
    Stick-breaking draw from a two-parameter Poisson-Dirichlet (Pitman-Yor)
    process (Pitman & Yor 1997), truncated to kx atoms and renormalized.
    Matches the qx ~ PY(d, alpha) used in Hernandez & Samengo Fig. 2a.
    """
    v = rng.beta(1.0 - d, alpha + d * np.arange(1, kx + 1))
    remaining = np.cumprod(1.0 - v)
    w = np.empty(kx)
    w[0] = v[0]
    w[1:] = v[1:] * remaining[:-1]
    w = w / w.sum()
    return rng.permutation(w)  # stick-breaking order is size-biased; shuffle it away


def sample_qx_uniform(kx: int) -> np.ndarray:
    return np.full(kx, 1.0 / kx)


QxKind = Literal["dirichlet", "pitman_yor", "uniform"]


def sample_qx(kx: int, kind: QxKind, rng: np.random.Generator, **kwargs) -> np.ndarray:
    """Dispatch to one of the q_x samplers above by name."""
    if kind == "dirichlet":
        return sample_qx_dirichlet(kx, rng=rng, **kwargs)
    if kind == "pitman_yor":
        return sample_qx_pitman_yor(kx, rng=rng, **kwargs)
    if kind == "uniform":
        return sample_qx_uniform(kx)
    raise ValueError(f"unknown qx_kind {kind!r}")


# --------------------------------------------------------------------------
# CRP partition of the kx states
# --------------------------------------------------------------------------

def sample_crp_partition(kx: int, alpha_crp: float, rng: np.random.Generator) -> np.ndarray:
    """
    Chinese Restaurant Process over kx items (order-exchangeable).
    Returns 0-indexed cluster labels z, shape (kx,).
    alpha_crp -> 0 gives (with high probability) a single cluster (K=1).
    """
    z = np.empty(kx, dtype=int)
    z[0] = 0
    sizes = [1]
    for i in range(1, kx):
        probs = np.array(sizes + [alpha_crp], dtype=float)
        probs /= probs.sum()
        c = rng.choice(len(probs), p=probs)
        if c == len(sizes):
            sizes.append(1)
        else:
            sizes[c] += 1
        z[i] = c
    return z


# --------------------------------------------------------------------------
# Per-cluster hyperparameters
# --------------------------------------------------------------------------

@dataclass
class ClusterParams:
    beta: np.ndarray  # (K,)      concentration per cluster
    mu: np.ndarray    # (K, ky)   center (target distribution) per cluster


def sample_cluster_params(K: int, ky: int,
                        beta_range: tuple[float, float],
                        rng: np.random.Generator,
                        center_mode: Literal["shared", "free"] = "shared",
                        shared_center: Optional[np.ndarray] = None,
                        center_concentration: float = 1.0) -> ClusterParams:
    """
    beta_c ~ log-uniform(beta_range)     (cf. the "uniform prior in information"
                                        motivation behind Eq. 19 of the paper)
    mu_c   = shared_center                                if center_mode == "shared"
             ~ Dirichlet(center_concentration * 1_ky)      if center_mode == "free"

    "shared": clusters differ only in spread (how tightly q_{y|x} hugs q_y) --
            the direct generalization of the paper's single-beta prior.
    "free":   clusters can also differ in location -- needed to represent e.g.
            their 3-delta example (peaks at 1/2, q0, 1-q0) as a mixture of
            well-fit Dirichlets rather than one badly-fit one.
    """
    log_beta = rng.uniform(np.log(beta_range[0]), np.log(beta_range[1]), size=K)
    beta = np.exp(log_beta)

    if center_mode == "shared":
        if shared_center is None:
            shared_center = np.full(ky, 1.0 / ky)
        mu = np.tile(shared_center, (K, 1))
    elif center_mode == "free":
        mu = rng.dirichlet(np.full(ky, center_concentration), size=K)
    else:
        raise ValueError(f"unknown center_mode {center_mode!r}")

    return ClusterParams(beta=beta, mu=mu)


def sample_conditional_probs(z: np.ndarray, cp: ClusterParams, rng: np.random.Generator) -> np.ndarray:
    """q_{y|x} ~ Dirichlet(beta_{z_x} * mu_{z_x}) independently for each x."""
    kx = len(z)
    ky = cp.mu.shape[1]
    q_ygivenx = np.empty((kx, ky))
    for x in range(kx):
        c = z[x]
        q_ygivenx[x] = rng.dirichlet(cp.beta[c] * cp.mu[c])
    return q_ygivenx


# --------------------------------------------------------------------------
# Sampling counts
# --------------------------------------------------------------------------

def sample_counts(qx: np.ndarray, N: int, rng: np.random.Generator) -> np.ndarray:
    """n_x ~ Multinomial(N, qx)."""
    return rng.multinomial(N, qx)


def sample_joint_counts(nx: np.ndarray,
                        q_ygivenx: np.ndarray,
                        rng: np.random.Generator) -> np.ndarray:
    """n_{x,y} ~ Multinomial(n_x, q_{y|x}), row by row. Returns (kx, ky) ints."""
    kx, ky = q_ygivenx.shape
    n = np.zeros((kx, ky), dtype=int)
    nz = np.nonzero(nx)[0]
    for x in nz:
        n[x] = rng.multinomial(nx[x], q_ygivenx[x])
    return n


# --------------------------------------------------------------------------
# Ground truth (fixed population) vs. observation (repeatable sampling)
#
# Kept as two separate steps, mirroring the paper's own evaluation protocol
# (Fig. 2a/2d): the true q_x, q_{y|x} are drawn ONCE, then many independent
# sets of N counts are drawn from that SAME fixed truth to study bias/variance.
# --------------------------------------------------------------------------

@dataclass
class GroundTruth:
    qx: np.ndarray
    q_ygivenx: np.ndarray
    z: np.ndarray
    cluster_params: ClusterParams
    I_true: float
    kx: int
    ky: int

    @property
    def K_true(self) -> int:
        return int(self.z.max()) + 1


def generate_ground_truth(kx: int, ky: int,
                        qx_kind: QxKind = "dirichlet", qx_kwargs: Optional[dict] = None,
                        alpha_crp: float = 1e-6,
                        beta_range: tuple[float, float] = (0.05, 50.0),
                        center_mode: Literal["shared", "free"] = "shared",
                        center_concentration: float = 1.0,
                        seed: Optional[int] = None) -> GroundTruth:
    """Draw the fixed, true (q_x, {q_{y|x}}) population once."""
    rng = np.random.default_rng(seed)
    qx_kwargs = qx_kwargs or {}

    qx = sample_qx(kx, qx_kind, rng, **qx_kwargs)

    z = sample_crp_partition(kx, alpha_crp, rng)
    K = int(z.max()) + 1

    cp = sample_cluster_params(
        K, ky, beta_range, rng,
        center_mode=center_mode,
        center_concentration=center_concentration,
    )

    q_ygivenx = sample_conditional_probs(z, cp, rng)
    I_true = exact_mutual_information(qx, q_ygivenx)

    return GroundTruth(qx=qx, q_ygivenx=q_ygivenx, z=z, cluster_params=cp,
                        I_true=I_true, kx=kx, ky=ky)


def generate_ground_truth_fixed_groups(kx: int, ky: int,
                                        proportions: list[float],
                                        betas: list[float],
                                        mus: Optional[list[np.ndarray]] = None,
                                        qx_kind: QxKind = "dirichlet",
                                        qx_kwargs: Optional[dict] = None,
                                        seed: Optional[int] = None) -> GroundTruth:
    """
    Controlled alternative to the CRP draw: exactly len(proportions) groups
    with specified (approximate) sizes and specified betas (and optionally
    specified centers mus; defaults to a shared center = uniform(1/ky) for all
    groups, i.e. groups differ only in spread). Useful for clean "does the
    fitted mixture recover known, well-separated clusters" tests, where a
    natural CRP draw's rich-get-richer tail would be a confound.
    """
    if abs(sum(proportions) - 1.0) > 1e-8:
        raise ValueError("proportions must sum to 1")
    rng = np.random.default_rng(seed)
    qx_kwargs = qx_kwargs or {}

    qx = sample_qx(kx, qx_kind, rng, **qx_kwargs)

    K = len(proportions)
    counts_per_group = np.round(np.array(proportions) * kx).astype(int)
    counts_per_group[-1] = kx - counts_per_group[:-1].sum()  # fix rounding
    z = np.repeat(np.arange(K), counts_per_group)
    rng.shuffle(z)

    beta_arr = np.asarray(betas, dtype=float)
    if mus is None:
        mu_arr = np.tile(np.full(ky, 1.0 / ky), (K, 1))
    else:
        mu_arr = np.asarray(mus, dtype=float)
    cp = ClusterParams(beta=beta_arr, mu=mu_arr)

    q_ygivenx = sample_conditional_probs(z, cp, rng)
    I_true = exact_mutual_information(qx, q_ygivenx)

    return GroundTruth(qx=qx, q_ygivenx=q_ygivenx, z=z, cluster_params=cp,
                        I_true=I_true, kx=kx, ky=ky)


def observe(gt: GroundTruth, N: int, seed: Optional[int] = None) -> np.ndarray:
    """Draw one (kx, ky) count matrix n_{xy} of N total samples from a fixed truth."""
    rng = np.random.default_rng(seed)
    nx = sample_counts(gt.qx, N, rng)
    return sample_joint_counts(nx, gt.q_ygivenx, rng)


@dataclass
class SimulatedDataset:
    n: np.ndarray                  # (kx, ky) joint counts -- THE DATA
    qx_true: np.ndarray
    q_ygivenx_true: np.ndarray
    z_true: np.ndarray
    cluster_params: ClusterParams
    I_true: float
    N: int
    kx: int
    ky: int

    @property
    def K_true(self) -> int:
        return int(self.z_true.max()) + 1


def simulate(kx: int, ky: int, N: int,
            qx_kind: QxKind = "dirichlet",
            qx_kwargs: Optional[dict] = None,
            alpha_crp: float = 1e-6,
            beta_range: tuple[float, float] = (0.05, 50.0),
            center_mode: Literal["shared", "free"] = "shared",
            center_concentration: float = 1.0,
            seed: Optional[int] = None) -> SimulatedDataset:
    """
    Convenience one-shot wrapper: draw a ground truth AND one observation of
    it in a single call (uses the same seed for both steps). For bias/variance
    studies across repeated draws of n at fixed truth, use generate_ground_truth()
    + observe() directly instead.

    alpha_crp is the CRP concentration governing the *partition of x-states*
    (not to be confused with beta, the per-cluster Dirichlet concentration).
    alpha_crp ~ 1e-6 (default) collapses to K=1 w.h.p., i.e. exactly the
    Hernandez & Samengo (2019) generative assumption.
    """
    gt = generate_ground_truth(kx, ky, qx_kind, qx_kwargs, alpha_crp, beta_range,
                                center_mode, center_concentration, seed)
    n = observe(gt, N, seed)
    return SimulatedDataset(
        n=n, qx_true=gt.qx, q_ygivenx_true=gt.q_ygivenx, z_true=gt.z,
        cluster_params=gt.cluster_params, I_true=gt.I_true, N=N, kx=kx, ky=ky,
    )
