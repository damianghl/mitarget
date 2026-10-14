"""Synthetic ground truths, scenarios and Bayes oracles."""

from .generative import (
    ClusterParams, GroundTruth, SimulatedDataset, generate_ground_truth,
    generate_ground_truth_fixed_groups, observe, simulate,
)
from .oracles import (
    PARITY_PRIORS, fit_q0_star, oracle_known_clusters_encoding, oracle_three_point_encoding,
    three_point_log_marginal_likelihood,
)
from .scenarios import (
    scenario_junk_mixture, scenario_parity, scenario_single_beta, scenario_sphere,
    scenario_two_clusters,
)

__all__ = [
    "ClusterParams", "GroundTruth", "SimulatedDataset", "generate_ground_truth",
    "generate_ground_truth_fixed_groups", "observe", "simulate",
    "PARITY_PRIORS", "fit_q0_star", "oracle_known_clusters_encoding",
    "oracle_three_point_encoding", "three_point_log_marginal_likelihood",
    "scenario_junk_mixture", "scenario_parity", "scenario_single_beta", "scenario_sphere",
    "scenario_two_clusters",
]
