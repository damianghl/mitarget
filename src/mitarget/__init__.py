"""
mitarget: empirical-Bayes target encoding and mutual-information estimation
for large, under-sampled categorical features.

Built on Hernandez & Samengo (2019), "Estimating the mutual information between
two discrete, asymmetric variables with limited samples", Entropy 21(6):623.
"""

from .baseline import BaselineFit, doubleton_rates, doubleton_test, fit_baseline, singleton_rate
from .competitors import (
    GLMMFit, NPMLEFit, catboost_classifier_encoding, catboost_ts_encoding, fit_glmm, fit_npmle,
    ordered_target_statistics,
)
from .core import counts_from_labels, fixed_m_encoding, plugin_information, plugin_target_encoding, qhat_y
from .crp import CRPFit, fit_crp_mixture
from .encoders import (
    METHODS, BayesianCategoricalClassifier, BayesianTargetEncoder, EncodingFit, fit_counts,
    mutual_information,
)
from .finite_mixture import FiniteMixtureFit, bic, fit_finite_mixture
from .freq_prior import FreqPriorFit, fit_freq_prior
from .metrics import excess_cross_entropy, exact_mutual_information, heldout_log_loss
from .mi_interval import (
    CredibleInterval, MIInterval, credible_interval, identified_interval, mi_interval, moment_estimates,
    worst_case_halfwidth,
)

__version__ = "0.1.0"

__all__ = [
    "BaselineFit", "doubleton_rates", "doubleton_test", "fit_baseline", "singleton_rate",
    "counts_from_labels", "fixed_m_encoding", "plugin_information", "plugin_target_encoding",
    "qhat_y",
    "GLMMFit", "NPMLEFit", "catboost_classifier_encoding", "catboost_ts_encoding", "fit_glmm",
    "fit_npmle", "ordered_target_statistics",
    "CRPFit", "fit_crp_mixture",
    "METHODS", "BayesianCategoricalClassifier", "BayesianTargetEncoder", "EncodingFit",
    "fit_counts", "mutual_information",
    "FiniteMixtureFit", "bic", "fit_finite_mixture",
    "FreqPriorFit", "fit_freq_prior",
    "excess_cross_entropy", "exact_mutual_information", "heldout_log_loss",
    "CredibleInterval", "MIInterval", "credible_interval", "identified_interval", "mi_interval",
    "moment_estimates", "worst_case_halfwidth",
]
