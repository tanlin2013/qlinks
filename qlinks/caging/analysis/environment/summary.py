from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from qlinks.caging.analysis.environment.contracts import (
    EnvironmentRemovalProbeReport,
    EnvironmentRemovalSummary,
)


def _environment_removal_summary(
    zero_reports: tuple[EnvironmentRemovalProbeReport, ...] | list[EnvironmentRemovalProbeReport],
) -> EnvironmentRemovalSummary:
    """Summarize the three supported exterior-removal mechanisms."""
    counts = {
        "no_environment_weight": 0,
        "projective_annihilation": 0,
        "same_local_cancellation_pattern": 0,
        "unsafe": 0,
    }
    for report in zero_reports:
        counts[report.removal_mechanism] += 1

    return EnvironmentRemovalSummary(
        n_no_environment_weight_probes=counts["no_environment_weight"],
        n_projective_annihilation_probes=counts["projective_annihilation"],
        n_same_local_cancellation_pattern_probes=(counts["same_local_cancellation_pattern"]),
        n_unsafe_probes=counts["unsafe"],
        n_projector_like_iz_targets=sum(
            report.n_projector_like_iz_targets for report in zero_reports
        ),
        n_unexpected_targets=sum(report.n_unexpected_targets for report in zero_reports),
        n_nonzero_complement_action_failures=sum(
            report.has_nonzero_complement_action for report in zero_reports
        ),
    )


def _safe_mean(values: NDArray[np.float64]) -> float:
    if values.size == 0:
        return 0.0
    return float(np.mean(values))


def _safe_max(values: NDArray[np.float64]) -> float:
    if values.size == 0:
        return 0.0
    return float(np.max(values))
