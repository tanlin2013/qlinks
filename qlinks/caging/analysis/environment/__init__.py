"""Exterior-environment reduction diagnostics for bounded local caging operators."""

from qlinks.caging.analysis.environment.contracts import (
    CollectiveCancellationMode,
    CollectiveCancellationReport,
    EnvironmentProbeDetailLabel,
    EnvironmentReductionConfig,
    EnvironmentRemovalMechanismLabel,
    EnvironmentRemovalProbeReport,
    EnvironmentRemovalSummary,
    ReducedIZMonitorComponentGroup,
    ReducedIZMonitorDecomposition,
    ReducedIZProbeSupport,
    SectorPolicy,
)
from qlinks.caging.analysis.environment.diagnosis import (
    diagnose_cage_environment_reduction,
    diagnose_environment_reduction,
)
from qlinks.caging.analysis.environment.monitor import (
    group_reduced_iz_monitor_reports,
    group_reduced_iz_reports_by_connected_support,
    group_reduced_iz_reports_by_exact_support,
    reduced_iz_component_groups_from_reports,
    reduced_iz_probe_support_from_report,
    select_reduced_iz_monitor_reports,
    select_reduced_iz_monitor_reports_from_zero_reports,
)
from qlinks.caging.analysis.environment.report import EnvironmentReductionReport
from qlinks.caging.analysis.environment.support import (
    support_key_for_zero_report,
    support_key_from_mask,
)
from qlinks.caging.analysis.transitions import LocalTransitionPattern

__all__ = [
    "CollectiveCancellationMode",
    "CollectiveCancellationReport",
    "EnvironmentProbeDetailLabel",
    "EnvironmentReductionConfig",
    "EnvironmentReductionReport",
    "EnvironmentRemovalMechanismLabel",
    "EnvironmentRemovalProbeReport",
    "EnvironmentRemovalSummary",
    "LocalTransitionPattern",
    "ReducedIZMonitorComponentGroup",
    "ReducedIZMonitorDecomposition",
    "ReducedIZProbeSupport",
    "SectorPolicy",
    "diagnose_cage_environment_reduction",
    "diagnose_environment_reduction",
    "group_reduced_iz_monitor_reports",
    "group_reduced_iz_reports_by_connected_support",
    "group_reduced_iz_reports_by_exact_support",
    "reduced_iz_component_groups_from_reports",
    "reduced_iz_probe_support_from_report",
    "select_reduced_iz_monitor_reports",
    "select_reduced_iz_monitor_reports_from_zero_reports",
    "support_key_for_zero_report",
    "support_key_from_mask",
]
