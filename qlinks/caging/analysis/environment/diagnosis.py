from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from qlinks.caging.analysis.environment.contracts import (
    EnvironmentReductionConfig,
)
from qlinks.caging.analysis.environment.discovery import (
    _active_frontier_zero_indices,
    _find_nontrivial_interference_zeros,
    _find_trivial_zero_indices,
    _resolve_environment_domain_mask,
)
from qlinks.caging.analysis.environment.mechanisms import (
    _annotate_collective_cancellations,
    _annotate_probe_mechanisms,
)
from qlinks.caging.analysis.environment.monitor import (
    _reduced_iz_region_variables_from_supports,
    reduced_iz_component_groups_from_reports,
    reduced_iz_probe_support_from_report,
    select_reduced_iz_monitor_reports_from_zero_reports,
)
from qlinks.caging.analysis.environment.operator import (
    _build_config_to_index,
    _ReducedLocalOperatorApplicationContext,
)
from qlinks.caging.analysis.environment.report import (
    EnvironmentReductionReport,
    _union_projector_like_annihilated_inputs,
    _zero_indices_with_indirect_projector_like,
    _zero_indices_with_mechanism,
    _zero_indices_with_nonzero_complement_action_failure,
    _zero_indices_with_source_projector_like,
    _zero_indices_with_unexpected_target_failure,
)
from qlinks.caging.analysis.environment.summary import (
    _environment_removal_summary,
    _safe_max,
    _safe_mean,
)
from qlinks.caging.analysis.transitions import transition_pattern_key
from qlinks.caging.results import CageState, cage_state_to_full_vector


def diagnose_cage_environment_reduction(
    cage_state: CageState,
    *,
    kinetic_matrix: sp.spmatrix | sp.sparray | NDArray,
    basis_configs: NDArray[np.integer],
    hilbert_size: int | None = None,
    sector_mask: NDArray[np.bool_] | None = None,
    config: EnvironmentReductionConfig | None = None,
) -> EnvironmentReductionReport:
    """Diagnose exterior-environment reduction for one compact cage state.

    Args:
        cage_state: Compact cage state returned by the caging solver.
        kinetic_matrix: Off-diagonal Hamiltonian or kinetic matrix used to
            identify interference zeros and local ``Z_h`` patterns.
        basis_configs: Integer array with shape ``(n_basis, n_variables)``.
            Rows are product-state configurations in the global constrained
            basis.
        hilbert_size: Full Hilbert-space dimension.  Defaults to
            ``basis_configs.shape[0]``.
        sector_mask: Optional mask selecting the sector used for local
            diagnostics.
        config: Numerical environment-reduction parameters.

    Returns:
        Environment-reduction report describing whether exterior degrees of
        freedom are safely removable and the mechanism used by each probe.
    """
    if config is None:
        config = EnvironmentReductionConfig()

    if hilbert_size is None:
        hilbert_size = int(basis_configs.shape[0])

    full_state = cage_state_to_full_vector(
        cage_state,
        hilbert_size=hilbert_size,
    )

    return diagnose_environment_reduction(
        full_state,
        kinetic_matrix=kinetic_matrix,
        basis_configs=basis_configs,
        sector_mask=sector_mask,
        config=config,
        metadata={
            "energy": cage_state.energy,
            "support_size": cage_state.support_size,
            "boundary_residual": cage_state.boundary_residual,
            "eigen_residual": cage_state.eigen_residual,
            "full_residual": cage_state.full_residual,
        },
    )


def diagnose_environment_reduction(
    full_state: NDArray[np.complex128],
    *,
    kinetic_matrix: sp.spmatrix | sp.sparray | NDArray,
    basis_configs: NDArray[np.integer],
    sector_mask: NDArray[np.bool_] | None = None,
    config: EnvironmentReductionConfig | None = None,
    metadata: dict[str, object] | None = None,
) -> EnvironmentReductionReport:
    """Diagnose exterior-environment reduction for a full Hilbert-space vector."""
    if config is None:
        config = EnvironmentReductionConfig()

    full_state = np.asarray(full_state, dtype=np.complex128)
    basis_configs = np.asarray(basis_configs)

    if basis_configs.ndim != 2:
        raise ValueError("basis_configs must have shape (n_basis, n_variables).")

    hilbert_size = int(full_state.size)
    if basis_configs.shape[0] != hilbert_size:
        raise ValueError("basis_configs.shape[0] must match full_state.size.")

    kinetic_csr = sp.csr_array(kinetic_matrix)

    support_mask = np.abs(full_state) > config.amplitude_tolerance
    support_size = int(np.count_nonzero(support_mask))
    support_fraction = support_size / float(hilbert_size)

    active_state_indices = np.flatnonzero(support_mask).astype(np.int64, copy=False)

    domain_mask = _resolve_environment_domain_mask(
        kinetic_csr,
        support_mask=support_mask,
        sector_mask=sector_mask,
        config=config,
    )

    active_domain_indices = active_state_indices[domain_mask[active_state_indices]].astype(
        np.int64,
        copy=False,
    )

    config_to_index = _build_config_to_index(basis_configs)
    local_operator_contexts: dict[
        tuple[int, ...],
        _ReducedLocalOperatorApplicationContext,
    ] = {}

    active_frontier_zero_indices = _active_frontier_zero_indices(
        kinetic_csr,
        support_mask=support_mask,
        domain_mask=domain_mask,
        active_state_indices=active_domain_indices,
    )

    zero_reports = _find_nontrivial_interference_zeros(
        full_state,
        kinetic_csr,
        basis_configs=basis_configs,
        config_to_index=config_to_index,
        support_mask=support_mask,
        domain_mask=domain_mask,
        active_frontier_zero_indices=active_frontier_zero_indices,
        active_state_indices=active_state_indices,
        active_domain_indices=active_domain_indices,
        local_operator_contexts=local_operator_contexts,
        config=config,
    )

    trivial_zero_indices = _find_trivial_zero_indices(
        full_state,
        kinetic_csr,
        support_mask=support_mask,
        domain_mask=domain_mask,
        active_frontier_zero_indices=active_frontier_zero_indices,
    )

    zero_reports = _annotate_probe_mechanisms(
        zero_reports,
        trivial_zero_indices=trivial_zero_indices,
        config=config,
    )

    zero_reports, collective_cancellation_reports = _annotate_collective_cancellations(
        zero_reports,
        full_state=full_state,
        basis_configs=basis_configs,
        config_to_index=config_to_index,
        domain_mask=domain_mask,
        active_domain_indices=active_domain_indices,
        local_operator_contexts=local_operator_contexts,
        config=config,
    )

    pattern_keys = {transition_pattern_key(report.local_transitions) for report in zero_reports}

    q_weights = np.array(
        [report.q_sector_weight for report in zero_reports],
        dtype=float,
    )
    reduced_norms = np.array(
        [report.reduced_action_norm for report in zero_reports],
        dtype=float,
    )
    complement_norms = np.array(
        [report.complement_action_norm for report in zero_reports],
        dtype=float,
    )
    n_complement_targets = sum(report.n_complement_targets for report in zero_reports)
    n_unexplained_complement_targets = sum(
        report.n_unexplained_complement_targets for report in zero_reports
    )
    n_trivial_targets = sum(report.n_trivial_targets for report in zero_reports)
    n_same_pattern_iz_targets = sum(report.n_same_pattern_iz_targets for report in zero_reports)
    n_projector_like_iz_targets = sum(report.n_projector_like_iz_targets for report in zero_reports)
    n_unexpected_targets = sum(report.n_unexpected_targets for report in zero_reports)
    unexpected_target_probe_failure_indices = _zero_indices_with_unexpected_target_failure(
        zero_reports
    )
    nonzero_complement_action_probe_failure_indices = (
        _zero_indices_with_nonzero_complement_action_failure(zero_reports)
    )
    q_empty_source_zero_indices = _zero_indices_with_mechanism(
        zero_reports,
        "q_empty",
    )
    same_pattern_zero_closure_indices = _zero_indices_with_mechanism(
        zero_reports,
        "closed_by_same_pattern_zeros",
    )
    source_projector_like_probe_indices = _zero_indices_with_source_projector_like(zero_reports)
    indirect_projector_like_probe_indices = _zero_indices_with_indirect_projector_like(zero_reports)
    projector_like_annihilated_input_indices = _union_projector_like_annihilated_inputs(
        zero_reports
    )
    # domain_blocked_source_zero_indices = _zero_indices_with_mechanism(
    #     zero_reports,
    #     "domain_blocked",
    # )  # noqa: F841
    projector_like_source_zero_indices = _zero_indices_with_mechanism(
        zero_reports,
        "projector_like",
    )
    collective_cancellation_source_zero_indices = _zero_indices_with_mechanism(
        zero_reports,
        "collective_cancellation",
    )
    invalid_source_zero_indices = _zero_indices_with_mechanism(
        zero_reports,
        "unexplained_leakage",
    )
    if len(zero_reports) == 0:
        fraction_removable = 0.0
    else:
        fraction_removable = float(np.mean([report.is_safely_removable for report in zero_reports]))

    reduced_iz_probe_supports = tuple(
        reduced_iz_probe_support_from_report(report) for report in zero_reports
    )
    reduced_iz_region_variable_indices = _reduced_iz_region_variables_from_supports(
        reduced_iz_probe_supports
    )
    default_reduced_iz_reports = select_reduced_iz_monitor_reports_from_zero_reports(
        tuple(zero_reports)
    )
    reduced_iz_monitor_component_groups = {
        decomposition: reduced_iz_component_groups_from_reports(
            default_reduced_iz_reports,
            decomposition=decomposition,
        )
        for decomposition in ("single_sum", "exact_support", "connected_support")
    }

    removal_summary = _environment_removal_summary(tuple(zero_reports))

    metadata = {} if metadata is None else dict(metadata)
    metadata.setdefault(
        "environment_domain_size",
        int(np.count_nonzero(domain_mask)),
    )
    metadata.setdefault(
        "environment_domain_fraction",
        float(np.count_nonzero(domain_mask)) / float(hilbert_size),
    )
    metadata.setdefault("sector_policy", config.sector_policy)

    return EnvironmentReductionReport(
        support_size=support_size,
        hilbert_size=hilbert_size,
        support_fraction=support_fraction,
        n_nontrivial_zeros=len(zero_reports),
        n_distinct_local_patterns=len(pattern_keys),
        n_complement_targets=n_complement_targets,
        n_unexplained_complement_targets=n_unexplained_complement_targets,
        fraction_probes_safely_removable=fraction_removable,
        n_q_empty_source_probes=int(q_empty_source_zero_indices.size),
        n_same_pattern_zero_closure_probes=int(same_pattern_zero_closure_indices.size),
        n_projector_like_source_probes=int(projector_like_source_zero_indices.size),
        n_invalid_source_probes=int(invalid_source_zero_indices.size),
        q_empty_source_zero_indices=q_empty_source_zero_indices,
        same_pattern_zero_closure_indices=(same_pattern_zero_closure_indices),
        projector_like_source_zero_indices=projector_like_source_zero_indices,
        n_collective_cancellation_source_probes=int(
            collective_cancellation_source_zero_indices.size
        ),
        collective_cancellation_source_zero_indices=(collective_cancellation_source_zero_indices),
        collective_cancellation_reports=collective_cancellation_reports,
        invalid_source_zero_indices=invalid_source_zero_indices,
        n_trivial_targets=n_trivial_targets,
        n_same_pattern_iz_targets=n_same_pattern_iz_targets,
        n_projector_like_iz_targets=n_projector_like_iz_targets,
        n_unexpected_targets=n_unexpected_targets,
        n_unexpected_target_probe_failures=int(unexpected_target_probe_failure_indices.size),
        n_nonzero_complement_action_probe_failures=int(
            nonzero_complement_action_probe_failure_indices.size
        ),
        unexpected_target_probe_failure_indices=(unexpected_target_probe_failure_indices),
        nonzero_complement_action_probe_failure_indices=(
            nonzero_complement_action_probe_failure_indices
        ),
        n_source_projector_like_probes=int(source_projector_like_probe_indices.size),
        n_indirect_projector_like_probes=int(indirect_projector_like_probe_indices.size),
        n_projector_like_annihilated_inputs=int(projector_like_annihilated_input_indices.size),
        source_projector_like_probe_indices=source_projector_like_probe_indices,
        indirect_projector_like_probe_indices=(indirect_projector_like_probe_indices),
        projector_like_annihilated_input_indices=(projector_like_annihilated_input_indices),
        mean_q_sector_weight=_safe_mean(q_weights),
        max_q_sector_weight=_safe_max(q_weights),
        mean_reduced_action_norm=_safe_mean(reduced_norms),
        max_reduced_action_norm=_safe_max(reduced_norms),
        mean_complement_action_norm=_safe_mean(complement_norms),
        max_complement_action_norm=_safe_max(complement_norms),
        zero_reports=tuple(zero_reports),
        reduced_iz_probe_supports=reduced_iz_probe_supports,
        reduced_iz_region_variable_indices=reduced_iz_region_variable_indices,
        reduced_iz_monitor_component_groups=reduced_iz_monitor_component_groups,
        removal_summary=removal_summary,
        metadata=metadata,
    )
