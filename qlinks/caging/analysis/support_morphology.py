"""Support-morphology diagnostics for caged-state analysis.

These finite-size descriptors are intentionally separate from exterior-environment
reduction. They characterize where a state has support; they do not classify a
caged eigenstate or certify that an exterior environment can be removed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from qlinks.caging.analysis.environment import (
    ReducedIZMonitorComponentGroup,
    ReducedIZMonitorDecomposition,
)

FockSupportMorphologyLabel: TypeAlias = Literal[
    "unknown",
    "finite_size_empty",
    "finite_size_singleton",
    "finite_size_sector_sparse",
    "finite_size_sector_dense",
    "finite_size_shell_sparse",
    "finite_size_shell_dense",
]
RealSpaceSupportMorphologyLabel: TypeAlias = Literal[
    "unknown",
    "frozen",
    "partially_active",
    "fully_active",
]


@dataclass(frozen=True, slots=True)
class SupportMorphologyConfig:
    """Numerical settings for finite-size support-morphology analysis."""

    amplitude_tolerance: float = 1e-10
    action_tolerance: float = 1e-9
    fock_dense_fraction_threshold: float = 0.5
    potential_shell_tolerance: float | None = None


@dataclass(frozen=True, slots=True)
class FockSupportMorphology:
    """Finite-size morphology diagnostics for the support in Fock space.

    The ``label`` is a finite-size proxy only.  Scaling labels such as
    finite, polynomial, or shell-extended require comparing a family of
    systems across sizes.
    """

    label: FockSupportMorphologyLabel = "unknown"
    support_size: int = 0
    effective_support_size: float = 0.0
    hilbert_size: int = 0
    support_fraction: float = 0.0
    effective_hilbert_fraction: float = 0.0
    boundary_size: int = 0
    boundary_to_support_ratio: float = 0.0
    support_internal_matrix_entries: int = 0
    potential_shell_value: complex | None = None
    potential_shell_size: int | None = None
    support_shell_fraction: float | None = None
    effective_shell_fraction: float | None = None
    potential_shell_residual: float | None = None

    @property
    def has_potential_shell(self) -> bool:
        return self.potential_shell_size is not None


@dataclass(frozen=True, slots=True)
class RealSpaceSupportMorphology:
    """Finite-size morphology diagnostics in the microscopic variable space.

    The variable indices are model-layout indices.  Connectivity, diameter,
    and winding/wrapping require lattice adjacency metadata and are therefore
    left to higher-level lattice-aware helpers.
    """

    label: RealSpaceSupportMorphologyLabel = "unknown"
    n_variables: int = 0
    active_variable_indices: tuple[int, ...] = ()
    active_variable_count: int = 0
    active_variable_fraction: float = 0.0
    frozen_variable_count: int = 0
    reduced_iz_region_variable_indices: tuple[int, ...] = ()
    reduced_iz_region_variable_count: int = 0
    reduced_iz_region_variable_fraction: float = 0.0
    exact_support_component_count: int = 0
    exact_support_component_sizes: tuple[int, ...] = ()
    connected_support_component_count: int = 0
    connected_support_component_sizes: tuple[int, ...] = ()


def analyze_fock_support_morphology(
    *,
    full_state: NDArray[np.complex128],
    kinetic_csr: sp.csr_array,
    support_mask: NDArray[np.bool_],
    active_frontier_zero_indices: NDArray[np.int64],
    potential_diagonal: NDArray | None,
    config: SupportMorphologyConfig,
) -> FockSupportMorphology:
    """Return finite-size Fock-space support morphology diagnostics."""
    support_indices = np.flatnonzero(support_mask)
    support_size = int(support_indices.size)
    hilbert_size = int(full_state.size)
    support_fraction = support_size / float(hilbert_size) if hilbert_size else 0.0

    weights = np.abs(full_state) ** 2
    state_norm_sq = float(np.sum(weights))
    weights_fourth_sum = float(np.sum(weights**2))
    if state_norm_sq > 0.0 and weights_fourth_sum > 0.0:
        effective_support_size = (state_norm_sq * state_norm_sq) / weights_fourth_sum
    else:
        effective_support_size = 0.0
    effective_hilbert_fraction = (
        effective_support_size / float(hilbert_size) if hilbert_size else 0.0
    )

    boundary_size = int(active_frontier_zero_indices.size)
    boundary_to_support_ratio = boundary_size / float(support_size) if support_size else 0.0
    if support_size:
        support_internal_matrix_entries = int(
            kinetic_csr[support_indices, :][:, support_indices].nnz
        )
    else:
        support_internal_matrix_entries = 0

    potential_shell_value: complex | None = None
    potential_shell_size: int | None = None
    support_shell_fraction: float | None = None
    effective_shell_fraction: float | None = None
    potential_shell_residual: float | None = None

    if potential_diagonal is not None:
        diagonal = np.asarray(potential_diagonal, dtype=np.complex128).reshape(-1)
        if diagonal.size != hilbert_size:
            raise ValueError("potential_diagonal must have length full_state.size.")
        if state_norm_sq > 0.0:
            potential_shell_value = complex(
                np.vdot(full_state, diagonal * full_state) / state_norm_sq
            )
            residual_vector = (diagonal - potential_shell_value) * full_state
            potential_shell_residual = float(np.linalg.norm(residual_vector))
            shell_tolerance = (
                float(config.potential_shell_tolerance)
                if config.potential_shell_tolerance is not None
                else max(float(config.action_tolerance), float(config.amplitude_tolerance))
            )
            if potential_shell_residual <= shell_tolerance:
                shell_mask = np.abs(diagonal - potential_shell_value) <= shell_tolerance
                potential_shell_size = int(np.count_nonzero(shell_mask))
                if potential_shell_size > 0:
                    support_shell_fraction = support_size / float(potential_shell_size)
                    effective_shell_fraction = effective_support_size / float(potential_shell_size)

    if support_size == 0:
        label: FockSupportMorphologyLabel = "finite_size_empty"
    elif support_size == 1:
        label = "finite_size_singleton"
    elif effective_shell_fraction is not None:
        if effective_shell_fraction >= config.fock_dense_fraction_threshold:
            label = "finite_size_shell_dense"
        else:
            label = "finite_size_shell_sparse"
    elif effective_hilbert_fraction >= config.fock_dense_fraction_threshold:
        label = "finite_size_sector_dense"
    else:
        label = "finite_size_sector_sparse"

    return FockSupportMorphology(
        label=label,
        support_size=support_size,
        effective_support_size=float(effective_support_size),
        hilbert_size=hilbert_size,
        support_fraction=float(support_fraction),
        effective_hilbert_fraction=float(effective_hilbert_fraction),
        boundary_size=boundary_size,
        boundary_to_support_ratio=float(boundary_to_support_ratio),
        support_internal_matrix_entries=support_internal_matrix_entries,
        potential_shell_value=potential_shell_value,
        potential_shell_size=potential_shell_size,
        support_shell_fraction=support_shell_fraction,
        effective_shell_fraction=effective_shell_fraction,
        potential_shell_residual=potential_shell_residual,
    )


def analyze_real_space_support_morphology(
    *,
    basis_configs: NDArray[np.integer],
    support_mask: NDArray[np.bool_],
    reduced_iz_region_variable_indices: tuple[int, ...],
    reduced_iz_monitor_component_groups: dict[
        ReducedIZMonitorDecomposition,
        tuple[ReducedIZMonitorComponentGroup, ...],
    ],
) -> RealSpaceSupportMorphology:
    """Return finite-size variable-space support morphology diagnostics."""
    n_variables = int(basis_configs.shape[1])
    support_configs = basis_configs[support_mask]

    if support_configs.shape[0] == 0 or n_variables == 0:
        active_variable_indices: tuple[int, ...] = ()
    else:
        reference = support_configs[0]
        active_mask = np.any(support_configs != reference, axis=0)
        active_variable_indices = tuple(int(index) for index in np.flatnonzero(active_mask))

    active_variable_count = len(active_variable_indices)
    active_variable_fraction = active_variable_count / float(n_variables) if n_variables else 0.0
    frozen_variable_count = n_variables - active_variable_count

    if n_variables == 0:
        label: RealSpaceSupportMorphologyLabel = "unknown"
    elif active_variable_count == 0:
        label = "frozen"
    elif active_variable_count == n_variables:
        label = "fully_active"
    else:
        label = "partially_active"

    exact_groups = reduced_iz_monitor_component_groups.get("exact_support", ())
    connected_groups = reduced_iz_monitor_component_groups.get("connected_support", ())
    exact_sizes = tuple(int(group.support_size) for group in exact_groups)
    connected_sizes = tuple(int(group.support_size) for group in connected_groups)
    reduced_count = len(reduced_iz_region_variable_indices)

    return RealSpaceSupportMorphology(
        label=label,
        n_variables=n_variables,
        active_variable_indices=active_variable_indices,
        active_variable_count=active_variable_count,
        active_variable_fraction=float(active_variable_fraction),
        frozen_variable_count=frozen_variable_count,
        reduced_iz_region_variable_indices=reduced_iz_region_variable_indices,
        reduced_iz_region_variable_count=reduced_count,
        reduced_iz_region_variable_fraction=(
            reduced_count / float(n_variables) if n_variables else 0.0
        ),
        exact_support_component_count=len(exact_groups),
        exact_support_component_sizes=exact_sizes,
        connected_support_component_count=len(connected_groups),
        connected_support_component_sizes=connected_sizes,
    )


@dataclass(frozen=True, slots=True)
class SupportMorphologyReport:
    """Finite-size Fock- and real-space support descriptors."""

    fock: FockSupportMorphology
    real_space: RealSpaceSupportMorphology


def analyze_support_morphology(
    *,
    full_state: NDArray[np.complex128],
    kinetic_matrix: sp.spmatrix | sp.sparray | NDArray,
    basis_configs: NDArray[np.integer],
    environment_report: object | None = None,
    potential_diagonal: NDArray | None = None,
    config: SupportMorphologyConfig | None = None,
) -> SupportMorphologyReport:
    """Analyze state-support morphology independently of environment reduction.

    ``environment_report`` is optional. When provided, its reduced-IZ region and
    component decomposition are used only to annotate real-space support; they
    do not affect the Fock-space morphology or any environment-removal claim.
    """
    if config is None:
        config = SupportMorphologyConfig()

    state = np.asarray(full_state, dtype=np.complex128).reshape(-1)
    configs = np.asarray(basis_configs)
    if configs.ndim != 2 or configs.shape[0] != state.size:
        raise ValueError("basis_configs must have shape (full_state.size, n_variables).")

    kinetic_csr = sp.csr_array(kinetic_matrix)
    support_mask = np.abs(state) > config.amplitude_tolerance
    support_indices = np.flatnonzero(support_mask)
    if support_indices.size:
        incoming = np.asarray(kinetic_csr[:, support_indices].count_nonzero(axis=1)).reshape(-1) > 0
        frontier = np.flatnonzero(incoming & ~support_mask).astype(np.int64, copy=False)
    else:
        frontier = np.array([], dtype=np.int64)

    if environment_report is None:
        region_variables: tuple[int, ...] = ()
        component_groups: dict[
            ReducedIZMonitorDecomposition,
            tuple[ReducedIZMonitorComponentGroup, ...],
        ] = {}
    else:
        region_variables = tuple(
            int(index)
            for index in getattr(environment_report, "reduced_iz_region_variable_indices", ())
        )
        component_groups = dict(
            getattr(environment_report, "reduced_iz_monitor_component_groups", {})
        )

    return SupportMorphologyReport(
        fock=analyze_fock_support_morphology(
            full_state=state,
            kinetic_csr=kinetic_csr,
            support_mask=support_mask,
            active_frontier_zero_indices=frontier,
            potential_diagonal=potential_diagonal,
            config=config,
        ),
        real_space=analyze_real_space_support_morphology(
            basis_configs=configs,
            support_mask=support_mask,
            reduced_iz_region_variable_indices=region_variables,
            reduced_iz_monitor_component_groups=component_groups,
        ),
    )
