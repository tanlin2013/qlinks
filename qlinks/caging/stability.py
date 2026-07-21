from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Literal, Sequence

import numpy as np
import numpy.typing as npt
import scipy.linalg as scipy_linalg
import scipy.sparse as scipy_sparse
import scipy.sparse.linalg as scipy_sparse_linalg

from qlinks.caging.invariant_subspace import invariant_boundary_nullspace
from qlinks.caging.local_search import (
    _qdm_flip_transition_from_action,
    _qdm_global_plaquette_actions,
)
from qlinks.caging.localization import IPRLocalizationConfig, localized_basis_by_many_start_ipr
from qlinks.caging.nullspace import as_dense_array, nullspace_svd
from qlinks.caging.periodic_sequence import (
    SquareQDMPeriodicProductInstance,
    SquareQDMPeriodicProductUnitCell,
)

CoefficientField = Literal["real", "complex"]


@dataclass(frozen=True, slots=True)
class CageHamiltonianBlocks:
    """Hamiltonian blocks associated with one proposed cage support.

    The basis is ordered conceptually as ``support + complement``.  The
    ``boundary`` block maps amplitudes on the support to the complementary
    Hilbert-space configurations.
    """

    support: npt.NDArray[np.int64]
    complement: npt.NDArray[np.int64]
    internal: object
    boundary: object
    external: object

    @property
    def hilbert_size(self) -> int:
        return int(self.support.size + self.complement.size)

    @property
    def support_size(self) -> int:
        return int(self.support.size)


@dataclass(frozen=True, slots=True)
class CageStabilityDiagnostic:
    """Static interference-kernel diagnostic for one Hamiltonian and support."""

    support: tuple[int, ...]
    boundary_singular_values: npt.NDArray[np.float64]
    boundary_rank: int
    boundary_nullity: int
    interference_gap: float | None
    boundary_kernel_basis: npt.NDArray[np.complex128]
    invariant_cage_basis: npt.NDArray[np.complex128]
    invariant_cage_dimension: int
    hermiticity_residual: float
    state_energy: complex | None = None
    state_boundary_residual: float | None = None
    state_internal_eigen_residual: float | None = None
    state_full_residual: float | None = None
    state_invariant_weight: float | None = None

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "support_size": len(self.support),
            "boundary_rank": self.boundary_rank,
            "boundary_nullity": self.boundary_nullity,
            "interference_gap": self.interference_gap,
            "invariant_cage_dimension": self.invariant_cage_dimension,
            "hermiticity_residual": self.hermiticity_residual,
            "state_energy": self.state_energy,
            "state_boundary_residual": self.state_boundary_residual,
            "state_internal_eigen_residual": self.state_internal_eigen_residual,
            "state_full_residual": self.state_full_residual,
            "state_invariant_weight": self.state_invariant_weight,
        }


@dataclass(frozen=True, slots=True)
class CageBranchPoint:
    """One point in a one-parameter cage-continuation scan."""

    parameter: float
    diagnostic: CageStabilityDiagnostic
    minimum_principal_overlap: float
    projector_distance_from_reference: float
    continued_state: npt.NDArray[np.complex128] | None
    continued_energy: complex | None
    continued_overlap_with_previous: float | None
    continued_overlap_with_reference: float | None
    continued_boundary_residual: float | None
    continued_internal_eigen_residual: float | None
    continued_full_residual: float | None

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "parameter": self.parameter,
            **self.diagnostic.to_summary_dict(),
            "minimum_principal_overlap": self.minimum_principal_overlap,
            "projector_distance_from_reference": self.projector_distance_from_reference,
            "continued_energy": self.continued_energy,
            "continued_overlap_with_previous": self.continued_overlap_with_previous,
            "continued_overlap_with_reference": self.continued_overlap_with_reference,
            "continued_boundary_residual": self.continued_boundary_residual,
            "continued_internal_eigen_residual": self.continued_internal_eigen_residual,
            "continued_full_residual": self.continued_full_residual,
        }


@dataclass(frozen=True, slots=True)
class CageBranchReport:
    """Result of scanning a cage support along a one-parameter deformation."""

    support: tuple[int, ...]
    reference_dimension: int
    points: tuple[CageBranchPoint, ...]
    tolerance: float

    @property
    def parameters(self) -> npt.NDArray[np.float64]:
        return np.asarray([point.parameter for point in self.points], dtype=np.float64)

    @property
    def interference_gaps(self) -> npt.NDArray[np.float64]:
        return np.asarray(
            [
                (
                    np.nan
                    if point.diagnostic.interference_gap is None
                    else point.diagnostic.interference_gap
                )
                for point in self.points
            ],
            dtype=np.float64,
        )

    @property
    def invariant_dimensions(self) -> npt.NDArray[np.int64]:
        return np.asarray(
            [point.diagnostic.invariant_cage_dimension for point in self.points],
            dtype=np.int64,
        )

    @property
    def continued_full_residuals(self) -> npt.NDArray[np.float64]:
        return np.asarray(
            [
                np.nan if point.continued_full_residual is None else point.continued_full_residual
                for point in self.points
            ],
            dtype=np.float64,
        )

    @property
    def fixed_state_full_residuals(self) -> npt.NDArray[np.float64]:
        return np.asarray(
            [
                (
                    np.nan
                    if point.diagnostic.state_full_residual is None
                    else point.diagnostic.state_full_residual
                )
                for point in self.points
            ],
            dtype=np.float64,
        )

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "support_size": len(self.support),
            "reference_dimension": self.reference_dimension,
            "tolerance": self.tolerance,
            "points": tuple(point.to_summary_dict() for point in self.points),
        }


@dataclass(frozen=True, slots=True)
class CageRecordStabilitySummary:
    """Record-wise stability data for a preferred basis of a degenerate cage manifold."""

    record_index: int
    signature: tuple[int, int]
    support_size: int
    inverse_participation_ratio: float
    classification_label: str | None
    n_collective_cancellation_source_probes: int | None
    formal_compatible_dimension: int
    exact_fixed_state_dimension: int
    tangent_only_dimension: int

    @property
    def requires_collective_cancellation(self) -> bool | None:
        if self.n_collective_cancellation_source_probes is None:
            return None
        return self.n_collective_cancellation_source_probes > 0

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "record_index": self.record_index,
            "signature": self.signature,
            "support_size": self.support_size,
            "inverse_participation_ratio": self.inverse_participation_ratio,
            "classification_label": self.classification_label,
            "n_collective_cancellation_source_probes": (
                self.n_collective_cancellation_source_probes
            ),
            "requires_collective_cancellation": self.requires_collective_cancellation,
            "formal_compatible_dimension": self.formal_compatible_dimension,
            "exact_fixed_state_dimension": self.exact_fixed_state_dimension,
            "tangent_only_dimension": self.tangent_only_dimension,
        }


@dataclass(frozen=True, slots=True)
class RandomCageStabilitySample:
    """One random multi-parameter deformation sample."""

    strength: float
    sample_index: int
    coefficients: tuple[float, ...]
    boundary_nullity: int
    invariant_cage_dimension: int
    interference_gap: float | None
    minimum_principal_overlap: float
    projector_distance_from_reference: float
    fixed_state_full_residual: float | None
    preserves_dimension: bool
    follows_reference_subspace: bool
    cage_survives: bool

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "strength": self.strength,
            "sample_index": self.sample_index,
            "coefficients": self.coefficients,
            "boundary_nullity": self.boundary_nullity,
            "invariant_cage_dimension": self.invariant_cage_dimension,
            "interference_gap": self.interference_gap,
            "minimum_principal_overlap": self.minimum_principal_overlap,
            "projector_distance_from_reference": self.projector_distance_from_reference,
            "fixed_state_full_residual": self.fixed_state_full_residual,
            "preserves_dimension": self.preserves_dimension,
            "follows_reference_subspace": self.follows_reference_subspace,
            "cage_survives": self.cage_survives,
        }


@dataclass(frozen=True, slots=True)
class RandomCageStabilityAggregate:
    """Aggregate random-ensemble statistics at one perturbation strength."""

    strength: float
    n_samples: int
    n_survived: int
    survival_fraction: float
    minimum_interference_gap: float | None
    median_interference_gap: float | None
    median_projector_distance: float
    median_minimum_principal_overlap: float

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "strength": self.strength,
            "n_samples": self.n_samples,
            "n_survived": self.n_survived,
            "survival_fraction": self.survival_fraction,
            "minimum_interference_gap": self.minimum_interference_gap,
            "median_interference_gap": self.median_interference_gap,
            "median_projector_distance": self.median_projector_distance,
            "median_minimum_principal_overlap": self.median_minimum_principal_overlap,
        }


@dataclass(frozen=True, slots=True)
class RandomCageStabilityReport:
    """Random compatible- or control-perturbation ensemble report."""

    support: tuple[int, ...]
    reference_dimension: int
    minimum_subspace_overlap: float
    samples: tuple[RandomCageStabilitySample, ...]
    aggregates: tuple[RandomCageStabilityAggregate, ...]
    random_seed: int | None
    tolerance: float

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "support_size": len(self.support),
            "reference_dimension": self.reference_dimension,
            "minimum_subspace_overlap": self.minimum_subspace_overlap,
            "random_seed": self.random_seed,
            "tolerance": self.tolerance,
            "aggregates": tuple(item.to_summary_dict() for item in self.aggregates),
        }


@dataclass(frozen=True, slots=True)
class PerturbationCompatibilityDiagnostic:
    """Compatibility of one perturbation with a selected cage vector."""

    perturbation_index: int
    boundary_residual: float
    eigenvector_residual: float
    first_order_boundary_obstruction_residual: float
    first_order_eigenstate_obstruction_residual: float
    preserves_state: bool
    first_order_boundary_compatible: bool
    first_order_eigenstate_compatible: bool

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "perturbation_index": self.perturbation_index,
            "boundary_residual": self.boundary_residual,
            "eigenvector_residual": self.eigenvector_residual,
            "first_order_boundary_obstruction_residual": (
                self.first_order_boundary_obstruction_residual
            ),
            "first_order_eigenstate_obstruction_residual": (
                self.first_order_eigenstate_obstruction_residual
            ),
            "preserves_state": self.preserves_state,
            "first_order_boundary_compatible": self.first_order_boundary_compatible,
            "first_order_eigenstate_compatible": self.first_order_eigenstate_compatible,
        }


@dataclass(frozen=True, slots=True)
class LinearizedCageObstructionReport:
    """First-order obstruction map for a basis of local perturbations."""

    coefficient_field: CoefficientField
    n_parameters: int
    obstruction_matrix: npt.NDArray[np.complex128]
    boundary_obstruction_matrix: npt.NDArray[np.complex128]
    tangent_operator: npt.NDArray[np.complex128] | None
    constraint_matrix: npt.NDArray[np.float64] | npt.NDArray[np.complex128]
    singular_values: npt.NDArray[np.float64]
    rank: int
    compatible_dimension: int
    compatible_coefficient_basis: npt.NDArray[np.float64] | npt.NDArray[np.complex128]
    left_boundary_nullspace: npt.NDArray[np.complex128]
    left_tangent_nullspace: npt.NDArray[np.complex128] | None
    perturbation_diagnostics: tuple[PerturbationCompatibilityDiagnostic, ...]
    tolerance: float

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "coefficient_field": self.coefficient_field,
            "n_parameters": self.n_parameters,
            "obstruction_dimension": int(self.constraint_matrix.shape[0]),
            "rank": self.rank,
            "compatible_dimension": self.compatible_dimension,
            "singular_values": tuple(float(value) for value in self.singular_values),
            "perturbations": tuple(
                item.to_summary_dict() for item in self.perturbation_diagnostics
            ),
            "tolerance": self.tolerance,
        }


@dataclass(frozen=True, slots=True)
class CageJacobianConditioningReport:
    """Conditioning spectrum of the gauge-fixed caged-eigenpair equations."""

    support: tuple[int, ...]
    energy: complex
    jacobian: npt.NDArray[np.complex128]
    singular_values: npt.NDArray[np.float64]
    rank: int
    nullity: int
    cage_gap: float
    internal_residual: float
    boundary_residual: float
    full_residual: float
    tolerance: float

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "support_size": len(self.support),
            "energy": self.energy,
            "jacobian_rows": int(self.jacobian.shape[0]),
            "jacobian_columns": int(self.jacobian.shape[1]),
            "rank": self.rank,
            "nullity": self.nullity,
            "cage_gap": self.cage_gap,
            "singular_values": tuple(float(value) for value in self.singular_values),
            "internal_residual": self.internal_residual,
            "boundary_residual": self.boundary_residual,
            "full_residual": self.full_residual,
            "tolerance": self.tolerance,
        }


@dataclass(frozen=True, slots=True)
class FixedCageStateCompatibilityReport:
    """Exact affine perturbation space that preserves one cage vector.

    A coefficient vector belongs to this space when its perturbation ``V``
    satisfies both ``delta_B phi = 0`` and
    ``(I - |phi><phi|) delta_A phi = 0``.  Therefore the same compact vector
    remains an exact eigenstate of ``H0 + lambda V`` for every ``lambda``.
    """

    coefficient_field: CoefficientField
    n_parameters: int
    action_matrix: npt.NDArray[np.complex128]
    constraint_matrix: npt.NDArray[np.float64] | npt.NDArray[np.complex128]
    singular_values: npt.NDArray[np.float64]
    rank: int
    compatible_dimension: int
    compatible_coefficient_basis: npt.NDArray[np.float64] | npt.NDArray[np.complex128]
    perturbation_residuals: npt.NDArray[np.float64]
    tolerance: float

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "coefficient_field": self.coefficient_field,
            "n_parameters": self.n_parameters,
            "constraint_dimension": int(self.constraint_matrix.shape[0]),
            "rank": self.rank,
            "compatible_dimension": self.compatible_dimension,
            "singular_values": tuple(float(value) for value in self.singular_values),
            "perturbation_residuals": tuple(float(value) for value in self.perturbation_residuals),
            "tolerance": self.tolerance,
        }


@dataclass(frozen=True, slots=True)
class CageCompatibilityHierarchyReport:
    """Compare first-order continuation with exact fixed-state compatibility."""

    first_order: LinearizedCageObstructionReport
    fixed_state: FixedCageStateCompatibilityReport
    tangent_only_coefficient_basis: npt.NDArray[np.float64] | npt.NDArray[np.complex128]
    fixed_subspace_inclusion_residual: float

    @property
    def tangent_only_dimension(self) -> int:
        """Number of first-order directions not preserving the state exactly."""
        return int(self.tangent_only_coefficient_basis.shape[1])

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "n_parameters": self.first_order.n_parameters,
            "first_order_compatible_dimension": self.first_order.compatible_dimension,
            "fixed_state_compatible_dimension": self.fixed_state.compatible_dimension,
            "tangent_only_dimension": self.tangent_only_dimension,
            "fixed_subspace_inclusion_residual": self.fixed_subspace_inclusion_residual,
            "tolerance": self.first_order.tolerance,
        }


@dataclass(frozen=True, slots=True)
class SupportEigenstateBranchPoint:
    """One support-local eigenstate tracked even after exact caging is lost."""

    parameter: float
    state: npt.NDArray[np.complex128]
    energy: complex
    overlap_with_previous: float
    overlap_with_reference: float
    boundary_residual: float
    internal_eigen_residual: float
    full_residual: float
    boundary_nullity: int
    invariant_cage_dimension: int
    interference_gap: float | None
    exact_cage: bool

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "parameter": self.parameter,
            "energy": self.energy,
            "overlap_with_previous": self.overlap_with_previous,
            "overlap_with_reference": self.overlap_with_reference,
            "boundary_residual": self.boundary_residual,
            "internal_eigen_residual": self.internal_eigen_residual,
            "full_residual": self.full_residual,
            "boundary_nullity": self.boundary_nullity,
            "invariant_cage_dimension": self.invariant_cage_dimension,
            "interference_gap": self.interference_gap,
            "exact_cage": self.exact_cage,
        }


@dataclass(frozen=True, slots=True)
class SupportEigenstateBranchReport:
    """Continuation of an internal eigenstate with its boundary leakage."""

    support: tuple[int, ...]
    points: tuple[SupportEigenstateBranchPoint, ...]
    tolerance: float

    @property
    def parameters(self) -> npt.NDArray[np.float64]:
        return np.asarray([point.parameter for point in self.points], dtype=np.float64)

    @property
    def boundary_residuals(self) -> npt.NDArray[np.float64]:
        return np.asarray(
            [point.boundary_residual for point in self.points],
            dtype=np.float64,
        )

    @property
    def full_residuals(self) -> npt.NDArray[np.float64]:
        return np.asarray([point.full_residual for point in self.points], dtype=np.float64)

    @property
    def exact_cage_flags(self) -> npt.NDArray[np.bool_]:
        return np.asarray([point.exact_cage for point in self.points], dtype=np.bool_)

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "support_size": len(self.support),
            "tolerance": self.tolerance,
            "points": tuple(point.to_summary_dict() for point in self.points),
        }


def partition_cage_hamiltonian(
    hamiltonian: object,
    support: Sequence[int] | npt.NDArray[np.integer],
) -> CageHamiltonianBlocks:
    """Partition a square Hamiltonian into support, boundary, and external blocks."""
    shape = getattr(hamiltonian, "shape", None)
    if shape is None or len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("hamiltonian must be a square matrix.")

    hilbert_size = int(shape[0])
    support_array = np.asarray(support, dtype=np.int64).reshape(-1)
    if np.any(support_array < 0) or np.any(support_array >= hilbert_size):
        raise ValueError("support contains an out-of-range basis index.")
    if np.unique(support_array).size != support_array.size:
        raise ValueError("support must not contain duplicate basis indices.")

    support_mask = np.zeros(hilbert_size, dtype=bool)
    support_mask[support_array] = True
    complement = np.flatnonzero(~support_mask).astype(np.int64, copy=False)

    internal = _matrix_subblock(hamiltonian, support_array, support_array)
    boundary = _matrix_subblock(hamiltonian, complement, support_array)
    external = _matrix_subblock(hamiltonian, complement, complement)

    return CageHamiltonianBlocks(
        support=support_array,
        complement=complement,
        internal=internal,
        boundary=boundary,
        external=external,
    )


def diagnose_cage_stability(
    hamiltonian: object,
    support: Sequence[int] | npt.NDArray[np.integer],
    *,
    state: npt.ArrayLike | None = None,
    tolerance: float = 1e-10,
    max_power: int | None = None,
    stabilization_rounds: int = 1,
) -> CageStabilityDiagnostic:
    """Diagnose the interference kernel and its internally invariant subspace.

    ``state`` can be either a compact support vector or a full Hilbert-space
    vector.  When supplied, the report distinguishes boundary leakage from the
    internal eigenvector residual.
    """
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")

    blocks = partition_cage_hamiltonian(hamiltonian, support)
    internal = as_dense_array(blocks.internal)
    boundary = as_dense_array(blocks.boundary)

    singular_values = scipy_linalg.svdvals(boundary).astype(np.float64, copy=False)
    boundary_rank = int(np.sum(singular_values > tolerance))
    boundary_nullity = int(blocks.support_size - boundary_rank)
    nonzero_singular_values = singular_values[singular_values > tolerance]
    interference_gap = (
        None if nonzero_singular_values.size == 0 else float(np.min(nonzero_singular_values))
    )
    boundary_kernel_basis = nullspace_svd(boundary, tolerance=tolerance)
    invariant_basis = invariant_boundary_nullspace(
        internal,
        boundary,
        tolerance=tolerance,
        max_power=max_power,
        stabilization_rounds=stabilization_rounds,
    )
    hermiticity_residual = float(np.linalg.norm(internal - internal.conj().T))

    state_energy: complex | None = None
    state_boundary_residual: float | None = None
    state_internal_eigen_residual: float | None = None
    state_full_residual: float | None = None
    state_invariant_weight: float | None = None

    if state is not None:
        local_state = _normalized_local_state(
            state,
            support=blocks.support,
            hilbert_size=blocks.hilbert_size,
            tolerance=tolerance,
        )
        action = internal @ local_state
        state_energy = complex(np.vdot(local_state, action))
        internal_residual_vector = action - state_energy * local_state
        boundary_residual_vector = boundary @ local_state
        state_boundary_residual = float(np.linalg.norm(boundary_residual_vector))
        state_internal_eigen_residual = float(np.linalg.norm(internal_residual_vector))
        state_full_residual = float(
            np.hypot(state_boundary_residual, state_internal_eigen_residual)
        )
        state_invariant_weight = float(np.linalg.norm(invariant_basis.conj().T @ local_state) ** 2)

    return CageStabilityDiagnostic(
        support=tuple(int(index) for index in blocks.support),
        boundary_singular_values=singular_values,
        boundary_rank=boundary_rank,
        boundary_nullity=boundary_nullity,
        interference_gap=interference_gap,
        boundary_kernel_basis=boundary_kernel_basis,
        invariant_cage_basis=invariant_basis,
        invariant_cage_dimension=int(invariant_basis.shape[1]),
        hermiticity_residual=hermiticity_residual,
        state_energy=state_energy,
        state_boundary_residual=state_boundary_residual,
        state_internal_eigen_residual=state_internal_eigen_residual,
        state_full_residual=state_full_residual,
        state_invariant_weight=state_invariant_weight,
    )


def scan_cage_stability_branch(
    base_hamiltonian: object,
    perturbation: object,
    support: Sequence[int] | npt.NDArray[np.integer],
    parameters: Sequence[float] | npt.NDArray[np.floating],
    *,
    reference_state: npt.ArrayLike | None = None,
    reference_subspace: npt.ArrayLike | None = None,
    tolerance: float = 1e-10,
    max_power: int | None = None,
    stabilization_rounds: int = 1,
) -> CageBranchReport:
    """Track a compact cage subspace along ``H(lambda) = H0 + lambda V``."""
    _validate_same_matrix_shape(base_hamiltonian, perturbation)
    parameter_array = np.asarray(parameters, dtype=np.float64).reshape(-1)
    if parameter_array.size == 0:
        raise ValueError("parameters must contain at least one value.")

    reference_diagnostic = diagnose_cage_stability(
        base_hamiltonian,
        support,
        state=reference_state,
        tolerance=tolerance,
        max_power=max_power,
        stabilization_rounds=stabilization_rounds,
    )
    support_array = np.asarray(reference_diagnostic.support, dtype=np.int64)
    normalized_reference_state = (
        None
        if reference_state is None
        else _normalized_local_state(
            reference_state,
            support=support_array,
            hilbert_size=int(base_hamiltonian.shape[0]),
            tolerance=tolerance,
        )
    )
    reference_basis = _reference_comparison_basis(
        reference_subspace=reference_subspace,
        reference_state=normalized_reference_state,
        fallback_basis=reference_diagnostic.invariant_cage_basis,
        support=support_array,
        hilbert_size=int(base_hamiltonian.shape[0]),
        tolerance=tolerance,
    )
    previous_state = normalized_reference_state
    points: list[CageBranchPoint] = []

    for parameter in parameter_array:
        hamiltonian = base_hamiltonian + float(parameter) * perturbation
        diagnostic = diagnose_cage_stability(
            hamiltonian,
            support_array,
            state=normalized_reference_state,
            tolerance=tolerance,
            max_power=max_power,
            stabilization_rounds=stabilization_rounds,
        )
        blocks = partition_cage_hamiltonian(hamiltonian, support_array)
        internal_matrix = as_dense_array(blocks.internal)
        matched_basis = _matched_cage_eigensubspace(
            internal_matrix,
            diagnostic.invariant_cage_basis,
            reference_basis=reference_basis,
            target_dimension=int(reference_basis.shape[1]),
            tolerance=tolerance,
        )
        principal_overlaps = subspace_principal_overlaps(
            reference_basis,
            matched_basis,
        )
        minimum_principal_overlap = (
            0.0 if principal_overlaps.size == 0 else float(np.min(principal_overlaps))
        )
        projector_distance = subspace_projector_distance(
            reference_basis,
            matched_basis,
        )

        continued_state, continued_energy = _continued_cage_eigenstate(
            internal_matrix,
            diagnostic.invariant_cage_basis,
            previous_state=previous_state,
            tolerance=tolerance,
        )

        overlap_with_previous: float | None = None
        overlap_with_reference: float | None = None
        boundary_residual: float | None = None
        internal_eigen_residual: float | None = None
        full_residual: float | None = None
        if continued_state is not None and continued_energy is not None:
            if previous_state is not None:
                overlap_with_previous = float(abs(np.vdot(previous_state, continued_state)))
            if normalized_reference_state is not None:
                overlap_with_reference = float(
                    abs(np.vdot(normalized_reference_state, continued_state))
                )
            boundary_residual = float(
                np.linalg.norm(as_dense_array(blocks.boundary) @ continued_state)
            )
            internal_eigen_residual = float(
                np.linalg.norm(
                    as_dense_array(blocks.internal) @ continued_state
                    - continued_energy * continued_state
                )
            )
            full_residual = float(np.hypot(boundary_residual, internal_eigen_residual))
            previous_state = continued_state

        points.append(
            CageBranchPoint(
                parameter=float(parameter),
                diagnostic=diagnostic,
                minimum_principal_overlap=minimum_principal_overlap,
                projector_distance_from_reference=projector_distance,
                continued_state=continued_state,
                continued_energy=continued_energy,
                continued_overlap_with_previous=overlap_with_previous,
                continued_overlap_with_reference=overlap_with_reference,
                continued_boundary_residual=boundary_residual,
                continued_internal_eigen_residual=internal_eigen_residual,
                continued_full_residual=full_residual,
            )
        )

    return CageBranchReport(
        support=tuple(int(index) for index in support_array),
        reference_dimension=int(reference_basis.shape[1]),
        points=tuple(points),
        tolerance=tolerance,
    )


def scan_support_eigenstate_branch(
    base_hamiltonian: object,
    perturbation: object,
    support: Sequence[int] | npt.NDArray[np.integer],
    parameters: Sequence[float] | npt.NDArray[np.floating],
    *,
    reference_state: npt.ArrayLike,
    tolerance: float = 1e-10,
    max_power: int | None = None,
    stabilization_rounds: int = 1,
) -> SupportEigenstateBranchReport:
    """Track the nearest support-local eigenstate and measure its leakage.

    Unlike :func:`scan_cage_stability_branch`, this continuation does not stop
    when the exact invariant cage subspace disappears.  At every parameter it
    follows the closest eigenspace of the internal support block. Within a
    degenerate eigenspace it selects the state with minimum boundary leakage,
    then reports ``||B phi||``. A first-order-compatible but non-integrable
    direction is therefore visible through a residual that starts at
    quadratic or higher order.
    """
    _validate_same_matrix_shape(base_hamiltonian, perturbation)
    parameter_array = np.asarray(parameters, dtype=np.float64).reshape(-1)
    if parameter_array.size == 0:
        raise ValueError("parameters must contain at least one value.")

    base_blocks = partition_cage_hamiltonian(base_hamiltonian, support)
    support_array = base_blocks.support
    normalized_reference_state = _normalized_local_state(
        reference_state,
        support=support_array,
        hilbert_size=base_blocks.hilbert_size,
        tolerance=tolerance,
    )
    previous_state = normalized_reference_state
    points: list[SupportEigenstateBranchPoint] = []

    for parameter in parameter_array:
        hamiltonian = base_hamiltonian + float(parameter) * perturbation
        diagnostic = diagnose_cage_stability(
            hamiltonian,
            support_array,
            tolerance=tolerance,
            max_power=max_power,
            stabilization_rounds=stabilization_rounds,
        )
        blocks = partition_cage_hamiltonian(hamiltonian, support_array)
        internal_matrix = as_dense_array(blocks.internal)
        boundary_matrix = as_dense_array(blocks.boundary)

        state, energy = _minimum_leakage_internal_eigenstate(
            internal_matrix,
            boundary_matrix,
            reference_state=previous_state,
            tolerance=tolerance,
        )
        overlap = np.vdot(previous_state, state)
        if abs(overlap) > tolerance:
            state = state * np.exp(-1j * np.angle(overlap))

        energy = complex(np.vdot(state, internal_matrix @ state))
        boundary_residual = float(np.linalg.norm(boundary_matrix @ state))
        internal_eigen_residual = float(np.linalg.norm(internal_matrix @ state - energy * state))
        full_residual = float(np.hypot(boundary_residual, internal_eigen_residual))
        overlap_with_previous = float(abs(np.vdot(previous_state, state)))
        overlap_with_reference = float(abs(np.vdot(normalized_reference_state, state)))

        points.append(
            SupportEigenstateBranchPoint(
                parameter=float(parameter),
                state=state,
                energy=energy,
                overlap_with_previous=overlap_with_previous,
                overlap_with_reference=overlap_with_reference,
                boundary_residual=boundary_residual,
                internal_eigen_residual=internal_eigen_residual,
                full_residual=full_residual,
                boundary_nullity=diagnostic.boundary_nullity,
                invariant_cage_dimension=diagnostic.invariant_cage_dimension,
                interference_gap=diagnostic.interference_gap,
                exact_cage=(full_residual <= tolerance),
            )
        )
        previous_state = state

    return SupportEigenstateBranchReport(
        support=tuple(int(index) for index in support_array),
        points=tuple(points),
        tolerance=tolerance,
    )


def random_cage_stability_ensemble(
    base_hamiltonian: object,
    perturbations: Sequence[object],
    support: Sequence[int] | npt.NDArray[np.integer],
    strengths: Sequence[float] | npt.NDArray[np.floating],
    *,
    n_samples: int = 100,
    reference_state: npt.ArrayLike | None = None,
    reference_subspace: npt.ArrayLike | None = None,
    target_dimension: int | None = None,
    minimum_subspace_overlap: float = 0.5,
    random_seed: int | None = None,
    tolerance: float = 1e-10,
    max_power: int | None = None,
    stabilization_rounds: int = 1,
) -> RandomCageStabilityReport:
    """Sample normalized random directions in a supplied perturbation basis."""
    if len(perturbations) == 0:
        raise ValueError("perturbations must contain at least one matrix.")
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1.")
    if not 0.0 <= minimum_subspace_overlap <= 1.0:
        raise ValueError("minimum_subspace_overlap must lie in [0, 1].")
    for perturbation in perturbations:
        _validate_same_matrix_shape(base_hamiltonian, perturbation)

    strength_array = np.asarray(strengths, dtype=np.float64).reshape(-1)
    if strength_array.size == 0:
        raise ValueError("strengths must contain at least one value.")

    reference_diagnostic = diagnose_cage_stability(
        base_hamiltonian,
        support,
        state=reference_state,
        tolerance=tolerance,
        max_power=max_power,
        stabilization_rounds=stabilization_rounds,
    )
    support_array = np.asarray(reference_diagnostic.support, dtype=np.int64)
    normalized_reference_state = (
        None
        if reference_state is None
        else _normalized_local_state(
            reference_state,
            support=support_array,
            hilbert_size=int(base_hamiltonian.shape[0]),
            tolerance=tolerance,
        )
    )
    reference_basis = _reference_comparison_basis(
        reference_subspace=reference_subspace,
        reference_state=normalized_reference_state,
        fallback_basis=reference_diagnostic.invariant_cage_basis,
        support=support_array,
        hilbert_size=int(base_hamiltonian.shape[0]),
        tolerance=tolerance,
    )
    reference_dimension = (
        int(reference_basis.shape[1]) if target_dimension is None else int(target_dimension)
    )
    if reference_dimension < 0:
        raise ValueError("target_dimension must be non-negative.")
    if reference_dimension > reference_basis.shape[1]:
        raise ValueError(
            "target_dimension cannot exceed the supplied reference-subspace dimension."
        )

    rng = np.random.default_rng(random_seed)
    samples: list[RandomCageStabilitySample] = []

    for strength in strength_array:
        for sample_index in range(n_samples):
            coefficients = rng.normal(size=len(perturbations))
            coefficient_norm = float(np.linalg.norm(coefficients))
            if coefficient_norm == 0.0:  # pragma: no cover - effectively impossible
                coefficients[0] = 1.0
                coefficient_norm = 1.0
            coefficients = coefficients / coefficient_norm

            hamiltonian = _matrix_linear_combination(
                base_hamiltonian,
                perturbations,
                float(strength) * coefficients,
            )
            diagnostic = diagnose_cage_stability(
                hamiltonian,
                support,
                state=reference_state,
                tolerance=tolerance,
                max_power=max_power,
                stabilization_rounds=stabilization_rounds,
            )
            blocks = partition_cage_hamiltonian(hamiltonian, support)
            matched_basis = _matched_cage_eigensubspace(
                as_dense_array(blocks.internal),
                diagnostic.invariant_cage_basis,
                reference_basis=reference_basis,
                target_dimension=reference_dimension,
                tolerance=tolerance,
            )
            principal_overlaps = subspace_principal_overlaps(
                reference_basis,
                matched_basis,
            )
            minimum_overlap = (
                0.0 if principal_overlaps.size == 0 else float(np.min(principal_overlaps))
            )
            projector_distance = subspace_projector_distance(
                reference_basis,
                matched_basis,
            )
            preserves_dimension = diagnostic.invariant_cage_dimension >= reference_dimension
            follows_reference = (
                reference_dimension == 0 or minimum_overlap >= minimum_subspace_overlap
            )
            survives = preserves_dimension and follows_reference

            samples.append(
                RandomCageStabilitySample(
                    strength=float(strength),
                    sample_index=sample_index,
                    coefficients=tuple(float(value) for value in coefficients),
                    boundary_nullity=diagnostic.boundary_nullity,
                    invariant_cage_dimension=diagnostic.invariant_cage_dimension,
                    interference_gap=diagnostic.interference_gap,
                    minimum_principal_overlap=minimum_overlap,
                    projector_distance_from_reference=projector_distance,
                    fixed_state_full_residual=diagnostic.state_full_residual,
                    preserves_dimension=preserves_dimension,
                    follows_reference_subspace=follows_reference,
                    cage_survives=survives,
                )
            )

    aggregates = tuple(
        _aggregate_random_samples(samples, strength=float(strength)) for strength in strength_array
    )
    return RandomCageStabilityReport(
        support=reference_diagnostic.support,
        reference_dimension=reference_dimension,
        minimum_subspace_overlap=minimum_subspace_overlap,
        samples=tuple(samples),
        aggregates=aggregates,
        random_seed=random_seed,
        tolerance=tolerance,
    )


def linearized_cage_obstruction(
    boundary_matrix: object,
    cage_state: npt.ArrayLike,
    boundary_perturbations: Sequence[object],
    *,
    internal_matrix: object | None = None,
    internal_perturbations: Sequence[object] | None = None,
    coefficient_field: CoefficientField = "real",
    tolerance: float = 1e-10,
) -> LinearizedCageObstructionReport:
    """Build first-order boundary and eigenstate obstruction maps.

    The boundary-only condition tests solvability of
    ``B delta_phi = -delta_B phi``.  When the internal blocks are supplied,
    the primary obstruction map additionally enforces the linearized
    eigenvalue equation and the gauge ``<phi|delta_phi> = 0``.

    For ``coefficient_field='real'``, real and imaginary parts are stacked so
    the returned coefficient nullspace corresponds to real coupling changes.
    """
    if coefficient_field not in {"real", "complex"}:
        raise ValueError("coefficient_field must be 'real' or 'complex'.")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")

    boundary = as_dense_array(boundary_matrix)
    if boundary.ndim != 2:
        raise ValueError("boundary_matrix must be 2D.")
    state = np.asarray(cage_state, dtype=np.complex128).reshape(-1)
    if state.size != boundary.shape[1]:
        raise ValueError("cage_state length must match the boundary column count.")
    state_norm = float(np.linalg.norm(state))
    if state_norm <= tolerance:
        raise ValueError("cage_state must have nonzero norm.")
    state = state / state_norm

    if len(boundary_perturbations) == 0:
        raise ValueError("boundary_perturbations must contain at least one matrix.")
    dense_boundary_perturbations = tuple(
        as_dense_array(perturbation) for perturbation in boundary_perturbations
    )
    if any(perturbation.shape != boundary.shape for perturbation in dense_boundary_perturbations):
        raise ValueError("every boundary perturbation must match boundary_matrix.shape.")

    if (internal_matrix is None) != (internal_perturbations is None):
        raise ValueError("internal_matrix and internal_perturbations must be supplied together.")
    if internal_perturbations is not None and len(internal_perturbations) != len(
        boundary_perturbations
    ):
        raise ValueError(
            "internal_perturbations must have the same length as boundary_perturbations."
        )

    dense_internal = None if internal_matrix is None else as_dense_array(internal_matrix)
    dense_internal_perturbations = (
        None
        if internal_perturbations is None
        else tuple(as_dense_array(perturbation) for perturbation in internal_perturbations)
    )
    if dense_internal is not None and dense_internal.shape != (state.size, state.size):
        raise ValueError("internal_matrix must be square with size cage_state.size.")
    if dense_internal_perturbations is not None and any(
        perturbation.shape != (state.size, state.size)
        for perturbation in dense_internal_perturbations
    ):
        raise ValueError("every internal perturbation must match internal_matrix.shape.")

    left_boundary_nullspace = nullspace_svd(boundary.conj().T, tolerance=tolerance)
    boundary_obstruction_columns = [
        left_boundary_nullspace.conj().T @ (perturbation @ state)
        for perturbation in dense_boundary_perturbations
    ]
    boundary_obstruction_matrix = np.column_stack(boundary_obstruction_columns).astype(
        np.complex128,
        copy=False,
    )

    tangent_operator: npt.NDArray[np.complex128] | None = None
    left_tangent_nullspace: npt.NDArray[np.complex128] | None = None
    eigenstate_obstruction_columns = boundary_obstruction_columns

    if dense_internal is not None and dense_internal_perturbations is not None:
        energy = complex(np.vdot(state, dense_internal @ state))
        boundary_zero_column = np.zeros((boundary.shape[0], 1), dtype=np.complex128)
        tangent_operator = np.block(
            [
                [boundary, boundary_zero_column],
                [
                    dense_internal - energy * np.eye(state.size, dtype=np.complex128),
                    -state[:, np.newaxis],
                ],
                [state.conj()[np.newaxis, :], np.zeros((1, 1), dtype=np.complex128)],
            ]
        )
        left_tangent_nullspace = nullspace_svd(
            tangent_operator.conj().T,
            tolerance=tolerance,
        )
        eigenstate_obstruction_columns = []
        for boundary_perturbation, internal_perturbation in zip(
            dense_boundary_perturbations,
            dense_internal_perturbations,
            strict=True,
        ):
            forcing = -np.concatenate(
                [
                    boundary_perturbation @ state,
                    internal_perturbation @ state,
                    np.zeros(1, dtype=np.complex128),
                ]
            )
            eigenstate_obstruction_columns.append(left_tangent_nullspace.conj().T @ forcing)

    obstruction_matrix = np.column_stack(eigenstate_obstruction_columns).astype(
        np.complex128,
        copy=False,
    )
    if coefficient_field == "real":
        constraint_matrix: npt.NDArray[np.float64] | npt.NDArray[np.complex128]
        constraint_matrix = np.vstack([obstruction_matrix.real, obstruction_matrix.imag])
    else:
        constraint_matrix = obstruction_matrix

    singular_values = scipy_linalg.svdvals(constraint_matrix).astype(
        np.float64,
        copy=False,
    )
    rank = int(np.sum(singular_values > tolerance))
    compatible_basis = nullspace_svd(constraint_matrix, tolerance=tolerance)
    if coefficient_field == "real":
        compatible_basis = np.real_if_close(compatible_basis, tol=1000).real

    perturbation_diagnostics: list[PerturbationCompatibilityDiagnostic] = []
    for perturbation_index, boundary_perturbation in enumerate(dense_boundary_perturbations):
        boundary_residual = float(np.linalg.norm(boundary_perturbation @ state))
        boundary_obstruction_residual = float(
            np.linalg.norm(boundary_obstruction_columns[perturbation_index])
        )
        eigenstate_obstruction_residual = float(
            np.linalg.norm(eigenstate_obstruction_columns[perturbation_index])
        )
        eigenvector_residual = np.nan
        if dense_internal_perturbations is not None:
            internal_action = dense_internal_perturbations[perturbation_index] @ state
            energy_shift = np.vdot(state, internal_action)
            eigenvector_residual = float(np.linalg.norm(internal_action - energy_shift * state))
        preserves_state = boundary_residual <= tolerance and (
            np.isnan(eigenvector_residual) or eigenvector_residual <= tolerance
        )
        perturbation_diagnostics.append(
            PerturbationCompatibilityDiagnostic(
                perturbation_index=perturbation_index,
                boundary_residual=boundary_residual,
                eigenvector_residual=float(eigenvector_residual),
                first_order_boundary_obstruction_residual=(boundary_obstruction_residual),
                first_order_eigenstate_obstruction_residual=(eigenstate_obstruction_residual),
                preserves_state=preserves_state,
                first_order_boundary_compatible=(boundary_obstruction_residual <= tolerance),
                first_order_eigenstate_compatible=(eigenstate_obstruction_residual <= tolerance),
            )
        )

    return LinearizedCageObstructionReport(
        coefficient_field=coefficient_field,
        n_parameters=len(boundary_perturbations),
        obstruction_matrix=obstruction_matrix,
        boundary_obstruction_matrix=boundary_obstruction_matrix,
        tangent_operator=tangent_operator,
        constraint_matrix=constraint_matrix,
        singular_values=singular_values,
        rank=rank,
        compatible_dimension=int(compatible_basis.shape[1]),
        compatible_coefficient_basis=compatible_basis,
        left_boundary_nullspace=left_boundary_nullspace,
        left_tangent_nullspace=left_tangent_nullspace,
        perturbation_diagnostics=tuple(perturbation_diagnostics),
        tolerance=tolerance,
    )


def fixed_cage_state_compatibility(
    boundary_matrix: object,
    cage_state: npt.ArrayLike,
    boundary_perturbations: Sequence[object],
    *,
    internal_perturbations: Sequence[object],
    coefficient_field: CoefficientField = "real",
    tolerance: float = 1e-10,
) -> FixedCageStateCompatibilityReport:
    """Find perturbation combinations that preserve one cage vector exactly.

    For an affine deformation ``H(lambda) = H0 + lambda V``, the compact state
    remains an exact eigenvector for every ``lambda`` precisely when

    ``delta_B phi = 0`` and
    ``(I - |phi><phi|) delta_A phi = 0``.

    The returned coefficient nullspace imposes these conditions collectively;
    individual perturbation terms need not preserve the state on their own.
    """
    if coefficient_field not in {"real", "complex"}:
        raise ValueError("coefficient_field must be 'real' or 'complex'.")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")
    if len(boundary_perturbations) == 0:
        raise ValueError("boundary_perturbations must contain at least one matrix.")
    if len(internal_perturbations) != len(boundary_perturbations):
        raise ValueError(
            "internal_perturbations must have the same length as boundary_perturbations."
        )

    boundary = as_dense_array(boundary_matrix)
    if boundary.ndim != 2:
        raise ValueError("boundary_matrix must be 2D.")
    state = np.asarray(cage_state, dtype=np.complex128).reshape(-1)
    if state.size != boundary.shape[1]:
        raise ValueError("cage_state length must match the boundary column count.")
    state_norm = float(np.linalg.norm(state))
    if state_norm <= tolerance:
        raise ValueError("cage_state must have nonzero norm.")
    state = state / state_norm

    dense_boundary_perturbations = tuple(
        as_dense_array(perturbation) for perturbation in boundary_perturbations
    )
    dense_internal_perturbations = tuple(
        as_dense_array(perturbation) for perturbation in internal_perturbations
    )
    if any(perturbation.shape != boundary.shape for perturbation in dense_boundary_perturbations):
        raise ValueError("every boundary perturbation must match boundary_matrix.shape.")
    if any(
        perturbation.shape != (state.size, state.size)
        for perturbation in dense_internal_perturbations
    ):
        raise ValueError("every internal perturbation must be square with size cage_state.size.")

    orthogonal_projector = np.eye(state.size, dtype=np.complex128) - np.outer(
        state,
        state.conj(),
    )
    action_columns = []
    for boundary_perturbation, internal_perturbation in zip(
        dense_boundary_perturbations,
        dense_internal_perturbations,
        strict=True,
    ):
        action_columns.append(
            np.concatenate(
                [
                    boundary_perturbation @ state,
                    orthogonal_projector @ (internal_perturbation @ state),
                ]
            )
        )
    action_matrix = np.column_stack(action_columns).astype(np.complex128, copy=False)
    if coefficient_field == "real":
        constraint_matrix: npt.NDArray[np.float64] | npt.NDArray[np.complex128]
        constraint_matrix = np.vstack([action_matrix.real, action_matrix.imag])
    else:
        constraint_matrix = action_matrix

    singular_values = scipy_linalg.svdvals(constraint_matrix).astype(
        np.float64,
        copy=False,
    )
    rank = int(np.sum(singular_values > tolerance))
    compatible_basis = nullspace_svd(constraint_matrix, tolerance=tolerance)
    if coefficient_field == "real":
        compatible_basis = np.real_if_close(compatible_basis, tol=1000).real

    return FixedCageStateCompatibilityReport(
        coefficient_field=coefficient_field,
        n_parameters=len(boundary_perturbations),
        action_matrix=action_matrix,
        constraint_matrix=constraint_matrix,
        singular_values=singular_values,
        rank=rank,
        compatible_dimension=int(compatible_basis.shape[1]),
        compatible_coefficient_basis=compatible_basis,
        perturbation_residuals=np.linalg.norm(action_matrix, axis=0).astype(
            np.float64,
            copy=False,
        ),
        tolerance=tolerance,
    )


def cage_jacobian_conditioning(
    internal_matrix: object,
    boundary_matrix: object,
    cage_state: npt.ArrayLike,
    *,
    support: Sequence[int] | None = None,
    tolerance: float = 1.0e-10,
) -> CageJacobianConditioningReport:
    """Evaluate the gauge-fixed Jacobian and ``Delta_cage``.

    The domain consists of state variations orthogonal to the normalized cage
    vector together with one energy variation.  This is the finite-dimensional
    Jacobian in Eq. (45) of the current deformation criterion.  Its smallest
    positive singular value is a conditioning scale after a deformation has
    passed the obstruction test; it does not by itself guarantee compatibility.
    """
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")
    internal = as_dense_array(internal_matrix)
    boundary = as_dense_array(boundary_matrix)
    if internal.ndim != 2 or internal.shape[0] != internal.shape[1]:
        raise ValueError("internal_matrix must be square.")
    if boundary.ndim != 2 or boundary.shape[1] != internal.shape[0]:
        raise ValueError("boundary_matrix must have one column per support state.")
    state = np.asarray(cage_state, dtype=np.complex128).reshape(-1)
    if state.size != internal.shape[0]:
        raise ValueError("cage_state length must match internal_matrix.shape[0].")
    norm = float(np.linalg.norm(state))
    if norm <= tolerance:
        raise ValueError("cage_state must have nonzero norm.")
    state = state / norm

    energy = complex(np.vdot(state, internal @ state))
    orthogonal_basis = nullspace_svd(state.conj()[np.newaxis, :], tolerance=tolerance)
    shifted = internal - energy * np.eye(state.size, dtype=np.complex128)
    state_column = -state[:, np.newaxis]
    boundary_energy_column = np.zeros((boundary.shape[0], 1), dtype=np.complex128)
    jacobian = np.block(
        [
            [shifted @ orthogonal_basis, state_column],
            [boundary @ orthogonal_basis, boundary_energy_column],
        ]
    ).astype(np.complex128, copy=False)

    singular_values = scipy_linalg.svdvals(jacobian).astype(np.float64, copy=False)
    rank = int(np.sum(singular_values > tolerance))
    nullity = int(jacobian.shape[1] - rank)
    positive = singular_values[singular_values > tolerance]
    gap = float(np.min(positive)) if positive.size else float("inf")

    internal_residual = float(np.linalg.norm(shifted @ state))
    boundary_residual = float(np.linalg.norm(boundary @ state))
    full_residual = float(np.hypot(internal_residual, boundary_residual))
    support_tuple = (
        tuple(range(state.size)) if support is None else tuple(int(index) for index in support)
    )
    if len(support_tuple) != state.size:
        raise ValueError("support length must match cage_state.size.")
    return CageJacobianConditioningReport(
        support=support_tuple,
        energy=energy,
        jacobian=jacobian,
        singular_values=singular_values,
        rank=rank,
        nullity=nullity,
        cage_gap=gap,
        internal_residual=internal_residual,
        boundary_residual=boundary_residual,
        full_residual=full_residual,
        tolerance=tolerance,
    )


def cage_jacobian_conditioning_from_hamiltonian(
    hamiltonian: object,
    support: Sequence[int] | npt.NDArray[np.integer],
    cage_state: npt.ArrayLike,
    *,
    tolerance: float = 1.0e-10,
) -> CageJacobianConditioningReport:
    """Build the cage Jacobian directly from a full Hamiltonian."""
    blocks = partition_cage_hamiltonian(hamiltonian, support)
    local_state = _normalized_local_state(
        cage_state,
        support=blocks.support,
        hilbert_size=blocks.hilbert_size,
        tolerance=tolerance,
    )
    return cage_jacobian_conditioning(
        blocks.internal,
        blocks.boundary,
        local_state,
        support=tuple(int(index) for index in blocks.support),
        tolerance=tolerance,
    )


def linearized_cage_obstruction_from_hamiltonians(
    base_hamiltonian: object,
    perturbations: Sequence[object],
    support: Sequence[int] | npt.NDArray[np.integer],
    cage_state: npt.ArrayLike,
    *,
    coefficient_field: CoefficientField = "real",
    tolerance: float = 1e-10,
) -> LinearizedCageObstructionReport:
    """Build the first-order obstruction map directly from full matrices."""
    base_blocks = partition_cage_hamiltonian(base_hamiltonian, support)
    perturbation_blocks = []
    for perturbation in perturbations:
        _validate_same_matrix_shape(base_hamiltonian, perturbation)
        perturbation_blocks.append(partition_cage_hamiltonian(perturbation, support))

    local_state = _normalized_local_state(
        cage_state,
        support=base_blocks.support,
        hilbert_size=base_blocks.hilbert_size,
        tolerance=tolerance,
    )
    return linearized_cage_obstruction(
        base_blocks.boundary,
        local_state,
        [blocks.boundary for blocks in perturbation_blocks],
        internal_matrix=base_blocks.internal,
        internal_perturbations=[blocks.internal for blocks in perturbation_blocks],
        coefficient_field=coefficient_field,
        tolerance=tolerance,
    )


def fixed_cage_state_compatibility_from_hamiltonians(
    base_hamiltonian: object,
    perturbations: Sequence[object],
    support: Sequence[int] | npt.NDArray[np.integer],
    cage_state: npt.ArrayLike,
    *,
    coefficient_field: CoefficientField = "real",
    tolerance: float = 1e-10,
) -> FixedCageStateCompatibilityReport:
    """Build the exact fixed-state compatibility space from full matrices."""
    base_blocks = partition_cage_hamiltonian(base_hamiltonian, support)
    perturbation_blocks = []
    for perturbation in perturbations:
        _validate_same_matrix_shape(base_hamiltonian, perturbation)
        perturbation_blocks.append(partition_cage_hamiltonian(perturbation, support))

    local_state = _normalized_local_state(
        cage_state,
        support=base_blocks.support,
        hilbert_size=base_blocks.hilbert_size,
        tolerance=tolerance,
    )
    return fixed_cage_state_compatibility(
        base_blocks.boundary,
        local_state,
        [blocks.boundary for blocks in perturbation_blocks],
        internal_perturbations=[blocks.internal for blocks in perturbation_blocks],
        coefficient_field=coefficient_field,
        tolerance=tolerance,
    )


def cage_compatibility_hierarchy_from_hamiltonians(
    base_hamiltonian: object,
    perturbations: Sequence[object],
    support: Sequence[int] | npt.NDArray[np.integer],
    cage_state: npt.ArrayLike,
    *,
    coefficient_field: CoefficientField = "real",
    tolerance: float = 1e-10,
) -> CageCompatibilityHierarchyReport:
    """Compare infinitesimal continuation and exact affine compatibility."""
    first_order = linearized_cage_obstruction_from_hamiltonians(
        base_hamiltonian,
        perturbations,
        support,
        cage_state,
        coefficient_field=coefficient_field,
        tolerance=tolerance,
    )
    fixed_state = fixed_cage_state_compatibility_from_hamiltonians(
        base_hamiltonian,
        perturbations,
        support,
        cage_state,
        coefficient_field=coefficient_field,
        tolerance=tolerance,
    )

    first_order_basis = np.asarray(first_order.compatible_coefficient_basis)
    fixed_state_basis = np.asarray(fixed_state.compatible_coefficient_basis)
    tangent_only_basis = subspace_complement_basis(
        first_order_basis,
        fixed_state_basis,
        tolerance=tolerance,
    )
    first_order_orthonormal = _orthonormal_columns(first_order_basis)
    fixed_state_orthonormal = _orthonormal_columns(fixed_state_basis)
    if fixed_state_orthonormal.shape[1] == 0:
        inclusion_residual = 0.0
    else:
        inclusion_residual = float(
            np.linalg.norm(
                fixed_state_orthonormal
                - first_order_orthonormal
                @ (first_order_orthonormal.conj().T @ fixed_state_orthonormal)
            )
        )

    if coefficient_field == "real":
        tangent_only_basis = np.real_if_close(tangent_only_basis, tol=1000).real

    return CageCompatibilityHierarchyReport(
        first_order=first_order,
        fixed_state=fixed_state,
        tangent_only_coefficient_basis=tangent_only_basis,
        fixed_subspace_inclusion_residual=inclusion_residual,
    )


def combine_perturbations_from_coefficients(
    perturbations: Sequence[object],
    coefficient_basis: npt.ArrayLike,
) -> tuple[object, ...]:
    """Convert coefficient-space basis vectors into Hamiltonian perturbations."""
    if len(perturbations) == 0:
        raise ValueError("perturbations must contain at least one matrix.")
    coefficients = np.asarray(coefficient_basis)
    if coefficients.ndim == 1:
        coefficients = coefficients[:, np.newaxis]
    if coefficients.ndim != 2 or coefficients.shape[0] != len(perturbations):
        raise ValueError("coefficient_basis must have shape (n_perturbations, n_combinations).")

    return tuple(
        _matrix_linear_combination(
            _zero_matrix_like(perturbations[0]),
            perturbations,
            coefficients[:, combination_index],
        )
        for combination_index in range(coefficients.shape[1])
    )


def subspace_principal_overlaps(
    basis_a: npt.ArrayLike,
    basis_b: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Return cosines of principal angles between two column subspaces."""
    orthonormal_a = _orthonormal_columns(basis_a)
    orthonormal_b = _orthonormal_columns(basis_b)
    if orthonormal_a.shape[1] == 0 or orthonormal_b.shape[1] == 0:
        return np.zeros(0, dtype=np.float64)
    overlaps = scipy_linalg.svdvals(orthonormal_a.conj().T @ orthonormal_b)
    return np.clip(overlaps.real, 0.0, 1.0).astype(np.float64, copy=False)


def subspace_complement_basis(
    parent_basis: npt.ArrayLike,
    child_basis: npt.ArrayLike,
    *,
    tolerance: float = 1e-10,
) -> npt.NDArray[np.complex128]:
    """Return the part of ``span(parent_basis)`` orthogonal to ``child_basis``.

    ``child_basis`` is required to lie inside the parent subspace up to the
    supplied tolerance. Empty column bases are supported.
    """
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")
    parent_array = np.asarray(parent_basis, dtype=np.complex128)
    child_array = np.asarray(child_basis, dtype=np.complex128)
    if parent_array.ndim == 1:
        parent_array = parent_array[:, np.newaxis]
    if child_array.ndim == 1:
        child_array = child_array[:, np.newaxis]
    if parent_array.ndim != 2 or child_array.ndim != 2:
        raise ValueError("parent_basis and child_basis must be vectors or 2D column bases.")
    if parent_array.shape[0] != child_array.shape[0]:
        raise ValueError("parent_basis and child_basis must have the same row count.")

    parent_orthonormal = _orthonormal_columns(parent_array)
    child_orthonormal = _orthonormal_columns(child_array)
    if child_orthonormal.shape[1] == 0:
        return parent_orthonormal
    if parent_orthonormal.shape[1] == 0:
        raise ValueError("a nonempty child subspace cannot lie inside an empty parent subspace.")

    child_projection = parent_orthonormal @ (parent_orthonormal.conj().T @ child_orthonormal)
    inclusion_residual = float(np.linalg.norm(child_orthonormal - child_projection))
    if inclusion_residual > tolerance * max(1.0, float(np.linalg.norm(child_orthonormal))):
        raise ValueError("child_basis is not contained in parent_basis within tolerance.")

    complement_coordinates = nullspace_svd(
        child_orthonormal.conj().T @ parent_orthonormal,
        tolerance=tolerance,
    )
    return (parent_orthonormal @ complement_coordinates).astype(
        np.complex128,
        copy=False,
    )


def estimate_power_law_exponent(
    parameters: npt.ArrayLike,
    residuals: npt.ArrayLike,
    *,
    minimum_parameter: float = 0.0,
    minimum_residual: float = 0.0,
) -> float | None:
    """Estimate ``residual ~ |parameter|**p`` by a log-log fit."""
    parameter_array = np.abs(np.asarray(parameters, dtype=np.float64).reshape(-1))
    residual_array = np.asarray(residuals, dtype=np.float64).reshape(-1)
    if parameter_array.size != residual_array.size:
        raise ValueError("parameters and residuals must have the same length.")
    mask = (
        np.isfinite(parameter_array)
        & np.isfinite(residual_array)
        & (parameter_array > minimum_parameter)
        & (residual_array > minimum_residual)
    )
    if np.count_nonzero(mask) < 2:
        return None
    design = np.column_stack(
        [
            np.log(parameter_array[mask]),
            np.ones(np.count_nonzero(mask), dtype=np.float64),
        ]
    )
    exponent, _intercept = np.linalg.lstsq(
        design,
        np.log(residual_array[mask]),
        rcond=None,
    )[0]
    return float(exponent)


def subspace_projector_distance(
    basis_a: npt.ArrayLike,
    basis_b: npt.ArrayLike,
) -> float:
    """Return the spectral norm ``||P_A - P_B||_2``."""
    orthonormal_a = _orthonormal_columns(basis_a)
    orthonormal_b = _orthonormal_columns(basis_b)
    if orthonormal_a.shape[0] != orthonormal_b.shape[0]:
        raise ValueError("basis_a and basis_b must have the same ambient dimension.")
    projector_a = orthonormal_a @ orthonormal_a.conj().T
    projector_b = orthonormal_b @ orthonormal_b.conj().T
    return float(np.linalg.norm(projector_a - projector_b, ord=2))


def _matrix_subblock(
    matrix: object,
    row_indices: npt.NDArray[np.int64],
    column_indices: npt.NDArray[np.int64],
) -> object:
    if scipy_sparse.issparse(matrix):
        return matrix[row_indices][:, column_indices]
    dense_matrix = np.asarray(matrix)
    return dense_matrix[np.ix_(row_indices, column_indices)]


def _validate_same_matrix_shape(matrix_a: object, matrix_b: object) -> None:
    if getattr(matrix_a, "shape", None) != getattr(matrix_b, "shape", None):
        raise ValueError("matrix shapes must match.")


def _normalized_local_state(
    state: npt.ArrayLike,
    *,
    support: npt.NDArray[np.int64],
    hilbert_size: int,
    tolerance: float,
) -> npt.NDArray[np.complex128]:
    state_array = np.asarray(state, dtype=np.complex128).reshape(-1)
    if state_array.size == hilbert_size:
        local_state = state_array[support]
        outside_weight = float(np.linalg.norm(np.delete(state_array, support)) ** 2)
        if outside_weight > tolerance**2:
            raise ValueError("full state has nonzero weight outside the proposed support.")
    elif state_array.size == support.size:
        local_state = state_array
    else:
        raise ValueError("state length must equal support size or full Hilbert-space size.")

    state_norm = float(np.linalg.norm(local_state))
    if state_norm <= tolerance:
        raise ValueError("state must have nonzero norm.")
    return (local_state / state_norm).astype(np.complex128, copy=False)


def _reference_comparison_basis(
    *,
    reference_subspace: npt.ArrayLike | None,
    reference_state: npt.NDArray[np.complex128] | None,
    fallback_basis: npt.NDArray[np.complex128],
    support: npt.NDArray[np.int64],
    hilbert_size: int,
    tolerance: float,
) -> npt.NDArray[np.complex128]:
    if reference_subspace is not None:
        return _normalized_local_subspace(
            reference_subspace,
            support=support,
            hilbert_size=hilbert_size,
            tolerance=tolerance,
        )
    if reference_state is not None:
        return reference_state[:, np.newaxis]
    return fallback_basis


def _normalized_local_subspace(
    subspace: npt.ArrayLike,
    *,
    support: npt.NDArray[np.int64],
    hilbert_size: int,
    tolerance: float,
) -> npt.NDArray[np.complex128]:
    array = np.asarray(subspace, dtype=np.complex128)
    if array.ndim == 1:
        array = array[:, np.newaxis]
    if array.ndim != 2:
        raise ValueError("reference_subspace must be a vector or a 2D column basis.")

    if array.shape[0] == hilbert_size:
        support_mask = np.zeros(hilbert_size, dtype=bool)
        support_mask[support] = True
        if np.linalg.norm(array[~support_mask, :]) > tolerance:
            raise ValueError("reference_subspace has nonzero weight outside the proposed support.")
        array = array[support, :]
    elif array.shape[0] != support.size:
        raise ValueError("reference_subspace row count must equal support size or Hilbert size.")

    if array.shape[1] == 0:
        return np.zeros((support.size, 0), dtype=np.complex128)
    left_vectors, singular_values, _right_vectors_h = scipy_linalg.svd(
        array,
        full_matrices=False,
    )
    rank = int(np.sum(singular_values > tolerance))
    if rank == 0:
        raise ValueError("reference_subspace must contain a nonzero vector.")
    return left_vectors[:, :rank].astype(np.complex128, copy=False)


def _orthonormal_columns(basis: npt.ArrayLike) -> npt.NDArray[np.complex128]:
    array = np.asarray(basis, dtype=np.complex128)
    if array.ndim != 2:
        raise ValueError("basis must be a 2D array with column vectors.")
    if array.shape[1] == 0:
        return np.zeros((array.shape[0], 0), dtype=np.complex128)
    orthonormal, _ = np.linalg.qr(array)
    return orthonormal[:, : array.shape[1]].astype(np.complex128, copy=False)


def _cage_eigensystem(
    internal_matrix: npt.NDArray[np.complex128],
    invariant_basis: npt.NDArray[np.complex128],
    *,
    tolerance: float,
) -> tuple[
    npt.NDArray[np.complex128],
    npt.NDArray[np.complex128],
    bool,
]:
    if invariant_basis.shape[1] == 0:
        return (
            np.zeros(0, dtype=np.complex128),
            np.zeros((internal_matrix.shape[0], 0), dtype=np.complex128),
            True,
        )

    reduced_matrix = invariant_basis.conj().T @ internal_matrix @ invariant_basis
    hermiticity_scale = max(1.0, float(np.linalg.norm(reduced_matrix)))
    is_hermitian = (
        np.linalg.norm(reduced_matrix - reduced_matrix.conj().T) <= tolerance * hermiticity_scale
    )
    if is_hermitian:
        eigenvalues, reduced_eigenvectors = np.linalg.eigh(
            0.5 * (reduced_matrix + reduced_matrix.conj().T)
        )
        eigenvalues = eigenvalues.astype(np.complex128)
    else:
        eigenvalues, reduced_eigenvectors = np.linalg.eig(reduced_matrix)

    cage_eigenvectors = invariant_basis @ reduced_eigenvectors
    column_norms = np.linalg.norm(cage_eigenvectors, axis=0)
    cage_eigenvectors = cage_eigenvectors / column_norms[np.newaxis, :]
    return eigenvalues, cage_eigenvectors, is_hermitian


def _minimum_leakage_internal_eigenstate(
    internal_matrix: npt.NDArray[np.complex128],
    boundary_matrix: npt.NDArray[np.complex128],
    *,
    reference_state: npt.NDArray[np.complex128],
    tolerance: float,
) -> tuple[npt.NDArray[np.complex128], complex]:
    """Select the least-leaking state in the closest internal eigenspace."""
    hermiticity_scale = max(1.0, float(np.linalg.norm(internal_matrix)))
    is_hermitian = (
        np.linalg.norm(internal_matrix - internal_matrix.conj().T) <= tolerance * hermiticity_scale
    )
    if not is_hermitian:
        eigenvalues, eigenvectors = np.linalg.eig(internal_matrix)
        overlaps = np.abs(eigenvectors.conj().T @ reference_state)
        selected_index = int(np.argmax(overlaps))
        state = eigenvectors[:, selected_index]
        state = state / np.linalg.norm(state)
        return state.astype(np.complex128, copy=False), complex(eigenvalues[selected_index])

    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (internal_matrix + internal_matrix.conj().T))
    energy_scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    best_state: npt.NDArray[np.complex128] | None = None
    best_energy = 0.0
    best_projection_overlap = -1.0
    best_boundary_residual = np.inf

    group_start = 0
    while group_start < eigenvalues.size:
        group_end = group_start + 1
        while (
            group_end < eigenvalues.size
            and abs(eigenvalues[group_end] - eigenvalues[group_start]) <= tolerance * energy_scale
        ):
            group_end += 1

        eigenspace = eigenvectors[:, group_start:group_end]
        reference_coordinates = eigenspace.conj().T @ reference_state
        projection_overlap = float(np.linalg.norm(reference_coordinates))

        leakage_gram = (boundary_matrix @ eigenspace).conj().T @ (boundary_matrix @ eigenspace)
        leakage_eigenvalues, leakage_eigenvectors = np.linalg.eigh(
            0.5 * (leakage_gram + leakage_gram.conj().T)
        )
        minimum_leakage_eigenvalue = float(max(0.0, leakage_eigenvalues[0]))
        minimum_mask = leakage_eigenvalues <= minimum_leakage_eigenvalue + tolerance * max(
            1.0, float(np.max(np.abs(leakage_eigenvalues)))
        )
        minimum_leakage_basis = leakage_eigenvectors[:, minimum_mask]
        projected_reference = minimum_leakage_basis.conj().T @ reference_coordinates
        if np.linalg.norm(projected_reference) > tolerance:
            local_state = minimum_leakage_basis @ (
                projected_reference / np.linalg.norm(projected_reference)
            )
        else:
            local_state = minimum_leakage_basis[:, 0]
        state = eigenspace @ local_state
        state = state / np.linalg.norm(state)
        boundary_residual = float(np.linalg.norm(boundary_matrix @ state))

        is_better = projection_overlap > best_projection_overlap + tolerance
        is_tied_and_less_leaky = (
            abs(projection_overlap - best_projection_overlap) <= tolerance
            and boundary_residual < best_boundary_residual
        )
        if is_better or is_tied_and_less_leaky:
            best_state = state.astype(np.complex128, copy=False)
            best_energy = float(eigenvalues[group_start])
            best_projection_overlap = projection_overlap
            best_boundary_residual = boundary_residual

        group_start = group_end

    if best_state is None:  # pragma: no cover - internal matrix is nonempty
        raise RuntimeError("could not select an internal support eigenstate.")
    return best_state, complex(best_energy)


def _matched_cage_eigensubspace(
    internal_matrix: npt.NDArray[np.complex128],
    invariant_basis: npt.NDArray[np.complex128],
    *,
    reference_basis: npt.NDArray[np.complex128],
    target_dimension: int,
    tolerance: float,
) -> npt.NDArray[np.complex128]:
    if target_dimension < 0:
        raise ValueError("target_dimension must be non-negative.")
    if target_dimension == 0:
        return np.zeros((internal_matrix.shape[0], 0), dtype=np.complex128)

    eigenvalues, eigenvectors, is_hermitian = _cage_eigensystem(
        internal_matrix,
        invariant_basis,
        tolerance=tolerance,
    )
    if eigenvectors.shape[1] == 0:
        return eigenvectors

    candidate_vectors: list[npt.NDArray[np.complex128]] = []
    candidate_scores: list[float] = []
    if is_hermitian:
        energy_scale = max(1.0, float(np.max(np.abs(eigenvalues))))
        group_start = 0
        while group_start < eigenvalues.size:
            group_end = group_start + 1
            while (
                group_end < eigenvalues.size
                and abs(eigenvalues[group_end] - eigenvalues[group_start])
                <= tolerance * energy_scale
            ):
                group_end += 1

            group_vectors = eigenvectors[:, group_start:group_end]
            overlap = group_vectors.conj().T @ reference_basis
            left_vectors, singular_values, _right_vectors_h = scipy_linalg.svd(
                overlap,
                full_matrices=True,
            )
            rotated_vectors = group_vectors @ left_vectors
            scores = np.zeros(rotated_vectors.shape[1], dtype=np.float64)
            scores[: singular_values.size] = singular_values**2
            candidate_vectors.extend(
                rotated_vectors[:, index] for index in range(rotated_vectors.shape[1])
            )
            candidate_scores.extend(float(score) for score in scores)
            group_start = group_end
    else:
        scores = np.sum(
            np.abs(reference_basis.conj().T @ eigenvectors) ** 2,
            axis=0,
        )
        candidate_vectors.extend(eigenvectors[:, index] for index in range(eigenvectors.shape[1]))
        candidate_scores.extend(float(score) for score in scores)

    selected_count = min(target_dimension, len(candidate_vectors))
    selected_indices = np.argsort(candidate_scores)[::-1][:selected_count]
    return np.column_stack([candidate_vectors[int(index)] for index in selected_indices]).astype(
        np.complex128, copy=False
    )


def _continued_cage_eigenstate(
    internal_matrix: npt.NDArray[np.complex128],
    invariant_basis: npt.NDArray[np.complex128],
    *,
    previous_state: npt.NDArray[np.complex128] | None,
    tolerance: float,
) -> tuple[npt.NDArray[np.complex128] | None, complex | None]:
    if invariant_basis.shape[1] == 0:
        return None, None

    if previous_state is None:
        eigenvalues, eigenvectors, _is_hermitian = _cage_eigensystem(
            internal_matrix,
            invariant_basis,
            tolerance=tolerance,
        )
        selected_state = eigenvectors[:, int(np.argmin(np.real(eigenvalues)))]
    else:
        selected_basis = _matched_cage_eigensubspace(
            internal_matrix,
            invariant_basis,
            reference_basis=previous_state[:, np.newaxis],
            target_dimension=1,
            tolerance=tolerance,
        )
        if selected_basis.shape[1] == 0:
            return None, None
        selected_state = selected_basis[:, 0]
        overlap = np.vdot(previous_state, selected_state)
        if abs(overlap) > tolerance:
            selected_state = selected_state * np.exp(-1j * np.angle(overlap))

    selected_energy = complex(np.vdot(selected_state, internal_matrix @ selected_state))
    return selected_state, selected_energy


def summarize_cage_record_stability(
    base_hamiltonian: object,
    perturbations: Sequence[object],
    records: Sequence[object],
    *,
    classification_reports: Sequence[object] | None = None,
    coefficient_field: CoefficientField = "real",
    tolerance: float = 1e-10,
) -> tuple[CageRecordStabilitySummary, ...]:
    """Compare preferred cage representatives inside one degenerate manifold.

    The records are intentionally treated as a chosen basis, such as the output
    of the IPR degenerate-basis strategy.  Manifold-level statements remain
    basis independent, whereas the returned record-wise quantities diagnose
    whether one preferred localized representative is more fragile than the
    others.
    """
    if classification_reports is not None and len(classification_reports) != len(records):
        raise ValueError("classification_reports length must match records.")

    summaries: list[CageRecordStabilitySummary] = []
    for record_index, record in enumerate(records):
        state = np.asarray(record.local_state, dtype=np.complex128).reshape(-1)
        norm = float(np.linalg.norm(state))
        if norm <= tolerance:
            raise ValueError("record local_state must have nonzero norm.")
        state = state / norm
        hierarchy = cage_compatibility_hierarchy_from_hamiltonians(
            base_hamiltonian,
            perturbations,
            record.support,
            state,
            coefficient_field=coefficient_field,
            tolerance=tolerance,
        )
        classification = (
            None if classification_reports is None else classification_reports[record_index]
        )
        summaries.append(
            CageRecordStabilitySummary(
                record_index=record_index,
                signature=tuple(int(value) for value in record.signature),
                support_size=int(np.asarray(record.support).size),
                inverse_participation_ratio=float(np.sum(np.abs(state) ** 4)),
                classification_label=(
                    None if classification is None else str(classification.label)
                ),
                n_collective_cancellation_source_probes=(
                    None
                    if classification is None
                    else int(classification.n_collective_cancellation_source_probes)
                ),
                formal_compatible_dimension=hierarchy.first_order.compatible_dimension,
                exact_fixed_state_dimension=hierarchy.fixed_state.compatible_dimension,
                tangent_only_dimension=hierarchy.tangent_only_dimension,
            )
        )
    return tuple(summaries)


def _matrix_linear_combination(
    base_matrix: object,
    perturbations: Sequence[object],
    coefficients: npt.ArrayLike,
) -> object:
    coefficient_array = np.asarray(coefficients).reshape(-1)
    if coefficient_array.size != len(perturbations):
        raise ValueError("coefficients length must match perturbations.")

    result = (
        base_matrix.copy() if hasattr(base_matrix, "copy") else np.array(base_matrix, copy=True)
    )
    for coefficient, perturbation in zip(coefficient_array, perturbations, strict=True):
        result = result + coefficient * perturbation
    return result


def _zero_matrix_like(matrix: object) -> object:
    if scipy_sparse.issparse(matrix):
        return scipy_sparse.csr_matrix(matrix.shape, dtype=np.complex128)
    return np.zeros_like(np.asarray(matrix), dtype=np.complex128)


def _aggregate_random_samples(
    samples: Sequence[RandomCageStabilitySample],
    *,
    strength: float,
) -> RandomCageStabilityAggregate:
    selected = [sample for sample in samples if sample.strength == strength]
    gaps = np.asarray(
        [sample.interference_gap for sample in selected if sample.interference_gap is not None],
        dtype=np.float64,
    )
    n_survived = sum(int(sample.cage_survives) for sample in selected)
    return RandomCageStabilityAggregate(
        strength=strength,
        n_samples=len(selected),
        n_survived=n_survived,
        survival_fraction=float(n_survived / len(selected)),
        minimum_interference_gap=None if gaps.size == 0 else float(np.min(gaps)),
        median_interference_gap=None if gaps.size == 0 else float(np.median(gaps)),
        median_projector_distance=float(
            np.median([sample.projector_distance_from_reference for sample in selected])
        ),
        median_minimum_principal_overlap=float(
            np.median([sample.minimum_principal_overlap for sample in selected])
        ),
    )


@dataclass(frozen=True, slots=True)
class FixedCageManifoldCompatibilityReport:
    """Exact perturbation space preserving a cage subspace as a whole.

    The basis vectors may rotate inside the manifold.  A perturbation is
    compatible when it neither leaks through the support boundary nor couples
    the selected manifold to its orthogonal complement inside the support.
    """

    coefficient_field: CoefficientField
    manifold_dimension: int
    n_parameters: int
    action_matrix: npt.NDArray[np.complex128]
    constraint_matrix: npt.NDArray[np.float64] | npt.NDArray[np.complex128]
    singular_values: npt.NDArray[np.float64]
    rank: int
    compatible_dimension: int
    compatible_coefficient_basis: npt.NDArray[np.float64] | npt.NDArray[np.complex128]
    perturbation_residuals: npt.NDArray[np.float64]
    tolerance: float

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "manifold_dimension": self.manifold_dimension,
            "n_parameters": self.n_parameters,
            "compatible_dimension": self.compatible_dimension,
            "rank": self.rank,
            "perturbation_residuals": tuple(float(x) for x in self.perturbation_residuals),
            "tolerance": self.tolerance,
        }


@dataclass(frozen=True, slots=True)
class ChiralIndexReport:
    """Finite-dimensional chiral index diagnostic for an off-diagonal block.

    For ``K = [[0, A^†], [A, 0]]``, the index is
    ``dim ker(A) - dim ker(A^†) = n_plus - n_minus``.  Only the signed index is
    stable under arbitrary chiral-symmetric rank-preserving deformations; any
    additional paired zero modes are interference-protected rather than
    index-protected.
    """

    n_plus: int
    n_minus: int
    rank: int
    kernel_plus_dimension: int
    kernel_minus_dimension: int
    index: int
    index_protected_plus_zero_modes: int
    index_protected_minus_zero_modes: int
    paired_zero_mode_count: int
    singular_gap: float | None
    tolerance: float

    @property
    def total_zero_mode_count(self) -> int:
        return self.kernel_plus_dimension + self.kernel_minus_dimension

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "n_plus": self.n_plus,
            "n_minus": self.n_minus,
            "rank": self.rank,
            "kernel_plus_dimension": self.kernel_plus_dimension,
            "kernel_minus_dimension": self.kernel_minus_dimension,
            "index": self.index,
            "index_protected_plus_zero_modes": self.index_protected_plus_zero_modes,
            "index_protected_minus_zero_modes": self.index_protected_minus_zero_modes,
            "paired_zero_mode_count": self.paired_zero_mode_count,
            "singular_gap": self.singular_gap,
            "tolerance": self.tolerance,
        }


def fixed_cage_manifold_compatibility(
    boundary_matrix: object,
    manifold_basis: npt.ArrayLike,
    boundary_perturbations: Sequence[object],
    *,
    internal_perturbations: Sequence[object],
    coefficient_field: CoefficientField = "real",
    tolerance: float = 1e-10,
) -> FixedCageManifoldCompatibilityReport:
    """Find affine perturbations preserving an entire cage manifold exactly."""
    if coefficient_field not in {"real", "complex"}:
        raise ValueError("coefficient_field must be 'real' or 'complex'.")
    if len(boundary_perturbations) == 0:
        raise ValueError("boundary_perturbations must contain at least one matrix.")
    if len(internal_perturbations) != len(boundary_perturbations):
        raise ValueError("internal_perturbations must match boundary_perturbations.")
    boundary = as_dense_array(boundary_matrix)
    raw_basis = np.asarray(manifold_basis, dtype=np.complex128)
    if raw_basis.ndim != 2 or raw_basis.shape[0] != boundary.shape[1]:
        raise ValueError("manifold_basis must have one row per support configuration.")
    basis, _ = np.linalg.qr(raw_basis)
    if basis.shape[1] == 0:
        raise ValueError("manifold_basis must contain at least one vector.")
    complement = nullspace_svd(basis.conj().T, tolerance=tolerance)
    columns = []
    for delta_b, delta_a in zip(boundary_perturbations, internal_perturbations, strict=True):
        dense_b = as_dense_array(delta_b)
        dense_a = as_dense_array(delta_a)
        if dense_b.shape != boundary.shape:
            raise ValueError("boundary perturbation shape mismatch.")
        if dense_a.shape != (boundary.shape[1], boundary.shape[1]):
            raise ValueError("internal perturbation shape mismatch.")
        columns.append(
            np.concatenate(
                [
                    (dense_b @ basis).reshape(-1),
                    (complement.conj().T @ dense_a @ basis).reshape(-1),
                ]
            )
        )
    action = np.column_stack(columns).astype(np.complex128, copy=False)
    constraint = np.vstack([action.real, action.imag]) if coefficient_field == "real" else action
    singular_values = scipy_linalg.svdvals(constraint).astype(np.float64, copy=False)
    rank = int(np.sum(singular_values > tolerance))
    compatible = nullspace_svd(constraint, tolerance=tolerance)
    if coefficient_field == "real":
        compatible = np.real_if_close(compatible, tol=1000).real
    return FixedCageManifoldCompatibilityReport(
        coefficient_field=coefficient_field,
        manifold_dimension=int(basis.shape[1]),
        n_parameters=len(boundary_perturbations),
        action_matrix=action,
        constraint_matrix=constraint,
        singular_values=singular_values,
        rank=rank,
        compatible_dimension=int(compatible.shape[1]),
        compatible_coefficient_basis=compatible,
        perturbation_residuals=np.linalg.norm(action, axis=0).astype(np.float64),
        tolerance=tolerance,
    )


def fixed_cage_manifold_compatibility_from_hamiltonians(
    base_hamiltonian: object,
    perturbations: Sequence[object],
    *,
    support: Sequence[int],
    manifold_states: npt.ArrayLike,
    coefficient_field: CoefficientField = "real",
    tolerance: float = 1e-10,
) -> FixedCageManifoldCompatibilityReport:
    """Hamiltonian wrapper for :func:`fixed_cage_manifold_compatibility`."""
    blocks = partition_cage_hamiltonian(base_hamiltonian, support)
    support_array = np.asarray(blocks.support, dtype=np.int64)
    states = np.asarray(manifold_states, dtype=np.complex128)
    if states.ndim != 2:
        raise ValueError("manifold_states must be a matrix with states in columns.")
    if states.shape[0] == blocks.hilbert_size:
        local_states = states[support_array, :]
    elif states.shape[0] == support_array.size:
        local_states = states
    else:
        raise ValueError("manifold_states row count must match Hilbert or support size.")
    perturbation_blocks = tuple(partition_cage_hamiltonian(p, support) for p in perturbations)
    return fixed_cage_manifold_compatibility(
        blocks.boundary,
        local_states,
        tuple(item.boundary for item in perturbation_blocks),
        internal_perturbations=tuple(item.internal for item in perturbation_blocks),
        coefficient_field=coefficient_field,
        tolerance=tolerance,
    )


def diagnose_chiral_index(
    off_diagonal_block: object,
    *,
    trim_isolated_rows: bool = True,
    trim_isolated_columns: bool = False,
    tolerance: float = 1e-10,
) -> ChiralIndexReport:
    """Diagnose index-protected and paired zero modes of a chiral block ``A``.

    Isolated rows are removed by default because a support-to-boundary block
    often includes complementary configurations that are not adjacent to the
    selected support.  Their trivial zero modes should not be mistaken for a
    chiral index of the interference network.
    """
    matrix = as_dense_array(off_diagonal_block)
    if matrix.ndim != 2:
        raise ValueError("off_diagonal_block must be 2D.")
    if trim_isolated_rows:
        matrix = matrix[np.linalg.norm(matrix, axis=1) > tolerance, :]
    if trim_isolated_columns:
        matrix = matrix[:, np.linalg.norm(matrix, axis=0) > tolerance]
    singular_values = scipy_linalg.svdvals(matrix).astype(np.float64, copy=False)
    rank = int(np.sum(singular_values > tolerance))
    n_minus, n_plus = matrix.shape
    kernel_plus = int(n_plus - rank)
    kernel_minus = int(n_minus - rank)
    index = int(kernel_plus - kernel_minus)
    nonzero = singular_values[singular_values > tolerance]
    gap = None if nonzero.size == 0 else float(np.min(nonzero))
    return ChiralIndexReport(
        n_plus=int(n_plus),
        n_minus=int(n_minus),
        rank=rank,
        kernel_plus_dimension=kernel_plus,
        kernel_minus_dimension=kernel_minus,
        index=index,
        index_protected_plus_zero_modes=max(index, 0),
        index_protected_minus_zero_modes=max(-index, 0),
        paired_zero_mode_count=min(kernel_plus, kernel_minus),
        singular_gap=gap,
        tolerance=tolerance,
    )


@dataclass(frozen=True, slots=True)
class RegionalChiralIndexEntry:
    """Chiral-index and zero-mode data for one selected support region."""

    region_index: int
    support: tuple[int, ...]
    active_boundary_size: int
    chiral_index: ChiralIndexReport
    target_weight: float | None
    target_boundary_residual: float | None
    target_is_regional_zero_mode: bool | None

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "region_index": self.region_index,
            "support_size": len(self.support),
            "active_boundary_size": self.active_boundary_size,
            "index": self.chiral_index.index,
            "kernel_plus_dimension": self.chiral_index.kernel_plus_dimension,
            "kernel_minus_dimension": self.chiral_index.kernel_minus_dimension,
            "index_protected_plus_zero_modes": (self.chiral_index.index_protected_plus_zero_modes),
            "paired_zero_mode_count": self.chiral_index.paired_zero_mode_count,
            "target_weight": self.target_weight,
            "target_boundary_residual": self.target_boundary_residual,
            "target_is_regional_zero_mode": self.target_is_regional_zero_mode,
        }


@dataclass(frozen=True, slots=True)
class LocalityRestrictedChiralProfileReport:
    """Regional chiral profile for a state under a prescribed locality cover."""

    entries: tuple[RegionalChiralIndexEntry, ...]
    covered_support: tuple[int, ...]
    uncovered_target_weight: float | None
    n_regional_target_zero_modes: int
    tolerance: float

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "n_regions": len(self.entries),
            "covered_support_size": len(self.covered_support),
            "uncovered_target_weight": self.uncovered_target_weight,
            "n_regional_target_zero_modes": self.n_regional_target_zero_modes,
            "tolerance": self.tolerance,
        }


@dataclass(frozen=True, slots=True)
class RegionalChiralKernelSpanReport:
    """Basis-independent overlap of a target manifold with regional kernels."""

    n_regions: int
    regional_raw_kernel_dimension: int
    regional_span_dimension: int
    target_dimension: int
    principal_overlaps: npt.NDArray[np.float64]
    captured_target_dimension: int
    uncaptured_target_dimension: int
    target_projector_residual: float
    tolerance: float

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "n_regions": self.n_regions,
            "regional_raw_kernel_dimension": self.regional_raw_kernel_dimension,
            "regional_span_dimension": self.regional_span_dimension,
            "target_dimension": self.target_dimension,
            "principal_overlaps": tuple(float(x) for x in self.principal_overlaps),
            "captured_target_dimension": self.captured_target_dimension,
            "uncaptured_target_dimension": self.uncaptured_target_dimension,
            "target_projector_residual": self.target_projector_residual,
            "tolerance": self.tolerance,
        }


def diagnose_locality_restricted_chiral_profile(
    hamiltonian: object,
    regions: Sequence[Sequence[int]],
    *,
    target_state: npt.ArrayLike | None = None,
    tolerance: float = 1e-10,
) -> LocalityRestrictedChiralProfileReport:
    """Diagnose chiral zero modes separately on prescribed support regions.

    The regions define the locality restriction.  For each region, the function
    forms the support-to-complement block and computes its finite-dimensional
    chiral index.  If ``target_state`` is supplied, its projection onto each
    region is tested against that regional boundary map.
    """
    matrix = as_dense_array(hamiltonian)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("hamiltonian must be a square matrix.")
    state = None
    if target_state is not None:
        state = np.asarray(target_state, dtype=np.complex128).reshape(-1)
        if state.size != matrix.shape[0]:
            raise ValueError("target_state size must match the Hilbert dimension.")
    entries: list[RegionalChiralIndexEntry] = []
    covered: set[int] = set()
    n_regional_zero_modes = 0
    for region_index, raw_region in enumerate(regions):
        support = tuple(sorted({int(index) for index in raw_region}))
        if not support:
            raise ValueError("each region must contain at least one index.")
        covered.update(support)
        blocks = partition_cage_hamiltonian(matrix, support)
        chiral = diagnose_chiral_index(blocks.boundary, tolerance=tolerance)
        active_boundary_size = int(
            np.sum(np.linalg.norm(as_dense_array(blocks.boundary), axis=1) > tolerance)
        )
        weight = residual = None
        is_zero = None
        if state is not None:
            local = state[np.asarray(support, dtype=np.int64)]
            weight = float(np.vdot(local, local).real)
            residual = float(np.linalg.norm(as_dense_array(blocks.boundary) @ local))
            is_zero = bool(weight > tolerance and residual <= tolerance)
            n_regional_zero_modes += int(is_zero)
        entries.append(
            RegionalChiralIndexEntry(
                region_index=region_index,
                support=support,
                active_boundary_size=active_boundary_size,
                chiral_index=chiral,
                target_weight=weight,
                target_boundary_residual=residual,
                target_is_regional_zero_mode=is_zero,
            )
        )
    uncovered_weight = None
    if state is not None:
        mask = np.ones(state.size, dtype=bool)
        if covered:
            mask[np.asarray(sorted(covered), dtype=np.int64)] = False
        uncovered_weight = float(np.vdot(state[mask], state[mask]).real)
    return LocalityRestrictedChiralProfileReport(
        entries=tuple(entries),
        covered_support=tuple(sorted(covered)),
        uncovered_target_weight=uncovered_weight,
        n_regional_target_zero_modes=n_regional_zero_modes,
        tolerance=tolerance,
    )


def regional_chiral_kernel_span(
    hamiltonian: object,
    regions: Sequence[Sequence[int]],
    target_manifold: npt.ArrayLike,
    *,
    tolerance: float = 1e-10,
) -> RegionalChiralKernelSpanReport:
    """Compare a target manifold with the direct span of regional chiral kernels."""
    matrix = as_dense_array(hamiltonian)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("hamiltonian must be a square matrix.")
    target = np.asarray(target_manifold, dtype=np.complex128)
    if target.ndim == 1:
        target = target[:, None]
    if target.ndim != 2 or target.shape[0] != matrix.shape[0]:
        raise ValueError("target_manifold must have one row per Hilbert state.")
    target_basis, _ = np.linalg.qr(target)
    embedded: list[npt.NDArray[np.complex128]] = []
    raw_dimension = 0
    for raw_region in regions:
        support = tuple(sorted({int(index) for index in raw_region}))
        if not support:
            raise ValueError("each region must contain at least one index.")
        blocks = partition_cage_hamiltonian(matrix, support)
        kernel = nullspace_svd(as_dense_array(blocks.boundary), tolerance=tolerance)
        raw_dimension += int(kernel.shape[1])
        for column in range(kernel.shape[1]):
            vector = np.zeros(matrix.shape[0], dtype=np.complex128)
            vector[np.asarray(support, dtype=np.int64)] = kernel[:, column]
            embedded.append(vector)
    if embedded:
        regional_matrix = np.column_stack(embedded)
        regional_basis = scipy_linalg.orth(regional_matrix, rcond=tolerance)
    else:
        regional_basis = np.zeros((matrix.shape[0], 0), dtype=np.complex128)
    overlaps = subspace_principal_overlaps(target_basis, regional_basis)
    captured = int(np.sum(overlaps >= 1.0 - tolerance))
    projector = regional_basis @ regional_basis.conj().T
    residual = float(np.linalg.norm((np.eye(matrix.shape[0]) - projector) @ target_basis))
    return RegionalChiralKernelSpanReport(
        n_regions=len(regions),
        regional_raw_kernel_dimension=raw_dimension,
        regional_span_dimension=int(regional_basis.shape[1]),
        target_dimension=int(target_basis.shape[1]),
        principal_overlaps=overlaps,
        captured_target_dimension=captured,
        uncaptured_target_dimension=int(target_basis.shape[1] - captured),
        target_projector_residual=residual,
        tolerance=tolerance,
    )


@dataclass(frozen=True, slots=True)
class RegionalCageQuotientReport:
    """Relative quotient of a target cage manifold by regional kernels.

    The quotient is represented canonically by the component of the target
    manifold orthogonal to the regional-kernel span.  It is therefore
    basis-independent up to a unitary rotation within the quotient itself.
    """

    target_dimension: int
    regional_span_dimension: int
    intersection_dimension: int
    quotient_dimension: int
    inclusion_residual: float
    quotient_basis: npt.NDArray[np.complex128]
    quotient_projector: npt.NDArray[np.complex128]
    tolerance: float

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "target_dimension": self.target_dimension,
            "regional_span_dimension": self.regional_span_dimension,
            "intersection_dimension": self.intersection_dimension,
            "quotient_dimension": self.quotient_dimension,
            "inclusion_residual": self.inclusion_residual,
            "tolerance": self.tolerance,
        }


def regional_cage_quotient(
    hamiltonian: object,
    regions: Sequence[Sequence[int]],
    target_manifold: npt.ArrayLike,
    *,
    tolerance: float = 1e-10,
) -> RegionalCageQuotientReport:
    """Construct ``target_manifold / regional_kernel_span`` numerically.

    The function first embeds every regional right kernel in the full Hilbert
    space, then projects the target manifold onto the orthogonal complement of
    their span.  If the regional span is contained in the target manifold, the
    resulting dimension is the usual quotient dimension.  ``inclusion_residual``
    diagnoses violations of that containment.
    """
    matrix = as_dense_array(hamiltonian)
    target = np.asarray(target_manifold, dtype=np.complex128)
    if target.ndim == 1:
        target = target[:, None]
    if target.ndim != 2 or target.shape[0] != matrix.shape[0]:
        raise ValueError("target_manifold must have one row per Hilbert state.")
    target_basis = scipy_linalg.orth(target, rcond=tolerance)
    embedded: list[npt.NDArray[np.complex128]] = []
    for raw_region in regions:
        support = tuple(sorted({int(index) for index in raw_region}))
        if not support:
            raise ValueError("each region must contain at least one index.")
        blocks = partition_cage_hamiltonian(matrix, support)
        kernel = nullspace_svd(as_dense_array(blocks.boundary), tolerance=tolerance)
        for column in range(kernel.shape[1]):
            vector = np.zeros(matrix.shape[0], dtype=np.complex128)
            vector[np.asarray(support, dtype=np.int64)] = kernel[:, column]
            embedded.append(vector)
    if embedded:
        regional_basis = scipy_linalg.orth(np.column_stack(embedded), rcond=tolerance)
    else:
        regional_basis = np.zeros((matrix.shape[0], 0), dtype=np.complex128)
    target_projector = target_basis @ target_basis.conj().T
    inclusion_residual = float(
        np.linalg.norm((np.eye(matrix.shape[0]) - target_projector) @ regional_basis)
    )
    regional_projector = regional_basis @ regional_basis.conj().T
    quotient_raw = (np.eye(matrix.shape[0]) - regional_projector) @ target_basis
    quotient_basis = scipy_linalg.orth(quotient_raw, rcond=tolerance)
    quotient_projector = quotient_basis @ quotient_basis.conj().T
    overlaps = subspace_principal_overlaps(target_basis, regional_basis)
    intersection = int(np.sum(overlaps >= 1.0 - tolerance))
    return RegionalCageQuotientReport(
        target_dimension=int(target_basis.shape[1]),
        regional_span_dimension=int(regional_basis.shape[1]),
        intersection_dimension=intersection,
        quotient_dimension=int(quotient_basis.shape[1]),
        inclusion_residual=inclusion_residual,
        quotient_basis=quotient_basis,
        quotient_projector=quotient_projector,
        tolerance=tolerance,
    )


@dataclass(frozen=True, slots=True)
class ManyBodyCLSGeneratorOrbitEntry:
    """Dimension of the translation orbit generated by one local cage seed."""

    generator_index: int
    orbit_dimension: int

    def to_summary_dict(self) -> dict[str, int]:
        return {
            "generator_index": self.generator_index,
            "orbit_dimension": self.orbit_dimension,
        }


@dataclass(frozen=True, slots=True)
class ManyBodyCLSCompletenessReport:
    """Many-body analogue of the CLS completeness defect.

    ``target_manifold`` is the complete exact cage manifold under study.
    ``local_generators`` contains bounded or otherwise locally generated cage
    seeds.  Their full translation orbit spans ``local_generator_basis``.  If
    this span is contained in the target manifold, the quotient

    ``target_manifold / local_generator_span``

    is the direct many-body counterpart of the flat-band sector missed by
    translated compact localized states.

    A positive quotient dimension is a finite-size completeness defect.  It is
    not, by itself, a topological invariant: persistence along a thermodynamic
    sequence and a noncontractible or otherwise quantized label must be tested
    separately.
    """

    hilbert_dimension: int
    target_dimension: int
    generator_seed_count: int
    generator_seed_span_dimension: int
    translated_generator_span_dimension: int
    intersection_dimension: int
    quotient_dimension: int
    generator_containment_residual: float
    orbit_entries: tuple[ManyBodyCLSGeneratorOrbitEntry, ...]
    target_basis: npt.NDArray[np.complex128]
    local_generator_basis: npt.NDArray[np.complex128]
    quotient_basis: npt.NDArray[np.complex128]
    quotient_projector: npt.NDArray[np.complex128]
    tolerance: float

    @property
    def is_locally_complete(self) -> bool:
        return self.quotient_dimension == 0

    @property
    def has_completeness_defect(self) -> bool:
        return self.quotient_dimension > 0

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "hilbert_dimension": self.hilbert_dimension,
            "target_dimension": self.target_dimension,
            "generator_seed_count": self.generator_seed_count,
            "generator_seed_span_dimension": self.generator_seed_span_dimension,
            "translated_generator_span_dimension": (self.translated_generator_span_dimension),
            "intersection_dimension": self.intersection_dimension,
            "quotient_dimension": self.quotient_dimension,
            "generator_containment_residual": self.generator_containment_residual,
            "is_locally_complete": self.is_locally_complete,
            "has_completeness_defect": self.has_completeness_defect,
            "orbit_dimensions": tuple(entry.orbit_dimension for entry in self.orbit_entries),
            "tolerance": self.tolerance,
        }


@dataclass(frozen=True, slots=True)
class ManyBodyCLSTranslationSector:
    """Multiplicity of one Abelian translation character."""

    momentum_indices: tuple[int, ...]
    momenta: tuple[float, ...]
    target_multiplicity: int
    local_generator_multiplicity: int
    quotient_multiplicity: int

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "momentum_indices": self.momentum_indices,
            "momenta": self.momenta,
            "target_multiplicity": self.target_multiplicity,
            "local_generator_multiplicity": self.local_generator_multiplicity,
            "quotient_multiplicity": self.quotient_multiplicity,
        }


@dataclass(frozen=True, slots=True)
class ManyBodyTopologicalLocalizationReport:
    """CLS-completeness defect resolved under commuting translations.

    This report deliberately uses the term ``candidate`` rather than assigning
    a winding number automatically.  Momentum resolution is well defined from
    the translation representation, while a real-space homology class requires
    additional geometric information about how a quotient state winds around
    the physical torus.
    """

    completeness: ManyBodyCLSCompletenessReport
    translation_orders: tuple[int, ...]
    target_translation_residual: float
    local_translation_residual: float
    quotient_translation_residual: float
    translation_commutator_residual: float
    sectors: tuple[ManyBodyCLSTranslationSector, ...]
    quotient_characters: tuple[complex, ...] | None
    tolerance: float

    @property
    def quotient_sector_signature(self) -> tuple[tuple[tuple[int, ...], int], ...]:
        return tuple(
            (sector.momentum_indices, sector.quotient_multiplicity)
            for sector in self.sectors
            if sector.quotient_multiplicity
        )

    @property
    def has_symmetry_resolved_quotient(self) -> bool:
        return self.completeness.quotient_dimension > 0 and (
            self.quotient_translation_residual <= 10.0 * self.tolerance
        )

    def to_summary_dict(self) -> dict[str, object]:
        return {
            **self.completeness.to_summary_dict(),
            "translation_orders": self.translation_orders,
            "target_translation_residual": self.target_translation_residual,
            "local_translation_residual": self.local_translation_residual,
            "quotient_translation_residual": self.quotient_translation_residual,
            "translation_commutator_residual": (self.translation_commutator_residual),
            "quotient_sector_signature": self.quotient_sector_signature,
            "quotient_characters": self.quotient_characters,
            "has_symmetry_resolved_quotient": self.has_symmetry_resolved_quotient,
        }


@dataclass(frozen=True, slots=True)
class ManyBodyCLSCompletenessSequencePoint:
    """One system size in a many-body CLS-completeness sequence."""

    size_label: str
    linear_sizes: tuple[int, ...]
    target_dimension: int
    local_generator_span_dimension: int
    quotient_dimension: int
    interference_gap: float | None = None
    exact_open_bond_dimension: int | None = None
    quotient_sector_signature: tuple[tuple[tuple[int, ...], int], ...] = ()

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "size_label": self.size_label,
            "linear_sizes": self.linear_sizes,
            "target_dimension": self.target_dimension,
            "local_generator_span_dimension": self.local_generator_span_dimension,
            "quotient_dimension": self.quotient_dimension,
            "interference_gap": self.interference_gap,
            "exact_open_bond_dimension": self.exact_open_bond_dimension,
            "quotient_sector_signature": self.quotient_sector_signature,
        }


@dataclass(frozen=True, slots=True)
class ManyBodyCLSCompletenessSequenceReport:
    """Finite-size persistence test for a CLS-completeness defect."""

    model_label: str
    points: tuple[ManyBodyCLSCompletenessSequencePoint, ...]

    @property
    def defect_dimensions(self) -> tuple[int, ...]:
        return tuple(point.quotient_dimension for point in self.points)

    @property
    def classification(self) -> str:
        defects = self.defect_dimensions
        if not defects:
            return "empty"
        if all(value == 0 for value in defects):
            return "locally_complete"
        if defects[0] > 0 and all(value == 0 for value in defects[1:]):
            return "finite_size_only_quotient"
        if all(value > 0 for value in defects):
            return "persistent_quotient_candidate"
        return "mixed_or_inconclusive"

    @property
    def has_persistent_defect(self) -> bool:
        return bool(self.points) and all(value > 0 for value in self.defect_dimensions)

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "model_label": self.model_label,
            "classification": self.classification,
            "defect_dimensions": self.defect_dimensions,
            "has_persistent_defect": self.has_persistent_defect,
            "points": tuple(point.to_summary_dict() for point in self.points),
        }


def _validate_hilbert_permutation(
    permutation: npt.ArrayLike,
    hilbert_dimension: int,
) -> npt.NDArray[np.int64]:
    values = np.asarray(permutation, dtype=np.int64).reshape(-1)
    if values.size != hilbert_dimension:
        raise ValueError("translation permutation has incompatible dimension.")
    if (
        np.unique(values).size != hilbert_dimension
        or np.any(values < 0)
        or np.any(values >= hilbert_dimension)
    ):
        raise ValueError("translation permutations must permute range(hilbert_dimension).")
    return values


def _apply_hilbert_permutation(
    vectors: npt.NDArray[np.complex128],
    permutation: npt.NDArray[np.int64],
) -> npt.NDArray[np.complex128]:
    transformed = np.zeros_like(vectors)
    transformed[permutation, :] = vectors
    return transformed


def _translation_group_permutations(
    permutations: Sequence[npt.ArrayLike],
    orders: Sequence[int],
    hilbert_dimension: int,
) -> tuple[npt.NDArray[np.int64], ...]:
    if len(permutations) != len(orders):
        raise ValueError("translation_permutations and translation_orders must match.")
    generators = tuple(
        _validate_hilbert_permutation(permutation, hilbert_dimension)
        for permutation in permutations
    )
    normalized_orders = tuple(int(order) for order in orders)
    if any(order <= 0 for order in normalized_orders):
        raise ValueError("translation orders must be positive.")

    identity = np.arange(hilbert_dimension, dtype=np.int64)
    powered: list[tuple[npt.NDArray[np.int64], ...]] = []
    for generator, order in zip(generators, normalized_orders, strict=True):
        powers = [identity]
        for _ in range(1, order):
            powers.append(generator[powers[-1]])
        if not np.array_equal(generator[powers[-1]], identity):
            raise ValueError("translation permutation order is inconsistent.")
        powered.append(tuple(power.copy() for power in powers))

    elements: list[npt.NDArray[np.int64]] = []
    for exponents in itertools.product(*(range(order) for order in normalized_orders)):
        combined = identity
        for axis, exponent in enumerate(exponents):
            combined = powered[axis][exponent][combined]
        elements.append(combined.copy())
    return tuple(elements)


def diagnose_many_body_cls_completeness(
    target_manifold: npt.ArrayLike,
    local_generators: npt.ArrayLike | None = None,
    *,
    translation_permutations: Sequence[npt.ArrayLike] = (),
    translation_orders: Sequence[int] = (),
    tolerance: float = 1e-10,
    require_generator_containment: bool = True,
) -> ManyBodyCLSCompletenessReport:
    """Compute the quotient missed by translated local cage generators."""
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")
    target = np.asarray(target_manifold, dtype=np.complex128)
    if target.ndim == 1:
        target = target[:, None]
    if target.ndim != 2 or target.shape[0] == 0:
        raise ValueError("target_manifold must be a nonempty vector or matrix.")
    hilbert_dimension = int(target.shape[0])
    target_basis = scipy_linalg.orth(target, rcond=tolerance)
    if target_basis.shape[1] == 0:
        raise ValueError("target_manifold must span a nonzero subspace.")

    if local_generators is None:
        seeds = np.zeros((hilbert_dimension, 0), dtype=np.complex128)
    else:
        seeds = np.asarray(local_generators, dtype=np.complex128)
        if seeds.ndim == 1:
            seeds = seeds[:, None]
        if seeds.ndim != 2 or seeds.shape[0] != hilbert_dimension:
            raise ValueError("local_generators must have one row per Hilbert state.")
    seed_basis = scipy_linalg.orth(seeds, rcond=tolerance)

    group = _translation_group_permutations(
        translation_permutations,
        translation_orders,
        hilbert_dimension,
    )
    if not group:
        group = (np.arange(hilbert_dimension, dtype=np.int64),)

    orbit_entries: list[ManyBodyCLSGeneratorOrbitEntry] = []
    orbit_vectors: list[npt.NDArray[np.complex128]] = []
    for generator_index in range(seeds.shape[1]):
        seed = seeds[:, generator_index : generator_index + 1]
        one_orbit = [_apply_hilbert_permutation(seed, element) for element in group]
        orbit_matrix = np.column_stack(one_orbit)
        orbit_dimension = int(scipy_linalg.orth(orbit_matrix, rcond=tolerance).shape[1])
        orbit_entries.append(
            ManyBodyCLSGeneratorOrbitEntry(
                generator_index=generator_index,
                orbit_dimension=orbit_dimension,
            )
        )
        orbit_vectors.extend(one_orbit)
    if orbit_vectors:
        local_basis = scipy_linalg.orth(np.column_stack(orbit_vectors), rcond=tolerance)
    else:
        local_basis = np.zeros((hilbert_dimension, 0), dtype=np.complex128)

    target_projector = target_basis @ target_basis.conj().T
    containment_residual = float(
        np.linalg.norm((np.eye(hilbert_dimension) - target_projector) @ local_basis)
    )
    if require_generator_containment and containment_residual > 10.0 * tolerance:
        raise ValueError(
            "translated local generators are not contained in the target manifold; "
            f"residual={containment_residual:.3e}."
        )
    overlaps = subspace_principal_overlaps(target_basis, local_basis)
    intersection = int(np.sum(overlaps >= 1.0 - tolerance))
    local_projector = local_basis @ local_basis.conj().T
    quotient_raw = (np.eye(hilbert_dimension) - local_projector) @ target_basis
    quotient_basis = scipy_linalg.orth(quotient_raw, rcond=tolerance)
    quotient_projector = quotient_basis @ quotient_basis.conj().T
    return ManyBodyCLSCompletenessReport(
        hilbert_dimension=hilbert_dimension,
        target_dimension=int(target_basis.shape[1]),
        generator_seed_count=int(seeds.shape[1]),
        generator_seed_span_dimension=int(seed_basis.shape[1]),
        translated_generator_span_dimension=int(local_basis.shape[1]),
        intersection_dimension=intersection,
        quotient_dimension=int(quotient_basis.shape[1]),
        generator_containment_residual=containment_residual,
        orbit_entries=tuple(orbit_entries),
        target_basis=target_basis,
        local_generator_basis=local_basis,
        quotient_basis=quotient_basis,
        quotient_projector=quotient_projector,
        tolerance=tolerance,
    )


def _abelian_translation_multiplicities(
    representations: Sequence[npt.NDArray[np.complex128]],
    orders: Sequence[int],
) -> dict[tuple[int, ...], int]:
    if not representations:
        return {(): 0}
    dimension = int(representations[0].shape[0])
    if dimension == 0:
        return {
            tuple(indices): 0
            for indices in itertools.product(*(range(int(order)) for order in orders))
        }
    powers = [
        tuple(np.linalg.matrix_power(rep, exponent) for exponent in range(int(order)))
        for rep, order in zip(representations, orders, strict=True)
    ]
    group_exponents = tuple(itertools.product(*(range(int(order)) for order in orders)))
    traces: dict[tuple[int, ...], complex] = {}
    for exponents in group_exponents:
        product = np.eye(dimension, dtype=np.complex128)
        for axis, exponent in enumerate(exponents):
            product = powers[axis][exponent] @ product
        traces[tuple(exponents)] = complex(np.trace(product))

    group_size = float(np.prod(np.asarray(orders, dtype=np.int64)))
    multiplicities: dict[tuple[int, ...], int] = {}
    for momentum_indices in itertools.product(*(range(int(order)) for order in orders)):
        value = 0.0j
        for exponents, trace in traces.items():
            phase = np.exp(
                -2.0j
                * np.pi
                * sum(
                    momentum_index * exponent / float(order)
                    for momentum_index, exponent, order in zip(
                        momentum_indices,
                        exponents,
                        orders,
                        strict=True,
                    )
                )
            )
            value += phase * trace
        multiplicities[tuple(int(index) for index in momentum_indices)] = max(
            0,
            int(np.rint(value.real / group_size)),
        )
    return multiplicities


def diagnose_many_body_topological_localization(
    target_manifold: npt.ArrayLike,
    local_generators: npt.ArrayLike | None,
    *,
    translation_permutations: Sequence[npt.ArrayLike],
    translation_orders: Sequence[int],
    tolerance: float = 1e-10,
) -> ManyBodyTopologicalLocalizationReport:
    """Resolve a many-body CLS-completeness quotient by lattice momentum."""
    completeness = diagnose_many_body_cls_completeness(
        target_manifold,
        local_generators,
        translation_permutations=translation_permutations,
        translation_orders=translation_orders,
        tolerance=tolerance,
    )
    permutations = tuple(
        _validate_hilbert_permutation(permutation, completeness.hilbert_dimension)
        for permutation in translation_permutations
    )
    orders = tuple(int(order) for order in translation_orders)
    target_representations: list[npt.NDArray[np.complex128]] = []
    local_representations: list[npt.NDArray[np.complex128]] = []
    quotient_representations: list[npt.NDArray[np.complex128]] = []
    target_residuals: list[float] = []
    local_residuals: list[float] = []
    quotient_residuals: list[float] = []
    for permutation in permutations:
        representation, residual = _subspace_symmetry_representation(
            completeness.target_basis,
            permutation,
        )
        target_representations.append(representation)
        target_residuals.append(residual)
        representation, residual = _subspace_symmetry_representation(
            completeness.local_generator_basis,
            permutation,
        )
        local_representations.append(representation)
        local_residuals.append(residual)
        representation, residual = _subspace_symmetry_representation(
            completeness.quotient_basis,
            permutation,
        )
        quotient_representations.append(representation)
        quotient_residuals.append(residual)

    commutator_residual = 0.0
    for first, second in itertools.combinations(target_representations, 2):
        commutator_residual = max(
            commutator_residual,
            float(np.linalg.norm(first @ second - second @ first)),
        )
    target_multiplicities = _abelian_translation_multiplicities(
        target_representations,
        orders,
    )
    local_multiplicities = _abelian_translation_multiplicities(
        local_representations,
        orders,
    )
    quotient_multiplicities = _abelian_translation_multiplicities(
        quotient_representations,
        orders,
    )
    sectors = tuple(
        ManyBodyCLSTranslationSector(
            momentum_indices=tuple(momentum_indices),
            momenta=tuple(
                2.0 * np.pi * index / float(order)
                for index, order in zip(momentum_indices, orders, strict=True)
            ),
            target_multiplicity=target_multiplicities[momentum_indices],
            local_generator_multiplicity=local_multiplicities[momentum_indices],
            quotient_multiplicity=quotient_multiplicities[momentum_indices],
        )
        for momentum_indices in itertools.product(*(range(order) for order in orders))
    )
    quotient_characters: tuple[complex, ...] | None = None
    if completeness.quotient_dimension == 1:
        quotient_characters = tuple(
            complex(representation[0, 0]) for representation in quotient_representations
        )
    return ManyBodyTopologicalLocalizationReport(
        completeness=completeness,
        translation_orders=orders,
        target_translation_residual=max(target_residuals, default=0.0),
        local_translation_residual=max(local_residuals, default=0.0),
        quotient_translation_residual=max(quotient_residuals, default=0.0),
        translation_commutator_residual=commutator_residual,
        sectors=sectors,
        quotient_characters=quotient_characters,
        tolerance=tolerance,
    )


@dataclass(frozen=True, slots=True)
class SignedBoundaryCycle:
    """Gauge-invariant signed holonomy on one bipartite boundary cycle."""

    rows: tuple[int, ...]
    columns: tuple[int, ...]
    edge_indices: tuple[int, ...]
    sign: int
    log_absolute_holonomy: float


@dataclass(frozen=True, slots=True)
class SignedBoundaryHolonomyReport:
    """Discrete and continuous cycle data of a support-to-boundary map."""

    n_rows: int
    n_columns: int
    n_edges: int
    n_components: int
    cycle_rank: int
    positive_cycle_count: int
    negative_cycle_count: int
    zero_edge_count: int
    cycles: tuple[SignedBoundaryCycle, ...]
    tolerance: float

    @property
    def sign_signature(self) -> tuple[int, ...]:
        return tuple(cycle.sign for cycle in self.cycles)

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "n_edges": self.n_edges,
            "n_components": self.n_components,
            "cycle_rank": self.cycle_rank,
            "positive_cycle_count": self.positive_cycle_count,
            "negative_cycle_count": self.negative_cycle_count,
            "zero_edge_count": self.zero_edge_count,
            "sign_signature": self.sign_signature,
            "tolerance": self.tolerance,
        }


@dataclass(frozen=True, slots=True)
class RelativeMod2CycleReport:
    """Relative cycle space of a full boundary graph modulo regional cycles."""

    full_cycle_dimension: int
    regional_cycle_span_dimension: int
    relative_cycle_dimension: int
    n_edges: int
    n_regions: int
    full_cycle_basis: npt.NDArray[np.uint8]
    regional_cycle_basis: npt.NDArray[np.uint8]
    relative_cycle_basis: npt.NDArray[np.uint8]
    edge_labels: tuple[tuple[int, int], ...]
    tolerance: float

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "full_cycle_dimension": self.full_cycle_dimension,
            "regional_cycle_span_dimension": self.regional_cycle_span_dimension,
            "relative_cycle_dimension": self.relative_cycle_dimension,
            "n_edges": self.n_edges,
            "n_regions": self.n_regions,
            "tolerance": self.tolerance,
        }


def _boundary_edge_labels(
    boundary: npt.NDArray[np.complex128], tolerance: float
) -> tuple[tuple[int, int], ...]:
    return tuple(
        (int(row), int(column))
        for row, column in zip(*np.nonzero(np.abs(boundary) > tolerance), strict=True)
    )


def _fundamental_bipartite_cycles(
    boundary: npt.NDArray[np.complex128],
    *,
    tolerance: float,
) -> tuple[tuple[tuple[str, int], ...], ...]:
    """Return a deterministic fundamental cycle basis of a bipartite graph."""
    import networkx as nx

    graph = nx.Graph()
    for row in range(boundary.shape[0]):
        graph.add_node(("r", row))
    for column in range(boundary.shape[1]):
        graph.add_node(("c", column))
    for row, column in _boundary_edge_labels(boundary, tolerance):
        graph.add_edge(("r", row), ("c", column))
    cycles: list[tuple[tuple[str, int], ...]] = []
    for component in sorted(
        nx.connected_components(graph), key=lambda nodes: min(nodes) if nodes else ("", -1)
    ):
        subgraph = graph.subgraph(component)
        root = min(component) if component else None
        for cycle in nx.cycle_basis(subgraph, root=root):
            if cycle:
                start = min(range(len(cycle)), key=lambda index: cycle[index])
                rotated = cycle[start:] + cycle[:start]
                reversed_cycle = [rotated[0], *reversed(rotated[1:])]
                canonical = min(tuple(rotated), tuple(reversed_cycle))
                cycles.append(canonical)
    return tuple(sorted(cycles))


def diagnose_signed_boundary_holonomy(
    boundary: object,
    *,
    tolerance: float = 1e-10,
) -> SignedBoundaryHolonomyReport:
    """Compute real signed holonomies on a fundamental cycle basis.

    Row and column rescalings cancel from the alternating cycle product.  The
    sign is therefore a discrete gauge invariant as long as no active edge
    crosses zero.  Complex-valued matrices are rejected because their natural
    invariant is a U(1) phase rather than a Z2 sign.
    """
    import networkx as nx

    matrix = np.asarray(as_dense_array(boundary), dtype=np.complex128)
    if matrix.ndim != 2:
        raise ValueError("boundary must be a matrix.")
    if np.max(np.abs(matrix.imag), initial=0.0) > tolerance:
        raise ValueError("signed holonomy requires a real boundary matrix.")
    real = matrix.real
    edge_labels = _boundary_edge_labels(matrix, tolerance)
    edge_to_index = {edge: index for index, edge in enumerate(edge_labels)}
    cycles_raw = _fundamental_bipartite_cycles(matrix, tolerance=tolerance)
    cycles: list[SignedBoundaryCycle] = []
    for nodes in cycles_raw:
        values: list[float] = []
        cycle_edges: list[int] = []
        rows: list[int] = []
        columns: list[int] = []
        for index, node in enumerate(nodes):
            next_node = nodes[(index + 1) % len(nodes)]
            if node[0] == "r":
                row, column = node[1], next_node[1]
            else:
                row, column = next_node[1], node[1]
            values.append(float(real[row, column]))
            cycle_edges.append(edge_to_index[(row, column)])
            if node[0] == "r":
                rows.append(node[1])
            else:
                columns.append(node[1])
        numerator = values[0::2]
        denominator = values[1::2]
        sign = int(np.prod(np.sign(numerator)) * np.prod(np.sign(denominator)))
        log_abs = float(np.sum(np.log(np.abs(numerator))) - np.sum(np.log(np.abs(denominator))))
        cycles.append(
            SignedBoundaryCycle(
                rows=tuple(rows),
                columns=tuple(columns),
                edge_indices=tuple(cycle_edges),
                sign=sign,
                log_absolute_holonomy=log_abs,
            )
        )
    graph = nx.Graph()
    graph.add_nodes_from(("r", row) for row in range(matrix.shape[0]))
    graph.add_nodes_from(("c", column) for column in range(matrix.shape[1]))
    graph.add_edges_from((("r", row), ("c", column)) for row, column in edge_labels)
    components = nx.number_connected_components(graph)
    cycle_rank = len(edge_labels) - graph.number_of_nodes() + components
    return SignedBoundaryHolonomyReport(
        n_rows=matrix.shape[0],
        n_columns=matrix.shape[1],
        n_edges=len(edge_labels),
        n_components=components,
        cycle_rank=int(cycle_rank),
        positive_cycle_count=sum(cycle.sign > 0 for cycle in cycles),
        negative_cycle_count=sum(cycle.sign < 0 for cycle in cycles),
        zero_edge_count=int(np.sum(np.abs(real) <= tolerance)),
        cycles=tuple(cycles),
        tolerance=tolerance,
    )


def _gf2_row_reduce(matrix: npt.NDArray[np.uint8]) -> tuple[npt.NDArray[np.uint8], tuple[int, ...]]:
    reduced = np.asarray(matrix, dtype=np.uint8).copy() % 2
    if reduced.ndim != 2:
        raise ValueError("GF(2) matrix must be two-dimensional.")
    pivots: list[int] = []
    row = 0
    for column in range(reduced.shape[1]):
        candidates = np.flatnonzero(reduced[row:, column])
        if candidates.size == 0:
            continue
        pivot = row + int(candidates[0])
        reduced[[row, pivot]] = reduced[[pivot, row]]
        for other in range(reduced.shape[0]):
            if other != row and reduced[other, column]:
                reduced[other] ^= reduced[row]
        pivots.append(column)
        row += 1
        if row == reduced.shape[0]:
            break
    return reduced[:row], tuple(pivots)


def _cycle_incidence_basis(
    boundary: npt.NDArray[np.complex128],
    edge_to_index: dict[tuple[int, int], int],
    *,
    tolerance: float,
) -> npt.NDArray[np.uint8]:
    cycles = _fundamental_bipartite_cycles(boundary, tolerance=tolerance)
    vectors = np.zeros((len(cycles), len(edge_to_index)), dtype=np.uint8)
    for cycle_index, nodes in enumerate(cycles):
        for index, node in enumerate(nodes):
            next_node = nodes[(index + 1) % len(nodes)]
            if node[0] == "r":
                edge = (node[1], next_node[1])
            else:
                edge = (next_node[1], node[1])
            vectors[cycle_index, edge_to_index[edge]] ^= 1
    reduced, _ = _gf2_row_reduce(vectors)
    return reduced


def diagnose_relative_mod2_cycles(
    boundary: object,
    regions: Sequence[Sequence[int]],
    *,
    tolerance: float = 1e-10,
) -> RelativeMod2CycleReport:
    """Compute full boundary cycles modulo cycles internal to support regions.

    ``regions`` contain column indices of the supplied boundary matrix.  Every
    regional graph includes those columns and all incident boundary rows.  The
    resulting quotient is a graph invariant over GF(2); it does not by itself
    identify which quotient cycles participate in a particular cage vector.
    """
    matrix = np.asarray(as_dense_array(boundary), dtype=np.complex128)
    if matrix.ndim != 2:
        raise ValueError("boundary must be a matrix.")
    edge_labels = _boundary_edge_labels(matrix, tolerance)
    edge_to_index = {edge: index for index, edge in enumerate(edge_labels)}
    full_basis = _cycle_incidence_basis(matrix, edge_to_index, tolerance=tolerance)
    regional_vectors: list[npt.NDArray[np.uint8]] = []
    for raw_region in regions:
        columns = tuple(sorted({int(column) for column in raw_region}))
        if any(column < 0 or column >= matrix.shape[1] for column in columns):
            raise IndexError("regional column index is outside the boundary matrix.")
        regional = np.zeros_like(matrix)
        regional[:, columns] = matrix[:, columns]
        basis = _cycle_incidence_basis(regional, edge_to_index, tolerance=tolerance)
        regional_vectors.extend(basis)
    if regional_vectors:
        regional_basis, _ = _gf2_row_reduce(np.asarray(regional_vectors, dtype=np.uint8))
    else:
        regional_basis = np.zeros((0, len(edge_labels)), dtype=np.uint8)
    combined = np.vstack((regional_basis, full_basis))
    combined_reduced, _ = _gf2_row_reduce(combined)
    relative_dimension = combined_reduced.shape[0] - regional_basis.shape[0]
    quotient_rows: list[npt.NDArray[np.uint8]] = []
    current = regional_basis.copy()
    current_rank = current.shape[0]
    for vector in full_basis:
        candidate = np.vstack((current, vector[None, :]))
        reduced, _ = _gf2_row_reduce(candidate)
        if reduced.shape[0] > current_rank:
            quotient_rows.append(vector.copy())
            current = reduced
            current_rank = reduced.shape[0]
    relative_basis = (
        np.asarray(quotient_rows, dtype=np.uint8)
        if quotient_rows
        else np.zeros((0, len(edge_labels)), dtype=np.uint8)
    )
    return RelativeMod2CycleReport(
        full_cycle_dimension=int(full_basis.shape[0]),
        regional_cycle_span_dimension=int(regional_basis.shape[0]),
        relative_cycle_dimension=int(relative_dimension),
        n_edges=len(edge_labels),
        n_regions=len(regions),
        full_cycle_basis=full_basis,
        regional_cycle_basis=regional_basis,
        relative_cycle_basis=relative_basis,
        edge_labels=edge_labels,
        tolerance=tolerance,
    )


@dataclass(frozen=True, slots=True)
class BoundaryCancellationCircuitEntry:
    """One regional dependency of the weighted boundary-column matroid."""

    region_index: int
    columns: tuple[int, ...]
    rank: int
    dependency_dimension: int
    singular_gap: float | None
    is_circuit: bool

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "region_index": self.region_index,
            "columns": self.columns,
            "rank": self.rank,
            "dependency_dimension": self.dependency_dimension,
            "singular_gap": self.singular_gap,
            "is_circuit": self.is_circuit,
        }


@dataclass(frozen=True, slots=True)
class BoundaryCancellationMatroidReport:
    """Weighted dependency quotient of a boundary matrix modulo local circuits.

    The columns of the boundary matrix represent a linear matroid.  Its
    dependency space is the cage kernel.  Regional kernels generate a local
    dependency subspace, and the remaining quotient records collective
    cancellation classes that cannot be assembled from those regional
    dependencies.
    """

    n_rows: int
    n_columns: int
    rank: int
    dependency_dimension: int
    singular_gap: float | None
    regional_entries: tuple[BoundaryCancellationCircuitEntry, ...]
    regional_dependency_span_dimension: int
    intersection_dimension: int
    relative_dependency_dimension: int
    inclusion_residual: float
    full_dependency_basis: npt.NDArray[np.complex128]
    regional_dependency_basis: npt.NDArray[np.complex128]
    relative_dependency_basis: npt.NDArray[np.complex128]
    relative_edge_flow_basis: npt.NDArray[np.complex128]
    edge_labels: tuple[tuple[int, int], ...]
    edge_flow_conservation_residual: float
    tolerance: float

    @property
    def regional_circuit_count(self) -> int:
        return sum(entry.is_circuit for entry in self.regional_entries)

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "rank": self.rank,
            "dependency_dimension": self.dependency_dimension,
            "singular_gap": self.singular_gap,
            "regional_dependency_dimensions": tuple(
                entry.dependency_dimension for entry in self.regional_entries
            ),
            "regional_dependency_span_dimension": (self.regional_dependency_span_dimension),
            "regional_circuit_count": self.regional_circuit_count,
            "intersection_dimension": self.intersection_dimension,
            "relative_dependency_dimension": self.relative_dependency_dimension,
            "inclusion_residual": self.inclusion_residual,
            "edge_flow_conservation_residual": (self.edge_flow_conservation_residual),
            "tolerance": self.tolerance,
        }


@dataclass(frozen=True, slots=True)
class BoundaryCancellationMatroidBranchPoint:
    """One parameter point in a weighted dependency-quotient scan."""

    parameter: float
    report: BoundaryCancellationMatroidReport

    def to_summary_dict(self) -> dict[str, object]:
        return {"parameter": self.parameter, **self.report.to_summary_dict()}


@dataclass(frozen=True, slots=True)
class BoundaryCancellationMatroidBranchReport:
    """Integer dependency data tracked along a boundary-matrix deformation."""

    points: tuple[BoundaryCancellationMatroidBranchPoint, ...]
    tolerance: float

    @property
    def parameters(self) -> npt.NDArray[np.float64]:
        return np.asarray([point.parameter for point in self.points], dtype=np.float64)

    @property
    def dependency_dimensions(self) -> npt.NDArray[np.int64]:
        return np.asarray(
            [point.report.dependency_dimension for point in self.points],
            dtype=np.int64,
        )

    @property
    def regional_dimensions(self) -> npt.NDArray[np.int64]:
        return np.asarray(
            [point.report.regional_dependency_span_dimension for point in self.points],
            dtype=np.int64,
        )

    @property
    def relative_dimensions(self) -> npt.NDArray[np.int64]:
        return np.asarray(
            [point.report.relative_dependency_dimension for point in self.points],
            dtype=np.int64,
        )

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "tolerance": self.tolerance,
            "points": tuple(point.to_summary_dict() for point in self.points),
        }


def _orthonormal_basis_absolute(
    matrix: npt.NDArray[np.complex128],
    *,
    tolerance: float,
) -> npt.NDArray[np.complex128]:
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2D.")
    if matrix.shape[1] == 0:
        return np.zeros((matrix.shape[0], 0), dtype=np.complex128)
    left_vectors, singular_values, _right_vectors_h = scipy_linalg.svd(
        matrix,
        full_matrices=False,
    )
    rank = int(np.sum(singular_values > tolerance))
    return left_vectors[:, :rank].astype(np.complex128, copy=False)


def _matrix_singular_gap(
    matrix: npt.NDArray[np.complex128],
    *,
    tolerance: float,
) -> float | None:
    singular_values = scipy_linalg.svdvals(matrix)
    nonzero = singular_values[singular_values > tolerance]
    if nonzero.size == 0:
        return None
    return float(np.min(nonzero))


def _is_column_circuit(
    matrix: npt.NDArray[np.complex128],
    columns: tuple[int, ...],
    *,
    tolerance: float,
) -> bool:
    if not columns:
        return False
    submatrix = matrix[:, np.asarray(columns, dtype=np.int64)]
    if nullspace_svd(submatrix, tolerance=tolerance).shape[1] != 1:
        return False
    for removed in range(len(columns)):
        reduced_columns = columns[:removed] + columns[removed + 1 :]
        reduced = matrix[:, np.asarray(reduced_columns, dtype=np.int64)]
        if nullspace_svd(reduced, tolerance=tolerance).shape[1] != 0:
            return False
    return True


def diagnose_boundary_cancellation_matroid(
    boundary_matrix: object,
    regions: Sequence[Sequence[int]],
    *,
    tolerance: float = 1e-10,
) -> BoundaryCancellationMatroidReport:
    """Diagnose weighted global dependencies modulo regional circuits.

    ``regions`` contains boundary-matrix column indices.  Every regional right
    kernel is embedded in the full column space.  Their span is quotiented from
    the complete right kernel, reducing the unweighted graph-cycle problem to
    cancellation relations that satisfy the actual matrix amplitudes.
    """
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")
    boundary = as_dense_array(boundary_matrix)
    if boundary.ndim != 2:
        raise ValueError("boundary_matrix must be 2D.")

    n_rows, n_columns = boundary.shape
    full_basis = nullspace_svd(boundary, tolerance=tolerance)
    rank = n_columns - full_basis.shape[1]
    regional_vectors: list[npt.NDArray[np.complex128]] = []
    regional_entries: list[BoundaryCancellationCircuitEntry] = []

    for region_index, raw_columns in enumerate(regions):
        columns = tuple(sorted({int(column) for column in raw_columns}))
        if not columns:
            raise ValueError("each region must contain at least one column.")
        if columns[0] < 0 or columns[-1] >= n_columns:
            raise ValueError("regional column index is outside boundary_matrix.")
        submatrix = boundary[:, np.asarray(columns, dtype=np.int64)]
        kernel = nullspace_svd(submatrix, tolerance=tolerance)
        regional_entries.append(
            BoundaryCancellationCircuitEntry(
                region_index=region_index,
                columns=columns,
                rank=len(columns) - int(kernel.shape[1]),
                dependency_dimension=int(kernel.shape[1]),
                singular_gap=_matrix_singular_gap(submatrix, tolerance=tolerance),
                is_circuit=_is_column_circuit(
                    boundary,
                    columns,
                    tolerance=tolerance,
                ),
            )
        )
        for kernel_column in range(kernel.shape[1]):
            embedded = np.zeros(n_columns, dtype=np.complex128)
            embedded[np.asarray(columns, dtype=np.int64)] = kernel[:, kernel_column]
            regional_vectors.append(embedded)

    if regional_vectors:
        regional_basis = _orthonormal_basis_absolute(
            np.column_stack(regional_vectors),
            tolerance=tolerance,
        )
    else:
        regional_basis = np.zeros((n_columns, 0), dtype=np.complex128)

    if full_basis.shape[1] == 0:
        inclusion_residual = float(np.linalg.norm(regional_basis))
        intersection_dimension = 0
        relative_basis = np.zeros((n_columns, 0), dtype=np.complex128)
    else:
        full_projector = full_basis @ full_basis.conj().T
        inclusion_residual = float(
            np.linalg.norm((np.eye(n_columns) - full_projector) @ regional_basis)
        )
        overlaps = subspace_principal_overlaps(full_basis, regional_basis)
        intersection_dimension = int(np.sum(overlaps >= 1.0 - tolerance))
        regional_projector = regional_basis @ regional_basis.conj().T
        relative_basis = _orthonormal_basis_absolute(
            (np.eye(n_columns) - regional_projector) @ full_basis,
            tolerance=tolerance,
        )

    edge_labels = _boundary_edge_labels(boundary, tolerance)
    relative_edge_flows = np.zeros(
        (len(edge_labels), relative_basis.shape[1]),
        dtype=np.complex128,
    )
    for edge_index, (row, column) in enumerate(edge_labels):
        relative_edge_flows[edge_index, :] = boundary[row, column] * relative_basis[column, :]
    row_sums = np.zeros(
        (n_rows, relative_basis.shape[1]),
        dtype=np.complex128,
    )
    for edge_index, (row, _column) in enumerate(edge_labels):
        row_sums[row, :] += relative_edge_flows[edge_index, :]

    return BoundaryCancellationMatroidReport(
        n_rows=n_rows,
        n_columns=n_columns,
        rank=rank,
        dependency_dimension=int(full_basis.shape[1]),
        singular_gap=_matrix_singular_gap(boundary, tolerance=tolerance),
        regional_entries=tuple(regional_entries),
        regional_dependency_span_dimension=int(regional_basis.shape[1]),
        intersection_dimension=intersection_dimension,
        relative_dependency_dimension=int(relative_basis.shape[1]),
        inclusion_residual=inclusion_residual,
        full_dependency_basis=full_basis,
        regional_dependency_basis=regional_basis,
        relative_dependency_basis=relative_basis,
        relative_edge_flow_basis=relative_edge_flows,
        edge_labels=edge_labels,
        edge_flow_conservation_residual=float(np.linalg.norm(row_sums)),
        tolerance=tolerance,
    )


def boundary_cancellation_matroid_from_hamiltonian(
    hamiltonian: object,
    support: Sequence[int],
    regions: Sequence[Sequence[int]],
    *,
    tolerance: float = 1e-10,
) -> BoundaryCancellationMatroidReport:
    """Hamiltonian wrapper using global Hilbert-space indices for regions."""
    blocks = partition_cage_hamiltonian(hamiltonian, support)
    column_by_index = {
        int(basis_index): column for column, basis_index in enumerate(blocks.support)
    }
    regional_columns: list[tuple[int, ...]] = []
    for raw_region in regions:
        try:
            regional_columns.append(
                tuple(column_by_index[int(basis_index)] for basis_index in raw_region)
            )
        except KeyError as error:
            raise ValueError("every regional index must belong to support.") from error
    return diagnose_boundary_cancellation_matroid(
        blocks.boundary,
        regional_columns,
        tolerance=tolerance,
    )


def scan_boundary_cancellation_matroid(
    base_boundary: object,
    perturbation_boundary: object,
    regions: Sequence[Sequence[int]],
    parameters: Sequence[float] | npt.NDArray[np.floating],
    *,
    tolerance: float = 1e-10,
) -> BoundaryCancellationMatroidBranchReport:
    """Track the weighted dependency quotient along an affine deformation."""
    base = as_dense_array(base_boundary)
    perturbation = as_dense_array(perturbation_boundary)
    if base.shape != perturbation.shape:
        raise ValueError("base_boundary and perturbation_boundary must have equal shapes.")
    parameter_array = np.asarray(parameters, dtype=np.float64).reshape(-1)
    if parameter_array.size == 0:
        raise ValueError("parameters must contain at least one value.")
    points = tuple(
        BoundaryCancellationMatroidBranchPoint(
            parameter=float(parameter),
            report=diagnose_boundary_cancellation_matroid(
                base + float(parameter) * perturbation,
                regions,
                tolerance=tolerance,
            ),
        )
        for parameter in parameter_array
    )
    return BoundaryCancellationMatroidBranchReport(
        points=points,
        tolerance=tolerance,
    )


@dataclass(frozen=True, slots=True)
class BoundaryCancellationMomentumPoint:
    """Weighted cancellation data for one Bloch momentum sector."""

    momentum_index: int
    momentum: float
    dependency_dimension: int
    regional_dependency_span_dimension: int
    relative_dependency_dimension: int
    singular_gap: float | None
    relative_singular_gap: float | None
    regional_inclusion_residual: float

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "momentum_index": self.momentum_index,
            "momentum": self.momentum,
            "dependency_dimension": self.dependency_dimension,
            "regional_dependency_span_dimension": (self.regional_dependency_span_dimension),
            "relative_dependency_dimension": self.relative_dependency_dimension,
            "singular_gap": self.singular_gap,
            "relative_singular_gap": self.relative_singular_gap,
            "regional_inclusion_residual": self.regional_inclusion_residual,
        }


@dataclass(frozen=True, slots=True)
class BoundaryCancellationScalingPoint:
    """One finite periodic repetition in a cancellation-matroid sequence."""

    n_repeats: int
    n_rows: int
    n_columns: int
    dependency_dimension: int
    regional_dependency_span_dimension: int
    relative_dependency_dimension: int
    relative_dependency_density: float
    relative_zero_momentum_indices: tuple[int, ...]
    minimum_relative_singular_gap: float | None
    minimum_positive_relative_singular_gap: float | None
    maximum_regional_inclusion_residual: float
    momentum_points: tuple[BoundaryCancellationMomentumPoint, ...]

    @property
    def relative_zero_momentum_count(self) -> int:
        return len(self.relative_zero_momentum_indices)

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "n_repeats": self.n_repeats,
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "dependency_dimension": self.dependency_dimension,
            "regional_dependency_span_dimension": (self.regional_dependency_span_dimension),
            "relative_dependency_dimension": self.relative_dependency_dimension,
            "relative_dependency_density": self.relative_dependency_density,
            "relative_zero_momentum_count": self.relative_zero_momentum_count,
            "relative_zero_momentum_indices": self.relative_zero_momentum_indices,
            "minimum_relative_singular_gap": self.minimum_relative_singular_gap,
            "minimum_positive_relative_singular_gap": (self.minimum_positive_relative_singular_gap),
            "maximum_regional_inclusion_residual": (self.maximum_regional_inclusion_residual),
        }


@dataclass(frozen=True, slots=True)
class PeriodicBoundaryCancellationScalingReport:
    """Thermodynamic diagnostic for a finite-range periodic boundary family.

    The repeated boundary map is block circulant,

    ``B_N = I_N tensor B_0 + sum_d S_N**d tensor C_d``.

    A discrete Fourier transform decomposes it into small Bloch symbols
    ``B(k) = B_0 + sum_d exp(i k d) C_d``.  The global weighted dependency,
    regional dependency, and relative dependency dimensions are therefore exact
    sums of the corresponding symbol dimensions.  This avoids constructing the
    exponentially large many-body Hilbert space and isolates whether the
    collective cancellation class forms an extensive flat zero band, survives
    only at isolated momenta, or is fully lifted by a local repeated coupling.
    """

    n_rows_per_cell: int
    n_columns_per_cell: int
    coupling_displacements: tuple[int, ...]
    points: tuple[BoundaryCancellationScalingPoint, ...]
    tolerance: float

    @property
    def repeat_counts(self) -> npt.NDArray[np.int64]:
        return np.asarray([point.n_repeats for point in self.points], dtype=np.int64)

    @property
    def relative_dependency_dimensions(self) -> npt.NDArray[np.int64]:
        return np.asarray(
            [point.relative_dependency_dimension for point in self.points],
            dtype=np.int64,
        )

    @property
    def relative_dependency_densities(self) -> npt.NDArray[np.float64]:
        return np.asarray(
            [point.relative_dependency_density for point in self.points],
            dtype=np.float64,
        )

    @property
    def minimum_positive_relative_gaps(self) -> npt.NDArray[np.float64]:
        return np.asarray(
            [
                (
                    np.nan
                    if point.minimum_positive_relative_singular_gap is None
                    else point.minimum_positive_relative_singular_gap
                )
                for point in self.points
            ],
            dtype=np.float64,
        )

    @property
    def relative_dimension_growth_exponent(self) -> float | None:
        return estimate_power_law_exponent(
            self.repeat_counts,
            self.relative_dependency_dimensions,
        )

    @property
    def positive_relative_gap_exponent(self) -> float | None:
        return self.estimate_positive_relative_gap_exponent()

    def estimate_positive_relative_gap_exponent(
        self,
        *,
        minimum_repeats: int = 0,
    ) -> float | None:
        """Fit the positive relative gap after an optional finite-size cutoff."""
        if minimum_repeats < 0:
            raise ValueError("minimum_repeats must be non-negative.")
        return estimate_power_law_exponent(
            self.repeat_counts,
            self.minimum_positive_relative_gaps,
            minimum_parameter=float(minimum_repeats),
        )

    @property
    def scaling_label(self) -> str:
        if all(point.relative_dependency_dimension == 0 for point in self.points):
            return "fully_lifted"
        if all(point.relative_zero_momentum_count == point.n_repeats for point in self.points):
            return "extensive_zero_band"
        if all(point.relative_zero_momentum_count > 0 for point in self.points):
            return "isolated_zero_momenta"
        return "mixed"

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "n_rows_per_cell": self.n_rows_per_cell,
            "n_columns_per_cell": self.n_columns_per_cell,
            "coupling_displacements": self.coupling_displacements,
            "scaling_label": self.scaling_label,
            "relative_dimension_growth_exponent": (self.relative_dimension_growth_exponent),
            "positive_relative_gap_exponent": self.positive_relative_gap_exponent,
            "points": tuple(point.to_summary_dict() for point in self.points),
            "tolerance": self.tolerance,
        }


def periodic_boundary_cancellation_symbol(
    base_boundary: object,
    coupling_terms: Sequence[tuple[int, object]],
    momentum: float,
) -> npt.NDArray[np.complex128]:
    """Return the Bloch boundary symbol for finite-range periodic couplings."""
    base = np.asarray(as_dense_array(base_boundary), dtype=np.complex128)
    if base.ndim != 2:
        raise ValueError("base_boundary must be a matrix.")
    symbol = base.copy()
    for displacement, raw_coupling in coupling_terms:
        coupling = np.asarray(as_dense_array(raw_coupling), dtype=np.complex128)
        if coupling.shape != base.shape:
            raise ValueError("every coupling matrix must match base_boundary shape.")
        symbol += np.exp(1.0j * float(momentum) * int(displacement)) * coupling
    if not np.all(np.isfinite(symbol)):
        raise ValueError("boundary symbol contains non-finite entries.")
    return symbol


def _fixed_regional_relative_singular_gap(
    boundary: npt.NDArray[np.complex128],
    regional_basis: npt.NDArray[np.complex128],
    *,
    relative_dependency_dimension: int,
    tolerance: float,
) -> float | None:
    if relative_dependency_dimension > 0:
        return 0.0
    if regional_basis.shape[1] == 0:
        complement = np.eye(boundary.shape[1], dtype=np.complex128)
    else:
        complement = nullspace_svd(regional_basis.conj().T, tolerance=tolerance)
    if complement.shape[1] == 0:
        return None
    singular_values = scipy_linalg.svdvals(boundary @ complement)
    if singular_values.size == 0:
        return None
    return float(np.min(singular_values))


def scan_periodic_boundary_cancellation_scaling(
    base_boundary: object,
    regions: Sequence[Sequence[int]],
    repeat_counts: Sequence[int] | npt.NDArray[np.integer],
    *,
    coupling_terms: Sequence[tuple[int, object]] = (),
    tolerance: float = 1e-10,
) -> PeriodicBoundaryCancellationScalingReport:
    """Scan exact weighted-dependency scaling in a 1D periodic repetition.

    The calculation is exact for the block-circulant boundary family specified
    by ``base_boundary`` and the finite-range ``coupling_terms``.  Each coupling
    term is ``(displacement, matrix)`` and contributes
    ``exp(i k displacement) * matrix`` to the Bloch symbol.  The same regional
    column cover is used in every cell and every momentum sector.
    """
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")
    base = np.asarray(as_dense_array(base_boundary), dtype=np.complex128)
    if base.ndim != 2:
        raise ValueError("base_boundary must be a matrix.")
    counts = tuple(int(value) for value in np.asarray(repeat_counts).reshape(-1))
    if not counts:
        raise ValueError("repeat_counts must contain at least one value.")
    if any(value <= 0 for value in counts):
        raise ValueError("repeat_counts must be positive.")
    if len(set(counts)) != len(counts):
        raise ValueError("repeat_counts must not contain duplicates.")

    normalized_couplings: list[tuple[int, npt.NDArray[np.complex128]]] = []
    for displacement, raw_coupling in coupling_terms:
        coupling = np.asarray(as_dense_array(raw_coupling), dtype=np.complex128)
        if coupling.shape != base.shape:
            raise ValueError("every coupling matrix must match base_boundary shape.")
        normalized_couplings.append((int(displacement), coupling))

    base_matroid = diagnose_boundary_cancellation_matroid(
        base,
        regions,
        tolerance=tolerance,
    )
    fixed_regional_basis = base_matroid.regional_dependency_basis
    fixed_regional_dimension = int(fixed_regional_basis.shape[1])

    scaling_points: list[BoundaryCancellationScalingPoint] = []
    for n_repeats in counts:
        momentum_points: list[BoundaryCancellationMomentumPoint] = []
        for momentum_index in range(n_repeats):
            momentum = float(2.0 * np.pi * momentum_index / n_repeats)
            symbol = periodic_boundary_cancellation_symbol(
                base,
                normalized_couplings,
                momentum,
            )
            inclusion_residual = float(np.linalg.norm(symbol @ fixed_regional_basis))
            inclusion_scale = max(1.0, float(np.linalg.norm(symbol)))
            if inclusion_residual > tolerance * inclusion_scale:
                raise ValueError(
                    "periodic coupling does not preserve the base regional "
                    "dependency span within tolerance."
                )
            full_basis = nullspace_svd(symbol, tolerance=tolerance)
            dependency_dimension = int(full_basis.shape[1])
            relative_dimension = dependency_dimension - fixed_regional_dimension
            if relative_dimension < 0:
                raise RuntimeError("regional dependency dimension exceeds the full symbol nullity.")
            momentum_points.append(
                BoundaryCancellationMomentumPoint(
                    momentum_index=momentum_index,
                    momentum=momentum,
                    dependency_dimension=dependency_dimension,
                    regional_dependency_span_dimension=fixed_regional_dimension,
                    relative_dependency_dimension=relative_dimension,
                    singular_gap=_matrix_singular_gap(symbol, tolerance=tolerance),
                    relative_singular_gap=(
                        _fixed_regional_relative_singular_gap(
                            symbol,
                            fixed_regional_basis,
                            relative_dependency_dimension=relative_dimension,
                            tolerance=tolerance,
                        )
                    ),
                    regional_inclusion_residual=inclusion_residual,
                )
            )

        total_dependency = sum(point.dependency_dimension for point in momentum_points)
        total_regional = sum(point.regional_dependency_span_dimension for point in momentum_points)
        total_relative = sum(point.relative_dependency_dimension for point in momentum_points)
        relative_gaps = tuple(
            point.relative_singular_gap
            for point in momentum_points
            if point.relative_singular_gap is not None
        )
        positive_relative_gaps = tuple(gap for gap in relative_gaps if gap > tolerance)
        scaling_points.append(
            BoundaryCancellationScalingPoint(
                n_repeats=n_repeats,
                n_rows=int(n_repeats * base.shape[0]),
                n_columns=int(n_repeats * base.shape[1]),
                dependency_dimension=int(total_dependency),
                regional_dependency_span_dimension=int(total_regional),
                relative_dependency_dimension=int(total_relative),
                relative_dependency_density=float(total_relative / n_repeats),
                relative_zero_momentum_indices=tuple(
                    point.momentum_index
                    for point in momentum_points
                    if point.relative_dependency_dimension > 0
                ),
                minimum_relative_singular_gap=(
                    None if not relative_gaps else float(min(relative_gaps))
                ),
                minimum_positive_relative_singular_gap=(
                    None if not positive_relative_gaps else float(min(positive_relative_gaps))
                ),
                maximum_regional_inclusion_residual=float(
                    max(
                        (point.regional_inclusion_residual for point in momentum_points),
                        default=0.0,
                    )
                ),
                momentum_points=tuple(momentum_points),
            )
        )

    return PeriodicBoundaryCancellationScalingReport(
        n_rows_per_cell=int(base.shape[0]),
        n_columns_per_cell=int(base.shape[1]),
        coupling_displacements=tuple(
            displacement for displacement, _coupling in normalized_couplings
        ),
        points=tuple(scaling_points),
        tolerance=tolerance,
    )


@dataclass(frozen=True, slots=True)
class QDMExplicitProductSupport:
    """Materialized finite support of a factorized periodic QDM cage.

    This object is intentionally limited to moderate product supports.  It is
    used to verify the physical finite-size boundary map without enumerating the
    full constrained Hilbert space.
    """

    configs: npt.NDArray[np.int64]
    amplitudes: npt.NDArray[np.complex128]
    block_support_indices: npt.NDArray[np.int64]

    def __post_init__(self) -> None:
        configs = np.asarray(self.configs, dtype=np.int64)
        amplitudes = np.asarray(self.amplitudes, dtype=np.complex128)
        indices = np.asarray(self.block_support_indices, dtype=np.int64)
        if configs.ndim != 2:
            raise ValueError("configs must be two-dimensional.")
        if amplitudes.ndim != 1 or amplitudes.size != configs.shape[0]:
            raise ValueError("amplitudes must have one entry per support config.")
        if indices.ndim != 2 or indices.shape[0] != configs.shape[0]:
            raise ValueError("block_support_indices must align with configs.")
        norm = float(np.linalg.norm(amplitudes))
        if norm == 0.0:
            raise ValueError("amplitudes must have nonzero norm.")
        object.__setattr__(self, "configs", configs.copy())
        object.__setattr__(self, "amplitudes", amplitudes / norm)
        object.__setattr__(self, "block_support_indices", indices.copy())

    @property
    def support_size(self) -> int:
        return int(self.configs.shape[0])

    @property
    def n_blocks(self) -> int:
        return int(self.block_support_indices.shape[1])


@dataclass(frozen=True, slots=True)
class QDMExplicitSupportBoundaryMap:
    """One-hop QDM boundary map built from an explicit support only."""

    support_configs: npt.NDArray[np.int64]
    shell_configs: npt.NDArray[np.int64]
    boundary: scipy_sparse.csr_matrix

    @property
    def support_size(self) -> int:
        return int(self.support_configs.shape[0])

    @property
    def shell_size(self) -> int:
        return int(self.shell_configs.shape[0])

    @property
    def n_transitions(self) -> int:
        return int(self.boundary.nnz)


@dataclass(frozen=True, slots=True)
class QDMLocalKineticCompatibilityReport:
    """Compatibility of independent plaquette kinetic couplings for one state."""

    plaquette_ids: tuple[int, ...]
    obstruction_matrix: npt.NDArray[np.complex128]
    singular_values: npt.NDArray[np.float64]
    rank: int
    compatible_dimension: int
    active_plaquette_ids: tuple[int, ...]
    equal_coupling_pairs: tuple[tuple[int, int], ...]
    singular_gap: float | None
    tolerance: float

    @property
    def compatible_fraction(self) -> float:
        if not self.plaquette_ids:
            return 1.0
        return float(self.compatible_dimension / len(self.plaquette_ids))

    @property
    def constraint_density(self) -> float:
        if not self.plaquette_ids:
            return 0.0
        return float(self.rank / len(self.plaquette_ids))

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "n_plaquette_terms": len(self.plaquette_ids),
            "n_active_leakage_plaquettes": len(self.active_plaquette_ids),
            "rank": self.rank,
            "compatible_dimension": self.compatible_dimension,
            "compatible_fraction": self.compatible_fraction,
            "constraint_density": self.constraint_density,
            "equal_coupling_pairs": self.equal_coupling_pairs,
            "singular_gap": self.singular_gap,
            "tolerance": self.tolerance,
        }


@dataclass(frozen=True, slots=True)
class QDMLocalPotentialCompatibilityReport:
    """Uniform-on-support test for independent plaquette potential couplings."""

    plaquette_ids: tuple[int, ...]
    obstruction_matrix: npt.NDArray[np.complex128]
    rank: int
    compatible_dimension: int
    varying_plaquette_ids: tuple[int, ...]
    tolerance: float

    @property
    def compatible_fraction(self) -> float:
        if not self.plaquette_ids:
            return 1.0
        return float(self.compatible_dimension / len(self.plaquette_ids))

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "n_plaquette_terms": len(self.plaquette_ids),
            "rank": self.rank,
            "compatible_dimension": self.compatible_dimension,
            "compatible_fraction": self.compatible_fraction,
            "varying_plaquette_ids": self.varying_plaquette_ids,
            "tolerance": self.tolerance,
        }


@dataclass(frozen=True, slots=True)
class QDMPhysicalCancellationScalingPoint:
    """Exact finite-support cancellation data for one periodic product member."""

    repeats: int
    system_size: tuple[int, int]
    n_blocks: int
    support_size: int
    shell_size: int
    n_boundary_transitions: int
    boundary_rank: int
    boundary_nullity: int
    interference_gap: float | None
    product_state_boundary_residual: float
    product_state_kernel_weight: float
    kinetic_compatibility: QDMLocalKineticCompatibilityReport
    potential_compatibility: QDMLocalPotentialCompatibilityReport

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "repeats": self.repeats,
            "system_size": self.system_size,
            "n_blocks": self.n_blocks,
            "support_size": self.support_size,
            "shell_size": self.shell_size,
            "n_boundary_transitions": self.n_boundary_transitions,
            "boundary_rank": self.boundary_rank,
            "boundary_nullity": self.boundary_nullity,
            "interference_gap": self.interference_gap,
            "product_state_boundary_residual": self.product_state_boundary_residual,
            "product_state_kernel_weight": self.product_state_kernel_weight,
            "kinetic_compatible_dimension": (self.kinetic_compatibility.compatible_dimension),
            "kinetic_constraint_rank": self.kinetic_compatibility.rank,
            "kinetic_compatible_fraction": (self.kinetic_compatibility.compatible_fraction),
            "kinetic_equal_coupling_pairs": (self.kinetic_compatibility.equal_coupling_pairs),
            "potential_compatible_dimension": (self.potential_compatibility.compatible_dimension),
            "potential_constraint_rank": self.potential_compatibility.rank,
            "potential_compatible_fraction": (self.potential_compatibility.compatible_fraction),
        }


@dataclass(frozen=True, slots=True)
class QDMPhysicalCancellationScalingReport:
    """Physical finite-size scaling for a certified periodic QDM product cage."""

    repeat_axis: str
    unit_cell_size: tuple[int, int]
    support_size_per_unit_cell: int
    points: tuple[QDMPhysicalCancellationScalingPoint, ...]
    tolerance: float

    @property
    def repeat_counts(self) -> npt.NDArray[np.int64]:
        return np.asarray([point.repeats for point in self.points], dtype=np.int64)

    @property
    def boundary_nullities(self) -> npt.NDArray[np.int64]:
        return np.asarray([point.boundary_nullity for point in self.points], dtype=np.int64)

    @property
    def interference_gaps(self) -> npt.NDArray[np.float64]:
        return np.asarray(
            [
                np.nan if point.interference_gap is None else point.interference_gap
                for point in self.points
            ],
            dtype=np.float64,
        )

    @property
    def kinetic_constraint_ranks(self) -> npt.NDArray[np.int64]:
        return np.asarray(
            [point.kinetic_compatibility.rank for point in self.points],
            dtype=np.int64,
        )

    @property
    def kinetic_compatible_fractions(self) -> npt.NDArray[np.float64]:
        return np.asarray(
            [point.kinetic_compatibility.compatible_fraction for point in self.points],
            dtype=np.float64,
        )

    @property
    def has_unique_product_kernel(self) -> bool:
        return bool(
            self.points
            and all(
                point.boundary_nullity == 1
                and point.product_state_kernel_weight >= 1.0 - 10.0 * self.tolerance
                for point in self.points
            )
        )

    @property
    def interference_gap_exponent(self) -> float | None:
        return estimate_power_law_exponent(
            self.repeat_counts,
            self.interference_gaps,
        )

    @property
    def kinetic_constraint_rank_exponent(self) -> float | None:
        return estimate_power_law_exponent(
            self.repeat_counts,
            self.kinetic_constraint_ranks,
        )

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "repeat_axis": self.repeat_axis,
            "unit_cell_size": self.unit_cell_size,
            "support_size_per_unit_cell": self.support_size_per_unit_cell,
            "has_unique_product_kernel": self.has_unique_product_kernel,
            "interference_gap_exponent": self.interference_gap_exponent,
            "kinetic_constraint_rank_exponent": (self.kinetic_constraint_rank_exponent),
            "points": tuple(point.to_summary_dict() for point in self.points),
            "tolerance": self.tolerance,
        }


def materialize_square_qdm_periodic_product_support(
    instance: SquareQDMPeriodicProductInstance,
    *,
    max_support_size: int = 4096,
) -> QDMExplicitProductSupport:
    """Materialize a moderate finite periodic-product support exactly."""
    if max_support_size < 1:
        raise ValueError("max_support_size must be positive.")
    support_size = int(instance.formal_support_size)
    if support_size > max_support_size:
        raise ValueError(
            "formal product support exceeds max_support_size: "
            f"{support_size} > {max_support_size}."
        )

    blocks = tuple(instance.blocks)
    support_ranges = tuple(range(int(block.support_size)) for block in blocks)
    support_tuples = tuple(itertools.product(*support_ranges))
    n_links = int(instance.model.lattice.num_links)
    configs = np.zeros((support_size, n_links), dtype=np.int64)
    amplitudes = np.ones(support_size, dtype=np.complex128)
    block_indices = np.zeros((support_size, len(blocks)), dtype=np.int64)
    exterior_ids = np.asarray(instance.padding.exterior_link_ids, dtype=np.int64)
    exterior_config = np.asarray(instance.padding.exterior_config, dtype=np.int64)

    for row_index, support_tuple in enumerate(support_tuples):
        if exterior_ids.size:
            configs[row_index, exterior_ids] = exterior_config
        for block_position, (block, support_index) in enumerate(
            zip(blocks, support_tuple, strict=True)
        ):
            index = int(support_index)
            configs[row_index, np.asarray(block.link_ids, dtype=np.int64)] = block.support_configs[
                index
            ]
            amplitudes[row_index] *= complex(block.amplitudes[index])
            block_indices[row_index, block_position] = index

    keys = {tuple(int(value) for value in row) for row in configs}
    if len(keys) != support_size:
        raise ValueError("materialized product support contains duplicate configurations.")
    return QDMExplicitProductSupport(
        configs=configs,
        amplitudes=amplitudes,
        block_support_indices=block_indices,
    )


def build_qdm_explicit_support_boundary(
    model: object,
    support_configs: object,
) -> QDMExplicitSupportBoundaryMap:
    """Build the support-to-exterior QDM kinetic map without a global basis."""
    configs = np.asarray(support_configs, dtype=np.int64)
    if configs.ndim != 2:
        raise ValueError("support_configs must be two-dimensional.")
    support_key_set = {tuple(int(value) for value in row) for row in configs}
    if len(support_key_set) != configs.shape[0]:
        raise ValueError("support_configs must not contain duplicates.")

    raw_entries: list[tuple[tuple[int, ...], int, complex]] = []
    shell_keys: set[tuple[int, ...]] = set()
    for source_index, source_config in enumerate(configs):
        for action in _qdm_global_plaquette_actions(model):
            transition = _qdm_flip_transition_from_action(source_config, action)
            if transition is None:
                continue
            final_config, coefficient = transition
            final_key = tuple(int(value) for value in final_config)
            if final_key in support_key_set:
                continue
            shell_keys.add(final_key)
            raw_entries.append((final_key, source_index, complex(coefficient)))

    ordered_shell_keys = tuple(sorted(shell_keys))
    row_by_key = {key: row for row, key in enumerate(ordered_shell_keys)}
    rows = [row_by_key[key] for key, _column, _value in raw_entries]
    columns = [column for _key, column, _value in raw_entries]
    values = [value for _key, _column, value in raw_entries]
    boundary = scipy_sparse.coo_matrix(
        (values, (rows, columns)),
        shape=(len(ordered_shell_keys), configs.shape[0]),
        dtype=np.complex128,
    ).tocsr()
    shell_configs = (
        np.asarray(ordered_shell_keys, dtype=np.int64)
        if ordered_shell_keys
        else np.empty((0, configs.shape[1]), dtype=np.int64)
    )
    return QDMExplicitSupportBoundaryMap(
        support_configs=configs.copy(),
        shell_configs=shell_configs,
        boundary=boundary,
    )


def _sparse_boundary_singular_data(
    boundary: scipy_sparse.csr_matrix,
    *,
    tolerance: float,
) -> tuple[int, int, float | None, npt.NDArray[np.complex128]]:
    # Direct SVD is deliberately used here.  Forming B^\dagger B squares the
    # condition number and turned exact cancellation modes into spurious
    # singular values of order sqrt(machine epsilon) in the QDM examples.
    dense = np.asarray(boundary.toarray(), dtype=np.complex128)
    _left, singular_values, right_vectors_h = scipy_linalg.svd(
        dense,
        full_matrices=boundary.shape[0] < boundary.shape[1],
    )
    rank = int(np.sum(singular_values > tolerance))
    nullity = int(boundary.shape[1] - rank)
    kernel = right_vectors_h.conj().T[:, rank:].astype(np.complex128, copy=False)
    positive = singular_values[singular_values > tolerance]
    gap = None if positive.size == 0 else float(np.min(positive))
    return rank, nullity, gap, kernel


def diagnose_qdm_local_kinetic_compatibility(
    model: object,
    support_configs: object,
    state: object,
    *,
    boundary_map: QDMExplicitSupportBoundaryMap | None = None,
    tolerance: float = 1e-10,
) -> QDMLocalKineticCompatibilityReport:
    """Find multiplicative plaquette-kinetic perturbations preserving one cage.

    Each coefficient independently rescales the kinetic operator already stored
    on one model plaquette.  For the uniform unit-coupling QDM this is the usual
    additive local-coupling basis.
    """
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")
    configs = np.asarray(support_configs, dtype=np.int64)
    amplitudes = np.asarray(state, dtype=np.complex128).reshape(-1)
    if amplitudes.size != configs.shape[0]:
        raise ValueError("state size must match support_configs rows.")
    norm = float(np.linalg.norm(amplitudes))
    if norm == 0.0:
        raise ValueError("state must have nonzero norm.")
    amplitudes = amplitudes / norm
    boundary_report = (
        build_qdm_explicit_support_boundary(model, configs)
        if boundary_map is None
        else boundary_map
    )
    support_keys = {tuple(int(value) for value in row) for row in configs}
    shell_row_by_key = {
        tuple(int(value) for value in row): index
        for index, row in enumerate(boundary_report.shell_configs)
    }
    actions = tuple(_qdm_global_plaquette_actions(model))
    obstruction = np.zeros(
        (boundary_report.shell_size, len(actions)),
        dtype=np.complex128,
    )
    for source_config, amplitude in zip(configs, amplitudes, strict=True):
        for action_index, action in enumerate(actions):
            transition = _qdm_flip_transition_from_action(source_config, action)
            if transition is None:
                continue
            final_config, coefficient = transition
            final_key = tuple(int(value) for value in final_config)
            if final_key in support_keys:
                continue
            obstruction[shell_row_by_key[final_key], action_index] += complex(
                coefficient
            ) * complex(amplitude)

    singular_values = scipy_linalg.svdvals(obstruction)
    rank = int(np.sum(singular_values > tolerance))
    column_norms = np.linalg.norm(obstruction, axis=0)
    active_indices = tuple(int(index) for index in np.flatnonzero(column_norms > tolerance))
    equal_pairs: list[tuple[int, int]] = []
    used: set[int] = set()
    for left in active_indices:
        if left in used:
            continue
        left_column = obstruction[:, left]
        for right in active_indices:
            if right <= left or right in used:
                continue
            right_column = obstruction[:, right]
            scale = max(
                1.0,
                float(np.linalg.norm(left_column)),
                float(np.linalg.norm(right_column)),
            )
            if np.linalg.norm(left_column + right_column) <= tolerance * scale:
                equal_pairs.append(
                    (
                        int(actions[left].plaquette_id),
                        int(actions[right].plaquette_id),
                    )
                )
                used.add(left)
                used.add(right)
                break

    positive = singular_values[singular_values > tolerance]
    return QDMLocalKineticCompatibilityReport(
        plaquette_ids=tuple(int(action.plaquette_id) for action in actions),
        obstruction_matrix=obstruction,
        singular_values=np.asarray(singular_values, dtype=np.float64),
        rank=rank,
        compatible_dimension=int(len(actions) - rank),
        active_plaquette_ids=tuple(int(actions[index].plaquette_id) for index in active_indices),
        equal_coupling_pairs=tuple(equal_pairs),
        singular_gap=None if positive.size == 0 else float(np.min(positive)),
        tolerance=tolerance,
    )


def diagnose_qdm_local_potential_compatibility(
    model: object,
    support_configs: object,
    *,
    tolerance: float = 1e-10,
) -> QDMLocalPotentialCompatibilityReport:
    """Find plaquette-potential perturbations uniform on an explicit support."""
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")
    configs = np.asarray(support_configs, dtype=np.int64)
    if configs.ndim != 2:
        raise ValueError("support_configs must be two-dimensional.")
    actions = tuple(_qdm_global_plaquette_actions(model))
    activity = np.zeros((configs.shape[0], len(actions)), dtype=np.complex128)
    for row_index, config in enumerate(configs):
        for action_index, action in enumerate(actions):
            if _qdm_flip_transition_from_action(config, action) is not None:
                activity[row_index, action_index] = 1.0
    obstruction = activity - activity[:1]
    singular_values = scipy_linalg.svdvals(obstruction)
    rank = int(np.sum(singular_values > tolerance))
    varying = tuple(
        int(actions[index].plaquette_id)
        for index in np.flatnonzero(np.linalg.norm(obstruction, axis=0) > tolerance)
    )
    return QDMLocalPotentialCompatibilityReport(
        plaquette_ids=tuple(int(action.plaquette_id) for action in actions),
        obstruction_matrix=obstruction,
        rank=rank,
        compatible_dimension=int(len(actions) - rank),
        varying_plaquette_ids=varying,
        tolerance=tolerance,
    )


def scan_square_qdm_periodic_product_cancellation_scaling(
    unit_cell: SquareQDMPeriodicProductUnitCell,
    repeat_counts: Sequence[int] | npt.NDArray[np.integer],
    *,
    max_support_size: int = 1024,
    tolerance: float = 1e-10,
) -> QDMPhysicalCancellationScalingReport:
    """Extract exact physical boundary maps for finite periodic embeddings.

    Unlike :func:`scan_periodic_boundary_cancellation_scaling`, this routine
    materializes the tensor-product support of the actual square-QDM sequence.
    It is therefore restricted to moderate repeats but includes every physical
    plaquette flip and every seam transition of the finite torus.
    """
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")
    counts = tuple(int(value) for value in np.asarray(repeat_counts).reshape(-1))
    if not counts or any(value <= 0 for value in counts):
        raise ValueError("repeat_counts must contain positive integers.")
    if len(set(counts)) != len(counts):
        raise ValueError("repeat_counts must not contain duplicates.")

    points: list[QDMPhysicalCancellationScalingPoint] = []
    for repeats in counts:
        instance = unit_cell.instantiate(repeats)
        support = materialize_square_qdm_periodic_product_support(
            instance,
            max_support_size=max_support_size,
        )
        boundary_map = build_qdm_explicit_support_boundary(
            instance.model,
            support.configs,
        )
        rank, nullity, gap, kernel = _sparse_boundary_singular_data(
            boundary_map.boundary,
            tolerance=tolerance,
        )
        boundary_residual = float(np.linalg.norm(boundary_map.boundary @ support.amplitudes))
        kernel_weight = float(np.linalg.norm(kernel.conj().T @ support.amplitudes) ** 2)
        kinetic = diagnose_qdm_local_kinetic_compatibility(
            instance.model,
            support.configs,
            support.amplitudes,
            boundary_map=boundary_map,
            tolerance=tolerance,
        )
        potential = diagnose_qdm_local_potential_compatibility(
            instance.model,
            support.configs,
            tolerance=tolerance,
        )
        points.append(
            QDMPhysicalCancellationScalingPoint(
                repeats=repeats,
                system_size=(int(instance.model.lx), int(instance.model.ly)),
                n_blocks=len(instance.blocks),
                support_size=support.support_size,
                shell_size=boundary_map.shell_size,
                n_boundary_transitions=boundary_map.n_transitions,
                boundary_rank=rank,
                boundary_nullity=nullity,
                interference_gap=gap,
                product_state_boundary_residual=boundary_residual,
                product_state_kernel_weight=kernel_weight,
                kinetic_compatibility=kinetic,
                potential_compatibility=potential,
            )
        )

    return QDMPhysicalCancellationScalingReport(
        repeat_axis=str(unit_cell.repeat_axis),
        unit_cell_size=(int(unit_cell.model.lx), int(unit_cell.model.ly)),
        support_size_per_unit_cell=int(unit_cell.support_size_per_unit_cell),
        points=tuple(points),
        tolerance=tolerance,
    )


SquareQDMColumnSymbol = tuple[int, int, int]
SquareQDMColumnWord = tuple[SquareQDMColumnSymbol, ...]


@dataclass(frozen=True, slots=True)
class QDMCyclicColumnGrammar:
    """Finite-range column grammar inferred from square-QDM support states.

    A column symbol stores ``(incoming_mask, outgoing_mask, vertical_mask)`` on
    a fixed-circumference cylinder.  ``allowed_windows`` contains every cyclic
    window of ``window_size`` columns observed in the reference support.
    """

    circumference: int
    reference_length: int
    window_size: int
    symbols: tuple[SquareQDMColumnSymbol, ...]
    allowed_windows: tuple[SquareQDMColumnWord, ...]

    def __post_init__(self) -> None:
        if self.circumference < 2:
            raise ValueError("circumference must be at least two.")
        if self.reference_length < 2:
            raise ValueError("reference_length must be at least two.")
        if self.window_size < 2:
            raise ValueError("window_size must be at least two.")
        if self.window_size > self.reference_length:
            raise ValueError("window_size cannot exceed reference_length.")
        if not self.symbols:
            raise ValueError("symbols must not be empty.")
        if not self.allowed_windows:
            raise ValueError("allowed_windows must not be empty.")
        if any(len(window) != self.window_size for window in self.allowed_windows):
            raise ValueError("every allowed window must have window_size entries.")

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "circumference": self.circumference,
            "reference_length": self.reference_length,
            "window_size": self.window_size,
            "n_symbols": len(self.symbols),
            "n_allowed_windows": len(self.allowed_windows),
        }


@dataclass(frozen=True, slots=True)
class QDMCyclicGrammarSupport:
    """Explicit finite support generated by one cyclic column grammar."""

    length: int
    words: tuple[SquareQDMColumnWord, ...]
    configs: npt.NDArray[np.int64]
    potential_value: complex

    def __post_init__(self) -> None:
        configs = np.asarray(self.configs, dtype=np.int64)
        if configs.ndim != 2:
            raise ValueError("configs must be two-dimensional.")
        if len(self.words) != configs.shape[0]:
            raise ValueError("words and configs must have the same number of rows.")
        if any(len(word) != self.length for word in self.words):
            raise ValueError("every word must have the requested length.")
        object.__setattr__(self, "configs", configs.copy())

    @property
    def support_size(self) -> int:
        return int(self.configs.shape[0])

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "length": self.length,
            "support_size": self.support_size,
            "potential_value": self.potential_value,
        }


@dataclass(frozen=True, slots=True)
class QDMLocalGrammarExtensionPoint:
    """One fixed-width extension test for a collective cage support grammar."""

    window_size: int
    length: int
    support_size: int
    shell_size: int
    boundary_nullity: int
    nullity_is_resolved: bool
    interference_gap: float | None
    product_translation_span_dimension: int
    kernel_product_intersection_dimension: int
    collective_quotient_dimension: int
    product_containment_residual: float
    principal_overlaps: npt.NDArray[np.float64]
    localized_support_sizes: tuple[int, ...]
    localized_iprs: tuple[float, ...]

    def __post_init__(self) -> None:
        overlaps = np.asarray(self.principal_overlaps, dtype=np.float64).reshape(-1)
        object.__setattr__(self, "principal_overlaps", overlaps.copy())

    @property
    def locality_extension_index(self) -> int:
        """Integer dimension of non-product cage modes in this local language."""
        return self.collective_quotient_dimension

    @property
    def kernel_is_exhausted_by_product_translations(self) -> bool:
        return bool(
            self.nullity_is_resolved
            and self.collective_quotient_dimension == 0
            and self.kernel_product_intersection_dimension == self.boundary_nullity
        )

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "window_size": self.window_size,
            "length": self.length,
            "support_size": self.support_size,
            "shell_size": self.shell_size,
            "boundary_nullity": self.boundary_nullity,
            "nullity_is_resolved": self.nullity_is_resolved,
            "interference_gap": self.interference_gap,
            "product_translation_span_dimension": self.product_translation_span_dimension,
            "kernel_product_intersection_dimension": (self.kernel_product_intersection_dimension),
            "collective_quotient_dimension": self.collective_quotient_dimension,
            "locality_extension_index": self.locality_extension_index,
            "product_containment_residual": self.product_containment_residual,
            "minimum_principal_overlap": (
                None
                if self.principal_overlaps.size == 0
                else float(np.min(self.principal_overlaps))
            ),
            "localized_support_sizes": self.localized_support_sizes,
            "localized_iprs": self.localized_iprs,
            "kernel_is_exhausted_by_product_translations": (
                self.kernel_is_exhausted_by_product_translations
            ),
        }


@dataclass(frozen=True, slots=True)
class QDMLocalGrammarExtensionReport:
    """Fixed-width locality test for a non-factorized collective extension."""

    reference_length: int
    circumference: int
    points: tuple[QDMLocalGrammarExtensionPoint, ...]
    tolerance: float

    @property
    def has_collective_extension(self) -> bool:
        return any(point.collective_quotient_dimension > 0 for point in self.points)

    @property
    def product_only_on_all_resolved_points(self) -> bool:
        return bool(
            self.points
            and all(
                point.kernel_is_exhausted_by_product_translations
                for point in self.points
                if point.nullity_is_resolved
            )
            and all(point.nullity_is_resolved for point in self.points)
        )

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "reference_length": self.reference_length,
            "circumference": self.circumference,
            "has_collective_extension": self.has_collective_extension,
            "product_only_on_all_resolved_points": (self.product_only_on_all_resolved_points),
            "points": tuple(point.to_summary_dict() for point in self.points),
            "tolerance": self.tolerance,
        }


@dataclass(frozen=True, slots=True)
class CyclicAmplitudeBondProfile:
    """Exact finite-state bond-rank profile of a cyclic column amplitude.

    The rank at a cut is the Schmidt rank of the sparse amplitude tensor after
    grouping the columns to the left and right of that cut.  The maximum cut
    rank is the minimal exact open-boundary MPS bond dimension for this finite
    state.  A periodic MPS of bond dimension ``D`` has Schmidt rank at most
    ``D**2``, giving the reported rigorous lower bound.
    """

    length: int
    support_size: int
    alphabet_size: int
    cut_ranks: tuple[int, ...]
    maximum_cut_rank: int
    periodic_bond_dimension_lower_bound: int
    translation_support_closed: bool
    translation_eigenvalue: complex | None
    translation_residual: float | None
    tolerance: float

    @property
    def exact_open_bond_dimension(self) -> int:
        return self.maximum_cut_rank

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "length": self.length,
            "support_size": self.support_size,
            "alphabet_size": self.alphabet_size,
            "cut_ranks": self.cut_ranks,
            "exact_open_bond_dimension": self.exact_open_bond_dimension,
            "periodic_bond_dimension_lower_bound": (self.periodic_bond_dimension_lower_bound),
            "translation_support_closed": self.translation_support_closed,
            "translation_eigenvalue": self.translation_eigenvalue,
            "translation_residual": self.translation_residual,
            "tolerance": self.tolerance,
        }


@dataclass(frozen=True, slots=True)
class SquareQDMTransferSectorMultiplicity:
    """Multiplicity of one lattice-momentum sector in a finite bond space."""

    momentum_x_index: int
    momentum_y_index: int
    momentum_x: float
    momentum_y: float
    kernel_multiplicity: int
    reference_multiplicity: int
    relative_multiplicity: int

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "momentum_x_index": self.momentum_x_index,
            "momentum_y_index": self.momentum_y_index,
            "momentum_x": self.momentum_x,
            "momentum_y": self.momentum_y,
            "kernel_multiplicity": self.kernel_multiplicity,
            "reference_multiplicity": self.reference_multiplicity,
            "relative_multiplicity": self.relative_multiplicity,
        }


@dataclass(frozen=True, slots=True)
class SquareQDMFiniteBondTransferInvariantReport:
    """Discrete spatial representation carried by a fixed-width cage kernel.

    The exact boundary kernel is treated as a finite transfer/bond space.  The
    reference subspace may contain known compact or regional cage modes.  The
    quotient then carries a basis-independent representation of the commuting
    translations.  Momentum-sector multiplicities are integers and can change
    only when the kernel/reference dimensions change or the spatial symmetry is
    broken.

    This is a physical symmetry representation, not automatically a virtual
    projective (SPT) invariant.  ``group_relation_residual`` explicitly checks
    that the quotient realizes the ordinary square-lattice relations.
    """

    system_size: tuple[int, int]
    support_size: int
    kernel_dimension: int
    reference_dimension: int
    relative_dimension: int
    reference_containment_residual: float
    kernel_symmetry_residual: float
    reference_symmetry_residual: float
    relative_symmetry_residual: float
    translation_commutator_residual: float
    group_relation_residual: float
    sectors: tuple[SquareQDMTransferSectorMultiplicity, ...]
    quotient_translation_x_character: complex | None
    quotient_translation_y_character: complex | None
    quotient_reflection_x_character: complex | None
    quotient_reflection_y_character: complex | None
    quotient_quarter_turn_character: complex | None
    tolerance: float

    @property
    def relative_trivial_sector_dimension(self) -> int:
        for sector in self.sectors:
            if sector.momentum_x_index == 0 and sector.momentum_y_index == 0:
                return sector.relative_multiplicity
        return 0

    @property
    def relative_sector_signature(self) -> tuple[tuple[int, int, int], ...]:
        return tuple(
            (
                sector.momentum_x_index,
                sector.momentum_y_index,
                sector.relative_multiplicity,
            )
            for sector in self.sectors
            if sector.relative_multiplicity
        )

    @property
    def has_one_dimensional_trivial_spatial_quotient(self) -> bool:
        if self.relative_dimension != 1 or self.relative_trivial_sector_dimension != 1:
            return False
        characters = (
            self.quotient_translation_x_character,
            self.quotient_translation_y_character,
            self.quotient_reflection_x_character,
            self.quotient_reflection_y_character,
            self.quotient_quarter_turn_character,
        )
        return all(
            value is None or abs(value - 1.0) <= 10.0 * self.tolerance for value in characters
        )

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "system_size": self.system_size,
            "support_size": self.support_size,
            "kernel_dimension": self.kernel_dimension,
            "reference_dimension": self.reference_dimension,
            "relative_dimension": self.relative_dimension,
            "relative_trivial_sector_dimension": self.relative_trivial_sector_dimension,
            "relative_sector_signature": self.relative_sector_signature,
            "reference_containment_residual": self.reference_containment_residual,
            "kernel_symmetry_residual": self.kernel_symmetry_residual,
            "reference_symmetry_residual": self.reference_symmetry_residual,
            "relative_symmetry_residual": self.relative_symmetry_residual,
            "translation_commutator_residual": self.translation_commutator_residual,
            "group_relation_residual": self.group_relation_residual,
            "quotient_translation_x_character": self.quotient_translation_x_character,
            "quotient_translation_y_character": self.quotient_translation_y_character,
            "quotient_reflection_x_character": self.quotient_reflection_x_character,
            "quotient_reflection_y_character": self.quotient_reflection_y_character,
            "quotient_quarter_turn_character": self.quotient_quarter_turn_character,
            "has_one_dimensional_trivial_spatial_quotient": (
                self.has_one_dimensional_trivial_spatial_quotient
            ),
            "tolerance": self.tolerance,
        }


@dataclass(frozen=True, slots=True)
class RealLocalSignObstructionReport:
    """Mod-two obstruction to a real finite-range local sign factorization.

    For cyclic support words ``w`` and real nonzero amplitudes ``psi_w``, let
    ``N[w, e]`` count occurrences of each local column window ``e``.  A real
    scalar local phase rule exists when the sign bits obey

    ``N s = sign(psi) (mod 2)``.

    The cokernel class of the sign vector is discrete.  It can change only if
    an amplitude crosses zero, the support language changes, or the real
    structure is abandoned.
    """

    window_size: int
    n_words: int
    n_local_windows: int
    incidence_rank_mod2: int
    augmented_rank_mod2: int
    obstruction_dimension: int
    magnitude_factorization_residual: float
    real_structure_residual: float
    obstruction_witness: npt.NDArray[np.uint8] | None
    local_sign_solution: npt.NDArray[np.uint8] | None

    def __post_init__(self) -> None:
        if self.obstruction_witness is not None:
            witness = np.asarray(self.obstruction_witness, dtype=np.uint8).reshape(-1) % 2
            object.__setattr__(self, "obstruction_witness", witness.copy())
        if self.local_sign_solution is not None:
            solution = np.asarray(self.local_sign_solution, dtype=np.uint8).reshape(-1) % 2
            object.__setattr__(self, "local_sign_solution", solution.copy())

    @property
    def is_obstructed(self) -> bool:
        return self.obstruction_dimension > 0

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "window_size": self.window_size,
            "n_words": self.n_words,
            "n_local_windows": self.n_local_windows,
            "incidence_rank_mod2": self.incidence_rank_mod2,
            "augmented_rank_mod2": self.augmented_rank_mod2,
            "obstruction_dimension": self.obstruction_dimension,
            "is_obstructed": self.is_obstructed,
            "magnitude_factorization_residual": self.magnitude_factorization_residual,
            "real_structure_residual": self.real_structure_residual,
            "obstruction_witness_weight": (
                None if self.obstruction_witness is None else int(np.sum(self.obstruction_witness))
            ),
        }


def square_qdm_column_words(
    model: object,
    support_configs: object,
) -> tuple[SquareQDMColumnWord, ...]:
    """Encode square-QDM configurations as periodic column-transition words."""
    configs = np.asarray(support_configs, dtype=np.int64)
    if configs.ndim != 2:
        raise ValueError("support_configs must be two-dimensional.")
    lx = int(model.lattice.lx)
    ly = int(model.lattice.ly)
    if configs.shape[1] != int(model.lattice.num_links):
        raise ValueError("support_configs width must match model links.")

    link_lookup: dict[tuple[int, int, str], int] = {}
    for link in model.lattice.links:
        x, y = model.lattice.sites[int(link.source)].cell
        link_lookup[(int(x), int(y), str(link.kind))] = int(link.id)

    words: list[SquareQDMColumnWord] = []
    for config in configs:
        symbols: list[SquareQDMColumnSymbol] = []
        for x in range(lx):
            incoming = 0
            outgoing = 0
            vertical = 0
            for y in range(ly):
                incoming |= int(config[link_lookup[((x - 1) % lx, y, "x")]]) << y
                outgoing |= int(config[link_lookup[(x, y, "x")]]) << y
                vertical |= int(config[link_lookup[(x, y, "y")]]) << y
            symbols.append((incoming, outgoing, vertical))
        words.append(tuple(symbols))
    return tuple(words)


def diagnose_cyclic_amplitude_bond_profile(
    column_words: Sequence[SquareQDMColumnWord],
    amplitudes: object,
    *,
    tolerance: float = 1e-10,
) -> CyclicAmplitudeBondProfile:
    """Compute exact finite-state bond ranks of a cyclic column amplitude."""
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")
    words = tuple(tuple(word) for word in column_words)
    if not words:
        raise ValueError("column_words must not be empty.")
    length = len(words[0])
    if length < 2 or any(len(word) != length for word in words):
        raise ValueError("all column words must have the same length of at least two.")
    if len(set(words)) != len(words):
        raise ValueError("column_words must not contain duplicates.")
    vector = np.asarray(amplitudes, dtype=np.complex128).reshape(-1)
    if vector.size != len(words):
        raise ValueError("amplitudes must have one entry per column word.")
    if float(np.linalg.norm(vector)) <= tolerance:
        raise ValueError("amplitudes must contain a nonzero state.")

    cut_ranks: list[int] = []
    for cut in range(1, length):
        prefixes = tuple(sorted({word[:cut] for word in words}))
        suffixes = tuple(sorted({word[cut:] for word in words}))
        prefix_index = {prefix: index for index, prefix in enumerate(prefixes)}
        suffix_index = {suffix: index for index, suffix in enumerate(suffixes)}
        matrix = np.zeros((len(prefixes), len(suffixes)), dtype=np.complex128)
        for word, amplitude in zip(words, vector, strict=True):
            matrix[prefix_index[word[:cut]], suffix_index[word[cut:]]] += amplitude
        singular_values = scipy_linalg.svdvals(matrix)
        cut_ranks.append(int(np.sum(singular_values > tolerance)))

    maximum_rank = max(cut_ranks)
    periodic_lower_bound = int(np.ceil(np.sqrt(maximum_rank)))
    amplitude_by_word = dict(zip(words, vector, strict=True))
    shifted_words = tuple(word[1:] + word[:1] for word in words)
    support_closed = all(word in amplitude_by_word for word in shifted_words)
    translation_eigenvalue: complex | None = None
    translation_residual: float | None = None
    if support_closed:
        shifted_vector = np.asarray(
            [amplitude_by_word[word] for word in shifted_words],
            dtype=np.complex128,
        )
        norm_squared = float(np.vdot(vector, vector).real)
        translation_eigenvalue = complex(np.vdot(vector, shifted_vector) / norm_squared)
        translation_residual = float(
            np.linalg.norm(shifted_vector - translation_eigenvalue * vector) / np.sqrt(norm_squared)
        )

    alphabet = {symbol for word in words for symbol in word}
    return CyclicAmplitudeBondProfile(
        length=length,
        support_size=len(words),
        alphabet_size=len(alphabet),
        cut_ranks=tuple(cut_ranks),
        maximum_cut_rank=maximum_rank,
        periodic_bond_dimension_lower_bound=periodic_lower_bound,
        translation_support_closed=support_closed,
        translation_eigenvalue=translation_eigenvalue,
        translation_residual=translation_residual,
        tolerance=tolerance,
    )


def infer_square_qdm_cyclic_column_grammar(
    model: object,
    support_configs: object,
    *,
    window_size: int = 3,
) -> QDMCyclicColumnGrammar:
    """Infer a finite-range cyclic support language from reference states."""
    words = square_qdm_column_words(model, support_configs)
    reference_length = int(model.lattice.lx)
    if window_size < 2 or window_size > reference_length:
        raise ValueError("window_size must lie between two and the reference length.")
    symbols = tuple(sorted({symbol for word in words for symbol in word}))
    windows = {
        tuple(word[(start + offset) % reference_length] for offset in range(window_size))
        for word in words
        for start in range(reference_length)
    }
    return QDMCyclicColumnGrammar(
        circumference=int(model.lattice.ly),
        reference_length=reference_length,
        window_size=int(window_size),
        symbols=symbols,
        allowed_windows=tuple(sorted(windows)),
    )


def enumerate_qdm_cyclic_column_grammar_words(
    grammar: QDMCyclicColumnGrammar,
    length: int,
    *,
    max_words: int = 100_000,
) -> tuple[SquareQDMColumnWord, ...]:
    """Enumerate periodic words accepted by a finite-range column grammar."""
    if length < grammar.window_size:
        raise ValueError("length must be at least the grammar window size.")
    if max_words < 1:
        raise ValueError("max_words must be positive.")

    allowed = set(grammar.allowed_windows)
    prefix_size = grammar.window_size - 1
    continuations: dict[SquareQDMColumnWord, list[SquareQDMColumnSymbol]] = defaultdict(list)
    for window in grammar.allowed_windows:
        continuations[window[:-1]].append(window[-1])
    for values in continuations.values():
        values.sort()

    words: set[SquareQDMColumnWord] = set()

    def closes_periodically(sequence: Sequence[SquareQDMColumnSymbol]) -> bool:
        return all(
            tuple(sequence[(start + offset) % length] for offset in range(grammar.window_size))
            in allowed
            for start in range(length)
        )

    for start_state in sorted(continuations):
        start_sequence = list(start_state)

        def visit(sequence: list[SquareQDMColumnSymbol]) -> None:
            if len(sequence) == length:
                if closes_periodically(sequence):
                    words.add(tuple(sequence))
                    if len(words) > max_words:
                        raise ValueError(
                            "grammar support exceeds max_words; increase the cap explicitly."
                        )
                return
            prefix = tuple(sequence[-prefix_size:])
            for symbol in continuations.get(prefix, ()):  # pragma: no branch
                visit([*sequence, symbol])

        visit(start_sequence)
    return tuple(sorted(words))


def materialize_square_qdm_cyclic_grammar_support(
    grammar: QDMCyclicColumnGrammar,
    target_model: object,
    *,
    potential_value: complex | None = None,
    max_words: int = 100_000,
    tolerance: float = 1e-10,
) -> QDMCyclicGrammarSupport:
    """Materialize the physical QDM configurations accepted by a grammar."""
    if int(target_model.lattice.ly) != grammar.circumference:
        raise ValueError("target circumference must match the grammar.")
    length = int(target_model.lattice.lx)
    words = enumerate_qdm_cyclic_column_grammar_words(
        grammar,
        length,
        max_words=max_words,
    )
    link_lookup: dict[tuple[int, int, str], int] = {}
    for link in target_model.lattice.links:
        x, y = target_model.lattice.sites[int(link.source)].cell
        link_lookup[(int(x), int(y), str(link.kind))] = int(link.id)

    configs = np.zeros(
        (len(words), int(target_model.lattice.num_links)),
        dtype=np.int64,
    )
    for row_index, word in enumerate(words):
        for x, (_incoming, outgoing, vertical) in enumerate(word):
            for y in range(grammar.circumference):
                configs[row_index, link_lookup[(x, y, "x")]] = (outgoing >> y) & 1
                configs[row_index, link_lookup[(x, y, "y")]] = (vertical >> y) & 1

    sectors = tuple(target_model.make_sectors())
    required_count = int(getattr(target_model, "required_count", 1))
    keep = np.ones(len(words), dtype=bool)
    for row_index, config in enumerate(configs):
        if not all(sector.is_satisfied(config) for sector in sectors):
            keep[row_index] = False
            continue
        for site_id in range(int(target_model.lattice.num_sites)):
            incident = np.asarray(
                target_model.lattice.incident_links(site_id),
                dtype=np.int64,
            )
            if int(np.sum(config[incident])) != required_count:
                keep[row_index] = False
                break
    configs = configs[keep]
    words = tuple(word for word, selected in zip(words, keep, strict=True) if selected)

    actions = _qdm_global_plaquette_actions(target_model)
    values = np.zeros(configs.shape[0], dtype=np.complex128)
    for action in actions:
        local_values = configs[:, action.links]
        flippable = np.all(local_values == action.pattern0, axis=1) | np.all(
            local_values == action.pattern1,
            axis=1,
        )
        values[flippable] += action.potential

    if potential_value is None:
        unique_values = np.unique(np.round(values, decimals=12))
        if unique_values.size != 1:
            raise ValueError(
                "grammar support spans multiple potential shells; provide potential_value."
            )
        selected_potential = complex(unique_values[0])
        shell_keep = np.ones(values.size, dtype=bool)
    else:
        selected_potential = complex(potential_value)
        shell_keep = np.isclose(values, selected_potential, atol=tolerance, rtol=0.0)
    configs = configs[shell_keep]
    words = tuple(word for word, selected in zip(words, shell_keep, strict=True) if selected)
    if configs.shape[0] == 0:
        raise ValueError("no grammar configurations remain in the requested potential shell.")
    return QDMCyclicGrammarSupport(
        length=length,
        words=words,
        configs=configs,
        potential_value=selected_potential,
    )


def _boundary_kernel_sparse_resolved(
    boundary: scipy_sparse.csr_matrix,
    *,
    tolerance: float,
    dense_column_limit: int,
    maximum_nullity: int,
) -> tuple[int, float | None, npt.NDArray[np.complex128], bool]:
    n_columns = int(boundary.shape[1])
    if n_columns <= dense_column_limit:
        _rank, nullity, gap, kernel = _sparse_boundary_singular_data(
            boundary,
            tolerance=tolerance,
        )
        return nullity, gap, kernel, True
    if n_columns <= 1:
        dense = np.asarray(boundary.toarray(), dtype=np.complex128)
        kernel = nullspace_svd(dense, tolerance=tolerance)
        return int(kernel.shape[1]), None, kernel, True

    gram = (boundary.conj().T @ boundary).asfptype()
    n_eigenpairs = min(maximum_nullity + 8, n_columns - 1)
    eigenvalues, eigenvectors = scipy_sparse_linalg.eigsh(
        gram,
        k=n_eigenpairs,
        which="SM",
        tol=max(tolerance * 0.1, 1e-13),
        maxiter=50_000,
    )
    order = np.argsort(eigenvalues)
    eigenvalues = np.real(eigenvalues[order])
    eigenvectors = np.asarray(eigenvectors[:, order], dtype=np.complex128)
    gram_scale = float(np.max(np.abs(gram.diagonal()))) if gram.shape[0] else 1.0
    squared_tolerance = max(
        tolerance**2,
        100.0 * np.finfo(np.float64).eps * max(1.0, gram_scale),
    )
    zero_mask = np.abs(eigenvalues) <= squared_tolerance
    nullity = int(np.sum(zero_mask))
    resolved = nullity < n_eigenpairs
    kernel = eigenvectors[:, zero_mask]
    if kernel.shape[1]:
        kernel, _upper = scipy_linalg.qr(kernel, mode="economic")
    positive = eigenvalues[eigenvalues > squared_tolerance]
    gap = None if positive.size == 0 else float(np.sqrt(np.min(positive)))
    return nullity, gap, np.asarray(kernel, dtype=np.complex128), resolved


def _translate_square_qdm_configs(
    model: object,
    configs: npt.NDArray[np.int64],
    *,
    dx: int,
    dy: int = 0,
) -> npt.NDArray[np.int64]:
    link_lookup: dict[tuple[int, int, str], int] = {}
    for link in model.lattice.links:
        x, y = model.lattice.sites[int(link.source)].cell
        link_lookup[(int(x), int(y), str(link.kind))] = int(link.id)
    translated = np.zeros_like(configs)
    lx = int(model.lattice.lx)
    ly = int(model.lattice.ly)
    for link in model.lattice.links:
        x, y = model.lattice.sites[int(link.source)].cell
        target_id = link_lookup[((int(x) + dx) % lx, (int(y) + dy) % ly, str(link.kind))]
        translated[:, target_id] = configs[:, int(link.id)]
    return translated


def _reflect_square_qdm_configs(
    model: object,
    configs: npt.NDArray[np.int64],
    *,
    axis: Literal["x", "y"],
) -> npt.NDArray[np.int64]:
    """Reflect square-QDM link configurations about a lattice coordinate axis."""
    if axis not in {"x", "y"}:
        raise ValueError("axis must be 'x' or 'y'.")
    link_lookup: dict[tuple[int, int, str], int] = {}
    for link in model.lattice.links:
        x, y = model.lattice.sites[int(link.source)].cell
        link_lookup[(int(x), int(y), str(link.kind))] = int(link.id)
    reflected = np.zeros_like(configs)
    lx = int(model.lattice.lx)
    ly = int(model.lattice.ly)
    for link in model.lattice.links:
        x, y = model.lattice.sites[int(link.source)].cell
        kind = str(link.kind)
        if axis == "x":
            target = (
                ((-int(x) - 1) % lx, int(y), kind)
                if kind == "x"
                else ((-int(x)) % lx, int(y), kind)
            )
        else:
            target = (
                (int(x), (-int(y)) % ly, kind)
                if kind == "x"
                else (int(x), (-int(y) - 1) % ly, kind)
            )
        reflected[:, link_lookup[target]] = configs[:, int(link.id)]
    return reflected


def _quarter_turn_square_qdm_configs(
    model: object,
    configs: npt.NDArray[np.int64],
) -> npt.NDArray[np.int64]:
    """Rotate square-QDM link configurations counterclockwise by ninety degrees."""
    lx = int(model.lattice.lx)
    ly = int(model.lattice.ly)
    if lx != ly:
        raise ValueError("a quarter turn requires a square torus.")
    link_lookup: dict[tuple[int, int, str], int] = {}
    for link in model.lattice.links:
        x, y = model.lattice.sites[int(link.source)].cell
        link_lookup[(int(x), int(y), str(link.kind))] = int(link.id)
    rotated = np.zeros_like(configs)
    for link in model.lattice.links:
        x, y = model.lattice.sites[int(link.source)].cell
        kind = str(link.kind)
        if kind == "x":
            target = ((-int(y)) % lx, int(x) % ly, "y")
        else:
            target = ((-int(y) - 1) % lx, int(x) % ly, "x")
        rotated[:, link_lookup[target]] = configs[:, int(link.id)]
    return rotated


def _support_symmetry_permutation(
    support_configs: npt.NDArray[np.int64],
    transformed_configs: npt.NDArray[np.int64],
    *,
    name: str,
) -> npt.NDArray[np.int64]:
    row_by_config = {
        tuple(int(value) for value in config): row_index
        for row_index, config in enumerate(support_configs)
    }
    if len(row_by_config) != support_configs.shape[0]:
        raise ValueError("support_configs must not contain duplicates.")
    permutation: list[int] = []
    for config in transformed_configs:
        row_index = row_by_config.get(tuple(int(value) for value in config))
        if row_index is None:
            raise ValueError(f"support is not invariant under {name}.")
        permutation.append(row_index)
    return np.asarray(permutation, dtype=np.int64)


def _subspace_symmetry_representation(
    basis: npt.NDArray[np.complex128],
    permutation: npt.NDArray[np.int64],
) -> tuple[npt.NDArray[np.complex128], float]:
    if basis.shape[1] == 0:
        return np.zeros((0, 0), dtype=np.complex128), 0.0
    transformed = np.zeros_like(basis)
    transformed[permutation, :] = basis
    representation = basis.conj().T @ transformed
    residual = float(np.linalg.norm(transformed - basis @ representation))
    return np.asarray(representation, dtype=np.complex128), residual


def _translation_sector_multiplicities(
    translation_x: npt.NDArray[np.complex128],
    translation_y: npt.NDArray[np.complex128],
    *,
    lx: int,
    ly: int,
    tolerance: float,
) -> dict[tuple[int, int], int]:
    dimension = int(translation_x.shape[0])
    if dimension == 0:
        return {}
    powers_x = [np.linalg.matrix_power(translation_x, power) for power in range(lx)]
    powers_y = [np.linalg.matrix_power(translation_y, power) for power in range(ly)]
    multiplicities: dict[tuple[int, int], int] = {}
    for momentum_x_index in range(lx):
        for momentum_y_index in range(ly):
            projector = np.zeros((dimension, dimension), dtype=np.complex128)
            for shift_x in range(lx):
                for shift_y in range(ly):
                    phase = np.exp(
                        -2j
                        * np.pi
                        * (momentum_x_index * shift_x / lx + momentum_y_index * shift_y / ly)
                    )
                    projector += phase * (powers_x[shift_x] @ powers_y[shift_y])
            projector /= lx * ly
            singular_values = scipy_linalg.svdvals(projector)
            multiplicity = int(np.sum(singular_values > 10.0 * tolerance))
            if multiplicity:
                multiplicities[(momentum_x_index, momentum_y_index)] = multiplicity
    return multiplicities


def _one_dimensional_character(
    representation: npt.NDArray[np.complex128] | None,
) -> complex | None:
    if representation is None or representation.shape != (1, 1):
        return None
    return complex(representation[0, 0])


def diagnose_square_qdm_finite_bond_transfer_invariant(
    model: object,
    support_configs: object,
    kernel_basis: npt.ArrayLike,
    reference_basis: npt.ArrayLike | None = None,
    *,
    tolerance: float = 1e-10,
) -> SquareQDMFiniteBondTransferInvariantReport:
    """Resolve a fixed-width cage kernel and its quotient into momentum sectors.

    The kernel is interpreted as the finite transfer/bond space admitted by the
    local support language.  ``reference_basis`` typically contains translated
    compact cages.  The relative momentum multiplicities are basis independent.
    """
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")
    configs = np.asarray(support_configs, dtype=np.int64)
    if configs.ndim != 2:
        raise ValueError("support_configs must be two-dimensional.")
    if configs.shape[1] != int(model.lattice.num_links):
        raise ValueError("support_configs width must match model links.")
    kernel = _orthonormal_basis_absolute(
        np.asarray(kernel_basis, dtype=np.complex128),
        tolerance=tolerance,
    )
    if kernel.shape[0] != configs.shape[0]:
        raise ValueError("kernel_basis row count must match support_configs.")
    if reference_basis is None:
        reference = np.zeros((configs.shape[0], 0), dtype=np.complex128)
    else:
        reference = _orthonormal_basis_absolute(
            np.asarray(reference_basis, dtype=np.complex128),
            tolerance=tolerance,
        )
        if reference.shape[0] != configs.shape[0]:
            raise ValueError("reference_basis row count must match support_configs.")
    reference_projection = kernel @ (kernel.conj().T @ reference)
    containment_residual = float(np.linalg.norm(reference - reference_projection))
    if containment_residual > tolerance * max(1.0, float(np.linalg.norm(reference))):
        raise ValueError("reference_basis is not contained in kernel_basis within tolerance.")
    relative = subspace_complement_basis(kernel, reference, tolerance=10.0 * tolerance)

    lx = int(model.lattice.lx)
    ly = int(model.lattice.ly)
    transformed_by_name = {
        "translation_x": _translate_square_qdm_configs(model, configs, dx=1, dy=0),
        "translation_y": _translate_square_qdm_configs(model, configs, dx=0, dy=1),
        "reflection_x": _reflect_square_qdm_configs(model, configs, axis="x"),
        "reflection_y": _reflect_square_qdm_configs(model, configs, axis="y"),
    }
    if lx == ly:
        transformed_by_name["quarter_turn"] = _quarter_turn_square_qdm_configs(
            model,
            configs,
        )
    permutations = {
        name: _support_symmetry_permutation(configs, transformed, name=name)
        for name, transformed in transformed_by_name.items()
    }

    def representations(
        basis: npt.NDArray[np.complex128],
    ) -> tuple[dict[str, npt.NDArray[np.complex128]], float]:
        values: dict[str, npt.NDArray[np.complex128]] = {}
        residual = 0.0
        for name, permutation in permutations.items():
            representation, current_residual = _subspace_symmetry_representation(
                basis,
                permutation,
            )
            values[name] = representation
            residual = max(residual, current_residual)
        return values, residual

    kernel_representations, kernel_residual = representations(kernel)
    reference_representations, reference_residual = representations(reference)
    relative_representations, relative_residual = representations(relative)

    kernel_multiplicities = _translation_sector_multiplicities(
        kernel_representations["translation_x"],
        kernel_representations["translation_y"],
        lx=lx,
        ly=ly,
        tolerance=tolerance,
    )
    reference_multiplicities = _translation_sector_multiplicities(
        reference_representations["translation_x"],
        reference_representations["translation_y"],
        lx=lx,
        ly=ly,
        tolerance=tolerance,
    )
    relative_multiplicities = _translation_sector_multiplicities(
        relative_representations["translation_x"],
        relative_representations["translation_y"],
        lx=lx,
        ly=ly,
        tolerance=tolerance,
    )
    sector_keys = sorted(
        set(kernel_multiplicities) | set(reference_multiplicities) | set(relative_multiplicities)
    )
    sectors = tuple(
        SquareQDMTransferSectorMultiplicity(
            momentum_x_index=momentum_x_index,
            momentum_y_index=momentum_y_index,
            momentum_x=float(2.0 * np.pi * momentum_x_index / lx),
            momentum_y=float(2.0 * np.pi * momentum_y_index / ly),
            kernel_multiplicity=kernel_multiplicities.get(
                (momentum_x_index, momentum_y_index),
                0,
            ),
            reference_multiplicity=reference_multiplicities.get(
                (momentum_x_index, momentum_y_index),
                0,
            ),
            relative_multiplicity=relative_multiplicities.get(
                (momentum_x_index, momentum_y_index),
                0,
            ),
        )
        for momentum_x_index, momentum_y_index in sector_keys
    )

    quotient_tx = relative_representations["translation_x"]
    quotient_ty = relative_representations["translation_y"]
    quotient_rx = relative_representations["reflection_x"]
    quotient_ry = relative_representations["reflection_y"]
    translation_commutator = float(
        np.linalg.norm(quotient_tx @ quotient_ty - quotient_ty @ quotient_tx)
    )
    relation_residuals = [translation_commutator]
    identity = np.eye(relative.shape[1], dtype=np.complex128)
    if relative.shape[1]:
        relation_residuals.extend(
            [
                float(np.linalg.norm(quotient_rx @ quotient_rx - identity)),
                float(np.linalg.norm(quotient_ry @ quotient_ry - identity)),
                float(
                    np.linalg.norm(
                        quotient_rx @ quotient_tx @ quotient_rx.conj().T - quotient_tx.conj().T
                    )
                ),
                float(
                    np.linalg.norm(
                        quotient_ry @ quotient_ty @ quotient_ry.conj().T - quotient_ty.conj().T
                    )
                ),
            ]
        )
        quarter_turn = relative_representations.get("quarter_turn")
        if quarter_turn is not None:
            relation_residuals.extend(
                [
                    float(np.linalg.norm(np.linalg.matrix_power(quarter_turn, 4) - identity)),
                    float(
                        np.linalg.norm(
                            quarter_turn @ quotient_tx @ quarter_turn.conj().T - quotient_ty
                        )
                    ),
                ]
            )
    group_relation_residual = max(relation_residuals, default=0.0)

    return SquareQDMFiniteBondTransferInvariantReport(
        system_size=(lx, ly),
        support_size=int(configs.shape[0]),
        kernel_dimension=int(kernel.shape[1]),
        reference_dimension=int(reference.shape[1]),
        relative_dimension=int(relative.shape[1]),
        reference_containment_residual=containment_residual,
        kernel_symmetry_residual=kernel_residual,
        reference_symmetry_residual=reference_residual,
        relative_symmetry_residual=relative_residual,
        translation_commutator_residual=translation_commutator,
        group_relation_residual=group_relation_residual,
        sectors=sectors,
        quotient_translation_x_character=_one_dimensional_character(quotient_tx),
        quotient_translation_y_character=_one_dimensional_character(quotient_ty),
        quotient_reflection_x_character=_one_dimensional_character(quotient_rx),
        quotient_reflection_y_character=_one_dimensional_character(quotient_ry),
        quotient_quarter_turn_character=_one_dimensional_character(
            relative_representations.get("quarter_turn")
        ),
        tolerance=tolerance,
    )


def _periodic_product_translation_span(
    unit_cell: SquareQDMPeriodicProductUnitCell,
    target_support: QDMCyclicGrammarSupport,
    *,
    tolerance: float,
    max_support_size: int,
) -> npt.NDArray[np.complex128]:
    if unit_cell.repeat_axis != "x":
        raise ValueError("collective fixed-width scan currently requires x repetition.")
    if target_support.length % int(unit_cell.model.lx) != 0:
        raise ValueError("target length must be a multiple of the product unit-cell length.")
    repeats = target_support.length // int(unit_cell.model.lx)
    instance = unit_cell.instantiate(repeats)
    product = materialize_square_qdm_periodic_product_support(
        instance,
        max_support_size=max_support_size,
    )
    row_by_key = {
        tuple(int(value) for value in row): index
        for index, row in enumerate(target_support.configs)
    }
    columns: list[npt.NDArray[np.complex128]] = []
    for dx in range(target_support.length):
        translated = _translate_square_qdm_configs(
            instance.model,
            product.configs,
            dx=dx,
        )
        vector = np.zeros(target_support.support_size, dtype=np.complex128)
        is_contained = True
        for config, amplitude in zip(translated, product.amplitudes, strict=True):
            row_index = row_by_key.get(tuple(int(value) for value in config))
            if row_index is None:
                is_contained = False
                break
            vector[row_index] += complex(amplitude)
        if not is_contained:
            continue
        norm = float(np.linalg.norm(vector))
        if norm > tolerance:
            columns.append(vector / norm)
    if not columns:
        return np.zeros((target_support.support_size, 0), dtype=np.complex128)
    candidates = np.column_stack(columns)
    left, singular_values, _right = scipy_linalg.svd(
        candidates,
        full_matrices=False,
    )
    rank = int(np.sum(singular_values > tolerance))
    return np.asarray(left[:, :rank], dtype=np.complex128)


def scan_square_qdm_collective_locality_extension(
    reference_model: object,
    reference_support_configs: object,
    product_unit_cell: SquareQDMPeriodicProductUnitCell,
    cases: Sequence[tuple[int, int]],
    *,
    potential_per_column: complex = 1.0,
    max_words: int = 100_000,
    max_product_support_size: int = 4096,
    dense_column_limit: int = 512,
    maximum_nullity: int = 32,
    ipr_restarts: int = 64,
    tolerance: float = 1e-9,
) -> QDMLocalGrammarExtensionReport:
    """Test non-factorized fixed-width extensions of a collective support.

    Each ``(window_size, length)`` case first generates every periodic support
    configuration compatible with the corresponding local column grammar.  The
    exact physical boundary kernel is then compared with the span of translated
    certified stripe-product cages.  A positive residual quotient is a genuine
    non-product collective extension within that finite-range support language.
    """
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")
    normalized_cases = tuple((int(window), int(length)) for window, length in cases)
    if not normalized_cases:
        raise ValueError("cases must not be empty.")
    if len(set(normalized_cases)) != len(normalized_cases):
        raise ValueError("cases must not contain duplicates.")

    grammar_by_window: dict[int, QDMCyclicColumnGrammar] = {}
    points: list[QDMLocalGrammarExtensionPoint] = []
    for window_size, length in normalized_cases:
        grammar = grammar_by_window.get(window_size)
        if grammar is None:
            grammar = infer_square_qdm_cyclic_column_grammar(
                reference_model,
                reference_support_configs,
                window_size=window_size,
            )
            grammar_by_window[window_size] = grammar
        target_model = replace(
            reference_model,
            lx=length,
            ly=grammar.circumference,
        )
        support = materialize_square_qdm_cyclic_grammar_support(
            grammar,
            target_model,
            potential_value=complex(potential_per_column) * length,
            max_words=max_words,
            tolerance=tolerance,
        )
        boundary_map = build_qdm_explicit_support_boundary(
            target_model,
            support.configs,
        )
        nullity, gap, kernel, resolved = _boundary_kernel_sparse_resolved(
            boundary_map.boundary,
            tolerance=tolerance,
            dense_column_limit=dense_column_limit,
            maximum_nullity=maximum_nullity,
        )
        product_span = _periodic_product_translation_span(
            product_unit_cell,
            support,
            tolerance=tolerance,
            max_support_size=max_product_support_size,
        )
        overlaps = subspace_principal_overlaps(kernel, product_span)
        intersection_dimension = int(np.sum(overlaps >= 1.0 - 10.0 * tolerance))
        collective_dimension = max(0, nullity - intersection_dimension)
        containment_residual = float(
            np.linalg.norm(product_span - kernel @ (kernel.conj().T @ product_span))
        )

        localized_support_sizes: tuple[int, ...] = ()
        localized_iprs: tuple[float, ...] = ()
        if kernel.shape[1] and kernel.shape[1] <= maximum_nullity:
            localized = localized_basis_by_many_start_ipr(
                kernel,
                config=IPRLocalizationConfig(
                    n_restarts=ipr_restarts,
                    candidate_count=max(ipr_restarts, 32),
                    random_seed=1234,
                    amplitude_tolerance=max(tolerance, 1e-8),
                    rank_tolerance=max(tolerance, 1e-8),
                ),
            )
            localized_support_sizes = tuple(
                int(np.sum(np.abs(localized[:, index]) > 10.0 * tolerance))
                for index in range(localized.shape[1])
            )
            localized_iprs = tuple(
                float(np.sum(np.abs(localized[:, index]) ** 4))
                for index in range(localized.shape[1])
            )

        points.append(
            QDMLocalGrammarExtensionPoint(
                window_size=window_size,
                length=length,
                support_size=support.support_size,
                shell_size=boundary_map.shell_size,
                boundary_nullity=nullity,
                nullity_is_resolved=resolved,
                interference_gap=gap,
                product_translation_span_dimension=int(product_span.shape[1]),
                kernel_product_intersection_dimension=intersection_dimension,
                collective_quotient_dimension=collective_dimension,
                product_containment_residual=containment_residual,
                principal_overlaps=overlaps,
                localized_support_sizes=localized_support_sizes,
                localized_iprs=localized_iprs,
            )
        )

    return QDMLocalGrammarExtensionReport(
        reference_length=int(reference_model.lattice.lx),
        circumference=int(reference_model.lattice.ly),
        points=tuple(points),
        tolerance=tolerance,
    )


def _gf2_rref(
    matrix: npt.ArrayLike,
) -> tuple[npt.NDArray[np.uint8], tuple[int, ...]]:
    array = np.asarray(matrix, dtype=np.uint8).copy() % 2
    row = 0
    pivots: list[int] = []
    for column in range(array.shape[1]):
        pivot = next(
            (index for index in range(row, array.shape[0]) if array[index, column]),
            None,
        )
        if pivot is None:
            continue
        array[[row, pivot]] = array[[pivot, row]]
        for index in range(array.shape[0]):
            if index != row and array[index, column]:
                array[index] ^= array[row]
        pivots.append(column)
        row += 1
        if row == array.shape[0]:
            break
    return array, tuple(pivots)


def _gf2_nullspace(matrix: npt.ArrayLike) -> npt.NDArray[np.uint8]:
    reduced, pivots = _gf2_rref(matrix)
    n_columns = reduced.shape[1]
    free_columns = [column for column in range(n_columns) if column not in pivots]
    basis = np.zeros((n_columns, len(free_columns)), dtype=np.uint8)
    for basis_index, free_column in enumerate(free_columns):
        vector = basis[:, basis_index]
        vector[free_column] = 1
        for row, pivot_column in enumerate(pivots):
            vector[pivot_column] = reduced[row, free_column]
    return basis


def diagnose_real_local_sign_obstruction(
    column_words: Sequence[SquareQDMColumnWord],
    amplitudes: object,
    *,
    window_size: int = 3,
    tolerance: float = 1e-10,
) -> RealLocalSignObstructionReport:
    """Diagnose the discrete real-sign class of a finite cage amplitude."""
    words = tuple(tuple(word) for word in column_words)
    if not words:
        raise ValueError("column_words must not be empty.")
    length = len(words[0])
    if any(len(word) != length for word in words):
        raise ValueError("all column words must have the same length.")
    if window_size < 1 or window_size > length:
        raise ValueError("window_size must lie between one and the word length.")
    vector = np.asarray(amplitudes, dtype=np.complex128).reshape(-1)
    if vector.size != len(words):
        raise ValueError("amplitudes must have one entry per column word.")
    if np.any(np.abs(vector) <= tolerance):
        raise ValueError("all amplitudes must be nonzero on the supplied support.")

    pivot_index = int(np.argmax(np.abs(vector)))
    phase = np.exp(-1j * np.angle(vector[pivot_index]))
    real_vector = phase * vector
    real_structure_residual = float(np.linalg.norm(np.imag(real_vector)))
    if real_structure_residual > 10.0 * tolerance:
        raise ValueError("amplitudes must admit a common real gauge.")
    signs = (np.real(real_vector) < 0.0).astype(np.uint8)

    windows = tuple(
        sorted(
            {
                tuple(word[(start + offset) % length] for offset in range(window_size))
                for word in words
                for start in range(length)
            }
        )
    )
    window_index = {window: index for index, window in enumerate(windows)}
    incidence = np.zeros((len(words), len(windows)), dtype=np.int64)
    for row_index, word in enumerate(words):
        for start in range(length):
            window = tuple(word[(start + offset) % length] for offset in range(window_size))
            incidence[row_index, window_index[window]] += 1

    # The first column is the physically irrelevant global sign.  Including
    # it makes the obstruction independent of the arbitrary overall phase of
    # the quantum state, which is essential when the cyclic word length is even.
    mod2_incidence = np.asarray(incidence % 2, dtype=np.uint8)
    gauge_incidence = np.column_stack([np.ones(len(words), dtype=np.uint8), mod2_incidence])
    _reduced, pivots = _gf2_rref(gauge_incidence)
    incidence_rank = len(pivots)
    augmented = np.column_stack([gauge_incidence, signs])
    _augmented_reduced, augmented_pivots = _gf2_rref(augmented)
    augmented_rank = len(augmented_pivots)
    obstruction_dimension = int(augmented_rank - incidence_rank)

    sign_solution: npt.NDArray[np.uint8] | None = None
    if obstruction_dimension == 0:
        reduced_augmented, pivot_columns = _gf2_rref(augmented)
        sign_solution = np.zeros(len(windows) + 1, dtype=np.uint8)
        for row_index, pivot_column in enumerate(pivot_columns):
            if pivot_column < len(windows) + 1:
                sign_solution[pivot_column] = reduced_augmented[row_index, -1]

    obstruction_witness: npt.NDArray[np.uint8] | None = None
    if obstruction_dimension:
        left_nullspace = _gf2_nullspace(gauge_incidence.T)
        for column_index in range(left_nullspace.shape[1]):
            candidate = left_nullspace[:, column_index]
            if int(np.dot(candidate, signs) % 2) == 1:
                obstruction_witness = candidate
                break

    logarithmic_magnitudes = np.log(np.abs(vector))
    real_gauge_incidence = np.column_stack(
        [np.ones(len(words), dtype=np.float64), incidence.astype(np.float64)]
    )
    local_log_weights, _residuals, _rank, _singular_values = np.linalg.lstsq(
        real_gauge_incidence,
        logarithmic_magnitudes,
        rcond=None,
    )
    magnitude_residual = float(
        np.linalg.norm(real_gauge_incidence @ local_log_weights - logarithmic_magnitudes)
    )
    return RealLocalSignObstructionReport(
        window_size=window_size,
        n_words=len(words),
        n_local_windows=len(windows),
        incidence_rank_mod2=incidence_rank,
        augmented_rank_mod2=augmented_rank,
        obstruction_dimension=obstruction_dimension,
        magnitude_factorization_residual=magnitude_residual,
        real_structure_residual=real_structure_residual,
        obstruction_witness=obstruction_witness,
        local_sign_solution=sign_solution,
    )


@dataclass(frozen=True, slots=True)
class LaurentPolynomialRootMode:
    """Kernel multiplicity of a Laurent constraint symbol at one root of unity."""

    repeat_count: int
    momentum_index: int
    primitive_order: int
    root: complex
    kernel_dimension: int
    free_dimension: int
    torsion_dimension: int
    singular_gap: float | None

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "repeat_count": self.repeat_count,
            "momentum_index": self.momentum_index,
            "primitive_order": self.primitive_order,
            "root": self.root,
            "kernel_dimension": self.kernel_dimension,
            "free_dimension": self.free_dimension,
            "torsion_dimension": self.torsion_dimension,
            "singular_gap": self.singular_gap,
        }


@dataclass(frozen=True, slots=True)
class LaurentPolynomialPeriodicPoint:
    """Periodic-kernel dimensions of a Laurent module at one repetition count."""

    repeat_count: int
    total_kernel_dimension: int
    free_kernel_dimension: int
    torsion_kernel_dimension: int
    root_modes: tuple[LaurentPolynomialRootMode, ...]

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "repeat_count": self.repeat_count,
            "total_kernel_dimension": self.total_kernel_dimension,
            "free_kernel_dimension": self.free_kernel_dimension,
            "torsion_kernel_dimension": self.torsion_kernel_dimension,
            "nonzero_torsion_root_count": sum(
                mode.torsion_dimension > 0 for mode in self.root_modes
            ),
        }


@dataclass(frozen=True, slots=True)
class LaurentPolynomialTorsionOrder:
    """Total torsion multiplicity carried by primitive roots of one order."""

    primitive_order: int
    multiplicity: int
    primitive_root_count: int

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "primitive_order": self.primitive_order,
            "multiplicity": self.multiplicity,
            "primitive_root_count": self.primitive_root_count,
        }


@dataclass(frozen=True, slots=True)
class LaurentPolynomialConstraintModuleReport:
    """Free and root-of-unity torsion data of a finite-range Laurent module.

    A translation-invariant finite-range constraint family is represented by

    ``B(z) = sum_d z**d B_d``

    over the Laurent ring ``C[z, z**-1]``.  The generic kernel dimension is the
    rank of the free module.  Additional kernel vectors occurring only at roots
    of unity are torsion modes on periodic systems.  Their primitive root orders
    are discrete and remain fixed until a determinantal factor changes.

    The calculation is numerical but uses only the small Bloch symbol.  It is
    exact up to ``tolerance`` for the supplied coefficient matrices and sampled
    roots of unity.
    """

    n_rows: int
    n_columns: int
    displacements: tuple[int, ...]
    generic_rank: int
    free_kernel_rank: int
    generic_rank_sample_count: int
    generic_rank_is_stable: bool
    periodic_points: tuple[LaurentPolynomialPeriodicPoint, ...]
    torsion_orders: tuple[LaurentPolynomialTorsionOrder, ...]
    tolerance: float

    @property
    def has_free_generators(self) -> bool:
        return self.free_kernel_rank > 0

    @property
    def has_root_of_unity_torsion(self) -> bool:
        return any(entry.multiplicity > 0 for entry in self.torsion_orders)

    @property
    def module_label(self) -> str:
        if self.has_free_generators and self.has_root_of_unity_torsion:
            return "free_plus_root_of_unity_torsion"
        if self.has_free_generators:
            return "free"
        if self.has_root_of_unity_torsion:
            return "root_of_unity_torsion"
        return "trivial_kernel"

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "displacements": self.displacements,
            "generic_rank": self.generic_rank,
            "free_kernel_rank": self.free_kernel_rank,
            "generic_rank_sample_count": self.generic_rank_sample_count,
            "generic_rank_is_stable": self.generic_rank_is_stable,
            "module_label": self.module_label,
            "torsion_orders": tuple(entry.to_summary_dict() for entry in self.torsion_orders),
            "periodic_points": tuple(point.to_summary_dict() for point in self.periodic_points),
            "tolerance": self.tolerance,
        }


@dataclass(frozen=True, slots=True)
class LaurentDimensionDivisibilityViolation:
    """Violation of root-set inclusion between two periodic lengths."""

    divisor_repeat_count: int
    multiple_repeat_count: int
    divisor_torsion_dimension: int
    multiple_torsion_dimension: int

    def to_summary_dict(self) -> dict[str, int]:
        return {
            "divisor_repeat_count": self.divisor_repeat_count,
            "multiple_repeat_count": self.multiple_repeat_count,
            "divisor_torsion_dimension": self.divisor_torsion_dimension,
            "multiple_torsion_dimension": self.multiple_torsion_dimension,
        }


@dataclass(frozen=True, slots=True)
class LaurentPeriodicDimensionConsistencyReport:
    """Necessary Laurent-module consistency test for observed periodic nullities.

    For a fixed Laurent symbol, every root of ``z**N - 1`` is also a root of
    ``z**M - 1`` whenever ``N`` divides ``M``.  After subtracting the extensive
    free contribution, periodic torsion dimensions must therefore be monotone
    under divisibility.  On a divisor-closed data set, Möbius inversion also
    gives non-negative primitive-order multiplicities.

    Passing these tests is necessary, not sufficient, for the data to come from
    one fixed finite-range translation-invariant Laurent module.
    """

    repeat_counts: tuple[int, ...]
    observed_dimensions: tuple[int, ...]
    assumed_free_rank: int
    torsion_dimensions: tuple[int, ...]
    divisibility_violations: tuple[LaurentDimensionDivisibilityViolation, ...]
    primitive_order_multiplicities: tuple[tuple[int, int], ...]
    incomplete_primitive_orders: tuple[int, ...]

    @property
    def has_negative_torsion_dimension(self) -> bool:
        return any(value < 0 for value in self.torsion_dimensions)

    @property
    def has_negative_primitive_multiplicity(self) -> bool:
        return any(value < 0 for _order, value in self.primitive_order_multiplicities)

    @property
    def passes_necessary_conditions(self) -> bool:
        return bool(
            not self.has_negative_torsion_dimension
            and not self.divisibility_violations
            and not self.has_negative_primitive_multiplicity
        )

    @property
    def obstruction_label(self) -> str:
        if self.has_negative_torsion_dimension:
            return "below_free_rank"
        if self.divisibility_violations:
            return "divisibility_violation"
        if self.has_negative_primitive_multiplicity:
            return "negative_primitive_multiplicity"
        if self.incomplete_primitive_orders:
            return "passes_partial_necessary_tests"
        return "passes_necessary_tests"

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "repeat_counts": self.repeat_counts,
            "observed_dimensions": self.observed_dimensions,
            "assumed_free_rank": self.assumed_free_rank,
            "torsion_dimensions": self.torsion_dimensions,
            "divisibility_violations": tuple(
                violation.to_summary_dict() for violation in self.divisibility_violations
            ),
            "primitive_order_multiplicities": self.primitive_order_multiplicities,
            "incomplete_primitive_orders": self.incomplete_primitive_orders,
            "passes_necessary_conditions": self.passes_necessary_conditions,
            "obstruction_label": self.obstruction_label,
        }


def _normalize_laurent_coefficient_terms(
    coefficient_terms: Sequence[tuple[int, object]],
) -> tuple[tuple[int, npt.NDArray[np.complex128]], ...]:
    if not coefficient_terms:
        raise ValueError("coefficient_terms must not be empty.")
    combined: dict[int, npt.NDArray[np.complex128]] = {}
    shape: tuple[int, int] | None = None
    for raw_displacement, raw_matrix in coefficient_terms:
        displacement = int(raw_displacement)
        matrix = np.asarray(as_dense_array(raw_matrix), dtype=np.complex128)
        if matrix.ndim != 2:
            raise ValueError("every Laurent coefficient must be a matrix.")
        if shape is None:
            shape = matrix.shape
        elif matrix.shape != shape:
            raise ValueError("every Laurent coefficient must have the same shape.")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("Laurent coefficients must contain finite values.")
        if displacement in combined:
            combined[displacement] = combined[displacement] + matrix
        else:
            combined[displacement] = matrix.copy()
    return tuple(sorted(combined.items()))


def laurent_polynomial_constraint_symbol(
    coefficient_terms: Sequence[tuple[int, object]],
    z: complex,
) -> npt.NDArray[np.complex128]:
    """Evaluate ``sum_d z**d B_d`` for one nonzero complex translation value."""
    point = complex(z)
    if not np.isfinite(point.real) or not np.isfinite(point.imag) or point == 0.0:
        raise ValueError("z must be a finite nonzero complex number.")
    normalized = _normalize_laurent_coefficient_terms(coefficient_terms)
    symbol = np.zeros_like(normalized[0][1], dtype=np.complex128)
    for displacement, matrix in normalized:
        symbol += point**displacement * matrix
    return symbol


def _laurent_symbol_rank_gap(
    symbol: npt.NDArray[np.complex128],
    *,
    tolerance: float,
) -> tuple[int, int, float | None]:
    singular_values = scipy_linalg.svdvals(symbol)
    rank = int(np.sum(singular_values > tolerance))
    nullity = int(symbol.shape[1] - rank)
    positive = singular_values[singular_values > tolerance]
    gap = None if positive.size == 0 else float(np.min(positive))
    return rank, nullity, gap


def _primitive_root_key(momentum_index: int, repeat_count: int) -> tuple[int, int]:
    divisor = int(np.gcd(momentum_index, repeat_count))
    order = repeat_count // divisor
    numerator = (momentum_index // divisor) % order
    return order, numerator


def diagnose_laurent_polynomial_constraint_module(
    coefficient_terms: Sequence[tuple[int, object]],
    repeat_counts: Sequence[int] | npt.NDArray[np.integer],
    *,
    generic_sample_count: int = 12,
    tolerance: float = 1e-10,
) -> LaurentPolynomialConstraintModuleReport:
    """Diagnose free and root-of-unity torsion sectors of ``B(z)``.

    ``generic_sample_count`` non-root-of-unity complex points are used to find
    the maximum symbol rank, which equals the rank over ``C(z)`` away from a
    nongeneric algebraic set.  Periodic kernels are then evaluated exactly at
    every root of unity for the requested repetition counts.
    """
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")
    if generic_sample_count < 3:
        raise ValueError("generic_sample_count must be at least three.")
    normalized = _normalize_laurent_coefficient_terms(coefficient_terms)
    counts = tuple(int(value) for value in np.asarray(repeat_counts).reshape(-1))
    if not counts or any(value <= 0 for value in counts):
        raise ValueError("repeat_counts must contain positive values.")
    if len(set(counts)) != len(counts):
        raise ValueError("repeat_counts must not contain duplicates.")

    sample_ranks: list[int] = []
    for sample_index in range(generic_sample_count):
        radius = 0.71 if sample_index % 2 == 0 else 1.37
        angle = np.sqrt(2.0) * (sample_index + 1)
        z = radius * np.exp(1.0j * angle)
        symbol = laurent_polynomial_constraint_symbol(normalized, z)
        rank, _nullity, _gap = _laurent_symbol_rank_gap(symbol, tolerance=tolerance)
        sample_ranks.append(rank)
    generic_rank = max(sample_ranks)
    n_rows, n_columns = normalized[0][1].shape
    free_rank = n_columns - generic_rank
    generic_rank_is_stable = sample_ranks.count(generic_rank) >= generic_sample_count // 2

    periodic_points: list[LaurentPolynomialPeriodicPoint] = []
    root_torsion: dict[tuple[int, int], int] = {}
    for repeat_count in counts:
        root_modes: list[LaurentPolynomialRootMode] = []
        total_kernel = 0
        for momentum_index in range(repeat_count):
            root = np.exp(2.0j * np.pi * momentum_index / repeat_count)
            symbol = laurent_polynomial_constraint_symbol(normalized, root)
            _rank, nullity, gap = _laurent_symbol_rank_gap(symbol, tolerance=tolerance)
            torsion = max(0, nullity - free_rank)
            order, numerator = _primitive_root_key(momentum_index, repeat_count)
            key = (order, numerator)
            previous = root_torsion.get(key)
            if previous is not None and previous != torsion:
                raise ValueError(
                    "inconsistent numerical nullity for the same primitive root; "
                    "increase tolerance or inspect symbol conditioning."
                )
            root_torsion[key] = torsion
            total_kernel += nullity
            root_modes.append(
                LaurentPolynomialRootMode(
                    repeat_count=repeat_count,
                    momentum_index=momentum_index,
                    primitive_order=order,
                    root=complex(root),
                    kernel_dimension=nullity,
                    free_dimension=free_rank,
                    torsion_dimension=torsion,
                    singular_gap=gap,
                )
            )
        periodic_points.append(
            LaurentPolynomialPeriodicPoint(
                repeat_count=repeat_count,
                total_kernel_dimension=total_kernel,
                free_kernel_dimension=repeat_count * free_rank,
                torsion_kernel_dimension=total_kernel - repeat_count * free_rank,
                root_modes=tuple(root_modes),
            )
        )

    by_order: dict[int, list[int]] = defaultdict(list)
    for (order, _numerator), multiplicity in root_torsion.items():
        if multiplicity > 0:
            by_order[order].append(multiplicity)
    torsion_orders = tuple(
        LaurentPolynomialTorsionOrder(
            primitive_order=order,
            multiplicity=int(sum(multiplicities)),
            primitive_root_count=len(multiplicities),
        )
        for order, multiplicities in sorted(by_order.items())
    )
    return LaurentPolynomialConstraintModuleReport(
        n_rows=n_rows,
        n_columns=n_columns,
        displacements=tuple(displacement for displacement, _matrix in normalized),
        generic_rank=generic_rank,
        free_kernel_rank=free_rank,
        generic_rank_sample_count=generic_sample_count,
        generic_rank_is_stable=generic_rank_is_stable,
        periodic_points=tuple(periodic_points),
        torsion_orders=torsion_orders,
        tolerance=tolerance,
    )


def diagnose_laurent_periodic_dimension_consistency(
    repeat_counts: Sequence[int] | npt.NDArray[np.integer],
    observed_dimensions: Sequence[int] | npt.NDArray[np.integer],
    *,
    assumed_free_rank: int = 0,
) -> LaurentPeriodicDimensionConsistencyReport:
    """Test necessary fixed-symbol constraints on observed periodic dimensions."""
    counts = tuple(int(value) for value in np.asarray(repeat_counts).reshape(-1))
    dimensions = tuple(int(value) for value in np.asarray(observed_dimensions).reshape(-1))
    if not counts or len(counts) != len(dimensions):
        raise ValueError("repeat_counts and observed_dimensions must have equal nonzero length.")
    if any(value <= 0 for value in counts):
        raise ValueError("repeat_counts must be positive.")
    if any(value < 0 for value in dimensions):
        raise ValueError("observed_dimensions must be non-negative.")
    if len(set(counts)) != len(counts):
        raise ValueError("repeat_counts must not contain duplicates.")
    if assumed_free_rank < 0:
        raise ValueError("assumed_free_rank must be non-negative.")

    ordered = sorted(zip(counts, dimensions, strict=True))
    counts = tuple(item[0] for item in ordered)
    dimensions = tuple(item[1] for item in ordered)
    torsion = tuple(
        dimension - repeat_count * assumed_free_rank
        for repeat_count, dimension in zip(counts, dimensions, strict=True)
    )
    torsion_by_count = dict(zip(counts, torsion, strict=True))

    violations: list[LaurentDimensionDivisibilityViolation] = []
    for divisor_count, divisor_torsion in zip(counts, torsion, strict=True):
        for multiple_count, multiple_torsion in zip(counts, torsion, strict=True):
            if multiple_count <= divisor_count or multiple_count % divisor_count:
                continue
            if multiple_torsion < divisor_torsion:
                violations.append(
                    LaurentDimensionDivisibilityViolation(
                        divisor_repeat_count=divisor_count,
                        multiple_repeat_count=multiple_count,
                        divisor_torsion_dimension=divisor_torsion,
                        multiple_torsion_dimension=multiple_torsion,
                    )
                )

    primitive: dict[int, int] = {}
    incomplete: list[int] = []
    for repeat_count in counts:
        proper_divisors = tuple(
            divisor for divisor in range(1, repeat_count) if repeat_count % divisor == 0
        )
        if any(divisor not in torsion_by_count for divisor in proper_divisors):
            incomplete.append(repeat_count)
            continue
        primitive[repeat_count] = torsion_by_count[repeat_count] - sum(
            primitive[divisor] for divisor in proper_divisors
        )

    return LaurentPeriodicDimensionConsistencyReport(
        repeat_counts=counts,
        observed_dimensions=dimensions,
        assumed_free_rank=int(assumed_free_rank),
        torsion_dimensions=torsion,
        divisibility_violations=tuple(violations),
        primitive_order_multiplicities=tuple(sorted(primitive.items())),
        incomplete_primitive_orders=tuple(incomplete),
    )
