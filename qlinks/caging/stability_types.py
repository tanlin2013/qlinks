"""Report and contract types for cage-stability diagnostics.

Numerical algorithms live in the focused
:mod:`qlinks.caging.stability_*` modules; this module contains the
immutable data returned by those algorithms. The separation is intentionally
mechanical:
it changes ownership, not the scientific definitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
import scipy.sparse as scipy_sparse

CoefficientField = Literal["real", "complex"]
SquareQDMColumnSymbol = tuple[int, int, int]
SquareQDMColumnWord = tuple[SquareQDMColumnSymbol, ...]


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


@dataclass(frozen=True, slots=True)
class BoundaryIncidenceCohomologyReport:
    """Cohomology of a two-channel support-to-boundary constraint map.

    Every active boundary row is interpreted as an edge joining the two support
    configurations on which it acts.  If the coefficient ratios define a flat
    multiplicative gauge, the boundary map is diagonally equivalent to an
    oriented graph-incidence matrix.  Its right kernel is then the zeroth
    cohomology of the support graph: one gauge-covariantly constant vector per
    connected component.

    ``betti_1`` counts graph cycles, but the cage vectors reported here belong
    to ``H^0`` rather than to the loop sector ``H^1``.  This distinction is
    important when comparing persistent many-body cages with noncontractible
    loop states in singular flat bands.
    """

    n_support_vertices: int
    n_boundary_rows: int
    n_active_constraints: int
    active_row_weight_histogram: tuple[tuple[int, int], ...]
    is_two_channel: bool
    equal_magnitude_residual: float | None
    gauge_flatness_residual: float | None
    incidence_residual: float | None
    connected_component_count: int | None
    betti_0: int | None
    betti_1: int | None
    kernel_dimension: int
    h0_intersection_dimension: int | None
    state_h0_weight: float | None
    interference_gap: float | None
    gauge_basis: npt.NDArray[np.complex128]
    edge_endpoints: tuple[tuple[int, int], ...]
    tolerance: float

    @property
    def is_flat_incidence_problem(self) -> bool:
        return bool(
            self.is_two_channel
            and self.gauge_flatness_residual is not None
            and self.gauge_flatness_residual <= 10.0 * self.tolerance
            and self.incidence_residual is not None
            and self.incidence_residual <= 10.0 * self.tolerance
        )

    @property
    def kernel_is_exact_h0(self) -> bool:
        return bool(
            self.is_flat_incidence_problem
            and self.betti_0 is not None
            and self.kernel_dimension == self.betti_0
            and self.h0_intersection_dimension == self.betti_0
        )

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "n_support_vertices": self.n_support_vertices,
            "n_boundary_rows": self.n_boundary_rows,
            "n_active_constraints": self.n_active_constraints,
            "active_row_weight_histogram": self.active_row_weight_histogram,
            "is_two_channel": self.is_two_channel,
            "equal_magnitude_residual": self.equal_magnitude_residual,
            "gauge_flatness_residual": self.gauge_flatness_residual,
            "incidence_residual": self.incidence_residual,
            "connected_component_count": self.connected_component_count,
            "betti_0": self.betti_0,
            "betti_1": self.betti_1,
            "kernel_dimension": self.kernel_dimension,
            "h0_intersection_dimension": self.h0_intersection_dimension,
            "state_h0_weight": self.state_h0_weight,
            "interference_gap": self.interference_gap,
            "is_flat_incidence_problem": self.is_flat_incidence_problem,
            "kernel_is_exact_h0": self.kernel_is_exact_h0,
            "tolerance": self.tolerance,
        }


@dataclass(frozen=True, slots=True)
class HardCoreLaurentLiftReport:
    """Hard-core many-body lift of a scalar Laurent transfer root.

    Support configurations are interpreted as fixed-particle-number binary
    words on a periodic chain.  Two-channel boundary constraints must exchange
    one occupied site with an adjacent empty site.  If every right-moving
    exchange transports the cage amplitude by the same factor ``zeta``, the
    many-body state is the hard-core lift of the scalar local relation whose
    Laurent symbol vanishes at ``z=zeta``.

    A unit-modulus root has a discrete primitive order, but the corresponding
    Toeplitz symbol is not Fredholm because it vanishes on the unit circle.
    The primitive order is therefore a translation/cyclotomic invariant of the
    exact interference rule, not a conventional Fredholm winding number.
    """

    length: int
    particle_number: int
    support_size: int
    exchange_constraint_count: int
    all_constraints_are_nearest_neighbor_exchanges: bool
    uniform_transport_factor: complex | None
    transport_residual: float | None
    primitive_root_order: int | None
    periodic_compatibility_residual: float | None
    amplitude_factorization_residual: float | None
    one_site_translation_character: complex | None
    one_site_translation_residual: float | None
    has_unit_circle_symbol_zero: bool
    incidence_cohomology: BoundaryIncidenceCohomologyReport
    tolerance: float

    @property
    def is_cyclotomic_hard_core_lift(self) -> bool:
        return bool(
            self.all_constraints_are_nearest_neighbor_exchanges
            and self.uniform_transport_factor is not None
            and self.transport_residual is not None
            and self.transport_residual <= 10.0 * self.tolerance
            and self.primitive_root_order is not None
            and self.periodic_compatibility_residual is not None
            and self.periodic_compatibility_residual <= 10.0 * self.tolerance
            and self.amplitude_factorization_residual is not None
            and self.amplitude_factorization_residual <= 10.0 * self.tolerance
            and self.incidence_cohomology.kernel_is_exact_h0
        )

    @property
    def toeplitz_fredholm_index_is_defined(self) -> bool:
        return not self.has_unit_circle_symbol_zero

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "length": self.length,
            "particle_number": self.particle_number,
            "support_size": self.support_size,
            "exchange_constraint_count": self.exchange_constraint_count,
            "all_constraints_are_nearest_neighbor_exchanges": (
                self.all_constraints_are_nearest_neighbor_exchanges
            ),
            "uniform_transport_factor": self.uniform_transport_factor,
            "transport_residual": self.transport_residual,
            "primitive_root_order": self.primitive_root_order,
            "periodic_compatibility_residual": self.periodic_compatibility_residual,
            "amplitude_factorization_residual": self.amplitude_factorization_residual,
            "one_site_translation_character": self.one_site_translation_character,
            "one_site_translation_residual": self.one_site_translation_residual,
            "has_unit_circle_symbol_zero": self.has_unit_circle_symbol_zero,
            "toeplitz_fredholm_index_is_defined": (self.toeplitz_fredholm_index_is_defined),
            "is_cyclotomic_hard_core_lift": self.is_cyclotomic_hard_core_lift,
            "incidence_cohomology": self.incidence_cohomology.to_summary_dict(),
            "tolerance": self.tolerance,
        }


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
class ReducedConstraintFredholmCandidateReport:
    """Reduced constraint map after quotienting a known exact kernel.

    The report deliberately distinguishes a square Fredholm-symbol candidate
    from a strictly rectangular injective map.  For a tall complex map, the
    polar part takes values in a complex Stiefel manifold with trivial first
    homotopy group, so a determinant winding is not intrinsic without an
    additional choice of codomain frame.
    """

    codomain_dimension: int
    domain_dimension: int
    kernel_dimension: int
    reduced_domain_dimension: int
    reduced_rank: int
    codomain_excess: int
    reduced_singular_values: npt.NDArray[np.float64]
    reduced_gap: float | None
    canonical_log_abs_determinant: float | None
    canonical_determinant_phase: float | None
    is_reduced_injective: bool
    is_square_symbol_candidate: bool
    classification: str
    tolerance: float

    @property
    def admits_intrinsic_scalar_winding(self) -> bool:
        return bool(self.is_reduced_injective and self.is_square_symbol_candidate)

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "codomain_dimension": self.codomain_dimension,
            "domain_dimension": self.domain_dimension,
            "kernel_dimension": self.kernel_dimension,
            "reduced_domain_dimension": self.reduced_domain_dimension,
            "reduced_rank": self.reduced_rank,
            "codomain_excess": self.codomain_excess,
            "reduced_gap": self.reduced_gap,
            "canonical_log_abs_determinant": self.canonical_log_abs_determinant,
            "canonical_determinant_phase": self.canonical_determinant_phase,
            "is_reduced_injective": self.is_reduced_injective,
            "is_square_symbol_candidate": self.is_square_symbol_candidate,
            "admits_intrinsic_scalar_winding": self.admits_intrinsic_scalar_winding,
            "classification": self.classification,
            "tolerance": self.tolerance,
        }


@dataclass(frozen=True, slots=True)
class QDMCompactCageReducedWindingPoint:
    """Reduced state- and coupling-space data for one fixed-width member."""

    repeats: int
    system_size: tuple[int, int]
    support_size: int
    shell_size: int
    state_complement: ReducedConstraintFredholmCandidateReport
    kinetic_term_count: int
    kinetic_compatible_dimension: int
    kinetic_quotient_dimension: int
    kinetic_quotient_singular_values: npt.NDArray[np.float64]
    kinetic_quotient_gap: float | None
    local_pair_offsets: tuple[tuple[int, int], ...]
    intercell_gram_norm: float
    unit_cell_gram_residual: float

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "repeats": self.repeats,
            "system_size": self.system_size,
            "support_size": self.support_size,
            "shell_size": self.shell_size,
            "state_reduced_domain_dimension": (self.state_complement.reduced_domain_dimension),
            "state_codomain_excess": self.state_complement.codomain_excess,
            "state_reduced_gap": self.state_complement.reduced_gap,
            "state_scalar_winding_eligible": (
                self.state_complement.admits_intrinsic_scalar_winding
            ),
            "kinetic_term_count": self.kinetic_term_count,
            "kinetic_compatible_dimension": self.kinetic_compatible_dimension,
            "kinetic_quotient_dimension": self.kinetic_quotient_dimension,
            "kinetic_quotient_gap": self.kinetic_quotient_gap,
            "local_pair_offsets": self.local_pair_offsets,
            "intercell_gram_norm": self.intercell_gram_norm,
            "unit_cell_gram_residual": self.unit_cell_gram_residual,
        }


@dataclass(frozen=True, slots=True)
class QDMCompactCageReducedWindingReport:
    """Fredholm-winding audit of the compact square-QDM cage sequence."""

    repeat_axis: str
    unit_cell_size: tuple[int, int]
    local_pair_offsets: tuple[tuple[int, int], ...]
    reduced_coupling_symbol: npt.NDArray[np.complex128]
    reduced_coupling_winding: int
    reduced_coupling_gap: float
    points: tuple[QDMCompactCageReducedWindingPoint, ...]
    classification: str
    tolerance: float

    @property
    def state_space_has_intrinsic_scalar_winding_candidate(self) -> bool:
        return bool(
            self.points
            and all(point.state_complement.admits_intrinsic_scalar_winding for point in self.points)
        )

    @property
    def has_uniform_fixed_width_gap(self) -> bool:
        if not self.points:
            return False
        gaps = np.asarray(
            [point.kinetic_quotient_gap for point in self.points],
            dtype=np.float64,
        )
        return bool(
            np.all(np.isfinite(gaps)) and np.max(np.abs(gaps - gaps[0])) <= 10.0 * self.tolerance
        )

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "repeat_axis": self.repeat_axis,
            "unit_cell_size": self.unit_cell_size,
            "local_pair_offsets": self.local_pair_offsets,
            "reduced_coupling_winding": self.reduced_coupling_winding,
            "reduced_coupling_gap": self.reduced_coupling_gap,
            "state_space_has_intrinsic_scalar_winding_candidate": (
                self.state_space_has_intrinsic_scalar_winding_candidate
            ),
            "has_uniform_fixed_width_gap": self.has_uniform_fixed_width_gap,
            "classification": self.classification,
            "points": tuple(point.to_summary_dict() for point in self.points),
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
