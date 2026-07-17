from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import numpy.typing as npt
import scipy.linalg as scipy_linalg
import scipy.sparse as scipy_sparse

from qlinks.caging.invariant_subspace import invariant_boundary_nullspace
from qlinks.caging.nullspace import as_dense_array, nullspace_svd

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
