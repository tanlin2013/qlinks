"""Core perturbative and continuation diagnostics for cage stability."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import numpy.typing as npt
import scipy.linalg as scipy_linalg
import scipy.sparse as scipy_sparse

from qlinks.caging.invariant_subspace import invariant_boundary_nullspace
from qlinks.caging.nullspace import as_dense_array, nullspace_svd
from qlinks.caging.stability.types import (
    CageBranchPoint,
    CageBranchReport,
    CageCompatibilityHierarchyReport,
    CageHamiltonianBlocks,
    CageJacobianConditioningReport,
    CageRecordStabilitySummary,
    CageStabilityDiagnostic,
    CoefficientField,
    FixedCageStateCompatibilityReport,
    LinearizedCageObstructionReport,
    PerturbationCompatibilityDiagnostic,
    RandomCageStabilityAggregate,
    RandomCageStabilityReport,
    RandomCageStabilitySample,
    SupportEigenstateBranchPoint,
    SupportEigenstateBranchReport,
)


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
