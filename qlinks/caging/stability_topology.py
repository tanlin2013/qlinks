"""Chiral, locality, CLS-completeness, and cohomological cage diagnostics."""

from __future__ import annotations

import itertools
from typing import Sequence

import numpy as np
import numpy.typing as npt
import scipy.linalg as scipy_linalg

from qlinks.caging.nullspace import as_dense_array, nullspace_svd
from qlinks.caging.stability_core import partition_cage_hamiltonian, subspace_principal_overlaps
from qlinks.caging.stability_symmetry import _subspace_symmetry_representation
from qlinks.caging.stability_types import (
    BoundaryIncidenceCohomologyReport,
    ChiralIndexReport,
    CoefficientField,
    FixedCageManifoldCompatibilityReport,
    HardCoreLaurentLiftReport,
    LocalityRestrictedChiralProfileReport,
    ManyBodyCLSCompletenessReport,
    ManyBodyCLSGeneratorOrbitEntry,
    ManyBodyCLSTranslationSector,
    ManyBodyTopologicalLocalizationReport,
    RegionalCageQuotientReport,
    RegionalChiralIndexEntry,
    RegionalChiralKernelSpanReport,
    RelativeMod2CycleReport,
    SignedBoundaryCycle,
    SignedBoundaryHolonomyReport,
)


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


def _positive_singular_gap(
    matrix: npt.NDArray[np.complex128],
    *,
    tolerance: float,
) -> tuple[int, float | None]:
    singular_values = scipy_linalg.svdvals(matrix)
    rank = int(np.sum(singular_values > tolerance))
    positive = singular_values[singular_values > tolerance]
    gap = None if positive.size == 0 else float(np.min(positive))
    return int(matrix.shape[1] - rank), gap


def diagnose_boundary_incidence_cohomology(
    boundary: object,
    state: npt.ArrayLike | None = None,
    *,
    tolerance: float = 1e-10,
) -> BoundaryIncidenceCohomologyReport:
    """Diagnose whether a boundary map is a flat two-channel incidence problem."""
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")
    matrix = np.asarray(as_dense_array(boundary), dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("boundary must be a matrix with at least one support column.")
    row_weights = np.sum(np.abs(matrix) > tolerance, axis=1).astype(np.int64)
    active_rows = np.flatnonzero(row_weights > 0)
    histogram = tuple(
        (int(weight), int(np.sum(row_weights[active_rows] == weight)))
        for weight in sorted(set(int(value) for value in row_weights[active_rows]))
    )
    kernel_dimension, interference_gap = _positive_singular_gap(
        matrix,
        tolerance=tolerance,
    )
    n_vertices = int(matrix.shape[1])
    if active_rows.size == 0 or np.any(row_weights[active_rows] != 2):
        return BoundaryIncidenceCohomologyReport(
            n_support_vertices=n_vertices,
            n_boundary_rows=int(matrix.shape[0]),
            n_active_constraints=int(active_rows.size),
            active_row_weight_histogram=histogram,
            is_two_channel=False,
            equal_magnitude_residual=None,
            gauge_flatness_residual=None,
            incidence_residual=None,
            connected_component_count=None,
            betti_0=None,
            betti_1=None,
            kernel_dimension=kernel_dimension,
            h0_intersection_dimension=None,
            state_h0_weight=None,
            interference_gap=interference_gap,
            gauge_basis=np.zeros((n_vertices, 0), dtype=np.complex128),
            edge_endpoints=(),
            tolerance=tolerance,
        )

    edges: list[tuple[int, int]] = []
    transports: list[complex] = []
    magnitude_residual = 0.0
    adjacency: list[list[tuple[int, int, bool]]] = [[] for _ in range(n_vertices)]
    for edge_index, row_index in enumerate(active_rows):
        columns = np.flatnonzero(np.abs(matrix[row_index]) > tolerance)
        first, second = int(columns[0]), int(columns[1])
        first_value = complex(matrix[row_index, first])
        second_value = complex(matrix[row_index, second])
        scale = max(abs(first_value), abs(second_value), tolerance)
        magnitude_residual = max(
            magnitude_residual,
            abs(abs(first_value) - abs(second_value)) / scale,
        )
        transport = -first_value / second_value
        edges.append((first, second))
        transports.append(transport)
        adjacency[first].append((second, edge_index, True))
        adjacency[second].append((first, edge_index, False))

    gauge = np.zeros(n_vertices, dtype=np.complex128)
    component_labels = np.full(n_vertices, -1, dtype=np.int64)
    flatness_residual = 0.0
    component_count = 0
    for start in range(n_vertices):
        if component_labels[start] >= 0:
            continue
        component_labels[start] = component_count
        gauge[start] = 1.0 + 0.0j
        stack = [start]
        while stack:
            current = stack.pop()
            for target, edge_index, forward in adjacency[current]:
                factor = transports[edge_index] if forward else 1.0 / transports[edge_index]
                candidate = factor * gauge[current]
                if component_labels[target] < 0:
                    component_labels[target] = component_count
                    gauge[target] = candidate
                    stack.append(target)
                else:
                    denominator = max(abs(candidate), abs(gauge[target]), tolerance)
                    flatness_residual = max(
                        flatness_residual,
                        abs(gauge[target] - candidate) / denominator,
                    )
        component_count += 1

    incidence_residual = 0.0
    for row_index, (first, second) in zip(active_rows, edges, strict=True):
        transformed = np.asarray(
            [matrix[row_index, first] * gauge[first], matrix[row_index, second] * gauge[second]],
            dtype=np.complex128,
        )
        scale = max(float(np.max(np.abs(transformed))), tolerance)
        incidence_residual = max(incidence_residual, abs(np.sum(transformed)) / scale)

    gauge_columns: list[npt.NDArray[np.complex128]] = []
    for component in range(component_count):
        vector = np.zeros(n_vertices, dtype=np.complex128)
        indices = np.flatnonzero(component_labels == component)
        vector[indices] = gauge[indices]
        norm = float(np.linalg.norm(vector))
        if norm > tolerance:
            gauge_columns.append(vector / norm)
    gauge_basis = (
        np.column_stack(gauge_columns)
        if gauge_columns
        else np.zeros((n_vertices, 0), dtype=np.complex128)
    )
    actual_kernel = nullspace_svd(matrix, tolerance=tolerance)
    overlaps = subspace_principal_overlaps(actual_kernel, gauge_basis)
    h0_intersection = int(np.sum(overlaps >= 1.0 - tolerance))
    state_weight: float | None = None
    if state is not None:
        vector = np.asarray(state, dtype=np.complex128).reshape(-1)
        if vector.size != n_vertices:
            raise ValueError("state must have one amplitude per support column.")
        norm = float(np.linalg.norm(vector))
        if norm <= tolerance:
            raise ValueError("state must be nonzero.")
        state_weight = float(np.linalg.norm(gauge_basis.conj().T @ (vector / norm)) ** 2)

    return BoundaryIncidenceCohomologyReport(
        n_support_vertices=n_vertices,
        n_boundary_rows=int(matrix.shape[0]),
        n_active_constraints=int(active_rows.size),
        active_row_weight_histogram=histogram,
        is_two_channel=True,
        equal_magnitude_residual=float(magnitude_residual),
        gauge_flatness_residual=float(flatness_residual),
        incidence_residual=float(incidence_residual),
        connected_component_count=component_count,
        betti_0=component_count,
        betti_1=int(len(edges) - n_vertices + component_count),
        kernel_dimension=kernel_dimension,
        h0_intersection_dimension=h0_intersection,
        state_h0_weight=state_weight,
        interference_gap=interference_gap,
        gauge_basis=gauge_basis,
        edge_endpoints=tuple(edges),
        tolerance=tolerance,
    )


def _smallest_root_of_unity_order(
    value: complex,
    *,
    maximum_order: int,
    tolerance: float,
) -> int | None:
    if maximum_order < 1 or abs(abs(value) - 1.0) > 10.0 * tolerance:
        return None
    for order in range(1, maximum_order + 1):
        if abs(value**order - 1.0) <= 10.0 * tolerance:
            return order
    return None


def diagnose_hard_core_laurent_lift(
    support_configs: npt.ArrayLike,
    amplitudes: npt.ArrayLike,
    boundary: object,
    *,
    raised_value: int = 1,
    maximum_root_order: int = 32,
    tolerance: float = 1e-10,
) -> HardCoreLaurentLiftReport:
    """Detect a uniform cyclotomic transfer rule in a hard-core cage shell."""
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")
    configs = np.asarray(support_configs, dtype=np.int64)
    if configs.ndim != 2 or configs.shape[0] == 0 or configs.shape[1] < 2:
        raise ValueError("support_configs must be a nonempty two-dimensional array.")
    vector = np.asarray(amplitudes, dtype=np.complex128).reshape(-1)
    if vector.size != configs.shape[0] or np.any(np.abs(vector) <= tolerance):
        raise ValueError("amplitudes must contain one nonzero value per support config.")
    matrix = np.asarray(as_dense_array(boundary), dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[1] != configs.shape[0]:
        raise ValueError("boundary columns must match support_configs rows.")

    cohomology = diagnose_boundary_incidence_cohomology(
        matrix,
        vector,
        tolerance=tolerance,
    )
    occupations = configs == int(raised_value)
    particle_counts = np.sum(occupations, axis=1)
    if np.any(particle_counts != particle_counts[0]):
        raise ValueError("support_configs must have fixed raised-site number.")
    length = int(configs.shape[1])
    particle_number = int(particle_counts[0])

    active_rows = np.flatnonzero(np.sum(np.abs(matrix) > tolerance, axis=1) > 0)
    exchange_ratios: list[complex] = []
    all_nearest = bool(cohomology.is_two_channel)
    if cohomology.is_two_channel:
        for row_index in active_rows:
            columns = np.flatnonzero(np.abs(matrix[row_index]) > tolerance)
            first, second = int(columns[0]), int(columns[1])
            first_occ = occupations[first]
            second_occ = occupations[second]
            removed = np.flatnonzero(first_occ & ~second_occ)
            added = np.flatnonzero(second_occ & ~first_occ)
            if removed.size != 1 or added.size != 1:
                all_nearest = False
                continue
            origin, target = int(removed[0]), int(added[0])
            if target == (origin + 1) % length:
                exchange_ratios.append(complex(vector[second] / vector[first]))
            elif origin == (target + 1) % length:
                exchange_ratios.append(complex(vector[first] / vector[second]))
            else:
                all_nearest = False
    transport: complex | None = None
    transport_residual: float | None = None
    primitive_order: int | None = None
    periodic_residual: float | None = None
    factorization_residual: float | None = None
    has_unit_circle_zero = False
    if all_nearest and exchange_ratios:
        transport = complex(np.mean(np.asarray(exchange_ratios, dtype=np.complex128)))
        scale = max(abs(transport), tolerance)
        transport_residual = float(max(abs(value - transport) / scale for value in exchange_ratios))
        primitive_order = _smallest_root_of_unity_order(
            transport,
            maximum_order=maximum_root_order,
            tolerance=tolerance,
        )
        periodic_residual = float(abs(transport**length - 1.0))
        exponents = np.sum(
            occupations * np.arange(length, dtype=np.int64)[None, :],
            axis=1,
        )
        predicted = transport**exponents
        coefficient = complex(np.vdot(predicted, vector) / np.vdot(predicted, predicted))
        factorization_residual = float(
            np.linalg.norm(vector - coefficient * predicted) / np.linalg.norm(vector)
        )
        has_unit_circle_zero = bool(abs(abs(transport) - 1.0) <= 10.0 * tolerance)

    amplitude_by_config = {
        tuple(int(value) for value in config): complex(amplitude)
        for config, amplitude in zip(configs, vector, strict=True)
    }
    translation_ratios: list[complex] = []
    for config, amplitude in zip(configs, vector, strict=True):
        translated = tuple(int(value) for value in np.roll(config, 1))
        translated_amplitude = amplitude_by_config.get(translated)
        if translated_amplitude is None:
            translation_ratios = []
            break
        translation_ratios.append(translated_amplitude / amplitude)
    translation_character: complex | None = None
    translation_residual: float | None = None
    if translation_ratios:
        translation_character = complex(
            np.mean(np.asarray(translation_ratios, dtype=np.complex128))
        )
        scale = max(abs(translation_character), tolerance)
        translation_residual = float(
            max(abs(value - translation_character) / scale for value in translation_ratios)
        )

    return HardCoreLaurentLiftReport(
        length=length,
        particle_number=particle_number,
        support_size=int(configs.shape[0]),
        exchange_constraint_count=len(exchange_ratios),
        all_constraints_are_nearest_neighbor_exchanges=all_nearest,
        uniform_transport_factor=transport,
        transport_residual=transport_residual,
        primitive_root_order=primitive_order,
        periodic_compatibility_residual=periodic_residual,
        amplitude_factorization_residual=factorization_residual,
        one_site_translation_character=translation_character,
        one_site_translation_residual=translation_residual,
        has_unit_circle_symbol_zero=has_unit_circle_zero,
        incidence_cohomology=cohomology,
        tolerance=tolerance,
    )


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
