"""Square-QDM-specific compact-cage and transfer diagnostics."""

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

from qlinks.caging.local_search import (
    _qdm_flip_transition_from_action,
    _qdm_global_plaquette_actions,
)
from qlinks.caging.localization import IPRLocalizationConfig, localized_basis_by_many_start_ipr
from qlinks.caging.nullspace import nullspace_svd
from qlinks.caging.periodic_sequence import (
    SquareQDMPeriodicProductInstance,
    SquareQDMPeriodicProductUnitCell,
)
from qlinks.caging.stability_boundary import _orthonormal_basis_absolute
from qlinks.caging.stability_core import (
    _orthonormal_columns,
    estimate_power_law_exponent,
    subspace_complement_basis,
    subspace_principal_overlaps,
)
from qlinks.caging.stability_symmetry import _subspace_symmetry_representation
from qlinks.caging.stability_types import (
    CyclicAmplitudeBondProfile,
    QDMCompactCageReducedWindingPoint,
    QDMCompactCageReducedWindingReport,
    QDMCyclicColumnGrammar,
    QDMCyclicGrammarSupport,
    QDMExplicitProductSupport,
    QDMExplicitSupportBoundaryMap,
    QDMLocalGrammarExtensionPoint,
    QDMLocalGrammarExtensionReport,
    QDMLocalKineticCompatibilityReport,
    QDMLocalPotentialCompatibilityReport,
    QDMPhysicalCancellationScalingPoint,
    RealLocalSignObstructionReport,
    ReducedConstraintFredholmCandidateReport,
    SquareQDMColumnSymbol,
    SquareQDMColumnWord,
    SquareQDMFiniteBondTransferInvariantReport,
    SquareQDMTransferSectorMultiplicity,
)


def diagnose_reduced_constraint_fredholm_candidate(
    constraint_map: object,
    *,
    kernel_basis: object | None = None,
    tolerance: float = 1e-10,
) -> ReducedConstraintFredholmCandidateReport:
    """Quotient an exact kernel and test whether a scalar winding is intrinsic.

    A square, injective reduced map can be the value of a Fredholm symbol once
    a translation-dependent family is supplied.  A strictly tall map has no
    intrinsic determinant phase: choosing a frame for its range can shift any
    apparent square-compression winding.
    """
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")
    matrix = constraint_map
    if scipy_sparse.issparse(matrix):
        matrix_shape = matrix.shape
        dense_for_kernel = None
    else:
        matrix = np.asarray(matrix, dtype=np.complex128)
        if matrix.ndim != 2:
            raise ValueError("constraint_map must be two-dimensional.")
        matrix_shape = matrix.shape
        dense_for_kernel = matrix
    codomain_dimension, domain_dimension = map(int, matrix_shape)

    if kernel_basis is None:
        if dense_for_kernel is None:
            dense_for_kernel = np.asarray(matrix.toarray(), dtype=np.complex128)
        kernel = nullspace_svd(dense_for_kernel, tolerance=tolerance)
    else:
        kernel = np.asarray(kernel_basis, dtype=np.complex128)
        if kernel.ndim == 1:
            kernel = kernel[:, np.newaxis]
        if kernel.ndim != 2 or kernel.shape[0] != domain_dimension:
            raise ValueError("kernel_basis has incompatible shape.")
        kernel = _orthonormal_columns(kernel)
        residual = np.linalg.norm(matrix @ kernel)
        scale = max(1.0, float(np.linalg.norm(kernel)))
        if residual > tolerance * scale:
            raise ValueError("kernel_basis is not annihilated by constraint_map.")

    kernel_dimension = int(kernel.shape[1])
    if kernel_dimension > domain_dimension:
        raise ValueError("kernel dimension cannot exceed the domain dimension.")
    if kernel_dimension == domain_dimension:
        complement = np.zeros((domain_dimension, 0), dtype=np.complex128)
    else:
        complement = scipy_linalg.null_space(
            kernel.conj().T,
            rcond=tolerance,
        ).astype(np.complex128, copy=False)
    reduced = np.asarray(matrix @ complement, dtype=np.complex128)
    singular_values = scipy_linalg.svdvals(reduced) if reduced.size else np.zeros(0)
    rank = int(np.sum(singular_values > tolerance))
    reduced_domain_dimension = int(complement.shape[1])
    injective = rank == reduced_domain_dimension
    positive = singular_values[singular_values > tolerance]
    gap = None if positive.size == 0 else float(np.min(positive))
    square_candidate = codomain_dimension == reduced_domain_dimension
    log_abs_det: float | None = None
    phase: float | None = None
    if injective and reduced_domain_dimension > 0:
        log_abs_det = float(np.sum(np.log(positive)))
        # The polar-frame compression is the positive matrix
        # sqrt(reduced^dagger reduced), whose determinant phase is zero.
        phase = 0.0
    if not injective:
        classification = "reduced_map_not_injective"
    elif square_candidate:
        classification = "square_fredholm_symbol_candidate"
    elif codomain_dimension > reduced_domain_dimension:
        classification = "rectangular_stiefel_no_intrinsic_winding"
    else:
        classification = "underdetermined_reduced_map"

    return ReducedConstraintFredholmCandidateReport(
        codomain_dimension=codomain_dimension,
        domain_dimension=domain_dimension,
        kernel_dimension=kernel_dimension,
        reduced_domain_dimension=reduced_domain_dimension,
        reduced_rank=rank,
        codomain_excess=int(codomain_dimension - reduced_domain_dimension),
        reduced_singular_values=np.asarray(singular_values, dtype=np.float64),
        reduced_gap=gap,
        canonical_log_abs_determinant=log_abs_det,
        canonical_determinant_phase=phase,
        is_reduced_injective=injective,
        is_square_symbol_candidate=square_candidate,
        classification=classification,
        tolerance=tolerance,
    )


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


def diagnose_square_qdm_compact_cage_reduced_winding(
    unit_cell: SquareQDMPeriodicProductUnitCell,
    repeat_counts: Sequence[int] | npt.NDArray[np.integer],
    *,
    max_support_size: int = 1024,
    tolerance: float = 1e-10,
) -> QDMCompactCageReducedWindingReport:
    """Audit scalar Fredholm winding for the compact QDM cage sequence.

    Two reductions are kept separate.  In state space the exact product cage is
    quotiented from the physical support-to-shell map.  This map remains
    strictly rectangular and therefore has no intrinsic determinant winding.
    In coupling space the cage-compatible directions are quotiented from the
    independently varying plaquette couplings.  The remaining two channels per
    repeated cell form a constant positive symbol and hence carry winding zero.
    """
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")
    counts = tuple(int(value) for value in np.asarray(repeat_counts).reshape(-1))
    if not counts or any(value <= 0 for value in counts):
        raise ValueError("repeat_counts must contain positive integers.")
    if len(set(counts)) != len(counts):
        raise ValueError("repeat_counts must not contain duplicates.")

    points: list[QDMCompactCageReducedWindingPoint] = []
    reference_pair_offsets: tuple[tuple[int, int], ...] | None = None
    reference_symbol: npt.NDArray[np.complex128] | None = None
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
        state_complement = diagnose_reduced_constraint_fredholm_candidate(
            boundary_map.boundary,
            kernel_basis=support.amplitudes[:, np.newaxis],
            tolerance=tolerance,
        )
        kinetic = diagnose_qdm_local_kinetic_compatibility(
            instance.model,
            support.configs,
            support.amplitudes,
            boundary_map=boundary_map,
            tolerance=tolerance,
        )
        if len(kinetic.plaquette_ids) % repeats:
            raise ValueError("plaquette count is not divisible by repeat count.")
        terms_per_cell = len(kinetic.plaquette_ids) // repeats
        column_by_id = {
            int(plaquette_id): index for index, plaquette_id in enumerate(kinetic.plaquette_ids)
        }
        pair_data: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
        for left_id, right_id in kinetic.equal_coupling_pairs:
            left_cell, left_offset = divmod(int(left_id), terms_per_cell)
            right_cell, right_offset = divmod(int(right_id), terms_per_cell)
            if left_cell != right_cell:
                raise ValueError("a kinetic compatibility pair crosses unit cells.")
            pair_data.append(
                (
                    left_cell,
                    (left_offset, right_offset),
                    (column_by_id[int(left_id)], column_by_id[int(right_id)]),
                )
            )
        pair_data.sort(key=lambda item: (item[0], item[1]))
        if len(pair_data) != kinetic.rank:
            raise ValueError("equal-coupling pairs do not span the kinetic obstruction quotient.")
        local_offsets = tuple(offset for cell, offset, _columns in pair_data if cell == 0)
        if not local_offsets:
            raise ValueError("no local kinetic obstruction channels were found.")
        if any(
            tuple(offset for cell, offset, _columns in pair_data if cell == cell_index)
            != local_offsets
            for cell_index in range(repeats)
        ):
            raise ValueError("kinetic obstruction pattern is not translation repeated.")
        if reference_pair_offsets is None:
            reference_pair_offsets = local_offsets
        elif local_offsets != reference_pair_offsets:
            raise ValueError("local kinetic obstruction pattern changes with size.")

        quotient_basis = np.zeros(
            (len(kinetic.plaquette_ids), len(pair_data)),
            dtype=np.complex128,
        )
        for quotient_index, (_cell, _offset, (left_column, right_column)) in enumerate(pair_data):
            quotient_basis[left_column, quotient_index] = 1.0 / np.sqrt(2.0)
            quotient_basis[right_column, quotient_index] = -1.0 / np.sqrt(2.0)
        effective = kinetic.obstruction_matrix @ quotient_basis
        singular_values = scipy_linalg.svdvals(effective)
        effective_rank = int(np.sum(singular_values > tolerance))
        if effective_rank != len(pair_data):
            raise ValueError("kinetic quotient map is not injective.")
        positive = singular_values[singular_values > tolerance]
        kinetic_gap = float(np.min(positive))
        gram = np.asarray(effective.conj().T @ effective, dtype=np.complex128)
        channels_per_cell = len(local_offsets)
        first_block = gram[:channels_per_cell, :channels_per_cell]
        unit_cell_symbol = scipy_linalg.sqrtm(first_block).astype(
            np.complex128,
            copy=False,
        )
        if reference_symbol is None:
            reference_symbol = unit_cell_symbol
        symbol_scale = max(1.0, float(np.linalg.norm(reference_symbol)))
        cell_block_residuals: list[float] = []
        off_cell_blocks: list[float] = []
        for left_cell in range(repeats):
            left_slice = slice(
                left_cell * channels_per_cell,
                (left_cell + 1) * channels_per_cell,
            )
            for right_cell in range(repeats):
                right_slice = slice(
                    right_cell * channels_per_cell,
                    (right_cell + 1) * channels_per_cell,
                )
                block = gram[left_slice, right_slice]
                if left_cell == right_cell:
                    block_symbol = scipy_linalg.sqrtm(block).astype(
                        np.complex128,
                        copy=False,
                    )
                    cell_block_residuals.append(
                        float(np.linalg.norm(block_symbol - reference_symbol))
                    )
                else:
                    off_cell_blocks.append(float(np.linalg.norm(block)))
        unit_cell_residual = max(cell_block_residuals, default=0.0) / symbol_scale
        intercell_norm = max(off_cell_blocks, default=0.0)

        points.append(
            QDMCompactCageReducedWindingPoint(
                repeats=repeats,
                system_size=(int(instance.model.lx), int(instance.model.ly)),
                support_size=support.support_size,
                shell_size=boundary_map.shell_size,
                state_complement=state_complement,
                kinetic_term_count=len(kinetic.plaquette_ids),
                kinetic_compatible_dimension=kinetic.compatible_dimension,
                kinetic_quotient_dimension=len(pair_data),
                kinetic_quotient_singular_values=np.asarray(
                    singular_values,
                    dtype=np.float64,
                ),
                kinetic_quotient_gap=kinetic_gap,
                local_pair_offsets=local_offsets,
                intercell_gram_norm=intercell_norm,
                unit_cell_gram_residual=unit_cell_residual,
            )
        )

    assert reference_pair_offsets is not None and reference_symbol is not None
    symbol_singular_values = scipy_linalg.svdvals(reference_symbol)
    symbol_gap = float(np.min(symbol_singular_values))
    symbol_det = complex(np.linalg.det(reference_symbol))
    if abs(symbol_det) <= tolerance:
        classification = "reduced_coupling_symbol_gapless"
    elif any(point.intercell_gram_norm > 10.0 * tolerance for point in points):
        classification = "intercell_reduced_coupling_requires_matrix_symbol"
    elif any(point.unit_cell_gram_residual > 10.0 * tolerance for point in points):
        classification = "reduced_coupling_symbol_not_size_stable"
    elif any(point.state_complement.admits_intrinsic_scalar_winding for point in points):
        classification = "state_square_symbol_requires_twist_test"
    else:
        classification = "local_constant_symbol_trivial_winding"

    return QDMCompactCageReducedWindingReport(
        repeat_axis=str(unit_cell.repeat_axis),
        unit_cell_size=(int(unit_cell.model.lx), int(unit_cell.model.ly)),
        local_pair_offsets=reference_pair_offsets,
        reduced_coupling_symbol=reference_symbol,
        reduced_coupling_winding=0,
        reduced_coupling_gap=symbol_gap,
        points=tuple(points),
        classification=classification,
        tolerance=tolerance,
    )


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
