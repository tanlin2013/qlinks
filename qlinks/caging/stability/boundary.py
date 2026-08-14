"""Boundary-cancellation matroid and periodic-scaling diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import numpy.typing as npt
import scipy.linalg as scipy_linalg

from qlinks.caging.nullspace import as_dense_array, nullspace_svd
from qlinks.caging.stability.core import (
    estimate_power_law_exponent,
    partition_cage_hamiltonian,
    subspace_principal_overlaps,
)
from qlinks.caging.stability.topology import _boundary_edge_labels
from qlinks.caging.stability.types import (
    BoundaryCancellationCircuitEntry,
    BoundaryCancellationMatroidBranchPoint,
    BoundaryCancellationMatroidBranchReport,
    BoundaryCancellationMatroidReport,
    BoundaryCancellationMomentumPoint,
    BoundaryCancellationScalingPoint,
)


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
