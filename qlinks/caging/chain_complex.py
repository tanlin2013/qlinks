from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import numpy.typing as npt
import scipy.linalg as scipy_linalg
import scipy.sparse as scipy_sparse

from qlinks.caging.nullspace import as_dense_array, nullspace_svd

ComplexArray = npt.NDArray[np.complex128]
FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class HamiltonianGraphChainComplex:
    """Finite Hamiltonian-graph caging complex ``C2 -> C1 -> C0``.

    ``constraint_map`` is the physical leakage/eigenvalue map ``D_E`` from
    support amplitudes to violated Hamiltonian rows. ``generator_map`` maps a
    chosen library of bounded-support cage motifs into the support-amplitude
    space. The chain condition is ``constraint_map @ generator_map == 0``.

    The basis conventions are:

    * columns of ``generator_map`` are local cage generators;
    * columns of ``ker(constraint_map)`` are all exact cage amplitudes on the
      chosen support shell;
    * ``H_1 = ker(D_E) / im(T_R)`` is the many-body CLS-completeness defect;
    * ``H_2 = ker(T_R)`` records linear relations among translated motifs.
    """

    constraint_map: ComplexArray
    generator_map: ComplexArray
    support_indices: tuple[int, ...] | None = None
    test_indices: tuple[int, ...] | None = None
    generator_labels: tuple[str, ...] = ()

    @property
    def c0_dimension(self) -> int:
        return int(self.constraint_map.shape[0])

    @property
    def c1_dimension(self) -> int:
        return int(self.constraint_map.shape[1])

    @property
    def c2_dimension(self) -> int:
        return int(self.generator_map.shape[1])

    @property
    def chain_residual(self) -> float:
        return float(np.linalg.norm(self.constraint_map @ self.generator_map))


@dataclass(frozen=True, slots=True)
class HamiltonianGraphHomologyReport:
    """Numerical homology/cohomology report for a caging chain complex."""

    c0_dimension: int
    c1_dimension: int
    c2_dimension: int
    constraint_rank: int
    generator_rank: int
    cage_dimension: int
    h1_dimension: int
    h2_dimension: int
    chain_residual: float
    relative_chain_residual: float
    generator_containment_residual: float
    cage_basis: ComplexArray
    local_generator_basis: ComplexArray
    h1_basis: ComplexArray
    h2_basis: ComplexArray
    cocycle_basis: ComplexArray
    hodge_operator: ComplexArray
    hodge_eigenvalues: FloatArray
    hodge_gap: float | None
    tolerance: float

    @property
    def nu_mb(self) -> int:
        """Return the finite-volume many-body CLS-completeness defect."""
        return self.h1_dimension

    @property
    def is_chain_complex(self) -> bool:
        return self.relative_chain_residual <= self.tolerance

    @property
    def is_locally_complete(self) -> bool:
        return self.h1_dimension == 0

    def pairing_matrix(self, cage_representatives: npt.ArrayLike | None = None) -> ComplexArray:
        """Pair dual cocycles with supplied cage representatives.

        With no argument, harmonic ``H_1`` representatives are used. Under the
        Euclidean inner product the returned matrix should be the identity up
        to numerical tolerance.
        """
        representatives = (
            self.h1_basis
            if cage_representatives is None
            else _as_column_matrix(cage_representatives, self.c1_dimension)
        )
        return np.asarray(self.cocycle_basis.conj().T @ representatives, dtype=np.complex128)

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "c0_dimension": self.c0_dimension,
            "c1_dimension": self.c1_dimension,
            "c2_dimension": self.c2_dimension,
            "constraint_rank": self.constraint_rank,
            "generator_rank": self.generator_rank,
            "cage_dimension": self.cage_dimension,
            "h1_dimension": self.h1_dimension,
            "nu_mb": self.nu_mb,
            "h2_dimension": self.h2_dimension,
            "chain_residual": self.chain_residual,
            "relative_chain_residual": self.relative_chain_residual,
            "generator_containment_residual": self.generator_containment_residual,
            "hodge_gap": self.hodge_gap,
            "is_chain_complex": self.is_chain_complex,
            "is_locally_complete": self.is_locally_complete,
            "tolerance": self.tolerance,
        }


@dataclass(frozen=True, slots=True)
class TermResolvedCagingReport:
    """Compare separately vanishing local channels with collective cancellation."""

    physical_constraint_map: ComplexArray
    resolved_constraint_map: ComplexArray
    physical_kernel_basis: ComplexArray
    resolved_kernel_basis: ComplexArray
    collective_quotient_basis: ComplexArray
    physical_nullity: int
    resolved_nullity: int
    collective_quotient_dimension: int
    resolved_containment_residual: float
    tolerance: float

    @property
    def has_collective_cancellation(self) -> bool:
        return self.collective_quotient_dimension > 0

    def channel_activity(self, states: npt.ArrayLike) -> FloatArray:
        """Return ``||\\widetilde D_E psi||`` for each supplied state column."""
        vectors = _as_column_matrix(states, self.physical_constraint_map.shape[1])
        values = np.linalg.norm(self.resolved_constraint_map @ vectors, axis=0)
        return np.asarray(values, dtype=np.float64)

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "physical_nullity": self.physical_nullity,
            "resolved_nullity": self.resolved_nullity,
            "collective_quotient_dimension": self.collective_quotient_dimension,
            "resolved_containment_residual": self.resolved_containment_residual,
            "has_collective_cancellation": self.has_collective_cancellation,
            "tolerance": self.tolerance,
        }


@dataclass(frozen=True, slots=True)
class MotifRadiusHomologyPoint:
    """One motif-radius point in a local-generator saturation scan."""

    radius: int
    generator_rank: int
    h1_dimension: int
    h2_dimension: int
    chain_residual: float
    hodge_gap: float | None

    @classmethod
    def from_report(
        cls,
        radius: int,
        report: HamiltonianGraphHomologyReport,
    ) -> MotifRadiusHomologyPoint:
        return cls(
            radius=int(radius),
            generator_rank=report.generator_rank,
            h1_dimension=report.h1_dimension,
            h2_dimension=report.h2_dimension,
            chain_residual=report.chain_residual,
            hodge_gap=report.hodge_gap,
        )


@dataclass(frozen=True, slots=True)
class MotifRadiusSaturationReport:
    """Track whether ``nu_MB`` stabilizes as the motif library grows."""

    points: tuple[MotifRadiusHomologyPoint, ...]
    plateau_length: int = 2
    tolerance: float = 1.0e-10

    @property
    def classification(self) -> str:
        if not self.points:
            return "empty"
        if self.tolerance <= 0.0:
            return "invalid_tolerance"
        if any(point.chain_residual > self.tolerance for point in self.points):
            return "invalid_chain_data"
        required = max(1, int(self.plateau_length))
        if len(self.points) < required:
            return "insufficient_radius_range"
        tail = self.points[-required:]
        h1_values = {point.h1_dimension for point in tail}
        generator_ranks = {point.generator_rank for point in tail}
        if len(h1_values) != 1 or len(generator_ranks) != 1:
            return "not_saturated"
        value = tail[-1].h1_dimension
        return "locally_complete" if value == 0 else "saturated_defect_candidate"

    @property
    def saturated_nu_mb(self) -> int | None:
        if self.classification not in {"locally_complete", "saturated_defect_candidate"}:
            return None
        return self.points[-1].h1_dimension

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "classification": self.classification,
            "saturated_nu_mb": self.saturated_nu_mb,
            "plateau_length": self.plateau_length,
            "tolerance": self.tolerance,
            "points": tuple(
                {
                    "radius": point.radius,
                    "generator_rank": point.generator_rank,
                    "h1_dimension": point.h1_dimension,
                    "h2_dimension": point.h2_dimension,
                    "chain_residual": point.chain_residual,
                    "hodge_gap": point.hodge_gap,
                }
                for point in self.points
            ),
        }


@dataclass(frozen=True, slots=True)
class LaurentPeriodicKernelPoint:
    """Kernel diagnostic for one Laurent operator on a twisted finite ring."""

    length: int
    twist: float
    rank: int
    nullity: int
    singular_values: FloatArray
    smallest_positive_singular_value: float | None

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "length": self.length,
            "twist": self.twist,
            "rank": self.rank,
            "nullity": self.nullity,
            "smallest_positive_singular_value": self.smallest_positive_singular_value,
        }


def build_hamiltonian_graph_chain_complex(
    hamiltonian: object,
    support_indices: Sequence[int],
    local_generators: npt.ArrayLike,
    *,
    energy: complex = 0.0,
    test_indices: Sequence[int] | None = None,
    generators_are_full_hilbert_vectors: bool = False,
    generator_labels: Sequence[str] = (),
) -> HamiltonianGraphChainComplex:
    """Build ``D_E`` and ``T_R`` from a Hamiltonian and a support shell.

    ``D_E`` is the selected-row block of ``(H - E I) P_support``. Local
    generators may be supplied either in support coordinates or as full
    Hilbert-space vectors.
    """
    shape = getattr(hamiltonian, "shape", None)
    if shape is None or len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("hamiltonian must be a square matrix.")
    hilbert_dimension = int(shape[0])

    support = _validate_indices(support_indices, hilbert_dimension, "support_indices")
    tests = (
        np.arange(hilbert_dimension, dtype=np.int64)
        if test_indices is None
        else _validate_indices(test_indices, hilbert_dimension, "test_indices")
    )
    constraint_map = _hamiltonian_constraint_block(
        hamiltonian,
        support,
        tests,
        energy=energy,
    )

    raw_generators = np.asarray(local_generators, dtype=np.complex128)
    if raw_generators.ndim == 1:
        raw_generators = raw_generators[:, None]
    if raw_generators.ndim != 2:
        raise ValueError("local_generators must be one- or two-dimensional.")
    if generators_are_full_hilbert_vectors:
        if raw_generators.shape[0] != hilbert_dimension:
            raise ValueError("full-Hilbert generators have incompatible dimension.")
        generator_map = raw_generators[support, :]
        outside = np.delete(raw_generators, support, axis=0)
        if np.linalg.norm(outside) > 1.0e-10:
            raise ValueError("full-Hilbert generators must be supported on support_indices.")
    else:
        generator_map = _as_column_matrix(raw_generators, support.size)

    labels = tuple(str(label) for label in generator_labels)
    if labels and len(labels) != generator_map.shape[1]:
        raise ValueError("generator_labels must match the number of generator columns.")

    return HamiltonianGraphChainComplex(
        constraint_map=np.asarray(constraint_map, dtype=np.complex128),
        generator_map=np.asarray(generator_map, dtype=np.complex128),
        support_indices=tuple(int(value) for value in support),
        test_indices=tuple(int(value) for value in tests),
        generator_labels=labels,
    )


def diagnose_hamiltonian_graph_homology(
    complex_: HamiltonianGraphChainComplex,
    *,
    tolerance: float = 1.0e-10,
    require_chain_condition: bool = True,
) -> HamiltonianGraphHomologyReport:
    """Compute finite-volume homology, dual cocycles, and the Hodge gap."""
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")

    d1 = _as_matrix(complex_.constraint_map)
    d2 = _as_matrix(complex_.generator_map)
    if d1.shape[1] != d2.shape[0]:
        raise ValueError("constraint_map and generator_map have incompatible dimensions.")

    chain_residual = float(np.linalg.norm(d1 @ d2))
    scale = max(1.0, float(np.linalg.norm(d1) * np.linalg.norm(d2)))
    normalized_chain_residual = chain_residual / scale
    if require_chain_condition and normalized_chain_residual > tolerance:
        raise ValueError(
            "constraint_map @ generator_map is nonzero; the supplied maps do not form "
            "a chain complex."
        )

    cage_basis = nullspace_svd(d1, tolerance=tolerance)
    local_basis = _orthonormal_column_space(d2, tolerance=tolerance)
    h2_basis = nullspace_svd(d2, tolerance=tolerance)

    cage_projector = cage_basis @ cage_basis.conj().T
    containment_residual = float(np.linalg.norm((np.eye(d1.shape[1]) - cage_projector) @ d2))
    h1_raw = (np.eye(d1.shape[1]) - local_basis @ local_basis.conj().T) @ cage_basis
    h1_basis = _orthonormal_column_space(h1_raw, tolerance=tolerance)

    # Use the orthogonal projector onto im(T_R), rather than ``T_R T_R^†``,
    # so the harmonic gap is invariant under changes of generator basis and
    # nonzero rescalings of individual motif columns.
    local_projector = local_basis @ local_basis.conj().T
    hodge = d1.conj().T @ d1 + local_projector
    hodge = np.asarray(0.5 * (hodge + hodge.conj().T), dtype=np.complex128)
    eigenvalues, eigenvectors = scipy_linalg.eigh(hodge)
    eigenvalues = np.asarray(np.real_if_close(eigenvalues), dtype=np.float64)
    zero_mask = np.abs(eigenvalues) <= tolerance
    harmonic_basis = np.asarray(eigenvectors[:, zero_mask], dtype=np.complex128)
    # The harmonic basis is simultaneously a canonical H_1 representative and
    # a dual H^1 cocycle basis under the Euclidean inner product.
    if harmonic_basis.shape[1] == h1_basis.shape[1]:
        h1_basis = harmonic_basis
    positive = eigenvalues[eigenvalues > tolerance]
    hodge_gap = float(positive[0]) if positive.size else None

    constraint_rank = _matrix_rank(d1, tolerance)
    generator_rank = _matrix_rank(d2, tolerance)
    cage_dimension = int(cage_basis.shape[1])
    h1_dimension = int(h1_basis.shape[1])
    h2_dimension = int(h2_basis.shape[1])

    expected_h1 = cage_dimension - generator_rank
    if normalized_chain_residual <= tolerance and h1_dimension != expected_h1:
        raise RuntimeError("inconsistent H_1 dimension; check the numerical tolerance.")

    return HamiltonianGraphHomologyReport(
        c0_dimension=int(d1.shape[0]),
        c1_dimension=int(d1.shape[1]),
        c2_dimension=int(d2.shape[1]),
        constraint_rank=constraint_rank,
        generator_rank=generator_rank,
        cage_dimension=cage_dimension,
        h1_dimension=h1_dimension,
        h2_dimension=h2_dimension,
        chain_residual=chain_residual,
        relative_chain_residual=normalized_chain_residual,
        generator_containment_residual=containment_residual,
        cage_basis=cage_basis,
        local_generator_basis=local_basis,
        h1_basis=h1_basis,
        h2_basis=h2_basis,
        cocycle_basis=h1_basis.copy(),
        hodge_operator=hodge,
        hodge_eigenvalues=eigenvalues,
        hodge_gap=hodge_gap,
        tolerance=tolerance,
    )


def diagnose_term_resolved_caging(
    local_constraint_maps: Sequence[object],
    *,
    coefficients: Sequence[complex] | None = None,
    tolerance: float = 1.0e-10,
) -> TermResolvedCagingReport:
    """Resolve robust channelwise zeros from collectively cancelled cages.

    Every local map must have the same row and column dimensions. The physical
    differential is their coefficient-weighted sum, while the term-resolved
    differential is the vertical stack of the individual weighted maps.
    """
    if not local_constraint_maps:
        raise ValueError("at least one local constraint map is required.")
    maps = tuple(_as_matrix(value) for value in local_constraint_maps)
    shape = maps[0].shape
    if any(value.shape != shape for value in maps):
        raise ValueError("all local constraint maps must have the same shape.")
    weights = (
        np.ones(len(maps), dtype=np.complex128)
        if coefficients is None
        else np.asarray(coefficients, dtype=np.complex128).reshape(-1)
    )
    if weights.size != len(maps):
        raise ValueError("coefficients must match local_constraint_maps.")

    weighted = tuple(weight * value for weight, value in zip(weights, maps, strict=True))
    physical = np.sum(np.stack(weighted, axis=0), axis=0)
    resolved = np.vstack(weighted)
    physical_kernel = nullspace_svd(physical, tolerance=tolerance)
    resolved_kernel = nullspace_svd(resolved, tolerance=tolerance)

    physical_projector = physical_kernel @ physical_kernel.conj().T
    containment = float(np.linalg.norm((np.eye(shape[1]) - physical_projector) @ resolved_kernel))
    collective_raw = (
        np.eye(shape[1]) - resolved_kernel @ resolved_kernel.conj().T
    ) @ physical_kernel
    collective_basis = _orthonormal_column_space(collective_raw, tolerance=tolerance)

    return TermResolvedCagingReport(
        physical_constraint_map=np.asarray(physical, dtype=np.complex128),
        resolved_constraint_map=np.asarray(resolved, dtype=np.complex128),
        physical_kernel_basis=physical_kernel,
        resolved_kernel_basis=resolved_kernel,
        collective_quotient_basis=collective_basis,
        physical_nullity=int(physical_kernel.shape[1]),
        resolved_nullity=int(resolved_kernel.shape[1]),
        collective_quotient_dimension=int(collective_basis.shape[1]),
        resolved_containment_residual=containment,
        tolerance=tolerance,
    )


def twisted_translation_matrix(length: int, twist: float = 0.0) -> ComplexArray:
    """Return the unitary one-site translation with ``T**L = exp(i twist)``."""
    if length <= 0:
        raise ValueError("length must be positive.")
    translation = np.zeros((length, length), dtype=np.complex128)
    for source in range(length - 1):
        translation[source + 1, source] = 1.0
    translation[0, length - 1] = np.exp(1.0j * float(twist))
    return translation


def periodic_laurent_operator(
    coefficients: Mapping[int, complex | npt.ArrayLike],
    length: int,
    *,
    twist: float = 0.0,
) -> ComplexArray:
    """Evaluate a finite Laurent-polynomial matrix at twisted translation.

    A scalar dictionary such as ``{0: 1, 1: 1}`` constructs ``I + T``. Matrix
    coefficients construct a block Laurent operator via Kronecker products.
    """
    if not coefficients:
        raise ValueError("coefficients must not be empty.")
    blocks: dict[int, ComplexArray] = {}
    block_shape: tuple[int, int] | None = None
    for shift, value in coefficients.items():
        array = np.asarray(value, dtype=np.complex128)
        if array.ndim == 0:
            array = array.reshape(1, 1)
        if array.ndim != 2:
            raise ValueError("Laurent coefficients must be scalars or matrices.")
        if block_shape is None:
            block_shape = array.shape
        if array.shape != block_shape:
            raise ValueError("all Laurent coefficient matrices must have the same shape.")
        blocks[int(shift)] = array
    assert block_shape is not None

    translation = twisted_translation_matrix(length, twist)
    result = np.zeros(
        (length * block_shape[0], length * block_shape[1]),
        dtype=np.complex128,
    )
    for shift, block in blocks.items():
        if shift >= 0:
            translated = np.linalg.matrix_power(translation, shift)
        else:
            translated = np.linalg.matrix_power(translation.conj().T, -shift)
        result += np.kron(translated, block)
    return result


def diagnose_periodic_laurent_kernel(
    coefficients: Mapping[int, complex | npt.ArrayLike],
    length: int,
    *,
    twist: float = 0.0,
    tolerance: float = 1.0e-10,
) -> LaurentPeriodicKernelPoint:
    """Compute the finite-ring nullity and smallest nonzero singular value."""
    operator = periodic_laurent_operator(coefficients, length, twist=twist)
    singular_values = np.asarray(scipy_linalg.svdvals(operator), dtype=np.float64)
    rank = int(np.sum(singular_values > tolerance))
    positive = np.sort(singular_values[singular_values > tolerance])
    gap = float(positive[0]) if positive.size else None
    return LaurentPeriodicKernelPoint(
        length=int(length),
        twist=float(twist),
        rank=rank,
        nullity=int(operator.shape[1] - rank),
        singular_values=singular_values,
        smallest_positive_singular_value=gap,
    )


def _hamiltonian_constraint_block(
    hamiltonian: object,
    support: npt.NDArray[np.int64],
    tests: npt.NDArray[np.int64],
    *,
    energy: complex,
) -> ComplexArray:
    """Return ``P_test (H-E) P_support`` without densifying the full matrix."""
    if scipy_sparse.issparse(hamiltonian):
        block = hamiltonian[tests, :][:, support].toarray()
    else:
        dense = np.asarray(hamiltonian, dtype=np.complex128)
        if dense.ndim != 2:
            raise ValueError("hamiltonian must be two-dimensional.")
        block = dense[np.ix_(tests, support)].copy()

    block = np.asarray(block, dtype=np.complex128)
    test_position = {int(index): row for row, index in enumerate(tests)}
    for column, support_index in enumerate(support):
        row = test_position.get(int(support_index))
        if row is not None:
            block[row, column] -= complex(energy)
    return block


def _as_matrix(matrix: object) -> ComplexArray:
    array = as_dense_array(matrix)
    if array.ndim != 2:
        raise ValueError("matrix must be two-dimensional.")
    return np.asarray(array, dtype=np.complex128)


def _as_column_matrix(vectors: npt.ArrayLike, row_count: int) -> ComplexArray:
    array = np.asarray(vectors, dtype=np.complex128)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[0] != row_count:
        raise ValueError("vectors have incompatible row dimension.")
    return array


def _orthonormal_column_space(matrix: object, *, tolerance: float) -> ComplexArray:
    array = _as_matrix(matrix)
    if array.shape[1] == 0:
        return np.zeros((array.shape[0], 0), dtype=np.complex128)
    return np.asarray(scipy_linalg.orth(array, rcond=tolerance), dtype=np.complex128)


def _matrix_rank(matrix: ComplexArray, tolerance: float) -> int:
    return int(np.sum(scipy_linalg.svdvals(matrix) > tolerance))


def _validate_indices(indices: Sequence[int], upper: int, name: str) -> npt.NDArray[np.int64]:
    values = np.asarray(tuple(indices), dtype=np.int64).reshape(-1)
    if values.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if np.any(values < 0) or np.any(values >= upper) or np.unique(values).size != values.size:
        raise ValueError(f"{name} must contain unique indices in range({upper}).")
    return values
