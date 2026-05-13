from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import scipy.sparse as scipy_sparse
from numpy.typing import NDArray

from qlinks.caging.candidate import CandidateSubgraph
from qlinks.caging.nullspace import as_dense_array, nullspace_svd


@dataclass(frozen=True)
class CandidateFilterContext:
    """
    Matrix-level context used by candidate prefilters.

    Parameters
    ----------
    hamiltonian:
        Full Hamiltonian matrix. This is optional for filters that only need
        kinetic_matrix or self_loop_values.

    kinetic_matrix:
        Off-diagonal or kinetic matrix. For ``H = K + V``, this should be ``K``.

    self_loop_values:
        Diagonal/self-loop data. This can be a one-dimensional array of scalar
        values, or a two-dimensional array of coefficient vectors. The latter is
        useful when ``V({lambda_i})`` depends on several parameters.

        Shape examples:
            ``(hilbert_size,)``
            ``(hilbert_size, n_parameters)``

    bipartition_labels:
        Integer labels for bipartite subsets. Usually entries are 0 and 1.
    """

    hamiltonian: object | None = None
    kinetic_matrix: object | None = None
    self_loop_values: NDArray[np.complex128] | None = None
    bipartition_labels: NDArray[np.int_] | None = None

    @property
    def hilbert_size(self) -> int:
        """Return the Hilbert-space size inferred from available data."""
        if self.hamiltonian is not None:
            return int(self.hamiltonian.shape[0])

        if self.kinetic_matrix is not None:
            return int(self.kinetic_matrix.shape[0])

        if self.self_loop_values is not None:
            return int(self.self_loop_values.shape[0])

        if self.bipartition_labels is not None:
            return int(self.bipartition_labels.shape[0])

        raise ValueError("Cannot infer hilbert_size from an empty context.")


@dataclass(frozen=True)
class CandidateFilterResult:
    """Result returned by a candidate prefilter."""

    accepted: bool
    reason: str = ""
    metadata: dict[str, object] | None = None


class CandidateFilter(Protocol):
    """Protocol for candidate prefilters."""

    def __call__(
        self,
        context: CandidateFilterContext,
        candidate: CandidateSubgraph,
    ) -> CandidateFilterResult:
        """Return whether the candidate passes this filter."""


def _matrix_shape(matrix: object) -> tuple[int, int]:
    return int(matrix.shape[0]), int(matrix.shape[1])


def _validate_square_matrix(matrix: object, *, name: str) -> None:
    row_count, column_count = _matrix_shape(matrix)

    if row_count != column_count:
        raise ValueError(f"{name} must be square.")


def _validate_support_indices(
    hilbert_size: int,
    support_indices: NDArray[np.int_],
) -> NDArray[np.int_]:
    support_indices = np.asarray(support_indices, dtype=np.int64)

    if support_indices.ndim != 1:
        raise ValueError("support_indices must be a 1D array.")

    if np.any(support_indices < 0) or np.any(support_indices >= hilbert_size):
        raise ValueError("support_indices contains out-of-range indices.")

    return support_indices


def extract_subblocks(
    matrix: object,
    support_indices: NDArray[np.int_],
) -> tuple[object, object, NDArray[np.int_]]:
    """
    Extract internal and boundary blocks for a candidate support.

    Returns
    -------
    internal_matrix:
        ``matrix[support, support]``.

    boundary_matrix:
        ``matrix[outside, support]``.

    outside_indices:
        Indices outside the support.
    """
    _validate_square_matrix(matrix, name="matrix")

    hilbert_size = int(matrix.shape[0])
    support_indices = _validate_support_indices(hilbert_size, support_indices)

    outside_mask = np.ones(hilbert_size, dtype=bool)
    outside_mask[support_indices] = False
    outside_indices = np.nonzero(outside_mask)[0]

    internal_matrix = matrix[support_indices, :][:, support_indices]
    boundary_matrix = matrix[outside_indices, :][:, support_indices]

    return internal_matrix, boundary_matrix, outside_indices


def diagonal_values(
    matrix: object,
    support_indices: NDArray[np.int_],
) -> NDArray[np.complex128]:
    """Return diagonal values of ``matrix`` on ``support_indices``."""
    if scipy_sparse.issparse(matrix):
        full_diagonal = matrix.diagonal()
    else:
        full_diagonal = np.diag(matrix)

    return np.asarray(full_diagonal[support_indices], dtype=np.complex128)


def has_uniform_values(
    values: NDArray[np.complex128],
    *,
    tolerance: float = 1e-10,
) -> bool:
    """
    Check whether all selected values are uniform.

    This supports both scalar values and coefficient vectors.

    Examples
    --------
    Scalar self-loop values:

        values.shape == (support_size,)

    Multi-parameter self-loop coefficients:

        values.shape == (support_size, n_parameters)
    """
    selected_values = np.asarray(values, dtype=np.complex128)

    if selected_values.shape[0] == 0:
        return False

    reference_value = selected_values[0]

    return bool(
        np.allclose(
            selected_values,
            reference_value,
            atol=tolerance,
            rtol=0.0,
        )
    )


def has_uniform_diagonal(
    matrix: object,
    support_indices: NDArray[np.int_],
    *,
    tolerance: float = 1e-10,
) -> bool:
    """Check whether a candidate support has uniform diagonal values."""
    local_diagonal = diagonal_values(matrix, support_indices)
    return has_uniform_values(local_diagonal, tolerance=tolerance)


def boundary_nullity(
    boundary_matrix: object,
    *,
    tolerance: float = 1e-10,
) -> int:
    """Return the nullity of the boundary-leakage matrix."""
    nullspace_basis = nullspace_svd(boundary_matrix, tolerance=tolerance)
    return int(nullspace_basis.shape[1])


def matrix_nullity(
    matrix: object,
    *,
    tolerance: float = 1e-10,
) -> int:
    """Return the nullity of a matrix."""
    nullspace_basis = nullspace_svd(matrix, tolerance=tolerance)
    return int(nullspace_basis.shape[1])


def _identity_matrix(
    matrix_size: int,
    *,
    sparse: bool,
) -> object:
    if sparse:
        return scipy_sparse.identity(
            matrix_size,
            dtype=np.complex128,
            format="csr",
        )

    return np.eye(matrix_size, dtype=np.complex128)


def _matrix_norm(matrix: object) -> float:
    if scipy_sparse.issparse(matrix):
        return float(scipy_sparse.linalg.norm(matrix))

    return float(np.linalg.norm(as_dense_array(matrix)))


def _vertical_stack(matrix_blocks: list[object]) -> object:
    if any(scipy_sparse.issparse(matrix_block) for matrix_block in matrix_blocks):
        return scipy_sparse.vstack(matrix_blocks, format="csr")

    dense_blocks = [as_dense_array(matrix_block) for matrix_block in matrix_blocks]
    return np.vstack(dense_blocks)


@dataclass(frozen=True)
class SupportSizeFilter:
    """Filter candidates by support size."""

    min_size: int = 2
    max_size: int | None = None

    def __call__(
        self,
        context: CandidateFilterContext,
        candidate: CandidateSubgraph,
    ) -> CandidateFilterResult:
        del context

        if candidate.size < self.min_size:
            return CandidateFilterResult(
                accepted=False,
                reason=f"support size {candidate.size} is smaller than {self.min_size}",
            )

        if self.max_size is not None and candidate.size > self.max_size:
            return CandidateFilterResult(
                accepted=False,
                reason=f"support size {candidate.size} is larger than {self.max_size}",
            )

        return CandidateFilterResult(
            accepted=True,
            reason="support size accepted",
            metadata={"support_size": candidate.size},
        )


@dataclass(frozen=True)
class UniformSelfLoopFilter:
    """
    Require uniform self-loop values on the candidate support.

    If ``context.self_loop_values`` is provided, this filter uses it directly.
    Otherwise, if ``use_hamiltonian_diagonal`` is true, it falls back to the
    diagonal of ``context.hamiltonian``.
    """

    tolerance: float = 1e-10
    use_hamiltonian_diagonal: bool = True

    def __call__(
        self,
        context: CandidateFilterContext,
        candidate: CandidateSubgraph,
    ) -> CandidateFilterResult:
        if context.self_loop_values is not None:
            all_self_loop_values = np.asarray(
                context.self_loop_values,
                dtype=np.complex128,
            )
            selected_values = all_self_loop_values[candidate.vertices]
        elif self.use_hamiltonian_diagonal and context.hamiltonian is not None:
            selected_values = diagonal_values(context.hamiltonian, candidate.vertices)
        else:
            return CandidateFilterResult(
                accepted=False,
                reason="no self-loop values or Hamiltonian diagonal available",
            )

        if not has_uniform_values(selected_values, tolerance=self.tolerance):
            return CandidateFilterResult(
                accepted=False,
                reason="self-loop values are not uniform on the support",
                metadata={"selected_values": selected_values},
            )

        return CandidateFilterResult(
            accepted=True,
            reason="self-loop values are uniform on the support",
            metadata={"self_loop_value": selected_values[0]},
        )


@dataclass(frozen=True)
class BoundaryNullityFilter:
    """
    Require the leakage matrix from support to outside to have nonzero nullity.

    By default this uses ``context.hamiltonian``. For ``H = K + V``, if the
    potential is diagonal, using ``context.kinetic_matrix`` is equivalent for
    the boundary block and is usually conceptually cleaner.
    """

    min_nullity: int = 1
    tolerance: float = 1e-10
    matrix_name: str = "kinetic"

    def __call__(
        self,
        context: CandidateFilterContext,
        candidate: CandidateSubgraph,
    ) -> CandidateFilterResult:
        if self.matrix_name == "kinetic":
            matrix = context.kinetic_matrix
        elif self.matrix_name == "hamiltonian":
            matrix = context.hamiltonian
        else:
            return CandidateFilterResult(
                accepted=False,
                reason=f"unknown matrix_name: {self.matrix_name}",
            )

        if matrix is None:
            return CandidateFilterResult(
                accepted=False,
                reason=f"{self.matrix_name} matrix is unavailable",
            )

        _internal_matrix, boundary_matrix, _outside_indices = extract_subblocks(
            matrix,
            candidate.vertices,
        )
        nullity = boundary_nullity(boundary_matrix, tolerance=self.tolerance)

        if nullity < self.min_nullity:
            return CandidateFilterResult(
                accepted=False,
                reason=(
                    f"boundary nullity {nullity} is smaller than "
                    f"{self.min_nullity}"
                ),
                metadata={"boundary_nullity": nullity},
            )

        return CandidateFilterResult(
            accepted=True,
            reason="boundary nullity accepted",
            metadata={"boundary_nullity": nullity},
        )


@dataclass(frozen=True)
class SameBipartitionSideFilter:
    """Require the whole support to lie on one bipartite subset."""

    def __call__(
        self,
        context: CandidateFilterContext,
        candidate: CandidateSubgraph,
    ) -> CandidateFilterResult:
        if context.bipartition_labels is None:
            return CandidateFilterResult(
                accepted=False,
                reason="bipartition labels are unavailable",
            )

        bipartition_labels = np.asarray(context.bipartition_labels, dtype=np.int64)
        selected_labels = bipartition_labels[candidate.vertices]
        unique_labels = np.unique(selected_labels)

        if unique_labels.size != 1:
            return CandidateFilterResult(
                accepted=False,
                reason="support is not contained in one bipartite subset",
                metadata={"selected_bipartition_labels": selected_labels},
            )

        return CandidateFilterResult(
            accepted=True,
            reason="support is contained in one bipartite subset",
            metadata={"bipartition_label": int(unique_labels[0])},
        )


@dataclass(frozen=True)
class ZeroInternalKineticFilter:
    """
    Require the internal kinetic block to be zero.

    This is useful for Type-1 cages on one side of a bipartite kinetic graph.
    If the graph is exactly bipartite and the support lies on one side, this
    should hold automatically, but this filter certifies it numerically.
    """

    tolerance: float = 1e-10

    def __call__(
        self,
        context: CandidateFilterContext,
        candidate: CandidateSubgraph,
    ) -> CandidateFilterResult:
        if context.kinetic_matrix is None:
            return CandidateFilterResult(
                accepted=False,
                reason="kinetic matrix is unavailable",
            )

        internal_kinetic_matrix, _boundary_matrix, _outside_indices = extract_subblocks(
            context.kinetic_matrix,
            candidate.vertices,
        )

        internal_norm = _matrix_norm(internal_kinetic_matrix)

        if internal_norm > self.tolerance:
            return CandidateFilterResult(
                accepted=False,
                reason="internal kinetic block is not zero",
                metadata={"internal_kinetic_norm": internal_norm},
            )

        return CandidateFilterResult(
            accepted=True,
            reason="internal kinetic block is zero",
            metadata={"internal_kinetic_norm": internal_norm},
        )


@dataclass(frozen=True)
class KineticTargetNullityFilter:
    """
    Require ``K_S - kappa I`` to have nonzero nullity for at least one kappa.

    This is a Type-2 prefilter. It does not include the boundary cancellation
    condition. It only checks whether the internal kinetic block can support
    the requested kinetic eigenvalue.
    """

    target_kappas: tuple[complex, ...]
    min_nullity: int = 1
    tolerance: float = 1e-10
    require_nonzero_kappa: bool = False

    def __call__(
        self,
        context: CandidateFilterContext,
        candidate: CandidateSubgraph,
    ) -> CandidateFilterResult:
        if context.kinetic_matrix is None:
            return CandidateFilterResult(
                accepted=False,
                reason="kinetic matrix is unavailable",
            )

        internal_kinetic_matrix, _boundary_matrix, _outside_indices = extract_subblocks(
            context.kinetic_matrix,
            candidate.vertices,
        )

        support_size = candidate.size
        identity_matrix = _identity_matrix(
            support_size,
            sparse=scipy_sparse.issparse(internal_kinetic_matrix),
        )

        accepted_kappas: list[complex] = []
        nullities_by_kappa: dict[complex, int] = {}

        for target_kappa in self.target_kappas:
            if self.require_nonzero_kappa and abs(target_kappa) <= self.tolerance:
                continue

            shifted_matrix = internal_kinetic_matrix - target_kappa * identity_matrix
            nullity = matrix_nullity(shifted_matrix, tolerance=self.tolerance)
            nullities_by_kappa[complex(target_kappa)] = nullity

            if nullity >= self.min_nullity:
                accepted_kappas.append(complex(target_kappa))

        if len(accepted_kappas) == 0:
            return CandidateFilterResult(
                accepted=False,
                reason="no target kappa has enough internal kinetic nullity",
                metadata={"nullities_by_kappa": nullities_by_kappa},
            )

        return CandidateFilterResult(
            accepted=True,
            reason="at least one target kappa has internal kinetic nullity",
            metadata={
                "accepted_kappas": accepted_kappas,
                "nullities_by_kappa": nullities_by_kappa,
            },
        )


@dataclass(frozen=True)
class CombinedBoundaryKineticTargetNullityFilter:
    """
    Require nonzero nullity of the combined matrix

        [K_out,S]
        [K_S - kappa I]

    for at least one target kappa.

    This is stronger than ``KineticTargetNullityFilter`` and is almost the
    fixed-kappa caging test. It is still useful as a prefilter before building
    full states or doing validation.
    """

    target_kappas: tuple[complex, ...]
    min_nullity: int = 1
    tolerance: float = 1e-10
    require_nonzero_kappa: bool = False

    def __call__(
        self,
        context: CandidateFilterContext,
        candidate: CandidateSubgraph,
    ) -> CandidateFilterResult:
        if context.kinetic_matrix is None:
            return CandidateFilterResult(
                accepted=False,
                reason="kinetic matrix is unavailable",
            )

        internal_kinetic_matrix, boundary_matrix, _outside_indices = extract_subblocks(
            context.kinetic_matrix,
            candidate.vertices,
        )

        support_size = candidate.size
        identity_matrix = _identity_matrix(
            support_size,
            sparse=scipy_sparse.issparse(internal_kinetic_matrix),
        )

        accepted_kappas: list[complex] = []
        combined_nullities_by_kappa: dict[complex, int] = {}

        for target_kappa in self.target_kappas:
            if self.require_nonzero_kappa and abs(target_kappa) <= self.tolerance:
                continue

            shifted_matrix = internal_kinetic_matrix - target_kappa * identity_matrix
            combined_matrix = _vertical_stack([boundary_matrix, shifted_matrix])

            combined_nullity = matrix_nullity(
                combined_matrix,
                tolerance=self.tolerance,
            )
            combined_nullities_by_kappa[complex(target_kappa)] = combined_nullity

            if combined_nullity >= self.min_nullity:
                accepted_kappas.append(complex(target_kappa))

        if len(accepted_kappas) == 0:
            return CandidateFilterResult(
                accepted=False,
                reason="no target kappa passes the combined boundary-kinetic test",
                metadata={"combined_nullities_by_kappa": combined_nullities_by_kappa},
            )

        return CandidateFilterResult(
            accepted=True,
            reason="at least one target kappa passes the combined test",
            metadata={
                "accepted_kappas": accepted_kappas,
                "combined_nullities_by_kappa": combined_nullities_by_kappa,
            },
        )


def run_candidate_filters(
    context: CandidateFilterContext,
    candidate: CandidateSubgraph,
    candidate_filters: list[CandidateFilter],
) -> CandidateFilterResult:
    """Run candidate filters sequentially and stop at the first rejection."""
    collected_metadata: dict[str, object] = {}

    for filter_index, candidate_filter in enumerate(candidate_filters):
        filter_result = candidate_filter(context, candidate)

        collected_metadata[f"filter_{filter_index}"] = {
            "filter_type": type(candidate_filter).__name__,
            "accepted": filter_result.accepted,
            "reason": filter_result.reason,
            "metadata": filter_result.metadata,
        }

        if not filter_result.accepted:
            return CandidateFilterResult(
                accepted=False,
                reason=filter_result.reason,
                metadata=collected_metadata,
            )

    return CandidateFilterResult(
        accepted=True,
        reason="all filters accepted the candidate",
        metadata=collected_metadata,
    )


def filter_candidates(
    context: CandidateFilterContext,
    candidates: list[CandidateSubgraph],
    candidate_filters: list[CandidateFilter],
) -> list[CandidateSubgraph]:
    """Return candidates accepted by all filters."""
    accepted_candidates: list[CandidateSubgraph] = []

    for candidate in candidates:
        filter_result = run_candidate_filters(
            context,
            candidate,
            candidate_filters,
        )

        if filter_result.accepted:
            accepted_candidates.append(candidate)

    return accepted_candidates


def make_type1_bipartite_prefilters(
    *,
    tolerance: float = 1e-10,
    min_boundary_nullity: int = 1,
    min_size: int = 2,
    max_size: int | None = None,
) -> list[CandidateFilter]:
    """
    Prefilters for Type-1 caged states.

    Intended physics:
        - kinetic graph is bipartite;
        - support lies on one bipartite subset;
        - internal kinetic block is zero;
        - self-loop values are uniform;
        - kappa = 0.
    """
    return [
        SupportSizeFilter(min_size=min_size, max_size=max_size),
        UniformSelfLoopFilter(tolerance=tolerance),
        SameBipartitionSideFilter(),
        ZeroInternalKineticFilter(tolerance=tolerance),
        BoundaryNullityFilter(
            min_nullity=min_boundary_nullity,
            tolerance=tolerance,
            matrix_name="kinetic",
        ),
    ]


def make_type2_integer_kappa_prefilters(
    *,
    target_kappas: tuple[complex, ...] = (2.0, -2.0),
    tolerance: float = 1e-10,
    min_boundary_nullity: int = 1,
    min_kinetic_nullity: int = 1,
    min_size: int = 2,
    max_size: int | None = None,
    use_combined_boundary_kinetic_test: bool = False,
) -> list[CandidateFilter]:
    """
    Prefilters for Type-2 caged states.

    Intended physics:
        - self-loop values are uniform;
        - kappa is a nonzero integer such as +2 or -2;
        - boundary cancellation is possible.

    If ``use_combined_boundary_kinetic_test`` is true, the final filter checks

        ker([K_out,S; K_S - kappa I])

    directly for each target kappa.
    """
    candidate_filters: list[CandidateFilter] = [
        SupportSizeFilter(min_size=min_size, max_size=max_size),
        UniformSelfLoopFilter(tolerance=tolerance),
        BoundaryNullityFilter(
            min_nullity=min_boundary_nullity,
            tolerance=tolerance,
            matrix_name="kinetic",
        ),
    ]

    if use_combined_boundary_kinetic_test:
        candidate_filters.append(
            CombinedBoundaryKineticTargetNullityFilter(
                target_kappas=target_kappas,
                min_nullity=min_kinetic_nullity,
                tolerance=tolerance,
                require_nonzero_kappa=True,
            )
        )
    else:
        candidate_filters.append(
            KineticTargetNullityFilter(
                target_kappas=target_kappas,
                min_nullity=min_kinetic_nullity,
                tolerance=tolerance,
                require_nonzero_kappa=True,
            )
        )

    return candidate_filters
