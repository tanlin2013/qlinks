from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Hashable

import numpy as np
import scipy.sparse as scipy_sparse
from numpy.typing import NDArray
from scipy.sparse.csgraph import connected_components

from qlinks.caging.candidate import (
    BOUNDARY_OVERLAP_MATRIX_METADATA_KEY,
    INTERNAL_KINETIC_MATRIX_METADATA_KEY,
    CandidateSubgraph,
)


@dataclass(frozen=True)
class VertexSignature:
    """Hashable vertex signature used for candidate partitioning."""

    values: tuple[Hashable, ...]


@dataclass(frozen=True)
class SparseKineticNeighborCache:
    """Reusable sparse-neighborhood view for cage-candidate partitioning.

    Type-1 candidate discovery only needs the *support pattern* of the
    boundary-overlap graph to split a self-loop/bipartition sector into
    connected components. Building the weighted overlap ``K[:, S]^† K[:, S]``
    for every full sector can be noticeably more expensive than the actual
    connectivity query. This cache keeps one CSC copy of the kinetic matrix and
    constructs the unweighted overlap graph by sharing sparse row-neighbor
    buckets. Weighted overlaps are still materialized later, but only for the
    final candidate components needed by the solver.
    """

    csc_matrix: scipy_sparse.csc_matrix

    @classmethod
    def from_matrix(cls, matrix: object) -> "SparseKineticNeighborCache":
        """Build a reusable CSC-neighborhood cache from a sparse matrix."""
        if not scipy_sparse.issparse(matrix):
            raise TypeError("SparseKineticNeighborCache requires a sparse matrix.")
        return cls(csc_matrix=scipy_sparse.csc_matrix(matrix))

    def boundary_overlap_adjacency(
        self,
        candidate_vertices: NDArray[np.int_],
    ) -> scipy_sparse.csr_matrix:
        """Return the unweighted shared-neighbor graph on candidate vertices.

        Vertices ``a`` and ``b`` are adjacent when the two sparse columns
        ``K[:, a]`` and ``K[:, b]`` share at least one nonzero row. This is the
        off-diagonal sparsity pattern of ``K[:, S]^† K[:, S]``.
        """
        vertices = np.asarray(candidate_vertices, dtype=np.int64)
        local_size = int(vertices.size)

        if local_size == 0:
            return scipy_sparse.csr_matrix((0, 0), dtype=np.int8)

        row_to_local_columns: dict[int, list[tuple[int, complex]]] = defaultdict(list)
        indptr = self.csc_matrix.indptr
        indices = self.csc_matrix.indices
        data = self.csc_matrix.data

        for local_column_index, vertex_index in enumerate(vertices):
            start = int(indptr[int(vertex_index)])
            stop = int(indptr[int(vertex_index) + 1])
            for entry_position in range(start, stop):
                row_to_local_columns[int(indices[entry_position])].append(
                    (local_column_index, complex(data[entry_position]))
                )

        overlap_entries: dict[tuple[int, int], complex] = defaultdict(complex)

        for local_columns in row_to_local_columns.values():
            if len(local_columns) < 2:
                continue
            for row_local_index, row_value in local_columns:
                for column_local_index, column_value in local_columns:
                    if row_local_index != column_local_index:
                        overlap_entries[(row_local_index, column_local_index)] += (
                            np.conj(row_value) * column_value
                        )

        nonzero_entries = [
            (row_index, column_index)
            for (row_index, column_index), value in overlap_entries.items()
            if value != 0.0
        ]

        if len(nonzero_entries) == 0:
            return scipy_sparse.csr_matrix((local_size, local_size), dtype=np.int8)

        row_indices, column_indices = zip(*nonzero_entries, strict=True)
        adjacency = scipy_sparse.csr_matrix(
            (
                np.ones(len(nonzero_entries), dtype=np.int8),
                (row_indices, column_indices),
            ),
            shape=(local_size, local_size),
        )
        return adjacency

    def boundary_overlap_matrix(
        self,
        candidate_vertices: NDArray[np.int_],
    ) -> scipy_sparse.csr_matrix:
        """Return the weighted boundary-overlap matrix for final candidates."""
        vertices = np.asarray(candidate_vertices, dtype=np.int64)
        column_block = self.csc_matrix[:, vertices]
        return (column_block.conj().T @ column_block).tocsr()


def _rounded_scalar_signature(
    value: complex,
    *,
    decimals: int,
) -> Hashable:
    """Convert a scalar value into a stable hashable signature."""
    real_part = round(float(np.real(value)), decimals)
    imag_part = round(float(np.imag(value)), decimals)

    if imag_part == 0.0:
        return real_part

    return real_part, imag_part


def _self_loop_signature(
    self_loop_value: np.ndarray | np.complexfloating | complex,
    *,
    decimals: int,
) -> Hashable:
    """Return a hashable signature for scalar or vector self-loop data."""
    value_array = np.asarray(self_loop_value)

    if value_array.ndim == 0:
        return _rounded_scalar_signature(complex(value_array), decimals=decimals)

    return tuple(
        _rounded_scalar_signature(complex(local_value), decimals=decimals)
        for local_value in value_array
    )


def _as_unweighted_adjacency(matrix: object) -> scipy_sparse.csr_matrix:
    """Convert a matrix to an unweighted sparse adjacency matrix."""
    adjacency_matrix = scipy_sparse.csr_matrix(matrix).copy()
    adjacency_matrix.setdiag(0)
    adjacency_matrix.eliminate_zeros()
    adjacency_matrix.data = np.ones_like(adjacency_matrix.data, dtype=np.int8)

    return adjacency_matrix.tocsr()


def _components_from_adjacency(
    adjacency_matrix: object,
    candidate_vertices: NDArray[np.int_],
    *,
    min_component_size: int,
) -> list[CandidateSubgraph]:
    """Split an adjacency matrix into connected candidate components."""
    sparse_adjacency = _as_unweighted_adjacency(adjacency_matrix)

    if sparse_adjacency.shape[0] == 0:
        return []

    component_count, component_labels = connected_components(
        sparse_adjacency,
        directed=False,
        connection="weak",
        return_labels=True,
    )

    candidate_subgraphs: list[CandidateSubgraph] = []

    for component_index in range(component_count):
        local_mask = component_labels == component_index

        if np.count_nonzero(local_mask) < min_component_size:
            continue

        support_indices = candidate_vertices[local_mask]
        candidate_subgraphs.append(CandidateSubgraph(vertices=support_indices))

    return candidate_subgraphs


def _submatrix(
    matrix: object,
    row_indices: NDArray[np.int_],
    column_indices: NDArray[np.int_],
) -> object:
    """Return ``matrix[row_indices, column_indices]`` for dense or sparse matrices."""
    return matrix[row_indices, :][:, column_indices]


def _column_block(
    matrix: object,
    column_indices: NDArray[np.int_],
) -> object:
    """Return ``matrix[:, column_indices]`` for dense or sparse matrices."""
    return matrix[:, column_indices]


def _local_indices_for_vertices(
    group_vertices: NDArray[np.int_],
    candidate_vertices: NDArray[np.int_],
) -> NDArray[np.int64]:
    """Return positions of candidate vertices within a parent vertex group."""
    local_index_by_vertex = {
        int(vertex_index): local_index for local_index, vertex_index in enumerate(group_vertices)
    }

    return np.asarray(
        [local_index_by_vertex[int(vertex_index)] for vertex_index in candidate_vertices],
        dtype=np.int64,
    )


def group_vertices_by_signature(
    *,
    self_loop_values: NDArray[np.complex128],
    bipartition_labels: NDArray[np.int_] | None = None,
    include_bipartition: bool = False,
    decimals: int = 12,
) -> dict[VertexSignature, NDArray[np.int_]]:
    """Group vertices by self-loop values and optionally bipartition labels."""
    self_loop_values = np.asarray(self_loop_values, dtype=np.complex128)

    if include_bipartition and bipartition_labels is None:
        raise ValueError("bipartition_labels are required when include_bipartition=True.")

    signature_to_vertices: dict[VertexSignature, list[int]] = defaultdict(list)

    for vertex_index in range(self_loop_values.shape[0]):
        signature_parts: list[Hashable] = []

        if include_bipartition:
            assert bipartition_labels is not None
            signature_parts.append(int(bipartition_labels[vertex_index]))

        signature_parts.append(
            _self_loop_signature(
                self_loop_values[vertex_index],
                decimals=decimals,
            )
        )

        vertex_signature = VertexSignature(values=tuple(signature_parts))
        signature_to_vertices[vertex_signature].append(vertex_index)

    return {
        vertex_signature: np.asarray(vertex_indices, dtype=np.int64)
        for vertex_signature, vertex_indices in signature_to_vertices.items()
    }


def type1_candidates_from_bipartite_self_loops(
    kinetic_matrix: object,
    self_loop_values: NDArray[np.complex128],
    bipartition_labels: NDArray[np.int_],
    *,
    min_component_size: int = 2,
    decimals: int = 12,
    neighbor_cache: SparseKineticNeighborCache | None = None,
) -> list[CandidateSubgraph]:
    """
    Generate Type-1 candidates from ``(bipartition_label, self_loop_value)``.

    Since same-side vertices of a bipartite kinetic graph have zero direct
    kinetic adjacency, components are defined by the boundary-overlap graph

        K[group, outside] @ K[outside, group].

    This is the generalized version of the old ``incidence_mat @ incidence_mat.T``
    workflow.
    """
    signature_groups = group_vertices_by_signature(
        self_loop_values=self_loop_values,
        bipartition_labels=bipartition_labels,
        include_bipartition=True,
        decimals=decimals,
    )

    if neighbor_cache is None and scipy_sparse.issparse(kinetic_matrix):
        neighbor_cache = SparseKineticNeighborCache.from_matrix(kinetic_matrix)

    candidate_subgraphs: list[CandidateSubgraph] = []

    for vertex_signature, group_vertices in signature_groups.items():
        if group_vertices.size < min_component_size:
            continue

        # Type-1 candidates are grouped by one bipartition side and uniform
        # self-loop value. The relevant connectivity is the support pattern of
        # K[:, S]^† K[:, S]. For sparse matrices, build that unweighted graph
        # from cached column-neighbor buckets and materialize the weighted
        # overlap only for final connected candidates.
        if neighbor_cache is None:
            column_block = _column_block(kinetic_matrix, group_vertices)
            boundary_overlap_matrix = column_block.conj().T @ column_block
            group_candidates = _components_from_adjacency(
                boundary_overlap_matrix,
                group_vertices,
                min_component_size=min_component_size,
            )
        else:
            boundary_overlap_matrix = None
            boundary_overlap_adjacency = neighbor_cache.boundary_overlap_adjacency(
                group_vertices,
            )
            group_candidates = _components_from_adjacency(
                boundary_overlap_adjacency,
                group_vertices,
                min_component_size=min_component_size,
            )

        for group_candidate in group_candidates:
            if boundary_overlap_matrix is None:
                if neighbor_cache is None:
                    raise RuntimeError("neighbor_cache unexpectedly missing.")
                candidate_boundary_overlap = neighbor_cache.boundary_overlap_matrix(
                    group_candidate.vertices,
                )
            else:
                local_indices = _local_indices_for_vertices(
                    group_vertices,
                    group_candidate.vertices,
                )
                candidate_boundary_overlap = _submatrix(
                    boundary_overlap_matrix,
                    local_indices,
                    local_indices,
                )

            candidate_internal_kinetic = np.zeros(
                (group_candidate.size, group_candidate.size),
                dtype=np.complex128,
            )

            candidate_subgraphs.append(
                CandidateSubgraph(
                    vertices=group_candidate.vertices,
                    metadata={
                        "candidate_type": "type1",
                        "signature": vertex_signature.values,
                        INTERNAL_KINETIC_MATRIX_METADATA_KEY: candidate_internal_kinetic,
                        BOUNDARY_OVERLAP_MATRIX_METADATA_KEY: candidate_boundary_overlap,
                    },
                )
            )

    return candidate_subgraphs


def type2_candidates_from_self_loops(
    kinetic_matrix: object,
    self_loop_values: NDArray[np.complex128],
    *,
    min_component_size: int = 2,
    decimals: int = 12,
) -> list[CandidateSubgraph]:
    """
    Generate Type-2 candidates from uniform self-loop sectors.

    Components are defined by the internal kinetic graph ``K[group, group]``.
    """
    signature_groups = group_vertices_by_signature(
        self_loop_values=self_loop_values,
        include_bipartition=False,
        decimals=decimals,
    )

    candidate_subgraphs: list[CandidateSubgraph] = []

    for vertex_signature, group_vertices in signature_groups.items():
        if group_vertices.size < min_component_size:
            continue

        internal_matrix = _submatrix(
            kinetic_matrix,
            group_vertices,
            group_vertices,
        )
        group_candidates = _components_from_adjacency(
            internal_matrix,
            group_vertices,
            min_component_size=min_component_size,
        )

        for group_candidate in group_candidates:
            local_indices = _local_indices_for_vertices(
                group_vertices,
                group_candidate.vertices,
            )
            candidate_internal_kinetic = _submatrix(
                internal_matrix,
                local_indices,
                local_indices,
            )

            candidate_subgraphs.append(
                CandidateSubgraph(
                    vertices=group_candidate.vertices,
                    metadata={
                        "candidate_type": "type2",
                        "signature": vertex_signature.values,
                        INTERNAL_KINETIC_MATRIX_METADATA_KEY: candidate_internal_kinetic,
                    },
                )
            )

    return candidate_subgraphs
