from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Hashable

import numpy as np
import scipy.sparse as scipy_sparse
from numpy.typing import NDArray
from scipy.sparse.csgraph import connected_components

from qlinks.caging.candidate import CandidateSubgraph
from qlinks.caging.prefilters import extract_subblocks


@dataclass(frozen=True)
class VertexSignature:
    """Hashable vertex signature used for candidate partitioning."""

    values: tuple[Hashable, ...]


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

    candidate_subgraphs: list[CandidateSubgraph] = []

    for vertex_signature, group_vertices in signature_groups.items():
        if group_vertices.size < min_component_size:
            continue

        _internal_matrix, boundary_matrix, _outside_indices = extract_subblocks(
            kinetic_matrix,
            group_vertices,
        )
        boundary_overlap_matrix = boundary_matrix.conj().T @ boundary_matrix
        group_candidates = _components_from_adjacency(
            boundary_overlap_matrix,
            group_vertices,
            min_component_size=min_component_size,
        )

        for group_candidate in group_candidates:
            candidate_subgraphs.append(
                CandidateSubgraph(
                    vertices=group_candidate.vertices,
                    metadata={
                        "candidate_type": "type1",
                        "signature": vertex_signature.values,
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

        internal_matrix, _boundary_matrix, _outside_indices = extract_subblocks(
            kinetic_matrix,
            group_vertices,
        )
        group_candidates = _components_from_adjacency(
            internal_matrix,
            group_vertices,
            min_component_size=min_component_size,
        )

        for group_candidate in group_candidates:
            candidate_subgraphs.append(
                CandidateSubgraph(
                    vertices=group_candidate.vertices,
                    metadata={
                        "candidate_type": "type2",
                        "signature": vertex_signature.values,
                    },
                )
            )

    return candidate_subgraphs
