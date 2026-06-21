from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from qlinks.visualizer.hamiltonian_graph import (
    EdgeColorRule,
    GraphBackend,
    HamiltonianGraphData,
    HamiltonianGraphStyle,
    HamiltonianGraphVisualizer,
    LayoutName,
    NodeColorRule,
)

VectorizationConvention = Literal["column_major", "row_major"]


def flatten_density_matrix(
    density_matrix: npt.ArrayLike,
    *,
    convention: VectorizationConvention = "column_major",
) -> npt.NDArray[np.complex128]:
    """Flatten a density matrix using the requested vectorization convention."""
    rho = np.asarray(density_matrix, dtype=np.complex128)

    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("density_matrix must be a square 2D array.")

    if convention == "column_major":
        return rho.reshape(-1, order="F").astype(np.complex128, copy=False)

    if convention == "row_major":
        return rho.reshape(-1, order="C").astype(np.complex128, copy=False)

    raise ValueError(f"Unsupported vectorization convention: {convention!r}")


def unflatten_operator_index(
    index: int,
    *,
    hilbert_dim: int,
    convention: VectorizationConvention = "column_major",
) -> tuple[int, int]:
    """Map a vectorized Liouville-space index to an operator basis pair.

    Args:
        index: Flat Liouville-space index.
        hilbert_dim: Hilbert-space dimension.
        convention: Vectorization convention used to flatten the operator.

    Returns:
        Pair ``(ket_index, bra_index)`` corresponding to ``|ket><bra|``.
    """
    if index < 0 or index >= hilbert_dim * hilbert_dim:
        raise ValueError("operator-space index is out of range.")

    if convention == "column_major":
        return index % hilbert_dim, index // hilbert_dim

    if convention == "row_major":
        return index // hilbert_dim, index % hilbert_dim

    raise ValueError(f"Unsupported vectorization convention: {convention!r}")


def operator_space_labels(
    *,
    hilbert_dim: int,
    convention: VectorizationConvention = "column_major",
    indices: npt.ArrayLike | None = None,
) -> list[str]:
    """Return labels ``|i><j|`` for Liouville-space nodes."""
    if indices is None:
        index_array = np.arange(hilbert_dim * hilbert_dim, dtype=np.int64)
    else:
        index_array = np.asarray(indices, dtype=np.int64)

    labels: list[str] = []

    for index in index_array:
        ket_index, bra_index = unflatten_operator_index(
            int(index),
            hilbert_dim=hilbert_dim,
            convention=convention,
        )
        labels.append(f"|{ket_index}><{bra_index}|")

    return labels


@dataclass(frozen=True)
class LiouvillianGraphVisualizer:
    """Directed graph visualizer for Liouvillian superoperators.

    Nodes are operator-space basis elements ``|i><j|``. Directed edges are
    nonzero off-diagonal Liouvillian matrix elements.
    """

    graph_visualizer: HamiltonianGraphVisualizer
    hilbert_dim: int
    vectorization: VectorizationConvention = "column_major"

    @classmethod
    def from_liouvillian(
        cls,
        liouvillian,
        *,
        hilbert_dim: int,
        density_matrix: npt.ArrayLike | None = None,
        vectorization: VectorizationConvention = "column_major",
        include_self_loops: bool = False,
        weight_tolerance: float = 0.0,
        style: HamiltonianGraphStyle | None = None,
    ) -> LiouvillianGraphVisualizer:
        """Construct a Liouvillian graph visualizer."""
        n_operator_space = hilbert_dim * hilbert_dim

        graph_visualizer = HamiltonianGraphVisualizer.from_sparse_matrix(
            liouvillian,
            include_self_loops=include_self_loops,
            weight_tolerance=weight_tolerance,
            directed=True,
            style=style,
        )

        if graph_visualizer.graph_data.n_vertices != n_operator_space:
            raise ValueError("liouvillian dimension must equal hilbert_dim ** 2.")

        state_vector = None

        if density_matrix is not None:
            state_vector = flatten_density_matrix(
                density_matrix,
                convention=vectorization,
            )

            if len(state_vector) != n_operator_space:
                raise ValueError("flattened density matrix must have length hilbert_dim ** 2.")

        labels = operator_space_labels(
            hilbert_dim=hilbert_dim,
            convention=vectorization,
        )

        old_data = graph_visualizer.graph_data

        graph_visualizer = HamiltonianGraphVisualizer(
            graph_data=HamiltonianGraphData(
                adjacency=old_data.adjacency,
                self_loop_values=old_data.self_loop_values,
                original_indices=old_data.original_indices,
                state_vector=state_vector,
                vertex_labels=labels,
                directed=True,
            ),
            style=graph_visualizer.style,
        )

        return cls(
            graph_visualizer=graph_visualizer,
            hilbert_dim=hilbert_dim,
            vectorization=vectorization,
        )

    @property
    def graph_data(self) -> HamiltonianGraphData:
        """Return the underlying graph data."""
        return self.graph_visualizer.graph_data

    def plot(
        self,
        *,
        backend: GraphBackend = "networkx",
        color_by: NodeColorRule = "state_amplitude_abs",
        edge_color_by: EdgeColorRule = "weight_complex",
        layout: LayoutName = "auto",
        **kwargs,
    ):
        """Draw the Liouvillian graph."""
        return self.graph_visualizer.plot(
            backend=backend,
            color_by=color_by,
            edge_color_by=edge_color_by,
            layout=layout,
            **kwargs,
        )

    def to_networkx(self):
        """Convert to a directed NetworkX graph."""
        return self.graph_visualizer.to_networkx()

    def to_igraph(self):
        """Convert to a directed igraph graph."""
        return self.graph_visualizer.to_igraph()

    def save_graph(self, *args, **kwargs) -> None:
        """Save the graph through the underlying visualizer."""
        self.graph_visualizer.save_graph(*args, **kwargs)
