from __future__ import annotations

from pathlib import Path

import igraph as ig
import numpy as np

from qlinks.models import SquareQDMModel


def sparse_hamiltonian_to_igraph(H, *, directed: bool = False) -> ig.Graph:
    """
    Convert a sparse Hamiltonian matrix into an igraph graph.

    Vertices:
        basis-state indices

    Edges:
        nonzero off-diagonal Hamiltonian matrix elements H[i, j]

    Edge attributes:
        weight = H[i, j]
        abs_weight = abs(H[i, j])
    """
    H = H.tocoo()

    n = H.shape[0]
    edges: list[tuple[int, int]] = []
    weights: list[complex] = []
    abs_weights: list[float] = []

    seen: set[tuple[int, int]] = set()

    for row, col, value in zip(H.row, H.col, H.data, strict=True):
        row = int(row)
        col = int(col)

        # Skip diagonal potential terms.
        if row == col:
            continue

        if directed:
            edge = (col, row)  # H[row, col] maps |col> -> |row>
        else:
            edge = tuple(sorted((row, col)))
            if edge in seen:
                continue
            seen.add(edge)

        edges.append(edge)
        weights.append(complex(value))
        abs_weights.append(float(abs(value)))

    graph = ig.Graph(n=n, edges=edges, directed=directed)
    graph.vs["basis_index"] = list(range(n))
    graph.es["weight"] = weights
    graph.es["abs_weight"] = abs_weights

    return graph


def main() -> None:
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    model = SquareQDMModel(
        lx=4,
        ly=4,
        boundary_condition="periodic",
        kinetic=-1.0,
        potential=0.7,
        required_count=1,
        winding_x=0,
        winding_y=0,
        winding_convention="electric",
    )

    print("Building QDM basis and Hamiltonian...")
    result = model.build(
        basis_solver="dfs",
        builder="sparse",
        sort_basis=True,
    )

    H = result.hamiltonian
    basis = result.basis

    print(f"Basis size: {basis.n_states}")
    print(f"H shape: {H.shape}")
    print(f"H nnz: {H.nnz}")

    graph = sparse_hamiltonian_to_igraph(H, directed=False)

    print(f"Graph vertices: {graph.vcount()}")
    print(f"Graph edges: {graph.ecount()}")

    # Layout choices:
    #   "fr" = Fruchterman-Reingold force layout
    #   "kk" = Kamada-Kawai
    #   "circle" = circular layout
    layout = graph.layout("fr")

    visual_style = {
        "layout": layout,
        "vertex_size": 8,
        "vertex_label": None,
        "vertex_color": "orange",
        "edge_width": 1.0,
        "edge_color": "gray",
        "bbox": (1000, 1000),
        "margin": 40,
    }

    out_png = out_dir / "qdm_4x4_pbc_hamiltonian_graph.png"
    ig.plot(graph, str(out_png), **visual_style)

    print(f"Saved graph plot to: {out_png}")

    # Optional: save graph in GraphML for inspection in Gephi/Cytoscape.
    # out_graphml = out_dir / "qdm_4x4_pbc_hamiltonian_graph.graphml"
    # graph.write_graphml(str(out_graphml))
    # print(f"Saved graphml to: {out_graphml}")


if __name__ == "__main__":
    main()
