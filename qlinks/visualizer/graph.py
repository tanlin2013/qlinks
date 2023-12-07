from dataclasses import astuple, dataclass, field
from itertools import product
from typing import Dict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt

from qlinks.lattice.component import Site, UnitVectors
from qlinks.lattice.square_lattice import SquareLattice


@dataclass(slots=True)
class GraphVisualizer:
    lattice: SquareLattice
    _enlarged_lattice: SquareLattice = field(init=False, repr=False)
    _graph: nx.Graph = field(default=None, repr=False)
    _pos: Dict = field(default=None, repr=False)
    _labels: Dict = field(default=None, repr=False)

    def __post_init__(self):
        self._enlarged_lattice = SquareLattice(self.lattice.length_x + 1, self.lattice.length_y + 1)
        self._graph = nx.from_numpy_array(
            self._boundary_flattened_adjacency_matrix(),
            parallel_edges=True,
            create_using=nx.DiGraph,
        )
        self._pos = {idx: astuple(site) for idx, site in enumerate(self._enlarged_lattice)}
        self._labels = {idx: str(astuple(self.lattice[site])) for idx, site in self._pos.items()}

    def _boundary_flattened_adjacency_matrix(self) -> npt.NDArray[np.int64]:
        """
        Flatten the periodic boundaries of the lattice by appending a copy of boundary sites
        to the right and the top, and return the corresponding adjacency matrix.

        Returns:
            The adjacency matrix with lattice boundary flattened.
        """
        adj_mat = np.zeros((self._enlarged_lattice.size, self._enlarged_lattice.size), dtype=int)
        for site, (unit_vector, direction) in product(self.lattice, zip(UnitVectors(), [0, 1])):
            orig_inds = (
                self.lattice.site_index(site) // 2,
                self.lattice.site_index(site + unit_vector) // 2,
            )
            link_val = self.lattice.links[2 * orig_inds[0] + direction]
            enlarged_inds = (
                self._enlarged_lattice.site_index(site) // 2,
                self._enlarged_lattice.site_index(site + unit_vector) // 2,
            )
            adj_mat[enlarged_inds] += link_val  # head_node to tail_node
            adj_mat[enlarged_inds[::-1]] += 1 - link_val  # tail_node to head_node

        for y in range(self.lattice.length_y):  # copy the left boundaries to the right
            left_inds = (
                self._enlarged_lattice.site_index(Site(0, y)) // 2,
                self._enlarged_lattice.site_index(Site(0, y + 1)) // 2,
            )
            right_inds = (
                self._enlarged_lattice.site_index(Site(self.lattice.length_x, y)) // 2,
                self._enlarged_lattice.site_index(Site(self.lattice.length_x, y + 1)) // 2,
            )
            adj_mat[right_inds] = adj_mat[left_inds]
            adj_mat[right_inds[::-1]] = adj_mat[left_inds[::-1]]

        for x in range(self.lattice.length_x):  # copy the bottom boundaries to the top
            bottom_inds = (
                self._enlarged_lattice.site_index(Site(x, 0)) // 2,
                self._enlarged_lattice.site_index(Site(x + 1, 0)) // 2,
            )
            top_inds = (
                self._enlarged_lattice.site_index(Site(x, self.lattice.length_y)) // 2,
                self._enlarged_lattice.site_index(Site(x + 1, self.lattice.length_y)) // 2,
            )
            adj_mat[top_inds] = adj_mat[bottom_inds]
            adj_mat[top_inds[::-1]] = adj_mat[bottom_inds[::-1]]
        return adj_mat

    def _plot_plaquette_variable(self, **kwargs) -> None:
        """Plot the plaquette variables on the lattice.

        Args:
            **kwargs: keyword arguments for :meth:`matplotlib.pyplot.text`.
        """
        fontsize = kwargs.pop("fontsize", 24)
        plaquette_var = {
            "1111": {"s": "◩", "color": "silver"},
            "1011": {"s": "↑", "color": "skyblue"},
            "0111": {"s": "→", "color": "salmon"},
            "0011": {"s": "♰", "color": "silver"},
            "1101": {"s": "↓", "color": "salmon"},
            "1001": {"s": "⬔", "color": "silver"},
            "0101": {"s": "↻", "color": "red"},
            "0001": {"s": "←", "color": "salmon"},
            "1110": {"s": "←", "color": "skyblue"},
            "1010": {"s": "↺", "color": "blue"},
            "0110": {"s": "⬕", "color": "silver"},
            "0010": {"s": "↓", "color": "skyblue"},
            "1100": {"s": "♱", "color": "silver"},
            "1000": {"s": "→", "color": "skyblue"},
            "0100": {"s": "↑", "color": "salmon"},
            "0000": {"s": "◪", "color": "silver"},
        }
        for plaquette in self.lattice.iter_plaquettes():
            center = (plaquette.site[0] + 0.5, plaquette.site[1] + 0.5)
            key = "".join(map(str, self.lattice.links[plaquette.link_index()]))
            plt.text(
                center[0],
                center[1],
                plaquette_var[key]["s"],
                fontsize=fontsize,
                color=plaquette_var[key]["color"],
                ha="center",
                va="center",
            )

    def plot(self, show: bool = True, **kwargs) -> None:
        with_labels = kwargs.pop("with_labels", True)
        node_color = kwargs.pop("node_color", "tab:orange")
        node_size = kwargs.pop("node_size", 1600)
        arrowstyle = kwargs.pop("arrowstyle", "simple")
        arrowsize = kwargs.pop("arrowsize", 30)
        width = kwargs.pop("width", 1)
        connectionstyle = kwargs.pop("connectionstyle", "arc3")
        alpha = kwargs.pop("alpha", 0.5)

        nx.draw(
            self._graph,
            self._pos,
            labels=self._labels,
            with_labels=with_labels,
            node_color=node_color,
            node_size=node_size,
            arrowstyle=arrowstyle,
            arrowsize=arrowsize,
            width=width,
            connectionstyle=connectionstyle,
            alpha=alpha,
        )
        self._plot_plaquette_variable(**kwargs)

        if show:
            plt.show()
