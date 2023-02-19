from dataclasses import astuple
from typing import TypeVar

import networkx as nx

from qlinks.lattice.square_lattice import LatticeState

AnySquareLattice = TypeVar("AnySquareLattice", bound=LatticeState)


class Graph:
    def __init__(self, lattice: AnySquareLattice):
        self._lattice = lattice
        self.graph = lattice.as_graph()
        self.pos = {idx: astuple(site) for idx, site in enumerate(self._lattice)}
        self.labels = {idx: str(site) for idx, site in self.pos.items()}

    def _all_arc_plot(self) -> None:
        nx.draw(
            self.graph,
            self.pos,
            labels=self.labels,
            with_labels=True,
            node_color="tab:orange",
            node_size=2000,
            arrowstyle="simple",
            arrowsize=30,
            width=1,
            connectionstyle="arc3,rad=0.2",
            alpha=0.5,
        )

    def _boundary_arc_plot(self) -> None:
        nx.draw_networkx_nodes(
            self.graph,
            self.pos,
            label=self.labels,
            node_color="tab:orange",
            node_size=2000,
            margins=(0.3, 0.3),
        )
        nx.draw_networkx_labels(self.graph, self.pos, labels=self.labels)

        # nx.draw_networkx_edges(
        #     self.graph, self.pos, edgelist=,
        #     arrowstyle="simple", arrowsize=30, width=1
        # )
        # nx.draw_networkx_edges(
        #     self.graph, self.pos, edgelist=
        # )

    def plot(self, all_arc_arrows: bool = True) -> None:
        if all_arc_arrows:
            self._all_arc_plot()
        else:
            self._boundary_arc_plot()
