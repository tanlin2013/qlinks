from dataclasses import astuple
from typing import TypeVar

import matplotlib.pyplot as plt
import networkx as nx

from qlinks.lattice.square_lattice import LatticeState

AnySquareLattice = TypeVar("AnySquareLattice", bound=LatticeState)


class Quiver:
    def __init__(self, lattice: AnySquareLattice):
        self._lattice = lattice

    def plot(self) -> None:
        pos_x, pos_y, arrow_x, arrow_y = [], [], [], []
        vertices = [self._lattice.get_vertex_links(site) for site in self._lattice]
        for vertex in vertices:
            for link in vertex:
                if link.flux > 0:
                    pos_x.append(link.site[0])
                    pos_y.append(link.site[1])
                    arrow_x.append(link.unit_vector[0])
                    arrow_y.append(link.unit_vector[1])
                else:
                    pos_x.append(self._lattice[link.site + link.unit_vector][0])
                    pos_y.append(self._lattice[link.site + link.unit_vector][1])
                    arrow_x.append(-1 * link.unit_vector[0])
                    arrow_y.append(-1 * link.unit_vector[1])

        plt.figure(figsize=(6, 4))
        plt.quiver(pos_x, pos_y, arrow_x, arrow_y, scale=1.5, scale_units="xy")
        plt.margins(0.4)

        pos = {astuple(site): astuple(site) for site in self._lattice}
        labels = {idx: str(site) for idx, site in pos.items()}
        graph = nx.grid_2d_graph(self._lattice.length, self._lattice.width)
        nx.draw(graph, pos, labels=labels, node_size=100)
