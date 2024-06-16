from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List

import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd
import pynauty
import scipy.sparse as sp
from scipy.linalg import null_space
from scipy.sparse.csgraph import connected_components
from sympy.combinatorics import Permutation, PermutationGroup

from qlinks import logger


@dataclass(slots=True)
class Automorphism:
    adj_mat: npt.NDArray | sp.sparray
    _graph: nx.Graph = field(init=False, repr=False)
    _df: pd.DataFrame = field(init=False, repr=False)

    def __post_init__(self):
        self._graph = nx.from_numpy_array(self.adj_mat)
        self._df = pd.DataFrame(
            {
                "degree": list(self.degree_series.values()),
                "bipartition": list(self.bipartition_series.values()),
            }
        )

    @staticmethod
    def group_indices_by_value(dictionary: Dict) -> Dict:
        index_groups = defaultdict(list)
        for index, value in dictionary.items():
            index_groups[value].append(index)
        return dict(index_groups)

    def characteristic_matrix(self, partition, normalized: bool = True) -> npt.NDArray:
        char_mat = np.zeros((self.n_nodes, len(partition)), dtype=int)
        for j, block in enumerate(partition):
            for i in block:
                char_mat[i][j] = 1
        if normalized:
            char_mat = char_mat @ np.sqrt(np.diagflat([1 / len(b) for b in partition]))
        return char_mat

    def quotient_matrix(self, partition) -> npt.NDArray:
        s = self.characteristic_matrix(partition, normalized=True)
        quotient = s.T @ self.adj_mat @ s
        if not np.allclose(self.adj_mat @ s, s @ quotient, atol=1e-12):
            logger.warn("The partition is not equitable.")
        return quotient

    @property
    def n_nodes(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def degree_series(self) -> Dict:
        return dict(self._graph.degree)

    @property
    def degree_partition(self):
        return self.group_indices_by_value(self.degree_series)

    @property
    def bipartition_series(self) -> Dict:
        top_nodes, bottom_nodes = nx.bipartite.sets(self._graph)
        bipartition_dict = {node: "A" for node in top_nodes}
        bipartition_dict.update({node: "B" for node in bottom_nodes})
        return dict(sorted(bipartition_dict.items()))

    @property
    def bipartition(self) -> Dict:
        top_nodes, bottom_nodes = nx.bipartite.sets(self._graph)
        return {"A": list(top_nodes), "B": list(bottom_nodes)}

    @property
    def joint_partition(self) -> Dict:
        return self._df.groupby(["degree", "bipartition"]).groups

    def joint_partition_indices(self):
        return [list(block) for block in self.joint_partition.values()]

    def automorphism_group(self, partition=None) -> PermutationGroup:
        ntg = pynauty.Graph(
            self.n_nodes,
            adjacency_dict=nx.to_dict_of_lists(self._graph),
            vertex_coloring=partition,
        )
        return PermutationGroup([Permutation(perm) for perm in pynauty.autgrp(ntg)[0]])

    @staticmethod
    def connected_null_space(mat, fill_zeros: bool = False) -> List[npt.NDArray]:
        n_components, labels = connected_components(
            mat, directed=False, connection="weak", return_labels=True
        )
        null_spaces = []
        for i in range(n_components):
            mask = (labels == i)  # fmt: skip
            if np.count_nonzero(mask) > 1:
                sub_mat = mat[np.ix_(mask, mask)]
                null_vecs = null_space(sub_mat.toarray())
                if fill_zeros:
                    new_null_vecs = np.zeros((mat.shape[0], null_vecs.shape[1]))
                    new_null_vecs[mask, :] = null_vecs
                    null_vecs = new_null_vecs
                null_spaces.append(null_vecs)
        return null_spaces

    @staticmethod
    def connected_eigh(mat):
        ...

    def type_1_scars(self, target_label: int, fill_zeros: bool = False) -> List[npt.NDArray]:
        parti_idx = self.joint_partition[target_label]
        mask = np.isin(np.arange(self.n_nodes), parti_idx)
        incidence_mat = self.adj_mat[np.ix_(mask, ~mask)]
        scars = self.connected_null_space(incidence_mat @ incidence_mat.T, fill_zeros)
        if fill_zeros:
            for i, scar in enumerate(scars):
                new_scar = np.zeros((self.n_nodes, scar.shape[1]))
                new_scar[mask, :] = scar
                scars[i] = new_scar
        return scars

    def type_3a_scars(self, target_degree: int, fill_zeros: bool = False):
        parti_idx = self.degree_partition[target_degree]
        mask = np.isin(np.arange(self.n_nodes), parti_idx)
        evals, evecs = np.linalg.eigh(sub_mat.toarray())
        evals = evals.round(12)
        scars = []
        for eval in np.unique(evals):
            scar = evecs[:, np.where(evals == eval)[0]]
            if np.allclose(incidence_mat.T @ scar, 0, atol=1e-12):
                logger.info(f"eval: {eval}, num of scars: {scar.shape[1]}")
                scars.append(scar)
        scars = np.hstack(scars)
        sub_mat = self.adj_mat[np.ix_(mask, mask)]
        incidence_mat = self.adj_mat[np.ix_(mask, ~mask)]
        if fill_zeros:
            scars = np.insert(scars, np.where(~mask)[0], 0, axis=0)
        return scars
