from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd
import pynauty
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from sympy.combinatorics import Permutation, PermutationGroup

from qlinks import logger
from qlinks.solver.linalg import eigh, null_space


@dataclass(slots=True)
class ScarDataHolder:
    evec: npt.NDArray[np.float64] = field(repr=False)
    eval: float = field(default=None)
    shape: Tuple[int, ...] = field(init=False)
    node_idx: npt.NDArray[np.int64] = field(default=None)
    mask: npt.NDArray[np.bool_] = field(default=None, repr=False)

    def __post_init__(self):
        self.shape = self.evec.shape


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
    def degree_partition(self) -> Dict:
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
        groups = self._df.groupby(["degree", "bipartition"]).groups
        return {k: list(v) for k, v in groups.items()}

    def automorphism_group(self, partition=None) -> PermutationGroup:
        ntg = pynauty.Graph(
            self.n_nodes,
            adjacency_dict=nx.to_dict_of_lists(self._graph),
            vertex_coloring=partition,
        )
        return PermutationGroup([Permutation(perm) for perm in pynauty.autgrp(ntg)[0]])

    @staticmethod
    def insert_zeros(arr, mask) -> npt.NDArray:
        new_arr = np.zeros((mask.size, arr.shape[1]))
        new_arr[mask, :] = arr
        return new_arr

    @staticmethod
    def connected_null_space(
        mat, fill_zeros: bool = False, k: Optional[int] = None
    ) -> List[ScarDataHolder]:
        n_components, labels = connected_components(
            mat, directed=False, connection="weak", return_labels=True
        )
        null_spaces = []
        for i in range(n_components):
            mask = (labels == i)  # fmt: skip
            if np.count_nonzero(mask) > 1:
                sub_mat = mat[np.ix_(mask, mask)]
                null_vecs = null_space(sub_mat, k)
                if null_vecs.size > 0:
                    payload = ScarDataHolder(evec=null_vecs, eval=0, mask=mask)
                    if fill_zeros:
                        payload.evec = Automorphism.insert_zeros(null_vecs, mask)
                    null_spaces.append(payload)
        return null_spaces

    @staticmethod
    def connected_eigh(
        mat,
        incidence_mat,
        fill_zeros: bool = False,
        k: Optional[int] = None,
        sigma: Optional[float] = None,
        **kwargs,
    ) -> List[ScarDataHolder]:
        n_components, labels = connected_components(
            mat, directed=False, connection="weak", return_labels=True
        )
        scars = []
        for i in range(n_components):
            mask = (labels == i)  # fmt: skip
            if np.count_nonzero(mask) <= 1:
                continue

            compo_mat = mat[np.ix_(mask, mask)]
            sub_incidence_mat = incidence_mat[mask, :]
            evals, evecs = eigh(compo_mat, k, sigma, **kwargs)
            evals = evals.round(12)

            for eval in np.unique(evals):
                if np.isclose(eval, 0, atol=1e-12):
                    continue

                unitary_mat = evecs[:, evals == eval]
                outer_boundary = sub_incidence_mat.T @ unitary_mat

                if np.allclose(outer_boundary, 0, atol=1e-12):
                    scar = unitary_mat
                elif unitary_mat.shape[1] > 1:
                    coef_mat = null_space(sp.csr_array(outer_boundary))
                    scar = unitary_mat @ coef_mat
                    non_zero_cols = ~np.all(np.abs(scar) < 1e-12, axis=0)
                    scar = scar[:, non_zero_cols]
                else:
                    continue

                if scar.size > 0:
                    logger.info(f"eval: {eval}, num of scars: {scar.shape[1]}")
                    payload = ScarDataHolder(evec=scar, eval=eval, mask=mask)
                    if fill_zeros:
                        payload.evec = Automorphism.insert_zeros(scar, mask)
                    scars.append(payload)
        return scars

    def type_1_scars(
        self, target_label: int, fill_zeros: bool = False, k: Optional[int] = None
    ) -> List[ScarDataHolder]:
        parti_idx = np.asarray(self.joint_partition[target_label])
        mask = np.isin(np.arange(self.n_nodes), parti_idx)
        incidence_mat = self.adj_mat[np.ix_(mask, ~mask)]
        scars = self.connected_null_space(incidence_mat @ incidence_mat.T, fill_zeros, k)
        for payload in scars:
            payload.node_idx = parti_idx[payload.mask]
            if fill_zeros:
                payload.evec = self.insert_zeros(payload.evec, mask)
        return scars

    def type_3a_scars(
        self,
        target_degree: int,
        fill_zeros: bool = False,
        k: Optional[int] = None,
        sigma: Optional[float] = None,
        **kwargs,
    ) -> List[ScarDataHolder]:
        parti_idx = np.asarray(self.degree_partition[target_degree])
        mask = np.isin(np.arange(self.n_nodes), parti_idx)
        sub_mat = self.adj_mat[np.ix_(mask, mask)]
        incidence_mat = self.adj_mat[np.ix_(mask, ~mask)]
        scars = self.connected_eigh(sub_mat, incidence_mat, fill_zeros, k, sigma, **kwargs)
        for payload in scars:
            payload.node_idx = parti_idx[payload.mask]
            if fill_zeros:
                payload.evec = self.insert_zeros(payload.evec, mask)
        return scars
