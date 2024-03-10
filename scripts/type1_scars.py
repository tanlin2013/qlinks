import concurrent.futures
import os
from itertools import repeat
from time import time

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import igraph
from ed import setup_dimer_model, setup_link_model  # noqa: F401
from tqdm import tqdm

from qlinks import logger
from qlinks.computation_basis import ComputationBasis
from qlinks.model import QuantumLinkModel

csv_file = "data/qlm_type1_scars.csv"


def matrix_nullity(mat):
    if mat.shape[0] < 2**12:
        return mat.shape[0] - np.linalg.matrix_rank(mat.toarray())
    else:
        s = sp.linalg.eigsh(
            mat, k=mat.shape[0] // 2, sigma=1e-12, which="SA", return_eigenvectors=False
        )
        return np.count_nonzero(np.abs(s) < 1e-12)


def task(ig, two_steps_mat, d):
    sub_ig = ig.induced_subgraph(np.where(two_steps_mat.diagonal() == d)[0])
    for c in sub_ig.connected_components(mode="weak"):
        mat = nx.to_scipy_sparse_array(sub_ig.subgraph(c).to_networkx())
        t0 = time()
        nullity = matrix_nullity(mat)
        logger.info(f"nullity calculation time: {time() - t0:.3e}s")
        return d, mat.shape[0], nullity


def task_wrapper(args):
    return task(*args)


if __name__ == "__main__":
    inputs = list(
        zip(
            [(8, 4), (6, 6), (8, 6)],  # lattice_shape
            # [1159166, 5482716, int(1e12)],  # n_solution
            repeat(1.0),  # coup_j
            repeat(1.0),  # coup_rk
        )
    )

    if not os.path.exists(csv_file):
        df = pd.DataFrame(
            columns=[
                "lattice_length_x",
                "lattice_length_y",
                "n_solution",
                "coup_j",
                "coup_rk",
                "degree",
                "subgraph_size",
                "nullity",
            ]
        )
        df.to_csv(csv_file, index=False)

    for lattice_shape, coup_j, coup_rk in tqdm(
        inputs, desc=" outer", position=0, leave=True
    ):
        # basis, model = setup_link_model(
        #     lattice_shape, n_solution, coup_j, coup_rk, max_steps=int(1e8)
        # )
        # basis.dataframe.to_parquet(
        #     f"data/qlm_{lattice_shape[0]}x{lattice_shape[1]}_lattice.parquet", index=False
        # )
        basis = ComputationBasis.from_parquet(f"data/qlm_{lattice_shape[0]}x{lattice_shape[1]}_lattice.parquet")
        model = QuantumLinkModel(coup_j, coup_rk, lattice_shape, basis)

        two_steps_mat = model.kinetic_term @ model.kinetic_term
        degree = np.unique(two_steps_mat.diagonal()).astype(int)
        logger.info(f"system size: {lattice_shape}, degree: {degree}")

        ig = igraph.Graph.from_networkx(nx.from_scipy_sparse_array(two_steps_mat))
        with concurrent.futures.ProcessPoolExecutor(max_workers=28) as executor:
            inner_inputs = [(ig, two_steps_mat, d) for d in degree]
            futures = [executor.submit(task_wrapper, args) for args in inner_inputs]
            with tqdm(total=len(inputs), desc=" inner", position=1, leave=True) as pbar:
                for future in concurrent.futures.as_completed(futures):
                    d, sub_ig_size, nullity = future.result()
                    if nullity > 0:
                        _df = pd.DataFrame(
                            {
                                "lattice_length_x": [lattice_shape[0]],
                                "lattice_length_y": [lattice_shape[1]],
                                "n_solution": [basis.n_states],
                                "coup_j": [coup_j],
                                "coup_rk": [coup_rk],
                                "degree": [d],
                                "subgraph_size": [sub_ig_size],
                                "nullity": [nullity],
                            }
                        )
                        _df.to_csv(csv_file, mode="a", index=False, header=False)
                        logger.info("\n\t" + _df.to_string(index=False).replace("\n", "\n\t"))
                    pbar.update(1)
