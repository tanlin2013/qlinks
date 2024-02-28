import concurrent.futures
import os
from itertools import repeat

import networkx as nx
import numpy as np
import pandas as pd
from ed import setup_dimer_model, setup_link_model  # noqa: F401
from tqdm import tqdm

csv_file = "qlm_type1_scars.csv"


def task(lattice_shape, n_solution, coup_j, coup_rk):
    basis, model = setup_link_model(lattice_shape, n_solution, coup_j, coup_rk)
    basis.dataframe.to_parquet(
        f"qlm_{lattice_shape[0]}x{lattice_shape[1]}_lattice.parquet", index=False
    )

    two_steps_mat = model.kinetic_term**2
    degree = np.unique(two_steps_mat.diagonal())
    g = nx.from_scipy_sparse_array(two_steps_mat)
    for d in degree:
        nodes = two_steps_mat.indices[np.where(two_steps_mat.diagonal() == d)]
        sub_g = nx.induced_subgraph(g, nodes)
        mat = nx.to_numpy_array(sub_g)
        nullity = mat.shape[0] - np.linalg.matrix_rank(mat)
        if nullity > 0:
            _df = pd.DataFrame(
                {
                    "lattice_shape": lattice_shape,
                    "n_solution": n_solution,
                    "coup_j": coup_j,
                    "coup_rk": coup_rk,
                    "degree": d,
                    "subgraph_size": mat.shape[0],
                    "nullity": nullity,
                }
            )
            _df.to_csv(csv_file, mode="a", index=False, header=False)


if __name__ == "__main__":

    inputs = zip(
        [(4, 2), (6, 2), (8, 2), (4, 4), (6, 4), (8, 4), (6, 6), (8, 6)],  # lattice_shape
        [38, 282, 2214, 990, 82810, 1159166, 5482716, int(1e8)],  # n_solution
        repeat(1.0),  # coup_j
        repeat(1.0),  # coup_rk
    )

    if not os.path.exists(csv_file):
        df = pd.DataFrame(
            columns=[
                "lattice_shape",
                "n_solution",
                "coup_j",
                "coup_rk",
                "degree",
                "subgraph_size",
                "nullity",
            ]
        )
        df.to_csv(csv_file, index=False)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        _ = tqdm(executor.map(task, inputs), total=len(inputs))
