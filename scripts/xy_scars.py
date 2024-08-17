import os
from threading import Lock

import numpy as np
import pandas as pd
import ray
from scipy.sparse.csgraph import connected_components

from qlinks import logger
from qlinks.distributed import map_on_ray
from qlinks.model.spin1_xy_model_1d import Spin1XYModel


csv_file = f"spin1_xy_scars.csv"


def setup_storage():
    if not os.path.exists(csv_file):
        df = pd.DataFrame(
            columns=[
                "n",
                "coup_j",
                "coup_h",
                "coup_d",
                "periodic",
                "sz",
                "nullity",
            ]
        )
        df.to_csv(csv_file, index=False)


@ray.remote(num_cpus=1, memory=1 * 1024**3)
def task_wrapper(args):
    return task(*args)


def task(n, coup_j, coup_h, coup_d, periodic):
    model = Spin1XYModel(n, coup_j, coup_h, coup_d, periodic)
    mat = model.hamiltonian.toarray()
    n_components, labels = connected_components(
        mat, directed=False, connection="weak", return_labels=True
    )
    for i in range(n_components):
        mask = (labels == i)  # fmt: skip
        if np.count_nonzero(mask) > 1:
            sub_mat = mat[np.ix_(mask, mask)]
            sz, = np.unique(np.diag(sub_mat)) / coup_h
            sub_mat -= coup_h * sz * np.eye(sub_mat.shape[0])
            sub_mat /= coup_j
            if n % 2 == 1 and periodic:
                non_bipartite_graph(n, coup_j, coup_h, coup_d, periodic, sz, sub_mat)
            else:
                two_step_mat = sub_mat @ sub_mat
                bipartite_graph(n, coup_j, coup_h, coup_d, periodic, sz, two_step_mat)
    return


def bipartite_graph(n, coup_j, coup_h, coup_d, periodic, sz, two_step_mat):
    n_components, labels = connected_components(
        two_step_mat, directed=False, connection="weak", return_labels=True
    )
    for i in range(n_components):
        mask = (labels == i)
        if np.count_nonzero(mask) > 1:
            sub_sub_mat = two_step_mat[np.ix_(mask, mask)]
            _df = pd.DataFrame.from_dict(
                {
                    "n": [n],
                    "coup_j": [coup_j],
                    "coup_h": [coup_h],
                    "coup_d": [coup_d],
                    "periodic": [periodic],
                    "sz": [sz],
                    "nullity": [sub_sub_mat.shape[0] - np.linalg.matrix_rank(sub_sub_mat)],
                }
            )
            with Lock():
                _df.to_csv(csv_file, mode="a", index=False, header=False)
            logger.info("\n\t" + _df.to_string(index=False).replace("\n", "\n\t"))
    return


def non_bipartite_graph(n, coup_j, coup_h, coup_d, periodic, sz, sub_mat):
    _df = pd.DataFrame.from_dict(
        {
            "n": [n],
            "coup_j": [coup_j],
            "coup_h": [coup_h],
            "coup_d": [coup_d],
            "periodic": [periodic],
            "sz": [sz],
            "nullity": [sub_mat.shape[0] - np.linalg.matrix_rank(sub_mat)],
        }
    )
    with Lock():
        _df.to_csv(csv_file, mode="a", index=False, header=False)
    logger.info("\n\t" + _df.to_string(index=False).replace("\n", "\n\t"))
    return


if __name__ == "__main__":
    coup_j, coup_h, coup_d = (1, 1, 0)
    inputs = [
        (n, coup_j, coup_h, coup_d, periodic)
        for n in range(3, 4)
        for periodic in [True, False]
    ]

    setup_storage()
    ray.init(num_cpus=2, log_to_driver=True)
    map_on_ray(task_wrapper, inputs)
