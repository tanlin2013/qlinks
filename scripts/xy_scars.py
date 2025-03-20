import os
from threading import Lock

import numpy as np
import pandas as pd
import ray

from qlinks import logger
from qlinks.distributed import map_on_ray
from qlinks.model.spin1_xy_model_1d import Spin1XYModel

csv_file = "spin1_xy_scars.csv"


def setup_storage():
    if not os.path.exists(csv_file):
        df = pd.DataFrame(
            columns=[
                "n",
                "coup_j",
                "coup_h",
                "coup_d",
                "coup_j3",
                "periodic",
                "eval",
                "kin",
                "pot",
                "sz",
            ]
        )
        df.to_csv(csv_file, index=False)


@ray.remote(num_cpus=1, memory=1 * 1024**3)
def task_wrapper(args):
    return task(*args)


def task(n, coup_j, coup_h, coup_d, coup_j3, periodic):
    model = Spin1XYModel(n, coup_j, coup_h, coup_d, coup_j3, periodic)
    mat = model.hamiltonian.toarray()
    evals, evecs = np.linalg.eigh(mat)

    sz = Spin1XYModel(n, coup_j, 1, 0, coup_j3, periodic).potential_term

    evecs_df = pd.DataFrame.from_dict(
        {
            "n": n,
            "coup_j": coup_j,
            "coup_h": coup_h,
            "coup_d": coup_d,
            "coup_j3": coup_j3,
            "periodic": periodic,
            "eval": evals,
            "kin": [(evec.T @ model.kinetic_term @ evec).item() for evec in evecs.T],
            "pot": [(evec.T @ model.potential_term @ evec).item() for evec in evecs.T],
            "sz": [(evec.T @ sz @ evec).item() for evec in evecs.T],
        }
    )
    _df = evecs_df[evecs_df["kin"].abs() < 1e-12].copy()
    with Lock():
        _df.to_csv(csv_file, mode="a", index=False, header=False)
    logger.info("\n\t" + _df.to_string(index=False).replace("\n", "\n\t"))
    return


if __name__ == "__main__":
    coup_j, coup_h, coup_d, coup_j3 = 2 * (np.random.rand(4) - 0.5)
    # fmt: off
    inputs = [
        (n, coup_j, coup_h, coup_d, coup_j3, periodic)
        for n in range(4, 5)
        for periodic in [True, False]
    ]
    # fmt: on

    setup_storage()
    ray.init(num_cpus=2, log_to_driver=True)
    map_on_ray(task_wrapper, inputs)
