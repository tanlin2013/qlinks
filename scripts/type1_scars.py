import os
from threading import Lock
from typing import Callable, List, Sequence

import numpy as np
import pandas as pd
import ray
from ed import setup_dimer_model, setup_link_model  # noqa: F401
from ray.remote_function import RemoteFunction
from tqdm import tqdm

from qlinks import logger
from qlinks.computation_basis import ComputationBasis
from qlinks.model.quantum_link_model import QuantumLinkModel
from qlinks.symmetry.automorphism import Automorphism


def map_on_ray(func: Callable, params: Sequence) -> List:
    """

    Args:
        func:
        params:

    Returns:

    Warnings:
        The results are not order-preserving as the order in input `params`.
    """

    def watch(obj_ids: List[ray.ObjectRef]):
        while obj_ids:
            done, obj_ids = ray.wait(obj_ids)
            yield ray.get(done[0])

    if not ray.is_initialized:
        ray.init()
    func = ray.remote(func) if not isinstance(func, RemoteFunction) else func
    jobs = [func.remote(i) for i in params]
    results = []
    for done_job in tqdm(watch(jobs), desc="Completed jobs", total=len(jobs)):
        results.append(done_job)
    ray.shutdown()
    return results


def setup_storage(model_name):
    csv_file = f"data/{model_name}_type1_scars.csv"
    if not os.path.exists(csv_file):
        df = pd.DataFrame(
            columns=[
                "length_x",
                "length_y",
                "n_solution",
                "label",
                "subgraph_size",
                "n_orbits",
                "degeneracy",
            ]
        )
        df.to_csv(csv_file, index=False)


def count_sets_with_elements(sets, array):
    count = 0
    for s in sets:
        if any(element in s for element in array):
            count += 1
    return count


def task(model, aut, label, model_name):
    csv_file = f"data/{model_name}_type1_scars.csv"
    scars = aut.type_1_scars(label, fill_zeros=True)
    orbits = aut.automorphism_group().orbits()
    for scar in scars:
        if scar.size > 0:
            vertex_id = np.where(np.any(np.abs(scar) > 1e-12, axis=1))[0]
            n_orbits = count_sets_with_elements(orbits, vertex_id)
            _df = pd.DataFrame(
                {
                    "length_x": [model.shape[0]],
                    "length_y": [model.shape[1]],
                    "n_solution": [model.basis.n_states],
                    "label": [label],
                    "subgraph_size": [len(vertex_id)],
                    "n_orbits": [n_orbits],
                    "degeneracy": [scar.shape[1]],
                }
            )
            with Lock():
                _df.to_csv(csv_file, mode="a", index=False, header=False)
            logger.info("\n\t" + _df.to_string(index=False).replace("\n", "\n\t"))


@ray.remote(num_cpus=1)
def task_wrapper(args):
    return task(*args)


if __name__ == "__main__":
    coup_j, coup_rk = (1, 1)
    inputs = [
        ["qdm", (6, 4)],  # 1456
        ["qdm", (8, 4)],  # 17412
        ["qlm", (6, 4)],  # 32810
        ["qdm", (6, 6)],  # 44176
        ["qlm", (8, 4)],  # 1159166
        ["qdm", (8, 6)],  # 1504896
        ["qlm", (6, 6)],  # 5482716
        # ["qlm", (8, 6)],
    ]  # model, lattice_shape

    for model_name, lattice_shape in inputs:
        setup_storage(model_name)
        basis = ComputationBasis.from_parquet(
            f"data/{model_name}_{lattice_shape[0]}x{lattice_shape[1]}_lattice.parquet"
        )

        model = QuantumLinkModel(coup_j, coup_rk, lattice_shape, basis)
        aut = Automorphism(-model.kinetic_term)

        ray.init(num_cpus=28, log_to_driver=True)
        map_on_ray(task_wrapper, [(model, aut, label, model_name) for label in aut.joint_partition])
