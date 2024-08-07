import os
from threading import Lock

import numpy as np
import pandas as pd
import ray
from ed import setup_dimer_model, setup_link_model  # noqa: F401

from qlinks import logger
from qlinks.computation_basis import ComputationBasis
from qlinks.distributed import map_on_ray
from qlinks.exceptions import InvalidOperationError
from qlinks.model.quantum_link_model import QuantumLinkModel
from qlinks.symmetry.automorphism import Automorphism
from qlinks.symmetry.gauss_law import GaussLaw


def setup_storage(model_name):
    csv_file = f"data/{model_name}_type3a_scars.csv"
    if not os.path.exists(csv_file):
        df = pd.DataFrame(
            columns=[
                "length_x",
                "length_y",
                "n_solution",
                "label",
                "subgraph_size",
                "n_orbits",
                "reduced_subgraph_size",
                "reduced_n_orbits",
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


def task(model, aut, label, model_name, k=None, sigma=None):
    csv_file = f"data/{model_name}_type3a_scars.csv"
    try:
        scars = aut.type_3a_scars(label, fill_zeros=True, k=k, sigma=sigma)
        orbits = aut.automorphism_group().orbits()
        for scar in scars:
            n_orbits = count_sets_with_elements(orbits, scar.node_idx)
            node_idx = np.where(np.any(np.abs(scar.evec) > 1e-12, axis=1))[0]
            reduced_n_orbits = count_sets_with_elements(orbits, node_idx)
            _df = pd.DataFrame(
                {
                    "length_x": [model.shape[0]],
                    "length_y": [model.shape[1]],
                    "n_solution": [model.basis.n_states],
                    "label": [label],
                    "subgraph_size": [scar.shape[0]],
                    "n_orbits": [n_orbits],
                    "reduced_subgraph_size": [len(node_idx)],
                    "reduced_n_orbits": [reduced_n_orbits],
                    "degeneracy": [scar.shape[1]],
                }
            )
            with Lock():
                _df.to_csv(csv_file, mode="a", index=False, header=False)
            logger.info("\n\t" + _df.to_string(index=False).replace("\n", "\n\t"))
        if scars:
            scar_storable = [scar.evec for scar in scars]
            np.savez(f"data/{model_name}_{model.shape}_type3a_scars_{label}.npz", *scar_storable)
    except Exception as e:
        logger.error(f"{model_name}, {model.shape}, {label}")
        logger.exception(f"{e}")
        pass


@ray.remote(num_cpus=1, memory=220 * 1024**3)
def task_wrapper(args):
    return task(*args)


if __name__ == "__main__":
    coup_j, coup_rk = (1, 1)
    inputs = [
        # ["qlm", (4, 2)],  # 38
        # ["qlm", (6, 2)],  # 282
        # ["qlm", (4, 4)],  # 990
        # ["qdm", (4, 2)],  # 16
        # ["qdm", (6, 2)],  # 76
        # ["qdm", (4, 4)],  # 132
        # ["qdm", (6, 4)],  # 1456
        # ["qdm", (8, 4)],  # 17412
        # ["qlm", (6, 4)],  # 32810
        # ["qdm", (6, 6)],  # 44176
        ["qlm", (8, 4)],  # 1159166
        # ["qdm", (8, 6)],  # 1504896
        # ["qlm", (6, 6)],  # 5482716
        # ["qlm", (8, 6)],
    ]  # model, lattice_shape

    for model_name, lattice_shape in inputs:
        setup_storage(model_name)
        try:
            basis = ComputationBasis.from_parquet(
                f"data/{model_name}_{lattice_shape[0]}x{lattice_shape[1]}_lattice.parquet"
            )
            model = QuantumLinkModel(coup_j, coup_rk, lattice_shape, basis)
        except (FileNotFoundError, InvalidOperationError):
            gauss_law = {
                "qlm": GaussLaw.from_zero_charge_distri,
                "qdm": GaussLaw.from_staggered_charge_distri,
            }[model_name](*lattice_shape, flux_sector=(0, 0))
            basis = gauss_law.solve()
            basis.to_parquet(
                f"data/{model_name}_{lattice_shape[0]}x{lattice_shape[1]}_lattice.parquet"
            )
            model = QuantumLinkModel(coup_j, coup_rk, lattice_shape, basis)

        aut = Automorphism(-model.kinetic_term)

        # cheat_sheet = [(16, 24, 2), (16, 24, -2)]  # k=24 is a random guess
        ray.init(num_cpus=2, log_to_driver=True)
        map_on_ray(
            task_wrapper,
            [(model, aut, label, model_name) for label in aut.degree_partition.keys()],
        )
