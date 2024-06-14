import concurrent.futures
import os

import pandas as pd
from tqdm import tqdm

from ed import setup_dimer_model, setup_link_model  # noqa: F401
from qlinks import logger
from qlinks.computation_basis import ComputationBasis
from qlinks.model.quantum_link_model import QuantumLinkModel
from qlinks.symmetry.automorphism import Automorphism


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
    scars = aut.type_1_scars(label, fill_zeros=False)
    if scars.size > 0:
        orbits = aut.automorphism_group().orbits()
        vertex_id = aut.joint_partition[label]
        n_orbits = count_sets_with_elements(orbits, vertex_id)
        _df = pd.DataFrame(
            {
                "length_x": [model.shape[0]],
                "length_y": [model.shape[1]],
                "n_solution": [model.basis.n_states],
                "label": [label],
                "subgraph_size": [scars.shape[0]],
                "n_orbits": [n_orbits],
                "degeneracy": [scars.shape[1]],
            }
        )
        _df.to_csv(csv_file, mode="a", index=False, header=False)
        logger.info("\n\t" + _df.to_string(index=False).replace("\n", "\n\t"))


def task_wrapper(args):
    return task(*args)


if __name__ == "__main__":
    coup_j, coup_rk = (1, 1)
    inputs = [
        ["qlm", (6, 4)],
        ["qlm", (8, 4)],
        ["qlm", (6, 6)],
        # ["qlm", (8, 6)],
        ["qdm", (6, 4)],
        ["qdm", (8, 4)],
        ["qdm", (6, 6)],
        ["qdm", (8, 6)],
    ]  # model, lattice_shape

    futures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=28) as executor:

        for model_name, lattice_shape in inputs:
            setup_storage(model_name)
            basis = ComputationBasis.from_parquet(
                f"data/{model_name}_{lattice_shape[0]}x{lattice_shape[1]}_lattice.parquet"
            )

            model = QuantumLinkModel(coup_j, coup_rk, lattice_shape, basis)
            aut = Automorphism(-model.kinetic_term)
            futures += [
                executor.submit(task_wrapper, (model, aut, label, model_name))
                for label in aut.joint_partition
            ]

        with tqdm(total=len(futures), leave=True) as pbar:
            for future in concurrent.futures.as_completed(futures):
                future.result()
                pbar.update(1)
