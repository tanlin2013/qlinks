import numpy as np
import pandas as pd

from qlinks.model import QuantumLinkModel
from qlinks.solver.deep_first_search import DeepFirstSearch
from qlinks.symmetry.gauss_law import GaussLaw


def setup_dimer_model(
    lattice_shape, n_solution, coup_j, coup_rk, flux_sector=(0, 0), max_steps=int(1e8)
):
    gauss_law = GaussLaw.from_staggered_charge_distri(*lattice_shape)
    gauss_law.flux_sector = flux_sector
    dfs = DeepFirstSearch(gauss_law, max_steps=max_steps)
    basis = gauss_law.to_basis(dfs.solve(n_solution))
    model = QuantumLinkModel(coup_j, coup_rk, lattice_shape, basis)
    return basis, model


def setup_link_model(
    lattice_shape, n_solution, coup_j, coup_rk, flux_sector=(0, 0), max_steps=int(1e8)
):
    gauss_law = GaussLaw.from_zero_charge_distri(*lattice_shape)
    gauss_law.flux_sector = flux_sector
    dfs = DeepFirstSearch(gauss_law, max_steps=max_steps)
    basis = gauss_law.to_basis(dfs.solve(n_solution))
    model = QuantumLinkModel(coup_j, coup_rk, lattice_shape, basis)
    return basis, model


if __name__ == "__main__":
    coup_j, coup_rk = (1, -0.7)  # dfs 3 mins 33 secs
    basis, model = setup_link_model(
        lattice_shape=(6, 4), n_solution=32810, coup_j=coup_j, coup_rk=coup_rk
    )

    evals, evecs = np.linalg.eigh(model.hamiltonian.todense())
    np.savez(
        f"qlm_6x4_coup_j_{coup_j}_coup_rk_{coup_rk}_eigs.npz",
        evals=evals,
        evecs=evecs,
    )

    evecs_df = pd.DataFrame.from_dict(
        {
            "eval": evals,
            "kin": [(evec.T @ model.kinetic_term @ evec).item() for evec in evecs.T],
            "pot": [(evec.T @ model.potential_term @ evec).item() for evec in evecs.T],
        }
    )
    evecs_df.to_parquet(f"qlm_6x4_coup_j_{coup_j}_coup_rk_{coup_rk}_eigs.parquet")
