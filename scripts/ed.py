import sys
from time import time

import numpy as np
import pandas as pd

from qlinks import logger
from qlinks.model import QuantumLinkModel
from qlinks.symmetry.gauss_law import GaussLaw


def setup_dimer_model(lattice_shape, coup_j, coup_rk, flux_sector=(0, 0)):
    t0 = time()
    gauss_law = GaussLaw.from_staggered_charge_distri(*lattice_shape, flux_sector=flux_sector)
    basis = gauss_law.solve()
    logger.info(f"lattice {lattice_shape}, DFS time elapsed: {time() - t0:.3e}s")
    logger.info(f"basis memory usage: {sys.getsizeof(basis.links) / (1024 ** 2):.3f} MB")
    basis.dataframe.to_parquet(
        f"data/qdm_{lattice_shape[0]}x{lattice_shape[1]}_lattice.parquet", index=False
    )
    model = QuantumLinkModel(coup_j, coup_rk, lattice_shape, basis)
    return basis, model


def setup_link_model(lattice_shape, coup_j, coup_rk, flux_sector=(0, 0)):
    t0 = time()
    gauss_law = GaussLaw.from_zero_charge_distri(*lattice_shape, flux_sector=flux_sector)
    basis = gauss_law.solve()
    logger.info(f"lattice {lattice_shape}, DFS time elapsed: {time() - t0:.3e}s")
    logger.info(f"basis memory usage: {sys.getsizeof(basis.links) / (1024 ** 2):.3f} MB")
    basis.dataframe.to_parquet(
        f"data/qlm_{lattice_shape[0]}x{lattice_shape[1]}_lattice.parquet", index=False
    )
    model = QuantumLinkModel(coup_j, coup_rk, lattice_shape, basis)
    return basis, model


if __name__ == "__main__":
    coup_j, coup_rk = (1, -0.7)
    basis, model = setup_link_model(
        lattice_shape=(6, 4), coup_j=coup_j, coup_rk=coup_rk
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
