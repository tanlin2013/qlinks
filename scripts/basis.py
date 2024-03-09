import concurrent.futures
from itertools import repeat
from time import time

from ed import setup_dimer_model, setup_link_model  # noqa: F401
from tqdm import tqdm

from qlinks import logger


def task(args):
    def task_wrapper(lattice_shape, n_solution, coup_j, coup_rk):
        t0 = time()
        basis, model = setup_link_model(
            lattice_shape, n_solution, coup_j, coup_rk, max_steps=int(1e14)
        )
        basis.dataframe.to_parquet(
            f"data/qlm_{lattice_shape[0]}x{lattice_shape[1]}_lattice.parquet", index=False
        )
        logger.info(f"lattice {lattice_shape}, time elapsed: {time() - t0:.3e}s")

    return task_wrapper(*args)


if __name__ == "__main__":
    inputs = list(
        zip(
            [(8, 4), (6, 6), (8, 6)],  # lattice_shape
            [1159166, 5482716, int(1e11)],  # n_solution
            repeat(1.0),  # coup_j
            repeat(1.0),  # coup_rk
        )
    )

    with tqdm(total=len(inputs)) as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(task, args) for args in inputs]
            for future in concurrent.futures.as_completed(futures):
                pbar.update(1)
                future.result()
