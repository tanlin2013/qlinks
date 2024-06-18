import time
from typing import Callable, List, Sequence

import ray
from ray.remote_function import RemoteFunction
from tqdm import tqdm


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
    time.sleep(1)  # wait for stdout to flush
    ray.shutdown()
    return results
