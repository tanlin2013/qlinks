from functools import wraps
from typing import Any, Callable
from time import perf_counter


def timeit(func: Callable) -> Callable:
    @wraps(func)
    def timeit_wrapper(*args, **kwargs) -> Any:
        start_time = perf_counter()
        result = func(*args, **kwargs)
        end_time = perf_counter()
        total_time = end_time - start_time
        print(f"Function [{func.__qualname__}] took {total_time:e} seconds")
        return result

    return timeit_wrapper
