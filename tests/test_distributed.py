import importlib.util

import pytest

from qlinks.distributed import (
    DistributedConfig,
    TaskFailure,
    map_tasks,
    map_tasks_serial,
)

ray_available = importlib.util.find_spec("ray") is not None


def test_map_tasks_serial_preserves_order() -> None:
    results = map_tasks_serial(
        lambda value: value * value,
        [3, 1, 2],
        show_progress=False,
    )

    assert results == [9, 1, 4]


def test_map_tasks_uses_serial_backend_by_default() -> None:
    results = map_tasks(
        lambda value: value + 1,
        [1, 2, 3],
        config=DistributedConfig(show_progress=False),
    )

    assert results == [2, 3, 4]


def test_map_tasks_serial_can_return_failures() -> None:
    def maybe_fail(value: int) -> int:
        if value == 2:
            raise ValueError("bad value")

        return value

    results = map_tasks_serial(
        maybe_fail,
        [1, 2, 3],
        show_progress=False,
        failure_mode="return",
    )

    assert results[0] == 1
    assert isinstance(results[1], TaskFailure)
    assert results[1].task_index == 1
    assert results[1].exception_type == "ValueError"
    assert results[2] == 3


def square_for_distributed_test(value: int) -> int:
    return value * value


@pytest.mark.skipif(not ray_available, reason="Ray is not installed.")
def test_map_tasks_ray_preserves_order() -> None:
    results = map_tasks(
        square_for_distributed_test,
        [3, 1, 2],
        config=DistributedConfig(
            backend="ray",
            preserve_order=True,
            show_progress=False,
            ray_init_kwargs={
                "num_cpus": 2,
                "ignore_reinit_error": True,
                "include_dashboard": False,
            },
            ray_shutdown=True,
        ),
    )

    assert results == [9, 1, 4]
