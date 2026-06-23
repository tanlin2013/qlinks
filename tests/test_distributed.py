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


def test_map_tasks_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Unsupported distributed backend"):
        map_tasks(
            lambda value: value,
            [1],
            config=DistributedConfig(
                backend="unknown",  # type: ignore[arg-type]
                show_progress=False,
            ),
        )


def test_map_tasks_serial_raises_by_default() -> None:
    def fail(_value: int) -> int:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        map_tasks_serial(fail, [1], show_progress=False)


def test_ray_task_wrapper_returns_failure() -> None:
    from qlinks.distributed import ray_task_wrapper

    def fail(_value: int) -> int:
        raise KeyError("missing")

    task_index, result = ray_task_wrapper(3, "input", fail, "return")

    assert task_index == 3
    assert isinstance(result, TaskFailure)
    assert result.task_index == 3
    assert result.task_input == "input"
    assert result.exception_type == "KeyError"
    assert "missing" in result.message


def test_run_indexed_task_raises_when_requested() -> None:
    from qlinks.distributed import _run_indexed_task

    def fail(_value: int) -> int:
        raise RuntimeError("stop")

    with pytest.raises(RuntimeError, match="stop"):
        _run_indexed_task(fail, 0, 1, "raise")


def test_map_tasks_ray_uses_fake_ray_backend(monkeypatch) -> None:
    import sys
    import types

    import qlinks.distributed as distributed

    class FakeObjectRef:
        def __init__(self, value):
            self.value = value

    class FakeRemoteFunction:
        def __init__(self, function):
            self.function = function

        def remote(self, *args):
            return FakeObjectRef(self.function(*args))

    class FakeRay(types.ModuleType):
        def __init__(self):
            super().__init__("ray")
            self.initialized = False
            self.init_kwargs = None
            self.remote_options = []
            self.shutdown_called = False

        def is_initialized(self):
            return self.initialized

        def init(self, **kwargs):
            self.initialized = True
            self.init_kwargs = kwargs

        def remote(self, *args, **kwargs):
            self.remote_options.append(kwargs)
            if args:
                return FakeRemoteFunction(args[0])

            def decorator(function):
                return FakeRemoteFunction(function)

            return decorator

        def wait(self, refs, num_returns=1):
            del num_returns
            return [refs[-1]], refs[:-1]

        def get(self, ref):
            return ref.value

        def shutdown(self):
            self.shutdown_called = True
            self.initialized = False

    fake_ray = FakeRay()
    monkeypatch.setitem(sys.modules, "ray", fake_ray)

    results = distributed.map_tasks_ray(
        lambda value: value * 10,
        [1, 2, 3],
        preserve_order=True,
        show_progress=False,
        ray_init_kwargs={"local_mode": True},
        ray_shutdown=True,
        num_cpus_per_task=0.5,
        num_gpus_per_task=0.25,
    )

    assert results == [10, 20, 30]
    assert fake_ray.init_kwargs == {"local_mode": True}
    assert fake_ray.remote_options == [{"num_cpus": 0.5, "num_gpus": 0.25}]
    assert fake_ray.shutdown_called is True
