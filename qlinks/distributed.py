"""Generic distributed task execution utilities.

This module intentionally does not depend on model, eigensolver, or caging
objects. It provides a small map-style abstraction that can be reused by
caging searches, parameter sweeps, and future workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Literal, Sequence, TypeVar

from tqdm import tqdm

InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")

DistributedBackend = Literal["serial", "ray"]
FailureMode = Literal["raise", "return"]


@dataclass(frozen=True, slots=True)
class DistributedConfig:
    """Configuration for generic distributed task execution."""

    backend: DistributedBackend = "serial"
    preserve_order: bool = True
    show_progress: bool = True
    progress_description: str = "Completed jobs"
    failure_mode: FailureMode = "raise"
    ray_init_kwargs: dict | None = None
    ray_shutdown: bool = False
    num_cpus_per_task: float | None = None
    num_gpus_per_task: float | None = None


@dataclass(frozen=True, slots=True)
class TaskFailure:
    """Container returned when a task fails and failure_mode='return'."""

    task_index: int
    task_input: object
    exception_type: str
    message: str


def map_tasks(
    function: Callable[[InputType], OutputType],
    inputs: Sequence[InputType],
    *,
    config: DistributedConfig | None = None,
) -> list[OutputType] | list[OutputType | TaskFailure]:
    """Map a function over inputs using the configured execution backend."""
    if config is None:
        config = DistributedConfig()

    if config.backend == "serial":
        return map_tasks_serial(
            function,
            inputs,
            preserve_order=config.preserve_order,
            show_progress=config.show_progress,
            progress_description=config.progress_description,
            failure_mode=config.failure_mode,
        )

    if config.backend == "ray":
        return map_tasks_ray(
            function,
            inputs,
            preserve_order=config.preserve_order,
            show_progress=config.show_progress,
            progress_description=config.progress_description,
            failure_mode=config.failure_mode,
            ray_init_kwargs=config.ray_init_kwargs,
            ray_shutdown=config.ray_shutdown,
            num_cpus_per_task=config.num_cpus_per_task,
            num_gpus_per_task=config.num_gpus_per_task,
        )

    raise ValueError(f"Unsupported distributed backend: {config.backend!r}")


def map_tasks_serial(
    function: Callable[[InputType], OutputType],
    inputs: Sequence[InputType],
    *,
    preserve_order: bool = True,
    show_progress: bool = True,
    progress_description: str = "Completed jobs",
    failure_mode: FailureMode = "raise",
) -> list[OutputType] | list[OutputType | TaskFailure]:
    """Map tasks serially.

    ``preserve_order`` is accepted for API compatibility. Serial execution
    naturally preserves input order.
    """
    del preserve_order

    iterator: Iterable[tuple[int, InputType]] = enumerate(inputs)

    if show_progress:
        iterator = tqdm(
            iterator,
            desc=progress_description,
            total=len(inputs),
        )

    results: list[OutputType | TaskFailure] = []

    for task_index, task_input in iterator:
        try:
            results.append(function(task_input))
        except Exception as error:
            if failure_mode == "raise":
                raise

            results.append(
                TaskFailure(
                    task_index=task_index,
                    task_input=task_input,
                    exception_type=type(error).__name__,
                    message=str(error),
                )
            )

    return results


def map_tasks_ray(
    function: Callable[[InputType], OutputType],
    inputs: Sequence[InputType],
    *,
    preserve_order: bool = True,
    show_progress: bool = True,
    progress_description: str = "Completed jobs",
    failure_mode: FailureMode = "raise",
    ray_init_kwargs: dict | None = None,
    ray_shutdown: bool = False,
    num_cpus_per_task: float | None = None,
    num_gpus_per_task: float | None = None,
) -> list[OutputType] | list[OutputType | TaskFailure]:
    """Map tasks with Ray."""
    try:
        import ray
    except ImportError as error:
        raise ImportError(
            "Ray is required for backend='ray'. Install ray before using "
            "DistributedConfig(backend='ray')."
        ) from error

    if ray_init_kwargs is None:
        ray_init_kwargs = {}

    if not ray.is_initialized():
        ray.init(**ray_init_kwargs)

    remote_options: dict[str, float] = {}

    if num_cpus_per_task is not None:
        remote_options["num_cpus"] = num_cpus_per_task

    if num_gpus_per_task is not None:
        remote_options["num_gpus"] = num_gpus_per_task

    remote_runner = ray.remote(**remote_options)(_run_indexed_task)

    object_refs = [
        remote_runner.remote(
            function,
            task_index,
            task_input,
            failure_mode,
        )
        for task_index, task_input in enumerate(inputs)
    ]

    pending_refs = list(object_refs)
    completion_results: list[tuple[int, OutputType | TaskFailure]] = []

    progress_bar = None

    if show_progress:
        progress_bar = tqdm(
            total=len(object_refs),
            desc=progress_description,
        )

    try:
        while pending_refs:
            done_refs, pending_refs = ray.wait(pending_refs, num_returns=1)
            indexed_result = ray.get(done_refs[0])
            completion_results.append(indexed_result)

            if progress_bar is not None:
                progress_bar.update(1)
    finally:
        if progress_bar is not None:
            progress_bar.close()

        if ray_shutdown:
            ray.shutdown()

    if preserve_order:
        completion_results.sort(key=lambda indexed_result: indexed_result[0])

    return [result for _task_index, result in completion_results]


def ray_task_wrapper(
    task_index: int,
    task_input: InputType,
    function: Callable[[InputType], OutputType],
    failure_mode: FailureMode,
) -> tuple[int, OutputType | TaskFailure]:
    """Run one task and return its index together with the result."""
    try:
        return task_index, function(task_input)
    except Exception as error:
        if failure_mode == "raise":
            raise

        return task_index, TaskFailure(
            task_index=task_index,
            task_input=task_input,
            exception_type=type(error).__name__,
            message=str(error),
        )


def _run_indexed_task(
    function: Callable[[InputType], OutputType],
    task_index: int,
    task_input: InputType,
    failure_mode: FailureMode,
) -> tuple[int, OutputType | TaskFailure]:
    """Run one indexed task."""
    try:
        return task_index, function(task_input)
    except Exception as error:
        if failure_mode == "raise":
            raise

        return task_index, TaskFailure(
            task_index=task_index,
            task_input=task_input,
            exception_type=type(error).__name__,
            message=str(error),
        )


def map_on_ray(
    func: Callable[[InputType], OutputType],
    params: Sequence[InputType],
) -> list[OutputType]:
    """Backward-compatible Ray map.

    Results preserve the historical behavior and are returned in completion
    order, not input order.
    """
    return map_tasks(
        func,
        params,
        config=DistributedConfig(
            backend="ray",
            preserve_order=False,
            show_progress=True,
            progress_description="Completed jobs",
            ray_shutdown=True,
        ),
    )
