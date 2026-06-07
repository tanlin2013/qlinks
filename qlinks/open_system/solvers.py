from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np

from qlinks.open_system.backend import (
    OpenSystemBackend,
    OpenSystemBackendName,
    get_open_system_backend,
)
from qlinks.open_system.diagnostics import verify_density_matrix
from qlinks.open_system.operators import (
    build_liouvillian,
    estimate_lindblad_scale,
    lindblad_rhs_density_matrix,
    unvectorize_density_matrix,
    vectorize_density_matrix,
)

LindbladSolverMethod = Literal[
    "auto",
    "krylov",
    "rk4_matrix",
    "rk4_liouville",
]


Rk4StepPolicy = Literal[
    "raise",
    "warn",
    "adaptive",
    "ignore",
]


@dataclass(frozen=True, slots=True)
class LindbladEvolutionOptions:
    method: LindbladSolverMethod = "auto"
    backend: OpenSystemBackendName = "scipy"
    rk4_step_policy: Rk4StepPolicy = "adaptive"
    max_dimension_for_liouvillian: int = 400
    max_dimension_for_krylov: int = 400
    max_rk4_step_scale: float = 0.05
    adaptive_tolerance: float = 1e-8
    min_substeps: int = 1
    max_substeps: int = 1024
    enforce_hermiticity: bool = True
    renormalize_trace: bool = True
    check_density_matrix: bool = True


@dataclass(frozen=True, slots=True)
class LindbladEvolutionResult:
    times: np.ndarray
    density_matrices: list[Any]
    method: str
    backend: str
    diagnostics: list[Any]
    n_substeps_per_interval: tuple[int, ...]


class LindbladProblem:
    def __init__(
        self,
        *,
        hamiltonian: Any,
        jumps: list[Any] | tuple[Any, ...],
        backend: OpenSystemBackendName | OpenSystemBackend = "scipy",
    ) -> None:
        self.backend = get_open_system_backend(backend)
        self.hamiltonian = hamiltonian
        self.jumps = tuple(jumps)

    @property
    def dim(self) -> int:
        return int(self.hamiltonian.shape[0])

    def build_liouvillian(self, *, sparse_format: str = "csc"):
        return build_liouvillian(
            self.hamiltonian,
            self.jumps,
            backend=self.backend,
            sparse_format=sparse_format,
        )

    def rhs(self, density_matrix: Any):
        return lindblad_rhs_density_matrix(
            density_matrix,
            hamiltonian=self.hamiltonian,
            jumps=self.jumps,
            backend=self.backend,
        )

    def evolve(
        self,
        density_matrix_initial: Any,
        times: np.ndarray,
        *,
        options: LindbladEvolutionOptions | None = None,
    ) -> LindbladEvolutionResult:
        if options is None:
            options = LindbladEvolutionOptions(backend=self.backend.name)

        times = np.asarray(times, dtype=np.float64)
        _validate_times(times)

        method = _choose_method(
            method=options.method,
            dim=self.dim,
            backend=self.backend,
            options=options,
        )

        if method == "krylov":
            return _evolve_krylov(
                problem=self,
                density_matrix_initial=density_matrix_initial,
                times=times,
                options=options,
            )

        if method == "rk4_liouville":
            return _evolve_rk4_liouville(
                problem=self,
                density_matrix_initial=density_matrix_initial,
                times=times,
                options=options,
            )

        if method == "rk4_matrix":
            return _evolve_rk4_matrix(
                problem=self,
                density_matrix_initial=density_matrix_initial,
                times=times,
                options=options,
            )

        raise ValueError(f"Unsupported Lindblad solver method: {method!r}")


def solve_lindblad(
    *,
    hamiltonian: Any,
    jumps: list[Any] | tuple[Any, ...],
    density_matrix_initial: Any,
    times: np.ndarray,
    method: LindbladSolverMethod = "auto",
    backend: OpenSystemBackendName = "scipy",
    options: LindbladEvolutionOptions | None = None,
) -> LindbladEvolutionResult:
    if options is None:
        options = LindbladEvolutionOptions(method=method, backend=backend)
    else:
        options = replace(options, method=method, backend=backend)

    problem = LindbladProblem(
        hamiltonian=hamiltonian,
        jumps=jumps,
        backend=backend,
    )
    return problem.evolve(
        density_matrix_initial,
        times,
        options=options,
    )


def _validate_times(times: np.ndarray) -> None:
    if times.ndim != 1 or times.size < 2:
        raise ValueError("times must be a one-dimensional array with at least two entries.")
    if not np.all(np.diff(times) > 0):
        raise ValueError("times must be strictly increasing.")


def _choose_method(
    *,
    method: LindbladSolverMethod,
    dim: int,
    backend: OpenSystemBackend,
    options: LindbladEvolutionOptions,
) -> str:
    if method != "auto":
        if method == "krylov" and not backend.supports_expm_multiply:
            raise ValueError(f"backend={backend.name!r} does not provide expm_multiply.")
        return method

    if (
        backend.name == "scipy"
        and backend.supports_expm_multiply
        and dim <= options.max_dimension_for_krylov
    ):
        return "krylov"

    if dim <= options.max_dimension_for_liouvillian:
        return "rk4_liouville"

    return "rk4_matrix"


def _rk4_step_matrix(problem: LindbladProblem, density_matrix: Any, step_size: float):
    rhs = problem.rhs

    slope_1 = rhs(density_matrix)
    slope_2 = rhs(density_matrix + 0.5 * step_size * slope_1)
    slope_3 = rhs(density_matrix + 0.5 * step_size * slope_2)
    slope_4 = rhs(density_matrix + step_size * slope_3)

    return density_matrix + (step_size / 6.0) * (slope_1 + 2.0 * slope_2 + 2.0 * slope_3 + slope_4)


def _rk4_step_matrix_adaptive(
    problem: LindbladProblem,
    density_matrix: Any,
    step_size: float,
    *,
    tolerance: float,
    max_substeps: int,
):
    substeps = 1

    while substeps <= max_substeps:
        coarse_step = step_size / substeps
        fine_step = coarse_step / 2.0

        coarse_state = density_matrix
        for _ in range(substeps):
            coarse_state = _rk4_step_matrix(problem, coarse_state, coarse_step)

        fine_state = density_matrix
        for _ in range(2 * substeps):
            fine_state = _rk4_step_matrix(problem, fine_state, fine_step)

        error = problem.backend.norm(fine_state - coarse_state)

        if error <= tolerance:
            return fine_state, 2 * substeps, error

        substeps *= 2

    raise RuntimeError(
        "Adaptive RK4 failed to reach the requested tolerance. "
        f"Last step_size={step_size}, tolerance={tolerance}, "
        f"max_substeps={max_substeps}."
    )


def _postprocess_density_matrix(
    density_matrix: Any,
    *,
    backend: OpenSystemBackend,
    enforce_hermiticity: bool,
    renormalize_trace: bool,
):
    array_module = backend.array_module

    if enforce_hermiticity:
        density_matrix = 0.5 * (density_matrix + density_matrix.conj().T)

    if renormalize_trace:
        trace = array_module.trace(density_matrix)
        if abs(complex(trace)) > 0:
            density_matrix = density_matrix / trace

    return density_matrix


def _check_rk4_step_size(
    *,
    step_size: float,
    scale: float,
    options: LindbladEvolutionOptions,
) -> None:
    scaled_step = abs(step_size) * scale

    if scaled_step <= options.max_rk4_step_scale:
        return

    message = (
        "RK4 time step may be too large: "
        f"dt * estimated_scale = {scaled_step:.3e}, "
        f"recommended <= {options.max_rk4_step_scale:.3e}. "
        "Use rk4_step_policy='adaptive', reduce dt, or use method='krylov'."
    )

    if options.rk4_step_policy == "raise":
        raise ValueError(message)

    if options.rk4_step_policy == "warn":
        warnings.warn(message, RuntimeWarning, stacklevel=2)


def _evolve_rk4_matrix(
    *,
    problem: LindbladProblem,
    density_matrix_initial: Any,
    times: np.ndarray,
    options: LindbladEvolutionOptions,
) -> LindbladEvolutionResult:
    backend = problem.backend
    density_matrix = backend.asarray(density_matrix_initial)

    density_matrices = [density_matrix.copy()]
    diagnostics = []
    substeps_used: list[int] = []

    if options.check_density_matrix:
        diagnostics.append(
            verify_density_matrix(
                backend.to_numpy(density_matrix),
            )
        )

    scale = estimate_lindblad_scale(
        hamiltonian=problem.hamiltonian,
        jumps=problem.jumps,
        backend=backend,
    )

    for time_index in range(times.size - 1):
        step_size = float(times[time_index + 1] - times[time_index])

        if options.rk4_step_policy == "adaptive":
            density_matrix, n_substeps, _error = _rk4_step_matrix_adaptive(
                problem,
                density_matrix,
                step_size,
                tolerance=options.adaptive_tolerance,
                max_substeps=options.max_substeps,
            )
        else:
            _check_rk4_step_size(
                step_size=step_size,
                scale=scale,
                options=options,
            )
            density_matrix = _rk4_step_matrix(problem, density_matrix, step_size)
            n_substeps = 1

        density_matrix = _postprocess_density_matrix(
            density_matrix,
            backend=backend,
            enforce_hermiticity=options.enforce_hermiticity,
            renormalize_trace=options.renormalize_trace,
        )

        density_matrices.append(density_matrix.copy())
        substeps_used.append(n_substeps)

        if options.check_density_matrix:
            diagnostics.append(
                verify_density_matrix(
                    backend.to_numpy(density_matrix),
                )
            )

    return LindbladEvolutionResult(
        times=times,
        density_matrices=density_matrices,
        method="rk4_matrix",
        backend=backend.name,
        diagnostics=diagnostics,
        n_substeps_per_interval=tuple(substeps_used),
    )


def _validate_uniform_times_for_krylov(times: np.ndarray) -> None:
    step_sizes = np.diff(times)
    if not np.allclose(step_sizes, step_sizes[0]):
        raise ValueError(
            "method='krylov' currently requires a uniform time grid because "
            "scipy.sparse.linalg.expm_multiply is called with start/stop/num."
        )


def _evolve_krylov(
    *,
    problem: LindbladProblem,
    density_matrix_initial: Any,
    times: np.ndarray,
    options: LindbladEvolutionOptions,
) -> LindbladEvolutionResult:
    if not problem.backend.supports_expm_multiply:
        raise ValueError(f"backend={problem.backend.name!r} does not support Krylov/expm_multiply.")

    _validate_uniform_times_for_krylov(times)

    liouvillian = problem.build_liouvillian(sparse_format="csc")
    dim = problem.dim
    vector_initial = vectorize_density_matrix(
        problem.backend.asarray(density_matrix_initial),
    )

    vectors = problem.backend.sparse_linalg_module.expm_multiply(
        liouvillian,
        vector_initial,
        start=float(times[0]),
        stop=float(times[-1]),
        num=int(times.size),
        endpoint=True,
    )

    density_matrices = [unvectorize_density_matrix(vector, dim) for vector in vectors]

    density_matrices = [
        _postprocess_density_matrix(
            density_matrix,
            backend=problem.backend,
            enforce_hermiticity=options.enforce_hermiticity,
            renormalize_trace=options.renormalize_trace,
        )
        for density_matrix in density_matrices
    ]

    diagnostics = []
    if options.check_density_matrix:
        diagnostics = [
            verify_density_matrix(problem.backend.to_numpy(density_matrix))
            for density_matrix in density_matrices
        ]

    return LindbladEvolutionResult(
        times=times,
        density_matrices=density_matrices,
        method="krylov",
        backend=problem.backend.name,
        diagnostics=diagnostics,
        n_substeps_per_interval=tuple(1 for _ in range(times.size - 1)),
    )


def _rk4_step_vector(
    liouvillian,
    vectorized_density_matrix,
    step_size: float,
):
    slope_1 = liouvillian @ vectorized_density_matrix
    slope_2 = liouvillian @ (vectorized_density_matrix + 0.5 * step_size * slope_1)
    slope_3 = liouvillian @ (vectorized_density_matrix + 0.5 * step_size * slope_2)
    slope_4 = liouvillian @ (vectorized_density_matrix + step_size * slope_3)

    return vectorized_density_matrix + (step_size / 6.0) * (
        slope_1 + 2.0 * slope_2 + 2.0 * slope_3 + slope_4
    )


def _rk4_step_vector_adaptive(
    liouvillian,
    vectorized_density_matrix,
    step_size: float,
    *,
    backend: OpenSystemBackend,
    tolerance: float,
    max_substeps: int,
):
    substeps = 1

    while substeps <= max_substeps:
        coarse_step_size = step_size / substeps
        fine_step_size = coarse_step_size / 2.0

        coarse_state = vectorized_density_matrix
        for _ in range(substeps):
            coarse_state = _rk4_step_vector(
                liouvillian,
                coarse_state,
                coarse_step_size,
            )

        fine_state = vectorized_density_matrix
        for _ in range(2 * substeps):
            fine_state = _rk4_step_vector(
                liouvillian,
                fine_state,
                fine_step_size,
            )

        error = backend.norm(fine_state - coarse_state)

        if error <= tolerance:
            return fine_state, 2 * substeps, error

        substeps *= 2

    raise RuntimeError(
        "Adaptive Liouville-space RK4 failed to reach the requested "
        f"tolerance={tolerance:.3e} with max_substeps={max_substeps}."
    )


def _evolve_rk4_liouville(
    *,
    problem: LindbladProblem,
    density_matrix_initial: Any,
    times: np.ndarray,
    options: LindbladEvolutionOptions,
) -> LindbladEvolutionResult:
    backend = problem.backend
    density_matrix = backend.asarray(density_matrix_initial)

    liouvillian = problem.build_liouvillian(sparse_format="csr")
    vectorized_density_matrix = vectorize_density_matrix(density_matrix)

    density_matrices = [density_matrix.copy()]
    diagnostics = []
    substeps_used: list[int] = []

    if options.check_density_matrix:
        diagnostics.append(
            verify_density_matrix(
                backend.to_numpy(density_matrix),
            )
        )

    scale = estimate_lindblad_scale(
        hamiltonian=problem.hamiltonian,
        jumps=problem.jumps,
        backend=backend,
    )

    for time_index in range(times.size - 1):
        step_size = float(times[time_index + 1] - times[time_index])

        if options.rk4_step_policy == "adaptive":
            vectorized_density_matrix, n_substeps, _error = _rk4_step_vector_adaptive(
                liouvillian,
                vectorized_density_matrix,
                step_size,
                backend=backend,
                tolerance=options.adaptive_tolerance,
                max_substeps=options.max_substeps,
            )
        else:
            _check_rk4_step_size(
                step_size=step_size,
                scale=scale,
                options=options,
            )
            vectorized_density_matrix = _rk4_step_vector(
                liouvillian,
                vectorized_density_matrix,
                step_size,
            )
            n_substeps = 1

        density_matrix = unvectorize_density_matrix(
            vectorized_density_matrix,
            problem.dim,
        )

        density_matrix = _postprocess_density_matrix(
            density_matrix,
            backend=backend,
            enforce_hermiticity=options.enforce_hermiticity,
            renormalize_trace=options.renormalize_trace,
        )

        # Keep the vectorized state consistent after postprocessing.
        vectorized_density_matrix = vectorize_density_matrix(density_matrix)

        density_matrices.append(density_matrix.copy())
        substeps_used.append(n_substeps)

        if options.check_density_matrix:
            diagnostics.append(
                verify_density_matrix(
                    backend.to_numpy(density_matrix),
                )
            )

    return LindbladEvolutionResult(
        times=times,
        density_matrices=density_matrices,
        method="rk4_liouville",
        backend=backend.name,
        diagnostics=diagnostics,
        n_substeps_per_interval=tuple(substeps_used),
    )
