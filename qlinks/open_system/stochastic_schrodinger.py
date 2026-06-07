from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np
from numpy.typing import NDArray

from qlinks.open_system.backend import (
    OpenSystemBackend,
    OpenSystemBackendName,
    as_backend_dense_array,
    get_open_system_backend,
)
from qlinks.open_system.states import random_pure_state

ArrayC = NDArray[np.complex128]
ArrayF = NDArray[np.float64]
StateSampler = Callable[[np.random.Generator], Any]


@dataclass(frozen=True, slots=True)
class McwfOptions:
    """Options for Monte Carlo wave-function sampling."""

    backend: OpenSystemBackendName = "scipy"
    n_trajectories: int = 128
    seed: int | None = None
    return_backend_arrays: bool = False
    store_states: bool = True
    store_trajectories: bool = False
    normalize_each_step: bool = True
    max_jump_probability: float = 0.1


@dataclass(frozen=True, slots=True)
class TrajectoryResult:
    """Single Monte Carlo wave-function trajectory."""

    times: NDArray[np.float64]
    states: list[Any]
    jump_times: NDArray[np.float64]
    jump_indices: NDArray[np.int64]
    norm_errors: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class EnsembleResult:
    """Ensemble-averaged MCWF result."""

    times: NDArray[np.float64]
    rho_t: list[ArrayC]
    trajectories: tuple[TrajectoryResult, ...] | None = None


def projector(state: Any) -> Any:
    """Return |state><state|."""
    state_column = state.reshape(-1, 1)
    return state_column @ state_column.conj().T


def expectation(state: Any, operator: Any) -> complex:
    """Return <state|operator|state>."""
    return complex(np.vdot(state, operator @ state))


def effective_hamiltonian(hamiltonian: Any, jumps: list[Any] | tuple[Any, ...]) -> Any:
    """Return H_eff = H - i/2 sum_mu J_mu^dagger J_mu."""
    effective = hamiltonian.copy()

    for jump in jumps:
        effective = effective - 0.5j * (jump.conj().T @ jump)

    return effective


def jump_probabilities(
    state: Any,
    jumps: list[Any] | tuple[Any, ...],
    step_size: float,
    *,
    backend: OpenSystemBackend,
) -> NDArray[np.float64]:
    """Return first-order jump probabilities dt <psi|J^dagger J|psi>."""
    array_module = backend.array_module
    probabilities: list[float] = []

    for jump in jumps:
        jumped_state = jump @ state
        rate = array_module.real(array_module.vdot(jumped_state, jumped_state))
        rate_float = float(backend.to_numpy(rate))
        probabilities.append(max(step_size * rate_float, 0.0))

    return np.asarray(probabilities, dtype=np.float64)


def choose_jump(
    probabilities: NDArray[np.float64],
    rng: np.random.Generator,
) -> int:
    """Choose a jump index according to normalized probabilities."""
    total_probability = float(np.sum(probabilities))

    if total_probability <= 0.0:
        raise ValueError("Total jump probability must be positive when choosing a jump.")

    return int(
        rng.choice(
            len(probabilities),
            p=probabilities / total_probability,
        )
    )


def evolve_no_jump_first_order(
    state: Any,
    effective_hamiltonian_matrix: Any,
    step_size: float,
) -> Any:
    """First-order no-jump evolution under H_eff."""
    return state - 1j * step_size * (effective_hamiltonian_matrix @ state)


def run_quantum_jump_trajectory(
    *,
    hamiltonian: Any,
    jumps: list[Any] | tuple[Any, ...],
    state_initial: Any,
    times: NDArray[np.float64],
    rng: np.random.Generator | int | None = None,
    backend: OpenSystemBackendName = "scipy",
    return_backend_arrays: bool = False,
    store_states: bool = True,
    normalize_each_step: bool = True,
    max_jump_probability: float = 0.1,
) -> TrajectoryResult:
    """Run one Monte Carlo wave-function trajectory.

    This uses a first-order no-jump propagator. It is therefore deliberately
    conservative: if the total jump probability in one time step is too large,
    it raises an error and asks the caller to refine the time grid.
    """
    times = np.asarray(times, dtype=np.float64)
    _validate_times_for_mcwf(times)

    generator = _rng_from_seed(rng)
    backend_obj = get_open_system_backend(backend)

    hamiltonian_backend = as_backend_dense_array(
        hamiltonian,
        backend=backend_obj,
        dtype=np.complex128,
    )
    jump_operators = [
        as_backend_dense_array(
            jump,
            backend=backend_obj,
            dtype=np.complex128,
        )
        for jump in jumps
    ]

    state = _normalize_backend_state(
        backend_obj.asarray(state_initial, dtype=np.complex128),
        backend=backend_obj,
    )

    effective_hamiltonian_matrix = effective_hamiltonian(
        hamiltonian_backend,
        jump_operators,
    )

    states: list[Any] = []
    if store_states:
        states.append(
            _maybe_to_numpy(state, backend=backend_obj, enabled=not return_backend_arrays)
        )

    jump_times: list[float] = []
    jump_indices: list[int] = []
    norm_errors: list[float] = []

    for time_index in range(times.size - 1):
        step_size = float(times[time_index + 1] - times[time_index])

        probabilities = jump_probabilities(
            state,
            jump_operators,
            step_size,
            backend=backend_obj,
        )
        total_jump_probability = float(np.sum(probabilities))

        if total_jump_probability > max_jump_probability:
            raise RuntimeError(
                "Time step is too large for first-order MCWF: "
                f"total jump probability={total_jump_probability:.6e}, "
                f"allowed maximum={max_jump_probability:.6e}. "
                "Use a finer time grid or increase max_jump_probability "
                "only if you know this is acceptable."
            )

        if generator.random() < total_jump_probability:
            jump_index = choose_jump(probabilities, generator)
            state = jump_operators[jump_index] @ state

            jump_times.append(float(times[time_index + 1]))
            jump_indices.append(jump_index)
        else:
            state = evolve_no_jump_first_order(
                state,
                effective_hamiltonian_matrix,
                step_size,
            )

        state_norm = _backend_state_norm(state, backend=backend_obj)
        norm_errors.append(abs(state_norm - 1.0))

        if normalize_each_step:
            if state_norm == 0.0:
                raise RuntimeError("The MCWF state reached zero norm. " "Try a smaller time step.")
            state = state / state_norm

        if store_states:
            states.append(
                _maybe_to_numpy(
                    state,
                    backend=backend_obj,
                    enabled=not return_backend_arrays,
                )
            )

    return TrajectoryResult(
        times=times,
        states=states,
        jump_times=np.asarray(jump_times, dtype=np.float64),
        jump_indices=np.asarray(jump_indices, dtype=np.int64),
        norm_errors=np.asarray(norm_errors, dtype=np.float64),
    )


def sample_lindblad_mcwf(
    *,
    hamiltonian: Any,
    jumps: list[Any] | tuple[Any, ...],
    times: NDArray[np.float64],
    state_initial: Any | None = None,
    state_sampler: StateSampler | None = None,
    options: McwfOptions | None = None,
    rng: np.random.Generator | int | None = None,
) -> EnsembleResult:
    """Estimate Lindblad evolution by averaging MCWF trajectories.

    Parameters
    ----------
    hamiltonian:
        Hamiltonian matrix.

    jumps:
        Lindblad jump operators.

    times:
        Strictly increasing time grid.

    state_initial:
        Fixed initial pure state. Used for every trajectory.

    state_sampler:
        Optional callable that samples an initial pure state from a NumPy RNG.
        Mutually exclusive with state_initial.

    options:
        MCWF options.

    rng:
        Optional RNG or seed. Overrides options.seed when provided.

    Returns
    -------
    EnsembleResult
        Contains ensemble-averaged density matrices rho_t and optionally the
        individual trajectories.
    """
    if options is None:
        options = McwfOptions()

    if options.n_trajectories <= 0:
        raise ValueError("options.n_trajectories must be positive.")

    if state_initial is not None and state_sampler is not None:
        raise ValueError("Pass only one of state_initial or state_sampler.")

    times = np.asarray(times, dtype=np.float64)
    _validate_times_for_mcwf(times)

    generator = _rng_from_seed(options.seed if rng is None else rng)
    backend_obj = get_open_system_backend(options.backend)

    hamiltonian_backend = as_backend_dense_array(
        hamiltonian,
        backend=backend_obj,
        dtype=np.complex128,
    )
    jump_operators = [
        as_backend_dense_array(
            jump,
            backend=backend_obj,
            dtype=np.complex128,
        )
        for jump in jumps
    ]

    dim = int(hamiltonian_backend.shape[0])

    if hamiltonian_backend.shape != (dim, dim):
        raise ValueError("hamiltonian must be a square matrix.")

    for jump in jump_operators:
        if jump.shape != (dim, dim):
            raise ValueError("Every jump operator must have shape (dim, dim).")

    rho_t = [np.zeros((dim, dim), dtype=np.complex128) for _ in range(times.size)]

    stored_trajectories: list[TrajectoryResult] | None
    if options.store_trajectories:
        stored_trajectories = []
    else:
        stored_trajectories = None

    child_seeds = generator.integers(
        low=0,
        high=np.iinfo(np.int64).max,
        size=options.n_trajectories,
        dtype=np.int64,
    )

    for child_seed in child_seeds:
        trajectory_rng = np.random.default_rng(int(child_seed))

        trajectory_state_initial = _initial_state_for_trajectory(
            dim=dim,
            state_initial=state_initial,
            state_sampler=state_sampler,
            rng=trajectory_rng,
        )

        trajectory = run_quantum_jump_trajectory(
            hamiltonian=hamiltonian_backend,
            jumps=jump_operators,
            state_initial=trajectory_state_initial,
            times=times,
            rng=trajectory_rng,
            backend=options.backend,
            return_backend_arrays=options.return_backend_arrays,
            store_states=True,
            normalize_each_step=options.normalize_each_step,
            max_jump_probability=options.max_jump_probability,
        )

        for time_index, state in enumerate(trajectory.states):
            state_numpy = _state_to_numpy(state, backend=backend_obj)
            rho_t[time_index] += projector(state_numpy)

        if stored_trajectories is not None:
            if options.store_states:
                stored_trajectories.append(trajectory)
            else:
                stored_trajectories.append(
                    TrajectoryResult(
                        times=trajectory.times,
                        states=[],
                        jump_times=trajectory.jump_times,
                        jump_indices=trajectory.jump_indices,
                        norm_errors=trajectory.norm_errors,
                    )
                )

    rho_t = [density_matrix / float(options.n_trajectories) for density_matrix in rho_t]

    return EnsembleResult(
        times=times,
        rho_t=rho_t,
        trajectories=(tuple(stored_trajectories) if stored_trajectories is not None else None),
    )


def observable_vs_time(
    rho_t: Iterable[ArrayC],
    observable: ArrayC,
) -> ArrayF:
    """Return Tr[rho(t) observable] for each density matrix."""
    observable = np.asarray(observable, dtype=np.complex128)

    values = [np.real_if_close(np.trace(density_matrix @ observable)) for density_matrix in rho_t]

    return np.asarray(values, dtype=np.float64)


def _validate_times_for_mcwf(times: NDArray[np.float64]) -> None:
    if times.ndim != 1 or times.size < 2:
        raise ValueError("times must be a one-dimensional array with at least two entries.")

    if not np.all(np.diff(times) > 0.0):
        raise ValueError("times must be strictly increasing.")


def _rng_from_seed(
    rng: np.random.Generator | int | None,
) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng

    return np.random.default_rng(rng)


def _initial_state_for_trajectory(
    *,
    dim: int,
    state_initial: Any | None,
    state_sampler: StateSampler | None,
    rng: np.random.Generator,
) -> ArrayC:
    if state_initial is not None:
        return np.asarray(state_initial, dtype=np.complex128)

    if state_sampler is not None:
        return np.asarray(state_sampler(rng), dtype=np.complex128)

    return random_pure_state(dim, rng=rng)


def _backend_state_norm(
    state: Any,
    *,
    backend: OpenSystemBackend,
) -> float:
    norm = backend.array_module.linalg.norm(state)
    return float(backend.to_numpy(norm))


def _normalize_backend_state(
    state: Any,
    *,
    backend: OpenSystemBackend,
) -> Any:
    norm = _backend_state_norm(state, backend=backend)

    if norm == 0.0:
        raise ValueError("state must be nonzero.")

    return state / norm


def _maybe_to_numpy(
    value: Any,
    *,
    backend: OpenSystemBackend,
    enabled: bool,
) -> Any:
    if not enabled:
        return value.copy()

    return backend.to_numpy(value.copy())


def _state_to_numpy(
    state: Any,
    *,
    backend: OpenSystemBackend,
) -> ArrayC:
    if isinstance(state, np.ndarray):
        return np.asarray(state, dtype=np.complex128)

    return np.asarray(backend.to_numpy(state), dtype=np.complex128)
