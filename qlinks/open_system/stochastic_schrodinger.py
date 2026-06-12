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
class _McwfPreparedOperators:
    backend: OpenSystemBackend
    hamiltonian: Any
    jumps: tuple[Any, ...]
    effective_hamiltonian_matrix: Any


@dataclass(frozen=True, slots=True)
class _JumpEvaluation:
    jumped_states: tuple[Any, ...]
    rates: NDArray[np.float64]

    @property
    def n_jumps(self) -> int:
        return int(self.rates.size)

    def probabilities(self, step_size: float) -> NDArray[np.float64]:
        if self.rates.size == 0:
            return np.zeros(0, dtype=np.float64)

        return np.maximum(step_size * self.rates, 0.0)


def _prepare_mcwf_operators(
    *,
    hamiltonian: Any,
    jumps: list[Any] | tuple[Any, ...],
    backend: OpenSystemBackendName | OpenSystemBackend = "scipy",
) -> _McwfPreparedOperators:
    backend_obj = get_open_system_backend(backend)
    hamiltonian_backend = as_backend_dense_array(
        hamiltonian,
        backend=backend_obj,
        dtype=np.complex128,
    )
    jump_operators = tuple(
        as_backend_dense_array(
            jump,
            backend=backend_obj,
            dtype=np.complex128,
        )
        for jump in jumps
    )
    effective_hamiltonian_matrix = effective_hamiltonian(
        hamiltonian_backend,
        jump_operators,
    )

    return _McwfPreparedOperators(
        backend=backend_obj,
        hamiltonian=hamiltonian_backend,
        jumps=jump_operators,
        effective_hamiltonian_matrix=effective_hamiltonian_matrix,
    )


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

    adaptive_time_step: bool = False
    adaptive_safety_factor: float = 0.8
    min_step_size: float = 1.0e-12
    max_substeps_per_interval: int = 100_000

    def validate(self) -> None:
        """Validate MCWF time-step control options."""
        if self.n_trajectories <= 0:
            raise ValueError("options.n_trajectories must be positive.")

        if self.max_jump_probability <= 0.0:
            raise ValueError("max_jump_probability must be positive.")

        if not (0.0 < self.adaptive_safety_factor <= 1.0):
            raise ValueError("adaptive_safety_factor must be in (0, 1].")

        if self.min_step_size <= 0.0:
            raise ValueError("min_step_size must be positive.")

        if self.max_substeps_per_interval <= 0:
            raise ValueError("max_substeps_per_interval must be positive.")

        if not self.adaptive_time_step and self.max_substeps_per_interval != 100_000:
            # Optional: probably do not need this check.
            pass


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
    return _evaluate_jumps(
        state,
        jumps,
        backend=backend,
    ).probabilities(step_size)


def _evaluate_jumps(
    state: Any,
    jumps: list[Any] | tuple[Any, ...],
    *,
    backend: OpenSystemBackend,
) -> _JumpEvaluation:
    """Return J_mu|psi> and rates ||J_mu|psi>||^2 for one MCWF state."""
    if not jumps:
        return _JumpEvaluation(
            jumped_states=(),
            rates=np.zeros(0, dtype=np.float64),
        )

    array_module = backend.array_module
    jumped_states: list[Any] = []
    rates: list[float] = []

    for jump in jumps:
        jumped_state = jump @ state
        rate = array_module.real(array_module.vdot(jumped_state, jumped_state))
        rate_float = float(backend.to_numpy(rate))
        jumped_states.append(jumped_state)
        rates.append(max(rate_float, 0.0))

    return _JumpEvaluation(
        jumped_states=tuple(jumped_states),
        rates=np.asarray(rates, dtype=np.float64),
    )


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
    adaptive_time_step: bool = False,
    adaptive_safety_factor: float = 0.8,
    min_step_size: float = 1.0e-12,
    max_substeps_per_interval: int = 100_000,
) -> TrajectoryResult:
    """Run one Monte Carlo wave-function trajectory.

    This uses a first-order no-jump propagator. It is therefore deliberately
    conservative: if the total jump probability in one time step is too large,
    it raises an error and asks the caller to refine the time grid.
    """
    prepared = _prepare_mcwf_operators(
        hamiltonian=hamiltonian,
        jumps=jumps,
        backend=backend,
    )
    return _run_quantum_jump_trajectory_prepared(
        prepared=prepared,
        state_initial=state_initial,
        times=times,
        rng=rng,
        return_backend_arrays=return_backend_arrays,
        store_states=store_states,
        normalize_each_step=normalize_each_step,
        max_jump_probability=max_jump_probability,
        adaptive_time_step=adaptive_time_step,
        adaptive_safety_factor=adaptive_safety_factor,
        min_step_size=min_step_size,
        max_substeps_per_interval=max_substeps_per_interval,
    )


def _run_quantum_jump_trajectory_prepared(
    *,
    prepared: _McwfPreparedOperators,
    state_initial: Any,
    times: NDArray[np.float64],
    rng: np.random.Generator | int | None = None,
    return_backend_arrays: bool = False,
    store_states: bool = True,
    state_callback: Callable[[int, Any], None] | None = None,
    normalize_each_step: bool = True,
    max_jump_probability: float = 0.1,
    adaptive_time_step: bool = False,
    adaptive_safety_factor: float = 0.8,
    min_step_size: float = 1.0e-12,
    max_substeps_per_interval: int = 100_000,
) -> TrajectoryResult:
    times = np.asarray(times, dtype=np.float64)
    _validate_times_for_mcwf(times)

    generator = _rng_from_seed(rng)
    backend_obj = prepared.backend
    jump_operators = prepared.jumps
    has_jump_operators = len(jump_operators) > 0

    state = _normalize_backend_state(
        backend_obj.asarray(state_initial, dtype=np.complex128),
        backend=backend_obj,
    )
    effective_hamiltonian_matrix = prepared.effective_hamiltonian_matrix

    states: list[Any] = []
    if store_states:
        states.append(
            _maybe_to_numpy(state, backend=backend_obj, enabled=not return_backend_arrays)
        )

    if state_callback is not None:
        state_callback(0, state)

    jump_times: list[float] = []
    jump_indices: list[int] = []
    norm_errors: list[float] = []

    for time_index in range(times.size - 1):
        interval_start = float(times[time_index])
        interval_stop = float(times[time_index + 1])
        current_time = interval_start
        substeps = 0

        while current_time < interval_stop:
            remaining_step = interval_stop - current_time
            step_size = remaining_step

            if has_jump_operators:
                jump_evaluation = _evaluate_jumps(
                    state,
                    jump_operators,
                    backend=backend_obj,
                )
                probabilities = jump_evaluation.probabilities(step_size)
                total_jump_probability = float(np.sum(probabilities))
            else:
                jump_evaluation = None
                probabilities = np.zeros(0, dtype=np.float64)
                total_jump_probability = 0.0

            if total_jump_probability > max_jump_probability:
                if not adaptive_time_step:
                    raise RuntimeError(
                        "Time step is too large for first-order MCWF: "
                        f"total jump probability={total_jump_probability:.6e}, "
                        f"allowed maximum={max_jump_probability:.6e}. "
                        "Use a finer time grid, enable adaptive_time_step=True, "
                        "or increase max_jump_probability only if you know this is "
                        "acceptable."
                    )

                scale = adaptive_safety_factor * max_jump_probability / total_jump_probability
                step_size = max(remaining_step * scale, min_step_size)

                if step_size >= remaining_step:
                    # Should only happen from roundoff, but keep the branch safe.
                    step_size = remaining_step

                assert jump_evaluation is not None
                probabilities = jump_evaluation.probabilities(step_size)
                total_jump_probability = float(np.sum(probabilities))

                if total_jump_probability > max_jump_probability:
                    raise RuntimeError(
                        "Adaptive MCWF failed to reduce the step enough: "
                        f"total jump probability={total_jump_probability:.6e}, "
                        f"allowed maximum={max_jump_probability:.6e}. "
                        "Try a smaller min_step_size or max_jump_probability."
                    )

            if step_size < min_step_size and remaining_step > min_step_size:
                raise RuntimeError(
                    "Adaptive MCWF reached min_step_size before completing "
                    "the requested time interval."
                )

            if jump_evaluation is not None and generator.random() < total_jump_probability:
                jump_index = choose_jump(probabilities, generator)
                state = jump_evaluation.jumped_states[jump_index]
                jump_times.append(float(current_time + step_size))
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
                    raise RuntimeError(
                        "The MCWF state reached zero norm. " "Try a smaller time step."
                    )
                state = state / state_norm

            current_time += step_size
            substeps += 1

            if substeps > max_substeps_per_interval:
                raise RuntimeError(
                    "Adaptive MCWF exceeded max_substeps_per_interval. "
                    "Try increasing max_jump_probability, increasing "
                    "max_substeps_per_interval, or checking jump rates."
                )

        if store_states:
            states.append(
                _maybe_to_numpy(
                    state,
                    backend=backend_obj,
                    enabled=not return_backend_arrays,
                )
            )

        if state_callback is not None:
            state_callback(time_index + 1, state)

    return TrajectoryResult(
        times=times,
        states=states,
        jump_times=np.asarray(jump_times, dtype=np.float64),
        jump_indices=np.asarray(jump_indices, dtype=np.int64),
        norm_errors=np.asarray(norm_errors, dtype=np.float64),
    )


def _can_use_vectorized_mcwf_ensemble(
    *,
    options: McwfOptions,
    backend: OpenSystemBackend,
) -> bool:
    """Return whether the ensemble can use the NumPy vectorized MCWF path."""
    return (
        backend.name == "scipy"
        and not options.store_trajectories
        and not options.adaptive_time_step
        and options.normalize_each_step
    )


def _sample_lindblad_mcwf_vectorized_scipy(
    *,
    prepared: _McwfPreparedOperators,
    dim: int,
    times: NDArray[np.float64],
    state_initial: Any | None,
    state_sampler: StateSampler | None,
    options: McwfOptions,
    rng: np.random.Generator,
) -> EnsembleResult:
    """Sample an MCWF ensemble with trajectory states stored as columns."""
    states = _initial_state_matrix_for_ensemble(
        dim=dim,
        n_trajectories=options.n_trajectories,
        state_initial=state_initial,
        state_sampler=state_sampler,
        rng=rng,
    )
    effective_hamiltonian_matrix = np.asarray(
        prepared.effective_hamiltonian_matrix,
        dtype=np.complex128,
    )
    jump_operators = tuple(np.asarray(jump, dtype=np.complex128) for jump in prepared.jumps)

    rho_t: list[ArrayC] = [_density_matrix_from_state_matrix(states)]

    for time_index in range(times.size - 1):
        step_size = float(times[time_index + 1] - times[time_index])

        jumped_states: tuple[ArrayC, ...] = ()
        probabilities = np.zeros((0, options.n_trajectories), dtype=np.float64)
        total_jump_probabilities = np.zeros(options.n_trajectories, dtype=np.float64)

        if jump_operators:
            jumped_states = tuple(jump @ states for jump in jump_operators)
            rates = np.asarray(
                [
                    np.einsum(
                        "ij,ij->j",
                        jumped_state.conj(),
                        jumped_state,
                    ).real
                    for jumped_state in jumped_states
                ],
                dtype=np.float64,
            )
            np.maximum(rates, 0.0, out=rates)
            probabilities = step_size * rates
            total_jump_probabilities = np.sum(probabilities, axis=0)

            max_total_jump_probability = float(np.max(total_jump_probabilities))
            if max_total_jump_probability > options.max_jump_probability:
                raise RuntimeError(
                    "Time step is too large for first-order MCWF: "
                    f"total jump probability={max_total_jump_probability:.6e}, "
                    f"allowed maximum={options.max_jump_probability:.6e}. "
                    "Use a finer time grid, enable adaptive_time_step=True, "
                    "or increase max_jump_probability only if you know this is "
                    "acceptable."
                )

        next_states = states - 1j * step_size * (effective_hamiltonian_matrix @ states)

        if jump_operators:
            jump_mask = rng.random(options.n_trajectories) < total_jump_probabilities
            if np.any(jump_mask):
                selected_jump_indices = _choose_jump_indices_vectorized(
                    probabilities=probabilities,
                    total_probabilities=total_jump_probabilities,
                    rng=rng,
                )
                for jump_index, jumped_state in enumerate(jumped_states):
                    selected_mask = jump_mask & (selected_jump_indices == jump_index)
                    if np.any(selected_mask):
                        next_states[:, selected_mask] = jumped_state[:, selected_mask]

        norms = np.linalg.norm(next_states, axis=0)
        if np.any(norms == 0.0):
            raise RuntimeError(
                "At least one MCWF state reached zero norm. " "Try a smaller time step."
            )

        states = next_states / norms.reshape(1, -1)
        rho_t.append(_density_matrix_from_state_matrix(states))

    return EnsembleResult(
        times=times,
        rho_t=rho_t,
        trajectories=None,
    )


def _choose_jump_indices_vectorized(
    *,
    probabilities: NDArray[np.float64],
    total_probabilities: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    cumulative_probabilities = np.cumsum(probabilities, axis=0)
    draws = rng.random(total_probabilities.size) * total_probabilities
    selected = np.sum(
        draws.reshape(1, -1) >= cumulative_probabilities,
        axis=0,
        dtype=np.int64,
    )
    return np.minimum(selected, probabilities.shape[0] - 1).astype(np.int64, copy=False)


def _initial_state_matrix_for_ensemble(
    *,
    dim: int,
    n_trajectories: int,
    state_initial: Any | None,
    state_sampler: StateSampler | None,
    rng: np.random.Generator,
) -> ArrayC:
    if state_initial is not None:
        state = _normalize_numpy_state(np.asarray(state_initial, dtype=np.complex128))
        return np.repeat(state.reshape(-1, 1), n_trajectories, axis=1)

    states = np.empty((dim, n_trajectories), dtype=np.complex128)
    child_seeds = rng.integers(
        low=0,
        high=np.iinfo(np.int64).max,
        size=n_trajectories,
        dtype=np.int64,
    )

    for trajectory_index, child_seed in enumerate(child_seeds):
        trajectory_rng = np.random.default_rng(int(child_seed))
        state = _initial_state_for_trajectory(
            dim=dim,
            state_initial=None,
            state_sampler=state_sampler,
            rng=trajectory_rng,
        )
        states[:, trajectory_index] = _normalize_numpy_state(state)

    return states


def _normalize_numpy_state(state: ArrayC) -> ArrayC:
    norm = float(np.linalg.norm(state))

    if norm == 0.0:
        raise ValueError("state must be nonzero.")

    return np.asarray(state / norm, dtype=np.complex128)


def _density_matrix_from_state_matrix(states: ArrayC) -> ArrayC:
    return (states @ states.conj().T) / float(states.shape[1])


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

    options.validate()

    if state_initial is not None and state_sampler is not None:
        raise ValueError("Pass only one of state_initial or state_sampler.")

    times = np.asarray(times, dtype=np.float64)
    _validate_times_for_mcwf(times)

    generator = _rng_from_seed(options.seed if rng is None else rng)
    prepared = _prepare_mcwf_operators(
        hamiltonian=hamiltonian,
        jumps=jumps,
        backend=options.backend,
    )
    backend_obj = prepared.backend
    hamiltonian_backend = prepared.hamiltonian

    dim = int(hamiltonian_backend.shape[0])

    if hamiltonian_backend.shape != (dim, dim):
        raise ValueError("hamiltonian must be a square matrix.")

    for jump in prepared.jumps:
        if jump.shape != (dim, dim):
            raise ValueError("Every jump operator must have shape (dim, dim).")

    if _can_use_vectorized_mcwf_ensemble(
        options=options,
        backend=backend_obj,
    ):
        return _sample_lindblad_mcwf_vectorized_scipy(
            prepared=prepared,
            dim=dim,
            times=times,
            state_initial=state_initial,
            state_sampler=state_sampler,
            options=options,
            rng=generator,
        )

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

    def accumulate_density_matrix(time_index: int, state: Any) -> None:
        state_numpy = _state_to_numpy(state, backend=backend_obj)
        rho_t[time_index] += projector(state_numpy)

    store_trajectory_states = bool(options.store_trajectories and options.store_states)

    for child_seed in child_seeds:
        trajectory_rng = np.random.default_rng(int(child_seed))

        trajectory_state_initial = _initial_state_for_trajectory(
            dim=dim,
            state_initial=state_initial,
            state_sampler=state_sampler,
            rng=trajectory_rng,
        )

        trajectory = _run_quantum_jump_trajectory_prepared(
            prepared=prepared,
            state_initial=trajectory_state_initial,
            times=times,
            rng=trajectory_rng,
            return_backend_arrays=options.return_backend_arrays,
            store_states=store_trajectory_states,
            state_callback=accumulate_density_matrix,
            normalize_each_step=options.normalize_each_step,
            max_jump_probability=options.max_jump_probability,
            adaptive_time_step=options.adaptive_time_step,
            adaptive_safety_factor=options.adaptive_safety_factor,
            min_step_size=options.min_step_size,
            max_substeps_per_interval=options.max_substeps_per_interval,
        )

        if stored_trajectories is not None:
            stored_trajectories.append(trajectory)

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
    if times.ndim != 1:
        raise ValueError("times must be a 1D array.")

    if times.size < 2:
        raise ValueError("times must contain at least two points.")

    if not np.all(np.isfinite(times)):
        raise ValueError("times must be finite.")

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
