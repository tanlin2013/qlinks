import numpy as np
import pytest

from qlinks.open_system.states import normalize_state
from qlinks.open_system.stochastic_schrodinger import TrajectoryResult, run_quantum_jump_trajectory
from tests.helpers.cupy import require_functional_cupy


def test_run_quantum_jump_trajectory_can_skip_state_storage(qubit_ops):
    hamiltonian = 0.5 * qubit_ops["sigma_x"]
    jump = np.sqrt(0.2) * qubit_ops["sigma_minus"]
    times = np.linspace(0.0, 0.5, 6)

    result = run_quantum_jump_trajectory(
        hamiltonian=hamiltonian,
        jumps=[jump],
        state_initial=qubit_ops["ket1"],
        times=times,
        rng=np.random.default_rng(0),
        store_states=False,
    )

    assert result.states == []
    np.testing.assert_allclose(result.times, times)
    assert result.norm_errors.shape == (len(times) - 1,)


def test_run_quantum_jump_trajectory_returns_dataclass(qubit_ops):
    hamiltonian = np.zeros((2, 2), dtype=np.complex128)
    jumps: list[np.ndarray] = []
    state_initial = qubit_ops["ket0"]
    times = np.linspace(0.0, 1.0, 6)

    result = run_quantum_jump_trajectory(
        hamiltonian=hamiltonian,
        jumps=jumps,
        state_initial=state_initial,
        times=times,
        rng=np.random.default_rng(1),
    )

    assert isinstance(result, TrajectoryResult)
    np.testing.assert_allclose(result.times, times)
    assert len(result.states) == len(times)
    assert result.jump_times.size == 0
    assert result.jump_indices.size == 0
    assert result.norm_errors.shape == (len(times) - 1,)


def test_run_quantum_jump_trajectory_preserves_state_without_h_or_jumps():
    hamiltonian = np.zeros((2, 2), dtype=np.complex128)
    jumps: list[np.ndarray] = []
    state_initial = normalize_state(np.array([1.0, 1.0], dtype=np.complex128))
    times = np.linspace(0.0, 1.0, 11)

    result = run_quantum_jump_trajectory(
        hamiltonian=hamiltonian,
        jumps=jumps,
        state_initial=state_initial,
        times=times,
        rng=np.random.default_rng(2),
    )

    for state in result.states:
        np.testing.assert_allclose(state, state_initial)
        assert np.linalg.norm(state) == pytest.approx(1.0)

    assert result.jump_times.size == 0
    assert result.jump_indices.size == 0


def test_run_quantum_jump_trajectory_allows_nonuniform_time_grid(qubit_ops):
    hamiltonian = np.zeros((2, 2), dtype=np.complex128)
    jumps: list[np.ndarray] = []
    state_initial = qubit_ops["ket0"]
    times = np.array([0.0, 0.1, 0.3, 0.6], dtype=np.float64)

    result = run_quantum_jump_trajectory(
        hamiltonian=hamiltonian,
        jumps=jumps,
        state_initial=state_initial,
        times=times,
        rng=np.random.default_rng(0),
    )

    assert len(result.states) == len(times)


def test_run_quantum_jump_trajectory_requires_strictly_increasing_times(qubit_ops):
    hamiltonian = np.zeros((2, 2), dtype=np.complex128)
    jumps: list[np.ndarray] = []
    state_initial = qubit_ops["ket0"]
    times = np.array([0.0, 0.1, 0.1], dtype=np.float64)

    with pytest.raises(ValueError, match="strictly increasing"):
        run_quantum_jump_trajectory(
            hamiltonian=hamiltonian,
            jumps=jumps,
            state_initial=state_initial,
            times=times,
            rng=np.random.default_rng(0),
        )


def test_run_quantum_jump_trajectory_requires_at_least_two_times(qubit_ops):
    hamiltonian = np.zeros((2, 2), dtype=np.complex128)
    jumps: list[np.ndarray] = []
    state_initial = qubit_ops["ket0"]
    times = np.array([0.0], dtype=np.float64)

    with pytest.raises(ValueError, match="at least two"):
        run_quantum_jump_trajectory(
            hamiltonian=hamiltonian,
            jumps=jumps,
            state_initial=state_initial,
            times=times,
            rng=np.random.default_rng(0),
        )


def test_run_quantum_jump_trajectory_raises_when_step_too_large(qubit_ops):
    hamiltonian = np.zeros((2, 2), dtype=np.complex128)
    decay_rate = 10.0
    jump = np.sqrt(decay_rate) * qubit_ops["sigma_minus"]
    state_initial = qubit_ops["ket1"]

    times = np.array([0.0, 0.2], dtype=np.float64)

    with pytest.raises(RuntimeError, match="Time step is too large"):
        run_quantum_jump_trajectory(
            hamiltonian=hamiltonian,
            jumps=[jump],
            state_initial=state_initial,
            times=times,
            rng=np.random.default_rng(0),
            max_jump_probability=0.1,
        )


def test_run_quantum_jump_trajectory_all_states_normalized(qubit_ops):
    hamiltonian = 0.5 * qubit_ops["sigma_x"]
    decay_rate = 0.3
    jump = np.sqrt(decay_rate) * qubit_ops["sigma_minus"]
    state_initial = qubit_ops["ket1"]
    times = np.linspace(0.0, 2.0, 101)

    result = run_quantum_jump_trajectory(
        hamiltonian=hamiltonian,
        jumps=[jump],
        state_initial=state_initial,
        times=times,
        rng=np.random.default_rng(42),
    )

    for state in result.states:
        assert np.linalg.norm(state) == pytest.approx(1.0, abs=1e-12)


def test_run_quantum_jump_trajectory_decay_jump_records_are_consistent(qubit_ops):
    hamiltonian = np.zeros((2, 2), dtype=np.complex128)
    decay_rate = 5.0
    jump = np.sqrt(decay_rate) * qubit_ops["sigma_minus"]
    state_initial = qubit_ops["ket1"]
    times = np.linspace(0.0, 5.0, 1001)

    result = run_quantum_jump_trajectory(
        hamiltonian=hamiltonian,
        jumps=[jump],
        state_initial=state_initial,
        times=times,
        rng=np.random.default_rng(123),
        max_jump_probability=0.1,
    )

    assert result.jump_times.size == result.jump_indices.size

    if result.jump_times.size > 0:
        first_jump_time = result.jump_times[0]
        jump_time_index = int(np.searchsorted(times, first_jump_time))

        for state in result.states[jump_time_index:]:
            np.testing.assert_allclose(
                state,
                qubit_ops["ket0"],
                atol=1e-12,
            )


@pytest.mark.gpu
def test_run_quantum_jump_trajectory_cupy_backend_optional(qubit_ops):
    require_functional_cupy()

    hamiltonian = np.zeros((2, 2), dtype=np.complex128)
    jumps: list[np.ndarray] = []
    times = np.linspace(0.0, 0.1, 3)

    result = run_quantum_jump_trajectory(
        hamiltonian=hamiltonian,
        jumps=jumps,
        state_initial=qubit_ops["ket0"],
        times=times,
        backend="cupy",
        return_backend_arrays=False,
    )

    assert len(result.states) == len(times)

    for state in result.states:
        assert isinstance(state, np.ndarray)


def test_quantum_jump_trajectory_raises_without_adaptive_step() -> None:
    hamiltonian = np.zeros((2, 2), dtype=np.complex128)
    jump = np.sqrt(10.0) * np.eye(2, dtype=np.complex128)
    state_initial = np.asarray([1.0, 0.0], dtype=np.complex128)
    times = np.asarray([0.0, 0.05], dtype=np.float64)

    with pytest.raises(RuntimeError, match="Time step is too large"):
        run_quantum_jump_trajectory(
            hamiltonian=hamiltonian,
            jumps=[jump],
            state_initial=state_initial,
            times=times,
            rng=1234,
            max_jump_probability=0.1,
            adaptive_time_step=False,
        )


def test_quantum_jump_trajectory_adaptive_step_succeeds() -> None:
    hamiltonian = np.zeros((2, 2), dtype=np.complex128)
    jump = np.sqrt(10.0) * np.eye(2, dtype=np.complex128)
    state_initial = np.asarray([1.0, 0.0], dtype=np.complex128)
    times = np.asarray([0.0, 0.05], dtype=np.float64)

    trajectory = run_quantum_jump_trajectory(
        hamiltonian=hamiltonian,
        jumps=[jump],
        state_initial=state_initial,
        times=times,
        rng=1234,
        max_jump_probability=0.1,
        adaptive_time_step=True,
        adaptive_safety_factor=0.8,
    )

    assert trajectory.times.shape == times.shape
    assert len(trajectory.states) == len(times)
    assert np.all(trajectory.jump_times >= times[0])
    assert np.all(trajectory.jump_times <= times[-1])
