import numpy as np
import pytest

from qlinks.open_system.states import density_matrix_from_state, normalize_state
from qlinks.open_system.stochastic_schrodinger import (
    EnsembleResult,
    McwfOptions,
    TrajectoryResult,
    choose_jump,
    effective_hamiltonian,
    evolve_no_jump_first_order,
    expectation,
    jump_probabilities,
    observable_vs_time,
    projector,
    run_quantum_jump_trajectory,
    sample_lindblad_mcwf,
)


@pytest.fixture
def qubit_ops():
    sigma_minus = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
    sigma_plus = sigma_minus.conj().T
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    identity = np.eye(2, dtype=np.complex128)

    ket0 = np.array([1.0, 0.0], dtype=np.complex128)
    ket1 = np.array([0.0, 1.0], dtype=np.complex128)

    return {
        "sigma_minus": sigma_minus,
        "sigma_plus": sigma_plus,
        "sigma_x": sigma_x,
        "sigma_z": sigma_z,
        "identity": identity,
        "ket0": ket0,
        "ket1": ket1,
    }


def test_projector_properties(qubit_ops):
    state = qubit_ops["ket1"]
    density_matrix = projector(state)

    expected = np.array(
        [[0.0, 0.0], [0.0, 1.0]],
        dtype=np.complex128,
    )

    assert density_matrix.shape == (2, 2)
    np.testing.assert_allclose(density_matrix, expected)
    np.testing.assert_allclose(density_matrix, density_matrix.conj().T)
    assert np.trace(density_matrix) == pytest.approx(1.0)


def test_density_matrix_from_state_matches_projector(qubit_ops):
    state = normalize_state(np.array([1.0, 1.0j], dtype=np.complex128))

    expected = projector(state)
    actual = density_matrix_from_state(state, normalize=False)

    np.testing.assert_allclose(actual, expected)


def test_expectation_matches_known_value(qubit_ops):
    value = expectation(
        qubit_ops["ket1"],
        qubit_ops["sigma_z"],
    )

    assert value == pytest.approx(-1.0)


def test_effective_hamiltonian_no_jumps_equals_hamiltonian(qubit_ops):
    hamiltonian = 0.5 * qubit_ops["sigma_x"]

    actual = effective_hamiltonian(hamiltonian, [])

    np.testing.assert_allclose(actual, hamiltonian)


def test_effective_hamiltonian_with_decay(qubit_ops):
    hamiltonian = np.zeros((2, 2), dtype=np.complex128)
    decay_rate = 2.0
    jump = np.sqrt(decay_rate) * qubit_ops["sigma_minus"]

    actual = effective_hamiltonian(hamiltonian, [jump])
    expected = hamiltonian - 0.5j * (jump.conj().T @ jump)

    np.testing.assert_allclose(actual, expected)

    expected_diag = np.array(
        [[0.0, 0.0], [0.0, -1.0j]],
        dtype=np.complex128,
    )
    np.testing.assert_allclose(actual, expected_diag)


def test_jump_probabilities_excited_state_decay(qubit_ops):
    from qlinks.open_system.backend import get_open_system_backend

    backend = get_open_system_backend("scipy")

    decay_rate = 3.0
    step_size = 0.1
    jump = np.sqrt(decay_rate) * qubit_ops["sigma_minus"]

    probabilities = jump_probabilities(
        qubit_ops["ket1"],
        [jump],
        step_size,
        backend=backend,
    )

    assert probabilities.shape == (1,)
    np.testing.assert_allclose(probabilities, [decay_rate * step_size])


def test_jump_probabilities_ground_state_zero(qubit_ops):
    from qlinks.open_system.backend import get_open_system_backend

    backend = get_open_system_backend("scipy")

    decay_rate = 3.0
    step_size = 0.1
    jump = np.sqrt(decay_rate) * qubit_ops["sigma_minus"]

    probabilities = jump_probabilities(
        qubit_ops["ket0"],
        [jump],
        step_size,
        backend=backend,
    )

    np.testing.assert_allclose(probabilities, [0.0])


def test_choose_jump_only_one_channel():
    rng = np.random.default_rng(123)
    probabilities = np.array([0.25], dtype=np.float64)

    for _ in range(10):
        assert choose_jump(probabilities, rng) == 0


def test_choose_jump_empirical_distribution():
    rng = np.random.default_rng(12345)
    probabilities = np.array([0.2, 0.3, 0.5], dtype=np.float64)

    counts = np.zeros(3, dtype=np.int64)
    n_samples = 20_000

    for _ in range(n_samples):
        jump_index = choose_jump(probabilities, rng)
        counts[jump_index] += 1

    frequencies = counts / n_samples
    target = probabilities / probabilities.sum()

    np.testing.assert_allclose(frequencies, target, atol=0.02)


def test_choose_jump_raises_when_total_probability_nonpositive():
    rng = np.random.default_rng(0)
    probabilities = np.array([0.0, 0.0], dtype=np.float64)

    with pytest.raises(ValueError, match="positive"):
        choose_jump(probabilities, rng)


def test_evolve_no_jump_first_order_identity_when_dt_zero(qubit_ops):
    state = qubit_ops["ket0"]
    hamiltonian = qubit_ops["sigma_x"]

    actual = evolve_no_jump_first_order(
        state,
        hamiltonian,
        0.0,
    )

    np.testing.assert_allclose(actual, state)


def test_evolve_no_jump_first_order_matches_manual_formula():
    effective = np.array(
        [[1.0, 0.0], [0.0, 2.0]],
        dtype=np.complex128,
    )
    state = np.array([1.0, 1.0j], dtype=np.complex128)
    step_size = 0.05

    expected = state - 1j * step_size * (effective @ state)
    actual = evolve_no_jump_first_order(state, effective, step_size)

    np.testing.assert_allclose(actual, expected)


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


def test_sample_lindblad_mcwf_returns_dataclass_with_fixed_state(qubit_ops):
    hamiltonian = np.zeros((2, 2), dtype=np.complex128)
    jumps: list[np.ndarray] = []
    times = np.linspace(0.0, 1.0, 6)

    result = sample_lindblad_mcwf(
        hamiltonian=hamiltonian,
        jumps=jumps,
        state_initial=qubit_ops["ket0"],
        times=times,
        options=McwfOptions(
            n_trajectories=5,
            seed=123,
            store_trajectories=False,
        ),
    )

    assert isinstance(result, EnsembleResult)
    np.testing.assert_allclose(result.times, times)
    assert len(result.rho_t) == len(times)
    assert result.trajectories is None


def test_sample_lindblad_mcwf_store_trajectories(qubit_ops):
    hamiltonian = np.zeros((2, 2), dtype=np.complex128)
    jumps: list[np.ndarray] = []
    times = np.linspace(0.0, 1.0, 4)

    result = sample_lindblad_mcwf(
        hamiltonian=hamiltonian,
        jumps=jumps,
        state_initial=qubit_ops["ket0"],
        times=times,
        options=McwfOptions(
            n_trajectories=3,
            seed=99,
            store_trajectories=True,
            store_states=True,
        ),
    )

    assert result.trajectories is not None
    assert len(result.trajectories) == 3

    for trajectory in result.trajectories:
        assert isinstance(trajectory, TrajectoryResult)
        assert len(trajectory.states) == len(times)


def test_sample_lindblad_mcwf_store_trajectories_without_states(qubit_ops):
    hamiltonian = np.zeros((2, 2), dtype=np.complex128)
    jumps: list[np.ndarray] = []
    times = np.linspace(0.0, 1.0, 4)

    result = sample_lindblad_mcwf(
        hamiltonian=hamiltonian,
        jumps=jumps,
        state_initial=qubit_ops["ket0"],
        times=times,
        options=McwfOptions(
            n_trajectories=3,
            seed=99,
            store_trajectories=True,
            store_states=False,
        ),
    )

    assert result.trajectories is not None
    assert len(result.trajectories) == 3

    for trajectory in result.trajectories:
        assert trajectory.states == []


def test_sample_lindblad_mcwf_accumulates_density_without_stored_states(qubit_ops):
    hamiltonian = 0.5 * qubit_ops["sigma_x"]
    jumps: list[np.ndarray] = []
    state_initial = qubit_ops["ket0"]
    times = np.linspace(0.0, 0.2, 5)

    expected_trajectory = run_quantum_jump_trajectory(
        hamiltonian=hamiltonian,
        jumps=jumps,
        state_initial=state_initial,
        times=times,
        rng=np.random.default_rng(11),
    )

    result = sample_lindblad_mcwf(
        hamiltonian=hamiltonian,
        jumps=jumps,
        state_initial=state_initial,
        times=times,
        options=McwfOptions(
            n_trajectories=1,
            seed=11,
            store_trajectories=True,
            store_states=False,
        ),
    )

    assert result.trajectories is not None
    assert len(result.trajectories) == 1
    assert result.trajectories[0].states == []

    for actual_density_matrix, expected_state in zip(
        result.rho_t,
        expected_trajectory.states,
    ):
        np.testing.assert_allclose(actual_density_matrix, projector(expected_state))


def test_sample_lindblad_mcwf_density_matrices_are_valid_for_trivial_case():
    hamiltonian = np.zeros((2, 2), dtype=np.complex128)
    jumps: list[np.ndarray] = []
    times = np.linspace(0.0, 1.0, 5)

    state = normalize_state(np.array([1.0, 1.0], dtype=np.complex128))

    result = sample_lindblad_mcwf(
        hamiltonian=hamiltonian,
        jumps=jumps,
        state_initial=state,
        times=times,
        options=McwfOptions(
            n_trajectories=10,
            seed=7,
            store_trajectories=False,
        ),
    )

    expected_density_matrix = projector(state)

    for density_matrix in result.rho_t:
        np.testing.assert_allclose(density_matrix, expected_density_matrix)
        np.testing.assert_allclose(density_matrix, density_matrix.conj().T)
        assert np.trace(density_matrix) == pytest.approx(1.0)


def test_sample_lindblad_mcwf_vectorized_fixed_state_no_trajectories(qubit_ops):
    hamiltonian = np.zeros((2, 2), dtype=np.complex128)
    jumps: list[np.ndarray] = []
    times = np.linspace(0.0, 0.4, 5)

    result = sample_lindblad_mcwf(
        hamiltonian=hamiltonian,
        jumps=jumps,
        state_initial=qubit_ops["ket0"],
        times=times,
        options=McwfOptions(
            n_trajectories=32,
            seed=123,
            store_trajectories=False,
        ),
    )

    assert result.trajectories is None
    for density_matrix in result.rho_t:
        np.testing.assert_allclose(density_matrix, projector(qubit_ops["ket0"]))


def test_sample_lindblad_mcwf_vectorized_raises_when_step_too_large(qubit_ops):
    hamiltonian = np.zeros((2, 2), dtype=np.complex128)
    jump = np.sqrt(10.0) * qubit_ops["sigma_minus"]
    times = np.asarray([0.0, 0.2], dtype=np.float64)

    with pytest.raises(RuntimeError, match="Time step is too large"):
        sample_lindblad_mcwf(
            hamiltonian=hamiltonian,
            jumps=[jump],
            state_initial=qubit_ops["ket1"],
            times=times,
            options=McwfOptions(
                n_trajectories=4,
                seed=123,
                store_trajectories=False,
                max_jump_probability=0.1,
            ),
        )


def test_sample_lindblad_mcwf_reproducible_with_seed(qubit_ops):
    hamiltonian = 0.5 * qubit_ops["sigma_x"]
    decay_rate = 0.4
    jumps = [np.sqrt(decay_rate) * qubit_ops["sigma_minus"]]
    times = np.linspace(0.0, 2.0, 51)

    options = McwfOptions(
        n_trajectories=200,
        seed=2024,
        store_trajectories=False,
    )

    result_1 = sample_lindblad_mcwf(
        hamiltonian=hamiltonian,
        jumps=jumps,
        state_initial=qubit_ops["ket1"],
        times=times,
        options=options,
    )
    result_2 = sample_lindblad_mcwf(
        hamiltonian=hamiltonian,
        jumps=jumps,
        state_initial=qubit_ops["ket1"],
        times=times,
        options=options,
    )

    for density_matrix_1, density_matrix_2 in zip(result_1.rho_t, result_2.rho_t):
        np.testing.assert_allclose(density_matrix_1, density_matrix_2)


def test_sample_lindblad_mcwf_accepts_state_sampler(qubit_ops):
    hamiltonian = np.zeros((2, 2), dtype=np.complex128)
    jumps: list[np.ndarray] = []
    times = np.array([0.0, 0.5, 1.0], dtype=np.float64)

    probability_ground = 0.7
    probability_excited = 0.3

    def state_sampler(rng: np.random.Generator):
        if rng.random() < probability_ground:
            return qubit_ops["ket0"]

        return qubit_ops["ket1"]

    result = sample_lindblad_mcwf(
        hamiltonian=hamiltonian,
        jumps=jumps,
        state_sampler=state_sampler,
        times=times,
        options=McwfOptions(
            n_trajectories=5000,
            seed=123,
            store_trajectories=False,
        ),
    )

    expected = np.array(
        [[probability_ground, 0.0], [0.0, probability_excited]],
        dtype=np.complex128,
    )

    for density_matrix in result.rho_t:
        np.testing.assert_allclose(density_matrix, expected, atol=0.03)


def test_sample_lindblad_mcwf_rejects_state_initial_and_sampler(qubit_ops):
    hamiltonian = np.zeros((2, 2), dtype=np.complex128)
    jumps: list[np.ndarray] = []
    times = np.array([0.0, 0.1], dtype=np.float64)

    def state_sampler(rng: np.random.Generator):
        return qubit_ops["ket0"]

    with pytest.raises(ValueError, match="only one"):
        sample_lindblad_mcwf(
            hamiltonian=hamiltonian,
            jumps=jumps,
            state_initial=qubit_ops["ket0"],
            state_sampler=state_sampler,
            times=times,
            options=McwfOptions(n_trajectories=1),
        )


def test_sample_lindblad_mcwf_rejects_nonpositive_trajectories(qubit_ops):
    hamiltonian = np.zeros((2, 2), dtype=np.complex128)
    jumps: list[np.ndarray] = []
    times = np.array([0.0, 0.1], dtype=np.float64)

    with pytest.raises(ValueError, match="positive"):
        sample_lindblad_mcwf(
            hamiltonian=hamiltonian,
            jumps=jumps,
            state_initial=qubit_ops["ket0"],
            times=times,
            options=McwfOptions(n_trajectories=0),
        )


def test_sample_lindblad_mcwf_decay_relaxes_toward_ground_state_on_average(qubit_ops):
    hamiltonian = np.zeros((2, 2), dtype=np.complex128)
    decay_rate = 1.0
    jumps = [np.sqrt(decay_rate) * qubit_ops["sigma_minus"]]
    times = np.linspace(0.0, 4.0, 201)

    result = sample_lindblad_mcwf(
        hamiltonian=hamiltonian,
        jumps=jumps,
        state_initial=qubit_ops["ket1"],
        times=times,
        options=McwfOptions(
            n_trajectories=500,
            seed=321,
            store_trajectories=False,
            max_jump_probability=0.1,
        ),
    )

    excited_projector = projector(qubit_ops["ket1"])
    excited_population = observable_vs_time(result.rho_t, excited_projector)

    assert excited_population[0] == pytest.approx(1.0, abs=1e-12)
    assert excited_population[-1] < 0.1
    assert excited_population[50] > excited_population[100] > excited_population[200]


def test_observable_vs_time_returns_correct_values(qubit_ops):
    density_matrix_0 = projector(qubit_ops["ket0"])
    density_matrix_1 = projector(qubit_ops["ket1"])
    density_matrix_plus = projector(normalize_state(np.array([1.0, 1.0], dtype=np.complex128)))

    values = observable_vs_time(
        [density_matrix_0, density_matrix_1, density_matrix_plus],
        qubit_ops["sigma_z"],
    )

    expected = np.array([1.0, -1.0, 0.0], dtype=np.float64)
    np.testing.assert_allclose(values, expected)


def test_observable_vs_time_shape(qubit_ops):
    rho_t = [projector(qubit_ops["ket0"]) for _ in range(7)]

    values = observable_vs_time(rho_t, qubit_ops["identity"])

    assert values.shape == (7,)
    np.testing.assert_allclose(values, 1.0)


def test_ensemble_trace_is_one(qubit_ops):
    hamiltonian = 0.25 * qubit_ops["sigma_x"]
    decay_rate = 0.7
    jumps = [np.sqrt(decay_rate) * qubit_ops["sigma_minus"]]
    times = np.linspace(0.0, 2.0, 101)

    result = sample_lindblad_mcwf(
        hamiltonian=hamiltonian,
        jumps=jumps,
        state_initial=qubit_ops["ket1"],
        times=times,
        options=McwfOptions(
            n_trajectories=500,
            seed=17,
            store_trajectories=False,
        ),
    )

    for density_matrix in result.rho_t:
        assert np.trace(density_matrix) == pytest.approx(1.0, abs=1e-12)


def test_ensemble_hermitian(qubit_ops):
    hamiltonian = 0.1 * qubit_ops["sigma_x"]
    decay_rate = 0.5
    jumps = [np.sqrt(decay_rate) * qubit_ops["sigma_minus"]]
    times = np.linspace(0.0, 1.0, 31)

    result = sample_lindblad_mcwf(
        hamiltonian=hamiltonian,
        jumps=jumps,
        state_initial=qubit_ops["ket1"],
        times=times,
        options=McwfOptions(
            n_trajectories=300,
            seed=101,
            store_trajectories=False,
        ),
    )

    for density_matrix in result.rho_t:
        np.testing.assert_allclose(
            density_matrix,
            density_matrix.conj().T,
            atol=1e-12,
        )


def test_sample_lindblad_mcwf_returns_requested_trajectories(qubit_ops):
    hamiltonian = np.zeros((2, 2), dtype=np.complex128)
    jump = qubit_ops["sigma_minus"]
    times = np.linspace(0.0, 0.1, 4)

    result = sample_lindblad_mcwf(
        hamiltonian=hamiltonian,
        jumps=[jump],
        state_initial=qubit_ops["ket1"],
        times=times,
        options=McwfOptions(
            backend="scipy",
            n_trajectories=3,
            seed=0,
            store_trajectories=True,
            store_states=True,
        ),
    )

    assert result.trajectories is not None
    assert len(result.trajectories) == 3
    assert result.times.shape == times.shape

    for trajectory in result.trajectories:
        assert len(trajectory.states) == len(times)


def test_run_quantum_jump_trajectory_cupy_backend_optional(qubit_ops):
    pytest.importorskip("cupy")
    pytest.importorskip("cupyx.scipy.sparse")

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


def test_sample_lindblad_mcwf_forwards_adaptive_options() -> None:
    hamiltonian = np.zeros((2, 2), dtype=np.complex128)
    jump = np.sqrt(10.0) * np.eye(2, dtype=np.complex128)
    state_initial = np.asarray([1.0, 0.0], dtype=np.complex128)
    times = np.asarray([0.0, 0.05], dtype=np.float64)

    result = sample_lindblad_mcwf(
        hamiltonian=hamiltonian,
        jumps=[jump],
        times=times,
        state_initial=state_initial,
        options=McwfOptions(
            n_trajectories=2,
            seed=1234,
            store_trajectories=True,
            store_states=True,
            max_jump_probability=0.1,
            adaptive_time_step=True,
        ),
    )

    assert len(result.rho_t) == len(times)
    assert result.trajectories is not None
    assert len(result.trajectories) == 2


def test_sample_lindblad_mcwf_reuses_prepared_effective_hamiltonian(monkeypatch, qubit_ops):
    import qlinks.open_system.stochastic_schrodinger as stochastic_schrodinger

    call_count = 0
    original_effective_hamiltonian = stochastic_schrodinger.effective_hamiltonian

    def counted_effective_hamiltonian(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_effective_hamiltonian(*args, **kwargs)

    monkeypatch.setattr(
        stochastic_schrodinger,
        "effective_hamiltonian",
        counted_effective_hamiltonian,
    )

    sample_lindblad_mcwf(
        hamiltonian=np.zeros((2, 2), dtype=np.complex128),
        jumps=[qubit_ops["sigma_minus"]],
        state_initial=qubit_ops["ket1"],
        times=np.linspace(0.0, 0.1, 3),
        options=McwfOptions(
            n_trajectories=4,
            seed=123,
            store_trajectories=False,
        ),
    )

    assert call_count == 1


@pytest.mark.manual
def test_sample_lindblad_mcwf_example_two_level_atom(qubit_ops):
    import matplotlib.pyplot as plt

    decay_rate = 1.0
    drive_strength = 0.5

    hamiltonian = 0.5 * drive_strength * qubit_ops["sigma_x"]
    jumps = [np.sqrt(decay_rate) * qubit_ops["sigma_minus"]]

    times = np.linspace(0.0, 10.0, 201)

    result = sample_lindblad_mcwf(
        hamiltonian=hamiltonian,
        jumps=jumps,
        state_initial=qubit_ops["ket1"],
        times=times,
        options=McwfOptions(
            n_trajectories=1000,
            seed=1234,
            store_trajectories=False,
            max_jump_probability=0.1,
        ),
    )

    excited_projector = projector(qubit_ops["ket1"])
    excited_population = observable_vs_time(result.rho_t, excited_projector)

    plt.figure(figsize=(7, 4.5))
    plt.plot(
        result.times,
        excited_population,
        label=r"$\langle 1|\rho(t)|1\rangle$",
        linestyle="--",
        marker="o",
    )
    plt.xlabel("t")
    plt.ylabel("Excited population")
    plt.legend()
    plt.tight_layout()
    plt.show()


def test_prepare_mcwf_operators_preserves_sparse_scipy_inputs(qubit_ops):
    import scipy.sparse as scipy_sparse

    from qlinks.open_system.stochastic_schrodinger import _prepare_mcwf_operators

    hamiltonian = scipy_sparse.csr_array(0.5 * qubit_ops["sigma_x"])
    jump = scipy_sparse.csr_array(np.sqrt(0.2) * qubit_ops["sigma_minus"])

    prepared = _prepare_mcwf_operators(
        hamiltonian=hamiltonian,
        jumps=[jump],
        backend="scipy",
        prefer_sparse_operators=True,
    )

    assert prepared.uses_sparse_operators
    assert not prepared.uses_sparse_rate_evaluator
    assert scipy_sparse.issparse(prepared.hamiltonian)
    assert scipy_sparse.issparse(prepared.jumps[0])
    assert scipy_sparse.issparse(prepared.effective_hamiltonian_matrix)


def test_vectorized_mcwf_sparse_matches_dense_fixed_seed(qubit_ops):
    import scipy.sparse as scipy_sparse

    hamiltonian = 0.5 * qubit_ops["sigma_x"]
    jump = np.sqrt(0.2) * qubit_ops["sigma_minus"]
    times = np.linspace(0.0, 0.2, 5)
    state_initial = normalize_state(np.array([1.0, 0.2j], dtype=np.complex128))

    dense_result = sample_lindblad_mcwf(
        hamiltonian=hamiltonian,
        jumps=[jump],
        times=times,
        state_initial=state_initial,
        options=McwfOptions(
            n_trajectories=16,
            seed=123,
            store_trajectories=False,
            prefer_sparse_operators=False,
        ),
    )
    sparse_result = sample_lindblad_mcwf(
        hamiltonian=scipy_sparse.csr_array(hamiltonian),
        jumps=[scipy_sparse.csr_array(jump)],
        times=times,
        state_initial=state_initial,
        options=McwfOptions(
            n_trajectories=16,
            seed=123,
            store_trajectories=False,
            prefer_sparse_operators=True,
        ),
    )

    for dense_rho, sparse_rho in zip(dense_result.rho_t, sparse_result.rho_t, strict=True):
        np.testing.assert_allclose(sparse_rho, dense_rho, atol=1e-14)


def test_jump_rates_state_matrix_do_not_require_retaining_jump_blocks(qubit_ops):
    import scipy.sparse as scipy_sparse

    from qlinks.open_system.stochastic_schrodinger import (
        _evaluate_jump_rates_state_matrix_numpy,
    )

    states = np.column_stack(
        [
            normalize_state(np.array([1.0, 0.0], dtype=np.complex128)),
            normalize_state(np.array([1.0, 1.0j], dtype=np.complex128)),
        ]
    )
    jumps = (
        scipy_sparse.csr_array(np.sqrt(0.2) * qubit_ops["sigma_minus"]),
        scipy_sparse.csr_array(np.sqrt(0.3) * qubit_ops["sigma_plus"]),
    )

    actual = _evaluate_jump_rates_state_matrix_numpy(states, jumps)
    expected = np.asarray(
        [np.einsum("ij,ij->j", (jump @ states).conj(), jump @ states).real for jump in jumps],
        dtype=np.float64,
    )

    np.testing.assert_allclose(actual, expected)


def test_build_sparse_jump_rate_evaluator_uses_row_sparse_jumps():
    import scipy.sparse as scipy_sparse

    from qlinks.open_system.stochastic_schrodinger import (
        _build_sparse_jump_rate_evaluator,
    )

    dim = 128
    jump0 = scipy_sparse.csr_array(
        ([1.0 + 0.0j], ([3], [5])),
        shape=(dim, dim),
        dtype=np.complex128,
    )
    jump1 = scipy_sparse.csr_array(
        ([2.0 + 0.0j], ([7], [11])),
        shape=(dim, dim),
        dtype=np.complex128,
    )

    evaluator = _build_sparse_jump_rate_evaluator((jump0, jump1))

    assert evaluator is not None
    assert evaluator.n_jumps == 2
    np.testing.assert_array_equal(evaluator.active_rows[0], np.asarray([3]))
    np.testing.assert_array_equal(evaluator.active_rows[1], np.asarray([7]))


def test_build_sparse_jump_rate_evaluator_rejects_row_dense_jumps():
    import scipy.sparse as scipy_sparse

    from qlinks.open_system.stochastic_schrodinger import (
        _build_sparse_jump_rate_evaluator,
    )

    jump = scipy_sparse.eye(8, format="csr", dtype=np.complex128)

    assert _build_sparse_jump_rate_evaluator((jump,)) is None


def test_sparse_jump_rate_evaluator_state_matrix_matches_full_sparse_rates():
    import scipy.sparse as scipy_sparse

    from qlinks.open_system.stochastic_schrodinger import (
        _build_sparse_jump_rate_evaluator,
        _evaluate_jump_rates_state_matrix_numpy,
        _evaluate_sparse_jump_rates_state_matrix_numpy,
    )

    dim = 128
    states = np.zeros((dim, 2), dtype=np.complex128)
    states[5, 0] = 1.0
    states[11, 1] = 1.0j
    jump0 = scipy_sparse.csr_array(
        ([1.0 + 0.0j], ([3], [5])),
        shape=(dim, dim),
        dtype=np.complex128,
    )
    jump1 = scipy_sparse.csr_array(
        ([2.0 + 0.0j], ([7], [11])),
        shape=(dim, dim),
        dtype=np.complex128,
    )
    jumps = (jump0, jump1)
    evaluator = _build_sparse_jump_rate_evaluator(jumps)
    assert evaluator is not None

    actual = _evaluate_sparse_jump_rates_state_matrix_numpy(states, evaluator)
    expected = _evaluate_jump_rates_state_matrix_numpy(states, jumps)

    np.testing.assert_allclose(actual, expected, atol=1e-14)


def test_vectorized_mcwf_sparse_rate_evaluator_matches_sparse_matmul(qubit_ops):
    import scipy.sparse as scipy_sparse

    dim = 128
    hamiltonian = scipy_sparse.csr_array((dim, dim), dtype=np.complex128)
    jumps = [
        scipy_sparse.csr_array(
            ([np.sqrt(0.2)], ([3], [5])),
            shape=(dim, dim),
            dtype=np.complex128,
        ),
        scipy_sparse.csr_array(
            ([np.sqrt(0.3)], ([7], [11])),
            shape=(dim, dim),
            dtype=np.complex128,
        ),
    ]
    state_initial = np.zeros(dim, dtype=np.complex128)
    state_initial[5] = 1.0
    times = np.linspace(0.0, 0.2, 5)

    baseline = sample_lindblad_mcwf(
        hamiltonian=hamiltonian,
        jumps=jumps,
        times=times,
        state_initial=state_initial,
        options=McwfOptions(
            n_trajectories=16,
            seed=123,
            store_trajectories=False,
            prefer_sparse_operators=True,
            prefer_sparse_rate_evaluator=False,
        ),
    )
    optimized = sample_lindblad_mcwf(
        hamiltonian=hamiltonian,
        jumps=jumps,
        times=times,
        state_initial=state_initial,
        options=McwfOptions(
            n_trajectories=16,
            seed=123,
            store_trajectories=False,
            prefer_sparse_operators=True,
            prefer_sparse_rate_evaluator=True,
        ),
    )

    for actual_rho, expected_rho in zip(optimized.rho_t, baseline.rho_t, strict=True):
        np.testing.assert_allclose(actual_rho, expected_rho, atol=1e-14)
