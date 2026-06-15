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


def test_sample_lindblad_mcwf_vectorized_adaptive_step_succeeds(qubit_ops):
    hamiltonian = np.zeros((2, 2), dtype=np.complex128)
    jump = np.sqrt(10.0) * qubit_ops["sigma_minus"]
    times = np.asarray([0.0, 0.2], dtype=np.float64)
    timing: dict[str, float] = {}

    result = sample_lindblad_mcwf(
        hamiltonian=hamiltonian,
        jumps=[jump],
        state_initial=qubit_ops["ket1"],
        times=times,
        options=McwfOptions(
            n_trajectories=4,
            seed=123,
            store_trajectories=False,
            store_density_matrices=False,
            max_jump_probability=0.1,
            adaptive_time_step=True,
            adaptive_safety_factor=0.8,
            timing_collector=timing,
        ),
    )

    assert result.trajectories is None
    assert result.rho_t == []
    assert timing["mcwf.rate_evaluation"] > 0.0
    assert timing["mcwf.count.adaptive_rate_reuses"] > 0.0
    assert timing["mcwf.count.grid_substeps"] > 0.0


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


def test_sample_lindblad_mcwf_rejects_nonpositive_event_segment_cap(qubit_ops):
    hamiltonian = np.zeros((2, 2), dtype=np.complex128)
    jumps: list[np.ndarray] = []
    times = np.array([0.0, 0.1], dtype=np.float64)

    with pytest.raises(ValueError, match="event_segment_probability_cap"):
        sample_lindblad_mcwf(
            hamiltonian=hamiltonian,
            jumps=jumps,
            state_initial=qubit_ops["ket0"],
            times=times,
            options=McwfOptions(
                n_trajectories=1,
                use_event_driven_jumps=True,
                event_segment_probability_cap=0.0,
            ),
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
    original_effective_hamiltonian = (
        stochastic_schrodinger._effective_hamiltonian_from_total_rate_operator
    )

    def counted_effective_hamiltonian(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_effective_hamiltonian(*args, **kwargs)

    monkeypatch.setattr(
        stochastic_schrodinger,
        "_effective_hamiltonian_from_total_rate_operator",
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


def test_sparse_jump_gram_sum_matches_sparse_matmul():
    import scipy.sparse as scipy_sparse

    from qlinks.open_system.stochastic_schrodinger import (
        _sparse_jump_gram_sum_csr,
    )

    dim = 16
    jump0 = scipy_sparse.csr_array(
        (
            np.array([1.0 + 2.0j, 3.0 - 1.0j, -0.5j], dtype=np.complex128),
            (np.array([2, 2, 7]), np.array([1, 4, 5])),
        ),
        shape=(dim, dim),
    )
    jump1 = scipy_sparse.csr_array(
        (
            np.array([0.25 + 0.1j, -1.0 + 0.2j], dtype=np.complex128),
            (np.array([3, 9]), np.array([8, 2])),
        ),
        shape=(dim, dim),
    )
    jumps = (jump0, jump1)

    actual = _sparse_jump_gram_sum_csr(jumps, shape=(dim, dim))
    assert actual is not None
    expected = sum((jump.conj().T @ jump for jump in jumps), scipy_sparse.csr_array((dim, dim)))

    np.testing.assert_allclose(actual.toarray(), expected.toarray(), atol=1e-14)


def test_sparse_jump_gram_sum_rejects_row_dense_jump():
    import scipy.sparse as scipy_sparse

    from qlinks.open_system.stochastic_schrodinger import (
        _sparse_jump_gram_sum_csr,
    )

    dim = 64
    jump = scipy_sparse.csr_array(
        (
            np.ones(dim, dtype=np.complex128),
            (np.zeros(dim, dtype=np.int64), np.arange(dim, dtype=np.int64)),
        ),
        shape=(dim, dim),
    )

    assert _sparse_jump_gram_sum_csr((jump,), shape=(dim, dim), max_row_nnz=32) is None


def test_effective_hamiltonian_sparse_many_jumps_matches_generic_path():
    import scipy.sparse as scipy_sparse

    dim = 16
    hamiltonian = scipy_sparse.csr_array(
        (
            np.array([0.2, -0.1j], dtype=np.complex128),
            (np.array([0, 5]), np.array([0, 3])),
        ),
        shape=(dim, dim),
    )
    jumps = (
        scipy_sparse.csr_array(
            (
                np.array([1.0 + 0.1j, -0.3j], dtype=np.complex128),
                (np.array([2, 2]), np.array([1, 4])),
            ),
            shape=(dim, dim),
        ),
        scipy_sparse.csr_array(
            (
                np.array([0.2 - 0.5j], dtype=np.complex128),
                (np.array([7]), np.array([5])),
            ),
            shape=(dim, dim),
        ),
    )

    actual = effective_hamiltonian(hamiltonian, jumps)
    expected = hamiltonian.copy()
    for jump in jumps:
        expected = expected - 0.5j * (jump.conj().T @ jump)

    assert scipy_sparse.issparse(actual)
    np.testing.assert_allclose(actual.toarray(), expected.toarray(), atol=1e-14)


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
        prefer_sparse_rate_evaluator=False,
    )

    assert prepared.uses_sparse_operators
    assert not prepared.uses_sparse_rate_evaluator
    assert scipy_sparse.issparse(prepared.hamiltonian)
    assert all(scipy_sparse.issparse(jump_operator) for jump_operator in prepared.jumps)
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
    np.testing.assert_array_equal(evaluator.row_columns[0][0], np.asarray([5]))
    np.testing.assert_allclose(evaluator.row_values[0][0], np.asarray([1.0 + 0.0j]))
    np.testing.assert_array_equal(evaluator.single_entry_columns[0], np.asarray([5]))
    np.testing.assert_allclose(evaluator.single_entry_weights[1], np.asarray([4.0]))


def test_sparse_jump_rate_evaluator_keeps_multi_entry_row_interference():
    import scipy.sparse as scipy_sparse

    from qlinks.open_system.stochastic_schrodinger import (
        _build_sparse_jump_rate_evaluator,
        _evaluate_sparse_jump_rates_numpy,
        _evaluate_sparse_jump_rates_state_matrix_numpy,
    )

    dim = 128
    jump = scipy_sparse.csr_array(
        (
            np.asarray([1.0 + 0.0j, 1.0 + 0.0j], dtype=np.complex128),
            (np.asarray([3, 3]), np.asarray([5, 6])),
        ),
        shape=(dim, dim),
        dtype=np.complex128,
    )
    evaluator = _build_sparse_jump_rate_evaluator((jump,))
    assert evaluator is not None
    assert evaluator.single_entry_columns[0] is None

    state = np.zeros(dim, dtype=np.complex128)
    state[5] = 1.0
    state[6] = 1.0
    states = np.column_stack([state, -state])

    # The two local amplitudes share one output row, so the rate is |1 + 1|^2,
    # not |1|^2 + |1|^2.
    np.testing.assert_allclose(_evaluate_sparse_jump_rates_numpy(state, evaluator), [4.0])
    np.testing.assert_allclose(
        _evaluate_sparse_jump_rates_state_matrix_numpy(states, evaluator),
        np.asarray([[4.0, 4.0]], dtype=np.float64),
    )


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


def test_sparse_jump_rate_evaluator_builds_single_entry_rate_matrix():
    import scipy.sparse as scipy_sparse

    from qlinks.open_system.stochastic_schrodinger import (
        _build_sparse_jump_rate_evaluator,
        _evaluate_jump_rates_state_matrix_numpy,
        _evaluate_sparse_jump_rates_numpy,
        _evaluate_sparse_jump_rates_state_matrix_numpy,
    )

    dim = 128
    jumps = tuple(
        scipy_sparse.csr_array(
            ([1.0 + 0.1j * index], ([index], [(index * 3) % dim])),
            shape=(dim, dim),
            dtype=np.complex128,
        )
        for index in range(16)
    )
    evaluator = _build_sparse_jump_rate_evaluator(jumps)
    assert evaluator is not None
    assert evaluator.single_entry_rate_matrix is not None
    assert evaluator.generic_jump_indices.size == 0

    rng = np.random.default_rng(123)
    state = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    states = rng.normal(size=(dim, 3)) + 1j * rng.normal(size=(dim, 3))

    expected_state = np.asarray(
        [max(float(np.vdot(jump @ state, jump @ state).real), 0.0) for jump in jumps],
        dtype=np.float64,
    )
    expected_matrix = _evaluate_jump_rates_state_matrix_numpy(states, jumps)

    np.testing.assert_allclose(_evaluate_sparse_jump_rates_numpy(state, evaluator), expected_state)
    np.testing.assert_allclose(
        _evaluate_sparse_jump_rates_state_matrix_numpy(states, evaluator),
        expected_matrix,
    )


def test_sparse_jump_rate_evaluator_builds_expanded_rate_operator_for_multi_entry_rows():
    import scipy.sparse as scipy_sparse

    from qlinks.open_system.stochastic_schrodinger import (
        _build_sparse_jump_rate_evaluator,
        _evaluate_jump_rates_state_matrix_numpy,
        _evaluate_sparse_jump_rates_numpy,
        _evaluate_sparse_jump_rates_state_matrix_numpy,
    )

    dim = 128
    jump0 = scipy_sparse.csr_array(
        (
            np.asarray([1.0 + 0.0j, 2.0j], dtype=np.complex128),
            (np.asarray([3, 3]), np.asarray([5, 6])),
        ),
        shape=(dim, dim),
        dtype=np.complex128,
    )
    jump1 = scipy_sparse.csr_array(
        (
            np.asarray([0.5 + 0.0j, -1.0 + 0.0j], dtype=np.complex128),
            (np.asarray([7, 9]), np.asarray([11, 12])),
        ),
        shape=(dim, dim),
        dtype=np.complex128,
    )
    jumps = (jump0, jump1)

    evaluator = _build_sparse_jump_rate_evaluator(jumps)
    assert evaluator is not None
    assert evaluator.expanded_rate_operator is not None
    np.testing.assert_array_equal(evaluator.expanded_rate_jump_indices, np.asarray([0, 1]))
    np.testing.assert_array_equal(evaluator.expanded_rate_row_splits, np.asarray([0, 1, 3]))

    rng = np.random.default_rng(123)
    state = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    states = rng.normal(size=(dim, 4)) + 1j * rng.normal(size=(dim, 4))

    expected_state = np.asarray(
        [max(float(np.vdot(jump @ state, jump @ state).real), 0.0) for jump in jumps],
        dtype=np.float64,
    )
    expected_matrix = _evaluate_jump_rates_state_matrix_numpy(states, jumps)

    np.testing.assert_allclose(_evaluate_sparse_jump_rates_numpy(state, evaluator), expected_state)
    np.testing.assert_allclose(
        _evaluate_sparse_jump_rates_state_matrix_numpy(states, evaluator),
        expected_matrix,
    )


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


def test_sample_lindblad_mcwf_populates_timing_collector(qubit_ops):
    timing: dict[str, float] = {}
    hamiltonian = 0.1 * qubit_ops["sigma_x"]
    jump = np.sqrt(0.2) * qubit_ops["sigma_minus"]
    times = np.linspace(0.0, 0.1, 4)

    sample_lindblad_mcwf(
        hamiltonian=hamiltonian,
        jumps=[jump],
        times=times,
        state_initial=qubit_ops["ket1"],
        options=McwfOptions(
            n_trajectories=4,
            seed=123,
            store_trajectories=False,
            timing_collector=timing,
        ),
    )

    assert timing["mcwf.operator_preparation"] >= 0.0
    assert timing["mcwf.initial_state_matrix"] >= 0.0
    assert timing["mcwf.rate_evaluation"] >= 0.0
    assert timing["mcwf.no_jump_propagation"] >= 0.0
    assert timing["mcwf.normalization"] >= 0.0
    assert timing["mcwf.density_accumulation"] >= 0.0


def test_sample_lindblad_mcwf_accumulates_existing_timing_collector_values(qubit_ops):
    timing: dict[str, float] = {"mcwf.rate_evaluation": 10.0}
    hamiltonian = np.zeros((2, 2), dtype=np.complex128)
    jump = np.sqrt(0.2) * qubit_ops["sigma_minus"]
    times = np.linspace(0.0, 0.1, 3)

    sample_lindblad_mcwf(
        hamiltonian=hamiltonian,
        jumps=[jump],
        times=times,
        state_initial=qubit_ops["ket1"],
        options=McwfOptions(
            n_trajectories=2,
            seed=123,
            store_trajectories=False,
            timing_collector=timing,
        ),
    )

    assert timing["mcwf.rate_evaluation"] >= 10.0


def test_sample_lindblad_mcwf_can_skip_density_matrices(qubit_ops):
    hamiltonian = np.zeros((2, 2), dtype=np.complex128)
    jump = np.sqrt(0.2) * qubit_ops["sigma_minus"]
    times = np.linspace(0.0, 0.1, 4)

    result = sample_lindblad_mcwf(
        hamiltonian=hamiltonian,
        jumps=[jump],
        times=times,
        state_initial=qubit_ops["ket1"],
        options=McwfOptions(
            n_trajectories=4,
            seed=123,
            store_trajectories=False,
            store_density_matrices=False,
        ),
    )

    assert result.rho_t == []
    assert result.state_snapshots is None


def test_density_matrix_from_state_matrix_validates_shape():
    from qlinks.open_system.stochastic_schrodinger import (
        density_matrix_from_state_matrix,
    )

    with pytest.raises(ValueError, match="2D array"):
        density_matrix_from_state_matrix(np.ones(2, dtype=np.complex128))

    with pytest.raises(ValueError, match="at least one trajectory"):
        density_matrix_from_state_matrix(np.zeros((2, 0), dtype=np.complex128))


def test_sample_lindblad_mcwf_state_snapshots_reconstruct_density(qubit_ops):
    from qlinks.open_system.stochastic_schrodinger import (
        density_matrix_from_state_matrix,
    )

    hamiltonian = 0.1 * qubit_ops["sigma_x"]
    jump = np.sqrt(0.2) * qubit_ops["sigma_minus"]
    times = np.linspace(0.0, 0.1, 4)

    result = sample_lindblad_mcwf(
        hamiltonian=hamiltonian,
        jumps=[jump],
        times=times,
        state_initial=qubit_ops["ket1"],
        options=McwfOptions(
            n_trajectories=4,
            seed=123,
            store_trajectories=False,
            store_density_matrices=True,
            store_state_snapshots=True,
        ),
    )

    assert result.state_snapshots is not None
    assert len(result.state_snapshots) == len(times)
    for snapshot, density_matrix in zip(result.state_snapshots, result.rho_t, strict=True):
        assert snapshot.shape == (2, 4)
        np.testing.assert_allclose(
            density_matrix_from_state_matrix(snapshot),
            density_matrix,
            atol=1e-14,
        )


def test_sample_lindblad_mcwf_state_snapshots_without_density(qubit_ops):
    hamiltonian = np.zeros((2, 2), dtype=np.complex128)
    jump = np.sqrt(0.2) * qubit_ops["sigma_minus"]
    times = np.linspace(0.0, 0.1, 3)

    result = sample_lindblad_mcwf(
        hamiltonian=hamiltonian,
        jumps=[jump],
        times=times,
        state_initial=qubit_ops["ket1"],
        options=McwfOptions(
            n_trajectories=3,
            seed=123,
            store_trajectories=False,
            store_density_matrices=False,
            store_state_snapshots=True,
        ),
    )

    assert result.rho_t == []
    assert result.state_snapshots is not None
    assert len(result.state_snapshots) == len(times)
    assert all(snapshot.shape == (2, 3) for snapshot in result.state_snapshots)


def test_sample_lindblad_mcwf_state_snapshots_nonvectorized(qubit_ops):
    hamiltonian = np.zeros((2, 2), dtype=np.complex128)
    jump = np.sqrt(0.2) * qubit_ops["sigma_minus"]
    times = np.linspace(0.0, 0.1, 3)

    result = sample_lindblad_mcwf(
        hamiltonian=hamiltonian,
        jumps=[jump],
        times=times,
        state_initial=qubit_ops["ket1"],
        options=McwfOptions(
            n_trajectories=2,
            seed=123,
            store_trajectories=True,
            store_states=False,
            store_density_matrices=False,
            store_state_snapshots=True,
        ),
    )

    assert result.rho_t == []
    assert result.trajectories is not None
    assert result.state_snapshots is not None
    assert len(result.state_snapshots) == len(times)
    assert all(snapshot.shape == (2, 2) for snapshot in result.state_snapshots)


def test_total_jump_rate_operator_matches_channel_rates():
    import scipy.sparse as scipy_sparse

    from qlinks.open_system.stochastic_schrodinger import (
        _evaluate_jump_rates_state_matrix_numpy,
        _evaluate_total_jump_rates_state_matrix_numpy,
        _total_jump_rate_operator,
    )

    dim = 16
    jump0 = scipy_sparse.csr_array(
        (
            np.asarray([1.0 + 0.0j, 0.5j], dtype=np.complex128),
            (np.asarray([2, 2]), np.asarray([3, 4])),
        ),
        shape=(dim, dim),
        dtype=np.complex128,
    )
    jump1 = scipy_sparse.csr_array(
        ([2.0 + 0.0j], ([5], [6])),
        shape=(dim, dim),
        dtype=np.complex128,
    )
    jumps = (jump0, jump1)
    rng = np.random.default_rng(123)
    states = rng.normal(size=(dim, 4)) + 1j * rng.normal(size=(dim, 4))

    gamma = _total_jump_rate_operator(jumps, shape=(dim, dim))
    assert gamma is not None

    expected = np.sum(_evaluate_jump_rates_state_matrix_numpy(states, jumps), axis=0)
    actual = _evaluate_total_jump_rates_state_matrix_numpy(states, gamma)

    np.testing.assert_allclose(actual, expected, atol=1e-14)


def test_vectorized_mcwf_total_rate_first_matches_channel_rate_path(qubit_ops):
    import scipy.sparse as scipy_sparse

    hamiltonian = scipy_sparse.csr_array(0.05 * qubit_ops["sigma_x"])
    jumps = [
        scipy_sparse.csr_array(np.sqrt(0.2) * qubit_ops["sigma_minus"]),
        scipy_sparse.csr_array(np.sqrt(0.1) * qubit_ops["sigma_z"]),
    ]
    times = np.linspace(0.0, 0.1, 6)

    baseline = sample_lindblad_mcwf(
        hamiltonian=hamiltonian,
        jumps=jumps,
        times=times,
        state_initial=qubit_ops["ket1"],
        options=McwfOptions(
            n_trajectories=32,
            seed=123,
            store_trajectories=False,
            prefer_sparse_operators=True,
            prefer_sparse_rate_evaluator=True,
            use_total_rate_first=False,
        ),
    )
    optimized = sample_lindblad_mcwf(
        hamiltonian=hamiltonian,
        jumps=jumps,
        times=times,
        state_initial=qubit_ops["ket1"],
        options=McwfOptions(
            n_trajectories=32,
            seed=123,
            store_trajectories=False,
            prefer_sparse_operators=True,
            prefer_sparse_rate_evaluator=True,
            use_total_rate_first=True,
        ),
    )

    for actual_rho, expected_rho in zip(optimized.rho_t, baseline.rho_t, strict=True):
        np.testing.assert_allclose(actual_rho, expected_rho, atol=1e-14)


def test_vectorized_mcwf_total_rate_first_skips_channel_rates_without_jumps(monkeypatch, qubit_ops):
    import scipy.sparse as scipy_sparse

    import qlinks.open_system.stochastic_schrodinger as stochastic_schrodinger

    calls = 0
    original = stochastic_schrodinger._evaluate_sparse_jump_rates_state_matrix_numpy

    def counted_channel_rates(states, evaluator):
        nonlocal calls
        calls += 1
        return original(states, evaluator)

    monkeypatch.setattr(
        stochastic_schrodinger,
        "_evaluate_sparse_jump_rates_state_matrix_numpy",
        counted_channel_rates,
    )
    monkeypatch.setattr(
        stochastic_schrodinger,
        "_should_use_total_rate_first",
        lambda *args, **kwargs: True,
    )

    hamiltonian = scipy_sparse.csr_array((2, 2), dtype=np.complex128)
    jump = scipy_sparse.csr_array(np.sqrt(0.2) * qubit_ops["sigma_minus"])
    times = np.linspace(0.0, 0.1, 5)

    result = sample_lindblad_mcwf(
        hamiltonian=hamiltonian,
        jumps=[jump],
        times=times,
        state_initial=qubit_ops["ket0"],
        options=McwfOptions(
            n_trajectories=8,
            seed=123,
            store_trajectories=False,
            store_density_matrices=False,
            prefer_sparse_operators=True,
            prefer_sparse_rate_evaluator=True,
            use_total_rate_first=True,
        ),
    )

    assert result.rho_t == []
    assert calls == 0


def test_sample_lindblad_mcwf_chunked_density_matches_unchunked(qubit_ops):
    hamiltonian = 0.05 * qubit_ops["sigma_x"]
    times = np.linspace(0.0, 0.1, 6)

    baseline = sample_lindblad_mcwf(
        hamiltonian=hamiltonian,
        jumps=[],
        times=times,
        state_initial=qubit_ops["ket1"],
        options=McwfOptions(
            n_trajectories=12,
            seed=123,
            store_trajectories=False,
            store_density_matrices=True,
            trajectory_chunk_size=None,
        ),
    )
    chunked = sample_lindblad_mcwf(
        hamiltonian=hamiltonian,
        jumps=[],
        times=times,
        state_initial=qubit_ops["ket1"],
        options=McwfOptions(
            n_trajectories=12,
            seed=123,
            store_trajectories=False,
            store_density_matrices=True,
            trajectory_chunk_size=5,
        ),
    )

    assert len(chunked.rho_t) == len(baseline.rho_t)
    for actual, expected in zip(chunked.rho_t, baseline.rho_t, strict=True):
        np.testing.assert_allclose(actual, expected, atol=1e-14)


def test_sample_lindblad_mcwf_chunked_state_snapshots_have_all_trajectories(qubit_ops):
    hamiltonian = np.zeros((2, 2), dtype=np.complex128)
    jump = np.sqrt(0.2) * qubit_ops["sigma_minus"]
    times = np.linspace(0.0, 0.1, 4)
    timing = {}

    result = sample_lindblad_mcwf(
        hamiltonian=hamiltonian,
        jumps=[jump],
        times=times,
        state_initial=qubit_ops["ket1"],
        options=McwfOptions(
            n_trajectories=10,
            seed=123,
            store_trajectories=False,
            store_density_matrices=False,
            store_state_snapshots=True,
            trajectory_chunk_size=4,
            timing_collector=timing,
        ),
    )

    assert result.rho_t == []
    assert result.state_snapshots is not None
    assert len(result.state_snapshots) == len(times)
    assert all(snapshot.shape == (2, 10) for snapshot in result.state_snapshots)
    assert timing["mcwf.chunk_merge"] >= 0.0


def test_mcwf_options_rejects_nonpositive_trajectory_chunk_size():
    with pytest.raises(ValueError, match="trajectory_chunk_size"):
        McwfOptions(trajectory_chunk_size=0).validate()


def test_sample_lindblad_mcwf_parallel_chunked_density_matches_unchunked(qubit_ops):
    hamiltonian = 0.05 * qubit_ops["sigma_x"]
    times = np.linspace(0.0, 0.1, 5)

    baseline = sample_lindblad_mcwf(
        hamiltonian=hamiltonian,
        jumps=[],
        times=times,
        state_initial=qubit_ops["ket1"],
        options=McwfOptions(
            n_trajectories=8,
            seed=123,
            store_trajectories=False,
            store_density_matrices=True,
            trajectory_chunk_size=None,
        ),
    )
    chunked = sample_lindblad_mcwf(
        hamiltonian=hamiltonian,
        jumps=[],
        times=times,
        state_initial=qubit_ops["ket1"],
        options=McwfOptions(
            n_trajectories=8,
            seed=123,
            store_trajectories=False,
            store_density_matrices=True,
            trajectory_chunk_size=4,
            trajectory_chunk_workers=2,
        ),
    )

    assert len(chunked.rho_t) == len(baseline.rho_t)
    for actual, expected in zip(chunked.rho_t, baseline.rho_t, strict=True):
        np.testing.assert_allclose(actual, expected, atol=1e-14)


def test_mcwf_options_rejects_nonpositive_trajectory_chunk_workers():
    with pytest.raises(ValueError, match="trajectory_chunk_workers"):
        McwfOptions(trajectory_chunk_workers=0).validate()


def test_sample_lindblad_mcwf_event_driven_no_jump_matches_fixed_grid(qubit_ops):
    hamiltonian = 0.05 * qubit_ops["sigma_x"]
    times = np.linspace(0.0, 0.2, 5)

    fixed = sample_lindblad_mcwf(
        hamiltonian=hamiltonian,
        jumps=[],
        times=times,
        state_initial=qubit_ops["ket1"],
        options=McwfOptions(
            n_trajectories=6,
            seed=123,
            store_trajectories=False,
            store_density_matrices=False,
            store_state_snapshots=True,
        ),
    )
    event_driven = sample_lindblad_mcwf(
        hamiltonian=hamiltonian,
        jumps=[],
        times=times,
        state_initial=qubit_ops["ket1"],
        options=McwfOptions(
            n_trajectories=6,
            seed=123,
            store_trajectories=False,
            store_density_matrices=False,
            store_state_snapshots=True,
            use_event_driven_jumps=True,
        ),
    )

    assert fixed.state_snapshots is not None
    assert event_driven.state_snapshots is not None
    for actual, expected in zip(
        event_driven.state_snapshots,
        fixed.state_snapshots,
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected, atol=1e-14)


def test_sample_lindblad_mcwf_event_driven_accepts_large_output_interval(qubit_ops):
    hamiltonian = np.zeros((2, 2), dtype=np.complex128)
    jump = np.sqrt(2.0) * qubit_ops["sigma_minus"]
    times = np.asarray([0.0, 2.0], dtype=np.float64)
    timing: dict[str, float] = {}

    result = sample_lindblad_mcwf(
        hamiltonian=hamiltonian,
        jumps=[jump],
        times=times,
        state_initial=qubit_ops["ket1"],
        options=McwfOptions(
            n_trajectories=16,
            seed=123,
            store_trajectories=False,
            store_density_matrices=False,
            store_state_snapshots=True,
            max_jump_probability=0.1,
            use_event_driven_jumps=True,
            timing_collector=timing,
        ),
    )

    assert result.rho_t == []
    assert result.state_snapshots is not None
    assert len(result.state_snapshots) == len(times)
    assert timing["mcwf.rate_evaluation"] > 0.0


def test_event_driven_survival_threshold_carries_across_output_intervals(qubit_ops):
    from qlinks.open_system.stochastic_schrodinger import (
        _advance_state_matrix_event_driven_numpy,
    )

    gamma = 0.1
    effective_hamiltonian = np.asarray(
        [[0.0, 0.0], [0.0, -0.5j * gamma]],
        dtype=np.complex128,
    )
    jump = np.sqrt(gamma) * qubit_ops["sigma_minus"]
    states = qubit_ops["ket1"].reshape(2, 1).copy()
    thresholds = np.asarray([0.5], dtype=np.float64)

    next_states, next_thresholds = _advance_state_matrix_event_driven_numpy(
        states=states,
        survival_thresholds=thresholds,
        step_size=1.0,
        effective_hamiltonian_matrix=effective_hamiltonian,
        total_jump_rate_operator=jump.conj().T @ jump,
        jump_operators=(jump,),
        sparse_jump_rate_evaluator=None,
        rng=np.random.default_rng(123),
        max_substeps=100,
        min_step_size=1.0e-12,
        max_jump_probability=10.0,
        adaptive_safety_factor=0.8,
        timing_collector={},
    )

    np.testing.assert_allclose(next_states[:, 0], qubit_ops["ket1"], atol=1e-14)
    np.testing.assert_allclose(next_thresholds, [0.5 / (1.0 - 0.5 * gamma) ** 2])


def test_event_driven_survival_threshold_jump_uses_norm_crossing(qubit_ops):
    from qlinks.open_system.stochastic_schrodinger import (
        _advance_state_matrix_event_driven_numpy,
    )

    gamma = 0.1
    effective_hamiltonian = np.asarray(
        [[0.0, 0.0], [0.0, -0.5j * gamma]],
        dtype=np.complex128,
    )
    jump = np.sqrt(gamma) * qubit_ops["sigma_minus"]
    states = qubit_ops["ket1"].reshape(2, 1).copy()
    thresholds = np.asarray([0.95], dtype=np.float64)

    next_states, _ = _advance_state_matrix_event_driven_numpy(
        states=states,
        survival_thresholds=thresholds,
        step_size=1.0,
        effective_hamiltonian_matrix=effective_hamiltonian,
        total_jump_rate_operator=jump.conj().T @ jump,
        jump_operators=(jump,),
        sparse_jump_rate_evaluator=None,
        rng=np.random.default_rng(123),
        max_substeps=100,
        min_step_size=1.0e-12,
        max_jump_probability=0.1,
        adaptive_safety_factor=0.8,
        timing_collector={},
    )

    np.testing.assert_allclose(next_states[:, 0], qubit_ops["ket0"], atol=1e-14)


def test_event_driven_substeps_zero_rate_hamiltonian_mixing(qubit_ops):
    from qlinks.open_system.stochastic_schrodinger import (
        _advance_state_matrix_event_driven_numpy,
    )

    jump_rate = 2.0
    hamiltonian = qubit_ops["sigma_x"]
    jump = np.sqrt(jump_rate) * qubit_ops["sigma_minus"]
    effective_hamiltonian = hamiltonian - 0.5j * (jump.conj().T @ jump)
    states = qubit_ops["ket0"].reshape(2, 1).copy()
    thresholds = np.asarray([0.95], dtype=np.float64)
    timing: dict[str, float] = {}

    _advance_state_matrix_event_driven_numpy(
        states=states,
        survival_thresholds=thresholds,
        step_size=2.0,
        effective_hamiltonian_matrix=effective_hamiltonian,
        total_jump_rate_operator=jump.conj().T @ jump,
        jump_operators=(jump,),
        sparse_jump_rate_evaluator=None,
        rng=np.random.default_rng(123),
        max_substeps=100_000,
        min_step_size=1.0e-12,
        max_jump_probability=0.1,
        adaptive_safety_factor=0.8,
        timing_collector=timing,
    )

    assert timing["mcwf.channel_rate_evaluation"] > 0.0
    assert timing["mcwf.selected_jump_application"] > 0.0


def test_event_driven_segment_limit_uses_jump_active_derivative_only():
    from qlinks.open_system.stochastic_schrodinger import _event_driven_segment_limits

    remaining_times = np.asarray([10.0, 10.0], dtype=np.float64)
    total_rates = np.asarray([0.0, 0.0], dtype=np.float64)
    jump_derivative_norms = np.asarray([0.0, 2.0], dtype=np.float64)

    segment_limits = _event_driven_segment_limits(
        remaining_times=remaining_times,
        total_rates=total_rates,
        jump_derivative_norms=jump_derivative_norms,
        max_jump_probability=0.1,
        adaptive_safety_factor=0.8,
    )

    assert segment_limits[0] == 10.0
    assert 0.0 < segment_limits[1] < 10.0


def test_jump_derivative_norms_ignore_dark_subspace_phase_rotation(qubit_ops):
    from qlinks.open_system.stochastic_schrodinger import (
        _jump_derivative_norms_state_matrix_numpy,
    )

    jump = qubit_ops["sigma_minus"]
    total_jump_rate_operator = jump.conj().T @ jump
    derivatives = (-1j * qubit_ops["ket0"]).reshape(2, 1)

    jump_derivative_norms = _jump_derivative_norms_state_matrix_numpy(
        derivatives,
        total_jump_rate_operator=total_jump_rate_operator,
        jump_operators=(jump,),
        sparse_jump_rate_evaluator=None,
        timing_collector={},
    )

    np.testing.assert_allclose(jump_derivative_norms, [0.0], atol=1e-14)


def test_sample_lindblad_mcwf_streams_target_fidelity_without_snapshots(qubit_ops):
    from qlinks.open_system.stochastic_schrodinger import (
        McwfOptions,
        sample_lindblad_mcwf,
    )

    times = np.linspace(0.0, 0.2, 3)
    options = McwfOptions(
        n_trajectories=4,
        seed=123,
        store_density_matrices=False,
        store_state_snapshots=False,
        fidelity_targets={"target": qubit_ops["ket1"]},
    )

    result = sample_lindblad_mcwf(
        hamiltonian=np.zeros((2, 2), dtype=np.complex128),
        jumps=[],
        state_initial=qubit_ops["ket1"],
        times=times,
        options=options,
    )

    assert result.rho_t == []
    assert result.state_snapshots is None
    assert result.target_fidelities is not None
    np.testing.assert_allclose(result.target_fidelities["target"], np.ones(times.size))


def test_sample_lindblad_mcwf_chunked_streamed_target_fidelity(qubit_ops):
    from qlinks.open_system.stochastic_schrodinger import (
        McwfOptions,
        sample_lindblad_mcwf,
    )

    times = np.linspace(0.0, 0.2, 3)
    options = McwfOptions(
        n_trajectories=6,
        seed=123,
        store_density_matrices=False,
        store_state_snapshots=False,
        trajectory_chunk_size=2,
        fidelity_targets={"target": qubit_ops["ket1"]},
    )

    result = sample_lindblad_mcwf(
        hamiltonian=np.zeros((2, 2), dtype=np.complex128),
        jumps=[],
        state_initial=qubit_ops["ket1"],
        times=times,
        options=options,
    )

    assert result.rho_t == []
    assert result.state_snapshots is None
    assert result.target_fidelities is not None
    np.testing.assert_allclose(result.target_fidelities["target"], np.ones(times.size))


def test_sample_lindblad_mcwf_event_krylov_no_jump_matches_unitary(qubit_ops):
    from qlinks.open_system.stochastic_schrodinger import (
        McwfOptions,
        sample_lindblad_mcwf,
    )

    times = np.asarray([0.0, np.pi / 2.0], dtype=np.float64)
    options = McwfOptions(
        n_trajectories=3,
        seed=123,
        store_density_matrices=False,
        store_state_snapshots=False,
        use_event_driven_jumps=True,
        event_no_jump_propagator="krylov",
        fidelity_targets={"target": qubit_ops["ket1"]},
    )

    result = sample_lindblad_mcwf(
        hamiltonian=qubit_ops["sigma_x"],
        jumps=[],
        state_initial=qubit_ops["ket0"],
        times=times,
        options=options,
    )

    assert result.target_fidelities is not None
    np.testing.assert_allclose(result.target_fidelities["target"][-1], 1.0, atol=1e-12)


def test_mcwf_options_reject_unknown_event_no_jump_propagator():
    from qlinks.open_system.stochastic_schrodinger import McwfOptions

    with pytest.raises(ValueError, match="event_no_jump_propagator"):
        McwfOptions(event_no_jump_propagator="bad").validate()
