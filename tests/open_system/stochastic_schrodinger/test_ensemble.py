import numpy as np
import pytest

from qlinks.open_system.states import normalize_state
from qlinks.open_system.stochastic_schrodinger import (
    EnsembleResult,
    McwfOptions,
    TrajectoryResult,
    observable_vs_time,
    projector,
    run_quantum_jump_trajectory,
    sample_lindblad_mcwf,
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
