import numpy as np
import pytest

from qlinks.open_sys.stochastic_schrodinger import (
    ArrayC,
    EnsembleResult,
    TrajectoryResult,
    _as_complex_array,
    choose_jump,
    effective_hamiltonian,
    evolve_no_jump_first_order,
    expectation,
    jump_probabilities,
    normalize_state,
    observable_vs_time,
    projector,
    run_quantum_jump_trajectory,
    sample_lindblad_mcwf,
)


@pytest.fixture
def qubit_ops():
    sigma_minus = np.array([[0, 1], [0, 0]], dtype=np.complex128)
    sigma_plus = sigma_minus.conj().T
    sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    ident = np.eye(2, dtype=np.complex128)

    ket0 = np.array([1.0, 0.0], dtype=np.complex128)
    ket1 = np.array([0.0, 1.0], dtype=np.complex128)

    return {
        "sigma_minus": sigma_minus,
        "sigma_plus": sigma_plus,
        "sigma_x": sigma_x,
        "sigma_z": sigma_z,
        "I": ident,
        "ket0": ket0,
        "ket1": ket1,
    }


def test_as_complex_array_casts_dtype():
    x = np.array([[1, 2], [3, 4]], dtype=np.float64)
    y = _as_complex_array(x)

    assert y.dtype == np.complex128
    assert np.allclose(y, x.astype(np.complex128))


def test_normalize_state_returns_unit_norm():
    psi = np.array([3.0 + 0.0j, 4.0 + 0.0j], dtype=np.complex128)
    psi_n = normalize_state(psi)

    assert np.isclose(np.linalg.norm(psi_n), 1.0)
    assert np.allclose(psi_n, psi / 5.0)


def test_normalize_state_raises_for_zero_vector():
    psi = np.zeros(4, dtype=np.complex128)
    with pytest.raises(ValueError, match="cannot normalize"):
        normalize_state(psi)


def test_projector_properties(qubit_ops):
    psi = qubit_ops["ket1"]
    rho = projector(psi)

    expected = np.array([[0, 0], [0, 1]], dtype=np.complex128)

    assert rho.shape == (2, 2)
    assert np.allclose(rho, expected)
    assert np.allclose(rho, rho.conj().T)
    assert np.isclose(np.trace(rho), 1.0)


def test_expectation_matches_known_value(qubit_ops):
    psi = qubit_ops["ket1"]
    sigma_z = qubit_ops["sigma_z"]

    val = expectation(psi, sigma_z)
    assert np.isclose(val, -1.0)


def test_effective_hamiltonian_no_jumps_equals_h(qubit_ops):
    H = 0.5 * qubit_ops["sigma_x"]
    H_eff = effective_hamiltonian(H, [])

    assert np.allclose(H_eff, H)


def test_effective_hamiltonian_with_decay(qubit_ops):
    H = np.zeros((2, 2), dtype=np.complex128)
    gamma = 2.0
    L = np.sqrt(gamma) * qubit_ops["sigma_minus"]

    H_eff = effective_hamiltonian(H, [L])

    expected = H - 0.5j * (L.conj().T @ L)
    assert np.allclose(H_eff, expected)

    # For sigma_- , L^\dagger L = gamma |1><1|
    expected_diag = np.array([[0.0, 0.0], [0.0, -1.0j]], dtype=np.complex128)
    assert np.allclose(H_eff, expected_diag)


def test_jump_probabilities_excited_state_decay(qubit_ops):
    gamma = 3.0
    dt = 0.1
    L = np.sqrt(gamma) * qubit_ops["sigma_minus"]

    psi = qubit_ops["ket1"]
    probs = jump_probabilities(psi, [L], dt)

    assert probs.shape == (1,)
    assert np.allclose(probs, [gamma * dt])


def test_jump_probabilities_ground_state_zero(qubit_ops):
    gamma = 3.0
    dt = 0.1
    L = np.sqrt(gamma) * qubit_ops["sigma_minus"]

    psi = qubit_ops["ket0"]
    probs = jump_probabilities(psi, [L], dt)

    assert np.allclose(probs, [0.0])


def test_choose_jump_only_one_channel():
    rng = np.random.default_rng(123)
    probs = np.array([0.25], dtype=np.float64)

    for _ in range(10):
        assert choose_jump(probs, rng) == 0


def test_choose_jump_empirical_distribution():
    rng = np.random.default_rng(12345)
    probs = np.array([0.2, 0.3, 0.5], dtype=np.float64)

    counts = np.zeros(3, dtype=int)
    n_samples = 20000
    for _ in range(n_samples):
        idx = choose_jump(probs, rng)
        counts[idx] += 1

    freq = counts / n_samples
    target = probs / probs.sum()

    assert np.allclose(freq, target, atol=0.02)


def test_choose_jump_raises_when_total_probability_nonpositive():
    rng = np.random.default_rng(0)
    probs = np.array([0.0, 0.0], dtype=np.float64)

    with pytest.raises(ValueError, match="must be positive"):
        choose_jump(probs, rng)


def test_evolve_no_jump_first_order_identity_when_dt_zero(qubit_ops):
    H = qubit_ops["sigma_x"]
    psi = qubit_ops["ket0"]

    psi_new = evolve_no_jump_first_order(psi, H, dt=0.0)
    assert np.allclose(psi_new, psi)


def test_evolve_no_jump_first_order_matches_manual_formula(qubit_ops):
    H_eff = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.complex128)
    psi = np.array([1.0, 1.0j], dtype=np.complex128)
    dt = 0.05

    expected = psi - 1j * dt * (H_eff @ psi)
    actual = evolve_no_jump_first_order(psi, H_eff, dt)

    assert np.allclose(actual, expected)


def test_run_quantum_jump_trajectory_returns_dataclass(qubit_ops):
    H = np.zeros((2, 2), dtype=np.complex128)
    jumps = []
    psi0 = qubit_ops["ket0"]
    times = np.linspace(0.0, 1.0, 6)

    result = run_quantum_jump_trajectory(H, jumps, psi0, times, rng=np.random.default_rng(1))

    assert isinstance(result, TrajectoryResult)
    assert np.allclose(result.times, times)
    assert len(result.states) == len(times)
    assert isinstance(result.jump_records, list)


def test_run_quantum_jump_trajectory_preserves_state_without_h_or_jumps(qubit_ops):
    H = np.zeros((2, 2), dtype=np.complex128)
    jumps = []
    psi0 = normalize_state(np.array([1.0, 1.0], dtype=np.complex128))
    times = np.linspace(0.0, 1.0, 11)

    result = run_quantum_jump_trajectory(H, jumps, psi0, times, rng=np.random.default_rng(2))

    for psi in result.states:
        assert np.allclose(psi, psi0)
        assert np.isclose(np.linalg.norm(psi), 1.0)

    assert result.jump_records == []


def test_run_quantum_jump_trajectory_requires_uniform_time_grid(qubit_ops):
    H = np.zeros((2, 2), dtype=np.complex128)
    jumps = []
    psi0 = qubit_ops["ket0"]
    times = np.array([0.0, 0.1, 0.3, 0.6], dtype=np.float64)

    with pytest.raises(ValueError, match="uniform time step"):
        run_quantum_jump_trajectory(H, jumps, psi0, times, rng=np.random.default_rng(0))


def test_run_quantum_jump_trajectory_requires_at_least_two_times(qubit_ops):
    H = np.zeros((2, 2), dtype=np.complex128)
    jumps = []
    psi0 = qubit_ops["ket0"]
    times = np.array([0.0], dtype=np.float64)

    with pytest.raises(ValueError, match="at least two points"):
        run_quantum_jump_trajectory(H, jumps, psi0, times, rng=np.random.default_rng(0))


def test_run_quantum_jump_trajectory_raises_when_dt_too_large(qubit_ops):
    H = np.zeros((2, 2), dtype=np.complex128)
    gamma = 10.0
    L = np.sqrt(gamma) * qubit_ops["sigma_minus"]
    psi0 = qubit_ops["ket1"]

    # dt = 0.2 -> p_jump = gamma * dt = 2.0 > 1
    times = np.array([0.0, 0.2], dtype=np.float64)

    with pytest.raises(RuntimeError, match="Time step too large"):
        run_quantum_jump_trajectory(H, [L], psi0, times, rng=np.random.default_rng(0))


def test_run_quantum_jump_trajectory_all_states_normalized(qubit_ops):
    H = 0.5 * qubit_ops["sigma_x"]
    gamma = 0.3
    L = np.sqrt(gamma) * qubit_ops["sigma_minus"]
    psi0 = qubit_ops["ket1"]
    times = np.linspace(0.0, 2.0, 101)

    result = run_quantum_jump_trajectory(H, [L], psi0, times, rng=np.random.default_rng(42))

    for psi in result.states:
        assert np.isclose(np.linalg.norm(psi), 1.0, atol=1e-12)


def test_run_quantum_jump_trajectory_decay_only_first_jump_stays_in_ground(qubit_ops):
    H = np.zeros((2, 2), dtype=np.complex128)
    gamma = 5.0
    L = np.sqrt(gamma) * qubit_ops["sigma_minus"]
    psi0 = qubit_ops["ket1"]
    times = np.linspace(0.0, 5.0, 1001)

    result = run_quantum_jump_trajectory(H, [L], psi0, times, rng=np.random.default_rng(123))

    # After a jump in pure decay with H=0, state should become |0>,
    # and then remain there because further jump probability is zero.
    if result.jump_records:
        first_jump_time, _ = result.jump_records[0]
        jump_index = np.searchsorted(times, first_jump_time)

        for psi in result.states[jump_index:]:
            assert np.allclose(psi, qubit_ops["ket0"], atol=1e-12)


def test_sample_lindblad_mcwf_returns_dataclass(qubit_ops):
    H = np.zeros((2, 2), dtype=np.complex128)
    jumps = []
    times = np.linspace(0.0, 1.0, 6)

    def psi0_sampler(rng):
        return qubit_ops["ket0"]

    result = sample_lindblad_mcwf(
        H=H,
        jumps=jumps,
        psi0_sampler=psi0_sampler,
        times=times,
        n_trajectories=5,
        seed=123,
        store_trajectories=False,
    )

    assert isinstance(result, EnsembleResult)
    assert np.allclose(result.times, times)
    assert len(result.rho_t) == len(times)
    assert result.trajectories is None


def test_sample_lindblad_mcwf_store_trajectories(qubit_ops):
    H = np.zeros((2, 2), dtype=np.complex128)
    jumps = []
    times = np.linspace(0.0, 1.0, 4)

    def psi0_sampler(rng):
        return qubit_ops["ket0"]

    result = sample_lindblad_mcwf(
        H=H,
        jumps=jumps,
        psi0_sampler=psi0_sampler,
        times=times,
        n_trajectories=3,
        seed=99,
        store_trajectories=True,
    )

    assert result.trajectories is not None
    assert len(result.trajectories) == 3
    assert all(isinstance(traj, TrajectoryResult) for traj in result.trajectories)


def test_sample_lindblad_mcwf_density_matrices_are_valid_for_trivial_case(qubit_ops):
    H = np.zeros((2, 2), dtype=np.complex128)
    jumps = []
    times = np.linspace(0.0, 1.0, 5)

    psi = normalize_state(np.array([1.0, 1.0], dtype=np.complex128))

    def psi0_sampler(rng):
        return psi

    result = sample_lindblad_mcwf(
        H=H,
        jumps=jumps,
        psi0_sampler=psi0_sampler,
        times=times,
        n_trajectories=10,
        seed=7,
        store_trajectories=False,
    )

    expected_rho = projector(psi)
    for rho in result.rho_t:
        assert np.allclose(rho, expected_rho)
        assert np.allclose(rho, rho.conj().T)
        assert np.isclose(np.trace(rho), 1.0)


def test_sample_lindblad_mcwf_reproducible_with_seed(qubit_ops):
    H = 0.5 * qubit_ops["sigma_x"]
    gamma = 0.4
    jumps = [np.sqrt(gamma) * qubit_ops["sigma_minus"]]
    times = np.linspace(0.0, 2.0, 51)

    def psi0_sampler(rng):
        return qubit_ops["ket1"]

    result1 = sample_lindblad_mcwf(
        H=H,
        jumps=jumps,
        psi0_sampler=psi0_sampler,
        times=times,
        n_trajectories=200,
        seed=2024,
        store_trajectories=False,
    )
    result2 = sample_lindblad_mcwf(
        H=H,
        jumps=jumps,
        psi0_sampler=psi0_sampler,
        times=times,
        n_trajectories=200,
        seed=2024,
        store_trajectories=False,
    )

    for rho1, rho2 in zip(result1.rho_t, result2.rho_t):
        assert np.allclose(rho1, rho2)


def test_sample_lindblad_mcwf_mixed_initial_sampler_average(qubit_ops):
    H = np.zeros((2, 2), dtype=np.complex128)
    jumps = []
    times = np.array([0.0, 0.5, 1.0], dtype=np.float64)

    p0 = 0.7
    p1 = 0.3

    def psi0_sampler(rng):
        return qubit_ops["ket0"] if rng.random() < p0 else qubit_ops["ket1"]

    result = sample_lindblad_mcwf(
        H=H,
        jumps=jumps,
        psi0_sampler=psi0_sampler,
        times=times,
        n_trajectories=5000,
        seed=123,
        store_trajectories=False,
    )

    expected = np.array([[p0, 0.0], [0.0, p1]], dtype=np.complex128)

    for rho in result.rho_t:
        assert np.allclose(rho, expected, atol=0.03)


def test_sample_lindblad_mcwf_decay_relaxes_toward_ground_state_on_average(qubit_ops):
    H = np.zeros((2, 2), dtype=np.complex128)
    gamma = 1.0
    jumps = [np.sqrt(gamma) * qubit_ops["sigma_minus"]]
    times = np.linspace(0.0, 4.0, 401)

    def psi0_sampler(rng):
        return qubit_ops["ket1"]

    result = sample_lindblad_mcwf(
        H=H,
        jumps=jumps,
        psi0_sampler=psi0_sampler,
        times=times,
        n_trajectories=4000,
        seed=321,
        store_trajectories=False,
    )

    P_exc = projector(qubit_ops["ket1"])
    excited_pop = observable_vs_time(result.rho_t, P_exc)

    # Starts near 1, ends much smaller
    assert excited_pop[0] == pytest.approx(1.0, abs=1e-12)
    assert excited_pop[-1] < 0.1

    # Should be broadly decreasing; allow for Monte Carlo noise
    assert excited_pop[50] > excited_pop[150] > excited_pop[300]


def test_observable_vs_time_returns_correct_values(qubit_ops):
    rho0 = projector(qubit_ops["ket0"])
    rho1 = projector(qubit_ops["ket1"])
    rho_plus = projector(normalize_state(np.array([1.0, 1.0], dtype=np.complex128)))

    sigma_z = qubit_ops["sigma_z"]
    vals = observable_vs_time([rho0, rho1, rho_plus], sigma_z)

    expected = np.array([1.0, -1.0, 0.0], dtype=np.float64)
    assert np.allclose(vals, expected)


def test_observable_vs_time_shape(qubit_ops):
    rho_t = [projector(qubit_ops["ket0"]) for _ in range(7)]
    O = qubit_ops["I"]

    vals = observable_vs_time(rho_t, O)
    assert vals.shape == (7,)
    assert np.allclose(vals, 1.0)


def test_trajectory_with_no_jumps_matches_manual_no_jump_evolution(qubit_ops):
    # No jumps, H=0 => state constant exactly
    H = np.zeros((2, 2), dtype=np.complex128)
    jumps = []
    psi0 = normalize_state(np.array([1.0, 2.0j], dtype=np.complex128))
    times = np.linspace(0.0, 1.0, 11)

    result = run_quantum_jump_trajectory(H, jumps, psi0, times, rng=np.random.default_rng(11))

    for psi in result.states:
        assert np.allclose(psi, psi0)


def test_ensemble_trace_is_one(qubit_ops):
    H = 0.25 * qubit_ops["sigma_x"]
    gamma = 0.7
    jumps = [np.sqrt(gamma) * qubit_ops["sigma_minus"]]
    times = np.linspace(0.0, 2.0, 101)

    def psi0_sampler(rng):
        return qubit_ops["ket1"]

    result = sample_lindblad_mcwf(
        H=H,
        jumps=jumps,
        psi0_sampler=psi0_sampler,
        times=times,
        n_trajectories=500,
        seed=17,
        store_trajectories=False,
    )

    for rho in result.rho_t:
        assert np.isclose(np.trace(rho), 1.0, atol=1e-12)


def test_ensemble_hermitian(qubit_ops):
    H = 0.1 * qubit_ops["sigma_x"]
    gamma = 0.5
    jumps = [np.sqrt(gamma) * qubit_ops["sigma_minus"]]
    times = np.linspace(0.0, 1.0, 31)

    def psi0_sampler(rng):
        return qubit_ops["ket1"]

    result = sample_lindblad_mcwf(
        H=H,
        jumps=jumps,
        psi0_sampler=psi0_sampler,
        times=times,
        n_trajectories=300,
        seed=101,
        store_trajectories=False,
    )

    for rho in result.rho_t:
        assert np.allclose(rho, rho.conj().T, atol=1e-12)


def _test_sample_lindblad_mcwf_example_two_level_atom(qubit_ops):
    """
    Example: two-level atom with spontaneous decay
    """
    import matplotlib.pyplot as plt

    gamma, omega = (1.0, 0.5)

    # Basis: |0> = ground, |1> = excited
    # Hamiltonian: H = (omega/2) sigma_x
    H = 0.5 * omega * qubit_ops["sigma_x"]

    # Jump operator: sqrt(gamma) * sigma_-
    jumps = [np.sqrt(gamma) * qubit_ops["sigma_minus"]]

    # Initial state fixed as excited state |1>
    psi_excited = np.array([0.0, 1.0], dtype=np.complex128)

    def psi0_sampler(rng: np.random.Generator) -> ArrayC:
        return psi_excited

    times = np.linspace(0.0, 10.0, 1001)
    result = sample_lindblad_mcwf(
        H=H,
        jumps=jumps,
        psi0_sampler=psi0_sampler,
        times=times,
        n_trajectories=2000,
        seed=1234,
        store_trajectories=False,
    )

    # Excited-state projector |1><1|
    P_excited = np.array([[0, 0], [0, 1]], dtype=np.complex128)

    excited_population = observable_vs_time(result.rho_t, P_excited)

    plt.figure(figsize=(7, 4.5))
    plt.plot(result.times, excited_population, label=r"$\langle 1|\rho(t)|1\rangle$")
    plt.xlabel("t")
    plt.ylabel("Excited population")
    plt.legend()
    plt.tight_layout()
    plt.show()
