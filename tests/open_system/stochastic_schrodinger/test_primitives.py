import numpy as np
import pytest

from qlinks.open_system.states import density_matrix_from_state, normalize_state
from qlinks.open_system.stochastic_schrodinger import (
    choose_jump,
    effective_hamiltonian,
    evolve_no_jump_first_order,
    expectation,
    jump_probabilities,
    projector,
)


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
