"""Structural sparse/vectorized MCWF kernel contracts.

Keep these tests focused on algorithmically meaningful behavior: sparse operator
preservation, the total-rate operator, vectorized channel evaluation, and the
``total-rate-first`` sampling strategy. Historical micro-kernel optimizations
belong in benchmarks, not permanent private-kernel contracts.
"""

import numpy as np

from qlinks.open_system.states import normalize_state
from qlinks.open_system.stochastic_schrodinger import (
    McwfOptions,
    effective_hamiltonian,
    sample_lindblad_mcwf,
)


def test_sample_lindblad_mcwf_reuses_prepared_effective_hamiltonian(monkeypatch, qubit_ops):
    import qlinks.open_system.stochastic_schrodinger as stochastic_schrodinger

    call_count = 0
    original = stochastic_schrodinger._effective_hamiltonian_from_total_rate_operator

    def counted_effective_hamiltonian(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original(*args, **kwargs)

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
        options=McwfOptions(n_trajectories=4, seed=123, store_trajectories=False),
    )

    assert call_count == 1


def test_effective_hamiltonian_sparse_many_jumps_matches_direct_sum():
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
    )

    assert prepared.uses_sparse_operators
    assert scipy_sparse.issparse(prepared.hamiltonian)
    assert all(scipy_sparse.issparse(jump_operator) for jump_operator in prepared.jumps)
    assert scipy_sparse.issparse(prepared.effective_hamiltonian_matrix)
    assert scipy_sparse.issparse(prepared.total_jump_rate_operator)


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


def test_jump_rates_state_matrix_match_direct_jump_actions(qubit_ops):
    import scipy.sparse as scipy_sparse

    from qlinks.open_system.stochastic_schrodinger import _evaluate_jump_rates_state_matrix_numpy

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


def test_total_jump_rate_operator_matches_channel_rates():
    import scipy.sparse as scipy_sparse

    from qlinks.open_system.stochastic_schrodinger import (
        _evaluate_jump_rates_state_matrix_numpy,
        _evaluate_total_jump_rates_state_matrix_numpy,
        _total_jump_rate_operator,
    )

    dim = 16
    jumps = (
        scipy_sparse.csr_array(
            (
                np.asarray([1.0 + 0.0j, 0.5j], dtype=np.complex128),
                (np.asarray([2, 2]), np.asarray([3, 4])),
            ),
            shape=(dim, dim),
        ),
        scipy_sparse.csr_array(
            ([2.0 + 0.0j], ([5], [6])),
            shape=(dim, dim),
            dtype=np.complex128,
        ),
    )
    rng = np.random.default_rng(123)
    states = rng.normal(size=(dim, 4)) + 1j * rng.normal(size=(dim, 4))

    gamma = _total_jump_rate_operator(jumps)
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
            use_total_rate_first=False,
        ),
    )
    total_rate_first = sample_lindblad_mcwf(
        hamiltonian=hamiltonian,
        jumps=jumps,
        times=times,
        state_initial=qubit_ops["ket1"],
        options=McwfOptions(
            n_trajectories=32,
            seed=123,
            store_trajectories=False,
            prefer_sparse_operators=True,
            use_total_rate_first=True,
        ),
    )

    for actual_rho, expected_rho in zip(total_rate_first.rho_t, baseline.rho_t, strict=True):
        np.testing.assert_allclose(actual_rho, expected_rho, atol=1e-14)


def test_total_rate_first_skips_channel_rates_when_no_jump_occurs(monkeypatch, qubit_ops):
    import scipy.sparse as scipy_sparse

    import qlinks.open_system.stochastic_schrodinger as stochastic_schrodinger

    calls = 0
    original = stochastic_schrodinger._evaluate_jump_rates_state_matrix_numpy

    def counted_channel_rates(states, jumps):
        nonlocal calls
        calls += 1
        return original(states, jumps)

    monkeypatch.setattr(
        stochastic_schrodinger,
        "_evaluate_jump_rates_state_matrix_numpy",
        counted_channel_rates,
    )

    result = sample_lindblad_mcwf(
        hamiltonian=scipy_sparse.csr_array((2, 2), dtype=np.complex128),
        jumps=[scipy_sparse.csr_array(np.sqrt(0.2) * qubit_ops["sigma_minus"])],
        times=np.linspace(0.0, 0.1, 5),
        state_initial=qubit_ops["ket0"],
        options=McwfOptions(
            n_trajectories=8,
            seed=123,
            store_trajectories=False,
            store_density_matrices=False,
            prefer_sparse_operators=True,
            use_total_rate_first=True,
        ),
    )

    assert result.rho_t == []
    assert calls == 0
