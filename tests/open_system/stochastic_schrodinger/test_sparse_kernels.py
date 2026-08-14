"""Internal optimized-MCWF kernel contracts.

These tests intentionally exercise private numerical kernels. Keep private-kernel
coupling isolated here rather than spreading it through behavioral MCWF tests.
"""

import numpy as np
import pytest

from qlinks.open_system.states import normalize_state
from qlinks.open_system.stochastic_schrodinger import (
    McwfOptions,
    effective_hamiltonian,
    sample_lindblad_mcwf,
)


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


def test_prepare_mcwf_operators_compresses_collinear_sparse_jumps(qubit_ops):
    scipy_sparse = pytest.importorskip("scipy.sparse")

    hamiltonian = scipy_sparse.csr_array(np.zeros((2, 2), dtype=np.complex128))
    base_jump = scipy_sparse.csr_array(qubit_ops["sigma_minus"])
    jumps = [base_jump, 2.0 * base_jump, scipy_sparse.csr_array(qubit_ops["sigma_plus"])]

    from qlinks.open_system.stochastic_schrodinger import _prepare_mcwf_operators

    uncompressed = _prepare_mcwf_operators(
        hamiltonian=hamiltonian,
        jumps=jumps,
        backend="scipy",
        prefer_sparse_operators=True,
        prefer_sparse_rate_evaluator=False,
    )
    compressed = _prepare_mcwf_operators(
        hamiltonian=hamiltonian,
        jumps=jumps,
        backend="scipy",
        prefer_sparse_operators=True,
        prefer_sparse_rate_evaluator=False,
        compress_collinear_jumps=True,
    )

    assert compressed.jump_compression_summary is not None
    assert compressed.jump_compression_summary.original_n_jumps == 3
    assert compressed.jump_compression_summary.compressed_n_jumps == 2
    assert compressed.jump_compression_summary.reduced_jump_count == 1
    assert all(scipy_sparse.issparse(jump) for jump in compressed.jumps)
    assert compressed.jump_compression_summary.compressed_total_nnz == 2

    np.testing.assert_allclose(
        compressed.total_jump_rate_operator.toarray(),
        uncompressed.total_jump_rate_operator.toarray(),
        atol=1e-14,
    )


def test_mcwf_options_rejects_nonpositive_jump_compression_tolerance():
    with pytest.raises(ValueError, match="jump_compression_tolerance"):
        McwfOptions(jump_compression_tolerance=0.0).validate()


def test_total_rate_action_reuse_matches_effective_hamiltonian_update(qubit_ops):
    import scipy.sparse as scipy_sparse

    from qlinks.open_system.stochastic_schrodinger import (
        _effective_hamiltonian_from_total_rate_operator,
        _evaluate_total_jump_rates_and_action_state_matrix_numpy,
        _total_jump_rate_operator,
    )

    hamiltonian = scipy_sparse.csr_array(0.37 * qubit_ops["sigma_x"])
    jumps = (
        scipy_sparse.csr_array(np.sqrt(0.23) * qubit_ops["sigma_minus"]),
        scipy_sparse.csr_array(np.sqrt(0.11) * qubit_ops["sigma_z"]),
    )
    states = np.array(
        [[1.0, 0.0, 1.0j], [0.0, 1.0, 1.0]],
        dtype=np.complex128,
    )
    states /= np.sqrt(np.sum(np.abs(states) ** 2, axis=0)).reshape(1, -1)
    step_size = 0.031

    total_rate_operator = _total_jump_rate_operator(jumps, shape=hamiltonian.shape)
    assert total_rate_operator is not None
    effective_hamiltonian_matrix = _effective_hamiltonian_from_total_rate_operator(
        hamiltonian,
        total_rate_operator,
    )

    _, total_rate_action = _evaluate_total_jump_rates_and_action_state_matrix_numpy(
        states,
        total_rate_operator,
    )
    reused_action_update = (
        states - 1j * step_size * (hamiltonian @ states) - 0.5 * step_size * total_rate_action
    )
    effective_hamiltonian_update = states - 1j * step_size * (effective_hamiltonian_matrix @ states)

    np.testing.assert_allclose(reused_action_update, effective_hamiltonian_update, atol=1e-14)
