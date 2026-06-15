import numpy as np
import pytest

from qlinks.open_system import (
    EnsembleResult,
    TrajectoryResult,
    analyze_lindblad_evolution,
    density_matrix_from_state,
    diagnose_absorbing_projector_symmetry,
    diagnose_dark_subspace,
    diagnose_jump_span,
    verify_density_matrix,
    verify_lindblad_final_state,
)


def test_diagnose_jump_span_detects_dense_dependencies():
    jump_a = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
    jump_b = 2.0 * jump_a
    jump_c = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.complex128)

    diagnostics = diagnose_jump_span([jump_a, jump_b, jump_c])

    assert diagnostics.n_jumps == 3
    assert diagnostics.dim == 2
    assert diagnostics.span_rank == 2
    assert diagnostics.dependent_jump_count == 1
    assert diagnostics.has_exact_dependencies
    assert diagnostics.compression_ratio == pytest.approx(2.0 / 3.0)
    assert diagnostics.max_normalized_overlap == pytest.approx(1.0)


def test_diagnose_jump_span_accepts_sparse_jumps():
    scipy_sparse = pytest.importorskip("scipy.sparse")
    jump_a = scipy_sparse.csr_array(
        ([1.0], ([0], [1])),
        shape=(2, 2),
        dtype=np.complex128,
    )
    jump_b = scipy_sparse.csr_array(
        ([1.0], ([1], [0])),
        shape=(2, 2),
        dtype=np.complex128,
    )

    diagnostics = diagnose_jump_span([jump_a, jump_b])

    assert diagnostics.span_rank == 2
    assert diagnostics.dependent_jump_count == 0
    assert diagnostics.total_jump_nnz == 2
    assert diagnostics.span_matrix_nnz == 2
    assert diagnostics.to_summary_dict()["span_rank"] == 2


def test_verify_density_matrix_for_target_pure_state():
    psi = np.array([1.0, 0.0], dtype=np.complex128)
    rho = np.outer(psi, psi.conj())

    result = verify_density_matrix(rho, target_state=psi)

    assert result.is_density_matrix
    assert result.fidelity_with_target == pytest.approx(1.0)
    assert result.purity == pytest.approx(1.0)
    assert result.trace_error < 1e-12
    assert result.hermiticity_error < 1e-12


def test_verify_lindblad_final_state_reports_lindblad_residual():
    sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    psi = np.array([1.0, 0.0], dtype=np.complex128)
    rho = np.outer(psi, psi.conj())

    result = verify_lindblad_final_state(
        rho,
        hamiltonian=sx,
        jumps=[],
        target_state=psi,
    )

    assert result.density_matrix.is_density_matrix
    assert result.lindblad_residual > 0.0


def test_verify_lindblad_final_state_accepts_hamiltonian_eigenstate():
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    psi = np.array([1.0, 0.0], dtype=np.complex128)
    rho = np.outer(psi, psi.conj())

    result = verify_lindblad_final_state(
        rho,
        hamiltonian=sz,
        jumps=[],
        target_state=psi,
    )

    assert result.lindblad_residual < 1e-12
    assert result.density_matrix.fidelity_with_target == pytest.approx(1.0)


def test_analyze_lindblad_evolution_reports_fidelity():
    state = np.array([1.0, 0.0], dtype=np.complex128)
    density_matrix = density_matrix_from_state(state)

    diagnostics = analyze_lindblad_evolution(
        [density_matrix, density_matrix],
        target_state=state,
    )

    np.testing.assert_allclose(diagnostics.fidelities, [1.0, 1.0])
    np.testing.assert_allclose(diagnostics.purities, [1.0, 1.0])


def test_analyze_lindblad_evolution_accepts_ensemble_state_snapshots():
    ket0 = np.array([1.0, 0.0], dtype=np.complex128)
    ket1 = np.array([0.0, 1.0], dtype=np.complex128)
    snapshot = np.column_stack([ket0, ket1])
    result = EnsembleResult(
        times=np.array([0.0, 1.0], dtype=np.float64),
        rho_t=[],
        state_snapshots=(snapshot, snapshot),
    )

    diagnostics = analyze_lindblad_evolution(
        ensemble_result=result,
        target_state=ket0,
    )

    assert diagnostics.source == "ensemble_result.state_snapshots"
    assert diagnostics.density_check_mode == "low_rank"
    np.testing.assert_allclose(diagnostics.times, [0.0, 1.0])
    np.testing.assert_array_equal(diagnostics.trajectory_counts, [2, 2])
    np.testing.assert_allclose(diagnostics.state_norm_errors, [0.0, 0.0])
    np.testing.assert_allclose(diagnostics.fidelities, [0.5, 0.5])
    np.testing.assert_allclose(diagnostics.purities, [0.5, 0.5])
    assert np.all(np.isnan(diagnostics.min_eigenvalues))


def test_analyze_lindblad_evolution_full_mode_materializes_state_snapshots():
    ket0 = np.array([1.0, 0.0], dtype=np.complex128)
    snapshot = np.column_stack([ket0, ket0])
    hamiltonian = np.zeros((2, 2), dtype=np.complex128)

    diagnostics = analyze_lindblad_evolution(
        state_snapshots=[snapshot],
        times=[0.0],
        target_state=ket0,
        hamiltonian=hamiltonian,
        jumps=[],
        density_check_mode="full",
    )

    assert diagnostics.source == "state_snapshots"
    assert diagnostics.density_check_mode == "full"
    np.testing.assert_allclose(diagnostics.fidelities, [1.0])
    np.testing.assert_allclose(diagnostics.purities, [1.0])
    np.testing.assert_allclose(diagnostics.lindblad_residuals, [0.0])
    np.testing.assert_allclose(diagnostics.min_eigenvalues, [0.0])


def test_analyze_lindblad_evolution_accepts_stored_trajectories():
    ket0 = np.array([1.0, 0.0], dtype=np.complex128)
    ket1 = np.array([0.0, 1.0], dtype=np.complex128)
    trajectories = (
        TrajectoryResult(
            times=np.array([0.0], dtype=np.float64),
            states=[ket0],
            jump_times=np.array([], dtype=np.float64),
            jump_indices=np.array([], dtype=np.int64),
            norm_errors=np.array([], dtype=np.float64),
        ),
        TrajectoryResult(
            times=np.array([0.0], dtype=np.float64),
            states=[ket1],
            jump_times=np.array([], dtype=np.float64),
            jump_indices=np.array([], dtype=np.int64),
            norm_errors=np.array([], dtype=np.float64),
        ),
    )

    diagnostics = analyze_lindblad_evolution(
        trajectories=trajectories,
        times=[0.0],
        target_state=ket0,
    )

    assert diagnostics.source == "trajectories"
    np.testing.assert_allclose(diagnostics.fidelities, [0.5])
    np.testing.assert_array_equal(diagnostics.trajectory_counts, [2])


def test_analyze_lindblad_evolution_requires_full_mode_for_snapshot_residuals():
    ket0 = np.array([1.0, 0.0], dtype=np.complex128)
    snapshot = np.column_stack([ket0, ket0])

    with pytest.raises(ValueError, match="Lindblad residuals require"):
        analyze_lindblad_evolution(
            state_snapshots=[snapshot],
            hamiltonian=np.zeros((2, 2), dtype=np.complex128),
            jumps=[],
            density_check_mode="low_rank",
        )


def test_diagnose_dark_subspace_amplitude_damping_unique_ground_state():
    hamiltonian = np.zeros((2, 2), dtype=np.complex128)
    jump = np.array(
        [[0.0, 1.0], [0.0, 0.0]],
        dtype=np.complex128,
    )
    target = np.array([1.0, 0.0], dtype=np.complex128)

    diagnostics = diagnose_dark_subspace(
        hamiltonian=hamiltonian,
        jumps=[jump],
        target_state=target,
        check_liouvillian_spectrum=True,
    )

    assert diagnostics.max_target_jump_residual < 1e-12
    assert diagnostics.target_liouvillian_residual < 1e-12
    assert diagnostics.common_jump_kernel_dimension == 1
    assert diagnostics.bad_common_jump_kernel_dimension == 0
    assert diagnostics.liouvillian_zero_mode_count == 1
    assert diagnostics.likely_unique_dark_state is True


def test_diagnose_dark_subspace_no_jumps_not_unique():
    hamiltonian = np.zeros((2, 2), dtype=np.complex128)
    target = np.array([1.0, 0.0], dtype=np.complex128)

    diagnostics = diagnose_dark_subspace(
        hamiltonian=hamiltonian,
        jumps=[],
        target_state=target,
        check_liouvillian_spectrum=True,
    )

    assert diagnostics.common_jump_kernel_dimension == 2
    assert diagnostics.bad_common_jump_kernel_dimension == 1
    assert diagnostics.liouvillian_zero_mode_count == 4
    assert diagnostics.likely_unique_dark_state is False


def test_diagnose_dark_subspace_dephasing_has_multiple_steady_modes():
    hamiltonian = np.zeros((2, 2), dtype=np.complex128)
    jump = np.array(
        [[1.0, 0.0], [0.0, -1.0]],
        dtype=np.complex128,
    )
    target = np.array([1.0, 0.0], dtype=np.complex128)

    diagnostics = diagnose_dark_subspace(
        hamiltonian=hamiltonian,
        jumps=[jump],
        target_state=target,
        check_liouvillian_spectrum=True,
    )

    # Note: target is not annihilated by jump, but is an eigenstate of jump.
    # So max jump residual is nonzero, while rho_target can still be stationary.
    assert diagnostics.max_target_jump_residual > 0.0
    assert diagnostics.target_liouvillian_residual < 1e-12
    assert diagnostics.liouvillian_zero_mode_count > 1
    assert diagnostics.likely_unique_dark_state is False


def test_diagnose_dark_subspace_raises_when_liouvillian_too_large():
    dim = 5
    hamiltonian = np.zeros((dim, dim), dtype=np.complex128)
    target = np.zeros(dim, dtype=np.complex128)
    target[0] = 1.0

    with pytest.raises(ValueError, match="Dense Liouvillian spectrum"):
        diagnose_dark_subspace(
            hamiltonian=hamiltonian,
            jumps=[],
            target_state=target,
            check_liouvillian_spectrum=True,
            max_liouvillian_dense_dimension=8,
        )


def test_diagnose_dark_subspace_can_skip_liouvillian_spectrum():
    dim = 5
    hamiltonian = np.zeros((dim, dim), dtype=np.complex128)
    target = np.zeros(dim, dtype=np.complex128)
    target[0] = 1.0

    diagnostics = diagnose_dark_subspace(
        hamiltonian=hamiltonian,
        jumps=[],
        target_state=target,
        check_liouvillian_spectrum=False,
        max_liouvillian_dense_dimension=8,
    )

    assert diagnostics.liouvillian_zero_mode_count is None
    assert diagnostics.liouvillian_spectral_gap is None
    assert diagnostics.likely_unique_dark_state is None


def test_absorbing_projector_symmetry_detects_isolated_dark_target():
    hamiltonian = np.zeros((2, 2), dtype=np.complex128)

    # J = |1><1| acts only in the orthogonal block.
    # |0> is dark, but there is no inflow from |1> to |0>.
    jump = np.array(
        [[0.0, 0.0], [0.0, 1.0]],
        dtype=np.complex128,
    )
    target = np.array([1.0, 0.0], dtype=np.complex128)

    diagnostics = diagnose_absorbing_projector_symmetry(
        hamiltonian=hamiltonian,
        jumps=[jump],
        target_state=target,
    )

    assert diagnostics.target_is_dark
    assert not diagnostics.has_recycling_inflow
    assert diagnostics.absorbing_projector_is_conserved
    assert diagnostics.has_absorbing_projector_symmetry
    assert diagnostics.max_inflow_norm < 1e-12


def test_absorbing_projector_symmetry_detects_recycling_inflow():
    hamiltonian = np.zeros((2, 2), dtype=np.complex128)

    # J = |0><1| pumps |1> into |0>.
    jump = np.array(
        [[0.0, 1.0], [0.0, 0.0]],
        dtype=np.complex128,
    )
    target = np.array([1.0, 0.0], dtype=np.complex128)

    diagnostics = diagnose_absorbing_projector_symmetry(
        hamiltonian=hamiltonian,
        jumps=[jump],
        target_state=target,
    )

    assert diagnostics.target_is_dark
    assert diagnostics.has_recycling_inflow
    assert not diagnostics.absorbing_projector_is_conserved
    assert not diagnostics.has_absorbing_projector_symmetry
    assert diagnostics.max_inflow_norm > 0.9


def test_absorbing_projector_symmetry_detects_hamiltonian_outflow():
    hamiltonian = np.array(
        [[0.0, 1.0], [1.0, 0.0]],
        dtype=np.complex128,
    )
    target = np.array([1.0, 0.0], dtype=np.complex128)

    diagnostics = diagnose_absorbing_projector_symmetry(
        hamiltonian=hamiltonian,
        jumps=[],
        target_state=target,
    )

    assert not diagnostics.absorbing_projector_is_conserved
    assert diagnostics.hamiltonian_commutator_norm > 0.9


def test_absorbing_projector_symmetry_rejects_matrix_target():
    hamiltonian = np.zeros((2, 2), dtype=np.complex128)
    target = np.eye(2, dtype=np.complex128)

    with pytest.raises(ValueError, match="one-dimensional"):
        diagnose_absorbing_projector_symmetry(
            hamiltonian=hamiltonian,
            jumps=[],
            target_state=target,
        )


def test_analyze_lindblad_evolution_accepts_streamed_target_fidelities_only():
    result = EnsembleResult(
        times=np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
        rho_t=[],
        target_fidelities={"target": np.asarray([0.0, 0.5, 1.0], dtype=np.float64)},
    )

    diagnostics = analyze_lindblad_evolution(ensemble_result=result)

    assert diagnostics.source == "ensemble_result.target_fidelities"
    assert diagnostics.density_check_mode == "streamed_fidelity"
    np.testing.assert_allclose(diagnostics.fidelities, [0.0, 0.5, 1.0])
    assert np.isnan(diagnostics.purities).all()
