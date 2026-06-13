import numpy as np
import scipy.sparse as scipy_sparse

from qlinks.caging import (
    CageSolverConfig,
    CandidateSubgraph,
    solve_candidate,
    solve_candidate_for_kinetic_targets,
    solve_candidates,
)
from qlinks.caging.candidate import (
    BOUNDARY_OVERLAP_MATRIX_METADATA_KEY,
    INTERNAL_KINETIC_MATRIX_METADATA_KEY,
)


def test_solve_candidate_finds_two_site_antisymmetric_cage() -> None:
    hamiltonian = np.array(
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.complex128,
    )
    candidate = CandidateSubgraph(vertices=np.array([0, 1]))

    config = CageSolverConfig(tolerance=1e-12)
    cage_states = solve_candidate(hamiltonian, candidate, config=config)

    assert len(cage_states) == 1

    cage_state = cage_states[0]

    np.testing.assert_allclose(cage_state.energy, -1.0, atol=1e-12)
    assert cage_state.boundary_residual <= 1e-12
    assert cage_state.eigen_residual <= 1e-12
    assert cage_state.full_residual is not None
    assert cage_state.full_residual <= 1e-12

    expected_local_state = np.array([1.0, -1.0], dtype=np.complex128)
    expected_local_state /= np.linalg.norm(expected_local_state)

    overlap = abs(np.vdot(expected_local_state, cage_state.local_state))
    np.testing.assert_allclose(overlap, 1.0, atol=1e-12)


def test_solve_candidate_returns_empty_for_non_invariant_boundary_nullspace() -> None:
    hamiltonian = np.array(
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.complex128,
    )
    candidate = CandidateSubgraph(vertices=np.array([0, 1]))

    config = CageSolverConfig(tolerance=1e-12)
    cage_states = solve_candidate(hamiltonian, candidate, config=config)

    assert cage_states == []


def test_solve_candidate_handles_sparse_hamiltonian() -> None:
    hamiltonian = scipy_sparse.csr_matrix(
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.complex128,
    )
    candidate = CandidateSubgraph(vertices=np.array([0, 1]))

    config = CageSolverConfig(tolerance=1e-12)
    cage_states = solve_candidate(hamiltonian, candidate, config=config)

    assert len(cage_states) == 1
    np.testing.assert_allclose(cage_states[0].energy, -1.0, atol=1e-12)


def test_solve_candidate_for_kinetic_targets_matches_generic_solver() -> None:
    kinetic_matrix = np.array(
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.complex128,
    )
    self_loop_values = np.full(3, 2.0, dtype=np.complex128)
    hamiltonian = kinetic_matrix + np.diag(self_loop_values)
    candidate = CandidateSubgraph(vertices=np.array([0, 1]))

    config = CageSolverConfig(tolerance=1e-12)
    generic_states = solve_candidate(hamiltonian, candidate, config=config)
    target_states = solve_candidate_for_kinetic_targets(
        hamiltonian,
        kinetic_matrix,
        self_loop_values,
        candidate,
        target_kappas=(-1.0,),
        config=config,
    )

    assert len(generic_states) == 1
    assert len(target_states) == 1

    np.testing.assert_allclose(target_states[0].energy, 1.0, atol=1e-12)
    np.testing.assert_allclose(
        target_states[0].energy,
        generic_states[0].energy,
        atol=1e-12,
    )

    overlap = abs(np.vdot(generic_states[0].local_state, target_states[0].local_state))
    np.testing.assert_allclose(overlap, 1.0, atol=1e-12)
    assert target_states[0].metadata["fixed_kappa_solver"] is True


def test_solve_candidate_for_kinetic_targets_exact_full_residual_rejects_mismatch() -> None:
    kinetic_matrix = np.array(
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.complex128,
    )
    self_loop_values = np.full(3, 2.0, dtype=np.complex128)
    hamiltonian = kinetic_matrix + np.diag(self_loop_values)
    hamiltonian[2, 0] += 0.5
    candidate = CandidateSubgraph(vertices=np.array([0, 1]))

    cage_states = solve_candidate_for_kinetic_targets(
        hamiltonian,
        kinetic_matrix,
        self_loop_values,
        candidate,
        target_kappas=(-1.0,),
        config=CageSolverConfig(
            tolerance=1e-12,
            validate_full_residual=True,
        ),
    )

    assert cage_states == []


def test_solve_candidate_for_kinetic_targets_rejects_nonuniform_self_loops() -> None:
    kinetic_matrix = np.array(
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.complex128,
    )
    self_loop_values = np.array([2.0, 3.0, 2.0], dtype=np.complex128)
    hamiltonian = kinetic_matrix + np.diag(self_loop_values)
    candidate = CandidateSubgraph(vertices=np.array([0, 1]))

    cage_states = solve_candidate_for_kinetic_targets(
        hamiltonian,
        kinetic_matrix,
        self_loop_values,
        candidate,
        target_kappas=(-1.0,),
        config=CageSolverConfig(tolerance=1e-12),
    )

    assert cage_states == []


def test_solve_candidate_for_kinetic_targets_rejects_boundary_leakage_without_full_residual() -> (
    None
):
    kinetic_matrix = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=np.complex128,
    )
    self_loop_values = np.zeros(2, dtype=np.complex128)
    hamiltonian = kinetic_matrix.copy()
    candidate = CandidateSubgraph(vertices=np.array([0]))

    cage_states = solve_candidate_for_kinetic_targets(
        hamiltonian,
        kinetic_matrix,
        self_loop_values,
        candidate,
        target_kappas=(0.0,),
        config=CageSolverConfig(
            tolerance=1e-12,
            validate_full_residual=False,
        ),
    )

    assert cage_states == []


def test_solve_candidate_for_kinetic_targets_uses_column_block_for_boundary_residual() -> None:
    kinetic_matrix = np.zeros((3, 3), dtype=np.complex128)
    self_loop_values = np.array([2.0, 2.0, 0.0], dtype=np.complex128)
    hamiltonian = kinetic_matrix + np.diag(self_loop_values)
    candidate = CandidateSubgraph(
        vertices=np.array([0, 1]),
        metadata={
            INTERNAL_KINETIC_MATRIX_METADATA_KEY: np.zeros((2, 2), dtype=np.complex128),
            # This mimics a tiny positive Gram-matrix roundoff error.  If the
            # boundary residual is recovered as sqrt(<psi|B^†B|psi>), one exact
            # cage is falsely rejected at tolerance 1e-10.  The actual kinetic
            # column block has no boundary leakage, so both states should pass.
            BOUNDARY_OVERLAP_MATRIX_METADATA_KEY: np.diag(
                np.array([1e-16, 0.0], dtype=np.complex128)
            ),
        },
    )

    cage_states = solve_candidate_for_kinetic_targets(
        hamiltonian,
        kinetic_matrix,
        self_loop_values,
        candidate,
        target_kappas=(0.0,),
        config=CageSolverConfig(
            tolerance=1e-10,
            validate_full_residual=True,
        ),
    )

    assert len(cage_states) == 2
    assert all(cage_state.boundary_residual <= 1e-10 for cage_state in cage_states)
    assert all(cage_state.full_residual is not None for cage_state in cage_states)
    assert all(cage_state.full_residual <= 1e-10 for cage_state in cage_states)


def test_solve_candidates_combines_results() -> None:
    hamiltonian = np.array(
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.complex128,
    )
    candidates = [
        CandidateSubgraph(vertices=np.array([0, 1])),
        CandidateSubgraph(vertices=np.array([0, 2])),
    ]

    config = CageSolverConfig(tolerance=1e-12)
    cage_states = solve_candidates(hamiltonian, candidates, config=config)

    assert len(cage_states) == 2
    np.testing.assert_allclose(
        sorted(cage_state.energy.real for cage_state in cage_states),
        [-1.0, -1.0],
        atol=1e-12,
    )


def test_solve_candidate_for_kinetic_targets_accumulates_timing_collector() -> None:
    kinetic_matrix = np.array(
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.complex128,
    )
    self_loop_values = np.full(3, 2.0, dtype=np.complex128)
    hamiltonian = kinetic_matrix + np.diag(self_loop_values)
    candidate = CandidateSubgraph(vertices=np.array([0, 1]))
    timing_collector: dict[str, float] = {}

    cage_states = solve_candidate_for_kinetic_targets(
        hamiltonian,
        kinetic_matrix,
        self_loop_values,
        candidate,
        target_kappas=(-1.0,),
        config=CageSolverConfig(
            tolerance=1e-12,
            timing_collector=timing_collector,
        ),
    )

    assert len(cage_states) == 1
    assert timing_collector["solver.candidate_blocks"] >= 0.0
    assert timing_collector["solver.fixed_kappa_nullspace"] >= 0.0
    assert timing_collector["solver.validation"] >= 0.0
