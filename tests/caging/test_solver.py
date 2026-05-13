import numpy as np
import scipy.sparse as scipy_sparse

from qlinks.caging import (
    CageSolverConfig,
    CandidateSubgraph,
    solve_candidate,
    solve_candidates,
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
