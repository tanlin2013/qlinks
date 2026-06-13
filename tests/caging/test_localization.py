import numpy as np

from qlinks.caging.candidate import CandidateSubgraph
from qlinks.caging.localization import (
    IPRLocalizationConfig,
    localized_basis_by_many_start_ipr,
)
from qlinks.caging.results import cage_state_to_full_vector
from qlinks.caging.solver import solve_candidate
from qlinks.caging.types import CageSolverConfig


def _support_set(vector: np.ndarray, *, tolerance: float = 1e-8) -> set[int]:
    """Return the numerical support of a vector as a plain Python int set."""
    return set(int(index) for index in np.flatnonzero(np.abs(vector) > tolerance))


def _make_two_degenerate_cage_hamiltonian() -> np.ndarray:
    """
    Build a small Hamiltonian with exactly two compact degenerate cages.

    Candidate vertices are [0, 1, 2, 3].
    Outside vertices are [4, 5].

    The compact eigenstates are

        phi_a = (|0> - |1>) / sqrt(2)
        phi_b = (|2> - |3>) / sqrt(2)

    both with energy 0.

    The two outside vertices impose two independent boundary constraints:

        x0 + x1 + x2 + x3 = 0
        x0 + x1 - x2 - x3 = 0

    Therefore the boundary-nullspace is exactly

        x0 + x1 = 0
        x2 + x3 = 0,

    so the caged space is two-dimensional.
    """
    hamiltonian = np.zeros((6, 6), dtype=np.complex128)

    # Internal two-site blocks:
    #
    # [[1, 1],
    #  [1, 1]]
    #
    # The antisymmetric vector has energy 0.
    hamiltonian[0, 0] = 1.0
    hamiltonian[1, 1] = 1.0
    hamiltonian[0, 1] = 1.0
    hamiltonian[1, 0] = 1.0

    hamiltonian[2, 2] = 1.0
    hamiltonian[3, 3] = 1.0
    hamiltonian[2, 3] = 1.0
    hamiltonian[3, 2] = 1.0

    # Outside vertex 4 imposes x0 + x1 + x2 + x3 = 0.
    for vertex in [0, 1, 2, 3]:
        hamiltonian[4, vertex] = 1.0
        hamiltonian[vertex, 4] = 1.0

    # Outside vertex 5 imposes x0 + x1 - x2 - x3 = 0.
    for vertex in [0, 1]:
        hamiltonian[5, vertex] = 1.0
        hamiltonian[vertex, 5] = 1.0

    for vertex in [2, 3]:
        hamiltonian[5, vertex] = -1.0
        hamiltonian[vertex, 5] = -1.0

    return hamiltonian


def test_many_start_ipr_separates_mixed_compact_states():
    vector_a = np.zeros(6, dtype=np.complex128)
    vector_b = np.zeros(6, dtype=np.complex128)

    vector_a[[0, 1]] = [1.0, -1.0]
    vector_b[[4, 5]] = [1.0, -1.0]

    vector_a /= np.linalg.norm(vector_a)
    vector_b /= np.linalg.norm(vector_b)

    mixed_1 = (vector_a + vector_b) / np.sqrt(2.0)
    mixed_2 = (vector_a - vector_b) / np.sqrt(2.0)

    basis = np.column_stack([mixed_1, mixed_2])

    config = IPRLocalizationConfig(
        n_restarts=128,
        max_iter=3000,
        candidate_count=64,
        random_seed=123,
        amplitude_tolerance=1e-12,
        rank_tolerance=1e-10,
        minimum_gap_ratio=10.0,
    )

    localized_basis = localized_basis_by_many_start_ipr(
        basis,
        config=config,
    )

    supports = [
        _support_set(localized_basis[:, index], tolerance=1e-8)
        for index in range(localized_basis.shape[1])
    ]

    assert {0, 1} in supports
    assert {4, 5} in supports


def test_solve_candidate_preserves_default_degenerate_strategy():
    """
    By default, degenerate cage localization is disabled.

    This test should not assume the raw SVD/eigensolver basis is mixed or local,
    because that is implementation-dependent. It only checks that the old
    strategy path is used and that the caged subspace is two-dimensional.
    """
    hamiltonian = _make_two_degenerate_cage_hamiltonian()
    candidate = CandidateSubgraph(
        vertices=np.array([0, 1, 2, 3], dtype=np.int64),
        label="two_degenerate_cages",
    )

    config = CageSolverConfig(
        tolerance=1e-10,
        validate_full_residual=True,
        degenerate_basis_strategy="none",
    )

    cage_states = solve_candidate(
        hamiltonian,
        candidate,
        config=config,
    )

    assert len(cage_states) == 2

    for cage_state in cage_states:
        assert cage_state.boundary_residual <= config.tolerance
        assert cage_state.eigen_residual <= config.tolerance
        assert cage_state.full_residual is not None
        assert cage_state.full_residual <= config.tolerance
        assert cage_state.metadata is not None
        assert cage_state.metadata["degenerate_basis_strategy"] == "none"
        assert cage_state.metadata["degenerate_group_size"] == 2


def test_solve_candidate_ipr_splits_degenerate_union_support():
    """
    With degenerate_basis_strategy='ipr', the solver should rotate the
    degenerate caged eigenspace and return the two tiny compact supports.
    """
    hamiltonian = _make_two_degenerate_cage_hamiltonian()
    candidate = CandidateSubgraph(
        vertices=np.array([0, 1, 2, 3], dtype=np.int64),
        label="two_degenerate_cages",
    )

    config = CageSolverConfig(
        tolerance=1e-10,
        validate_full_residual=True,
        degenerate_basis_strategy="ipr",
        ipr_n_restarts=128,
        ipr_max_iter=3000,
        ipr_candidate_count=64,
        ipr_random_seed=123,
    )

    cage_states = solve_candidate(
        hamiltonian,
        candidate,
        config=config,
    )

    assert len(cage_states) == 2

    supports = [set(int(index) for index in cage_state.support) for cage_state in cage_states]

    assert {0, 1} in supports
    assert {2, 3} in supports

    for cage_state in cage_states:
        assert cage_state.support_size == 2
        assert cage_state.support_size < candidate.size
        assert cage_state.boundary_residual <= config.tolerance
        assert cage_state.eigen_residual <= config.tolerance
        assert cage_state.full_residual is not None
        assert cage_state.full_residual <= config.tolerance
        assert cage_state.metadata is not None
        assert cage_state.metadata["degenerate_basis_strategy"] == "ipr"
        assert cage_state.metadata["degenerate_group_size"] == 2

        full_vector = cage_state_to_full_vector(
            cage_state,
            hilbert_size=hamiltonian.shape[0],
        )

        residual = np.linalg.norm(hamiltonian @ full_vector - cage_state.energy * full_vector)
        assert residual <= config.tolerance


def test_many_start_ipr_avoids_per_restart_orthonormalization(monkeypatch):
    import qlinks.caging.localization as localization

    basis = np.eye(4, 2, dtype=np.complex128)
    call_count = 0
    original = localization._orthonormalize_columns

    def counted_orthonormalize(matrix, *, tolerance):
        nonlocal call_count
        call_count += 1
        return original(matrix, tolerance=tolerance)

    monkeypatch.setattr(localization, "_orthonormalize_columns", counted_orthonormalize)

    config = IPRLocalizationConfig(
        candidate_count=5,
        max_iter=2,
        random_seed=123,
        rank_tolerance=1e-10,
    )

    localized_basis_by_many_start_ipr(basis, config=config)

    # One call prepares the cached working basis; one final call
    # orthonormalizes the selected localized output basis. The old slow path
    # effectively did one extra orthonormalization per restart.
    assert call_count == 2


def test_many_start_ipr_rank_completion_patience_stops_after_unique_supports(monkeypatch):
    import qlinks.caging.localization as localization

    basis = np.eye(4, 2, dtype=np.complex128)
    supports = [
        np.array([True, False, False, False]),
        np.array([False, True, False, False]),
        np.array([False, False, True, False]),
        np.array([False, False, False, True]),
    ]
    calls = 0

    def fake_maximize(working_basis, *, config, rng):
        nonlocal calls
        support_mask = supports[min(calls, len(supports) - 1)]
        local_state = np.zeros(working_basis.basis.shape[0], dtype=np.complex128)
        local_state[np.flatnonzero(support_mask)[0]] = 1.0
        calls += 1
        return localization.LocalizedState(
            local_state=local_state,
            coefficients=np.zeros(working_basis.dimension, dtype=np.complex128),
            support_mask=support_mask,
            ipr_value=1.0,
        )

    monkeypatch.setattr(localization, "_maximize_ipr_once_with_working_basis", fake_maximize)

    config = IPRLocalizationConfig(
        candidate_count=10,
        rank_completion_patience=0,
        random_seed=123,
        rank_tolerance=1e-10,
    )

    localized_basis = localized_basis_by_many_start_ipr(basis, config=config)

    assert calls == 2
    assert localized_basis.shape[1] == 2
