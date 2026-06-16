import numpy as np

from qlinks.open_system import (
    build_local_recycling_jumps_from_regions,
    detect_two_pattern_recycling_structure,
    local_rank_one_matrix_unit_expansion,
    local_reduced_density_matrix_from_state,
    scan_local_recycling_candidates,
    score_recycling_jump,
)


def test_local_reduced_density_matrix_product_state_has_nullspace() -> None:
    basis_configs = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ],
        dtype=np.int64,
    )
    state = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)

    rdm = local_reduced_density_matrix_from_state(
        basis_configs=basis_configs,
        state=state,
        variable_indices=(0,),
    )

    assert rdm.local_dim == 2
    assert rdm.support_rank == 1
    assert rdm.nullity == 1


def test_local_reduced_density_matrix_bell_state_has_full_one_site_support() -> None:
    basis_configs = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ],
        dtype=np.int64,
    )
    state = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.complex128)
    state = state / np.linalg.norm(state)

    rdm = local_reduced_density_matrix_from_state(
        basis_configs=basis_configs,
        state=state,
        variable_indices=(0,),
    )

    assert rdm.support_rank == 2
    assert rdm.nullity == 0


def test_scan_local_recycling_candidates_finds_product_state_pump() -> None:
    basis_configs = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ],
        dtype=np.int64,
    )
    state = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)

    result = scan_local_recycling_candidates(
        basis_configs=basis_configs,
        target_state=state,
        variable_indices=(0,),
        inflow_tolerance=1e-12,
    )

    assert result.n_candidates >= 1

    candidate = result.best_candidates[0]

    assert candidate.target_residual < 1e-12
    assert candidate.inflow_norm > 0.0
    assert candidate.jump.shape == (4, 4)


def test_local_rank_one_matrix_unit_expansion_for_two_pattern_jump() -> None:
    local_patterns = ((0,), (1,))
    alpha = np.array([1.0, -1.0], dtype=np.complex128) / np.sqrt(2.0)
    beta = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)

    terms = local_rank_one_matrix_unit_expansion(
        local_patterns=local_patterns,
        alpha=alpha,
        beta=beta,
        tolerance=1e-12,
    )

    assert len(terms) == 4

    coefficients = {(term.target_pattern, term.source_pattern): term.coefficient for term in terms}

    assert np.isclose(coefficients[((0,), (0,))], 0.5)
    assert np.isclose(coefficients[((0,), (1,))], 0.5)
    assert np.isclose(coefficients[((1,), (0,))], -0.5)
    assert np.isclose(coefficients[((1,), (1,))], -0.5)


def test_detect_two_pattern_recycling_structure() -> None:
    basis_configs = np.array(
        [
            [0],
            [1],
        ],
        dtype=np.int64,
    )

    # Target state is |-> = (|0> - |1>) / sqrt(2).
    state = np.array([1.0, -1.0], dtype=np.complex128) / np.sqrt(2.0)

    result = scan_local_recycling_candidates(
        basis_configs=basis_configs,
        target_state=state,
        variable_indices=(0,),
        inflow_tolerance=1e-12,
    )

    assert result.n_candidates == 1

    candidate = result.best_candidates[0]
    structure = detect_two_pattern_recycling_structure(
        candidate=candidate,
        local_patterns=result.reduced_density_matrix.local_patterns,
        tolerance=1e-10,
    )

    assert structure is not None
    assert structure.pattern_a == (0,)
    assert structure.pattern_b == (1,)
    assert len(structure.matrix_unit_terms) == 4


def test_build_local_recycling_jumps_from_regions_selects_two_pattern_jump() -> None:
    basis_configs = np.array(
        [
            [0],
            [1],
        ],
        dtype=np.int64,
    )

    state = np.array([1.0, -1.0], dtype=np.complex128) / np.sqrt(2.0)

    result = build_local_recycling_jumps_from_regions(
        basis_configs=basis_configs,
        target_state=state,
        regions=((0,),),
        source="local_rdm_two_pattern",
        max_jumps_per_region=1,
        inflow_tolerance=1e-12,
    )

    assert result.n_jumps == 1
    assert len(result.selections) == 1
    assert result.selections[0].two_pattern_structure is not None
    assert result.jumps[0].shape == (2, 2)


def test_score_recycling_jump_matches_dense_projector_formula() -> None:
    rng = np.random.default_rng(123)
    dense_jump = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    sparse_jump = np.asarray(dense_jump, dtype=np.complex128)
    state = rng.normal(size=4) + 1j * rng.normal(size=4)
    state = state.astype(np.complex128)
    state = state / np.linalg.norm(state)

    projector_target = np.outer(state, state.conj())
    projector_orthogonal = np.eye(state.size, dtype=np.complex128) - projector_target

    expected = (
        float(np.linalg.norm(dense_jump @ state)),
        float(np.linalg.norm(projector_target @ dense_jump @ projector_orthogonal)),
        float(np.linalg.norm(projector_orthogonal @ dense_jump @ projector_target)),
        float(np.linalg.norm(dense_jump @ projector_target - projector_target @ dense_jump)),
    )

    observed = score_recycling_jump(
        jump=sparse_jump,
        target_state=state,
    )

    assert np.allclose(observed, expected)


def test_embed_local_pattern_operator_reuses_transition_context_consistently() -> None:
    basis_configs = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ],
        dtype=np.int64,
    )
    state = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)

    result = scan_local_recycling_candidates(
        basis_configs=basis_configs,
        target_state=state,
        variable_indices=(0,),
        inflow_tolerance=1e-12,
    )

    assert result.n_candidates >= 1
    assert result.best_candidates[0].jump.nnz == 2


def test_build_local_recycling_jumps_reuses_duplicate_region_scan() -> None:
    basis_configs = np.array(
        [
            [0],
            [1],
        ],
        dtype=np.int64,
    )
    state = np.array([1.0, -1.0], dtype=np.complex128) / np.sqrt(2.0)

    result = build_local_recycling_jumps_from_regions(
        basis_configs=basis_configs,
        target_state=state,
        regions=((0,), (0,)),
        source="local_rdm_two_pattern",
        max_jumps_per_region=1,
        inflow_tolerance=1e-12,
    )

    assert len(result.scan_results) == 2
    assert result.scan_results[0] is result.scan_results[1]
    assert result.n_jumps == 2


def test_build_local_recycling_jumps_null_basis_selects_all_null_directions() -> None:
    basis_configs = np.arange(4, dtype=np.int64).reshape(4, 1)
    state = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)

    result = build_local_recycling_jumps_from_regions(
        basis_configs=basis_configs,
        target_state=state,
        regions=((0,),),
        source="local_rdm_null_basis",
        max_jumps_per_region=1,
        inflow_tolerance=1e-12,
    )

    assert result.scan_results[0].reduced_density_matrix.nullity == 3
    assert result.n_jumps == 3
    assert sorted(selection.candidate.beta_index for selection in result.selections) == [0, 1, 2]
