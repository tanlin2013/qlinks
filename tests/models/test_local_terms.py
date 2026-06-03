import numpy as np

from qlinks.models import SquareQDMModel, SquareQLMModel


def _assert_sparse_allclose(actual, expected, atol=1e-12):
    diff = actual - expected
    if diff.nnz:
        assert np.max(np.abs(diff.data)) < atol


def test_square_qdm_local_term_descriptors_count():
    model = SquareQDMModel(
        lx=2,
        ly=2,
        boundary_condition="open",
        coup_kin=-1.0,
        coup_pot=1.0,
    )

    kinetic_terms = model.local_term_descriptors(operator_kind="kinetic")
    potential_terms = model.local_term_descriptors(operator_kind="potential")
    all_terms = model.local_term_descriptors()

    n_plaquettes = len(model.plaquette_ids())

    assert len(kinetic_terms) == n_plaquettes
    assert len(potential_terms) == n_plaquettes
    assert len(all_terms) == 2 * n_plaquettes

    assert all(term.term_kind == "plaquette" for term in kinetic_terms)
    assert all(term.operator_kind == "kinetic" for term in kinetic_terms)
    assert all(term.operator_kind == "potential" for term in potential_terms)
    assert all(len(term.support_links) == 4 for term in kinetic_terms)


def test_square_qdm_local_kinetic_terms_reconstruct_aggregate():
    model = SquareQDMModel(
        lx=2,
        ly=2,
        boundary_condition="open",
        coup_kin=-1.0,
        coup_pot=1.0,
    )

    result = model.build(
        basis_solver="dfs",
        builder="sparse",
        backend="scipy",
        sort_basis=True,
        on_missing="raise",
    )

    local_terms = model.local_term_descriptors(operator_kind="kinetic")
    local_matrices = [
        model.build_local_term(
            term,
            result,
            builder="sparse",
            backend="scipy",
        )
        for term in local_terms
    ]

    reconstructed = sum(local_matrices[1:], local_matrices[0])

    _assert_sparse_allclose(reconstructed, result.kinetic)


def test_square_qdm_local_potential_terms_reconstruct_aggregate():
    model = SquareQDMModel(
        lx=2,
        ly=2,
        boundary_condition="open",
        coup_kin=-1.0,
        coup_pot=1.0,
    )

    result = model.build(
        basis_solver="dfs",
        builder="sparse",
        backend="scipy",
        sort_basis=True,
        on_missing="raise",
    )

    local_terms = model.local_term_descriptors(operator_kind="potential")
    local_matrices = [
        model.build_local_term(
            term,
            result,
            builder="sparse",
            backend="scipy",
        )
        for term in local_terms
    ]

    reconstructed = sum(local_matrices[1:], local_matrices[0])

    _assert_sparse_allclose(reconstructed, result.potential)


def test_square_qlm_local_term_descriptors_count():
    model = SquareQLMModel(
        lx=2,
        ly=2,
        boundary_condition="open",
        coup_kin=-1.0,
        coup_pot=1.0,
        charges=0,
    )

    kinetic_terms = model.local_term_descriptors(operator_kind="kinetic")
    potential_terms = model.local_term_descriptors(operator_kind="potential")

    n_plaquettes = len(model.plaquette_ids())

    assert len(kinetic_terms) == n_plaquettes
    assert len(potential_terms) == n_plaquettes

    assert all(term.term_kind == "plaquette" for term in kinetic_terms)
    assert all(len(term.support_links) == 4 for term in kinetic_terms)


def test_square_qlm_local_kinetic_terms_reconstruct_aggregate():
    model = SquareQLMModel(
        lx=2,
        ly=2,
        boundary_condition="open",
        coup_kin=-1.0,
        coup_pot=1.0,
        charges=0,
    )

    result = model.build(
        basis_solver="dfs",
        builder="sparse",
        backend="scipy",
        sort_basis=True,
        on_missing="raise",
    )

    local_terms = model.local_term_descriptors(operator_kind="kinetic")
    local_matrices = [
        model.build_local_term(
            term,
            result,
            builder="sparse",
            backend="scipy",
        )
        for term in local_terms
    ]

    reconstructed = sum(local_matrices[1:], local_matrices[0])

    _assert_sparse_allclose(reconstructed, result.kinetic)


def test_square_qlm_local_potential_terms_reconstruct_aggregate():
    model = SquareQLMModel(
        lx=2,
        ly=2,
        boundary_condition="open",
        coup_kin=-1.0,
        coup_pot=1.0,
        charges=0,
    )

    result = model.build(
        basis_solver="dfs",
        builder="sparse",
        backend="scipy",
        sort_basis=True,
        on_missing="raise",
    )

    local_terms = model.local_term_descriptors(operator_kind="potential")
    local_matrices = [
        model.build_local_term(
            term,
            result,
            builder="sparse",
            backend="scipy",
        )
        for term in local_terms
    ]

    reconstructed = sum(local_matrices[1:], local_matrices[0])

    _assert_sparse_allclose(reconstructed, result.potential)
