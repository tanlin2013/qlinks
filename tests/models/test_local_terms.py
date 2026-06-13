import pytest

from qlinks.models import SquareQDMModel, SquareQLMModel
from tests.helpers.assertions import assert_sparse_allclose


@pytest.mark.parametrize(
    ("model_factory", "expected_support_size"),
    [
        (
            lambda: SquareQDMModel(
                lx=2,
                ly=2,
                boundary_condition="open",
                coup_kin=-1.0,
                coup_pot=1.0,
            ),
            4,
        ),
        (
            lambda: SquareQLMModel(
                lx=2,
                ly=2,
                boundary_condition="open",
                coup_kin=-1.0,
                coup_pot=1.0,
                charges=0,
            ),
            4,
        ),
    ],
)
def test_local_term_descriptors_count(model_factory, expected_support_size) -> None:
    model = model_factory()

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
    assert all(len(term.support_links) == expected_support_size for term in kinetic_terms)


@pytest.mark.parametrize(
    ("model_factory", "operator_kind", "expected_matrix_name"),
    [
        (
            lambda: SquareQDMModel(
                lx=2,
                ly=2,
                boundary_condition="open",
                coup_kin=-1.0,
                coup_pot=1.0,
            ),
            "kinetic",
            "kinetic",
        ),
        (
            lambda: SquareQDMModel(
                lx=2,
                ly=2,
                boundary_condition="open",
                coup_kin=-1.0,
                coup_pot=1.0,
            ),
            "potential",
            "potential",
        ),
        (
            lambda: SquareQLMModel(
                lx=2,
                ly=2,
                boundary_condition="open",
                coup_kin=-1.0,
                coup_pot=1.0,
                charges=0,
            ),
            "kinetic",
            "kinetic",
        ),
        (
            lambda: SquareQLMModel(
                lx=2,
                ly=2,
                boundary_condition="open",
                coup_kin=-1.0,
                coup_pot=1.0,
                charges=0,
            ),
            "potential",
            "potential",
        ),
    ],
)
def test_local_terms_reconstruct_aggregate(
    model_factory,
    operator_kind: str,
    expected_matrix_name: str,
) -> None:
    model = model_factory()
    result = model.build(
        basis_solver="dfs",
        builder="sparse",
        backend="scipy",
        sort_basis=True,
        on_missing="raise",
    )

    local_terms = model.local_term_descriptors(operator_kind=operator_kind)
    local_matrices = [
        model.build_local_term(term, result, builder="sparse", backend="scipy")
        for term in local_terms
    ]

    reconstructed = sum(local_matrices[1:], local_matrices[0])
    expected = getattr(result, expected_matrix_name)

    assert_sparse_allclose(reconstructed, expected)


def test_local_term_descriptor_caches_support_link_set() -> None:
    term = SquareQDMModel(lx=2, ly=2, boundary_condition="open").local_term_descriptors(
        operator_kind="kinetic"
    )[0]

    assert term.support_link_set == frozenset(term.support_links)
    assert term.support_link_set is term.support_link_set
    assert term.is_inside_links(set(term.support_links))
    assert term.is_disjoint_from_links(set())
