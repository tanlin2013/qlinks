import numpy as np
import pytest

from qlinks.constraints import DimerCoveringConstraint


def test_dimer_constraint_open_chain_middle_site(chain_3_open, binary_chain_3_link_layout) -> None:
    constraint = DimerCoveringConstraint.from_lattice_site(
        lattice=chain_3_open,
        layout=binary_chain_3_link_layout,
        site_id=1,
        required_count=1,
    )

    np.testing.assert_array_equal(constraint.link_ids, np.array([0, 1]))

    assert constraint.is_satisfied(np.array([1, 0]))
    assert constraint.is_satisfied(np.array([0, 1]))
    assert not constraint.is_satisfied(np.array([1, 1]))
    assert not constraint.is_satisfied(np.array([0, 0]))


def test_dimer_constraint_all_sites_chain_perfect_matching(
    chain_4_open, binary_chain_4_link_layout
) -> None:
    constraints = DimerCoveringConstraint.all_sites(
        lattice=chain_4_open,
        layout=binary_chain_4_link_layout,
        required_counts=1,
    )

    # Perfect matching on links 0 and 2.
    config = np.array([1, 0, 1])

    assert len(constraints) == 4
    assert all(c.is_satisfied(config) for c in constraints)


def test_dimer_constraint_square_site(square_2x2_open, square_2x2_open_binary_link_layout) -> None:
    constraint = DimerCoveringConstraint.from_lattice_site(
        lattice=square_2x2_open,
        layout=square_2x2_open_binary_link_layout,
        site_id=0,
        required_count=1,
    )

    assert constraint.is_satisfied(np.array([1, 0, 0, 0]))
    assert constraint.is_satisfied(np.array([0, 1, 0, 0]))
    assert not constraint.is_satisfied(np.array([1, 1, 0, 0]))


def test_dimer_all_sites_rejects_bad_shape(chain_3_open, binary_chain_3_link_layout) -> None:
    with pytest.raises(ValueError, match="required_counts must have shape"):
        DimerCoveringConstraint.all_sites(
            lattice=chain_3_open,
            layout=binary_chain_3_link_layout,
            required_counts=np.array([1, 1]),
        )
