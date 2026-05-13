import numpy as np
import pytest

from qlinks.lattice import ChainLattice, SquareLattice
from qlinks.operators import PXPSpinFlipOperator
from qlinks.variables import LocalSpace, VariableLayout


def test_pxp_spin_flip_allowed_chain_middle() -> None:
    lattice = ChainLattice(3, boundary_condition="open")
    layout = VariableLayout.from_lattice_sites(lattice, LocalSpace.binary())

    op = PXPSpinFlipOperator(
        layout=layout,
        lattice=lattice,
        site_id=1,
        coefficient=2.0,
    )

    actions = op.apply(np.array([0, 0, 0]))

    assert len(actions) == 1
    assert actions[0].coefficient == 2.0 + 0j
    np.testing.assert_array_equal(actions[0].config, np.array([0, 1, 0]))


def test_pxp_spin_flip_blocked_chain_middle() -> None:
    lattice = ChainLattice(3, boundary_condition="open")
    layout = VariableLayout.from_lattice_sites(lattice, LocalSpace.binary())

    op = PXPSpinFlipOperator(
        layout=layout,
        lattice=lattice,
        site_id=1,
    )

    assert op.apply(np.array([1, 0, 0])) == ()
    assert op.apply(np.array([0, 0, 1])) == ()
    assert op.apply(np.array([1, 0, 1])) == ()


def test_pxp_spin_flip_boundary_site() -> None:
    lattice = ChainLattice(3, boundary_condition="open")
    layout = VariableLayout.from_lattice_sites(lattice, LocalSpace.binary())

    op = PXPSpinFlipOperator(
        layout=layout,
        lattice=lattice,
        site_id=0,
    )

    actions = op.apply(np.array([0, 0, 1]))
    assert len(actions) == 1
    np.testing.assert_array_equal(actions[0].config, np.array([1, 0, 1]))

    assert op.apply(np.array([0, 1, 0])) == ()


def test_pxp_spin_flip_square_site() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    layout = VariableLayout.from_lattice_sites(lattice, LocalSpace.binary())

    op = PXPSpinFlipOperator(
        layout=layout,
        lattice=lattice,
        site_id=0,
    )

    # Site 0 neighbors are 1 and 2.
    actions = op.apply(np.array([0, 0, 0, 1]))
    assert len(actions) == 1
    np.testing.assert_array_equal(actions[0].config, np.array([1, 0, 0, 1]))

    assert op.apply(np.array([0, 1, 0, 0])) == ()
    assert op.apply(np.array([0, 0, 1, 0])) == ()


def test_pxp_rejects_non_binary_site_space() -> None:
    lattice = ChainLattice(2, boundary_condition="open")
    layout = VariableLayout.from_lattice_sites(lattice, LocalSpace.spin_half_flux())

    with pytest.raises(ValueError, match=r"requires local-space values \[0, 1\]"):
        PXPSpinFlipOperator(
            layout=layout,
            lattice=lattice,
            site_id=0,
        )
