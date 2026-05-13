import numpy as np
import pytest

from qlinks.lattice import ChainLattice
from qlinks.operators import (
    SpinOneXYBondOperator,
    spin_one_lower_amplitude,
    spin_one_raise_amplitude,
)
from qlinks.operators.base import OperatorAction
from qlinks.variables import LocalSpace, VariableLayout


def test_spin_one_raise_lower_amplitudes() -> None:
    assert np.isclose(spin_one_raise_amplitude(-1), np.sqrt(2))
    assert np.isclose(spin_one_raise_amplitude(0), np.sqrt(2))
    assert spin_one_raise_amplitude(1) == 0.0

    assert spin_one_lower_amplitude(-1) == 0.0
    assert np.isclose(spin_one_lower_amplitude(0), np.sqrt(2))
    assert np.isclose(spin_one_lower_amplitude(1), np.sqrt(2))


def test_spin_one_xy_bond_affected_variables() -> None:
    lattice = ChainLattice(2, boundary_condition="open")
    layout = VariableLayout.from_lattice_sites(
        lattice,
        LocalSpace.spin_one(),
    )

    op = SpinOneXYBondOperator(
        layout=layout,
        lattice=lattice,
        link_id=0,
        coefficient=1.0,
    )

    np.testing.assert_array_equal(
        np.sort(op.affected_variables()),
        np.array([0, 1], dtype=np.int64),
    )


@pytest.mark.parametrize(
    "config, expected_configs",
    [
        (
            np.array([-1, 1], dtype=np.int64),
            [np.array([0, 0], dtype=np.int64)],
        ),
        (
            np.array([1, -1], dtype=np.int64),
            [np.array([0, 0], dtype=np.int64)],
        ),
        (
            np.array([0, 0], dtype=np.int64),
            [
                np.array([1, -1], dtype=np.int64),
                np.array([-1, 1], dtype=np.int64),
            ],
        ),
        (
            np.array([1, 1], dtype=np.int64),
            [],
        ),
        (
            np.array([-1, -1], dtype=np.int64),
            [],
        ),
    ],
)
def test_spin_one_xy_bond_apply(
    config: np.ndarray,
    expected_configs: list[np.ndarray],
) -> None:
    lattice = ChainLattice(2, boundary_condition="open")
    layout = VariableLayout.from_lattice_sites(
        lattice,
        LocalSpace.spin_one(),
    )

    op = SpinOneXYBondOperator(
        layout=layout,
        lattice=lattice,
        link_id=0,
        coefficient=2.0,
    )

    actions = op.apply(config)

    assert len(actions) == len(expected_configs)

    actual_configs = [action.config for action in actions]

    for expected in expected_configs:
        assert any(np.array_equal(actual, expected) for actual in actual_configs)

    for action in actions:
        assert isinstance(action, OperatorAction)
        # For spin-1 allowed exchange, matrix element is J_xy.
        np.testing.assert_allclose(
            action.coefficient,
            complex(2.0),
            atol=1e-12,
        )


def test_spin_one_xy_bond_apply_does_not_mutate_input() -> None:
    lattice = ChainLattice(2, boundary_condition="open")
    layout = VariableLayout.from_lattice_sites(
        lattice,
        LocalSpace.spin_one(),
    )

    op = SpinOneXYBondOperator(
        layout=layout,
        lattice=lattice,
        link_id=0,
        coefficient=1.0,
    )

    config = np.array([0, 0], dtype=np.int64)
    original = config.copy()

    _ = op.apply(config)

    np.testing.assert_array_equal(config, original)


def test_spin_one_xy_bond_rejects_wrong_local_space() -> None:
    lattice = ChainLattice(2, boundary_condition="open")
    layout = VariableLayout.from_lattice_sites(
        lattice,
        LocalSpace.binary(),
    )

    with pytest.raises(ValueError, match=r"\{-1, 0, \+1\}"):
        SpinOneXYBondOperator(
            layout=layout,
            lattice=lattice,
            link_id=0,
            coefficient=1.0,
        )


def test_spin_one_xy_bond_validates_config_values() -> None:
    lattice = ChainLattice(2, boundary_condition="open")
    layout = VariableLayout.from_lattice_sites(
        lattice,
        LocalSpace.spin_one(),
    )

    op = SpinOneXYBondOperator(
        layout=layout,
        lattice=lattice,
        link_id=0,
        coefficient=1.0,
    )

    bad_config = np.array([2, 0], dtype=np.int64)

    with pytest.raises(ValueError):
        op.apply(bad_config)


def test_operator_variable_indices_accessor_returns_copy() -> None:
    lattice = ChainLattice(2, boundary_condition="open")
    layout = VariableLayout.from_lattice_sites(
        lattice,
        LocalSpace.spin_one(),
    )

    op = SpinOneXYBondOperator(
        layout=layout,
        lattice=lattice,
        link_id=0,
        coefficient=1.0,
    )

    indices = op.variable_indices
    indices[0] = 999

    np.testing.assert_array_equal(
        op.variable_indices,
        np.array([0, 1], dtype=np.int64),
    )
