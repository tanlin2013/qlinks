import numpy as np
import pytest

from qlinks.basis import Basis
from qlinks.variables import LocalSpace, VariableLayout


def test_basis_from_states() -> None:
    layout = VariableLayout.from_sites(3, LocalSpace.binary())

    states = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ],
        dtype=np.int64,
    )

    basis = Basis.from_states(layout, states)

    assert basis.n_states == 3
    assert basis.n_variables == 3

    np.testing.assert_array_equal(basis.state(1), np.array([1, 0, 0]))

    assert basis.get_index(np.array([0, 0, 0])) == 0
    assert basis.get_index(np.array([1, 0, 0])) == 1
    assert basis.get_index(np.array([0, 1, 0])) == 2
    assert basis.get_index(np.array([1, 1, 1])) is None


def test_basis_contains() -> None:
    layout = VariableLayout.from_sites(2, LocalSpace.binary())
    basis = Basis.from_states(
        layout,
        np.array(
            [
                [0, 0],
                [1, 0],
            ]
        ),
    )

    assert np.array([0, 0]) in basis
    assert np.array([1, 1]) not in basis


def test_basis_empty() -> None:
    layout = VariableLayout.from_sites(2, LocalSpace.binary())
    basis = Basis.empty(layout)

    assert basis.n_states == 0
    assert basis.states.shape == (0, 2)


def test_basis_rejects_wrong_shape() -> None:
    layout = VariableLayout.from_sites(3, LocalSpace.binary())

    with pytest.raises(ValueError, match="Expected states with 3 variables"):
        Basis.from_states(layout, np.array([[0, 1]]))


def test_basis_rejects_invalid_value() -> None:
    layout = VariableLayout.from_sites(2, LocalSpace.binary())

    with pytest.raises(ValueError, match="outside local space|not allowed"):
        Basis.from_states(layout, np.array([[0, 2]]))


def test_basis_duplicate_states_rejected() -> None:
    layout = VariableLayout.from_sites(2, LocalSpace.binary())

    with pytest.raises(ValueError, match="Duplicate configuration"):
        Basis.from_states(
            layout,
            np.array(
                [
                    [0, 0],
                    [0, 0],
                ]
            ),
        )


def test_require_index() -> None:
    layout = VariableLayout.from_sites(2, LocalSpace.binary())
    basis = Basis.from_states(layout, np.array([[0, 0], [1, 0]]))

    assert basis.require_index(np.array([1, 0])) == 1

    with pytest.raises(KeyError):
        basis.require_index(np.array([1, 1]))


def test_iter_states_copy() -> None:
    layout = VariableLayout.from_sites(2, LocalSpace.binary())
    basis = Basis.from_states(layout, np.array([[0, 0], [1, 0]]))

    states = list(basis.iter_states(copy=True))
    states[0][0] = 1

    np.testing.assert_array_equal(basis.state(0), np.array([0, 0]))
