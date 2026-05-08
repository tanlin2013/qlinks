import numpy as np
import pytest

from qlinks.variables import LocalSpace


def test_binary_local_space() -> None:
    space = LocalSpace.binary()

    assert space.dim == 2
    np.testing.assert_array_equal(space.values, np.array([0, 1]))
    assert space.contains(0)
    assert space.contains(1)
    assert not space.contains(2)


def test_spin_half_flux_space() -> None:
    space = LocalSpace.spin_half_flux()

    assert space.dim == 2
    np.testing.assert_array_equal(space.values, np.array([-1, 1]))
    assert space.contains(-1)
    assert space.contains(1)
    assert not space.contains(0)


def test_reject_empty_local_space() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        LocalSpace.from_values([])


def test_reject_duplicate_values() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        LocalSpace.from_values([0, 1, 1])


def test_reject_non_1d_values() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        LocalSpace(np.array([[0, 1]]))


def test_validate_value() -> None:
    space = LocalSpace.binary()

    space.validate_value(0)
    space.validate_value(1)

    with pytest.raises(ValueError, match="not allowed"):
        space.validate_value(2)


def test_validate_array() -> None:
    space = LocalSpace.binary()

    space.validate_array(np.array([0, 1, 0]))

    with pytest.raises(ValueError, match="outside local space"):
        space.validate_array(np.array([0, 2, 1]))


def test_value_code_conversion() -> None:
    space = LocalSpace.from_values([-1, 0, 1])

    assert space.value_to_code(-1) == 0
    assert space.value_to_code(0) == 1
    assert space.value_to_code(1) == 2

    assert space.code_to_value(0) == -1
    assert space.code_to_value(1) == 0
    assert space.code_to_value(2) == 1

    with pytest.raises(ValueError, match="not allowed"):
        space.value_to_code(3)

    with pytest.raises(ValueError, match="outside valid range"):
        space.code_to_value(3)
    