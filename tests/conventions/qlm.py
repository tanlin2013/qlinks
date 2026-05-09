import numpy as np
import pytest

from qlinks.conventions import (
    square_qdm_staggered_charges,
    staggered_charges_from_sites,
)
from qlinks.lattice import ChainLattice, SquareLattice


def test_square_qdm_staggered_charges_2_by_2_even_positive() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")

    charges = square_qdm_staggered_charges(
        lattice,
        magnitude=2,
        convention="even_positive",
    )

    expected = np.array([2, -2, -2, 2], dtype=np.int64)

    np.testing.assert_array_equal(charges, expected)


def test_square_qdm_staggered_charges_2_by_2_odd_positive() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")

    charges = square_qdm_staggered_charges(
        lattice,
        magnitude=2,
        convention="odd_positive",
    )

    expected = np.array([-2, 2, 2, -2], dtype=np.int64)

    np.testing.assert_array_equal(charges, expected)


def test_staggered_charges_chain() -> None:
    lattice = ChainLattice(4, boundary_condition="open")

    charges = staggered_charges_from_sites(
        lattice,
        magnitude=1,
        convention="even_positive",
    )

    expected = np.array([1, -1, 1, -1], dtype=np.int64)

    np.testing.assert_array_equal(charges, expected)


def test_staggered_charges_rejects_negative_magnitude() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")

    with pytest.raises(ValueError, match="magnitude"):
        square_qdm_staggered_charges(
            lattice,
            magnitude=-1,
        )


def test_staggered_charges_rejects_bad_convention() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")

    with pytest.raises(ValueError, match="convention"):
        square_qdm_staggered_charges(
            lattice,
            convention="bad",  # type: ignore[arg-type]
        )
