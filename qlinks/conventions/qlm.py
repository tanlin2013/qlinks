from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt

from qlinks.lattice import LatticeGraph, SquareLattice

ChargeNormalization = Literal["integer_flux", "spin_half"]
SublatticeSignConvention = Literal["even_positive", "odd_positive"]


def staggered_charges_from_sites(
    lattice: LatticeGraph,
    *,
    magnitude: int = 1,
    convention: SublatticeSignConvention = "even_positive",
) -> npt.NDArray[np.int64]:
    """Generate staggered background charges on lattice sites.

    The charge sign is assigned by the parity of the unit-cell coordinate,
    ``eta(r) = (-1) ** sum(cell coordinates)``.  On a square lattice this is
    ``eta(x, y) = (-1) ** (x + y)``.

    Args:
        lattice: Lattice whose sites carry the Gauss-law charges.
        magnitude: Absolute value of the staggered charge.
        convention: Which sublattice receives the positive charge.
            ``"even_positive"`` assigns ``+magnitude`` to the even sublattice;
            ``"odd_positive"`` reverses the sign.

    Returns:
        Integer charge array ordered by site id.
    """
    if magnitude < 0:
        raise ValueError("magnitude must be non-negative.")

    if convention not in ("even_positive", "odd_positive"):
        raise ValueError("convention must be 'even_positive' or 'odd_positive'.")

    charges = np.empty(lattice.num_sites, dtype=np.int64)

    for site in lattice.sites:
        parity = sum(int(c) for c in site.cell) % 2

        if convention == "even_positive":
            sign = 1 if parity == 0 else -1
        else:
            sign = -1 if parity == 0 else 1

        charges[int(site.id)] = sign * int(magnitude)

    return charges


def square_qdm_staggered_charges(
    lattice: SquareLattice,
    *,
    magnitude: int | None = None,
    charge_normalization: ChargeNormalization = "spin_half",
    convention: SublatticeSignConvention = "even_positive",
) -> npt.NDArray[np.int64]:
    """Return staggered charges for the square QDM-to-QLM mapping.

    With spin-half flux variables ``E_l in {-1, +1}``, the close-packed QDM
    constraint maps to a staggered Gauss law with charge magnitude one.

    Args:
        lattice: Square lattice whose sites receive charges.
        magnitude: Optional explicit charge magnitude.  If omitted, the
            magnitude is inferred from ``charge_normalization``.
        charge_normalization: ``"spin_half"`` gives user charges ``±1``;
            ``"integer_flux"`` gives user charges ``±2``.
        convention: Which sublattice receives the positive charge.

    Returns:
        Integer charge array ordered by site id.
    """
    if magnitude is None:
        if charge_normalization == "spin_half":
            magnitude = 1
        elif charge_normalization == "integer_flux":
            magnitude = 2
        else:
            raise ValueError("charge_normalization must be 'integer_flux' or 'spin_half'.")

    return staggered_charges_from_sites(
        lattice,
        magnitude=magnitude,
        convention=convention,
    )
