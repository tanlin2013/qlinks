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
    """
    Generate staggered background charges on lattice sites.

    The charge is assigned by the parity of the unit-cell coordinate:

        eta(r) = (-1) ** sum(cell coordinates)

    For a square lattice, this is simply:

        eta(x, y) = (-1) ** (x + y)

    Parameters
    ----------
    lattice:
        Lattice whose sites carry the Gauss-law charges.

    magnitude:
        Absolute value of the staggered charge.

    convention:
        "even_positive":
            even sublattice gets +magnitude,
            odd sublattice gets -magnitude.

        "odd_positive":
            even sublattice gets -magnitude,
            odd sublattice gets +magnitude.

    Returns
    -------
    charges:
        Integer array of shape (lattice.num_sites,), ordered by site id.
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
    """
    Staggered charges for the square-lattice QDM-to-QLM mapping.

    For the current spin-half flux convention E_l in {-1, +1}, the close-packed
    QDM constraint

        one dimer touching each site

    maps naturally to a staggered Gauss law with charge magnitude 1 on the
    square lattice.

    Parameters
    ----------
    lattice:
        SquareLattice instance.

    charge_normalization:
        "spin_half":
            The user-facing charges are ±1, matching the spin-half flux convention.

        "integer_flux":
            The user-facing charges are ±2, matching the integer-flux convention
            where E_l in {-2, 0, +2}.

    convention:
        Which sublattice receives the positive charge.

    Returns
    -------
    charges:
        Integer array ordered by site id.
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
