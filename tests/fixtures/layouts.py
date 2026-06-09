from __future__ import annotations

import pytest

from qlinks.variables import LocalSpace, VariableLayout


@pytest.fixture(scope="session")
def square_2x2_pbc_flux_layout(square_2x2_pbc):
    return VariableLayout.from_lattice_links(
        square_2x2_pbc,
        LocalSpace.spin_half_flux(),
    )


@pytest.fixture(scope="session")
def square_4x4_pbc_binary_link_layout(square_4x4_pbc):
    return VariableLayout.from_lattice_links(
        square_4x4_pbc,
        LocalSpace.binary(),
    )


@pytest.fixture(scope="session")
def honeycomb_3x3_pbc_flux_layout(honeycomb_3x3_pbc):
    return VariableLayout.from_lattice_links(
        honeycomb_3x3_pbc,
        LocalSpace.spin_half_flux(),
    )


@pytest.fixture(scope="session")
def triangular_3x3_pbc_binary_link_layout(triangular_3x3_pbc):
    return VariableLayout.from_lattice_links(
        triangular_3x3_pbc,
        LocalSpace.binary(),
    )
