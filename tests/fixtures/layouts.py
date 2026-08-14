from __future__ import annotations

import pytest

from qlinks.variables import LocalSpace, VariableLayout


@pytest.fixture(scope="session")
def binary_site_layout_3() -> VariableLayout:
    return VariableLayout.from_sites(3, LocalSpace.binary())


@pytest.fixture(scope="session")
def binary_site_layout_5() -> VariableLayout:
    return VariableLayout.from_sites(5, LocalSpace.binary())


@pytest.fixture(scope="session")
def spin_half_chain_3_link_layout(chain_3_open) -> VariableLayout:
    return VariableLayout.from_lattice_links(chain_3_open, LocalSpace.spin_half_flux())


@pytest.fixture(scope="session")
def binary_chain_3_link_layout(chain_3_open) -> VariableLayout:
    return VariableLayout.from_lattice_links(chain_3_open, LocalSpace.binary())


@pytest.fixture(scope="session")
def binary_chain_4_link_layout(chain_4_open) -> VariableLayout:
    return VariableLayout.from_lattice_links(chain_4_open, LocalSpace.binary())


@pytest.fixture(scope="session")
def square_2x2_open_binary_link_layout(square_2x2_open):
    return VariableLayout.from_lattice_links(
        square_2x2_open,
        LocalSpace.binary(),
    )
