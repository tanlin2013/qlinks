from __future__ import annotations

import pytest

from qlinks.lattice import ChainLattice, SquareLattice


@pytest.fixture(scope="session")
def chain_3_open() -> ChainLattice:
    return ChainLattice(3, boundary_condition="open")


@pytest.fixture(scope="session")
def chain_4_open() -> ChainLattice:
    return ChainLattice(4, boundary_condition="open")


@pytest.fixture(scope="session")
def square_2x2_open() -> SquareLattice:
    return SquareLattice(2, 2, boundary_condition="open")
