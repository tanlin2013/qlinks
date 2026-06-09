from __future__ import annotations

import pytest

from qlinks.lattice import (
    ChainLattice,
    HoneycombLattice,
    SquareLattice,
    TriangularLattice,
)


@pytest.fixture(scope="session")
def chain_3_open() -> ChainLattice:
    return ChainLattice(3, boundary_condition="open")


@pytest.fixture(scope="session")
def chain_4_open() -> ChainLattice:
    return ChainLattice(4, boundary_condition="open")


@pytest.fixture(scope="session")
def square_2x2_open() -> SquareLattice:
    return SquareLattice(2, 2, boundary_condition="open")


@pytest.fixture(scope="session")
def square_2x2_pbc() -> SquareLattice:
    return SquareLattice(2, 2, boundary_condition="periodic")


@pytest.fixture(scope="session")
def square_4x4_pbc() -> SquareLattice:
    return SquareLattice(4, 4, boundary_condition="periodic")


@pytest.fixture(scope="session")
def triangular_3x3_pbc() -> TriangularLattice:
    return TriangularLattice(3, 3, boundary_condition="periodic")


@pytest.fixture(scope="session")
def honeycomb_3x3_pbc() -> HoneycombLattice:
    return HoneycombLattice(3, 3, boundary_condition="periodic")
