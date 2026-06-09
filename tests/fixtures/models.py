from __future__ import annotations

import pytest

from qlinks.models import (
    HoneycombQDMModel,
    HoneycombQLMModel,
    SquareQDMModel,
    SquareQLMModel,
    TriangularQDMModel,
)


@pytest.fixture
def square_qdm_2x2_open() -> SquareQDMModel:
    return SquareQDMModel(
        lx=2,
        ly=2,
        boundary_condition="open",
        coup_kin=-1.0,
        coup_pot=2.0,
    )


@pytest.fixture
def square_qlm_2x2_open_zero_charge() -> SquareQLMModel:
    return SquareQLMModel(
        lx=2,
        ly=2,
        boundary_condition="open",
        coup_kin=-1.0,
        coup_pot=2.0,
        charges=0,
    )


@pytest.fixture
def square_qlm_4x2_staggered_pbc_w00() -> SquareQLMModel:
    return SquareQLMModel.from_staggered_background(
        lx=4,
        ly=2,
        boundary_condition="periodic",
        coup_kin=-1.0,
        coup_pot=0.0,
        winding_x=0,
        winding_y=0,
    )


@pytest.fixture
def triangular_qdm_3x3_pbc_z2() -> TriangularQDMModel:
    return TriangularQDMModel(
        lx=3,
        ly=3,
        boundary_condition="periodic",
        winding_a=0,
        winding_b=1,
    )


@pytest.fixture
def honeycomb_qdm_2x2_open() -> HoneycombQDMModel:
    return HoneycombQDMModel(
        lx=2,
        ly=2,
        boundary_condition="open",
        coup_kin=-1.0,
        coup_pot=1.0,
    )


@pytest.fixture
def honeycomb_qlm_3x3_pbc_fractional_winding() -> HoneycombQLMModel:
    return HoneycombQLMModel(
        lx=3,
        ly=3,
        boundary_condition="periodic",
        charges=0,
        winding_x="1/2",
        winding_y="1/2",
    )
