from __future__ import annotations

import pytest

from qlinks.models import HoneycombQDMModel, SquareQDMModel


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
def honeycomb_qdm_2x2_open() -> HoneycombQDMModel:
    return HoneycombQDMModel(
        lx=2,
        ly=2,
        boundary_condition="open",
        coup_kin=-1.0,
        coup_pot=1.0,
    )
