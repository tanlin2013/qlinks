"""Manual integration tests for square-lattice QDM/QLM caged states.

Run manually with:

    pytest tests/caging/manual/test_square_qdm_qlm_cages.py -m manual

These tests are intentionally not part of the normal CI path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pytest

from qlinks.caging import (
    CageSearchConfig,
    CageSearcher,
)
from qlinks.models import (
    SquareQDMModel,
    SquareQLMModel,
)

ModelName = Literal["qdm", "qlm"]


@dataclass(frozen=True)
class SquareCageCase:
    """Expected caging data for one square-lattice model."""

    model_name: ModelName
    lattice_shape: tuple[int, int]
    expected_counts_by_signature: dict[tuple[int, int], int]
    max_support_size: int
    winding_x: int = 0
    winding_y: int = 0


# Fill these with your known small-size counts.
#
# I leave them as explicit test data rather than deriving them inside the test,
# because this is an integration/regression test.
SQUARE_CAGE_CASES = [
    SquareCageCase(
        model_name="qdm",
        lattice_shape=(4, 4),
        winding_x=0,
        winding_y=0,
        expected_counts_by_signature={
            (0, 4): 9,
            (0, 6): 1,
        },
        max_support_size=48,
    ),
    SquareCageCase(
        model_name="qlm",
        lattice_shape=(4, 4),
        winding_x=0,
        winding_y=0,
        expected_counts_by_signature={
            (0, 8): 26,
            (0, 6): 12,
            (2, 8): 6,
            (-2, 8): 6,
        },
        max_support_size=224,
    ),
]


def _build_square_model_result(square_case: SquareCageCase):
    """Build a square QDM/QLM model result."""
    lattice_size_x, lattice_size_y = square_case.lattice_shape

    if square_case.model_name == "qdm":
        model = SquareQDMModel(
            lx=lattice_size_x,
            ly=lattice_size_y,
            boundary_condition="periodic",
            winding_x=square_case.winding_x,
            winding_y=square_case.winding_y,
            winding_convention="electric",
            kinetic=1.0,
            potential=1.0,
        )
        builder_name = "sparse"
    elif square_case.model_name == "qlm":
        model = SquareQLMModel(
            lx=lattice_size_x,
            ly=lattice_size_y,
            boundary_condition="periodic",
            winding_x=square_case.winding_x,
            winding_y=square_case.winding_y,
            charges=0,
            kinetic=1.0,
            potential=1.0,
        )
        builder_name = "bitmask"
    else:
        raise ValueError(f"Unsupported model_name: {square_case.model_name}")

    return model.build(
        basis_solver="dfs",
        builder=builder_name,
        backend="scipy",
        sort_basis=True,
    )


@pytest.mark.manual
@pytest.mark.parametrize("square_case", SQUARE_CAGE_CASES)
def test_square_qdm_qlm_cage_counts_by_signature(
    square_case: SquareCageCase,
) -> None:
    build_result = _build_square_model_result(square_case)

    search_type = "qdm" if square_case.model_name == "qdm" else "qlm"

    searcher = CageSearcher.from_model_build_result(
        build_result,
        config=CageSearchConfig(
            search_type=search_type,
            tolerance=1e-10,
        ),
    )

    result = searcher.run()

    assert result.counts_by_signature == square_case.expected_counts_by_signature
