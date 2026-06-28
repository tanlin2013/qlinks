from dataclasses import dataclass
from typing import Callable

import pytest

from qlinks.models import (
    HoneycombQDMModel,
    HoneycombQLMModel,
    KagomeQDMModel,
    KagomeQLMModel,
    QDMModel,
    QLMModel,
    TriangularQDMModel,
    TriangularQLMModel,
)


@dataclass(frozen=True)
class ModelSmokeCase:
    name: str
    factory: Callable[[], object]
    expected_plaquette_count: int | None = None


MODEL_SMOKE_CASES = [
    ModelSmokeCase(
        name="triangular_qdm_open",
        factory=lambda: TriangularQDMModel(
            lx=2,
            ly=2,
            boundary_condition="open",
            coup_kin=-1.0,
            coup_pot=0.0,
        ),
    ),
    ModelSmokeCase(
        name="honeycomb_qdm_open",
        factory=lambda: HoneycombQDMModel(
            lx=2,
            ly=2,
            boundary_condition="open",
            coup_kin=-1.0,
            coup_pot=0.0,
        ),
    ),
    ModelSmokeCase(
        name="triangular_qlm_open",
        factory=lambda: TriangularQLMModel(
            lx=2,
            ly=2,
            boundary_condition="open",
            coup_kin=-1.0,
            coup_pot=0.0,
            charges=0,
        ),
    ),
    ModelSmokeCase(
        name="honeycomb_qlm_open",
        factory=lambda: HoneycombQLMModel(
            lx=2,
            ly=2,
            boundary_condition="open",
            coup_kin=-1.0,
            coup_pot=0.0,
            charges=0,
        ),
        expected_plaquette_count=1,
    ),
]


@pytest.mark.parametrize("case", MODEL_SMOKE_CASES, ids=lambda c: c.name)
def test_model_smoke(case: ModelSmokeCase) -> None:
    model = case.factory()
    basis = model.build_basis(solver="dfs", sort=True)

    assert basis.n_states >= 0

    if case.expected_plaquette_count is None:
        assert len(model.plaquette_ids()) > 0
    else:
        assert len(model.plaquette_ids()) == case.expected_plaquette_count


def test_kagome_models_expose_z2_winding_sectors() -> None:
    qdm = KagomeQDMModel(lx=2, ly=3, boundary_condition="periodic")
    qlm = KagomeQLMModel(lx=2, ly=2, boundary_condition="periodic")

    assert qdm.allowed_sector_labels() == {"winding_a": (0, 1), "winding_b": (0, 1)}
    assert qlm.allowed_sector_labels() == {"winding_a": (0, 1), "winding_b": (0, 1)}

    qdm_sector = KagomeQDMModel(
        lx=2,
        ly=3,
        boundary_condition="periodic",
        winding_a=0,
        winding_b=0,
    )
    assert len(qdm_sector.make_sectors()) == 2


def test_generic_model_helpers_construct_kagome_models() -> None:
    assert isinstance(QDMModel.kagome(2, 3, boundary_condition="periodic"), KagomeQDMModel)
    assert isinstance(QLMModel.kagome(2, 2, boundary_condition="periodic"), KagomeQLMModel)
