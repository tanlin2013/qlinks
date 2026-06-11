from __future__ import annotations

import pytest

from qlinks.models import SquareQDMModel, SquareQLMModel
from tests.helpers.assertions import (
    assert_same_physical_flux_basis_order,
    assert_sparse_allclose,
)


def _plaquette_coupling_dict(model, values: list[complex]) -> dict[int, complex]:
    plaquette_ids = [int(p) for p in model.plaquette_ids()]
    assert len(plaquette_ids) >= len(values)

    return {
        plaquette_id: complex(value)
        for plaquette_id, value in zip(plaquette_ids, values, strict=False)
    }


def _constant_coupling_from_plaquettes(model, value: complex) -> dict[int, complex]:
    return {int(p): complex(value) for p in model.plaquette_ids()}


@pytest.mark.parametrize("builder", ["sparse", "bitmask"])
def test_square_qdm_constant_plaquette_coup_kin_matches_scalar(builder: str) -> None:
    scalar_model = SquareQDMModel(
        lx=3,
        ly=3,
        boundary_condition="open",
        coup_kin=-1.25,
        coup_pot=0.0,
    )
    mapped_model = SquareQDMModel(
        lx=3,
        ly=3,
        boundary_condition="open",
        coup_kin=_constant_coupling_from_plaquettes(scalar_model, -1.25),
        coup_pot=0.0,
    )

    scalar_result = scalar_model.build(builder=builder)
    mapped_result = mapped_model.build(builder=builder)

    assert_sparse_allclose(mapped_result.kinetic, scalar_result.kinetic)
    assert_sparse_allclose(mapped_result.hamiltonian, scalar_result.hamiltonian)


@pytest.mark.parametrize("builder", ["sparse", "bitmask"])
def test_square_qdm_dict_and_callable_coup_kin_match(builder: str) -> None:
    reference_model = SquareQDMModel(
        lx=3,
        ly=3,
        boundary_condition="open",
        coup_kin=-1.0,
        coup_pot=0.0,
    )

    coup_kin = {
        int(p): (-1.0 + 0.25 * index) for index, p in enumerate(reference_model.plaquette_ids())
    }

    def coup_kin_fn(plaquette_id: int) -> complex:
        return coup_kin[int(plaquette_id)]

    dict_model = SquareQDMModel(
        lx=3,
        ly=3,
        boundary_condition="open",
        coup_kin=coup_kin,
        coup_pot=0.0,
    )
    callable_model = SquareQDMModel(
        lx=3,
        ly=3,
        boundary_condition="open",
        coup_kin=coup_kin_fn,
        coup_pot=0.0,
    )

    dict_result = dict_model.build(builder=builder)
    callable_result = callable_model.build(builder=builder)

    assert_sparse_allclose(callable_result.kinetic, dict_result.kinetic)
    assert_sparse_allclose(callable_result.hamiltonian, dict_result.hamiltonian)


def test_square_qdm_sparse_and_bitmask_match_with_plaquette_coup_kin() -> None:
    reference_model = SquareQDMModel(
        lx=3,
        ly=3,
        boundary_condition="open",
        coup_kin=-1.0,
        coup_pot=0.0,
    )

    coup_kin = {
        int(p): (-1.0 + 0.2j * index) for index, p in enumerate(reference_model.plaquette_ids())
    }

    sparse_model = SquareQDMModel(
        lx=3,
        ly=3,
        boundary_condition="open",
        coup_kin=coup_kin,
        coup_pot=0.0,
    )
    bitmask_model = SquareQDMModel(
        lx=3,
        ly=3,
        boundary_condition="open",
        coup_kin=coup_kin,
        coup_pot=0.0,
    )

    sparse_result = sparse_model.build(builder="sparse", sort_basis=False)
    bitmask_result = bitmask_model.build(builder="bitmask", sort_basis=False)

    assert_same_physical_flux_basis_order(sparse_result, bitmask_result)
    assert_sparse_allclose(bitmask_result.kinetic, sparse_result.kinetic)
    assert_sparse_allclose(bitmask_result.hamiltonian, sparse_result.hamiltonian)


@pytest.mark.parametrize("builder", ["sparse", "bitmask"])
def test_square_qdm_dict_and_callable_coup_pot_match(builder: str) -> None:
    reference_model = SquareQDMModel(
        lx=3,
        ly=3,
        boundary_condition="open",
        coup_kin=0.0,
        coup_pot=1.0,
    )

    coup_pot = {
        int(p): (0.5 + 0.1 * index) for index, p in enumerate(reference_model.plaquette_ids())
    }

    def coup_pot_fn(plaquette_id: int) -> complex:
        return coup_pot[int(plaquette_id)]

    dict_model = SquareQDMModel(
        lx=3,
        ly=3,
        boundary_condition="open",
        coup_kin=0.0,
        coup_pot=coup_pot,
    )
    callable_model = SquareQDMModel(
        lx=3,
        ly=3,
        boundary_condition="open",
        coup_kin=0.0,
        coup_pot=coup_pot_fn,
    )

    dict_result = dict_model.build(builder=builder)
    callable_result = callable_model.build(builder=builder)

    assert_sparse_allclose(callable_result.potential, dict_result.potential)
    assert_sparse_allclose(callable_result.hamiltonian, dict_result.hamiltonian)


def _square_qlm_model(
    *,
    coup_kin,
    coup_pot,
) -> SquareQLMModel:
    model = SquareQLMModel.from_staggered_background(
        lx=4,
        ly=2,
        boundary_condition="periodic",
        coup_kin=coup_kin,
        coup_pot=coup_pot,
        winding_x=0,
        winding_y=0,
    )

    basis = model.build_basis()
    assert basis.n_states > 0

    return model


@pytest.mark.parametrize("builder", ["sparse", "bitmask"])
def test_square_qlm_constant_plaquette_coup_kin_matches_scalar(builder: str) -> None:
    scalar_model = _square_qlm_model(
        coup_kin=-0.75,
        coup_pot=0.0,
    )
    mapped_model = _square_qlm_model(
        coup_kin=_constant_coupling_from_plaquettes(scalar_model, -0.75),
        coup_pot=0.0,
    )

    scalar_result = scalar_model.build(builder=builder)
    mapped_result = mapped_model.build(builder=builder)

    assert_sparse_allclose(mapped_result.kinetic, scalar_result.kinetic)
    assert_sparse_allclose(mapped_result.hamiltonian, scalar_result.hamiltonian)


@pytest.mark.parametrize("builder", ["sparse", "bitmask"])
def test_square_qlm_dict_and_callable_coup_kin_match(builder: str) -> None:
    reference_model = _square_qlm_model(
        coup_kin=-1.0,
        coup_pot=0.0,
    )

    coup_kin = {
        int(p): (-1.0 + 0.15 * index) for index, p in enumerate(reference_model.plaquette_ids())
    }

    def coup_kin_fn(plaquette_id: int) -> complex:
        return coup_kin[int(plaquette_id)]

    dict_model = _square_qlm_model(
        coup_kin=coup_kin,
        coup_pot=0.0,
    )
    callable_model = _square_qlm_model(
        coup_kin=coup_kin_fn,
        coup_pot=0.0,
    )

    dict_result = dict_model.build(builder=builder)
    callable_result = callable_model.build(builder=builder)

    assert_sparse_allclose(callable_result.kinetic, dict_result.kinetic)
    assert_sparse_allclose(callable_result.hamiltonian, dict_result.hamiltonian)


def test_square_qlm_sparse_and_bitmask_match_with_plaquette_coup_kin() -> None:
    reference_model = _square_qlm_model(
        coup_kin=-1.0,
        coup_pot=0.0,
    )

    coup_kin = {
        int(p): (-1.0 + 0.1j * index) for index, p in enumerate(reference_model.plaquette_ids())
    }

    sparse_model = _square_qlm_model(
        coup_kin=coup_kin,
        coup_pot=0.0,
    )
    bitmask_model = _square_qlm_model(
        coup_kin=coup_kin,
        coup_pot=0.0,
    )

    sparse_result = sparse_model.build(builder="sparse", sort_basis=True)
    bitmask_result = bitmask_model.build(builder="bitmask", sort_basis=True)

    assert_same_physical_flux_basis_order(sparse_result, bitmask_result)
    assert_sparse_allclose(bitmask_result.kinetic, sparse_result.kinetic)
    assert_sparse_allclose(bitmask_result.hamiltonian, sparse_result.hamiltonian)


@pytest.mark.parametrize("builder", ["sparse", "bitmask"])
def test_square_qlm_dict_and_callable_coup_pot_match(builder: str) -> None:
    reference_model = _square_qlm_model(
        coup_kin=0.0,
        coup_pot=1.0,
    )

    coup_pot = {
        int(p): (0.25 + 0.2 * index) for index, p in enumerate(reference_model.plaquette_ids())
    }

    def coup_pot_fn(plaquette_id: int) -> complex:
        return coup_pot[int(plaquette_id)]

    dict_model = _square_qlm_model(
        coup_kin=0.0,
        coup_pot=coup_pot,
    )
    callable_model = _square_qlm_model(
        coup_kin=0.0,
        coup_pot=coup_pot_fn,
    )

    dict_result = dict_model.build(builder=builder)
    callable_result = callable_model.build(builder=builder)

    assert_sparse_allclose(callable_result.potential, dict_result.potential)
    assert_sparse_allclose(callable_result.hamiltonian, dict_result.hamiltonian)


@pytest.mark.parametrize("model_factory", [SquareQDMModel, _square_qlm_model])
def test_missing_plaquette_coupling_key_raises(model_factory) -> None:
    if model_factory is SquareQDMModel:
        model = SquareQDMModel(
            lx=3,
            ly=3,
            boundary_condition="open",
            coup_kin={0: -1.0},
            coup_pot=0.0,
        )
    else:
        model = _square_qlm_model(
            coup_kin={0: -1.0},
            coup_pot=0.0,
        )

    with pytest.raises(KeyError, match="coup_kin"):
        model.build(builder="sparse")
