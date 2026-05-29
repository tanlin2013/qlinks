import numpy as np
import pytest

from qlinks.basis import Basis
from qlinks.builders import is_hermitian_sparse
from qlinks.conventions import square_qdm_staggered_charges
from qlinks.encoded import binary_encoded_basis_from_flux_basis
from qlinks.models import (
    HoneycombQLMModel,
    QLMModel,
    SquareQLMModel,
    TriangularQLMModel,
)


def test_square_qlm_manual_single_plaquette_sparse() -> None:
    model = SquareQLMModel(
        lx=2,
        ly=2,
        boundary_condition="open",
        coup_kin=-1.0,
        coup_pot=2.0,
        charges=0,
    )

    layout = model.layout
    plaquette_id = model.plaquette_ids()[0]

    link_ids = model.lattice.plaquette_links(plaquette_id)
    variable_indices = np.asarray(
        [layout.link_variable_index(int(link_id)) for link_id in link_ids],
        dtype=np.int64,
    )
    orientation_pattern = model.lattice.plaquette_orientations(plaquette_id)

    state_forward = -np.ones(layout.n_variables, dtype=np.int64)
    state_backward = -np.ones(layout.n_variables, dtype=np.int64)

    state_forward[variable_indices] = orientation_pattern
    state_backward[variable_indices] = -orientation_pattern

    basis = Basis.from_states(
        layout,
        np.array(
            [
                state_forward,
                state_backward,
            ],
            dtype=np.int64,
        ),
    )

    result = model.build(
        basis=basis,
        builder="sparse",
        on_missing="raise",
    )

    expected = np.array(
        [
            [2, -1],
            [-1, 2],
        ],
        dtype=np.complex128,
    )

    assert result.kinetic is not None
    assert result.potential is not None
    np.testing.assert_allclose(result.hamiltonian.toarray(), expected)
    assert is_hermitian_sparse(result.hamiltonian)


def test_square_qlm_bitmask_matches_sparse() -> None:
    model = SquareQLMModel(
        lx=2,
        ly=2,
        boundary_condition="open",
        coup_kin=-1.0,
        coup_pot=2.0,
        charges=0,
    )

    layout = model.layout
    plaquette_id = model.plaquette_ids()[0]
    orientation_pattern = model.lattice.plaquette_orientations(plaquette_id)

    flux_basis = Basis.from_states(
        layout,
        np.array(
            [
                orientation_pattern,
                -orientation_pattern,
            ],
            dtype=np.int64,
        ),
    )

    encoded_basis = binary_encoded_basis_from_flux_basis(flux_basis, sort=False)

    sparse_result = model.build(
        basis=flux_basis,
        builder="sparse",
    )

    bitmask_result = model.build(
        basis=encoded_basis,
        builder="bitmask",
    )

    np.testing.assert_allclose(
        sparse_result.hamiltonian.toarray(),
        bitmask_result.hamiltonian.toarray(),
    )


def test_square_qlm_optimized_matches_sparse_for_kinetic_only() -> None:
    model = SquareQLMModel(
        lx=2,
        ly=2,
        boundary_condition="open",
        coup_kin=-1.0,
        coup_pot=0.0,
        charges=0,
    )

    layout = model.layout
    plaquette_id = model.plaquette_ids()[0]
    orientation_pattern = model.lattice.plaquette_orientations(plaquette_id)

    basis = Basis.from_states(
        layout,
        np.array(
            [
                orientation_pattern,
                -orientation_pattern,
            ],
            dtype=np.int64,
        ),
    )

    sparse_result = model.build(
        basis=basis,
        builder="sparse",
    )

    optimized_result = model.build(
        basis=basis,
        builder="optimized",
    )

    np.testing.assert_allclose(
        sparse_result.hamiltonian.toarray(),
        optimized_result.hamiltonian.toarray(),
    )


def test_square_qlm_potential_rejects_optimized() -> None:
    model = SquareQLMModel(
        lx=2,
        ly=2,
        boundary_condition="open",
        coup_kin=-1.0,
        coup_pot=1.0,
        charges=0,
    )

    layout = model.layout
    plaquette_id = model.plaquette_ids()[0]
    orientation_pattern = model.lattice.plaquette_orientations(plaquette_id)

    basis = Basis.from_states(
        layout,
        np.array(
            [
                orientation_pattern,
                -orientation_pattern,
            ],
            dtype=np.int64,
        ),
    )

    import pytest

    with pytest.raises(NotImplementedError, match="potential"):
        model.build(
            basis=basis,
            builder="optimized",
        )


@pytest.mark.parametrize(
    (
        "lx",
        "ly",
        "winding_x",
        "winding_y",
        "charge_pattern",
        "charge_normalization",
        "expected_n_states",
    ),
    [
        # Zero-charge QLM, integer-flux convention.
        (4, 4, None, None, "zero", "integer_flux", 2970),
        (4, 4, 0, 0, "zero", "integer_flux", 990),
        # Staggered-background QLM matching square QDM.
        # Spin-half convention: user-facing charges are ±1.
        (4, 4, None, None, "staggered", "spin_half", 272),
        (4, 4, 0, 0, "staggered", "spin_half", 132),
    ],
)
def test_square_qlm_known_basis_counts(
    lx: int,
    ly: int,
    winding_x: int | None,
    winding_y: int | None,
    charge_pattern: str,
    charge_normalization: str,
    expected_n_states: int,
) -> None:
    model0 = SquareQLMModel(
        lx=lx,
        ly=ly,
        boundary_condition="periodic",
    )

    if charge_pattern == "zero":
        charges = 0
    elif charge_pattern == "staggered":
        charges = square_qdm_staggered_charges(
            model0.lattice,
            charge_normalization=charge_normalization,
            convention="even_positive",
        )
    else:
        raise ValueError(f"Unknown charge pattern: {charge_pattern}")

    model = SquareQLMModel(
        lx=lx,
        ly=ly,
        boundary_condition="periodic",
        winding_x=winding_x,
        winding_y=winding_y,
        charges=charges,
        charge_normalization=charge_normalization,
    )

    basis = model.build_basis(
        solver="dfs",
        sort=True,
    )

    assert basis.n_states == expected_n_states


@pytest.mark.parametrize("builder_name", ["sparse", "bitmask"])
def test_square_qlm_kinetic_nonzero(builder_name: str) -> None:
    model = SquareQLMModel(
        lx=4,
        ly=4,
        boundary_condition="periodic",
        winding_x=0,
        winding_y=0,
        charges=0,
        coup_kin=1.0,
        coup_pot=0.0,
    )

    build_result = model.build(
        basis_solver="dfs",
        builder=builder_name,
        backend="scipy",
        sort_basis=True,
        on_missing="raise",
    )

    assert build_result.kinetic.nnz > 0


def test_square_qlm_basis_smoke() -> None:
    charges = np.array([-1, 0, 0, 1], dtype=np.int64)

    model = SquareQLMModel(
        lx=2,
        ly=2,
        boundary_condition="open",
        coup_kin=-1.0,
        coup_pot=0.0,
        charges=charges,
    )

    basis = model.build_basis(solver="dfs", sort=True)

    assert basis.n_states >= 1


def test_square_qlm_from_qdm_staggered_background_defaults_to_spin_half_pm1() -> None:
    model = SquareQLMModel.from_staggered_background(
        lx=2,
        ly=2,
        boundary_condition="open",
    )

    np.testing.assert_array_equal(
        np.asarray(model.charges),
        np.array([1, -1, -1, 1], dtype=np.int64),
    )
    assert model.charge_normalization == "spin_half"


def test_triangular_qlm_smoke() -> None:
    model = TriangularQLMModel(
        lx=2,
        ly=2,
        boundary_condition="open",
        coup_kin=-1.0,
        coup_pot=0.0,
        charges=0,
    )

    basis = model.build_basis(
        solver="dfs",
        sort=True,
    )

    assert basis.n_states >= 0
    assert len(model.plaquette_ids()) > 0


def test_honeycomb_qlm_smoke() -> None:
    model = HoneycombQLMModel(
        lx=2,
        ly=2,
        boundary_condition="open",
        coup_kin=-1.0,
        coup_pot=0.0,
        charges=0,
    )

    basis = model.build_basis(
        solver="dfs",
        sort=True,
    )

    assert basis.n_states >= 0
    assert len(model.plaquette_ids()) == 1


def test_qlmmodel_factory_methods() -> None:
    tri = QLMModel.triangular(2, 2)
    honey = QLMModel.honeycomb(2, 2)

    assert isinstance(tri, TriangularQLMModel)
    assert isinstance(honey, HoneycombQLMModel)


def test_triangular_qlm_z2_sector_builds() -> None:
    model = TriangularQLMModel(
        lx=3,
        ly=3,
        boundary_condition="periodic",
        winding_a=0,
        winding_b=1,
        charges=0,
    )

    basis = model.build_basis(solver="dfs", sort=True)

    assert basis.n_states >= 0


def test_honeycomb_qlm_winding_sector_builds() -> None:
    model = HoneycombQLMModel(
        lx=3,
        ly=3,
        boundary_condition="periodic",
        coup_kin=-1.0,
        coup_pot=0.0,
        charges=0,
        winding_x="1/2",
        winding_y="1/2",
    )

    basis = model.build_basis(
        solver="dfs",
        sort=True,
    )

    assert basis.n_states >= 0

    sectors = model.make_sectors(model.layout)
    for state in basis.states:
        assert all(sector.is_satisfied(state) for sector in sectors)


@pytest.mark.parametrize("builder_name", ["sparse", "bitmask"])
def test_square_qlm_2x2_pbc_kinetic_preserves_gauss_law(
    builder_name: str,
) -> None:
    model = SquareQLMModel(
        lx=2,
        ly=2,
        boundary_condition="periodic",
        charges=0,
        coup_kin=1.0,
        coup_pot=0.0,
    )

    build_result = model.build(
        basis_solver="dfs",
        builder=builder_name,
        backend="scipy",
        sort_basis=True,
        on_missing="raise",
    )

    assert build_result.kinetic is not None
    assert build_result.kinetic.nnz > 0


def _debug_config_against_model(model, config: np.ndarray) -> None:
    print("config =", config.tolist())

    for constraint_index, constraint in enumerate(model.make_constraints(model.layout)):
        result = constraint.check(config)
        print(
            "constraint",
            constraint_index,
            type(constraint).__name__,
            result.satisfied,
            result.message,
        )

    for sector_index, sector in enumerate(model.make_sectors(model.layout)):
        result = sector.check(config)
        print(
            "sector",
            sector_index,
            type(sector).__name__,
            result.satisfied,
            result.message,
        )


@pytest.mark.parametrize("builder_name", ["sparse", "bitmask"])
def test_square_qlm_2x2_pbc_w00_kinetic_preserves_basis(
    builder_name: str,
) -> None:
    model = SquareQLMModel(
        lx=2,
        ly=2,
        boundary_condition="periodic",
        winding_x=0,
        winding_y=0,
        charges=0,
        coup_kin=1.0,
        coup_pot=0.0,
    )

    build_result = model.build(
        basis_solver="dfs",
        builder=builder_name,
        backend="scipy",
        sort_basis=True,
        on_missing="raise",
    )

    assert build_result.kinetic is not None
    assert build_result.kinetic.nnz > 0


@pytest.mark.parametrize("builder_name", ["sparse", "bitmask"])
def test_square_qlm_2x2_pbc_potential_nonzero(
    builder_name: str,
) -> None:
    model = SquareQLMModel(
        lx=2,
        ly=2,
        boundary_condition="periodic",
        charges=0,
        coup_kin=0.0,
        coup_pot=1.0,
    )

    build_result = model.build(
        basis_solver="dfs",
        builder=builder_name,
        backend="scipy",
        sort_basis=True,
        on_missing="raise",
    )

    assert build_result.potential is not None
    assert build_result.potential.nnz > 0


@pytest.mark.parametrize(
    ("kinetic", "potential"),
    [
        (1.0, 0.0),
        (0.0, 1.0),
        (1.0, 1.0),
    ],
)
def test_square_qlm_2x2_sparse_and_bitmask_match(
    kinetic: float,
    potential: float,
) -> None:
    model = SquareQLMModel(
        lx=2,
        ly=2,
        boundary_condition="periodic",
        charges=0,
        coup_kin=kinetic,
        coup_pot=potential,
    )

    sparse_result = model.build(
        basis_solver="dfs",
        builder="sparse",
        backend="scipy",
        sort_basis=True,
        on_missing="raise",
    )

    bitmask_result = model.build(
        basis=sparse_result.basis,
        builder="bitmask",
        backend="scipy",
        sort_basis=False,
        on_missing="raise",
    )

    difference_matrix = sparse_result.hamiltonian - bitmask_result.hamiltonian
    difference_matrix.eliminate_zeros()

    assert difference_matrix.nnz == 0
