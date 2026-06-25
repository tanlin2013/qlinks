import numpy as np

from qlinks.basis import basis_configs_from_build_result
from qlinks.models import SpinOneXYChainModel
from qlinks.open_system import (
    build_spin_one_xy_lindblad_construction,
    spin_one_xy_scar_tower_states,
)
from tests.helpers.assertions import assert_sparse_allclose


def test_spin_one_xy_scar_tower_states_on_full_basis() -> None:
    model = SpinOneXYChainModel(length=3, boundary_condition="open")
    result = model.build(builder="sparse", basis_solver="dfs", sort_basis=True)

    states, labels = spin_one_xy_scar_tower_states(
        basis_configs=basis_configs_from_build_result(result),
        length=model.length,
    )

    assert labels == ("S_0", "S_1", "S_2", "S_3")
    assert states.shape == (3**3, 4)
    np.testing.assert_allclose(states.conj().T @ states, np.eye(4), atol=1e-12)


def test_spin_one_xy_lindblad_construction_annihilates_full_scar_tower() -> None:
    model = SpinOneXYChainModel(
        length=4,
        boundary_condition="periodic",
        j_xy=1.0,
        h_z=1.0,
        d_z=1.0,
    )
    result = model.build(builder="sparse", basis_solver="dfs", sort_basis=True)

    construction = build_spin_one_xy_lindblad_construction(
        model=model,
        build_result=result,
        left_multiplier="sx",
        gamma=1.0,
    )

    assert construction.n_jumps == model.lattice.num_links
    assert construction.target_labels == ("S_0", "S_1", "S_2", "S_3", "S_4")
    assert construction.max_jump_residual < 1e-12
    assert construction.hamiltonian_closure_residual < 1e-12


def test_spin_one_xy_bond_detectors_sum_to_xy_kinetic_term() -> None:
    model = SpinOneXYChainModel(length=3, boundary_condition="open", j_xy=1.0)
    result = model.build(builder="sparse", basis_solver="dfs", sort_basis=True)

    construction = build_spin_one_xy_lindblad_construction(
        model=model,
        build_result=result,
        left_multiplier="identity",
    )

    detector_sum = sum(construction.detectors, 0 * result.hamiltonian)

    assert result.kinetic is not None
    assert_sparse_allclose(detector_sum, result.kinetic)


def test_spin_one_xy_lindblad_construction_in_total_sz_sector() -> None:
    model = SpinOneXYChainModel(
        length=4,
        boundary_condition="periodic",
        j_xy=1.0,
        h_z=1.0,
        d_z=1.0,
        total_sz=0,
    )
    result = model.build(builder="sparse", basis_solver="dfs", sort_basis=True)

    construction = build_spin_one_xy_lindblad_construction(
        model=model,
        build_result=result,
        left_multiplier="sz",
    )

    assert result.basis.n_states == 19
    assert construction.total_sz == 0
    assert construction.target_labels == ("S_2",)
    assert construction.n_jumps == model.lattice.num_links
    assert construction.max_jump_residual < 1e-12
    assert construction.hamiltonian_closure_residual < 1e-12
