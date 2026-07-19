from __future__ import annotations

import numpy as np
import pytest

from qlinks.caging import (
    SquareQDMPEPSAnsatz,
    build_square_qdm_peps_finite_cluster_problem,
    build_square_qdm_rectangular_tile_tensor_basis,
    build_square_qdm_singlet_peps_ansatz,
    quimb_available,
    square_qdm_two_plaquette_singlet_blocks,
)
from qlinks.models import SquareQDMModel


def _host_model() -> SquareQDMModel:
    return SquareQDMModel(
        lx=8,
        ly=8,
        boundary_condition="periodic",
        coup_kin=1.0,
        coup_pot=0.0,
    )


def _tile_basis():
    return build_square_qdm_rectangular_tile_tensor_basis(
        _host_model(),
        tile_shape=(3, 2),
        origin=(2, 2),
    )


def test_rectangular_vertex_tensor_basis_enforces_local_dimer_constraint() -> None:
    basis = _tile_basis()

    assert basis.tile_shape == (3, 2)
    assert basis.owned_link_ids.size == 12
    assert basis.physical_dimension == 71
    assert basis.n_entries == 108
    assert basis.tensor_shape == (8, 4, 8, 4, 71)
    assert np.count_nonzero(basis.structural_tensor_data()) == 108
    assert np.isclose(basis.compression_ratio, 71 / 4096)


def test_singlet_core_initializes_a_genuine_two_dimensional_unit_tensor() -> None:
    model = _host_model()
    singlet = next(
        block
        for block in square_qdm_two_plaquette_singlet_blocks(
            model,
            directions=("x",),
        )
        if set(block.anchor_cells) == {(2, 2), (3, 2)}
    )

    ansatz = build_square_qdm_singlet_peps_ansatz(
        model,
        singlet,
        origin=(2, 2),
    )

    assert ansatz.n_parameters == 108
    assert ansatz.metadata["n_core_compatible_entries"] == 2
    assert np.count_nonzero(ansatz.parameters) == 2
    assert np.isclose(np.linalg.norm(ansatz.parameters), 1.0)


@pytest.mark.skipif(not quimb_available(), reason="quimb is not installed")
def test_quimb_network_counts_all_dimer_coverings_on_a_short_torus() -> None:
    basis = _tile_basis()
    ansatz = SquareQDMPEPSAnsatz(
        tile_basis=basis,
        parameters=np.ones(basis.n_entries, dtype=np.complex128),
    )

    network = ansatz.to_quimb_tensor_network(n_tiles_x=2, n_tiles_y=2)
    norm_squared = network.norm(squared=True, optimize="greedy")
    model = SquareQDMModel(
        lx=6,
        ly=4,
        boundary_condition="periodic",
        coup_kin=1.0,
        coup_pot=0.0,
    )

    assert network.num_tensors == 4
    assert len(network.outer_inds()) == 4
    assert np.isclose(norm_squared, model.build_basis().n_states)
    with pytest.raises(ValueError, match="at least three tiles"):
        ansatz.to_quimb_peps(n_tiles_x=2, n_tiles_y=2)
    structured = ansatz.to_quimb_peps(n_tiles_x=3, n_tiles_y=3)
    assert structured.Lx == 3
    assert structured.Ly == 3


def test_exact_small_torus_problem_reproduces_singlet_product_leakage() -> None:
    host = _host_model()
    singlet = next(
        block
        for block in square_qdm_two_plaquette_singlet_blocks(
            host,
            directions=("x",),
        )
        if set(block.anchor_cells) == {(2, 2), (3, 2)}
    )
    ansatz = build_square_qdm_singlet_peps_ansatz(
        host,
        singlet,
        origin=(2, 2),
    )
    model = SquareQDMModel(
        lx=6,
        ly=4,
        boundary_condition="periodic",
        winding_x=0,
        winding_y=0,
        coup_kin=1.0,
        coup_pot=0.0,
    )

    problem = build_square_qdm_peps_finite_cluster_problem(
        model,
        ansatz.tile_basis,
    )
    report = problem.diagnose(ansatz.parameters)

    assert problem.hilbert_dimension == 1456
    assert problem.n_tiles_x == 2
    assert problem.n_tiles_y == 2
    assert problem.n_tiles == 4
    assert report.nonzero_basis_amplitudes == 16
    assert np.isclose(report.energy, 0.0)
    assert np.isclose(report.energy_variance, 3.0)
    assert np.isclose(report.residual, np.sqrt(3.0))
    assert np.isclose(problem.loss(ansatz.parameters), 3.0)


@pytest.mark.skipif(not quimb_available(), reason="quimb is not installed")
def test_quimb_optimizer_can_be_constructed_for_exact_cluster_loss() -> None:
    basis = _tile_basis()
    model = SquareQDMModel(
        lx=6,
        ly=4,
        boundary_condition="periodic",
        winding_x=0,
        winding_y=0,
        coup_kin=1.0,
        coup_pot=0.0,
    )
    problem = build_square_qdm_peps_finite_cluster_problem(model, basis)

    optimizer = problem.make_quimb_optimizer(
        np.ones(basis.n_entries, dtype=np.complex128),
        progbar=False,
    )

    assert type(optimizer).__name__ == "TNOptimizer"


@pytest.mark.skipif(not quimb_available(), reason="quimb is not installed")
def test_compact_autograd_optimizer_uses_only_allowed_entries() -> None:
    from qlinks.caging import autograd_available

    if not autograd_available():
        pytest.skip("autograd is not installed")
    host = _host_model()
    singlet = next(
        block
        for block in square_qdm_two_plaquette_singlet_blocks(host, directions=("x",))
        if set(block.anchor_cells) == {(2, 2), (3, 2)}
    )
    ansatz = build_square_qdm_singlet_peps_ansatz(host, singlet, origin=(2, 2))
    model = SquareQDMModel(
        lx=6,
        ly=4,
        boundary_condition="periodic",
        winding_x=0,
        winding_y=0,
        coup_kin=1.0,
        coup_pot=0.0,
    )
    problem = build_square_qdm_peps_finite_cluster_problem(model, ansatz.tile_basis)
    initial = problem.perturb_parameters(ansatz.parameters, scale=1.0e-2, seed=0)
    loss, gradient = problem.loss_and_gradient_autograd(initial)
    optimizer = problem.make_quimb_optimizer(initial, progbar=False)

    assert optimizer.d == ansatz.tile_basis.n_entries == 108
    assert np.isclose(loss, problem.loss(initial))
    assert np.linalg.norm(gradient) > 0.0


@pytest.mark.skipif(not quimb_available(), reason="quimb is not installed")
def test_short_autograd_optimization_reduces_exact_variance() -> None:
    from qlinks.caging import autograd_available

    if not autograd_available():
        pytest.skip("autograd is not installed")
    host = _host_model()
    singlet = next(
        block
        for block in square_qdm_two_plaquette_singlet_blocks(host, directions=("x",))
        if set(block.anchor_cells) == {(2, 2), (3, 2)}
    )
    ansatz = build_square_qdm_singlet_peps_ansatz(host, singlet, origin=(2, 2))
    model = SquareQDMModel(
        lx=6,
        ly=4,
        boundary_condition="periodic",
        winding_x=0,
        winding_y=0,
        coup_kin=1.0,
        coup_pot=0.0,
    )
    problem = build_square_qdm_peps_finite_cluster_problem(model, ansatz.tile_basis)
    result = problem.optimize_with_quimb(
        ansatz.parameters,
        max_steps=2,
        noise_scale=1.0e-2,
        seed=0,
        progbar=False,
    )

    assert result.metadata["optimizer_dimension"] == 108
    assert result.final_loss < result.initial_loss
    assert result.to_ansatz(ansatz.tile_basis).n_parameters == 108


def test_tile_periodic_chiral_rule_anticommutes_with_square_qdm_kinetic_term() -> None:
    from qlinks.caging import (
        build_square_qdm_type1_peps_problem,
        infer_square_qdm_tile_chiral_parity_rule,
    )
    from qlinks.caging.search import bipartition_labels

    basis = _tile_basis()
    model = SquareQDMModel(
        lx=6,
        ly=4,
        boundary_condition="periodic",
        winding_x=0,
        winding_y=0,
        coup_kin=1.0,
        coup_pot=0.0,
    )
    build_result = model.build()
    states = np.asarray(build_result.basis.states, dtype=np.int8)
    graph_labels = bipartition_labels(build_result.kinetic)
    rule = infer_square_qdm_tile_chiral_parity_rule(
        model,
        states,
        build_result.kinetic,
        basis,
        reference_labels=graph_labels,
    )

    assert rule.validate_kinetic_matrix(states, build_result.kinetic)
    assert rule.metadata["tile_periodic"] is True
    assert rule.n_edge_equations == 6
    charges = rule.tile_physical_charges(model, basis)
    assert charges.shape == (basis.physical_dimension,)
    assert set(np.unique(charges)) == {0, 1}

    problem = build_square_qdm_type1_peps_problem(model, basis)
    assert problem.parity_rule is not None
    assert problem.parity_rule.metadata["tile_periodic"] is True


def test_type1_peps_objective_separates_kinetic_and_potential_conditions() -> None:
    from qlinks.caging import build_square_qdm_type1_peps_problem

    host = _host_model()
    singlet = next(
        block
        for block in square_qdm_two_plaquette_singlet_blocks(host, directions=("x",))
        if set(block.anchor_cells) == {(2, 2), (3, 2)}
    )
    ansatz = build_square_qdm_singlet_peps_ansatz(host, singlet, origin=(2, 2))
    model = SquareQDMModel(
        lx=6,
        ly=4,
        boundary_condition="periodic",
        winding_x=0,
        winding_y=0,
        coup_kin=1.0,
        coup_pot=0.0,
    )
    problem = build_square_qdm_type1_peps_problem(
        model,
        ansatz.tile_basis,
        reference_parameters=ansatz.parameters,
    )
    report = problem.diagnose(ansatz.parameters)

    assert np.isclose(report.retained_chiral_weight, 1.0)
    assert np.isclose(report.discarded_chiral_weight, 0.0)
    assert np.isclose(report.kinetic_interference_norm, 3.0)
    assert np.isclose(report.kinetic_interference_density, 3.0 / 24.0)
    assert np.isclose(report.potential_variance, 0.0)
    assert np.isclose(report.total_variance, report.kinetic_interference_norm)
    assert np.isclose(report.objective, report.kinetic_interference_density)
    assert report.n_nonzero_interference_targets == 48


def test_type1_problem_accepts_an_existing_type1_cage_record() -> None:
    from qlinks.caging import (
        CageSearchConfig,
        CageSearcher,
        build_square_qdm_type1_peps_problem,
    )

    host = SquareQDMModel(
        lx=6,
        ly=6,
        boundary_condition="periodic",
        coup_kin=1.0,
        coup_pot=1.0,
    )
    tile_basis = build_square_qdm_rectangular_tile_tensor_basis(
        host,
        tile_shape=(2, 2),
        origin=(2, 2),
    )
    model = SquareQDMModel(
        lx=4,
        ly=4,
        boundary_condition="periodic",
        winding_x=0,
        winding_y=0,
        coup_kin=1.0,
        coup_pot=1.0,
    )
    build_result = model.build()
    cages = CageSearcher.from_model_build_result(
        build_result,
        config=CageSearchConfig(search_type="type1"),
    ).run()
    record = cages[(0, 4), 0]
    problem = build_square_qdm_type1_peps_problem(
        model,
        tile_basis,
        cage_record=record,
    )

    assert problem.target_chiral_label in {0, 1}
    assert np.isclose(problem.target_potential_value, 4.0)
    assert np.unique(problem.chiral_labels[record.support]).tolist() == [
        problem.target_chiral_label
    ]
    assert problem.parity_rule is not None
    assert problem.parity_rule.metadata["tile_periodic"] is True


@pytest.mark.skipif(not quimb_available(), reason="quimb is not installed")
def test_type1_autograd_loss_matches_separated_diagnostic() -> None:
    from qlinks.caging import autograd_available, build_square_qdm_type1_peps_problem

    if not autograd_available():
        pytest.skip("autograd is not installed")
    host = _host_model()
    singlet = next(
        block
        for block in square_qdm_two_plaquette_singlet_blocks(host, directions=("x",))
        if set(block.anchor_cells) == {(2, 2), (3, 2)}
    )
    ansatz = build_square_qdm_singlet_peps_ansatz(host, singlet, origin=(2, 2))
    model = SquareQDMModel(
        lx=6,
        ly=4,
        boundary_condition="periodic",
        winding_x=0,
        winding_y=0,
        coup_kin=1.0,
        coup_pot=0.0,
    )
    problem = build_square_qdm_type1_peps_problem(
        model,
        ansatz.tile_basis,
        reference_parameters=ansatz.parameters,
    )
    initial = problem.base_problem.perturb_parameters(
        ansatz.parameters,
        scale=1.0e-2,
        seed=0,
    )
    loss, gradient = problem.loss_and_gradient_autograd(initial)

    assert np.isclose(loss, problem.loss(initial))
    assert gradient.shape == (ansatz.tile_basis.n_entries,)
    assert np.linalg.norm(gradient) > 0.0


@pytest.mark.skipif(not quimb_available(), reason="quimb is not installed")
def test_native_chiral_peps_projects_the_global_fock_subset() -> None:
    from qlinks.caging import (
        SquareQDMChiralPEPSAnsatz,
        build_square_qdm_type1_peps_problem,
    )

    basis = _tile_basis()
    sector_model = SquareQDMModel(
        lx=6,
        ly=4,
        boundary_condition="periodic",
        winding_x=0,
        winding_y=0,
        coup_kin=1.0,
        coup_pot=0.0,
    )
    problem = build_square_qdm_type1_peps_problem(sector_model, basis)
    ansatz = SquareQDMChiralPEPSAnsatz.from_type1_problem(
        problem,
        np.ones(basis.n_entries),
    )
    network = ansatz.to_quimb_tensor_network(n_tiles_x=2, n_tiles_y=2)
    norm_squared = float(network.norm(squared=True, optimize="greedy").real)

    unrestricted_model = SquareQDMModel(
        lx=6,
        ly=4,
        boundary_condition="periodic",
        coup_kin=1.0,
        coup_pot=0.0,
    )
    unrestricted_states = np.asarray(unrestricted_model.build_basis().states, dtype=np.int8)
    even_count = int(np.count_nonzero(problem.parity_rule.labels(unrestricted_states) == 0))
    degeneracy = ansatz.charge_degeneracy(n_tiles_x=2, n_tiles_y=2)

    assert ansatz.tensor_shape == (16, 8, 16, 8, 71)
    assert ansatz.n_nonzero_tensor_entries == 8 * basis.n_entries
    assert np.isclose(norm_squared, even_count * degeneracy**2)


@pytest.mark.skipif(not quimb_available(), reason="quimb is not installed")
def test_short_type1_quimb_optimization_reduces_the_separated_objective() -> None:
    from qlinks.caging import autograd_available, build_square_qdm_type1_peps_problem

    if not autograd_available():
        pytest.skip("autograd is not installed")
    host = _host_model()
    singlet = next(
        block
        for block in square_qdm_two_plaquette_singlet_blocks(host, directions=("x",))
        if set(block.anchor_cells) == {(2, 2), (3, 2)}
    )
    ansatz = build_square_qdm_singlet_peps_ansatz(host, singlet, origin=(2, 2))
    model = SquareQDMModel(
        lx=6,
        ly=2,
        boundary_condition="periodic",
        winding_x=0,
        winding_y=0,
        coup_kin=1.0,
        coup_pot=0.0,
    )
    problem = build_square_qdm_type1_peps_problem(
        model,
        ansatz.tile_basis,
        reference_parameters=ansatz.parameters,
    )
    result = problem.optimize_with_quimb(
        ansatz.parameters,
        max_steps=1,
        noise_scale=1.0e-2,
        seed=0,
        progbar=False,
    )

    assert result.improved
    assert result.final_report.objective < result.initial_report.objective
