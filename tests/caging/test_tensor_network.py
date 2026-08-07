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

pytestmark = pytest.mark.integration


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


def test_native_type1_block_matches_post_projected_state_and_objective() -> None:
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
    parameters = problem.base_problem.perturb_parameters(
        ansatz.parameters,
        scale=1.0e-2,
        seed=3,
    )
    projected = problem.projected_state_vector(parameters)
    native = problem.native_state_vector(parameters)
    native_report = problem.diagnose_native(parameters)
    projected_report = problem.diagnose(parameters)

    assert problem.kinetic_interference_matrix.shape == (
        problem.opposite_basis_indices.size,
        problem.target_basis_indices.size,
    )
    assert np.allclose(projected[problem.target_basis_indices], native)
    assert np.count_nonzero(projected[problem.opposite_basis_indices]) == 0
    assert np.isclose(
        native_report.kinetic_interference_density,
        projected_report.kinetic_interference_density,
    )
    assert np.isclose(
        native_report.potential_variance_density,
        projected_report.potential_variance_density,
    )
    assert np.isclose(problem.loss(parameters), native_report.objective)


def test_native_chiral_ansatz_selects_the_same_fock_subset_without_projection() -> None:
    from qlinks.caging import (
        SquareQDMChiralPEPSAnsatz,
        build_square_qdm_type1_peps_problem,
    )

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
    problem = build_square_qdm_type1_peps_problem(model, basis)
    parameters = np.linspace(0.1, 1.0, basis.n_entries)
    ansatz = SquareQDMChiralPEPSAnsatz.from_type1_problem(problem, parameters)
    native_mask = ansatz.native_sector_mask(problem.base_problem)
    native_state = ansatz.finite_cluster_state_vector(problem.base_problem)
    projected_state = problem.projected_state_vector(parameters)

    assert np.array_equal(native_mask, problem.chiral_mask.astype(bool))
    assert np.allclose(native_state, projected_state)
    assert ansatz.global_charge_sector == problem.parity_rule.offset


def test_joint_type1_problem_aggregates_components_separately() -> None:
    from qlinks.caging import (
        build_square_qdm_type1_joint_cluster_problem,
        build_square_qdm_type1_peps_problem,
        validate_square_qdm_type1_peps_on_clusters,
    )

    host = _host_model()
    singlet = next(
        block
        for block in square_qdm_two_plaquette_singlet_blocks(host, directions=("x",))
        if set(block.anchor_cells) == {(2, 2), (3, 2)}
    )
    ansatz = build_square_qdm_singlet_peps_ansatz(host, singlet, origin=(2, 2))
    problems = {
        label: build_square_qdm_type1_peps_problem(
            SquareQDMModel(
                lx=lx,
                ly=ly,
                boundary_condition="periodic",
                winding_x=0,
                winding_y=0,
                coup_kin=1.0,
                coup_pot=0.0,
            ),
            ansatz.tile_basis,
            reference_parameters=ansatz.parameters,
        )
        for label, (lx, ly) in {
            "6x2": (6, 2),
            "6x4": (6, 4),
        }.items()
    }
    report = validate_square_qdm_type1_peps_on_clusters(
        ansatz.parameters,
        problems,
        aggregation_power=4.0,
    )
    joint = build_square_qdm_type1_joint_cluster_problem(
        problems,
        aggregation_power=4.0,
    )

    expected_kinetic = (
        np.mean([record.report.kinetic_interference_density**4 for record in report.records])
        ** 0.25
    )
    assert np.isclose(report.kinetic_aggregate, expected_kinetic)
    assert np.isclose(report.potential_aggregate, 0.0)
    assert np.isclose(report.objective, expected_kinetic)
    assert np.isclose(joint.loss(ansatz.parameters), report.objective)
    assert report.worst_cluster_label in {"6x2", "6x4"}


def test_joint_type1_autograd_matches_cross_cluster_diagnostic() -> None:
    from qlinks.caging import (
        autograd_available,
        build_square_qdm_type1_joint_cluster_problem,
        build_square_qdm_type1_peps_problem,
    )

    if not autograd_available():
        pytest.skip("autograd is not installed")
    host = _host_model()
    singlet = next(
        block
        for block in square_qdm_two_plaquette_singlet_blocks(host, directions=("x",))
        if set(block.anchor_cells) == {(2, 2), (3, 2)}
    )
    ansatz = build_square_qdm_singlet_peps_ansatz(host, singlet, origin=(2, 2))
    problems = {
        label: build_square_qdm_type1_peps_problem(
            SquareQDMModel(
                lx=lx,
                ly=ly,
                boundary_condition="periodic",
                winding_x=0,
                winding_y=0,
                coup_kin=1.0,
                coup_pot=0.0,
            ),
            ansatz.tile_basis,
            reference_parameters=ansatz.parameters,
        )
        for label, (lx, ly) in {"6x2": (6, 2), "6x4": (6, 4)}.items()
    }
    joint = build_square_qdm_type1_joint_cluster_problem(problems)
    parameters = problems["6x2"].base_problem.perturb_parameters(
        ansatz.parameters,
        scale=1.0e-2,
        seed=4,
    )
    loss, gradient = joint.loss_and_gradient_autograd(parameters)

    assert np.isclose(loss, joint.loss(parameters))
    assert gradient.shape == (ansatz.tile_basis.n_entries,)
    assert np.linalg.norm(gradient) > 0.0


@pytest.mark.skipif(not quimb_available(), reason="quimb is not installed")
def test_short_joint_type1_quimb_optimization_reduces_shared_objective() -> None:
    from qlinks.caging import (
        autograd_available,
        build_square_qdm_type1_joint_cluster_problem,
        build_square_qdm_type1_peps_problem,
    )

    if not autograd_available():
        pytest.skip("autograd is not installed")
    host = _host_model()
    singlet = next(
        block
        for block in square_qdm_two_plaquette_singlet_blocks(host, directions=("x",))
        if set(block.anchor_cells) == {(2, 2), (3, 2)}
    )
    ansatz = build_square_qdm_singlet_peps_ansatz(host, singlet, origin=(2, 2))
    problems = {
        label: build_square_qdm_type1_peps_problem(
            SquareQDMModel(
                lx=lx,
                ly=ly,
                boundary_condition="periodic",
                winding_x=0,
                winding_y=0,
                coup_kin=1.0,
                coup_pot=0.0,
            ),
            ansatz.tile_basis,
            reference_parameters=ansatz.parameters,
        )
        for label, (lx, ly) in {"6x2": (6, 2), "6x4": (6, 4)}.items()
    }
    joint = build_square_qdm_type1_joint_cluster_problem(problems)
    result = joint.optimize_with_quimb(
        ansatz.parameters,
        max_steps=1,
        noise_scale=1.0e-2,
        seed=0,
        progbar=False,
    )

    assert result.improved
    assert result.final_validation.objective < result.initial_validation.objective
    assert set(result.final_validation.by_label) == {"6x2", "6x4"}


def test_type1_interference_decomposition_resolves_interior_and_seam_cancellation() -> None:
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
    decomposition = problem.interference_decomposition(ansatz.parameters)
    by_class = decomposition.by_class

    assert np.isclose(decomposition.total_norm_squared, 3.0)
    assert np.isclose(decomposition.reconstruction_residual, 0.0)
    assert decomposition.dominant_seam_class == "y_seam"
    assert np.isclose(by_class["interior"].residual_norm_squared, 0.0)
    assert np.isclose(by_class["interior"].incoherent_norm_squared, 4.0)
    assert np.isclose(by_class["x_seam"].residual_norm_squared, 1.0)
    assert np.isclose(by_class["y_seam"].residual_norm_squared, 2.0)
    assert np.isclose(by_class["corner"].residual_norm_squared, 0.0)
    assert np.isclose(decomposition.global_cancellation_fraction, 4.0 / 7.0)


def test_type1_seam_sensitivity_builds_targeted_period_two_enlargement() -> None:
    from qlinks.caging import (
        SquareQDMType1AdaptivePEPSFiniteClusterProblem,
        build_square_qdm_type1_adaptive_parameterization,
        build_square_qdm_type1_peps_problem,
    )

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
    probe = problem.base_problem.perturb_parameters(
        ansatz.parameters,
        scale=1.0e-3,
        seed=0,
    )
    sensitivity = problem.interference_parameter_sensitivity(probe, "y_seam")
    parameterization = build_square_qdm_type1_adaptive_parameterization(
        problem,
        ansatz.parameters,
        max_selected_entries=6,
        probe_scale=1.0e-3,
        seed=0,
    )
    adaptive = SquareQDMType1AdaptivePEPSFiniteClusterProblem.from_problem(
        problem,
        parameterization,
    )
    lifted = parameterization.lift_parameters(ansatz.parameters)
    base_report = problem.diagnose_native(ansatz.parameters)
    adaptive_report = adaptive.diagnose(lifted)

    assert sensitivity.loss > 0.0
    assert np.linalg.norm(sensitivity.gradient) > 0.0
    assert parameterization.split_axis == "y"
    assert parameterization.selected_entry_indices.size == 6
    assert parameterization.n_parameters == ansatz.tile_basis.n_entries + 6
    assert adaptive.parameter_indices.shape == problem.target_entry_parameter_indices.shape
    assert np.isclose(
        adaptive_report.kinetic_interference_density,
        base_report.kinetic_interference_density,
    )


@pytest.mark.scientific
def test_type1_adaptive_x_split_can_be_shared_across_longitudinal_clusters() -> None:
    """Protect cross-size sharing of one adaptive PEPS parameterization.

    The 12x2 cluster is retained because this regression checks longitudinal
    extensibility rather than only one finite-cluster construction.
    """
    from qlinks.caging import (
        SquareQDMType1AdaptiveParameterization,
        build_square_qdm_type1_adaptive_joint_cluster_problem,
        build_square_qdm_type1_peps_problem,
    )

    host = _host_model()
    singlet = next(
        block
        for block in square_qdm_two_plaquette_singlet_blocks(host, directions=("x",))
        if set(block.anchor_cells) == {(2, 2), (3, 2)}
    )
    ansatz = build_square_qdm_singlet_peps_ansatz(host, singlet, origin=(2, 2))
    problems = {
        label: build_square_qdm_type1_peps_problem(
            SquareQDMModel(
                lx=lx,
                ly=ly,
                boundary_condition="periodic",
                winding_x=0,
                winding_y=0,
                coup_kin=1.0,
                coup_pot=0.0,
            ),
            ansatz.tile_basis,
            reference_parameters=ansatz.parameters,
        )
        for label, (lx, ly) in {"6x2": (6, 2), "6x4": (6, 4), "12x2": (12, 2)}.items()
    }
    parameterization = SquareQDMType1AdaptiveParameterization(
        tile_basis=ansatz.tile_basis,
        selected_entry_indices=np.asarray((8, 9, 21, 31), dtype=np.int64),
        split_axis="x",
    )
    joint = build_square_qdm_type1_adaptive_joint_cluster_problem(
        problems,
        parameterization,
    )
    lifted = parameterization.lift_parameters(ansatz.parameters)
    validation = joint.diagnose(lifted)

    assert parameterization.n_parameters == 112
    assert set(validation.by_label) == {"6x2", "6x4", "12x2"}
    for label, problem in problems.items():
        assert np.isclose(
            validation.by_label[label].report.kinetic_interference_density,
            problem.diagnose_native(ansatz.parameters).kinetic_interference_density,
        )


@pytest.mark.scientific
def test_type1_adaptive_exact_gradient_and_short_optimization_reduce_joint_loss() -> None:
    """Protect the exact joint gradient and its optimization descent claim.

    This combines two finite clusters because the shared-parameter objective is
    the scientific contract; a one-cluster unit case would not exercise it.
    """
    from qlinks.caging import (
        SquareQDMType1AdaptiveParameterization,
        build_square_qdm_type1_adaptive_joint_cluster_problem,
        build_square_qdm_type1_peps_problem,
    )

    host = _host_model()
    singlet = next(
        block
        for block in square_qdm_two_plaquette_singlet_blocks(host, directions=("x",))
        if set(block.anchor_cells) == {(2, 2), (3, 2)}
    )
    ansatz = build_square_qdm_singlet_peps_ansatz(host, singlet, origin=(2, 2))
    problems = {
        label: build_square_qdm_type1_peps_problem(
            SquareQDMModel(
                lx=lx,
                ly=ly,
                boundary_condition="periodic",
                winding_x=0,
                winding_y=0,
                coup_kin=1.0,
                coup_pot=0.0,
            ),
            ansatz.tile_basis,
            reference_parameters=ansatz.parameters,
        )
        for label, (lx, ly) in {"6x2": (6, 2), "6x4": (6, 4)}.items()
    }
    parameterization = SquareQDMType1AdaptiveParameterization(
        tile_basis=ansatz.tile_basis,
        selected_entry_indices=np.asarray((8, 9, 21, 31), dtype=np.int64),
        split_axis="x",
    )
    joint = build_square_qdm_type1_adaptive_joint_cluster_problem(
        problems,
        parameterization,
    )
    initial = parameterization.lift_parameters(ansatz.parameters)
    probe = initial + 1.0e-3 * np.random.default_rng(2).normal(size=initial.size)
    loss, gradient = joint.loss_and_gradient_exact(probe)
    entry_index = ansatz.tile_basis.n_entries
    epsilon = 1.0e-6
    plus = probe.copy()
    minus = probe.copy()
    plus[entry_index] += epsilon
    minus[entry_index] -= epsilon
    finite_difference = (joint.loss(plus) - joint.loss(minus)) / (2.0 * epsilon)
    result = joint.optimize_with_scipy(
        initial,
        max_steps=2,
        noise_scale=1.0e-3,
        seed=0,
    )

    assert np.isclose(loss, joint.loss(probe))
    assert np.isclose(gradient[entry_index], finite_difference, rtol=1.0e-4, atol=1.0e-7)
    assert result.improved
    assert result.final_validation.objective < result.initial_validation.objective
