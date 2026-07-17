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
