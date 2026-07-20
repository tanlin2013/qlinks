from __future__ import annotations

import numpy as np

from qlinks.caging import (
    SquareQDMPEPSOptimizationResult,
    SquareQDMPEPSResidualReport,
    build_square_qdm_rectangular_tile_tensor_basis,
)
from qlinks.models import SquareQDMModel
from qlinks.visualizer import SquareQDMTensorNetworkVisualizer


def _basis():
    model = SquareQDMModel(
        lx=8,
        ly=8,
        boundary_condition="periodic",
        coup_kin=1.0,
        coup_pot=0.0,
    )
    return build_square_qdm_rectangular_tile_tensor_basis(
        model,
        tile_shape=(3, 2),
        origin=(2, 2),
    )


def test_tensor_network_visualizer_draws_graph_entry_and_parameters() -> None:
    basis = _basis()
    visualizer = SquareQDMTensorNetworkVisualizer(basis)

    network_ax = visualizer.plot_network(n_tiles_x=3, n_tiles_y=2)
    entry_ax = visualizer.plot_entry(0)
    parameter_ax = visualizer.plot_parameter_magnitudes(np.ones(basis.n_entries))

    assert network_ax.get_title()
    assert entry_ax.get_title()
    assert parameter_ax.get_ylabel() == "Amplitude magnitude"


def test_tensor_network_visualizer_draws_optimization_history() -> None:
    basis = _basis()
    report_initial = SquareQDMPEPSResidualReport(1.0, 0.0, 1.0, 1.0, 2, 4)
    report_final = SquareQDMPEPSResidualReport(1.0, 0.0, 0.5, 0.25, 4, 4)
    result = SquareQDMPEPSOptimizationResult(
        initial_parameters=np.ones(basis.n_entries),
        optimized_parameters=np.ones(basis.n_entries),
        loss_history=(1.0, 0.5, 0.25),
        initial_report=report_initial,
        final_report=report_final,
        requested_steps=2,
        optimizer="L-BFGS-B",
        autodiff_backend="autograd",
    )

    axis = SquareQDMTensorNetworkVisualizer(basis).plot_optimization_history(result)

    assert axis.get_yscale() == "log"
    assert axis.get_ylabel()


def test_tensor_network_visualizer_draws_type1_diagnostics() -> None:
    from qlinks.caging import (
        SquareQDMType1PEPSResidualReport,
        build_square_qdm_type1_peps_problem,
    )

    basis = _basis()
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
    report = SquareQDMType1PEPSResidualReport(
        norm_before_projection=1.0,
        norm_after_projection=0.8,
        retained_chiral_weight=0.64,
        discarded_chiral_weight=0.36,
        target_chiral_label=0,
        kinetic_interference_norm=2.4,
        kinetic_interference_density=0.1,
        potential_mean=0.0,
        potential_variance=0.0,
        potential_variance_density=0.0,
        total_variance=2.4,
        objective=0.1,
        max_interference_residual=0.2,
        n_nonzero_interference_targets=12,
        nonzero_projected_amplitudes=8,
        hilbert_dimension=1456,
    )
    visualizer = SquareQDMTensorNetworkVisualizer(basis)
    component_axis = visualizer.plot_type1_components(report)
    charge_axis = visualizer.plot_chiral_physical_charges(problem.parity_rule, model)

    assert component_axis.get_ylabel() == "Normalized diagnostic"
    assert charge_axis.get_title() == "Tile-local chiral charges"


def test_tensor_network_visualizer_draws_type1_cluster_validation() -> None:
    from qlinks.caging import (
        SquareQDMType1ClusterValidationRecord,
        SquareQDMType1ClusterValidationReport,
        SquareQDMType1PEPSResidualReport,
    )

    def report(kinetic: float, potential: float):
        return SquareQDMType1PEPSResidualReport(
            norm_before_projection=1.0,
            norm_after_projection=1.0,
            retained_chiral_weight=1.0,
            discarded_chiral_weight=0.0,
            target_chiral_label=0,
            kinetic_interference_norm=kinetic * 24.0,
            kinetic_interference_density=kinetic,
            potential_mean=0.0,
            potential_variance=potential * 24.0,
            potential_variance_density=potential,
            total_variance=(kinetic + potential) * 24.0,
            objective=kinetic + potential,
            max_interference_residual=0.1,
            n_nonzero_interference_targets=2,
            nonzero_projected_amplitudes=4,
            hilbert_dimension=8,
        )

    validation = SquareQDMType1ClusterValidationReport(
        records=(
            SquareQDMType1ClusterValidationRecord("6x2", 12, report(0.2, 0.01)),
            SquareQDMType1ClusterValidationRecord("6x4", 24, report(0.15, 0.02)),
        ),
    )
    axis = SquareQDMTensorNetworkVisualizer(_basis()).plot_type1_cluster_validation(validation)

    assert axis.get_ylabel() == "Density"
    assert len(axis.patches) == 4


def test_tensor_network_visualizer_draws_seam_decomposition_and_adaptive_entries() -> None:
    from qlinks.caging import (
        SquareQDMType1AdaptiveParameterization,
        SquareQDMType1InterferenceClassRecord,
        SquareQDMType1InterferenceDecomposition,
        SquareQDMType1ParameterSensitivity,
    )

    basis = _basis()
    decomposition = SquareQDMType1InterferenceDecomposition(
        total_norm_squared=3.0,
        reconstruction_residual=0.0,
        plaquette_records=(),
        class_records=(
            SquareQDMType1InterferenceClassRecord("interior", 8, 0.0, 4.0, 0.0, 0.0, 0),
            SquareQDMType1InterferenceClassRecord("x_seam", 4, 1.0, 1.0, 1.0, 0.25, 16),
            SquareQDMType1InterferenceClassRecord("y_seam", 8, 2.0, 2.0, 2.0, 0.25, 32),
            SquareQDMType1InterferenceClassRecord("corner", 4, 0.0, 0.0, 0.0, 0.0, 0),
        ),
    )
    sensitivity = SquareQDMType1ParameterSensitivity(
        plaquette_class="y_seam",
        gradient=np.linspace(-1.0, 1.0, basis.n_entries),
        scores=np.abs(np.linspace(-1.0, 1.0, basis.n_entries)),
        entry_coordinates=basis.entry_coordinates,
        loss=0.1,
    )
    parameterization = SquareQDMType1AdaptiveParameterization(
        tile_basis=basis,
        selected_entry_indices=np.asarray((8, 9, 21, 31), dtype=np.int64),
        split_axis="y",
    )
    visualizer = SquareQDMTensorNetworkVisualizer(basis)
    decomposition_axis = visualizer.plot_type1_interference_decomposition(decomposition)
    sensitivity_axis = visualizer.plot_type1_parameter_sensitivity(sensitivity, max_entries=8)
    adaptive_axis = visualizer.plot_type1_adaptive_parameterization(parameterization)

    assert decomposition_axis.get_ylabel() == "Residual norm squared"
    assert sensitivity_axis.get_ylabel() == "Absolute loss gradient"
    assert adaptive_axis.get_ylabel() == "Independent tile classes"
