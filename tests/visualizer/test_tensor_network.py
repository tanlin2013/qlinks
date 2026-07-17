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
