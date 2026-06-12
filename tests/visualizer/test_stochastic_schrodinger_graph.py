import os

import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy.sparse as sp

from qlinks.visualizer.hamiltonian_graph import HamiltonianGraphStyle
from qlinks.visualizer.stochastic_schrodinger_graph import (
    StochasticSchrodingerGraphVisualizer,
    as_stochastic_trajectory,
)


def test_as_stochastic_trajectory_validates_shape() -> None:
    times = np.asarray([0.0, 0.1, 0.2])
    states = np.ones((3, 4), dtype=np.complex128)

    trajectory = as_stochastic_trajectory(times=times, states=states)

    assert trajectory.n_times == 3
    assert trajectory.hilbert_dim == 4


def test_as_stochastic_trajectory_rejects_mismatched_time_count() -> None:
    with pytest.raises(ValueError, match="same n_times"):
        as_stochastic_trajectory(
            times=np.asarray([0.0, 0.1]),
            states=np.ones((3, 4), dtype=np.complex128),
        )


def test_stochastic_visualizer_frame_uses_selected_state_vector() -> None:
    hamiltonian = sp.csr_array(
        np.asarray(
            [
                [0.0, 1.0],
                [1.0, 0.0],
            ],
            dtype=np.complex128,
        )
    )

    times = np.asarray([0.0, 1.0])
    states = np.asarray(
        [
            [1.0, 0.0],
            [0.6, 0.8j],
        ],
        dtype=np.complex128,
    )

    visualizer = StochasticSchrodingerGraphVisualizer.from_trajectory(
        times=times,
        states=states,
        hamiltonian=hamiltonian,
    )

    values = visualizer.graph_visualizer.node_values(
        color_by="state_weight",
        state_vector=visualizer.trajectory.state_at(1),
    )

    np.testing.assert_allclose(values, np.asarray([0.36, 0.64]))


def test_stochastic_visualizer_jump_operators_are_directed_layers() -> None:
    hamiltonian = sp.csr_array(np.zeros((2, 2), dtype=np.complex128))

    jump = sp.csr_array(
        np.asarray(
            [
                [0.0, 1.0],
                [0.0, 0.0],
            ],
            dtype=np.complex128,
        )
    )

    visualizer = StochasticSchrodingerGraphVisualizer.from_trajectory(
        times=np.asarray([0.0]),
        states=np.asarray([[1.0, 0.0]], dtype=np.complex128),
        hamiltonian=hamiltonian,
        jump_operators=[jump],
    )

    assert len(visualizer.jump_visualizers) == 1

    jump_graph = visualizer.jump_visualizers[0].to_networkx()

    assert jump_graph.is_directed()
    assert set(jump_graph.edges) == {(0, 1)}


def test_stochastic_visualizer_plot_frame_smoke() -> None:
    hamiltonian = sp.csr_array(
        np.asarray(
            [
                [0.0, 1.0],
                [1.0, 0.0],
            ],
            dtype=np.complex128,
        )
    )

    visualizer = StochasticSchrodingerGraphVisualizer.from_trajectory(
        times=np.asarray([0.0, 1.0]),
        states=np.asarray(
            [
                [1.0, 0.0],
                [0.5, np.sqrt(3.0) / 2.0],
            ],
            dtype=np.complex128,
        ),
        hamiltonian=hamiltonian,
        style=HamiltonianGraphStyle(
            colorbar=False,
            edge_colorbar=False,
        ),
    )

    ax = visualizer.plot_frame(
        1,
        backend="networkx",
        layout="circle",
        show=False,
    )

    assert ax is not None


def test_stochastic_visualizer_animation_smoke() -> None:
    hamiltonian = sp.csr_array(
        np.asarray(
            [
                [0.0, 1.0],
                [1.0, 0.0],
            ],
            dtype=np.complex128,
        )
    )

    visualizer = StochasticSchrodingerGraphVisualizer.from_trajectory(
        times=np.asarray([0.0, 1.0]),
        states=np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.complex128,
        ),
        hamiltonian=hamiltonian,
        style=HamiltonianGraphStyle(
            colorbar=False,
            edge_colorbar=False,
        ),
    )

    animation = visualizer.animate(
        layout="circle",
        interval=10,
        repeat=False,
    )

    assert animation is not None


def test_stochastic_visualizer_animation_redraw_each_frame_smoke() -> None:
    hamiltonian = sp.csr_array(
        np.asarray(
            [
                [0.0, 1.0],
                [1.0, 0.0],
            ],
            dtype=np.complex128,
        )
    )

    visualizer = StochasticSchrodingerGraphVisualizer.from_trajectory(
        times=np.asarray([0.0, 1.0]),
        states=np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.complex128,
        ),
        hamiltonian=hamiltonian,
        style=HamiltonianGraphStyle(
            colorbar=False,
            edge_colorbar=False,
        ),
    )

    animation = visualizer.animate(
        layout="circle",
        interval=10,
        repeat=False,
        redraw_each_frame=True,
    )

    assert animation is not None


@pytest.mark.manual
@pytest.mark.skipif(
    os.environ.get("QLINKS_SHOW_PLOTS") != "1",
    reason="Set QLINKS_SHOW_PLOTS=1 to show manual plots.",
)
def test_manual_stochastic_schrodinger_animation() -> None:
    """Manual animation smoke test for a stochastic-Schrödinger trajectory."""
    import matplotlib

    matplotlib.use("TkAgg")  # Force interactive GUI window

    n_frames = 80
    times = np.linspace(0.0, 8.0, n_frames)

    # A small six-state Hilbert-space graph with complex hopping.
    hamiltonian = sp.csr_array(
        np.asarray(
            [
                [0.0, 1.0, 0.4j, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.8, 0.5j, 0.0, 0.0],
                [-0.4j, 0.8, 0.0, 1.0, 0.0, 0.0],
                [0.0, -0.5j, 1.0, 0.0, 0.7, 0.3j],
                [0.0, 0.0, 0.0, 0.7, 0.0, 1.0],
                [0.0, 0.0, 0.0, -0.3j, 1.0, 0.0],
            ],
            dtype=np.complex128,
        )
    )

    # Toy trajectory: two moving wave packets plus time-dependent phases.
    states = np.zeros((n_frames, 6), dtype=np.complex128)

    centers = 2.5 + 2.0 * np.sin(0.75 * times)
    basis_positions = np.arange(6, dtype=np.float64)

    for frame, (time, center) in enumerate(zip(times, centers, strict=True)):
        envelope = np.exp(-0.5 * ((basis_positions - center) / 0.9) ** 2)
        phase = np.exp(1.0j * (0.9 * basis_positions * time))
        state = envelope * phase

        # Add a weak second component so the animation is not just translation.
        state += (
            0.25
            * np.exp(-0.5 * ((basis_positions - (5.0 - center)) / 0.7) ** 2)
            * np.exp(-1.4j * basis_positions * time)
        )

        state = state / np.linalg.norm(state)
        states[frame] = state

    jump_operator = sp.csr_array(
        np.asarray(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.4],
                [0.0, 0.0, 0.0, 0.0, 0.3, 0.0],
                [0.0, 0.0, 0.0, 0.2, 0.0, 0.0],
                [0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
                [0.0, 0.05, 0.0, 0.0, 0.0, 0.0],
                [0.02, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.complex128,
        )
    )

    visualizer = StochasticSchrodingerGraphVisualizer.from_trajectory(
        times=times,
        states=states,
        hamiltonian=hamiltonian,
        jump_operators=[jump_operator],
        basis_labels=[f"|{index}>" for index in range(6)],
        style=HamiltonianGraphStyle(
            figure_size=(6.2, 5.6),
            label_vertices=True,
            vertex_size=120.0,
            edge_width=2.2,
            edge_alpha=0.65,
            colorbar=True,
            edge_colorbar=False,
        ),
    )

    animation = visualizer.animate(
        layout="circle",
        color_by="probability",
        edge_color_by="weight_complex",
        interval=80,
        repeat=True,
    )

    plt.show()

    assert animation is not None
