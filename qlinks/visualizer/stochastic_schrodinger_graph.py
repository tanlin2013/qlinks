from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import numpy.typing as npt

from qlinks.visualizer.hamiltonian_graph import (
    EdgeColorRule,
    GraphBackend,
    HamiltonianGraphData,
    HamiltonianGraphStyle,
    HamiltonianGraphVisualizer,
    LayoutName,
    NodeColorRule,
)

TrajectoryNodeColorRule = Literal[
    "probability",
    "amplitude_abs",
    "amplitude_real",
    "amplitude_imag",
    "phase",
]


@dataclass(frozen=True)
class StochasticSchrodingerTrajectory:
    """One stochastic Schrödinger / quantum trajectory."""

    times: npt.NDArray[np.float64]
    states: npt.NDArray[np.complex128]

    @property
    def n_times(self) -> int:
        return int(self.states.shape[0])

    @property
    def hilbert_dim(self) -> int:
        return int(self.states.shape[1])

    def state_at(self, frame: int) -> npt.NDArray[np.complex128]:
        return np.asarray(self.states[int(frame)], dtype=np.complex128)


def as_stochastic_trajectory(
    *,
    times: npt.ArrayLike,
    states: npt.ArrayLike,
) -> StochasticSchrodingerTrajectory:
    """Validate and normalize trajectory arrays."""
    time_array = np.asarray(times, dtype=np.float64)
    state_array = np.asarray(states, dtype=np.complex128)

    if time_array.ndim != 1:
        raise ValueError("times must be a 1D array.")

    if state_array.ndim != 2:
        raise ValueError("states must be a 2D array with shape (n_times, dim).")

    if len(time_array) != state_array.shape[0]:
        raise ValueError("times and states must have the same n_times.")

    return StochasticSchrodingerTrajectory(
        times=time_array,
        states=state_array,
    )


@dataclass(frozen=True)
class StochasticSchrodingerGraphVisualizer:
    """Graph visualizer for stochastic Schrödinger trajectories.

    Nodes are Hilbert-space basis states. Node colors are taken from a
    time-dependent stochastic state vector psi(t).
    """

    graph_visualizer: HamiltonianGraphVisualizer
    trajectory: StochasticSchrodingerTrajectory
    jump_visualizers: tuple[HamiltonianGraphVisualizer, ...] = ()

    @classmethod
    def from_trajectory(
        cls,
        *,
        times: npt.ArrayLike,
        states: npt.ArrayLike,
        hamiltonian=None,
        jump_operators: Sequence | None = None,
        basis_labels: Sequence[str] | None = None,
        weight_tolerance: float = 0.0,
        style: HamiltonianGraphStyle | None = None,
    ) -> StochasticSchrodingerGraphVisualizer:
        """Construct a stochastic Schrödinger trajectory visualizer."""
        trajectory = as_stochastic_trajectory(times=times, states=states)

        if hamiltonian is None:
            # No transition graph: isolated nodes.
            import scipy.sparse as sp

            hamiltonian = sp.csr_array(
                (trajectory.hilbert_dim, trajectory.hilbert_dim),
                dtype=np.complex128,
            )

        graph_visualizer = HamiltonianGraphVisualizer.from_sparse_matrix(
            hamiltonian,
            weight_tolerance=weight_tolerance,
            style=style,
        )

        if graph_visualizer.graph_data.n_vertices != trajectory.hilbert_dim:
            raise ValueError("hamiltonian dimension must match trajectory Hilbert dimension.")

        old_data = graph_visualizer.graph_data

        graph_visualizer = HamiltonianGraphVisualizer(
            graph_data=HamiltonianGraphData(
                adjacency=old_data.adjacency,
                self_loop_values=old_data.self_loop_values,
                original_indices=old_data.original_indices,
                state_vector=trajectory.state_at(0),
                vertex_labels=basis_labels,
                directed=False,
            ),
            style=graph_visualizer.style,
        )

        jump_visualizers_list: list[HamiltonianGraphVisualizer] = []

        for jump_operator in jump_operators or ():
            jump_visualizer = HamiltonianGraphVisualizer.from_sparse_matrix(
                jump_operator,
                weight_tolerance=weight_tolerance,
                directed=True,
                style=style,
            )

            if jump_visualizer.graph_data.n_vertices != trajectory.hilbert_dim:
                raise ValueError("jump operator dimension must match trajectory Hilbert dimension.")

            jump_visualizers_list.append(jump_visualizer)

        return cls(
            graph_visualizer=graph_visualizer,
            trajectory=trajectory,
            jump_visualizers=tuple(jump_visualizers_list),
        )

    def plot_frame(
        self,
        frame: int,
        *,
        backend: GraphBackend = "networkx",
        layout: LayoutName = "auto",
        color_by: TrajectoryNodeColorRule = "probability",
        edge_color_by: EdgeColorRule = "constant",
        title: str | None = None,
        show: bool = True,
        ax=None,
        **layout_kwargs,
    ):
        """Plot one trajectory frame."""
        state = self.trajectory.state_at(frame)
        time = float(self.trajectory.times[int(frame)])

        if title is None:
            title = f"stochastic trajectory, t={time:.4g}"

        return self.graph_visualizer.plot(
            backend=backend,
            layout=layout,
            color_by=_trajectory_color_to_node_color(color_by),
            edge_color_by=edge_color_by,
            state_vector=state,
            title=title,
            show=show,
            ax=ax,
            **layout_kwargs,
        )

    def animate(
        self,
        *,
        layout: LayoutName = "auto",
        color_by: TrajectoryNodeColorRule = "probability",
        edge_color_by: EdgeColorRule = "constant",
        interval: int = 100,
        repeat: bool = True,
        save_path: str | Path | None = None,
        colorbar: bool = False,
        **layout_kwargs,
    ):
        """Animate the trajectory on a fixed graph layout."""
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        # Reuse the existing layout helper indirectly by drawing the first frame.
        fig, ax = plt.subplots(figsize=self.graph_visualizer.style.figure_size)

        # Simpler first implementation: redraw each frame.
        # Later we can optimize by updating PathCollection colors only.
        def update(frame: int):
            ax.clear()

            frame_style = replace(
                self.graph_visualizer.style,
                colorbar=colorbar,
                edge_colorbar=False if not colorbar else self.graph_visualizer.style.edge_colorbar,
            )

            frame_visualizer = HamiltonianGraphVisualizer(
                graph_data=self.graph_visualizer.graph_data,
                style=frame_style,
            )

            frame_wrapper = StochasticSchrodingerGraphVisualizer(
                graph_visualizer=frame_visualizer,
                trajectory=self.trajectory,
                jump_visualizers=self.jump_visualizers,
            )

            frame_wrapper.plot_frame(
                frame,
                backend="networkx",
                layout=layout,
                color_by=color_by,
                edge_color_by=edge_color_by,
                show=False,
                ax=ax,
                **layout_kwargs,
            )

            return ax.collections

        animation = FuncAnimation(
            fig,
            update,
            frames=self.trajectory.n_times,
            interval=interval,
            repeat=repeat,
        )

        if save_path is not None:
            animation.save(str(save_path))

        return animation


def _trajectory_color_to_node_color(
    color_by: TrajectoryNodeColorRule,
) -> NodeColorRule:
    if color_by == "probability":
        return "state_weight"

    if color_by == "amplitude_abs":
        return "state_amplitude_abs"

    if color_by == "amplitude_real":
        return "state_amplitude_real"

    if color_by == "amplitude_imag":
        return "state_amplitude_imag"

    if color_by == "phase":
        return "state_phase"

    raise ValueError(f"Unsupported trajectory color rule: {color_by!r}")
