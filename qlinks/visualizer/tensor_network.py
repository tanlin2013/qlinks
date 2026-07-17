"""Visualization helpers for the constrained square-QDM PEPS prototype."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.patches import Circle, Rectangle

from qlinks.caging.tensor_network import (
    SquareQDMPEPSOptimizationResult,
    SquareQDMRectangularTileTensorBasis,
)


def _bits(pattern: int, count: int) -> tuple[int, ...]:
    return tuple((int(pattern) >> index) & 1 for index in range(int(count)))


def _axes(ax: Axes | None, *, figsize: tuple[float, float]) -> tuple[Any, Axes]:
    if ax is not None:
        return ax.figure, ax
    return plt.subplots(figsize=figsize)


@dataclass(frozen=True, slots=True)
class SquareQDMTensorNetworkVisualStyle:
    """Plotting controls for the square-QDM tensor-network visualizer."""

    tensor_size: float = 0.46
    site_radius: float = 0.06
    occupied_width: float = 3.0
    empty_width: float = 0.8
    physical_leg_length: float = 0.42
    wrap_offset: float = 0.22
    parameter_floor: float = 1.0e-14


@dataclass(slots=True)
class SquareQDMTensorNetworkVisualizer:
    """Visualize the PEPS graph, local tensor entries, and optimization traces."""

    tile_basis: SquareQDMRectangularTileTensorBasis
    style: SquareQDMTensorNetworkVisualStyle = SquareQDMTensorNetworkVisualStyle()

    def plot_network(
        self,
        *,
        n_tiles_x: int = 3,
        n_tiles_y: int = 2,
        periodic: bool = True,
        show_bond_dimensions: bool = True,
        show_physical_legs: bool = True,
        ax: Axes | None = None,
        title: str | None = None,
    ) -> Axes:
        """Draw the repeated tensor graph and grouped virtual dimensions."""
        if n_tiles_x <= 0 or n_tiles_y <= 0:
            raise ValueError("n_tiles_x and n_tiles_y must be positive.")
        _, axis = _axes(ax, figsize=(2.0 + 1.7 * n_tiles_x, 1.8 + 1.5 * n_tiles_y))

        positions = {
            (x, y): (float(x), float(y))
            for y in range(int(n_tiles_y))
            for x in range(int(n_tiles_x))
        }
        for (x, y), (px, py) in positions.items():
            if x + 1 < n_tiles_x:
                axis.plot((px, px + 1.0), (py, py), linewidth=1.5)
                if show_bond_dimensions:
                    axis.text(
                        px + 0.5,
                        py + 0.06,
                        f"$D_x={self.tile_basis.right_dimension}$",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )
            elif periodic and n_tiles_x > 1:
                offset = self.style.wrap_offset
                axis.plot(
                    (px, px + offset, -offset, 0.0),
                    (py, py + offset, py + offset, py),
                    linestyle="--",
                    linewidth=1.0,
                )
            if y + 1 < n_tiles_y:
                axis.plot((px, px), (py, py + 1.0), linewidth=1.5)
                if show_bond_dimensions:
                    axis.text(
                        px + 0.05,
                        py + 0.5,
                        f"$D_y={self.tile_basis.up_dimension}$",
                        ha="left",
                        va="center",
                        fontsize=8,
                        rotation=90,
                    )
            elif periodic and n_tiles_y > 1:
                offset = self.style.wrap_offset
                axis.plot(
                    (px, px + offset, px + offset, px),
                    (py, py + offset, -offset, 0.0),
                    linestyle="--",
                    linewidth=1.0,
                )

        for (x, y), (px, py) in positions.items():
            half = self.style.tensor_size / 2.0
            axis.add_patch(
                Rectangle((px - half, py - half), 2.0 * half, 2.0 * half, fill=True, alpha=0.85)
            )
            axis.text(px, py, f"A\n({x},{y})", ha="center", va="center", fontsize=9)
            if show_physical_legs:
                axis.plot(
                    (px, px), (py - half, py - half - self.style.physical_leg_length), linewidth=1.2
                )
                axis.text(
                    px,
                    py - half - self.style.physical_leg_length - 0.05,
                    f"$d={self.tile_basis.physical_dimension}$",
                    ha="center",
                    va="top",
                    fontsize=8,
                )

        axis.set_aspect("equal")
        axis.set_xlim(-0.65, n_tiles_x - 0.35)
        axis.set_ylim(-0.95, n_tiles_y - 0.35)
        axis.axis("off")
        axis.set_title(title or "Square-QDM constrained PEPS")
        return axis

    def plot_entry(
        self,
        entry_index: int,
        *,
        ax: Axes | None = None,
        title: str | None = None,
        show_empty_links: bool = True,
    ) -> Axes:
        """Draw one allowed local tensor entry as a dimer configuration."""
        entry_index = int(entry_index)
        if entry_index < 0 or entry_index >= self.tile_basis.n_entries:
            raise IndexError("entry_index is outside the allowed tensor-entry range.")
        _, axis = _axes(
            ax,
            figsize=(2.0 + self.tile_basis.tile_lx, 2.0 + self.tile_basis.tile_ly),
        )
        coordinate = self.tile_basis.entry_coordinates[entry_index]
        up, right, down, left, physical_index = (int(value) for value in coordinate)
        configuration = self.tile_basis.physical_configurations[physical_index]

        link_values = {
            tuple(key): int(configuration[position])
            for position, key in enumerate(self.tile_basis.owned_link_keys)
        }
        for x in range(self.tile_basis.tile_lx):
            for y in range(self.tile_basis.tile_ly):
                axis.add_patch(Circle((x, y), radius=self.style.site_radius, fill=True))
                for kind, delta in (("x", (1.0, 0.0)), ("y", (0.0, 1.0))):
                    occupied = link_values[(x, y, kind)]
                    if occupied or show_empty_links:
                        axis.plot(
                            (x, x + delta[0]),
                            (y, y + delta[1]),
                            linewidth=(
                                self.style.occupied_width if occupied else self.style.empty_width
                            ),
                            alpha=1.0 if occupied else 0.25,
                        )

        for y, occupied in enumerate(_bits(left, self.tile_basis.tile_ly)):
            if occupied or show_empty_links:
                axis.plot(
                    (-1.0, 0.0),
                    (y, y),
                    linewidth=self.style.occupied_width if occupied else self.style.empty_width,
                    alpha=1.0 if occupied else 0.25,
                )
        for x, occupied in enumerate(_bits(down, self.tile_basis.tile_lx)):
            if occupied or show_empty_links:
                axis.plot(
                    (x, x),
                    (-1.0, 0.0),
                    linewidth=self.style.occupied_width if occupied else self.style.empty_width,
                    alpha=1.0 if occupied else 0.25,
                )

        axis.add_patch(
            Rectangle(
                (-0.15, -0.15),
                self.tile_basis.tile_lx - 0.7,
                self.tile_basis.tile_ly - 0.7,
                fill=False,
                linestyle="--",
                linewidth=1.0,
            )
        )
        axis.set_aspect("equal")
        axis.set_xlim(-1.15, self.tile_basis.tile_lx + 0.15)
        axis.set_ylim(-1.15, self.tile_basis.tile_ly + 0.15)
        axis.axis("off")
        axis.set_title(
            title
            or (
                f"Tensor entry {entry_index}: physical={physical_index}, "
                f"(u,r,d,l)=({up},{right},{down},{left})"
            )
        )
        return axis

    def plot_parameter_magnitudes(
        self,
        parameters: npt.ArrayLike,
        *,
        max_entries: int = 24,
        ax: Axes | None = None,
        title: str | None = None,
    ) -> Axes:
        """Plot the largest compact tensor-entry amplitudes."""
        values = np.asarray(parameters, dtype=np.complex128).reshape(-1)
        if values.size != self.tile_basis.n_entries:
            raise ValueError(f"parameters must have size {self.tile_basis.n_entries}.")
        max_entries = max(1, min(int(max_entries), values.size))
        order = np.argsort(np.abs(values))[::-1][:max_entries]
        _, axis = _axes(ax, figsize=(max(6.0, 0.32 * max_entries), 3.6))
        axis.bar(np.arange(max_entries), np.abs(values[order]))
        axis.set_xticks(np.arange(max_entries), [str(index) for index in order], rotation=90)
        axis.set_xlabel("Tensor-entry index")
        axis.set_ylabel("Amplitude magnitude")
        axis.set_title(title or f"Largest {max_entries} unit-tensor amplitudes")
        axis.grid(axis="y", alpha=0.25)
        return axis

    def plot_optimization_history(
        self,
        result: SquareQDMPEPSOptimizationResult | Sequence[float],
        *,
        ax: Axes | None = None,
        log_scale: bool = True,
        title: str | None = None,
    ) -> Axes:
        """Plot the exact finite-cluster variance during optimization."""
        if isinstance(result, SquareQDMPEPSOptimizationResult):
            losses = np.asarray(result.loss_history, dtype=np.float64)
        else:
            losses = np.asarray(tuple(result), dtype=np.float64)
        if losses.size == 0:
            raise ValueError("At least one loss value is required.")
        _, axis = _axes(ax, figsize=(6.0, 3.8))
        axis.plot(np.arange(losses.size), losses, marker="o", markersize=3)
        if log_scale and np.all(losses > 0.0):
            axis.set_yscale("log")
        axis.set_xlabel("Function evaluation")
        axis.set_ylabel(r"Energy variance $\mathcal{V}_H$")
        axis.set_title(title or "PEPS finite-cluster optimization")
        axis.grid(alpha=0.25)
        return axis
