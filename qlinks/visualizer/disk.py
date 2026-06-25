from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle

from qlinks.lattice import SquareLattice
from qlinks.variables import VariableLayout
from qlinks.visualizer.basis import (
    BasisConfigLabelStyle,
    VisualizerBackend,
    _amplitude_label,
    _select_cage_record,
    _zero_indices_for_mechanism,
    _zero_mechanism_label_map,
    automatic_grid_shape,
    format_basis_config,
)


@dataclass(frozen=True, slots=True)
class QuantumDiskVisualStyle:
    """Visual style for the square quantum disk basis visualizers."""

    site_marker_size: float = 35.0
    site_marker_color: str = "lightgray"
    site_marker_alpha: float = 0.65
    site_label_fontsize: float = 8.0
    site_value_fontsize: float = 8.0

    disk_radius: float = 0.32
    disk_face_color: str = "tab:blue"
    disk_edge_color: str = "black"
    disk_alpha: float = 0.82
    disk_linewidth: float = 0.9

    blockade_edge_color: str = "lightgray"
    blockade_edge_linewidth: float = 0.9
    blockade_edge_alpha: float = 0.75

    hop_bond_color: str = "tab:purple"
    hop_bond_linewidth: float = 1.1
    hop_bond_alpha: float = 0.75
    hop_bond_linestyle: str = "--"

    occupied_site_value_color: str = "white"
    empty_site_value_color: str = "dimgray"

    axis_margin: float = 0.65


@dataclass(frozen=True, slots=True)
class _QuantumDiskGridRenderCache:
    """Reusable geometry cache for quantum disk basis-state plots."""

    site_xy: npt.NDArray[np.float64]
    site_variable_indices: npt.NDArray[np.int64]
    site_labels: tuple[str, ...]
    blockade_segments: npt.NDArray[np.float64]
    hop_segments: npt.NDArray[np.float64]
    xlim: tuple[float, float]
    ylim: tuple[float, float]


@dataclass(frozen=True)
class QuantumDiskConfigurationVisualizer:
    """Draw one quantum disk basis configuration on a square lattice.

    The visualizer interprets binary site variables as disk occupations.  It is
    intentionally site-based and therefore avoids the link/dimer-specific
    drawing conventions used by :class:`BasisConfigurationVisualizer`.
    """

    lattice: SquareLattice
    layout: VariableLayout | None = None
    style: QuantumDiskVisualStyle = field(default_factory=QuantumDiskVisualStyle)
    hop_pairs: tuple[tuple[int, int], ...] = ()
    coordinate_scale: float = 1.0
    site_label_style: str = "site_id"

    @classmethod
    def from_model(
        cls,
        model,
        *,
        style: QuantumDiskVisualStyle | None = None,
        coordinate_scale: float = 1.0,
        site_label_style: str = "site_id",
    ) -> QuantumDiskConfigurationVisualizer:
        """Construct a disk visualizer from a disk model instance."""
        hop_pairs: list[tuple[int, int]] = []
        if hasattr(model, "hop_families") and hasattr(model, "diagonal_hop_pairs"):
            for family in model.hop_families:
                hop_pairs.extend((int(i), int(j)) for i, j in model.diagonal_hop_pairs(family))

        return cls(
            lattice=model.lattice,
            layout=model.layout,
            style=style if style is not None else QuantumDiskVisualStyle(),
            hop_pairs=tuple(hop_pairs),
            coordinate_scale=coordinate_scale,
            site_label_style=site_label_style,
        )

    @classmethod
    def from_build_result(
        cls,
        result,
        *,
        style: QuantumDiskVisualStyle | None = None,
        coordinate_scale: float = 1.0,
        site_label_style: str = "site_id",
    ) -> QuantumDiskConfigurationVisualizer:
        """Construct a disk visualizer from a model build result."""
        return cls.from_model(
            result.model,
            style=style,
            coordinate_scale=coordinate_scale,
            site_label_style=site_label_style,
        )

    def _as_config(self, config: npt.ArrayLike) -> npt.NDArray[np.int64]:
        arr = np.asarray(config, dtype=np.int64)
        if arr.ndim != 1:
            raise ValueError("config must be one-dimensional.")

        if self.layout is not None:
            self.layout.validate_config(arr)
        elif arr.size != self.lattice.num_sites:
            raise ValueError(
                "Without a VariableLayout, quantum disk configs must have "
                f"exactly {self.lattice.num_sites} entries."
            )

        return arr

    def _validate_config_batch(self, states: npt.NDArray[np.int64]) -> None:
        if states.ndim != 2:
            raise ValueError("states must have shape (n_variables,) or (n_states, n_variables).")
        if self.layout is not None:
            self.layout.validate_batch(states)
        elif states.shape[1] != self.lattice.num_sites:
            raise ValueError(
                "Without a VariableLayout, quantum disk configs must have "
                f"exactly {self.lattice.num_sites} entries."
            )

    def _site_variable_index(self, site_id: int) -> int:
        if self.layout is None:
            return int(site_id)
        return int(self.layout.site_variable_index(int(site_id)))

    def _site_label(self, site_id: int) -> str:
        site = self.lattice.sites[int(site_id)]
        if self.site_label_style == "none":
            return ""
        if self.site_label_style == "site_id":
            return str(int(site_id))
        if self.site_label_style == "cell":
            return f"({int(site.cell[0])},{int(site.cell[1])})"
        if self.site_label_style == "site_id_cell":
            return f"{int(site_id)}\n({int(site.cell[0])},{int(site.cell[1])})"
        raise ValueError(
            "site_label_style must be one of 'none', 'site_id', 'cell', or 'site_id_cell'."
        )

    def _site_xy(self) -> npt.NDArray[np.float64]:
        positions = [
            self.lattice.site_embedded_position(int(site_id))[:2]
            for site_id in self.lattice.site_ids
        ]
        xy = np.asarray(positions, dtype=float)
        return xy * float(self.coordinate_scale)

    def _segments_from_site_pairs(
        self,
        pairs: Sequence[tuple[int, int]],
        site_xy: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        if not pairs:
            return np.empty((0, 2, 2), dtype=float)
        return np.asarray(
            [
                [site_xy[int(source)], site_xy[int(target)]]
                for source, target in pairs
                if int(source) != int(target)
            ],
            dtype=float,
        ).reshape(-1, 2, 2)

    def _blockade_pairs(self) -> tuple[tuple[int, int], ...]:
        return tuple((int(link.source), int(link.target)) for link in self.lattice.links)

    def build_render_cache(self) -> _QuantumDiskGridRenderCache:
        """Build reusable disk-plot geometry for this lattice/layout/style."""
        site_xy = self._site_xy()
        variable_indices = np.asarray(
            [self._site_variable_index(int(site_id)) for site_id in self.lattice.site_ids],
            dtype=np.int64,
        )
        site_labels = tuple(self._site_label(int(site_id)) for site_id in self.lattice.site_ids)
        blockade_segments = self._segments_from_site_pairs(self._blockade_pairs(), site_xy)
        hop_segments = self._segments_from_site_pairs(self.hop_pairs, site_xy)

        margin = float(self.style.axis_margin) * float(self.coordinate_scale)
        if site_xy.size == 0:
            xlim = (-margin, margin)
            ylim = (-margin, margin)
        else:
            xlim = (float(np.min(site_xy[:, 0]) - margin), float(np.max(site_xy[:, 0]) + margin))
            ylim = (float(np.min(site_xy[:, 1]) - margin), float(np.max(site_xy[:, 1]) + margin))

        return _QuantumDiskGridRenderCache(
            site_xy=site_xy,
            site_variable_indices=variable_indices,
            site_labels=site_labels,
            blockade_segments=blockade_segments,
            hop_segments=hop_segments,
            xlim=xlim,
            ylim=ylim,
        )

    def plot(
        self,
        config: npt.ArrayLike,
        *,
        ax=None,
        show: bool = True,
        backend: VisualizerBackend = "matplotlib",
        with_site_labels: bool = True,
        with_site_values: bool = False,
        with_empty_sites: bool = True,
        with_blockade_edges: bool = True,
        with_hop_bonds: bool = False,
        title: str | None = None,
        render_cache: _QuantumDiskGridRenderCache | None = None,
    ):
        """Plot one quantum disk basis state."""
        arr = self._as_config(config)
        if render_cache is None:
            render_cache = self.build_render_cache()
        if ax is None:
            _, ax = plt.subplots()
        return self._plot_with_render_cache(
            arr,
            ax=ax,
            render_cache=render_cache,
            show=show,
            backend=backend,
            with_site_labels=with_site_labels,
            with_site_values=with_site_values,
            with_empty_sites=with_empty_sites,
            with_blockade_edges=with_blockade_edges,
            with_hop_bonds=with_hop_bonds,
            title=title,
        )

    def _plot_with_render_cache(
        self,
        config: npt.NDArray[np.int64],
        *,
        ax,
        render_cache: _QuantumDiskGridRenderCache,
        show: bool = True,
        backend: VisualizerBackend = "matplotlib",
        with_site_labels: bool = True,
        with_site_values: bool = False,
        with_empty_sites: bool = True,
        with_blockade_edges: bool = True,
        with_hop_bonds: bool = False,
        title: str | None = None,
    ):
        if backend != "matplotlib":
            raise ValueError("Quantum disk visualizers currently support backend='matplotlib'.")

        style = self.style
        if with_blockade_edges and render_cache.blockade_segments.size:
            ax.add_collection(
                LineCollection(
                    render_cache.blockade_segments,
                    colors=style.blockade_edge_color,
                    linewidths=style.blockade_edge_linewidth,
                    alpha=style.blockade_edge_alpha,
                    zorder=1,
                )
            )

        if with_hop_bonds and render_cache.hop_segments.size:
            ax.add_collection(
                LineCollection(
                    render_cache.hop_segments,
                    colors=style.hop_bond_color,
                    linewidths=style.hop_bond_linewidth,
                    alpha=style.hop_bond_alpha,
                    linestyles=style.hop_bond_linestyle,
                    zorder=2,
                )
            )

        values = config[render_cache.site_variable_indices]
        occupied_mask = values != 0

        if with_empty_sites and render_cache.site_xy.size:
            empty_xy = render_cache.site_xy[~occupied_mask]
            if empty_xy.size:
                ax.scatter(
                    empty_xy[:, 0],
                    empty_xy[:, 1],
                    s=style.site_marker_size,
                    color=style.site_marker_color,
                    alpha=style.site_marker_alpha,
                    zorder=3,
                )

        radius = float(style.disk_radius) * float(self.coordinate_scale)
        for xy in render_cache.site_xy[occupied_mask]:
            ax.add_patch(
                Circle(
                    (float(xy[0]), float(xy[1])),
                    radius=radius,
                    facecolor=style.disk_face_color,
                    edgecolor=style.disk_edge_color,
                    linewidth=style.disk_linewidth,
                    alpha=style.disk_alpha,
                    zorder=4,
                )
            )

        if with_site_labels or with_site_values:
            for site_index, (xy, value) in enumerate(
                zip(render_cache.site_xy, values, strict=True)
            ):
                pieces: list[str] = []
                if with_site_labels:
                    label = render_cache.site_labels[site_index]
                    if label:
                        pieces.append(label)
                if with_site_values:
                    pieces.append(str(int(value)))
                if not pieces:
                    continue
                occupied = bool(occupied_mask[site_index])
                ax.text(
                    float(xy[0]),
                    float(xy[1]),
                    "\n".join(pieces),
                    ha="center",
                    va="center",
                    fontsize=(
                        style.site_value_fontsize if with_site_values else style.site_label_fontsize
                    ),
                    color=(
                        style.occupied_site_value_color
                        if occupied
                        else style.empty_site_value_color
                    ),
                    zorder=5,
                )

        ax.set_aspect("equal")
        ax.set_xlim(*render_cache.xlim)
        ax.set_ylim(*render_cache.ylim)
        ax.set_xticks([])
        ax.set_yticks([])
        if title is not None:
            ax.set_title(title)

        if show:
            plt.show()
        return ax

    def save(
        self,
        config: npt.ArrayLike,
        path: str | Path,
        *,
        dpi: int = 200,
        show: bool = False,
        **plot_kwargs,
    ) -> None:
        fig, ax = plt.subplots()
        self.plot(config, ax=ax, show=show, **plot_kwargs)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


@dataclass(frozen=True)
class QuantumDiskBasisGridVisualizer:
    """Plot batches of quantum disk basis states.

    The public methods intentionally mirror the generic
    :class:`BasisGridVisualizer` API: ``plot`` for arbitrary basis-state batches,
    ``plot_cage_support`` for cage records, and ``plot_interference_zeros`` for
    classification reports.
    """

    lattice: SquareLattice
    layout: VariableLayout | None = None
    style: QuantumDiskVisualStyle = field(default_factory=QuantumDiskVisualStyle)
    hop_pairs: tuple[tuple[int, int], ...] = ()
    coordinate_scale: float = 1.0
    site_label_style: str = "site_id"

    @classmethod
    def from_model(
        cls,
        model,
        *,
        style: QuantumDiskVisualStyle | None = None,
        coordinate_scale: float = 1.0,
        site_label_style: str = "site_id",
    ) -> QuantumDiskBasisGridVisualizer:
        single = QuantumDiskConfigurationVisualizer.from_model(
            model,
            style=style,
            coordinate_scale=coordinate_scale,
            site_label_style=site_label_style,
        )
        return cls(
            lattice=single.lattice,
            layout=single.layout,
            style=single.style,
            hop_pairs=single.hop_pairs,
            coordinate_scale=single.coordinate_scale,
            site_label_style=single.site_label_style,
        )

    @classmethod
    def from_build_result(
        cls,
        result,
        *,
        style: QuantumDiskVisualStyle | None = None,
        coordinate_scale: float = 1.0,
        site_label_style: str = "site_id",
    ) -> QuantumDiskBasisGridVisualizer:
        return cls.from_model(
            result.model,
            style=style,
            coordinate_scale=coordinate_scale,
            site_label_style=site_label_style,
        )

    def _single_visualizer(self) -> QuantumDiskConfigurationVisualizer:
        return QuantumDiskConfigurationVisualizer(
            lattice=self.lattice,
            layout=self.layout,
            style=self.style,
            hop_pairs=self.hop_pairs,
            coordinate_scale=self.coordinate_scale,
            site_label_style=self.site_label_style,
        )

    def build_render_cache(self) -> _QuantumDiskGridRenderCache:
        """Build a reusable render cache for repeated disk-grid plotting."""
        return self._single_visualizer().build_render_cache()

    def plot(
        self,
        states: npt.ArrayLike,
        *,
        nrows: int | None = None,
        ncols: int | None = None,
        start_index: int = 0,
        labels: Sequence[str] | None = None,
        show_config_label: bool = False,
        config_label_style: BasisConfigLabelStyle = "compact",
        config_label_max_length: int = 48,
        figsize: tuple[float, float] | None = None,
        show: bool = True,
        backend: VisualizerBackend = "matplotlib",
        suptitle: str | None = None,
        suptitle_y: float = 0.995,
        tight_layout_rect: tuple[float, float, float, float] | None = None,
        single_plot_kwargs: dict | None = None,
        render_cache: _QuantumDiskGridRenderCache | None = None,
    ):
        """Plot a batch of quantum disk basis configurations."""
        arr = np.asarray(states, dtype=np.int64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError("states must have shape (n_variables,) or (n_states, n_variables).")
        if arr.shape[0] == 0:
            raise ValueError("states must contain at least one configuration.")

        single = self._single_visualizer()
        single._validate_config_batch(arr)
        if render_cache is None:
            render_cache = single.build_render_cache()

        rows, cols = automatic_grid_shape(arr.shape[0], nrows=nrows, ncols=ncols)
        if labels is not None and len(labels) != arr.shape[0]:
            raise ValueError("labels must have the same length as states.")
        if figsize is None:
            figsize = (2.8 * cols, 2.8 * rows)

        fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
        if single_plot_kwargs is None:
            single_plot_kwargs = {}
        plot_kwargs = dict(single_plot_kwargs)
        plot_kwargs.pop("title", None)
        plot_kwargs.pop("show", None)
        plot_kwargs.pop("backend", None)
        plot_kwargs.pop("ax", None)
        plot_kwargs.pop("render_cache", None)

        for k in range(rows * cols):
            ax = axes.flat[k]
            if k >= arr.shape[0]:
                ax.axis("off")
                continue

            if labels is None:
                title = f"state {start_index + k}"
            else:
                title = labels[k]

            if show_config_label:
                config_text = format_basis_config(
                    arr[k],
                    style=config_label_style,
                    max_length=config_label_max_length,
                )
                if config_text:
                    title = f"{title}\n{config_text}"

            single._plot_with_render_cache(
                arr[k],
                ax=ax,
                render_cache=render_cache,
                show=False,
                backend=backend,
                title=title,
                **plot_kwargs,
            )

        if suptitle is not None:
            fig.suptitle(suptitle, y=suptitle_y)

        if tight_layout_rect is None:
            tight_layout_rect = (0.0, 0.0, 1.0, 1.0 if suptitle is None else 0.96)
        fig.tight_layout(rect=tight_layout_rect)

        if show:
            plt.show()
        return fig, axes

    def plot_basis(self, basis, **plot_kwargs):
        """Plot all states in a Basis-like object with a ``states`` attribute."""
        return self.plot(basis.states, **plot_kwargs)

    def plot_cage_support(
        self,
        result_or_record,
        *,
        basis_configs: npt.ArrayLike,
        signature: tuple[int, int] | None = None,
        record_index: int = 0,
        max_states: int | None = None,
        show_amplitudes: bool = True,
        amplitude_digits: int = 3,
        labels: Sequence[str] | None = None,
        suptitle: str | None = None,
        **plot_kwargs,
    ):
        """Plot the disk basis states in one cage support."""
        basis_configs = np.asarray(basis_configs)
        record = _select_cage_record(
            result_or_record,
            signature=signature,
            record_index=record_index,
        )
        support = np.asarray(record.support, dtype=np.int64)
        local_state = np.asarray(record.local_state, dtype=np.complex128)
        if max_states is not None:
            support = support[:max_states]
            local_state = local_state[:max_states]
        states = basis_configs[support]

        if labels is None:
            if show_amplitudes:
                labels = [
                    _amplitude_label(
                        basis_index=int(index),
                        amplitude=complex(amplitude),
                        digits=amplitude_digits,
                    )
                    for index, amplitude in zip(support, local_state, strict=True)
                ]
            else:
                labels = [f"basis {int(index)}" for index in support]

        if suptitle is None:
            suptitle = (
                f"Quantum disk cage support, signature={record.signature}, "
                f"support size={record.support.size}"
            )

        return self.plot(states, labels=labels, suptitle=suptitle, **plot_kwargs)

    def plot_interference_zeros(
        self,
        classification_report,
        *,
        basis_configs: npt.ArrayLike,
        mechanism: str = "all",
        max_states: int | None = None,
        labels: Sequence[str] | None = None,
        suptitle: str | None = None,
        **plot_kwargs,
    ):
        """Plot disk basis states corresponding to nontrivial interference zeros."""
        basis_configs = np.asarray(basis_configs)
        zero_indices = _zero_indices_for_mechanism(classification_report, mechanism)
        if max_states is not None:
            zero_indices = zero_indices[:max_states]
        states = basis_configs[zero_indices]
        mechanism_labels = _zero_mechanism_label_map(classification_report)

        if labels is None:
            labels = [
                f"zero {int(index)}\n{mechanism_labels.get(int(index), mechanism)}"
                for index in zero_indices
            ]

        if suptitle is None:
            if mechanism == "all":
                suptitle = f"Quantum disk interference zeros ({zero_indices.size} states)"
            else:
                suptitle = (
                    f"Quantum disk interference zeros: {mechanism} " f"({zero_indices.size} states)"
                )

        return self.plot(states, labels=labels, suptitle=suptitle, **plot_kwargs)


def plot_quantum_disk_basis_grid(
    states: npt.ArrayLike,
    *,
    lattice: SquareLattice | None = None,
    layout: VariableLayout | None = None,
    model=None,
    result=None,
    nrows: int | None = None,
    ncols: int | None = None,
    start_index: int = 0,
    labels: Sequence[str] | None = None,
    show_config_label: bool = False,
    config_label_style: BasisConfigLabelStyle = "compact",
    config_label_max_length: int = 48,
    style: QuantumDiskVisualStyle | None = None,
    figsize: tuple[float, float] | None = None,
    show: bool = True,
    backend: VisualizerBackend = "matplotlib",
    suptitle: str | None = None,
    single_plot_kwargs: dict | None = None,
):
    """Functional wrapper around :class:`QuantumDiskBasisGridVisualizer`."""
    if result is not None:
        visualizer = QuantumDiskBasisGridVisualizer.from_build_result(
            result,
            style=style,
        )
    elif model is not None:
        visualizer = QuantumDiskBasisGridVisualizer.from_model(
            model,
            style=style,
        )
    else:
        if lattice is None:
            raise ValueError("Either result, model, or lattice must be provided.")
        visualizer = QuantumDiskBasisGridVisualizer(
            lattice=lattice,
            layout=layout,
            style=style if style is not None else QuantumDiskVisualStyle(),
        )

    return visualizer.plot(
        states,
        nrows=nrows,
        ncols=ncols,
        start_index=start_index,
        labels=labels,
        show_config_label=show_config_label,
        config_label_style=config_label_style,
        config_label_max_length=config_label_max_length,
        figsize=figsize,
        show=show,
        backend=backend,
        suptitle=suptitle,
        single_plot_kwargs=single_plot_kwargs,
    )
