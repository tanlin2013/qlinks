from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from qlinks.lattice import LatticeGraph
from qlinks.variables import VariableKind, VariableLayout
from qlinks.visualizer.basis.configuration import BasisConfigurationVisualizer
from qlinks.visualizer.basis.formatting import automatic_grid_shape, format_basis_config
from qlinks.visualizer.basis.render_cache import _BasisGridRenderCache
from qlinks.visualizer.basis.styles import (
    BasisConfigLabelStyle,
    BasisVisualizerTheme,
    LinkPlotMode,
    LinkVisualStyle,
    LocalBasisShadowStyle,
    MatrixElementValueRole,
    PeriodicImageMode,
    PlaquetteSymbolStyle,
    SiteLabelStyle,
    VisualizerBackend,
    _basis_visualizer_theme_defaults,
)


@dataclass(frozen=True, slots=True)
class _LocalStructurePlotEntry:
    """One local-structure basis pattern to draw."""

    variable_indices: tuple[int, ...]
    pattern: tuple[int, ...]
    label: str
    plaquette_symbols: PlaquetteSymbolStyle
    show_pattern_label: bool = True


def _format_local_structure_coefficient(value: complex) -> str:
    if abs(value.imag) <= 1e-10:
        return f"{value.real:.3g}"
    return f"{value:.3g}"


def _structure_component_prefix(readout: Any) -> str:
    component_index = getattr(readout, "component_index", None)
    if component_index is None:
        return ""
    return f"comp {component_index}: "


def _local_structure_entries_from_readout_report(
    report: Any,
    *,
    max_structures: int | None,
    max_basis_states: int | None,
    include_frozen: bool,
    max_frozen: int | None,
    coherent_plaquette_symbols: PlaquetteSymbolStyle,
    frozen_plaquette_symbols: PlaquetteSymbolStyle,
) -> list[_LocalStructurePlotEntry]:
    readout = report.readout
    variable_indices = tuple(int(index) for index in readout.variable_indices)
    prefix = _structure_component_prefix(readout)
    entries: list[_LocalStructurePlotEntry] = []

    coherent_pairs = tuple(getattr(report, "coherent_pairs", ()))
    if max_structures is not None:
        coherent_pairs = coherent_pairs[: int(max_structures)]

    max_states = 2 if max_basis_states is None else max(0, int(max_basis_states))

    for pair_index, pair in enumerate(coherent_pairs):
        pair_kind = "singlet" if bool(getattr(pair, "is_singlet_like", False)) else "coherent"
        coeff_labels = ["+1/sqrt(2)"]
        sign_label = str(getattr(pair, "sign_label", "+"))
        if sign_label == "+":
            coeff_labels.append("+1/sqrt(2)")
        elif sign_label == "-":
            coeff_labels.append("-1/sqrt(2)")
        else:
            _coeff = _format_local_structure_coefficient(
                getattr(pair, "relative_phase", 1.0 + 0.0j)
            )
            coeff_labels.append(f"({_coeff})/sqrt(2)")

        patterns = [
            tuple(int(v) for v in pair.pattern_a),
            tuple(int(v) for v in pair.pattern_b),
        ]
        for state_index, (pattern, coeff_label) in enumerate(
            zip(patterns[:max_states], coeff_labels[:max_states], strict=True)
        ):
            entries.append(
                _LocalStructurePlotEntry(
                    variable_indices=variable_indices,
                    pattern=pattern,
                    label=(
                        f"{prefix}{pair_kind} {pair_index}, state {state_index}\n"
                        f"{coeff_label}, weight={float(getattr(pair, 'weight', 0.0)):.3g}"
                    ),
                    plaquette_symbols=coherent_plaquette_symbols,
                    show_pattern_label=True,
                )
            )

    if include_frozen:
        classical_sectors = tuple(getattr(report, "classical_sectors", ()))
        if max_frozen is not None:
            classical_sectors = classical_sectors[: int(max_frozen)]
        for sector_index, sector in enumerate(classical_sectors):
            entries.append(
                _LocalStructurePlotEntry(
                    variable_indices=variable_indices,
                    pattern=tuple(int(v) for v in sector.pattern),
                    label=(
                        f"{prefix}frozen {sector_index}\n"
                        f"weight={float(getattr(sector, 'weight', 0.0)):.3g}"
                    ),
                    plaquette_symbols=frozen_plaquette_symbols,
                    show_pattern_label=True,
                )
            )

    return entries


@dataclass(frozen=True)
class LocalBasisGridVisualizer:
    """Plot local basis patterns on top of the full lattice geometry.

    This visualizer is intended for local reduced-density-matrix and local
    recycler readouts.  It embeds each local pattern into a synthetic or
    user-supplied full-lattice background, draws the full lattice with the
    usual :class:`BasisConfigurationVisualizer` geometry, and shadows every
    site/link outside ``variable_indices``.  A full constrained-basis
    configuration is therefore optional; only the finite local basis is needed
    for the local variables being inspected.
    """

    lattice: LatticeGraph
    layout: VariableLayout | None = None
    style: LinkVisualStyle | None = None
    theme: BasisVisualizerTheme = "research"
    shadow_style: LocalBasisShadowStyle = field(default_factory=LocalBasisShadowStyle)
    periodic_image_mode: PeriodicImageMode = "positive_patch"
    collapse_duplicate_visual_links: bool = True
    coordinate_scale: float = 1.0
    coordinate_transform: npt.ArrayLike | None = None
    # Use the same compact sublattice-first convention commonly used for
    # honeycomb full-basis plots, e.g. ``A(0, 0)`` rather than ``(0, 0), A``.
    # Single-sublattice lattices are unaffected because the base formatter
    # omits the sublattice label when there is only one basis offset.
    site_label_style: SiteLabelStyle = "sublattice_cell"

    def __post_init__(self) -> None:
        defaults = _basis_visualizer_theme_defaults(self.theme)
        if self.style is None:
            object.__setattr__(self, "style", defaults.style)

    def _single_visualizer(self) -> BasisConfigurationVisualizer:
        return BasisConfigurationVisualizer(
            lattice=self.lattice,
            layout=self.layout,
            theme=self.theme,
            style=self.style,
            periodic_image_mode=self.periodic_image_mode,
            collapse_duplicate_visual_links=self.collapse_duplicate_visual_links,
            coordinate_scale=self.coordinate_scale,
            coordinate_transform=self.coordinate_transform,
            site_label_style=self.site_label_style,
        )

    def build_render_cache(
        self,
        *,
        reference_config: npt.ArrayLike | None = None,
        mode: LinkPlotMode = "auto",
        plaquette_symbols: PlaquetteSymbolStyle = "none",
    ) -> _BasisGridRenderCache:
        """Build a reusable render cache for local-basis plots."""
        reference = self._resolve_reference_config(reference_config)
        return self._single_visualizer().build_grid_render_cache(
            reference_config=reference,
            mode=mode,
            plaquette_symbols=plaquette_symbols,
        )

    def plot(
        self,
        local_patterns: npt.ArrayLike,
        *,
        variable_indices: Sequence[int],
        reference_config: npt.ArrayLike | None = None,
        nrows: int | None = None,
        ncols: int | None = None,
        start_index: int = 0,
        labels: Sequence[str] | None = None,
        show_local_pattern_label: bool = True,
        config_label_style: BasisConfigLabelStyle = "compact",
        config_label_max_length: int = 48,
        mode: LinkPlotMode = "auto",
        plaquette_symbols: PlaquetteSymbolStyle = "none",
        figsize: tuple[float, float] | None = None,
        show: bool = True,
        backend: VisualizerBackend = "matplotlib",
        suptitle: str | None = None,
        suptitle_y: float = 0.995,
        tight_layout_rect: tuple[float, float, float, float] | None = None,
        single_plot_kwargs: dict | None = None,
        render_cache: _BasisGridRenderCache | None = None,
        local_operator: npt.ArrayLike | None = None,
        show_only_nonzero_matrix_elements: bool = False,
        matrix_element_tolerance: float = 1e-10,
        show_matrix_element_values: bool = False,
        matrix_element_value_role: MatrixElementValueRole = "both",
        max_matrix_element_values_per_pattern: int = 6,
        matrix_element_value_precision: int = 3,
    ):
        """Plot local patterns, highlighting only ``variable_indices``.

        Parameters
        ----------
        local_patterns:
            Local basis patterns with shape ``(n_patterns, n_local_variables)``.
            For a single local variable, a one-dimensional input is interpreted
            as several one-variable patterns.
        variable_indices:
            Indices in the full configuration array corresponding to the local
            pattern entries.
        reference_config:
            Optional full configuration used as the background outside the local
            support.  If omitted, a synthetic background is used.  Nonlocal
            variables are shadowed, so the synthetic values are not meant to be
            interpreted as a physical basis state.
        local_operator:
            Optional local matrix/operator in the same pattern order.  When
            ``show_only_nonzero_matrix_elements=True``, only patterns appearing
            in a nonzero row or column of this matrix are drawn.
        show_matrix_element_values:
            If true, append nonzero local matrix entries touching each displayed
            pattern to the subplot title.  Rows are labelled as outgoing
            ``<target|O|this>`` entries and columns as incoming
            ``<this|O|source>`` entries.
        """
        variable_key = _normalize_local_variable_indices(variable_indices)
        reference = self._resolve_reference_config(reference_config)
        patterns = _as_local_basis_patterns(
            local_patterns,
            n_local_variables=len(variable_key),
        )

        if patterns.shape[0] == 0:
            raise ValueError("local_patterns must contain at least one pattern.")

        operator_array = None
        if local_operator is not None:
            operator_array = np.asarray(local_operator, dtype=np.complex128)
            if operator_array.shape != (patterns.shape[0], patterns.shape[0]):
                raise ValueError(
                    "local_operator shape must match the number of local patterns: "
                    f"{operator_array.shape} != {(patterns.shape[0], patterns.shape[0])}."
                )

        displayed_pattern_indices = np.arange(patterns.shape[0], dtype=np.int64)
        if show_only_nonzero_matrix_elements:
            if operator_array is None:
                raise ValueError(
                    "local_operator is required when show_only_nonzero_matrix_elements=True."
                )
            selected_pattern_indices = _nonzero_local_operator_pattern_indices(
                operator_array,
                n_patterns=patterns.shape[0],
                tolerance=matrix_element_tolerance,
            )
            if selected_pattern_indices.size == 0:
                raise ValueError("No local patterns participate in nonzero matrix elements.")
            patterns = patterns[selected_pattern_indices]
            labels = _select_local_pattern_labels(labels, selected_pattern_indices)
            displayed_pattern_indices = selected_pattern_indices

        matrix_element_labels = None
        if show_matrix_element_values:
            if operator_array is None:
                raise ValueError("local_operator is required when show_matrix_element_values=True.")
            matrix_element_labels = _matrix_element_value_labels_for_patterns(
                operator_array,
                displayed_pattern_indices=displayed_pattern_indices,
                tolerance=matrix_element_tolerance,
                role=matrix_element_value_role,
                max_terms_per_pattern=max_matrix_element_values_per_pattern,
                precision=matrix_element_value_precision,
            )

        embedded_configs = _embed_local_patterns(
            reference_config=reference,
            local_patterns=patterns,
            variable_indices=variable_key,
        )

        single_visualizer = self._single_visualizer()
        single_visualizer._validate_config_batch_for_cached_grid(embedded_configs)

        if render_cache is None:
            render_cache = single_visualizer.build_grid_render_cache(
                reference_config=embedded_configs[0],
                mode=mode,
                plaquette_symbols=plaquette_symbols,
            )

        active_link_mask, active_node_mask = self._active_artist_masks(
            variable_indices=variable_key,
            render_cache=render_cache,
        )

        rows, cols = automatic_grid_shape(
            patterns.shape[0],
            nrows=nrows,
            ncols=ncols,
        )

        if labels is not None and len(labels) != patterns.shape[0]:
            raise ValueError("labels must have the same length as local_patterns.")

        if figsize is None:
            figsize = (3.0 * cols, 3.0 * rows)

        fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)

        if single_plot_kwargs is None:
            single_plot_kwargs = {}

        plot_kwargs = dict(single_plot_kwargs)
        with_site_labels = bool(
            plot_kwargs.pop(
                "with_site_labels",
                single_visualizer._theme_defaults.with_site_labels,
            )
        )
        with_coordinate_labels = bool(
            plot_kwargs.pop(
                "with_coordinate_labels",
                single_visualizer._theme_defaults.with_coordinate_labels,
            )
        )
        with_site_values = bool(plot_kwargs.pop("with_site_values", False))
        with_link_values = bool(plot_kwargs.pop("with_link_values", False))
        with_link_ids = bool(plot_kwargs.pop("with_link_ids", False))
        with_plaquette_symbols = bool(plot_kwargs.pop("with_plaquette_symbols", False))
        plaquette_symbol_values = plot_kwargs.pop("plaquette_symbol_values", None)
        plot_kwargs.pop("title", None)
        plot_kwargs.pop("show", None)
        plot_kwargs.pop("backend", None)
        plot_kwargs.pop("ax", None)
        plot_kwargs.pop("mode", None)

        # Constructor-only options; do not pass them to the single-state renderer.
        plot_kwargs.pop("style", None)
        plot_kwargs.pop("shadow_style", None)
        plot_kwargs.pop("periodic_image_mode", None)
        plot_kwargs.pop("collapse_duplicate_visual_links", None)
        plot_kwargs.pop("coordinate_scale", None)
        plot_kwargs.pop("coordinate_transform", None)
        plot_kwargs.pop("site_label_style", None)

        for k in range(rows * cols):
            ax = axes.flat[k]

            if k >= patterns.shape[0]:
                ax.axis("off")
                continue

            if labels is None:
                title = f"local {start_index + k}"
            else:
                title = labels[k]

            if show_local_pattern_label:
                pattern_text = format_basis_config(
                    patterns[k],
                    style=config_label_style,
                    max_length=config_label_max_length,
                )
                if pattern_text:
                    title = f"{title}\n{pattern_text}"

            if matrix_element_labels is not None and matrix_element_labels[k]:
                title = f"{title}\n{matrix_element_labels[k]}"

            single_visualizer._plot_local_basis_with_grid_render_cache(
                embedded_configs[k],
                ax=ax,
                render_cache=render_cache,
                active_link_mask=active_link_mask,
                active_node_mask=active_node_mask,
                shadow_style=self.shadow_style,
                show=False,
                backend=backend,
                with_site_labels=with_site_labels,
                with_coordinate_labels=with_coordinate_labels,
                with_site_values=with_site_values,
                with_link_values=with_link_values,
                with_link_ids=with_link_ids,
                with_plaquette_symbols=with_plaquette_symbols
                and render_cache.plaquette_symbol_style != "none",
                plaquette_symbol_values=plaquette_symbol_values,
                title=title,
                **plot_kwargs,
            )

        if suptitle is not None:
            fig.suptitle(suptitle, y=suptitle_y)

        if tight_layout_rect is None:
            if suptitle is None:
                tight_layout_rect = (0.0, 0.0, 1.0, 1.0)
            else:
                tight_layout_rect = (0.0, 0.0, 1.0, 0.96)

        fig.tight_layout(rect=tight_layout_rect)

        if show:
            plt.show()

        return fig, axes

    def plot_readout(
        self,
        readout,
        *,
        reference_config: npt.ArrayLike | None = None,
        labels: Sequence[str] | None = None,
        suptitle: str | None = None,
        show_only_nonzero_matrix_elements: bool = True,
        matrix_element_tolerance: float = 1e-10,
        show_matrix_element_values: bool = False,
        matrix_element_value_role: MatrixElementValueRole = "both",
        max_matrix_element_values_per_pattern: int = 6,
        matrix_element_value_precision: int = 3,
        **plot_kwargs,
    ):
        """Plot the local patterns exposed by a local-RDM-style readout.

        The method intentionally uses duck typing so the visualizer does not
        depend on ``qlinks.caging`` or ``qlinks.open_system``.
        """
        if not hasattr(readout, "local_patterns") or not hasattr(readout, "variable_indices"):
            readout_type = type(readout).__name__
            raise TypeError(
                "plot_readout expects a local matrix readout with local_patterns "
                f"and variable_indices; got {readout_type}.  "
                "Use workflow.local_operator_readouts(), "
                "workflow.recycled_recycler_readouts(), or "
                "workflow.targeted_operator_readouts() for local-basis plots.  "
                "workflow.detector_readouts() returns global detector coefficient readouts."
            )

        if suptitle is None:
            component_index = getattr(readout, "component_index", None)
            if component_index is None:
                suptitle = "Local basis patterns"
            else:
                suptitle = f"Local basis patterns, component {component_index}"

        local_operator = _local_operator_from_readout(readout)
        matrix_unit_terms = (
            None if local_operator is not None else getattr(readout, "matrix_unit_terms", None)
        )

        local_patterns = readout.local_patterns
        if (
            show_only_nonzero_matrix_elements
            and local_operator is None
            and matrix_unit_terms is not None
        ):
            selected_pattern_indices = _nonzero_matrix_unit_pattern_indices(
                matrix_unit_terms=matrix_unit_terms,
                local_patterns=local_patterns,
            )
            local_patterns = _select_local_patterns(local_patterns, selected_pattern_indices)
            labels = _select_local_pattern_labels(labels, selected_pattern_indices)
            show_only_nonzero_matrix_elements = False

        return self.plot(
            local_patterns,
            variable_indices=readout.variable_indices,
            reference_config=reference_config,
            labels=labels,
            suptitle=suptitle,
            local_operator=local_operator,
            show_only_nonzero_matrix_elements=show_only_nonzero_matrix_elements
            and local_operator is not None,
            matrix_element_tolerance=matrix_element_tolerance,
            show_matrix_element_values=show_matrix_element_values,
            matrix_element_value_role=matrix_element_value_role,
            max_matrix_element_values_per_pattern=max_matrix_element_values_per_pattern,
            matrix_element_value_precision=matrix_element_value_precision,
            **plot_kwargs,
        )

    def plot_structure_readout(
        self,
        structure_report: Any,
        *,
        reference_config: npt.ArrayLike | None = None,
        max_structures: int | None = None,
        max_basis_states: int | None = None,
        include_frozen: bool = True,
        max_frozen: int | None = None,
        nrows: int | None = None,
        ncols: int | None = None,
        mode: LinkPlotMode = "auto",
        coherent_plaquette_symbols: PlaquetteSymbolStyle = "auto",
        frozen_plaquette_symbols: PlaquetteSymbolStyle = "none",
        figsize: tuple[float, float] | None = None,
        show: bool = True,
        backend: VisualizerBackend = "matplotlib",
        suptitle: str | None = None,
        suptitle_y: float = 0.995,
        tight_layout_rect: tuple[float, float, float, float] | None = None,
        single_plot_kwargs: dict | None = None,
    ):
        """Visualize local entangled structures from one readout report.

        Each coherent pair is shown explicitly as a linear superposition of its
        basis patterns. Frozen/classical sectors are optionally shown afterward
        without plaquette symbols.
        """
        entries = _local_structure_entries_from_readout_report(
            structure_report,
            max_structures=max_structures,
            max_basis_states=max_basis_states,
            include_frozen=include_frozen,
            max_frozen=max_frozen,
            coherent_plaquette_symbols=coherent_plaquette_symbols,
            frozen_plaquette_symbols=frozen_plaquette_symbols,
        )
        if not entries:
            raise ValueError("No local structures are available to visualize.")

        if suptitle is None:
            readout = getattr(structure_report, "readout", None)
            component_index = None if readout is None else getattr(readout, "component_index", None)
            if component_index is None:
                suptitle = "Local entangled structures"
            else:
                suptitle = f"Local entangled structures, component {component_index}"

        return self._plot_local_structure_entries(
            entries,
            reference_config=reference_config,
            nrows=nrows,
            ncols=ncols,
            mode=mode,
            figsize=figsize,
            show=show,
            backend=backend,
            suptitle=suptitle,
            suptitle_y=suptitle_y,
            tight_layout_rect=tight_layout_rect,
            single_plot_kwargs=single_plot_kwargs,
        )

    def plot_structure_report(
        self,
        structure_report: Any,
        *,
        reference_config: npt.ArrayLike | None = None,
        max_readouts: int | None = None,
        max_structures_per_readout: int | None = None,
        max_basis_states: int | None = None,
        include_frozen: bool = True,
        max_frozen_per_readout: int | None = None,
        nrows: int | None = None,
        ncols: int | None = None,
        mode: LinkPlotMode = "auto",
        coherent_plaquette_symbols: PlaquetteSymbolStyle = "auto",
        frozen_plaquette_symbols: PlaquetteSymbolStyle = "none",
        figsize: tuple[float, float] | None = None,
        show: bool = True,
        backend: VisualizerBackend = "matplotlib",
        suptitle: str | None = None,
        suptitle_y: float = 0.995,
        tight_layout_rect: tuple[float, float, float, float] | None = None,
        single_plot_kwargs: dict | None = None,
    ):
        """Visualize entangled local structures from a cage-level structure report."""
        readout_reports = tuple(getattr(structure_report, "readout_reports", ()))
        if max_readouts is not None:
            readout_reports = readout_reports[: int(max_readouts)]

        entries: list[_LocalStructurePlotEntry] = []
        for report in readout_reports:
            entries.extend(
                _local_structure_entries_from_readout_report(
                    report,
                    max_structures=max_structures_per_readout,
                    max_basis_states=max_basis_states,
                    include_frozen=include_frozen,
                    max_frozen=max_frozen_per_readout,
                    coherent_plaquette_symbols=coherent_plaquette_symbols,
                    frozen_plaquette_symbols=frozen_plaquette_symbols,
                )
            )

        if not entries:
            raise ValueError("No local structures are available to visualize.")

        if suptitle is None:
            suptitle = "Local entangled structures"

        return self._plot_local_structure_entries(
            entries,
            reference_config=reference_config,
            nrows=nrows,
            ncols=ncols,
            mode=mode,
            figsize=figsize,
            show=show,
            backend=backend,
            suptitle=suptitle,
            suptitle_y=suptitle_y,
            tight_layout_rect=tight_layout_rect,
            single_plot_kwargs=single_plot_kwargs,
        )

    def _plot_local_structure_entries(
        self,
        entries: Sequence[_LocalStructurePlotEntry],
        *,
        reference_config: npt.ArrayLike | None = None,
        nrows: int | None = None,
        ncols: int | None = None,
        mode: LinkPlotMode = "auto",
        figsize: tuple[float, float] | None = None,
        show: bool = True,
        backend: VisualizerBackend = "matplotlib",
        suptitle: str | None = None,
        suptitle_y: float = 0.995,
        tight_layout_rect: tuple[float, float, float, float] | None = None,
        single_plot_kwargs: dict | None = None,
    ):
        reference = self._resolve_reference_config(reference_config)
        single_visualizer = self._single_visualizer()
        rows, cols = automatic_grid_shape(len(entries), nrows=nrows, ncols=ncols)

        if figsize is None:
            figsize = (3.2 * cols, 3.2 * rows)

        fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)

        if single_plot_kwargs is None:
            single_plot_kwargs = {}

        plot_kwargs = dict(single_plot_kwargs)
        with_site_labels = bool(plot_kwargs.pop("with_site_labels", True))
        with_site_values = bool(plot_kwargs.pop("with_site_values", False))
        with_link_values = bool(plot_kwargs.pop("with_link_values", False))
        with_link_ids = bool(plot_kwargs.pop("with_link_ids", False))
        with_plaquette_symbols_kw = plot_kwargs.pop("with_plaquette_symbols", True)
        plaquette_symbol_values = plot_kwargs.pop("plaquette_symbol_values", None)
        plot_kwargs.pop("title", None)
        plot_kwargs.pop("show", None)
        plot_kwargs.pop("backend", None)
        plot_kwargs.pop("ax", None)
        plot_kwargs.pop("mode", None)
        plot_kwargs.pop("style", None)
        plot_kwargs.pop("shadow_style", None)
        plot_kwargs.pop("periodic_image_mode", None)
        plot_kwargs.pop("collapse_duplicate_visual_links", None)
        plot_kwargs.pop("coordinate_scale", None)
        plot_kwargs.pop("coordinate_transform", None)
        plot_kwargs.pop("site_label_style", None)

        render_cache_map: dict[
            tuple[tuple[int, ...], PlaquetteSymbolStyle], _BasisGridRenderCache
        ] = {}
        mask_map: dict[
            tuple[tuple[int, ...], PlaquetteSymbolStyle],
            tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]],
        ] = {}

        for k, entry in enumerate(entries):
            ax = axes.flat[k]
            variable_indices = _normalize_local_variable_indices(entry.variable_indices)
            pattern_batch = _as_local_basis_patterns(
                entry.pattern, n_local_variables=len(variable_indices)
            )
            embedded_configs = _embed_local_patterns(
                reference_config=reference,
                local_patterns=pattern_batch,
                variable_indices=variable_indices,
            )
            config = embedded_configs[0]

            cache_key = (variable_indices, entry.plaquette_symbols)
            render_cache = render_cache_map.get(cache_key)
            if render_cache is None:
                render_cache = single_visualizer.build_grid_render_cache(
                    reference_config=config,
                    mode=mode,
                    plaquette_symbols=entry.plaquette_symbols,
                )
                render_cache_map[cache_key] = render_cache
                mask_map[cache_key] = self._active_artist_masks(
                    variable_indices=variable_indices,
                    render_cache=render_cache,
                )
            active_link_mask, active_node_mask = mask_map[cache_key]

            single_visualizer._plot_local_basis_with_grid_render_cache(
                config,
                ax=ax,
                render_cache=render_cache,
                active_link_mask=active_link_mask,
                active_node_mask=active_node_mask,
                shadow_style=self.shadow_style,
                show=False,
                backend=backend,
                with_site_labels=with_site_labels,
                with_site_values=with_site_values,
                with_link_values=with_link_values,
                with_link_ids=with_link_ids,
                with_plaquette_symbols=bool(with_plaquette_symbols_kw)
                and entry.plaquette_symbols != "none"
                and render_cache.plaquette_symbol_style != "none",
                plaquette_symbol_values=plaquette_symbol_values,
                title=entry.label,
                **plot_kwargs,
            )

            if entry.show_pattern_label:
                pattern_text = format_basis_config(entry.pattern, style="compact", max_length=64)
                ax.set_title(f"{entry.label}\n{pattern_text}")

        for k in range(len(entries), rows * cols):
            axes.flat[k].axis("off")

        if suptitle is not None:
            fig.suptitle(suptitle, y=suptitle_y)

        if tight_layout_rect is None:
            tight_layout_rect = (0.0, 0.0, 1.0, 0.96 if suptitle is not None else 1.0)

        fig.tight_layout(rect=tight_layout_rect)

        if show:
            plt.show()

        return fig, axes

    def _resolve_reference_config(
        self,
        reference_config: npt.ArrayLike | None,
    ) -> npt.NDArray[np.int64]:
        if reference_config is None:
            if self.layout is not None:
                return np.asarray(self.layout.default_config(), dtype=np.int64)
            return np.zeros(self.lattice.num_links, dtype=np.int64)

        reference = np.asarray(reference_config, dtype=np.int64)
        if reference.ndim != 1:
            raise ValueError("reference_config must be one-dimensional.")

        if self.layout is not None:
            self.layout.validate_config(reference)
        elif reference.size < self.lattice.num_links:
            raise ValueError(
                "Without a VariableLayout, reference_config must contain at least "
                f"{self.lattice.num_links} link values."
            )

        return reference

    def _active_artist_masks(
        self,
        *,
        variable_indices: tuple[int, ...],
        render_cache: _BasisGridRenderCache,
    ) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
        active_variables = set(variable_indices)
        active_link_mask = np.asarray(
            [int(index) in active_variables for index in render_cache.link_variable_indices],
            dtype=bool,
        )

        active_site_ids: set[int] = set()
        active_link_ids: set[int] = set()

        if self.layout is None:
            active_link_ids.update(int(index) for index in variable_indices)
        else:
            for variable_index in variable_indices:
                spec = self.layout.spec(int(variable_index))
                if spec.kind == VariableKind.SITE:
                    active_site_ids.add(int(spec.geometry_index))
                elif spec.kind == VariableKind.LINK:
                    active_link_ids.add(int(spec.geometry_index))

        for draw_link in render_cache.draw_links:
            if int(draw_link.link_id) not in active_link_ids:
                continue
            active_site_ids.add(int(draw_link.source_site))
            active_site_ids.add(int(draw_link.target_site))

        active_node_mask = np.asarray(
            [
                int(node.site_id) in active_site_ids
                or int(render_cache.site_variable_indices[index]) in active_variables
                for index, node in enumerate(render_cache.draw_nodes)
            ],
            dtype=bool,
        )

        return active_link_mask, active_node_mask


def _normalize_local_variable_indices(variable_indices: Sequence[int]) -> tuple[int, ...]:
    out = tuple(int(index) for index in variable_indices)

    if len(set(out)) != len(out):
        raise ValueError("variable_indices must not contain duplicates.")

    if any(index < 0 for index in out):
        raise ValueError("variable_indices must be non-negative.")

    return out


def _as_local_basis_patterns(
    local_patterns: npt.ArrayLike,
    *,
    n_local_variables: int,
) -> npt.NDArray[np.int64]:
    patterns = np.asarray(local_patterns, dtype=np.int64)

    if patterns.ndim == 1:
        if n_local_variables == 0:
            if patterns.size != 0:
                raise ValueError("Empty local supports require empty local patterns.")
            patterns = patterns.reshape(1, 0)
        elif n_local_variables == 1:
            patterns = patterns.reshape(-1, 1)
        elif patterns.size == n_local_variables:
            patterns = patterns.reshape(1, -1)
        else:
            raise ValueError(
                "local_patterns has incompatible shape for the supplied variable_indices."
            )

    if patterns.ndim != 2:
        raise ValueError("local_patterns must be one- or two-dimensional.")

    if patterns.shape[1] != n_local_variables:
        raise ValueError("local_patterns must have one column for each supplied variable index.")

    return patterns


def _select_local_patterns(
    local_patterns,
    selected_indices: npt.NDArray[np.int64],
) -> tuple[tuple[int, ...], ...]:
    pattern_tuple = tuple(tuple(int(value) for value in pattern) for pattern in local_patterns)
    return tuple(pattern_tuple[int(index)] for index in selected_indices)


def _format_matrix_element_value(
    value: complex,
    *,
    precision: int,
    tolerance: float,
) -> str:
    real = float(np.real(value))
    imag = float(np.imag(value))
    if abs(real) <= tolerance:
        real = 0.0
    if abs(imag) <= tolerance:
        imag = 0.0

    if imag == 0.0:
        return f"{real:.{precision}g}"
    if real == 0.0:
        return f"{imag:.{precision}g}i"

    sign = "+" if imag >= 0.0 else "-"
    return f"{real:.{precision}g}{sign}{abs(imag):.{precision}g}i"


def _matrix_element_value_labels_for_patterns(
    local_operator: npt.NDArray[np.complex128],
    *,
    displayed_pattern_indices: npt.NDArray[np.int64],
    tolerance: float,
    role: MatrixElementValueRole,
    max_terms_per_pattern: int,
    precision: int,
) -> list[str]:
    if role not in ("row", "column", "both"):
        raise ValueError("matrix_element_value_role must be 'row', 'column', or 'both'.")

    max_terms = max(int(max_terms_per_pattern), 0)
    labels: list[str] = []
    for pattern_index in displayed_pattern_indices:
        index = int(pattern_index)
        terms: list[str] = []

        if role in ("row", "both"):
            row_sources = np.flatnonzero(np.abs(local_operator[index, :]) > tolerance)
            for source_index in row_sources:
                value = local_operator[index, int(source_index)]
                value_text = _format_matrix_element_value(
                    value,
                    precision=precision,
                    tolerance=tolerance,
                )
                terms.append(f"{index}←{int(source_index)}:{value_text}")

        if role in ("column", "both"):
            column_targets = np.flatnonzero(np.abs(local_operator[:, index]) > tolerance)
            for target_index in column_targets:
                if role == "both" and int(target_index) == index:
                    # The diagonal element has already appeared in the row list.
                    continue
                value = local_operator[int(target_index), index]
                value_text = _format_matrix_element_value(
                    value,
                    precision=precision,
                    tolerance=tolerance,
                )
                terms.append(f"{int(target_index)}←{index}:{value_text}")

        if max_terms == 0 or not terms:
            labels.append("")
            continue

        clipped_terms = terms[:max_terms]
        if len(terms) > max_terms:
            clipped_terms.append(f"… {len(terms) - max_terms} more")
        labels.append("; ".join(clipped_terms))

    return labels


def _select_local_pattern_labels(
    labels: Sequence[str] | None,
    selected_indices: npt.NDArray[np.int64],
) -> list[str] | None:
    if labels is None:
        return [f"local {int(index)}" for index in selected_indices]

    if len(labels) < int(np.max(selected_indices, initial=-1)) + 1:
        raise ValueError("labels must have the same length as local_patterns.")

    return [labels[int(index)] for index in selected_indices]


def _nonzero_local_operator_pattern_indices(
    local_operator: npt.ArrayLike,
    *,
    n_patterns: int,
    tolerance: float,
) -> npt.NDArray[np.int64]:
    operator = np.asarray(local_operator, dtype=np.complex128)

    if operator.shape != (n_patterns, n_patterns):
        raise ValueError(
            "local_operator shape must match the number of local patterns: "
            f"{operator.shape} != {(n_patterns, n_patterns)}."
        )

    nonzero = np.abs(operator) > float(tolerance)
    active = np.any(nonzero, axis=0) | np.any(nonzero, axis=1)
    return np.flatnonzero(active).astype(np.int64, copy=False)


def _local_operator_from_readout(readout) -> npt.NDArray[np.complex128] | None:
    for attribute in ("density_matrix", "local_operator"):
        if hasattr(readout, attribute):
            value = getattr(readout, attribute)
            if value is not None:
                return np.asarray(value, dtype=np.complex128)

    reduced_density_matrix = getattr(readout, "reduced_density_matrix", None)
    if reduced_density_matrix is not None and hasattr(reduced_density_matrix, "density_matrix"):
        return np.asarray(reduced_density_matrix.density_matrix, dtype=np.complex128)

    return None


def _nonzero_matrix_unit_pattern_indices(
    *,
    matrix_unit_terms,
    local_patterns,
) -> npt.NDArray[np.int64]:
    pattern_to_index = {
        tuple(int(value) for value in pattern): index
        for index, pattern in enumerate(local_patterns)
    }
    selected: set[int] = set()

    for term in matrix_unit_terms:
        for attribute in ("target_pattern", "source_pattern"):
            pattern = tuple(int(value) for value in getattr(term, attribute))
            if pattern not in pattern_to_index:
                raise ValueError(
                    f"matrix-unit term contains pattern {pattern} not present in local_patterns."
                )
            selected.add(int(pattern_to_index[pattern]))

    return np.asarray(sorted(selected), dtype=np.int64)


def _embed_local_patterns(
    *,
    reference_config: npt.NDArray[np.int64],
    local_patterns: npt.NDArray[np.int64],
    variable_indices: tuple[int, ...],
) -> npt.NDArray[np.int64]:
    if any(index >= reference_config.size for index in variable_indices):
        raise ValueError("variable_indices are outside reference_config.")

    configs = np.repeat(reference_config.reshape(1, -1), local_patterns.shape[0], axis=0)

    if variable_indices:
        configs[:, list(variable_indices)] = local_patterns

    return configs


def plot_local_basis_grid(
    lattice: LatticeGraph,
    local_patterns: npt.ArrayLike,
    *,
    variable_indices: Sequence[int],
    reference_config: npt.ArrayLike | None = None,
    layout: VariableLayout | None = None,
    nrows: int | None = None,
    ncols: int | None = None,
    start_index: int = 0,
    labels: Sequence[str] | None = None,
    show_local_pattern_label: bool = True,
    config_label_style: BasisConfigLabelStyle = "compact",
    config_label_max_length: int = 48,
    backend: VisualizerBackend = "matplotlib",
    mode: LinkPlotMode = "auto",
    plaquette_symbols: PlaquetteSymbolStyle = "none",
    periodic_image_mode: PeriodicImageMode = "positive_patch",
    collapse_duplicate_visual_links: bool = True,
    coordinate_scale: float = 1.0,
    coordinate_transform: npt.ArrayLike | None = None,
    site_label_style: SiteLabelStyle = "sublattice_cell",
    theme: BasisVisualizerTheme = "research",
    style: LinkVisualStyle | None = None,
    shadow_style: LocalBasisShadowStyle | None = None,
    figsize: tuple[float, float] | None = None,
    show: bool = True,
    suptitle: str | None = None,
    single_plot_kwargs: dict | None = None,
    render_cache: _BasisGridRenderCache | None = None,
    local_operator: npt.ArrayLike | None = None,
    show_only_nonzero_matrix_elements: bool = False,
    matrix_element_tolerance: float = 1e-10,
    show_matrix_element_values: bool = False,
    matrix_element_value_role: MatrixElementValueRole = "both",
    max_matrix_element_values_per_pattern: int = 6,
    matrix_element_value_precision: int = 3,
):
    """Functional wrapper around :class:`LocalBasisGridVisualizer`."""

    visualizer = LocalBasisGridVisualizer(
        lattice=lattice,
        layout=layout,
        style=style,
        theme=theme,
        shadow_style=shadow_style if shadow_style is not None else LocalBasisShadowStyle(),
        periodic_image_mode=periodic_image_mode,
        collapse_duplicate_visual_links=collapse_duplicate_visual_links,
        coordinate_scale=coordinate_scale,
        coordinate_transform=coordinate_transform,
        site_label_style=site_label_style,
    )

    return visualizer.plot(
        local_patterns,
        variable_indices=variable_indices,
        reference_config=reference_config,
        nrows=nrows,
        ncols=ncols,
        start_index=start_index,
        labels=labels,
        show_local_pattern_label=show_local_pattern_label,
        config_label_style=config_label_style,
        config_label_max_length=config_label_max_length,
        mode=mode,
        plaquette_symbols=plaquette_symbols,
        figsize=figsize,
        show=show,
        backend=backend,
        suptitle=suptitle,
        single_plot_kwargs=single_plot_kwargs,
        render_cache=render_cache,
        local_operator=local_operator,
        show_only_nonzero_matrix_elements=show_only_nonzero_matrix_elements,
        matrix_element_tolerance=matrix_element_tolerance,
        show_matrix_element_values=show_matrix_element_values,
        matrix_element_value_role=matrix_element_value_role,
        max_matrix_element_values_per_pattern=max_matrix_element_values_per_pattern,
        matrix_element_value_precision=matrix_element_value_precision,
    )


def plot_local_structure_readout(
    lattice: LatticeGraph,
    structure_report: Any,
    *,
    reference_config: npt.ArrayLike | None = None,
    layout: VariableLayout | None = None,
    max_structures: int | None = None,
    max_basis_states: int | None = None,
    include_frozen: bool = True,
    max_frozen: int | None = None,
    nrows: int | None = None,
    ncols: int | None = None,
    backend: VisualizerBackend = "matplotlib",
    mode: LinkPlotMode = "auto",
    coherent_plaquette_symbols: PlaquetteSymbolStyle = "auto",
    frozen_plaquette_symbols: PlaquetteSymbolStyle = "none",
    periodic_image_mode: PeriodicImageMode = "positive_patch",
    collapse_duplicate_visual_links: bool = True,
    coordinate_scale: float = 1.0,
    coordinate_transform: npt.ArrayLike | None = None,
    site_label_style: SiteLabelStyle = "sublattice_cell",
    theme: BasisVisualizerTheme = "research",
    style: LinkVisualStyle | None = None,
    shadow_style: LocalBasisShadowStyle | None = None,
    figsize: tuple[float, float] | None = None,
    show: bool = True,
    suptitle: str | None = None,
    single_plot_kwargs: dict | None = None,
):
    """Functional wrapper around :meth:`LocalBasisGridVisualizer.plot_structure_readout`."""
    visualizer = LocalBasisGridVisualizer(
        lattice=lattice,
        layout=layout,
        style=style,
        theme=theme,
        shadow_style=shadow_style if shadow_style is not None else LocalBasisShadowStyle(),
        periodic_image_mode=periodic_image_mode,
        collapse_duplicate_visual_links=collapse_duplicate_visual_links,
        coordinate_scale=coordinate_scale,
        coordinate_transform=coordinate_transform,
        site_label_style=site_label_style,
    )
    return visualizer.plot_structure_readout(
        structure_report,
        reference_config=reference_config,
        max_structures=max_structures,
        max_basis_states=max_basis_states,
        include_frozen=include_frozen,
        max_frozen=max_frozen,
        nrows=nrows,
        ncols=ncols,
        mode=mode,
        coherent_plaquette_symbols=coherent_plaquette_symbols,
        frozen_plaquette_symbols=frozen_plaquette_symbols,
        figsize=figsize,
        show=show,
        backend=backend,
        suptitle=suptitle,
        single_plot_kwargs=single_plot_kwargs,
    )


def plot_local_structure_report(
    lattice: LatticeGraph,
    structure_report: Any,
    *,
    reference_config: npt.ArrayLike | None = None,
    layout: VariableLayout | None = None,
    max_readouts: int | None = None,
    max_structures_per_readout: int | None = None,
    max_basis_states: int | None = None,
    include_frozen: bool = True,
    max_frozen_per_readout: int | None = None,
    nrows: int | None = None,
    ncols: int | None = None,
    backend: VisualizerBackend = "matplotlib",
    mode: LinkPlotMode = "auto",
    coherent_plaquette_symbols: PlaquetteSymbolStyle = "auto",
    frozen_plaquette_symbols: PlaquetteSymbolStyle = "none",
    periodic_image_mode: PeriodicImageMode = "positive_patch",
    collapse_duplicate_visual_links: bool = True,
    coordinate_scale: float = 1.0,
    coordinate_transform: npt.ArrayLike | None = None,
    site_label_style: SiteLabelStyle = "sublattice_cell",
    theme: BasisVisualizerTheme = "research",
    style: LinkVisualStyle | None = None,
    shadow_style: LocalBasisShadowStyle | None = None,
    figsize: tuple[float, float] | None = None,
    show: bool = True,
    suptitle: str | None = None,
    single_plot_kwargs: dict | None = None,
):
    """Functional wrapper around :meth:`LocalBasisGridVisualizer.plot_structure_report`."""
    visualizer = LocalBasisGridVisualizer(
        lattice=lattice,
        layout=layout,
        style=style,
        theme=theme,
        shadow_style=shadow_style if shadow_style is not None else LocalBasisShadowStyle(),
        periodic_image_mode=periodic_image_mode,
        collapse_duplicate_visual_links=collapse_duplicate_visual_links,
        coordinate_scale=coordinate_scale,
        coordinate_transform=coordinate_transform,
        site_label_style=site_label_style,
    )
    return visualizer.plot_structure_report(
        structure_report,
        reference_config=reference_config,
        max_readouts=max_readouts,
        max_structures_per_readout=max_structures_per_readout,
        max_basis_states=max_basis_states,
        include_frozen=include_frozen,
        max_frozen_per_readout=max_frozen_per_readout,
        nrows=nrows,
        ncols=ncols,
        mode=mode,
        coherent_plaquette_symbols=coherent_plaquette_symbols,
        frozen_plaquette_symbols=frozen_plaquette_symbols,
        figsize=figsize,
        show=show,
        backend=backend,
        suptitle=suptitle,
        single_plot_kwargs=single_plot_kwargs,
    )
