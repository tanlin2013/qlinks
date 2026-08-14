from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

BasisConfigLabelStyle = Literal["none", "compact", "array"]
BasisVisualizerTheme = Literal["research", "paper"]
LinkPlotMode = Literal["auto", "arrows", "dimers", "values"]
PeriodicImageMode = Literal["none", "positive_patch"]
PlaquetteSymbolMode = Literal["binary", "flux"]
PlaquetteSymbolStyle = Literal["auto", "none", "circulation", "resonance"]
SiteLabelStyle = Literal["cell", "cell_sublattice", "sublattice_cell", "site_id"]
VisualizerBackend = Literal["matplotlib", "networkx"]
MatrixElementValueRole = Literal["row", "column", "both"]


@dataclass(frozen=True, slots=True)
class LinkVisualStyle:
    """
    Basic visual style for link drawing.
    """

    node_size: float = 180.0
    node_color: str = "tab:orange"
    node_face_color: str | None = None
    node_edge_color: str | None = None
    node_linewidth: float | None = None
    edge_color: str = "black"
    empty_edge_color: str = "lightgray"

    arrow_linewidth: float = 1.1
    arrow_alpha: float = 0.85
    arrow_mutation_scale: float | None = None
    arrow_shrink_points: float | None = None

    occupied_width: float = 2.0
    empty_width: float = 0.8
    occupied_alpha: float = 0.9
    empty_alpha: float = 0.5

    site_label_fontsize: float | None = None
    link_label_fontsize: float | None = None
    plaquette_symbol_fontsize: float = 22.0
    vulnerable_link_arrow_length_fraction: float = 1.1
    plaquette_symbol_offset: tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True, slots=True)
class _BasisVisualizerThemeDefaults:
    """Resolved presentation defaults for a basis-visualizer theme."""

    style: LinkVisualStyle
    with_site_labels: bool
    with_coordinate_labels: bool
    axes_padding: float
    panel_size: float
    title_fontsize: float | None
    coordinate_label_fontsize: float | None
    coordinate_axis_label_fontsize: float | None
    coordinate_label_color: str
    qdm_filled_flippable_color: str
    qdm_hollow_flippable_color: str
    qdm_vulnerable_color: str | None
    qdm_nonflippable_symbol: tuple[str, str] | None


def _basis_visualizer_theme_defaults(
    theme: BasisVisualizerTheme,
) -> _BasisVisualizerThemeDefaults:
    """Return presentation defaults for a named basis-visualizer theme."""
    if theme == "research":
        return _BasisVisualizerThemeDefaults(
            style=LinkVisualStyle(),
            with_site_labels=True,
            with_coordinate_labels=False,
            axes_padding=0.5,
            panel_size=3.0,
            title_fontsize=None,
            coordinate_label_fontsize=None,
            coordinate_axis_label_fontsize=None,
            coordinate_label_color="0.30",
            qdm_filled_flippable_color="blue",
            qdm_hollow_flippable_color="red",
            qdm_vulnerable_color=None,
            qdm_nonflippable_symbol=None,
        )

    if theme == "paper":
        return _BasisVisualizerThemeDefaults(
            style=LinkVisualStyle(
                node_size=22.0,
                node_color="black",
                node_face_color="white",
                node_edge_color="black",
                node_linewidth=0.9,
                edge_color="black",
                empty_edge_color="0.72",
                arrow_linewidth=0.9,
                arrow_alpha=1.0,
                arrow_mutation_scale=6.5,
                arrow_shrink_points=1.0,
                occupied_width=2.8,
                empty_width=0.65,
                occupied_alpha=1.0,
                empty_alpha=0.55,
                site_label_fontsize=6.0,
                link_label_fontsize=5.5,
                plaquette_symbol_fontsize=13.0,
                vulnerable_link_arrow_length_fraction=0.95,
            ),
            with_site_labels=False,
            with_coordinate_labels=False,
            axes_padding=0.18,
            panel_size=2.35,
            title_fontsize=8.0,
            coordinate_label_fontsize=8.5,
            coordinate_axis_label_fontsize=11.0,
            coordinate_label_color="0.25",
            qdm_filled_flippable_color="#0072B2",
            qdm_hollow_flippable_color="#D55E00",
            qdm_vulnerable_color="0.45",
            qdm_nonflippable_symbol=("×", "0.60"),
        )

    raise ValueError("theme must be 'research' or 'paper'.")


def basis_visual_style(theme: BasisVisualizerTheme = "research") -> LinkVisualStyle:
    """Return the default :class:`LinkVisualStyle` for a named basis theme.

    ``"research"`` reproduces the historical qlinks appearance. ``"paper"`` uses a compact
    publication style with hollow lattice sites and the paper QDM plaquette convention.
    The returned dataclass is immutable and can be customized with
    :func:`dataclasses.replace` when a figure needs a small local override.
    """
    return _basis_visualizer_theme_defaults(theme).style


@dataclass(frozen=True, slots=True)
class LocalBasisShadowStyle:
    """Visual style for variables outside a displayed local support.

    ``LocalBasisGridVisualizer`` embeds local basis patterns into a full lattice
    configuration.  Variables in the selected local support are drawn normally;
    all other site/link variables are drawn with this shadow style so the global
    lattice context remains visible without visually competing with the local
    state.
    """

    shadow_node_color: str = "lightgray"
    shadow_node_alpha: float = 0.18
    shadow_link_color: str = "lightgray"
    shadow_link_alpha: float = 0.22
    shadow_link_width_scale: float = 0.75
    label_shadowed_variables: bool = False
