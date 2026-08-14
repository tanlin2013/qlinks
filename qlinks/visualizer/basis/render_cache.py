from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from qlinks.visualizer.basis.styles import PlaquetteSymbolStyle


@dataclass(frozen=True, slots=True)
class _DrawNode:
    key: tuple[int, tuple[int, ...]]
    site_id: int
    image_shift: tuple[int, ...]
    position: tuple[float, float]


@dataclass(frozen=True, slots=True)
class _DrawLink:
    link_id: int
    source_key: tuple[int, tuple[int, ...]]
    target_key: tuple[int, tuple[int, ...]]
    source_site: int
    target_site: int
    source_position: tuple[float, float]
    target_position: tuple[float, float]


@dataclass(frozen=True, slots=True)
class _DrawPlaquette:
    plaquette_id: int
    image_shift: tuple[int, ...]
    visual_cell: tuple[int, ...]
    center: tuple[float, ...]
    link_ids: tuple[int, ...] = ()
    link_orientations: tuple[int, ...] = ()
    link_midpoints: tuple[tuple[float, float], ...] = ()


@dataclass(frozen=True, slots=True)
class _BasisGridRenderCache:
    """Reusable drawing cache for :class:`BasisGridVisualizer`.

    The cache stores geometry-only primitives plus resolved layout indices for
    one set of visualizer options.  It is useful when plotting many basis states
    or repeatedly plotting multiple batches with the same lattice/layout/style.

    Build instances with :meth:`BasisGridVisualizer.build_render_cache` rather
    than constructing them manually.  Treat all attributes as read-only
    implementation details.
    """

    mode: Literal["arrows", "dimers", "values"]
    plaquette_symbol_style: PlaquetteSymbolStyle
    draw_nodes: tuple[_DrawNode, ...]
    draw_links: tuple[_DrawLink, ...]
    draw_plaquettes: tuple[_DrawPlaquette, ...]
    link_variable_indices: npt.NDArray[np.int64]
    site_variable_indices: npt.NDArray[np.int64]
    node_xy: npt.NDArray[np.float64]
    link_source_xy: npt.NDArray[np.float64]
    link_target_xy: npt.NDArray[np.float64]
    link_segments: npt.NDArray[np.float64]
    link_midpoints: npt.NDArray[np.float64]
    site_labels: tuple[str, ...]
    plaquette_link_variable_indices: tuple[tuple[int, ...], ...]
    plaquette_orientations: tuple[tuple[int, ...], ...]
    plaquette_centers: npt.NDArray[np.float64]
    plaquette_midpoints: tuple[tuple[tuple[float, float], ...], ...]
    square_qlm_link_variable_indices: tuple[tuple[int, ...] | None, ...]
