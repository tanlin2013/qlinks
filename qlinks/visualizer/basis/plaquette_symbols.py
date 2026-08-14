from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import numpy.typing as npt
from matplotlib.patches import FancyArrowPatch

from qlinks.lattice import SquareLattice
from qlinks.visualizer.basis.render_cache import _DrawLink, _DrawPlaquette

_SQUARE_QLM_PLAQUETTE_SYMBOLS: dict[str, dict[str, str]] = {
    "1111": {"s": "◩", "color": "silver"},
    "1011": {"s": "↑", "color": "skyblue"},
    "0111": {"s": "→", "color": "salmon"},
    "0011": {"s": "♰", "color": "silver"},
    "1101": {"s": "↓", "color": "salmon"},
    "1001": {"s": "⬔", "color": "silver"},
    "0101": {"s": "↻", "color": "red"},
    "0001": {"s": "←", "color": "salmon"},
    "1110": {"s": "←", "color": "skyblue"},
    "1010": {"s": "↺", "color": "blue"},
    "0110": {"s": "⬕", "color": "silver"},
    "0010": {"s": "↓", "color": "skyblue"},
    "1100": {"s": "♱", "color": "silver"},
    "1000": {"s": "→", "color": "skyblue"},
    "0100": {"s": "↑", "color": "salmon"},
    "0000": {"s": "◪", "color": "silver"},
}


class _BasisConfigurationPlaquetteSymbolMixin:
    def _draw_square_qlm_plaquette_symbols(
        self,
        *,
        ax,
        config: npt.NDArray[np.int64],
        draw_plaquettes: list[_DrawPlaquette],
    ) -> None:
        """Draw the square-QLM-specific 16-symbol plaquette overlay."""
        if not isinstance(self.lattice, SquareLattice):
            return

        for draw_plaquette in draw_plaquettes:
            plaquette = self.lattice.plaquettes[draw_plaquette.plaquette_id]

            if len(plaquette.links) != 4:
                continue

            visual_cell = self._square_visual_cell_from_center(draw_plaquette.center)

            link_values = self._square_visual_qlm_symbol_link_values(
                config,
                tuple(int(value) for value in visual_cell),
            )
            key = self._plaquette_key(link_values)
            symbol_info = _SQUARE_QLM_PLAQUETTE_SYMBOLS.get(key)

            if symbol_info is None:
                continue

            center = draw_plaquette.center

            ax.text(
                center[0],
                center[1],
                symbol_info["s"],
                fontsize=self.style.plaquette_symbol_fontsize,
                color=symbol_info["color"],
                ha="center",
                va="center",
                zorder=6,
            )

    @staticmethod
    def _is_binary_link_pattern(values: Sequence[int]) -> bool:
        return set(int(value) for value in values) <= {0, 1}

    @staticmethod
    def _vulnerable_color_from_target_symbol(symbol_info: tuple[str, str]) -> str:
        """Return the arrow color for a one-link-away plaquette.

        Blue target symbols get skyblue arrows.
        Red target symbols get salmon arrows.
        """
        _symbol, color = symbol_info

        if color == "blue":
            return "skyblue"

        if color == "red":
            return "salmon"

        return color

    def _theme_qdm_resonance_symbol(
        self,
        values: Sequence[int],
    ) -> tuple[str, str] | None:
        """Return the QDM resonance marker using the active presentation theme."""
        symbol_info = self._qdm_resonance_symbol(values)

        if symbol_info is None:
            return None

        symbol, _color = symbol_info
        defaults = self._theme_defaults

        if symbol == "◆":
            return symbol, defaults.qdm_filled_flippable_color

        return symbol, defaults.qdm_hollow_flippable_color

    def _theme_qdm_vulnerable_color(self, inferred_color: str) -> str:
        """Resolve one-vulnerable-link color for the active presentation theme."""
        return self._theme_defaults.qdm_vulnerable_color or inferred_color

    def _draw_theme_qdm_nonflippable_symbol(
        self,
        *,
        ax,
        center: Sequence[float],
    ) -> None:
        """Draw the paper-theme nonflippable marker, if the theme requests one."""
        symbol_info = self._theme_defaults.qdm_nonflippable_symbol

        if symbol_info is None:
            return

        symbol, color = symbol_info
        ax.annotate(
            symbol,
            xy=(float(center[0]), float(center[1])),
            xytext=self.style.plaquette_symbol_offset,
            textcoords="offset points",
            fontsize=self.style.plaquette_symbol_fontsize,
            color=color,
            ha="center",
            va="center",
            zorder=6,
        )

    @staticmethod
    def _qdm_one_vulnerable_link(
        values: Sequence[int],
    ) -> tuple[int, str] | None:
        """Return the unique link whose flip makes a QDM plaquette resonant.

        Returns
        -------
        tuple[int, str] | None
            ``(vulnerable_link_index, arrow_color)`` if exactly one binary
            link flip turns the plaquette into a QDM resonance pattern.
        """
        values_tuple = tuple(int(value) for value in values)

        if len(values_tuple) < 4:
            return None

        if len(values_tuple) % 2 != 0:
            return None

        if not _BasisConfigurationPlaquetteSymbolMixin._is_binary_link_pattern(values_tuple):
            return None

        # Already resonant: draw the diamond, not the vulnerable-link arrow.
        if _BasisConfigurationPlaquetteSymbolMixin._qdm_resonance_symbol(values_tuple) is not None:
            return None

        candidates: list[tuple[int, str]] = []

        for index, value in enumerate(values_tuple):
            flipped = list(values_tuple)
            flipped[index] = 1 - int(value)

            symbol_info = _BasisConfigurationPlaquetteSymbolMixin._qdm_resonance_symbol(flipped)

            if symbol_info is None:
                continue

            candidates.append(
                (
                    index,
                    _BasisConfigurationPlaquetteSymbolMixin._vulnerable_color_from_target_symbol(
                        symbol_info
                    ),
                )
            )

        if len(candidates) != 1:
            return None

        return candidates[0]

    @staticmethod
    def _flux_one_vulnerable_link(
        values: Sequence[int],
        orientations: Sequence[int],
    ) -> tuple[int, str] | None:
        """Return the unique link whose sign flip makes a flux plaquette circulate.

        This is the QLM analogue of the one-vulnerable-link square symbols.
        """
        values_tuple = tuple(int(value) for value in values)
        orientations_tuple = tuple(int(orientation) for orientation in orientations)

        if len(values_tuple) != len(orientations_tuple):
            return None

        if len(values_tuple) < 4:
            return None

        # Already circulating: draw the circular arrow, not the vulnerable-link arrow.
        if (
            _BasisConfigurationPlaquetteSymbolMixin._flux_circulation_symbol(
                values_tuple,
                orientations_tuple,
            )
            is not None
        ):
            return None

        # Zero is not a signed flux direction.
        if any(value == 0 for value in values_tuple):
            return None

        candidates: list[tuple[int, str]] = []

        for index, value in enumerate(values_tuple):
            flipped = list(values_tuple)
            flipped[index] = -int(value)

            symbol_info = _BasisConfigurationPlaquetteSymbolMixin._flux_circulation_symbol(
                flipped,
                orientations_tuple,
            )

            if symbol_info is None:
                continue

            candidates.append(
                (
                    index,
                    _BasisConfigurationPlaquetteSymbolMixin._vulnerable_color_from_target_symbol(
                        symbol_info
                    ),
                )
            )

        if len(candidates) != 1:
            return None

        return candidates[0]

    def _draw_vulnerable_link_arrow(
        self,
        *,
        ax,
        center: Sequence[float],
        link_midpoint: Sequence[float],
        color: str,
    ) -> None:
        """Draw an arrow centered at the plaquette center toward a vulnerable link."""
        center_array = np.asarray(center, dtype=float)
        midpoint_array = np.asarray(link_midpoint, dtype=float)

        direction = midpoint_array - center_array
        distance = float(np.linalg.norm(direction))

        if distance <= 1e-12:
            return

        # A value < 1 keeps the arrow inside the plaquette and avoids placing the
        # arrow head directly on top of the link/dimer/flux arrow.
        arrow_length_fraction = self.style.vulnerable_link_arrow_length_fraction
        arrow_vector = arrow_length_fraction * direction

        start = center_array - 0.5 * arrow_vector
        end = center_array + 0.5 * arrow_vector

        fontsize = float(self.style.plaquette_symbol_fontsize)
        mutation_scale = fontsize
        linewidth = max(1.0, 0.12 * fontsize)

        arrow = FancyArrowPatch(
            posA=(float(start[0]), float(start[1])),
            posB=(float(end[0]), float(end[1])),
            arrowstyle="->",
            mutation_scale=mutation_scale,
            linewidth=linewidth,
            color=color,
            zorder=7,
        )
        ax.add_patch(arrow)

    @staticmethod
    def _qdm_resonance_symbol(values: Sequence[int]) -> tuple[str, str] | None:
        """Return a QDM resonance marker for alternating binary dimers.

        The input values must already be in canonical visual cyclic order.

        Pattern 1010... -> blue ◆
        Pattern 0101... -> red ◇
        """
        values_tuple = tuple(int(value) for value in values)

        if len(values_tuple) < 4:
            return None

        if len(values_tuple) % 2 != 0:
            return None

        if not _BasisConfigurationPlaquetteSymbolMixin._is_binary_link_pattern(values_tuple):
            return None

        pattern_a = tuple(1 if i % 2 == 0 else 0 for i in range(len(values_tuple)))
        pattern_b = tuple(0 if i % 2 == 0 else 1 for i in range(len(values_tuple)))

        if values_tuple == pattern_a:
            return "◆", "blue"

        if values_tuple == pattern_b:
            return "◇", "red"

        return None

    @staticmethod
    def _flux_circulation_symbol(
        values: Sequence[int],
        orientations: Sequence[int],
    ) -> tuple[str, str] | None:
        """Return QLM-like flux circulation symbol.

        This is for signed flux values, not binary QDM dimers.
        """
        if len(values) != len(orientations):
            return None

        oriented_values = [
            int(value) * int(orientation)
            for value, orientation in zip(values, orientations, strict=True)
        ]

        # Zero should not count as negative circulation.
        if any(value == 0 for value in oriented_values):
            return None

        if all(value > 0 for value in oriented_values):
            return "↺", "blue"

        if all(value < 0 for value in oriented_values):
            return "↻", "red"

        return None

    def _draw_resonance_plaquette_symbols(
        self,
        *,
        ax,
        config: npt.NDArray[np.int64],
        draw_plaquettes: list[_DrawPlaquette],
        plaquette_symbol_values: Mapping[int, tuple[str, str]] | None = None,
    ) -> None:
        for draw_plaquette in draw_plaquettes:
            plaquette_id = int(draw_plaquette.plaquette_id)

            if plaquette_symbol_values is not None:
                symbol_info = plaquette_symbol_values.get(plaquette_id)

                if symbol_info is None:
                    continue

                symbol, color = symbol_info
                center = draw_plaquette.center

                ax.annotate(
                    symbol,
                    xy=(center[0], center[1]),
                    xytext=self.style.plaquette_symbol_offset,
                    textcoords="offset points",
                    fontsize=self.style.plaquette_symbol_fontsize,
                    color=color,
                    ha="center",
                    va="center",
                    zorder=6,
                )
                continue

            # existing visualizer-inferred fallback
            link_ids = tuple(int(link_id) for link_id in draw_plaquette.link_ids)

            if len(link_ids) == 0:
                plaquette = self.lattice.plaquettes[draw_plaquette.plaquette_id]
                link_ids = tuple(int(link_id) for link_id in plaquette.links)

            values = [self.link_value(config, int(link_id)) for link_id in link_ids]

            symbol_info = self._theme_qdm_resonance_symbol(values)

            if symbol_info is not None:
                symbol, color = symbol_info
                center = draw_plaquette.center

                ax.annotate(
                    symbol,
                    xy=(center[0], center[1]),
                    xytext=self.style.plaquette_symbol_offset,
                    textcoords="offset points",
                    fontsize=self.style.plaquette_symbol_fontsize,
                    color=color,
                    ha="center",
                    va="center",
                    zorder=6,
                )
                continue

            vulnerable_info = self._qdm_one_vulnerable_link(values)

            if vulnerable_info is None:
                self._draw_theme_qdm_nonflippable_symbol(
                    ax=ax,
                    center=draw_plaquette.center,
                )
                continue

            vulnerable_index, color = vulnerable_info
            color = self._theme_qdm_vulnerable_color(color)

            if vulnerable_index >= len(draw_plaquette.link_midpoints):
                continue

            self._draw_vulnerable_link_arrow(
                ax=ax,
                center=draw_plaquette.center,
                link_midpoint=draw_plaquette.link_midpoints[vulnerable_index],
                color=color,
            )

    def _draw_circulation_plaquette_symbols(
        self,
        *,
        ax,
        config: npt.NDArray[np.int64],
        draw_plaquettes: list[_DrawPlaquette],
    ) -> None:
        text_items: list[tuple[int, Sequence[float], str, str]] = []

        for draw_plaquette in draw_plaquettes:
            if isinstance(self.lattice, SquareLattice) and len(draw_plaquette.link_ids) == 4:
                symbol_info = self._square_qlm_symbol_info(
                    config=config,
                    draw_plaquette=draw_plaquette,
                )

                if symbol_info is None:
                    continue

                symbol, color = symbol_info
                text_items.append(
                    (
                        int(draw_plaquette.plaquette_id),
                        draw_plaquette.center,
                        symbol,
                        color,
                    )
                )
                continue

            values = tuple(
                self.link_value(config, int(link_id)) for link_id in draw_plaquette.link_ids
            )
            orientations = tuple(int(x) for x in draw_plaquette.link_orientations)

            symbol_info = self._flux_circulation_symbol(values, orientations)

            if symbol_info is not None:
                symbol, color = symbol_info
                text_items.append(
                    (
                        int(draw_plaquette.plaquette_id),
                        draw_plaquette.center,
                        symbol,
                        color,
                    )
                )
                continue

            vulnerable_info = self._flux_one_vulnerable_link(values, orientations)

            if vulnerable_info is None:
                continue

            vulnerable_index, color = vulnerable_info

            if vulnerable_index >= len(draw_plaquette.link_midpoints):
                continue

            self._draw_vulnerable_link_arrow(
                ax=ax,
                center=draw_plaquette.center,
                link_midpoint=draw_plaquette.link_midpoints[vulnerable_index],
                color=color,
            )

        for _plaquette_id, center, symbol, color in text_items:
            ax.annotate(
                symbol,
                xy=(center[0], center[1]),
                xytext=self.style.plaquette_symbol_offset,
                textcoords="offset points",
                fontsize=self.style.plaquette_symbol_fontsize,
                color=color,
                ha="center",
                va="center",
                zorder=6,
            )

    def _square_qlm_symbol_info(
        self,
        *,
        config: npt.NDArray[np.int64],
        draw_plaquette: _DrawPlaquette,
    ) -> tuple[str, str] | None:
        """Return the legacy square QLM glyph for a square plaquette.

        The legacy _SQUARE_QLM_PLAQUETTE_SYMBOLS table uses the visual key
        convention

            bottom, left, right, top

        not the generic square primitive order

            bottom, right, top, left.

        Therefore we must adapt the current visual plaquette cell back to the
        legacy key convention before looking up the table.
        """
        if not isinstance(self.lattice, SquareLattice):
            return None

        if len(draw_plaquette.link_ids) != 4:
            return None

        if len(draw_plaquette.visual_cell) >= 2 and all(
            int(value) >= 0 for value in draw_plaquette.visual_cell[:2]
        ):
            visual_cell = (
                int(draw_plaquette.visual_cell[0]),
                int(draw_plaquette.visual_cell[1]),
            )
        else:
            visual_cell = self._square_visual_cell_from_center(
                draw_plaquette.center,
            )

        values = self._square_visual_qlm_symbol_link_values(
            config,
            visual_cell,
        )

        key = self._plaquette_key(values)

        payload = _SQUARE_QLM_PLAQUETTE_SYMBOLS.get(key)

        if payload is None:
            return None

        return payload["s"], payload["color"]

    def _plaquette_center_2d(
        self,
        site_ids: Sequence[int],
    ) -> tuple[float, float]:
        positions = [
            self._xy(tuple(self.lattice.site_positions[int(site_id)])) for site_id in site_ids
        ]
        center = np.mean(np.asarray(positions, dtype=float), axis=0)
        return float(center[0]), float(center[1])

    @staticmethod
    def _points_along_link(value: int) -> bool:
        """
        Link-arrow convention.

        Positive flux or binary 1 points along stored link orientation.
        Negative flux or binary 0 points opposite.
        """

        return value > 0

    def _square_visual_cell_from_center(
        self,
        center: npt.ArrayLike,
    ) -> tuple[int, int]:
        """Infer square-lattice visual cell from a drawn plaquette center.

        In the positive-patch drawing, the visual plaquette at cell (x, y) is
        centered at approximately (x + 1/2, y + 1/2), up to coordinate transforms.
        """
        center_array = np.asarray(center, dtype=float)

        # If coordinate transforms/scales are applied before storing draw centers,
        # this helper assumes draw centers are already in plotting coordinates.
        # For the default square plotting, this is correct.
        cell_x = int(round(float(center_array[0]) - 0.5))
        cell_y = int(round(float(center_array[1]) - 0.5))

        return cell_x, cell_y

    def _square_visual_link_id(
        self,
        *,
        cell: tuple[int, int],
        kind: str,
    ) -> int:
        """Return the square-lattice link id at a visual cell and kind."""
        if not isinstance(self.lattice, SquareLattice):
            raise TypeError("Expected SquareLattice.")

        cell_x = int(cell[0])
        cell_y = int(cell[1])

        lattice_x = cell_x % int(self.lattice.lx)
        lattice_y = cell_y % int(self.lattice.ly)

        for link in self.lattice.links:
            source_site = self.lattice.sites[int(link.source)]

            if tuple(source_site.cell) == (lattice_x, lattice_y) and link.kind == kind:
                return int(link.id)

        raise KeyError(f"No {kind}-link found at cell {(lattice_x, lattice_y)}.")

    def _square_visual_qlm_symbol_link_values(
        self,
        config: npt.ArrayLike,
        visual_cell: tuple[int, int],
    ) -> list[int]:
        """Return square-QLM symbol values from the drawn visual plaquette.

        Key convention:
            bottom, left, right, top

        These values follow the visible positive-patch arrows, not the abstract
        periodic plaquette object's stored boundary. This matters for small PBC
        lattices such as 2x2.
        """
        cell_x = int(visual_cell[0])
        cell_y = int(visual_cell[1])

        bottom_link = self._square_visual_link_id(
            cell=(cell_x, cell_y),
            kind="x",
        )
        left_link = self._square_visual_link_id(
            cell=(cell_x, cell_y),
            kind="y",
        )
        right_link = self._square_visual_link_id(
            cell=(cell_x + 1, cell_y),
            kind="y",
        )
        top_link = self._square_visual_link_id(
            cell=(cell_x, cell_y + 1),
            kind="x",
        )

        return [
            self.link_value(config, bottom_link),
            self.link_value(config, left_link),
            self.link_value(config, right_link),
            self.link_value(config, top_link),
        ]

    @staticmethod
    def _cyclic_order_score(
        candidate_link_ids: Sequence[int],
        physical_link_ids: Sequence[int],
    ) -> tuple[int, int]:
        """Score how well a visual cyclic order matches a physical link order.

        Lower is better.

        Returns
        -------
        tuple[int, int]
            (mismatch_count, reversed_flag)
        """
        candidate = tuple(int(x) for x in candidate_link_ids)
        physical = tuple(int(x) for x in physical_link_ids)

        if len(candidate) != len(physical):
            return (10**9, 1)

        n = len(physical)
        best = (10**9, 1)

        for reversed_flag, order in enumerate((physical, tuple(reversed(physical)))):
            for shift in range(n):
                rotated = order[shift:] + order[:shift]
                mismatches = sum(int(a != b) for a, b in zip(candidate, rotated, strict=True))
                best = min(best, (mismatches, reversed_flag))

        return best

    def _visual_plaquette_representative_score_for_physical_links(
        self,
        draw_links: tuple[_DrawLink, ...],
        *,
        physical_link_ids: tuple[int, ...],
        preferred_center: npt.NDArray[np.float64] | None = None,
    ) -> tuple[int, int, float, float, float, float]:
        """Score a closed visual representative for a physical plaquette.

        Lower is preferred:
            1. visual cyclic link order matches physical plaquette link order;
            2. optional closeness to a preferred center;
            3. lower visual center;
            4. left visual center;
            5. compactness.
        """
        center = self._closed_visual_plaquette_center(draw_links)
        canonical_draw_links = self._canonical_visual_cycle_draw_links(draw_links)
        candidate_link_ids = tuple(int(draw_link.link_id) for draw_link in canonical_draw_links)

        order_score = self._cyclic_order_score(
            candidate_link_ids,
            physical_link_ids,
        )

        if preferred_center is None:
            center_distance = 0.0
        else:
            center_distance = float(
                np.linalg.norm(
                    np.asarray(center, dtype=float) - np.asarray(preferred_center, dtype=float)
                )
            )

        compactness = self._visual_plaquette_compactness_score(draw_links)

        return (
            int(order_score[0]),
            int(order_score[1]),
            center_distance,
            float(center[1]),
            float(center[0]),
            float(compactness),
        )

    @staticmethod
    def _plaquette_key(values: list[int]) -> str:
        bits = [1 if value > 0 else 0 for value in values]
        return "".join(str(bit) for bit in bits)
