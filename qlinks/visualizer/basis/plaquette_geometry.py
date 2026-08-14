from __future__ import annotations

import math
from itertools import product
from typing import Mapping, Sequence

import numpy as np
import numpy.typing as npt

from qlinks.lattice import (
    BoundaryCondition,
    HoneycombLattice,
    KagomeLattice,
    SquareLattice,
    TriangularLattice,
)
from qlinks.visualizer.basis.render_cache import _DrawLink, _DrawPlaquette
from qlinks.visualizer.basis.styles import PlaquetteSymbolStyle


class _BasisConfigurationPlaquetteGeometryMixin:
    def _draw_square_generic_plaquette_primitives(self) -> list[_DrawPlaquette]:
        """Build square plaquette primitives for generic resonance/circulation.

        Unlike the old generic fallback, this is cell based. On a square PBC
        positive patch, each visual cell gets its own plaquette center and local
        boundary links. This prevents distinct plaquettes on a small torus from
        collapsing to the same visual center.
        """
        if not isinstance(self.lattice, SquareLattice):
            return []

        if self.lattice.ndim != 2:
            return []

        spans = self._cell_spans()
        lx = int(spans[0])
        ly = int(spans[1])

        period_vectors = self._period_vectors_2d()
        unit_vectors = np.zeros_like(period_vectors)
        unit_vectors[0] = period_vectors[0] / float(lx)
        unit_vectors[1] = period_vectors[1] / float(ly)

        plaquette_by_cell = self._square_plaquette_by_cell_fallback()

        draw_plaquettes: list[_DrawPlaquette] = []

        if self.lattice.boundary_condition != BoundaryCondition.PERIODIC:
            for plaquette in self.lattice.plaquettes:
                if len(plaquette.links) != 4:
                    continue

                center = self._plaquette_center_2d(plaquette.sites)

                draw_plaquettes.append(
                    _DrawPlaquette(
                        plaquette_id=int(plaquette.id),
                        image_shift=tuple(0 for _ in range(self.lattice.ndim)),
                        visual_cell=tuple(-1 for _ in range(self.lattice.ndim)),
                        center=center,
                        link_ids=tuple(int(link_id) for link_id in plaquette.links),
                        link_orientations=tuple(
                            int(orientation) for orientation in plaquette.orientations
                        ),
                        link_midpoints=self._square_generic_link_midpoints_from_center(
                            center=center,
                            unit_vectors=unit_vectors,
                        ),
                    )
                )

            return self._collapse_duplicate_draw_plaquettes(draw_plaquettes)

        for x in range(lx):
            for y in range(ly):
                visual_cell = (x, y)
                base_cell = (x % lx, y % ly)

                plaquette_id = plaquette_by_cell.get(base_cell)

                if plaquette_id is None:
                    flat_index = x * ly + y
                    if flat_index < self.lattice.num_plaquettes:
                        plaquette_id = int(self.lattice.plaquettes[flat_index].id)

                if plaquette_id is None:
                    continue

                lower_left_site_id = self._site_id_from_cell(base_cell)
                if lower_left_site_id is None:
                    continue

                image_shift = self._image_shift_for_visual_cell(
                    site_id=lower_left_site_id,
                    visual_cell=visual_cell,
                )
                if image_shift is None:
                    continue

                lower_left_position = np.asarray(
                    self._visual_site_position(
                        site_id=lower_left_site_id,
                        image_shift=image_shift,
                        period_vectors=period_vectors,
                    ),
                    dtype=float,
                )

                center_arr = lower_left_position + 0.5 * unit_vectors[0] + 0.5 * unit_vectors[1]
                center = (float(center_arr[0]), float(center_arr[1]))

                bottom_link = self._square_visual_link_id(
                    cell=visual_cell,
                    kind="x",
                )
                right_link = self._square_visual_link_id(
                    cell=(visual_cell[0] + 1, visual_cell[1]),
                    kind="y",
                )
                top_link = self._square_visual_link_id(
                    cell=(visual_cell[0], visual_cell[1] + 1),
                    kind="x",
                )
                left_link = self._square_visual_link_id(
                    cell=visual_cell,
                    kind="y",
                )

                draw_plaquettes.append(
                    _DrawPlaquette(
                        plaquette_id=int(plaquette_id),
                        image_shift=image_shift,
                        visual_cell=visual_cell,
                        center=center,
                        link_ids=(
                            int(bottom_link),
                            int(right_link),
                            int(top_link),
                            int(left_link),
                        ),
                        link_orientations=(1, 1, -1, -1),
                        link_midpoints=self._square_generic_link_midpoints_from_center(
                            center=center,
                            unit_vectors=unit_vectors,
                        ),
                    )
                )

        return self._collapse_duplicate_draw_plaquettes(draw_plaquettes)

    @staticmethod
    def _square_generic_link_midpoints_from_center(
        *,
        center: tuple[float, float],
        unit_vectors: npt.NDArray[np.float64],
    ) -> tuple[tuple[float, float], ...]:
        """Return bottom/right/top/left local edge midpoints for a square cell."""
        center_arr = np.asarray(center, dtype=float)

        bottom = center_arr - 0.5 * unit_vectors[1]
        right = center_arr + 0.5 * unit_vectors[0]
        top = center_arr + 0.5 * unit_vectors[1]
        left = center_arr - 0.5 * unit_vectors[0]

        return (
            (float(bottom[0]), float(bottom[1])),
            (float(right[0]), float(right[1])),
            (float(top[0]), float(top[1])),
            (float(left[0]), float(left[1])),
        )

    def _site_id_from_cell(
        self,
        cell: tuple[int, ...],
        *,
        sublattice: int = 0,
    ) -> int | None:
        for site in self.lattice.sites:
            if tuple(int(c) for c in site.cell) == tuple(int(c) for c in cell):
                if int(site.sublattice) == int(sublattice):
                    return int(site.id)

        return None

    def _square_plaquette_by_cell_fallback(self) -> dict[tuple[int, int], int]:
        """
        Map square plaquettes to base cells.

        This tries several conventions, because different square-lattice builders
        may store plaquette metadata differently.

        Priority:
            1. plaquette.cell or plaquette.anchor_cell if available
            2. lower-left cell inferred from plaquette sites
            3. row-major plaquette ordering fallback
        """
        if not isinstance(self.lattice, SquareLattice):
            return {}

        spans = self._cell_spans()
        lx = int(spans[0])
        ly = int(spans[1])

        out: dict[tuple[int, int], int] = {}

        # 1. Use explicit plaquette metadata if present.
        for plaquette in self.lattice.plaquettes:
            cell = None

            if hasattr(plaquette, "cell"):
                cell = plaquette.cell

            elif hasattr(plaquette, "anchor_cell"):
                cell = plaquette.anchor_cell

            if cell is None:
                continue

            c = tuple(int(x) for x in cell)
            if len(c) < 2:
                continue

            out[(c[0] % lx, c[1] % ly)] = int(plaquette.id)

        if out:
            return out

        # 2. Try to infer from plaquette sites.
        #
        # For non-wrapping plaquettes this is simply min x, min y.
        # For wrapping plaquettes this may be ambiguous, so this is only a best effort.
        for plaquette in self.lattice.plaquettes:
            if len(plaquette.sites) == 0:
                continue

            cells = np.asarray(
                [self.lattice.sites[int(site_id)].cell for site_id in plaquette.sites],
                dtype=np.int64,
            )

            if cells.shape[1] != 2:
                continue

            xs = cells[:, 0] % lx
            ys = cells[:, 1] % ly

            # If the plaquette spans the PBC seam, the lower-left cell is the
            # largest coordinate before wrapping, not min. Detect this by spread.
            if xs.max() - xs.min() > lx / 2:
                x0 = int(xs.max())
            else:
                x0 = int(xs.min())

            if ys.max() - ys.min() > ly / 2:
                y0 = int(ys.max())
            else:
                y0 = int(ys.min())

            out[(x0 % lx, y0 % ly)] = int(plaquette.id)

        if out:
            return out

        # 3. Last-resort row-major fallback.
        #
        # This assumes plaquette id/order follows:
        #   (0,0), (0,1), ..., (0,ly-1), (1,0), ...
        for x in range(lx):
            for y in range(ly):
                flat_index = x * ly + y
                if flat_index < self.lattice.num_plaquettes:
                    out[(x, y)] = int(self.lattice.plaquettes[flat_index].id)

        return out

    @staticmethod
    def _draw_link_midpoint(draw_link: _DrawLink) -> tuple[float, float]:
        source = np.asarray(draw_link.source_position, dtype=float)
        target = np.asarray(draw_link.target_position, dtype=float)
        midpoint = 0.5 * (source + target)
        return float(midpoint[0]), float(midpoint[1])

    def _canonical_visual_cycle_link_ids(
        self,
        draw_links: tuple[_DrawLink, ...],
    ) -> tuple[int, ...]:
        """Return link ids in canonical visual cyclic order."""
        canonical_links = self._canonical_visual_cycle_draw_links(draw_links)
        return tuple(int(draw_link.link_id) for draw_link in canonical_links)

    def _canonical_visual_cycle_orientations(
        self,
        *,
        plaquette_id: int,
        canonical_link_ids: tuple[int, ...],
    ) -> tuple[int, ...]:
        """Return plaquette orientations reordered to canonical visual link order."""
        plaquette = self.lattice.plaquettes[plaquette_id]

        orientation_by_link_id = {
            int(link_id): int(orientation)
            for link_id, orientation in zip(
                plaquette.links,
                plaquette.orientations,
                strict=True,
            )
        }

        return tuple(int(orientation_by_link_id[int(link_id)]) for link_id in canonical_link_ids)

    def _canonical_visual_cycle_draw_links(
        self,
        draw_links: tuple[_DrawLink, ...],
    ) -> tuple[_DrawLink, ...]:
        """Return draw links in canonical visual cyclic order.

        Convention:
        1. sort edge midpoints counterclockwise around the visual center;
        2. rotate so the first edge has the lowest midpoint y, then lowest x.
        """
        center = self._closed_visual_plaquette_center(draw_links)

        records: list[tuple[float, float, float, _DrawLink]] = []

        for draw_link in draw_links:
            source = np.asarray(draw_link.source_position, dtype=float)
            target = np.asarray(draw_link.target_position, dtype=float)
            midpoint = 0.5 * (source + target)

            angle = math.atan2(
                float(midpoint[1] - center[1]),
                float(midpoint[0] - center[0]),
            )

            records.append(
                (
                    angle,
                    float(midpoint[1]),
                    float(midpoint[0]),
                    draw_link,
                )
            )

        records.sort(key=lambda item: item[0])

        start = min(
            range(len(records)),
            key=lambda i: (records[i][1], records[i][2]),
        )

        rotated = records[start:] + records[:start]

        return tuple(record[3] for record in rotated)

    def _draw_plaquette_primitives(self) -> list[_DrawPlaquette]:
        """Build visual plaquette primitives for generic plaquette symbols.

        This method is intentionally style-independent. The same primitives are
        used by QLM circulation symbols, QDM resonance symbols, and one-vulnerable
        link arrows.
        """
        if self.lattice.num_plaquettes == 0:
            return []

        if isinstance(self.lattice, SquareLattice):
            return self._draw_square_generic_plaquette_primitives()

        return self._draw_generic_non_square_plaquette_primitives()

    def _draw_generic_non_square_plaquette_primitives(self) -> list[_DrawPlaquette]:
        """Build generic non-square plaquette primitives."""
        _draw_nodes, draw_links = self._draw_primitives()

        draw_links_by_link_id: dict[int, list[_DrawLink]] = {}
        for draw_link in draw_links:
            draw_links_by_link_id.setdefault(int(draw_link.link_id), []).append(draw_link)

        draw_plaquettes: list[_DrawPlaquette] = []

        for plaquette in self.lattice.plaquettes:
            link_ids = tuple(int(link_id) for link_id in plaquette.links)

            if not self._is_supported_circulation_plaquette(link_ids):
                continue

            candidate_lists = [draw_links_by_link_id.get(link_id, []) for link_id in link_ids]

            if any(len(candidates) == 0 for candidates in candidate_lists):
                continue

            selected = self._select_closed_visual_plaquette(
                candidate_lists,
                physical_link_ids=link_ids,
                preferred_center=None,
            )

            if selected is None:
                continue

            center = self._closed_visual_plaquette_center(selected)
            canonical_draw_links = self._canonical_visual_cycle_draw_links(selected)

            canonical_link_ids = tuple(int(draw_link.link_id) for draw_link in canonical_draw_links)
            canonical_orientations = self._canonical_visual_cycle_orientations_from_draw_links(
                center=center,
                canonical_draw_links=canonical_draw_links,
            )
            canonical_midpoints = tuple(
                self._draw_link_midpoint(draw_link) for draw_link in canonical_draw_links
            )

            draw_plaquettes.append(
                _DrawPlaquette(
                    plaquette_id=int(plaquette.id),
                    image_shift=tuple(0 for _ in range(self.lattice.ndim)),
                    visual_cell=tuple(-1 for _ in range(self.lattice.ndim)),
                    center=(float(center[0]), float(center[1])),
                    link_ids=canonical_link_ids,
                    link_orientations=canonical_orientations,
                    link_midpoints=canonical_midpoints,
                )
            )

        return self._collapse_duplicate_draw_plaquettes(draw_plaquettes)

    def _canonical_visual_cycle_orientations_from_draw_links(
        self,
        *,
        center: Sequence[float],
        canonical_draw_links: tuple[_DrawLink, ...],
    ) -> tuple[int, ...]:
        """Return orientations of drawn links relative to the local visual cycle.

        +1 means the stored draw-link direction agrees with the local cyclic
        boundary direction. -1 means it opposes it.
        """
        center_array = np.asarray(center, dtype=float)

        orientations: list[int] = []

        for draw_link in canonical_draw_links:
            source = np.asarray(draw_link.source_position, dtype=float)
            target = np.asarray(draw_link.target_position, dtype=float)
            midpoint = 0.5 * (source + target)

            radial = midpoint - center_array
            tangent_ccw = np.asarray([-radial[1], radial[0]], dtype=float)

            link_vector = target - source

            orientation = 1 if float(np.dot(link_vector, tangent_ccw)) >= 0.0 else -1
            orientations.append(orientation)

        return tuple(orientations)

    def _is_supported_circulation_plaquette(
        self,
        link_ids: tuple[int, ...],
    ) -> bool:
        """Return whether a plaquette should receive a circulation symbol."""
        n_links = len(link_ids)

        if isinstance(self.lattice, SquareLattice):
            return n_links == 4

        if isinstance(self.lattice, TriangularLattice):
            # For triangular-lattice QDM/QLM resonance, the relevant plaquette is
            # a rhombus, not an elementary triangle.
            return n_links == 4

        if isinstance(self.lattice, (HoneycombLattice, KagomeLattice)):
            return n_links == 6

        # Conservative generic fallback.
        return n_links >= 4

    def _select_closed_visual_plaquette(
        self,
        candidate_lists: list[list[_DrawLink]],
        *,
        physical_link_ids: tuple[int, ...],
        preferred_center: npt.NDArray[np.float64] | None = None,
    ) -> tuple[_DrawLink, ...] | None:
        """Choose the preferred closed visual representative of a plaquette."""
        best: tuple[_DrawLink, ...] | None = None
        best_score: tuple[int, int, float, float, float, float] | None = None

        for candidate_tuple in product(*candidate_lists):
            selected = tuple(candidate_tuple)

            if not self._draw_links_form_closed_cycle(selected):
                continue

            score = self._visual_plaquette_representative_score_for_physical_links(
                selected,
                physical_link_ids=physical_link_ids,
                preferred_center=preferred_center,
            )

            if best_score is None or score < best_score:
                best = selected
                best_score = score

        return best

    def _visual_plaquette_representative_score(
        self,
        draw_links: tuple[_DrawLink, ...],
        *,
        preferred_center: npt.NDArray[np.float64] | None = None,
    ) -> tuple[float, float, float, float]:
        """Score visual plaquette representatives.

        Lower score is preferred:
            1. closeness to the plaquette's natural local center;
            2. lower visual center;
            3. left visual center.
            4. compactness;

        This avoids moving actual top-row small-torus plaquettes down to the
        bottom row while still choosing deterministic representatives among
        duplicate PBC images.
        """
        center = self._closed_visual_plaquette_center(draw_links)
        compactness = self._visual_plaquette_compactness_score(draw_links)

        if preferred_center is None:
            center_distance = 0.0
        else:
            center_distance = float(
                np.linalg.norm(
                    np.asarray(center, dtype=float) - np.asarray(preferred_center, dtype=float)
                )
            )

        return (
            center_distance,
            float(center[1]),
            float(center[0]),
            float(compactness),
        )

    def _draw_links_form_closed_cycle(
        self,
        draw_links: tuple[_DrawLink, ...],
        *,
        decimals: int = 10,
    ) -> bool:
        """Return True iff drawn links form one closed polygon.

        This rejects open paths, disconnected pieces, doubled links, and
        incorrectly assembled periodic images.
        """
        if len(draw_links) < 3:
            return False

        def key(position: tuple[float, float]) -> tuple[float, float]:
            return tuple(np.round(np.asarray(position, dtype=float), decimals=decimals))

        adjacency: dict[tuple[float, float], set[tuple[float, float]]] = {}

        for draw_link in draw_links:
            source = key(draw_link.source_position)
            target = key(draw_link.target_position)

            if source == target:
                return False

            adjacency.setdefault(source, set()).add(target)
            adjacency.setdefault(target, set()).add(source)

        # A simple closed n-link polygon has exactly n vertices, and every vertex
        # has degree 2.
        if len(adjacency) != len(draw_links):
            return False

        if any(len(neighbors) != 2 for neighbors in adjacency.values()):
            return False

        # Check connectedness.
        start = next(iter(adjacency))
        visited = {start}
        stack = [start]

        while stack:
            node = stack.pop()
            for neighbor in adjacency[node]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                stack.append(neighbor)

        return len(visited) == len(adjacency)

    def _visual_plaquette_compactness_score(
        self,
        draw_links: tuple[_DrawLink, ...],
    ) -> float:
        """Score a closed visual plaquette; smaller means more compact."""
        positions = self._closed_visual_plaquette_vertices(draw_links)
        xy = np.asarray(positions, dtype=float)

        mins = np.min(xy, axis=0)
        maxs = np.max(xy, axis=0)

        # Prefer compact representatives. This avoids choosing a plaquette image
        # stretched across the torus when a local positive-patch representative
        # exists.
        bbox = maxs - mins
        return float(np.dot(bbox, bbox))

    def _closed_visual_plaquette_vertices(
        self,
        draw_links: tuple[_DrawLink, ...],
        *,
        decimals: int = 10,
    ) -> list[np.ndarray]:
        """Return unique vertices of a closed drawn plaquette."""
        vertices: list[np.ndarray] = []
        seen: set[tuple[float, float]] = set()

        for draw_link in draw_links:
            for position in (draw_link.source_position, draw_link.target_position):
                arr = np.asarray(position, dtype=float)
                key = tuple(np.round(arr, decimals=decimals))
                if key in seen:
                    continue
                seen.add(key)
                vertices.append(arr)

        return vertices

    def _closed_visual_plaquette_center(
        self,
        draw_links: tuple[_DrawLink, ...],
    ) -> np.ndarray:
        """Return the center of a closed drawn plaquette."""
        vertices = self._closed_visual_plaquette_vertices(draw_links)

        if len(vertices) == 0:
            raise ValueError("Cannot compute center of an empty plaquette.")

        return np.mean(np.asarray(vertices, dtype=float), axis=0)

    @staticmethod
    def _draw_link_distance_to_point(
        draw_link: _DrawLink,
        point: npt.ArrayLike,
    ) -> float:
        """Distance from a drawn link midpoint to a point."""
        source = np.asarray(draw_link.source_position, dtype=float)
        target = np.asarray(draw_link.target_position, dtype=float)
        midpoint = 0.5 * (source + target)

        return float(np.linalg.norm(midpoint - np.asarray(point, dtype=float)))

    @staticmethod
    def _unique_positions(
        positions: list[npt.NDArray[np.float64]],
        *,
        decimals: int = 10,
    ) -> list[npt.NDArray[np.float64]]:
        """Remove duplicate plotting positions."""
        out: list[npt.NDArray[np.float64]] = []
        seen: set[tuple[float, float]] = set()

        for position in positions:
            position_array = np.asarray(position, dtype=float)
            key = tuple(np.round(position_array, decimals=decimals).tolist())

            if key in seen:
                continue

            seen.add(key)
            out.append(position_array)

        return out

    def _torus_translation_vectors(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Return full-system torus translation vectors."""
        primitive_vectors = getattr(self.lattice, "primitive_vectors", None)

        if primitive_vectors is None:
            return None

        primitive_vectors = tuple(np.asarray(vector, dtype=float) for vector in primitive_vectors)

        lattice_x = getattr(self.lattice, "lx", None)
        lattice_y = getattr(self.lattice, "ly", None)

        if lattice_x is None or lattice_y is None:
            shape = getattr(self.lattice, "shape", None)

            if shape is None:
                return None

            lattice_x = shape[0]
            lattice_y = shape[1]

        return (
            int(lattice_x) * primitive_vectors[0],
            int(lattice_y) * primitive_vectors[1],
        )

    def _apply_visual_transform(self, position: npt.ArrayLike) -> np.ndarray:
        """Apply coordinate scale and transform to one position."""
        position_array = np.asarray(position, dtype=float)

        if self.coordinate_transform is not None:
            transform = np.asarray(self.coordinate_transform, dtype=float)
            position_array = transform @ position_array

        return self.coordinate_scale * position_array

    def _nearest_periodic_image(
        self,
        position: np.ndarray,
        reference: np.ndarray,
    ) -> np.ndarray:
        """Return the torus image of ``position`` nearest to ``reference``.

        Important:
            For a finite PBC lattice, the periodic translations are the full torus
            periods, not the primitive lattice vectors.
        """
        translations = self._torus_translation_vectors()

        if translations is None:
            return position

        translation_x, translation_y = translations

        best_position = np.asarray(position, dtype=float)
        best_distance = np.linalg.norm(best_position - reference)

        for shift_x in (-1, 0, 1):
            for shift_y in (-1, 0, 1):
                candidate = (
                    np.asarray(position, dtype=float)
                    + shift_x * translation_x
                    + shift_y * translation_y
                )
                distance = np.linalg.norm(candidate - reference)

                if distance < best_distance:
                    best_distance = distance
                    best_position = candidate

        return best_position

    def _collapse_duplicate_draw_plaquettes(
        self,
        draw_plaquettes: list[_DrawPlaquette],
    ) -> list[_DrawPlaquette]:
        """Collapse multiple representatives of the same physical plaquette."""
        by_plaquette_id: dict[int, _DrawPlaquette] = {}

        for draw_plaquette in draw_plaquettes:
            plaquette_id = int(draw_plaquette.plaquette_id)

            existing = by_plaquette_id.get(plaquette_id)
            if existing is None:
                by_plaquette_id[plaquette_id] = draw_plaquette
                continue

            if self._draw_plaquette_position_score(
                draw_plaquette
            ) < self._draw_plaquette_position_score(existing):
                by_plaquette_id[plaquette_id] = draw_plaquette

        return list(by_plaquette_id.values())

    @staticmethod
    def _draw_plaquette_position_score(
        draw_plaquette: _DrawPlaquette,
    ) -> tuple[float, float]:
        """Lower-left preference for duplicate plaquette representatives."""
        center = tuple(float(value) for value in draw_plaquette.center)
        return (
            float(center[1]),
            float(center[0]),
        )

    def _draw_plaquette_symbols(
        self,
        *,
        ax,
        config: npt.NDArray[np.int64],
        style: PlaquetteSymbolStyle,
        draw_plaquettes: list[_DrawPlaquette],
        plaquette_symbol_values: Mapping[int, tuple[str, str]] | None = None,
    ) -> None:
        if style == "none":
            return

        if style == "circulation":
            self._draw_circulation_plaquette_symbols(
                ax=ax,
                config=config,
                draw_plaquettes=draw_plaquettes,
            )
            return

        if style == "resonance":
            self._draw_resonance_plaquette_symbols(
                ax=ax,
                config=config,
                draw_plaquettes=draw_plaquettes,
                plaquette_symbol_values=plaquette_symbol_values,
            )
            return

        raise ValueError(
            "plaquette_symbol_style must be 'auto', 'none', 'circulation', or 'resonance'."
        )
