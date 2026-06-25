from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from qlinks.constraints import NearestNeighborBlockadeConstraint, TotalValueSector
from qlinks.constraints.disk import (
    DiskDiagonalFamily,
    SquareDiskDiagonalLineSumSector,
    square_disk_line_labels,
)
from qlinks.lattice import BoundaryCondition, SquareLattice
from qlinks.models.base import (
    BasisSolverName,
    HamiltonianBuilderName,
    HamiltonianModelBase,
    HamiltonianTermSpec,
    normalize_sector_labels_for_display,
    validate_builder_name,
)
from qlinks.models.local_terms import LocalOperatorKind, LocalTermDescriptor, LocalTermKind
from qlinks.operators import LocalValueDiagonalOperator
from qlinks.operators.disk import DiskDiagonalHopOperator, DiskDiagonalHopProjector
from qlinks.variables import LocalSpace, VariableLayout

DiskHopFamily = Literal["x_plus_y", "x_minus_y"]


@dataclass(frozen=True)
class SquareQuantumDiskModel(HamiltonianModelBase):
    """Square-lattice quantum disk model with diagonal hopping.

    Variables:
        Binary disk occupations ``n_i in {0, 1}`` on square-lattice sites.  The
        sites should be interpreted as disk centers; this leaves enough geometry
        metadata for a later basis visualizer without introducing a new cell
        variable kind yet.

    Constraint:
        Optional nearest-neighbor hard-core exclusion.

    Kinetic term:
        A disk hops along selected diagonal lattice lines.  A hop in
        ``family='x_plus_y'`` uses displacement ``(+1, -1)`` and preserves every
        ``x + y`` line sum.  A hop in ``family='x_minus_y'`` uses displacement
        ``(+1, +1)`` and preserves every ``x - y`` line sum.

    Diagonal/topological sectors:
        ``disk_number`` fixes the total particle number.  ``x_plus_y_sums`` and
        ``x_minus_y_sums`` fix the conserved diagonal line sums.  The model
        rejects incompatible diagonal sectors when the selected hop families do
        not preserve them.
    """

    lx: int = 2
    ly: int = 2
    boundary_condition: BoundaryCondition | str = BoundaryCondition.OPEN
    coup_kin: complex = -1.0
    coup_pot: complex = 0.0
    chemical_potential: complex = 0.0
    hop_families: tuple[DiskHopFamily, ...] = ("x_plus_y",)
    hard_core_nearest_neighbor: bool = True
    disk_number: int | None = None
    x_plus_y_sums: tuple[int, ...] | None = None
    x_minus_y_sums: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if len(self.hop_families) == 0:
            raise ValueError("hop_families must contain at least one diagonal family.")
        invalid = set(self.hop_families) - {"x_plus_y", "x_minus_y"}
        if invalid:
            raise ValueError("hop_families entries must be 'x_plus_y' or 'x_minus_y'.")

    def _make_lattice(self) -> SquareLattice:
        return SquareLattice(
            self.lx,
            self.ly,
            boundary_condition=self.boundary_condition,
        )

    def _make_layout(self) -> VariableLayout:
        return VariableLayout.from_lattice_sites(
            self.lattice,
            LocalSpace.binary(),
        )

    def disk_site_ids(self) -> tuple[int, ...]:
        """Return the site ids that carry disk variables."""
        return tuple(int(site_id) for site_id in self.lattice.site_ids)

    def disk_coordinates(self) -> dict[int, tuple[int, int]]:
        """Return disk-center coordinates for future visualization code."""
        return {int(site.id): (int(site.cell[0]), int(site.cell[1])) for site in self.lattice.sites}

    def basis_visualization_data(self, config) -> dict[str, object]:
        """Return lightweight disk-occupation data for a future visualizer."""
        arr = np.asarray(config, dtype=np.int64)
        self.layout.validate_config(arr)
        return {
            "kind": "quantum_disk",
            "site_occupations": {
                int(site_id): int(arr[self.layout.site_variable_index(int(site_id))])
                for site_id in self.disk_site_ids()
            },
            "site_coordinates": self.disk_coordinates(),
            "boundary_condition": str(self.lattice.boundary_condition),
        }

    def diagonal_line_labels(self, family: DiskDiagonalFamily) -> tuple[int, ...]:
        return square_disk_line_labels(self.lattice, family=family)

    def make_constraints(self, layout: VariableLayout | None = None):
        if layout is None:
            layout = self.layout
        if not self.hard_core_nearest_neighbor:
            return ()
        return NearestNeighborBlockadeConstraint.from_lattice(
            self.lattice,
            layout,
            occupied_value=1,
        )

    def _validate_diagonal_sector_compatibility(self) -> None:
        families = set(self.hop_families)
        if self.x_plus_y_sums is not None and families - {"x_plus_y"}:
            raise ValueError(
                "x_plus_y_sums is a symmetry sector only when all hops preserve "
                "x+y, i.e. hop_families must be ('x_plus_y',)."
            )
        if self.x_minus_y_sums is not None and families - {"x_minus_y"}:
            raise ValueError(
                "x_minus_y_sums is a symmetry sector only when all hops preserve "
                "x-y, i.e. hop_families must be ('x_minus_y',)."
            )

    def make_sectors(self, layout: VariableLayout | None = None):
        if layout is None:
            layout = self.layout

        self._validate_diagonal_sector_compatibility()

        sectors = []
        if self.disk_number is not None:
            sectors.append(
                TotalValueSector(
                    layout=layout,
                    target=int(self.disk_number),
                    variable_indices=np.asarray(
                        [
                            layout.site_variable_index(int(site_id))
                            for site_id in self.disk_site_ids()
                        ],
                        dtype=np.int64,
                    ),
                    name="disk_number_sector",
                )
            )

        if self.x_plus_y_sums is not None:
            sectors.append(
                SquareDiskDiagonalLineSumSector(
                    layout=layout,
                    lattice=self.lattice,
                    family="x_plus_y",
                    target=tuple(int(v) for v in self.x_plus_y_sums),
                )
            )

        if self.x_minus_y_sums is not None:
            sectors.append(
                SquareDiskDiagonalLineSumSector(
                    layout=layout,
                    lattice=self.lattice,
                    family="x_minus_y",
                    target=tuple(int(v) for v in self.x_minus_y_sums),
                )
            )

        return tuple(sectors)

    def allowed_sector_labels(self):
        return normalize_sector_labels_for_display(self._allowed_sector_labels())

    def _allowed_sector_labels(self) -> dict[str, tuple[object, ...]]:
        labels: dict[str, tuple[object, ...]] = {
            "disk_number": tuple(range(self.lattice.num_sites + 1)),
        }
        labels["x_plus_y_lines"] = self.diagonal_line_labels("x_plus_y")
        labels["x_minus_y_lines"] = self.diagonal_line_labels("x_minus_y")
        return labels

    def nonempty_sector_labels(self, *args, **kwargs):
        return normalize_sector_labels_for_display(self._nonempty_sector_labels(*args, **kwargs))

    def _nonempty_sector_labels(
        self,
        *,
        solver: BasisSolverName = "dfs",
    ) -> dict[str, tuple[object, ...]]:
        nonempty_disk_numbers: list[int] = []
        for disk_number in range(self.lattice.num_sites + 1):
            trial = replace(self, disk_number=disk_number)
            if trial.has_basis_state(solver=solver):
                nonempty_disk_numbers.append(int(disk_number))
        return {"disk_number": tuple(nonempty_disk_numbers)}

    def diagonal_hop_pairs(self, family: DiskHopFamily) -> tuple[tuple[int, int], ...]:
        return DiskDiagonalHopOperator.pairs_for_family(self.lattice, family=family)

    def make_kinetic_operators(
        self,
        layout: VariableLayout | None = None,
        *,
        builder: HamiltonianBuilderName = "sparse",
    ) -> tuple[object, ...]:
        validate_builder_name(builder)
        if builder != "sparse":
            raise NotImplementedError("SquareQuantumDiskModel currently supports builder='sparse'.")
        if layout is None:
            layout = self.layout

        operators: list[object] = []
        for family in self.hop_families:
            for site_i, site_j in self.diagonal_hop_pairs(family):
                operators.append(
                    DiskDiagonalHopOperator(
                        layout=layout,
                        lattice=self.lattice,
                        source_site=int(site_i),
                        target_site=int(site_j),
                        coefficient=self.coup_kin,
                        enforce_nearest_neighbor_blockade=self.hard_core_nearest_neighbor,
                    )
                )
                operators.append(
                    DiskDiagonalHopOperator(
                        layout=layout,
                        lattice=self.lattice,
                        source_site=int(site_j),
                        target_site=int(site_i),
                        coefficient=np.conjugate(self.coup_kin),
                        enforce_nearest_neighbor_blockade=self.hard_core_nearest_neighbor,
                    )
                )
        return tuple(operators)

    def make_potential_operators(
        self,
        layout: VariableLayout | None = None,
        *,
        builder: HamiltonianBuilderName = "sparse",
    ) -> tuple[object, ...]:
        validate_builder_name(builder)
        if builder != "sparse":
            raise NotImplementedError("SquareQuantumDiskModel currently supports builder='sparse'.")
        if layout is None:
            layout = self.layout

        operators: list[object] = []
        if self.chemical_potential != 0:
            for site_id in self.disk_site_ids():
                operators.append(
                    LocalValueDiagonalOperator(
                        layout=layout,
                        variable_index=int(layout.site_variable_index(int(site_id))),
                        coefficient=self.chemical_potential,
                        name=f"disk_number_{site_id}",
                    )
                )

        if self.coup_pot != 0:
            for family in self.hop_families:
                for site_i, site_j in self.diagonal_hop_pairs(family):
                    operators.append(
                        DiskDiagonalHopProjector(
                            layout=layout,
                            source_site=int(site_i),
                            target_site=int(site_j),
                            coefficient=self.coup_pot,
                        )
                    )
                    operators.append(
                        DiskDiagonalHopProjector(
                            layout=layout,
                            source_site=int(site_j),
                            target_site=int(site_i),
                            coefficient=self.coup_pot,
                        )
                    )
        return tuple(operators)

    def make_terms(
        self,
        layout: VariableLayout,
        *,
        builder: HamiltonianBuilderName = "sparse",
    ) -> tuple[HamiltonianTermSpec, ...]:
        kinetic = self.make_kinetic_operators(layout, builder=builder)
        potential = self.make_potential_operators(layout, builder=builder)
        terms = [
            HamiltonianTermSpec.from_operators(
                name="kinetic",
                operators=kinetic,
                kind="kinetic",
            )
        ]
        if potential:
            terms.append(
                HamiltonianTermSpec.from_operators(
                    name="potential",
                    operators=potential,
                    kind="potential",
                )
            )
        return tuple(terms)

    def local_term_descriptors(
        self,
        *,
        operator_kind: LocalOperatorKind | None = None,
        term_kind: LocalTermKind | None = None,
    ) -> tuple[LocalTermDescriptor, ...]:
        if term_kind not in (None, "bond"):
            return ()
        descriptors: list[LocalTermDescriptor] = []
        term_id = 0
        for family in self.hop_families:
            for site_i, site_j in self.diagonal_hop_pairs(family):
                support_sites = (int(site_i), int(site_j))
                support_variables = tuple(
                    int(self.layout.site_variable_index(site_id)) for site_id in support_sites
                )
                if operator_kind in (None, "kinetic", "hamiltonian"):
                    descriptors.append(
                        LocalTermDescriptor(
                            term_id=term_id,
                            term_kind="bond",
                            operator_kind="kinetic",
                            support_links=(),
                            support_sites=support_sites,
                            support_variables=support_variables,
                            label=f"T_{family}_{site_i}_{site_j}",
                        )
                    )
                if self.coup_pot != 0 and operator_kind in (None, "potential", "hamiltonian"):
                    descriptors.append(
                        LocalTermDescriptor(
                            term_id=term_id,
                            term_kind="bond",
                            operator_kind="potential",
                            support_links=(),
                            support_sites=support_sites,
                            support_variables=support_variables,
                            label=f"P_{family}_{site_i}_{site_j}",
                        )
                    )
                term_id += 1
        return tuple(descriptors)

    def make_local_term(
        self,
        descriptor: LocalTermDescriptor,
        layout: VariableLayout,
        *,
        builder: HamiltonianBuilderName = "sparse",
    ) -> HamiltonianTermSpec:
        validate_builder_name(builder)
        if builder != "sparse":
            raise NotImplementedError("SquareQuantumDiskModel currently supports builder='sparse'.")
        if descriptor.term_kind != "bond":
            raise ValueError("SquareQuantumDiskModel local terms currently support bond terms.")
        if len(descriptor.support_sites) != 2:
            raise ValueError("Disk bond descriptors must contain exactly two support sites.")

        site_i, site_j = (int(site) for site in descriptor.support_sites)

        if descriptor.operator_kind == "kinetic":
            operators = (
                DiskDiagonalHopOperator(
                    layout=layout,
                    lattice=self.lattice,
                    source_site=site_i,
                    target_site=site_j,
                    coefficient=self.coup_kin,
                    enforce_nearest_neighbor_blockade=self.hard_core_nearest_neighbor,
                ),
                DiskDiagonalHopOperator(
                    layout=layout,
                    lattice=self.lattice,
                    source_site=site_j,
                    target_site=site_i,
                    coefficient=np.conjugate(self.coup_kin),
                    enforce_nearest_neighbor_blockade=self.hard_core_nearest_neighbor,
                ),
            )
            return HamiltonianTermSpec.from_operators(
                name=f"kinetic_{descriptor.term_id}",
                operators=operators,
                kind="kinetic",
            )

        if descriptor.operator_kind == "potential":
            operators = (
                DiskDiagonalHopProjector(
                    layout=layout,
                    source_site=site_i,
                    target_site=site_j,
                    coefficient=self.coup_pot,
                ),
                DiskDiagonalHopProjector(
                    layout=layout,
                    source_site=site_j,
                    target_site=site_i,
                    coefficient=self.coup_pot,
                ),
            )
            return HamiltonianTermSpec.from_operators(
                name=f"potential_{descriptor.term_id}",
                operators=operators,
                kind="potential",
            )

        raise ValueError(
            "descriptor.operator_kind must be 'kinetic' or 'potential' for SquareQuantumDiskModel."
        )


QuantumDiskModel = SquareQuantumDiskModel
