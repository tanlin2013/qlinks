from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from qlinks.constraints import (
    DimerCoveringConstraint,
    SquareQDMElectricWindingSector,
    SquareWindingSector,
    TriangularZ2WindingSector,
)
from qlinks.encoded import (
    BitmaskAlternatingPlaquetteFlipOperator,
    BitmaskQDMFlipOperator,
    bitmask_alternating_flippability_projectors,
    bitmask_qdm_flippability_projectors,
)
from qlinks.lattice import (
    BoundaryCondition,
    HoneycombLattice,
    LatticeGraph,
    SquareLattice,
    TriangularLattice,
)
from qlinks.models.base import (
    HamiltonianBuilderName,
    HamiltonianModelBase,
    HamiltonianTermSpec,
    validate_builder_name,
)
from qlinks.operators import (
    PlaquettePatternOperator,
    UpdatePlaquettePatternOperator,
    alternating_binary_flippability_projectors,
    qdm_flippability_projectors,
)
from qlinks.variables import LocalSpace, VariableLayout


@dataclass(frozen=True)
class QDMBase(HamiltonianModelBase):
    """
    Shared implementation for link-binary quantum dimer models.

    Subclasses provide the lattice geometry by implementing `_make_lattice()`.
    They may also override `plaquette_ids()` or `make_sectors()` for
    geometry-specific topological sectors.

    Variables:
        n_l in {0, 1}

    Constraint:
        sum of occupied links touching each site = required_count

    Hamiltonian:
        H = kinetic * sum_p flip_p
          + potential * sum_p flippability_p
    """

    kinetic: complex = -1.0
    potential: complex = 0.0
    required_count: int = 1

    def _make_layout(self) -> VariableLayout:
        return VariableLayout.from_lattice_links(
            self.lattice,
            LocalSpace.binary(),
        )

    def make_constraints(
        self,
        layout: VariableLayout | None = None,
    ):
        if layout is None:
            layout = self.layout

        return DimerCoveringConstraint.all_sites(
            lattice=self.lattice,
            layout=layout,
            required_counts=self.required_count,
        )

    def make_sectors(
        self,
        layout: VariableLayout | None = None,
    ):
        """
        Default QDM sector list.

        Geometry-specific subclasses can override this.
        """
        return ()

    def plaquette_ids(self) -> list[int]:
        """
        Plaquettes used by the QDM resonance move.

        The lattice may define qdm_plaquette_ids() to select only the relevant
        resonance loops. For example, triangular QDM should use rhombi rather
        than elementary triangles.
        """

        if hasattr(self.lattice, "qdm_plaquette_ids"):
            return list(self.lattice.qdm_plaquette_ids())

        return [int(p) for p in self.lattice.plaquette_ids]

    def make_kinetic_operators(
        self,
        layout: VariableLayout | None = None,
        *,
        builder: HamiltonianBuilderName = "sparse",
    ) -> tuple[object, ...]:
        validate_builder_name(builder)

        if layout is None:
            layout = self.layout

        if builder == "sparse":
            return tuple(
                PlaquettePatternOperator.alternating_binary_flip(
                    layout=layout,
                    lattice=self.lattice,
                    plaquette_id=int(p),
                    coefficient=self.kinetic,
                )
                for p in self.plaquette_ids()
            )

        if builder == "bitmask":
            return tuple(
                BitmaskAlternatingPlaquetteFlipOperator(
                    layout=layout,
                    lattice=self.lattice,
                    plaquette_id=int(p),
                    coefficient=self.kinetic,
                )
                for p in self.plaquette_ids()
            )

        if builder == "optimized":
            raise NotImplementedError(
                "Generic optimized QDM operators are not implemented yet. "
                "Use builder='sparse' or builder='bitmask'."
            )

        raise ValueError(f"Unsupported builder: {builder}")

    def make_potential_operators(
        self,
        layout: VariableLayout | None = None,
        *,
        builder: HamiltonianBuilderName = "sparse",
    ) -> tuple[object, ...]:
        validate_builder_name(builder)

        if self.potential == 0:
            return ()

        if layout is None:
            layout = self.layout

        operators: list[object] = []

        if builder == "sparse":
            for p in self.plaquette_ids():
                operators.extend(
                    alternating_binary_flippability_projectors(
                        layout=layout,
                        lattice=self.lattice,
                        plaquette_id=int(p),
                        coefficient=self.potential,
                    )
                )
            return tuple(operators)

        if builder == "bitmask":
            for p in self.plaquette_ids():
                operators.extend(
                    bitmask_alternating_flippability_projectors(
                        layout=layout,
                        lattice=self.lattice,
                        plaquette_id=int(p),
                        coefficient=self.potential,
                    )
                )
            return tuple(operators)

        if builder == "optimized":
            raise NotImplementedError(
                "Generic optimized QDM potential is not implemented yet. "
                "Use builder='sparse' or builder='bitmask'."
            )

        raise ValueError(f"Unsupported builder: {builder}")

    def make_operators(
        self,
        layout: VariableLayout | None = None,
        *,
        builder: HamiltonianBuilderName = "sparse",
    ) -> tuple[object, ...]:
        return self.make_kinetic_operators(layout, builder=builder) + self.make_potential_operators(
            layout, builder=builder
        )

    def make_terms(
        self,
        layout: VariableLayout,
        *,
        builder: HamiltonianBuilderName = "sparse",
    ) -> tuple[HamiltonianTermSpec, ...]:
        kinetic_operators = self.make_kinetic_operators(
            layout,
            builder=builder,
        )

        potential_operators = self.make_potential_operators(
            layout,
            builder=builder,
        )

        terms: list[HamiltonianTermSpec] = [
            HamiltonianTermSpec.from_operators(
                name="kinetic",
                operators=kinetic_operators,
                kind="kinetic",
            )
        ]

        if potential_operators:
            terms.append(
                HamiltonianTermSpec.from_operators(
                    name="potential",
                    operators=potential_operators,
                    kind="potential",
                )
            )

        return tuple(terms)


@dataclass(frozen=True)
class SquareQDMModel(QDMBase):
    """
    Square-lattice quantum dimer model.

    This subclass keeps square-specific functionality, especially winding
    sectors.

    winding_convention:
        "cut_count":
            raw count of occupied wrapping links.

        "electric":
            staggered electric-flux winding compatible with the square QDM
            to staggered-charge QLM mapping.
    """

    lx: int = 2
    ly: int = 2
    boundary_condition: BoundaryCondition | str = BoundaryCondition.OPEN
    winding_x: int | None = None
    winding_y: int | None = None
    winding_convention: Literal["cut_count", "electric"] = "electric"

    def _make_lattice(self) -> SquareLattice:
        return SquareLattice(
            self.lx,
            self.ly,
            boundary_condition=self.boundary_condition,
        )

    def plaquette_ids(self) -> list[int]:
        return [int(p) for p in self.lattice.plaquette_ids]

    def make_sectors(
        self,
        layout: VariableLayout | None = None,
    ):
        if layout is None:
            layout = self.layout

        sectors = []

        if self.winding_convention == "cut_count":
            winding_cls = SquareWindingSector
        elif self.winding_convention == "electric":
            winding_cls = SquareQDMElectricWindingSector
        else:
            raise ValueError("winding_convention must be 'cut_count' or 'electric'.")

        if self.winding_x is not None:
            sectors.append(
                winding_cls(
                    layout=layout,
                    lattice=self.lattice,
                    direction="x",
                    target=self.winding_x,
                )
            )

        if self.winding_y is not None:
            sectors.append(
                winding_cls(
                    layout=layout,
                    lattice=self.lattice,
                    direction="y",
                    target=self.winding_y,
                )
            )

        return tuple(sectors)

    def make_kinetic_operators(
        self,
        layout: VariableLayout | None = None,
        *,
        builder: HamiltonianBuilderName = "sparse",
    ) -> tuple[object, ...]:
        """
        Square QDM keeps the specialized length-4 qdm_flip implementation.

        The generic alternating plaquette flip would also work for square
        plaquettes, but the specialized operator preserves the old API and
        square-QDM convention explicitly.
        """

        validate_builder_name(builder)

        if layout is None:
            layout = self.layout

        if builder == "sparse":
            return tuple(
                PlaquettePatternOperator.qdm_flip(
                    layout=layout,
                    lattice=self.lattice,
                    plaquette_id=int(p),
                    coefficient=self.kinetic,
                )
                for p in self.plaquette_ids()
            )

        if builder == "optimized":
            return tuple(
                UpdatePlaquettePatternOperator.qdm_flip(
                    layout=layout,
                    lattice=self.lattice,
                    plaquette_id=int(p),
                    coefficient=self.kinetic,
                )
                for p in self.plaquette_ids()
            )

        if builder == "bitmask":
            return tuple(
                BitmaskQDMFlipOperator(
                    layout=layout,
                    lattice=self.lattice,
                    plaquette_id=int(p),
                    coefficient=self.kinetic,
                )
                for p in self.plaquette_ids()
            )

        raise ValueError(f"Unsupported builder: {builder}")

    def make_potential_operators(
        self,
        layout: VariableLayout | None = None,
        *,
        builder: HamiltonianBuilderName = "sparse",
    ) -> tuple[object, ...]:
        """
        Square QDM keeps the specialized 1010/0101 flippability projectors.
        """

        validate_builder_name(builder)

        if self.potential == 0:
            return ()

        if layout is None:
            layout = self.layout

        operators: list[object] = []

        if builder == "sparse":
            for p in self.plaquette_ids():
                operators.extend(
                    qdm_flippability_projectors(
                        layout=layout,
                        lattice=self.lattice,
                        plaquette_id=int(p),
                        coefficient=self.potential,
                    )
                )
            return tuple(operators)

        if builder == "bitmask":
            for p in self.plaquette_ids():
                operators.extend(
                    bitmask_qdm_flippability_projectors(
                        layout=layout,
                        lattice=self.lattice,
                        plaquette_id=int(p),
                        coefficient=self.potential,
                    )
                )
            return tuple(operators)

        if builder == "optimized":
            raise NotImplementedError(
                "Optimized QDM potential is not implemented as update actions yet. "
                "Use builder='sparse' or builder='bitmask'."
            )

        raise ValueError(f"Unsupported builder: {builder}")


@dataclass(frozen=True)
class TriangularQDMModel(QDMBase):
    """
    Triangular-lattice QDM.

    The QDM resonance plaquettes are rhombi/lozenges, not elementary triangles.
    """

    lx: int = 2
    ly: int = 2
    boundary_condition: BoundaryCondition | str = BoundaryCondition.OPEN
    winding_a: int | None = None
    winding_b: int | None = None

    def _make_lattice(self) -> TriangularLattice:
        return TriangularLattice(
            self.lx,
            self.ly,
            boundary_condition=self.boundary_condition,
            include_triangles=True,
            include_rhombi=True,
        )

    def plaquette_ids(self) -> list[int]:
        return list(self.lattice.qdm_plaquette_ids())

    def make_sectors(
        self,
        layout: VariableLayout | None = None,
    ):
        if layout is None:
            layout = self.layout

        sectors = []

        if self.winding_a is not None:
            sectors.append(
                TriangularZ2WindingSector(
                    layout=layout,
                    lattice=self.lattice,
                    direction="a",
                    target=self.winding_a,
                    value_convention="binary",
                )
            )

        if self.winding_b is not None:
            sectors.append(
                TriangularZ2WindingSector(
                    layout=layout,
                    lattice=self.lattice,
                    direction="b",
                    target=self.winding_b,
                    value_convention="binary",
                )
            )

        return tuple(sectors)


@dataclass(frozen=True)
class HoneycombQDMModel(QDMBase):
    """
    Honeycomb-lattice QDM.

    The QDM resonance plaquettes are hexagons.
    """

    lx: int = 2
    ly: int = 2
    boundary_condition: BoundaryCondition | str = BoundaryCondition.OPEN

    def _make_lattice(self) -> HoneycombLattice:
        return HoneycombLattice(
            self.lx,
            self.ly,
            boundary_condition=self.boundary_condition,
        )

    def plaquette_ids(self) -> list[int]:
        return list(self.lattice.qdm_plaquette_ids())


@dataclass(frozen=True)
class QDMModel(QDMBase):
    """
    Generic lattice-backed QDM model.

    Use this when you already have a LatticeGraph instance.

    For named geometries, prefer:
        SquareQDMModel
        TriangularQDMModel
        HoneycombQDMModel
    """

    lattice_input: LatticeGraph | None = None

    def _make_lattice(self) -> LatticeGraph:
        if self.lattice_input is None:
            raise ValueError(
                "QDMModel requires lattice_input. "
                "Use SquareQDMModel, TriangularQDMModel, or HoneycombQDMModel "
                "for built-in geometries."
            )
        return self.lattice_input

    @classmethod
    def triangular(
        cls,
        lx: int,
        ly: int,
        *,
        boundary_condition: BoundaryCondition | str = BoundaryCondition.OPEN,
        kinetic: complex = -1.0,
        potential: complex = 0.0,
        required_count: int = 1,
    ) -> TriangularQDMModel:
        return TriangularQDMModel(
            lx=lx,
            ly=ly,
            boundary_condition=boundary_condition,
            kinetic=kinetic,
            potential=potential,
            required_count=required_count,
        )

    @classmethod
    def honeycomb(
        cls,
        lx: int,
        ly: int,
        *,
        boundary_condition: BoundaryCondition | str = BoundaryCondition.OPEN,
        kinetic: complex = -1.0,
        potential: complex = 0.0,
        required_count: int = 1,
    ) -> HoneycombQDMModel:
        return HoneycombQDMModel(
            lx=lx,
            ly=ly,
            boundary_condition=boundary_condition,
            kinetic=kinetic,
            potential=potential,
            required_count=required_count,
        )
