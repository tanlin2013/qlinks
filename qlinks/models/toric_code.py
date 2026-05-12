from __future__ import annotations

from dataclasses import dataclass

from qlinks.lattice import BoundaryCondition, SquareLattice
from qlinks.models.base import (
    HamiltonianBuilderName,
    HamiltonianModelBase,
    HamiltonianTermSpec,
    validate_builder_name,
)
from qlinks.operators import (
    ToricCodePlaquetteFluxOperator,
    ToricCodeStarFlipOperator,
)
from qlinks.variables import LocalSpace, VariableLayout


@dataclass(frozen=True)
class ToricCodeModel(HamiltonianModelBase):
    """
    Standard toric code on a square lattice with PBC.

    Variables live on links in the Z basis:

        z_l in {-1, +1}

    Hamiltonian:

        H = -electric * sum_v A_v - magnetic * sum_p B_p

    where A_v flips all incident links and B_p is diagonal in the Z basis.
    """

    lx: int = 2
    ly: int = 2
    boundary_condition: BoundaryCondition | str = BoundaryCondition.PERIODIC
    electric: complex = 1.0
    magnetic: complex = 1.0

    def _make_lattice(self) -> SquareLattice:
        boundary_condition = BoundaryCondition(self.boundary_condition)

        if boundary_condition != BoundaryCondition.PERIODIC:
            raise ValueError("ToricCodeModel currently supports only periodic boundary conditions.")

        return SquareLattice(
            self.lx,
            self.ly,
            boundary_condition=boundary_condition,
        )

    def _make_layout(self) -> VariableLayout:
        return VariableLayout.from_lattice_links(
            self.lattice,
            LocalSpace.spin_half_flux(),
        )

    def make_constraints(
        self,
        layout: VariableLayout | None = None,
    ):
        return ()

    def make_sectors(
        self,
        layout: VariableLayout | None = None,
    ):
        return ()

    def make_star_operators(
        self,
        layout: VariableLayout | None = None,
        *,
        builder: HamiltonianBuilderName = "sparse",
    ) -> tuple[object, ...]:
        validate_builder_name(builder)

        if builder != "sparse":
            raise NotImplementedError("ToricCodeModel currently supports only builder='sparse'.")

        if layout is None:
            layout = self.layout

        return tuple(
            ToricCodeStarFlipOperator(
                layout=layout,
                lattice=self.lattice,
                site_id=int(site_id),
                coefficient=-complex(self.electric),
            )
            for site_id in self.lattice.site_ids
        )

    def make_plaquette_operators(
        self,
        layout: VariableLayout | None = None,
        *,
        builder: HamiltonianBuilderName = "sparse",
    ) -> tuple[object, ...]:
        validate_builder_name(builder)

        if builder != "sparse":
            raise NotImplementedError("ToricCodeModel currently supports only builder='sparse'.")

        if layout is None:
            layout = self.layout

        return tuple(
            ToricCodePlaquetteFluxOperator(
                layout=layout,
                lattice=self.lattice,
                plaquette_id=int(plaquette_id),
                coefficient=-complex(self.magnetic),
            )
            for plaquette_id in self.lattice.plaquette_ids
        )

    def make_operators(
        self,
        layout: VariableLayout | None = None,
        *,
        builder: HamiltonianBuilderName = "sparse",
    ) -> tuple[object, ...]:
        return self.make_star_operators(
            layout,
            builder=builder,
        ) + self.make_plaquette_operators(
            layout,
            builder=builder,
        )

    def make_terms(
        self,
        layout: VariableLayout,
        *,
        builder: HamiltonianBuilderName = "sparse",
    ) -> tuple[HamiltonianTermSpec, ...]:
        star_operators = self.make_star_operators(
            layout,
            builder=builder,
        )
        plaquette_operators = self.make_plaquette_operators(
            layout,
            builder=builder,
        )

        return (
            HamiltonianTermSpec.from_operators(
                name="kinetic",
                operators=star_operators,
                kind="kinetic",
            ),
            HamiltonianTermSpec.from_operators(
                name="potential",
                operators=plaquette_operators,
                kind="potential",
            ),
        )
