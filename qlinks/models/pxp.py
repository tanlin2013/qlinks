from __future__ import annotations

from dataclasses import dataclass

from qlinks.constraints import NearestNeighborBlockadeConstraint
from qlinks.encoded import BitmaskPXPSpinFlipOperator
from qlinks.lattice import BoundaryCondition, ChainLattice, SquareLattice
from qlinks.models.base import (
    HamiltonianBuilderName,
    HamiltonianModelBase,
    HamiltonianTermSpec,
    validate_builder_name,
)
from qlinks.operators import PXPSpinFlipOperator, UpdatePXPSpinFlipOperator
from qlinks.variables import LocalSpace, VariableLayout


@dataclass(frozen=True)
class PXPModel(HamiltonianModelBase):
    """
    PXP/Rydberg blockade model.

    Variables:
        binary site occupations n_i in {0, 1}

    Constraint:
        no two neighboring sites can both be occupied.

    Hamiltonian:
        H = omega * sum_i P_neighbors X_i P_neighbors

    The constrained basis already enforces the blockade, and the operator
    applies spin flips only when neighboring sites are unoccupied.
    """

    lattice_input: ChainLattice | SquareLattice
    omega: complex = 1.0

    @classmethod
    def chain(
        cls,
        length: int,
        *,
        boundary_condition: BoundaryCondition | str = BoundaryCondition.OPEN,
        omega: complex = 1.0,
    ) -> PXPModel:
        return cls(
            lattice_input=ChainLattice(
                length,
                boundary_condition=boundary_condition,
            ),
            omega=omega,
        )

    @classmethod
    def square(
        cls,
        lx: int,
        ly: int,
        *,
        boundary_condition: BoundaryCondition | str = BoundaryCondition.OPEN,
        omega: complex = 1.0,
    ) -> PXPModel:
        return cls(
            lattice_input=SquareLattice(
                lx,
                ly,
                boundary_condition=boundary_condition,
            ),
            omega=omega,
        )

    def _make_lattice(self) -> ChainLattice | SquareLattice:
        return self.lattice_input

    def _make_layout(self) -> VariableLayout:
        return VariableLayout.from_lattice_sites(
            self.lattice,
            LocalSpace.binary(),
        )

    def make_constraints(
        self,
        layout: VariableLayout | None = None,
    ):
        if layout is None:
            layout = self.layout

        return NearestNeighborBlockadeConstraint.from_lattice(
            self.lattice,
            layout,
            occupied_value=1,
        )

    def make_sectors(
        self,
        layout: VariableLayout | None = None,
    ):
        return ()

    def make_operators(
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
                PXPSpinFlipOperator(
                    layout=layout,
                    lattice=self.lattice,
                    site_id=int(site_id),
                    coefficient=self.omega,
                )
                for site_id in self.lattice.site_ids
            )

        if builder == "optimized":
            return tuple(
                UpdatePXPSpinFlipOperator(
                    layout=layout,
                    lattice=self.lattice,
                    site_id=int(site_id),
                    coefficient=self.omega,
                )
                for site_id in self.lattice.site_ids
            )

        if builder == "bitmask":
            return tuple(
                BitmaskPXPSpinFlipOperator(
                    layout=layout,
                    lattice=self.lattice,
                    site_id=int(site_id),
                    coefficient=self.omega,
                )
                for site_id in self.lattice.site_ids
            )

        raise ValueError(f"Unsupported builder: {builder}")

    def make_terms(
        self,
        layout: VariableLayout,
        *,
        builder: HamiltonianBuilderName = "sparse",
    ) -> tuple[HamiltonianTermSpec, ...]:
        return (
            HamiltonianTermSpec.from_operators(
                name="kinetic",
                operators=self.make_operators(
                    layout,
                    builder=builder,
                ),
                kind="kinetic",
            ),
        )
