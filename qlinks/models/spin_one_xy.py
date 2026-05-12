from __future__ import annotations

from dataclasses import dataclass

from qlinks.lattice import BoundaryCondition, ChainLattice
from qlinks.models.base import (
    HamiltonianBuilderName,
    HamiltonianModelBase,
    HamiltonianTermSpec,
    validate_builder_name,
)
from qlinks.operators import SpinOneXYBondOperator
from qlinks.variables import LocalSpace, VariableLayout


@dataclass(frozen=True)
class SpinOneXYChainModel(HamiltonianModelBase):
    """
    Spin-1 XY chain in the S^z product basis.

    Local basis:

        m_i in {-1, 0, +1}

    Hamiltonian:

        H = J_xy * sum_<ij> (S^x_i S^x_j + S^y_i S^y_j)
          = J_xy/2 * sum_<ij> (S^+_i S^-_j + S^-_i S^+_j)

    No constraints are imposed at this stage.
    """

    length: int
    boundary_condition: BoundaryCondition | str = BoundaryCondition.OPEN
    j_xy: complex = 1.0

    def _make_lattice(self) -> ChainLattice:
        return ChainLattice(
            self.length,
            boundary_condition=self.boundary_condition,
        )

    def _make_layout(self) -> VariableLayout:
        return VariableLayout.from_lattice_sites(
            self.lattice,
            LocalSpace.spin_one(),
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

    def make_operators(
        self,
        layout: VariableLayout | None = None,
        *,
        builder: HamiltonianBuilderName = "sparse",
    ) -> tuple[object, ...]:
        validate_builder_name(builder)

        if builder != "sparse":
            raise NotImplementedError(
                "SpinOneXYChainModel currently supports only builder='sparse'."
            )

        if layout is None:
            layout = self.layout

        return tuple(
            SpinOneXYBondOperator(
                layout=layout,
                lattice=self.lattice,
                link_id=int(link_id),
                coefficient=self.j_xy,
            )
            for link_id in self.lattice.link_ids
        )

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
