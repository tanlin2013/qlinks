from __future__ import annotations

from dataclasses import dataclass

from qlinks.lattice import BoundaryCondition, ChainLattice
from qlinks.models.base import (
    HamiltonianBuilderName,
    HamiltonianModelBase,
    HamiltonianTermSpec,
    validate_builder_name,
)
from qlinks.operators import (
    LocalSquareValueDiagonalOperator,
    LocalValueDiagonalOperator,
    SpinOneXYBondOperator,
    UpdateSpinOneXYBondOperator,
)
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
    h_z: complex = 0.0
    d_z: complex = 0.0

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
                SpinOneXYBondOperator(
                    layout=layout,
                    lattice=self.lattice,
                    link_id=int(link_id),
                    coefficient=self.j_xy,
                )
                for link_id in self.lattice.link_ids
            )

        if builder == "optimized":
            return tuple(
                UpdateSpinOneXYBondOperator(
                    layout=layout,
                    lattice=self.lattice,
                    link_id=int(link_id),
                    coefficient=self.j_xy,
                )
                for link_id in self.lattice.link_ids
            )

        raise NotImplementedError(
            "SpinOneXYChainModel currently supports kinetic terms only for "
            "builder='sparse' or builder='optimized'."
        )

    def make_potential_operators(
        self,
        layout: VariableLayout | None = None,
        *,
        builder: HamiltonianBuilderName = "sparse",
    ) -> tuple[object, ...]:
        validate_builder_name(builder)

        if layout is None:
            layout = self.layout

        operators: list[object] = []

        if builder not in ("sparse", "optimized"):
            if self.h_z == 0 and self.d_z == 0:
                return ()

            raise NotImplementedError(
                "SpinOneXYChainModel currently supports potential terms only for "
                "builder='sparse' or builder='optimized'."
            )

        for site_id in self.lattice.site_ids:
            variable_index = int(layout.site_variable_index(int(site_id)))

            if self.h_z != 0:
                operators.append(
                    LocalValueDiagonalOperator(
                        layout=layout,
                        variable_index=variable_index,
                        coefficient=self.h_z,
                        name="spin_one_zeeman_z",
                    )
                )

            if self.d_z != 0:
                operators.append(
                    LocalSquareValueDiagonalOperator(
                        layout=layout,
                        variable_index=variable_index,
                        coefficient=self.d_z,
                        name="spin_one_single_ion_anisotropy",
                    )
                )

        return tuple(operators)

    def make_operators(
        self,
        layout: VariableLayout | None = None,
        *,
        builder: HamiltonianBuilderName = "sparse",
    ) -> tuple[object, ...]:
        return (
            *self.make_kinetic_operators(layout, builder=builder),
            *self.make_potential_operators(layout, builder=builder),
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

        terms = [
            HamiltonianTermSpec.from_operators(
                name="kinetic",
                operators=kinetic_operators,
                kind="kinetic",
            ),
        ]

        if len(potential_operators) > 0:
            terms.append(
                HamiltonianTermSpec.from_operators(
                    name="potential",
                    operators=potential_operators,
                    kind="potential",
                )
            )

        return tuple(terms)
