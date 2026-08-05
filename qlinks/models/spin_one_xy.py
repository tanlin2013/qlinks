from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy import typing as npt

from qlinks.constraints import TotalValueSector
from qlinks.lattice import BoundaryCondition, ChainLattice
from qlinks.models.base import (
    HamiltonianBuilderName,
    HamiltonianModelBase,
    HamiltonianTermSpec,
    validate_builder_name,
)
from qlinks.models.local_terms import (
    LocalOperatorKind,
    LocalTermDescriptor,
    LocalTermKind,
)
from qlinks.operators import (
    LocalSquareValueDiagonalOperator,
    LocalValueDiagonalOperator,
    SpinOneXYBondOperator,
    SpinOneXYPairOperator,
    UpdateSpinOneXYBondOperator,
    UpdateSpinOneXYPairOperator,
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
    total_sz: int | None = None
    extra_xy_couplings: tuple[tuple[int, int, complex], ...] = ()
    h_z_by_site: tuple[complex, ...] | None = None
    d_z_by_site: tuple[complex, ...] | None = None

    def __post_init__(self) -> None:
        if self.length <= 0:
            raise ValueError("length must be positive.")

        couplings: list[tuple[int, int, complex]] = []
        for raw in self.extra_xy_couplings:
            if len(raw) != 3:
                raise ValueError("each extra_xy_coupling must be (site_i, site_j, coefficient).")
            site_i, site_j, coefficient = int(raw[0]), int(raw[1]), complex(raw[2])
            if site_i == site_j:
                raise ValueError("extra XY couplings require distinct sites.")
            if not (0 <= site_i < self.length and 0 <= site_j < self.length):
                raise ValueError("extra XY coupling site is outside the chain.")
            couplings.append((site_i, site_j, coefficient))
        object.__setattr__(self, "extra_xy_couplings", tuple(couplings))

        for name in ("h_z_by_site", "d_z_by_site"):
            values = getattr(self, name)
            if values is None:
                continue
            normalized = tuple(complex(value) for value in values)
            if len(normalized) != self.length:
                raise ValueError(f"{name} must have length equal to the chain length.")
            object.__setattr__(self, name, normalized)

    def _site_coefficient(self, name: str, site_id: int) -> complex:
        by_site = getattr(self, f"{name}_by_site")
        if by_site is not None:
            return complex(by_site[int(site_id)])
        return complex(getattr(self, name))

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
        if layout is None:
            layout = self.layout

        if self.total_sz is None:
            return ()

        return (
            TotalValueSector(
                layout=layout,
                target=int(self.total_sz),
                name="total_sz_sector",
            ),
        )

    def make_kinetic_operators(
        self,
        layout: VariableLayout | None = None,
        *,
        builder: HamiltonianBuilderName = "sparse",
    ) -> tuple[object, ...]:
        validate_builder_name(builder)

        if layout is None:
            layout = self.layout

        operators: list[object] = []
        for link_id in self.lattice.link_ids:
            if builder == "sparse":
                operators.append(
                    SpinOneXYBondOperator(
                        layout=layout,
                        lattice=self.lattice,
                        link_id=int(link_id),
                        coefficient=self.j_xy,
                    )
                )
            elif builder == "optimized":
                operators.append(
                    UpdateSpinOneXYBondOperator(
                        layout=layout,
                        lattice=self.lattice,
                        link_id=int(link_id),
                        coefficient=self.j_xy,
                    )
                )
            else:
                raise NotImplementedError(
                    "SpinOneXYChainModel currently supports kinetic terms only for "
                    "builder='sparse' or builder='optimized'."
                )

        for site_i, site_j, coefficient in self.extra_xy_couplings:
            if builder == "sparse":
                operators.append(
                    SpinOneXYPairOperator(
                        layout=layout,
                        site_i=site_i,
                        site_j=site_j,
                        coefficient=coefficient,
                    )
                )
            elif builder == "optimized":
                operators.append(
                    UpdateSpinOneXYPairOperator(
                        layout=layout,
                        site_i=site_i,
                        site_j=site_j,
                        coefficient=coefficient,
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

        if layout is None:
            layout = self.layout

        if builder not in ("sparse", "optimized"):
            has_potential = any(
                self._site_coefficient(name, site_id) != 0
                for name in ("h_z", "d_z")
                for site_id in range(self.length)
            )
            if not has_potential:
                return ()
            raise NotImplementedError(
                "SpinOneXYChainModel currently supports potential terms only for "
                "builder='sparse' or builder='optimized'."
            )

        operators: list[object] = []
        for site_id in self.lattice.site_ids:
            variable_index = int(layout.site_variable_index(int(site_id)))
            h_value = self._site_coefficient("h_z", int(site_id))
            d_value = self._site_coefficient("d_z", int(site_id))

            if h_value != 0:
                operators.append(
                    LocalValueDiagonalOperator(
                        layout=layout,
                        variable_index=variable_index,
                        coefficient=h_value,
                        name="spin_one_zeeman_z",
                    )
                )

            if d_value != 0:
                operators.append(
                    LocalSquareValueDiagonalOperator(
                        layout=layout,
                        variable_index=variable_index,
                        coefficient=d_value,
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

    def local_term_descriptors(
        self,
        *,
        operator_kind: LocalOperatorKind | None = None,
        term_kind: LocalTermKind | None = None,
    ) -> tuple[LocalTermDescriptor, ...]:
        """Return site/pair local terms for generic diagnostics and builders."""
        descriptors: list[LocalTermDescriptor] = []

        if term_kind in (None, "bond") and operator_kind in (None, "kinetic", "hamiltonian"):
            for link in self.lattice.links:
                support_sites = (int(link.source), int(link.target))
                support_variables = tuple(
                    int(self.layout.site_variable_index(site_id)) for site_id in support_sites
                )
                descriptors.append(
                    LocalTermDescriptor(
                        term_id=int(link.id),
                        term_kind="bond",
                        operator_kind="kinetic",
                        support_links=(int(link.id),),
                        support_sites=support_sites,
                        support_variables=support_variables,
                        label=f"XY_{link.source}_{link.target}",
                    )
                )

            offset = len(self.lattice.links)
            for pair_index, (site_i, site_j, _coefficient) in enumerate(self.extra_xy_couplings):
                support_sites = (int(site_i), int(site_j))
                support_variables = tuple(
                    int(self.layout.site_variable_index(site_id)) for site_id in support_sites
                )
                descriptors.append(
                    LocalTermDescriptor(
                        term_id=offset + pair_index,
                        term_kind="bond",
                        operator_kind="kinetic",
                        support_links=(),
                        support_sites=support_sites,
                        support_variables=support_variables,
                        label=f"XY_pair_{pair_index}_{site_i}_{site_j}",
                    )
                )

        if term_kind in (None, "site") and operator_kind in (None, "potential", "hamiltonian"):
            for site_id in self.lattice.site_ids:
                variable_index = int(self.layout.site_variable_index(int(site_id)))
                if self._site_coefficient("h_z", int(site_id)) != 0:
                    descriptors.append(
                        LocalTermDescriptor(
                            term_id=int(site_id),
                            term_kind="site",
                            operator_kind="potential",
                            support_links=(),
                            support_sites=(int(site_id),),
                            support_variables=(variable_index,),
                            label=f"Sz_{site_id}",
                        )
                    )
                if self._site_coefficient("d_z", int(site_id)) != 0:
                    descriptors.append(
                        LocalTermDescriptor(
                            term_id=int(site_id),
                            term_kind="site",
                            operator_kind="potential",
                            support_links=(),
                            support_sites=(int(site_id),),
                            support_variables=(variable_index,),
                            label=f"Sz2_{site_id}",
                        )
                    )

        return tuple(descriptors)

    def make_local_term(
        self,
        descriptor: LocalTermDescriptor,
        layout: VariableLayout,
        *,
        builder: HamiltonianBuilderName = "sparse",
    ) -> HamiltonianTermSpec:
        validate_builder_name(builder)

        if descriptor.term_kind == "bond" and descriptor.operator_kind == "kinetic":
            n_links = len(self.lattice.links)
            if int(descriptor.term_id) < n_links:
                operator = (
                    SpinOneXYBondOperator(
                        layout=layout,
                        lattice=self.lattice,
                        link_id=int(descriptor.term_id),
                        coefficient=self.j_xy,
                    )
                    if builder == "sparse"
                    else UpdateSpinOneXYBondOperator(
                        layout=layout,
                        lattice=self.lattice,
                        link_id=int(descriptor.term_id),
                        coefficient=self.j_xy,
                    )
                )
            else:
                pair_index = int(descriptor.term_id) - n_links
                try:
                    site_i, site_j, coefficient = self.extra_xy_couplings[pair_index]
                except IndexError as exc:
                    raise ValueError("unknown spin-one XY pair descriptor.") from exc
                operator = (
                    SpinOneXYPairOperator(
                        layout=layout,
                        site_i=site_i,
                        site_j=site_j,
                        coefficient=coefficient,
                    )
                    if builder == "sparse"
                    else UpdateSpinOneXYPairOperator(
                        layout=layout,
                        site_i=site_i,
                        site_j=site_j,
                        coefficient=coefficient,
                    )
                )
            return HamiltonianTermSpec.from_operators(
                name=f"kinetic_{descriptor.term_id}",
                operators=(operator,),
                kind="kinetic",
            )

        if descriptor.term_kind == "site" and descriptor.operator_kind == "potential":
            site_id = int(descriptor.term_id)
            variable_index = int(layout.site_variable_index(site_id))
            operators: list[object] = []

            if descriptor.label is None or str(descriptor.label).startswith("Sz_"):
                h_value = self._site_coefficient("h_z", site_id)
                if h_value != 0:
                    operators.append(
                        LocalValueDiagonalOperator(
                            layout=layout,
                            variable_index=variable_index,
                            coefficient=h_value,
                            name="spin_one_zeeman_z",
                        )
                    )

            if descriptor.label is None or str(descriptor.label).startswith("Sz2_"):
                d_value = self._site_coefficient("d_z", site_id)
                if d_value != 0:
                    operators.append(
                        LocalSquareValueDiagonalOperator(
                            layout=layout,
                            variable_index=variable_index,
                            coefficient=d_value,
                            name="spin_one_single_ion_anisotropy",
                        )
                    )

            return HamiltonianTermSpec.from_operators(
                name=f"potential_{descriptor.label or site_id}",
                operators=tuple(operators),
                kind="potential",
            )

        raise ValueError(
            "SpinOneXYChainModel local terms support pair kinetic terms and site potential terms."
        )


def spin_one_xy_scar_tower_states(
    *,
    basis_configs: npt.NDArray[np.integer],
    length: int | None = None,
    site_phase_offset: int = 0,
    normalize: bool = True,
    include_zero: bool = False,
) -> tuple[npt.NDArray[np.complex128], tuple[str, ...]]:
    """Return the spin-1 XY scar tower in a supplied product/sector basis.

    The tower is generated by ``(Q^dagger)^n |-1,...,-1>`` with
    ``Q^dagger = sum_j (-1)^(j + site_phase_offset) (S^+_j)^2``.  Up to a
    state-dependent normalization, the nonzero amplitudes are on configurations
    with ``n`` sites at ``+1`` and all remaining sites at ``-1``.

    If ``basis_configs`` is already restricted to one total-Sz sector, only the
    corresponding tower vector is nonzero unless ``include_zero=True``.
    """
    configs = np.asarray(basis_configs, dtype=np.int64)
    if configs.ndim != 2:
        raise ValueError("basis_configs must have shape (n_basis, n_variables).")

    n_basis, n_variables = configs.shape
    if length is None:
        length = int(n_variables)
    if int(length) != int(n_variables):
        raise ValueError("length must match the number of spin variables in basis_configs.")

    states: list[npt.NDArray[np.complex128]] = []
    labels: list[str] = []

    for n_raised in range(int(length) + 1):
        vector = np.zeros(n_basis, dtype=np.complex128)
        for basis_index, config in enumerate(configs):
            if np.any((config != -1) & (config != 1)):
                continue
            raised_sites = np.flatnonzero(config == 1)
            if raised_sites.size != n_raised:
                continue
            sign_power = int(np.sum(raised_sites) + site_phase_offset * n_raised)
            vector[basis_index] = -1.0 if sign_power % 2 else 1.0

        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            if include_zero:
                states.append(vector)
                labels.append(f"S_{n_raised}")
            continue
        if normalize:
            vector = vector / norm
        states.append(vector)
        labels.append(f"S_{n_raised}")

    if len(states) == 0:
        return np.zeros((n_basis, 0), dtype=np.complex128), ()

    return np.column_stack(states).astype(np.complex128, copy=False), tuple(labels)


@dataclass(frozen=True, slots=True)
class SpinOneXYTowerThermalActivities:
    """Exact fixed-magnetization witness activities for the pi-bimagnon tower.

    ``xy_matrix_element`` is the qlinks convention: it is the matrix element
    connecting ``|00>`` with ``|+->``.  In the manuscript convention of
    Eq. (104), ``xy_matrix_element = 2 J``.
    """

    length: int
    total_sz: int
    sector_dimension: int
    one_zero_count: int
    two_site_remainder_count: int
    y2_activity: float
    directed_q_activity: float
    z2_activity: float
    p0_limit: float
    y2_limit: float
    directed_q_limit: float
    z2_limit: float
    xy_matrix_element: complex

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "length": self.length,
            "total_sz": self.total_sz,
            "sector_dimension": self.sector_dimension,
            "one_zero_count": self.one_zero_count,
            "two_site_remainder_count": self.two_site_remainder_count,
            "y2_activity": self.y2_activity,
            "directed_q_activity": self.directed_q_activity,
            "z2_activity": self.z2_activity,
            "p0_limit": self.p0_limit,
            "y2_limit": self.y2_limit,
            "directed_q_limit": self.directed_q_limit,
            "z2_limit": self.z2_limit,
            "xy_matrix_element": self.xy_matrix_element,
        }


@dataclass(frozen=True, slots=True)
class SpinOneXYPhaseCompatibilityReport:
    """Bondwise compatibility of a generalized tower phase with XY exchanges."""

    residuals: tuple[complex, ...]
    pairs: tuple[tuple[int, int], ...]
    couplings: tuple[complex, ...]
    phases: tuple[complex, ...]

    @property
    def max_residual(self) -> float:
        return max((abs(value) for value in self.residuals), default=0.0)

    @property
    def is_compatible(self) -> bool:
        return self.max_residual <= 1.0e-10

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "pairs": self.pairs,
            "couplings": self.couplings,
            "phases": self.phases,
            "residuals": self.residuals,
            "max_residual": self.max_residual,
            "is_compatible": self.is_compatible,
        }


def spin_one_xy_periodic_range_couplings(
    *,
    length: int,
    distance: int,
    coefficient: complex,
) -> tuple[tuple[int, int, complex], ...]:
    """Return unique undirected periodic pairs at one separation.

    The ordered orientation is chosen from ``r`` to ``r + distance`` before
    duplicate undirected pairs are removed.  Real coefficients therefore give
    the usual translation-invariant exchange.  For complex coefficients the
    orientation fixes the Peierls phase convention.
    """
    if length <= 1:
        raise ValueError("length must exceed one.")
    step = int(distance) % int(length)
    if step == 0:
        raise ValueError("distance must not be a multiple of length.")
    seen: set[frozenset[int]] = set()
    pairs: list[tuple[int, int, complex]] = []
    for site_i in range(int(length)):
        site_j = (site_i + step) % int(length)
        key = frozenset((site_i, site_j))
        if key in seen:
            continue
        seen.add(key)
        pairs.append((site_i, site_j, complex(coefficient)))
    return tuple(pairs)


def spin_one_xy_hxy_h3_model(
    *,
    length: int,
    j: complex = 1.0,
    j3: complex = 0.1,
    total_sz: int | None = None,
    h_z: complex = 0.0,
    d_z: complex = 0.0,
) -> SpinOneXYChainModel:
    """Return the periodic manuscript Hamiltonian ``H_XY + H_3``.

    The manuscript convention is

    ``H_XY = J sum_r (S_r^+ S_{r+1}^- + h.c.)`` and
    ``H_3  = J3 sum_r (S_r^+ S_{r+3}^- + h.c.)``.

    :class:`SpinOneXYChainModel` uses the conventional ``J_xy/2`` prefactor
    for the ladder-operator form, so the corresponding qlinks coefficients are
    ``j_xy=2*J`` and ``extra_xy_coupling=2*J3``.  The third-neighbor term is
    phase compatible with the staggered tower on even periodic chains.
    """
    if length <= 0:
        raise ValueError("length must be positive.")
    return SpinOneXYChainModel(
        length=int(length),
        boundary_condition=BoundaryCondition.PERIODIC,
        j_xy=2.0 * complex(j),
        h_z=complex(h_z),
        d_z=complex(d_z),
        total_sz=total_sz,
        extra_xy_couplings=spin_one_xy_periodic_range_couplings(
            length=int(length),
            distance=3,
            coefficient=2.0 * complex(j3),
        ),
    )


def spin_one_xy_hxy_h3_imaginary_j2_model(
    *,
    length: int,
    j: complex = 1.0,
    j3: complex = 0.1,
    kappa: float = 0.0,
    total_sz: int | None = None,
    h_z: complex = 0.0,
    d_z: complex = 0.0,
) -> SpinOneXYChainModel:
    """Return ``H_XY + H_3 + i kappa H_2^-`` on a periodic chain.

    In manuscript ladder-operator conventions,

    ``H_2^-(kappa) = i kappa sum_r (S_r^+ S_{r+2}^- - h.c.)``.

    The corresponding qlinks pair coefficient is ``2 i kappa``.  For the
    staggered ``Q=pi`` bimagnon tower, real odd-range exchanges and purely
    imaginary even-range exchanges separately satisfy the exact bondwise
    cancellation rule.  Thus this family continuously contains
    :func:`spin_one_xy_hxy_h3_model` at ``kappa=0`` while preserving the same
    tower and its zero energy.
    """
    if length <= 0:
        raise ValueError("length must be positive.")
    if not np.isfinite(float(kappa)):
        raise ValueError("kappa must be finite.")
    if int(length) == 4 and abs(float(kappa)) > 0.0:
        raise ValueError(
            "a translation-invariant oriented second-neighbor phase is ambiguous at length=4"
        )
    extra = list(
        spin_one_xy_periodic_range_couplings(
            length=int(length),
            distance=3,
            coefficient=2.0 * complex(j3),
        )
    )
    if abs(float(kappa)) > 0.0:
        extra.extend(
            spin_one_xy_periodic_range_couplings(
                length=int(length),
                distance=2,
                coefficient=2.0j * float(kappa),
            )
        )
    return SpinOneXYChainModel(
        length=int(length),
        boundary_condition=BoundaryCondition.PERIODIC,
        j_xy=2.0 * complex(j),
        h_z=complex(h_z),
        d_z=complex(d_z),
        total_sz=total_sz,
        extra_xy_couplings=tuple(extra),
    )


def spin_one_xy_fixed_magnetization_dimension(length: int, total_sz: int) -> int:
    """Return ``[z^M](z^-1 + 1 + z)^L`` by exact dynamic programming."""
    if length < 0:
        raise ValueError("length must be non-negative.")
    counts = {0: 1}
    for _ in range(int(length)):
        updated: dict[int, int] = {}
        for magnetization, count in counts.items():
            for local_value in (-1, 0, 1):
                key = magnetization + local_value
                updated[key] = updated.get(key, 0) + count
        counts = updated
    return int(counts.get(int(total_sz), 0))


def spin_one_xy_tower_thermal_activities(
    *,
    length: int,
    total_sz: int,
    xy_matrix_element: complex = 1.0,
) -> SpinOneXYTowerThermalActivities:
    """Evaluate the exact finite-L ratios and their fixed-density limits.

    The returned quantities are ``Tr(rho Y_r^2)``, the one-sided directed
    activity ``Tr(rho A_r^dagger A_r)``, and
    ``Tr(rho Z_{r,r+1}^2)`` in the infinite-temperature fixed-magnetization
    ensemble.  They correspond to the local channels in the current draft
    after identifying ``xy_matrix_element = 2 J``.
    """
    if length < 2:
        raise ValueError("length must be at least two.")
    dimension = spin_one_xy_fixed_magnetization_dimension(length, total_sz)
    if dimension == 0:
        raise ValueError("requested fixed-magnetization sector is empty.")
    one_zero = spin_one_xy_fixed_magnetization_dimension(length - 1, total_sz)
    remainder = spin_one_xy_fixed_magnetization_dimension(length - 2, total_sz)
    matrix_element = complex(xy_matrix_element)
    y2 = float(one_zero / dimension)
    directed_q = float(2.0 * abs(matrix_element) ** 2 * remainder / dimension)
    z2 = float(4.0 * abs(matrix_element) ** 2 * remainder / dimension)
    q = float(total_sz) / float(length)
    if abs(q) > 1.0:
        raise ValueError("magnetization density must lie in [-1, 1].")
    p0 = float((np.sqrt(max(4.0 - 3.0 * q * q, 0.0)) - 1.0) / 3.0)
    return SpinOneXYTowerThermalActivities(
        length=int(length),
        total_sz=int(total_sz),
        sector_dimension=dimension,
        one_zero_count=one_zero,
        two_site_remainder_count=remainder,
        y2_activity=y2,
        directed_q_activity=directed_q,
        z2_activity=z2,
        p0_limit=p0,
        y2_limit=p0,
        directed_q_limit=float(2.0 * abs(matrix_element) ** 2 * p0**2),
        z2_limit=float(4.0 * abs(matrix_element) ** 2 * p0**2),
        xy_matrix_element=matrix_element,
    )


def spin_one_xy_phase_compatibility(
    couplings: tuple[tuple[int, int, complex], ...],
    *,
    phases: npt.ArrayLike,
) -> SpinOneXYPhaseCompatibilityReport:
    """Check ``t* eta_i + t eta_j = 0`` for every Hermitian pair exchange."""
    eta = np.asarray(phases, dtype=np.complex128).reshape(-1)
    if eta.size == 0:
        raise ValueError("phases must not be empty.")
    if np.any(np.abs(np.abs(eta) - 1.0) > 1.0e-10):
        raise ValueError("tower phases must have unit modulus.")
    pairs: list[tuple[int, int]] = []
    values: list[complex] = []
    residuals: list[complex] = []
    for site_i, site_j, coupling in couplings:
        i, j, value = int(site_i), int(site_j), complex(coupling)
        if not (0 <= i < eta.size and 0 <= j < eta.size):
            raise ValueError("coupling site lies outside the phase array.")
        pairs.append((i, j))
        values.append(value)
        residuals.append(np.conj(value) * eta[i] + value * eta[j])
    return SpinOneXYPhaseCompatibilityReport(
        residuals=tuple(residuals),
        pairs=tuple(pairs),
        couplings=tuple(values),
        phases=tuple(complex(value) for value in eta),
    )
