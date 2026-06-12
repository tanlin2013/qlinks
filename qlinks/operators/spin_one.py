from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from qlinks.lattice import ChainLattice
from qlinks.operators.base import BaseLocalOperator, OperatorAction
from qlinks.operators.updates import BaseLocalUpdateOperator, LocalUpdateAction
from qlinks.variables import VariableLayout


def spin_one_raise_amplitude(m: int) -> float:
    """
    Spin-1 S^+ amplitude.

        S^+ |m> = sqrt(S(S+1) - m(m+1)) |m+1>

    for S=1 and m in {-1, 0, 1}.
    """
    if m >= 1:
        return 0.0

    return float(np.sqrt(2 - m * (m + 1)))


def spin_one_lower_amplitude(m: int) -> float:
    """
    Spin-1 S^- amplitude.

        S^- |m> = sqrt(S(S+1) - m(m-1)) |m-1>

    for S=1 and m in {-1, 0, 1}.
    """
    if m <= -1:
        return 0.0

    return float(np.sqrt(2 - m * (m - 1)))


@dataclass(frozen=True, slots=True)
class SpinOneXYBondOperator(BaseLocalOperator):
    """
    Spin-1 XY bond operator on sites i,j.

        H_ij = J_xy * (S^x_i S^x_j + S^y_i S^y_j)
             = J_xy/2 * (S^+_i S^-_j + S^-_i S^+_j)

    The computational basis is the S^z product basis with values:

        m_i in {-1, 0, +1}
    """

    layout: VariableLayout
    lattice: ChainLattice
    link_id: int
    coefficient: complex = 1.0
    name: str = "spin_one_xy_bond"

    def __post_init__(self) -> None:
        link = self.lattice.links[int(self.link_id)]

        site_i = int(link.source)
        site_j = int(link.target)

        variable_indices = np.asarray(
            [
                self._site_variable_index(site_i),
                self._site_variable_index(site_j),
            ],
            dtype=np.int64,
        )

        self._validate_local_spaces(
            variable_indices,
            {-1, 0, 1},
            operator_name=type(self).__name__,
        )

        object.__setattr__(
            self,
            "_site_ids",
            self._cached_array([site_i, site_j]),
        )
        object.__setattr__(
            self,
            "_variable_indices",
            self._cached_array(variable_indices),
        )

    @property
    def site_ids(self) -> npt.NDArray[np.int64]:
        return self._copy_indices(self._site_ids)

    @property
    def variable_indices(self) -> npt.NDArray[np.int64]:
        return self._copy_indices(self._variable_indices)

    def affected_variables(self) -> npt.NDArray[np.int64]:
        return self._copy_indices(self._variable_indices)

    def apply(self, config: npt.ArrayLike) -> tuple[OperatorAction, ...]:
        arr = self._as_config(config)

        i, j = (int(x) for x in self._variable_indices)
        mi = int(arr[i])
        mj = int(arr[j])

        actions: list[OperatorAction] = []

        # S^+_i S^-_j
        amp_i_plus = spin_one_raise_amplitude(mi)
        amp_j_minus = spin_one_lower_amplitude(mj)

        if amp_i_plus != 0.0 and amp_j_minus != 0.0:
            new = arr.copy()
            new[i] = mi + 1
            new[j] = mj - 1

            coeff = 0.5 * complex(self.coefficient) * amp_i_plus * amp_j_minus
            actions.append(OperatorAction(coeff, new))

        # S^-_i S^+_j
        amp_i_minus = spin_one_lower_amplitude(mi)
        amp_j_plus = spin_one_raise_amplitude(mj)

        if amp_i_minus != 0.0 and amp_j_plus != 0.0:
            new = arr.copy()
            new[i] = mi - 1
            new[j] = mj + 1

            coeff = 0.5 * complex(self.coefficient) * amp_i_minus * amp_j_plus
            actions.append(OperatorAction(coeff, new))

        return tuple(actions)


@dataclass(frozen=True, slots=True)
class UpdateSpinOneXYBondOperator(BaseLocalUpdateOperator):
    """Update-action version of :class:`SpinOneXYBondOperator`.

    The operator returns only local two-site updates instead of allocating full
    output configurations. It is intended for ``OptimizedSparseHamiltonianBuilder``.
    """

    layout: VariableLayout
    lattice: ChainLattice
    link_id: int
    coefficient: complex = 1.0
    name: str = "update_spin_one_xy_bond"

    def __post_init__(self) -> None:
        link = self.lattice.links[int(self.link_id)]

        site_i = int(link.source)
        site_j = int(link.target)

        variable_indices = np.asarray(
            [
                self.layout.site_variable_index(site_i),
                self.layout.site_variable_index(site_j),
            ],
            dtype=np.int64,
        )

        for variable_index in variable_indices:
            values = set(self.layout.local_space(int(variable_index)).values.tolist())
            if values != {-1, 0, 1}:
                raise ValueError(
                    "UpdateSpinOneXYBondOperator requires local-space values " "[-1, 0, 1]."
                )

        object.__setattr__(
            self,
            "_site_ids",
            np.asarray([site_i, site_j], dtype=np.int64),
        )
        object.__setattr__(
            self,
            "_variable_indices",
            variable_indices,
        )

    @property
    def site_ids(self) -> npt.NDArray[np.int64]:
        return self._site_ids.copy()

    @property
    def variable_indices(self) -> npt.NDArray[np.int64]:
        return self._variable_indices.copy()

    def affected_variables(self) -> npt.NDArray[np.int64]:
        return self._variable_indices.copy()

    def apply_update(self, config: npt.ArrayLike) -> tuple[LocalUpdateAction, ...]:
        arr = self._as_config(config)

        i, j = (int(x) for x in self._variable_indices)
        mi = int(arr[i])
        mj = int(arr[j])

        actions: list[LocalUpdateAction] = []

        amp_i_plus = spin_one_raise_amplitude(mi)
        amp_j_minus = spin_one_lower_amplitude(mj)

        if amp_i_plus != 0.0 and amp_j_minus != 0.0:
            coeff = 0.5 * complex(self.coefficient) * amp_i_plus * amp_j_minus
            actions.append(
                LocalUpdateAction(
                    coefficient=coeff,
                    variable_indices=self._variable_indices,
                    new_values=np.asarray([mi + 1, mj - 1], dtype=np.int64),
                )
            )

        amp_i_minus = spin_one_lower_amplitude(mi)
        amp_j_plus = spin_one_raise_amplitude(mj)

        if amp_i_minus != 0.0 and amp_j_plus != 0.0:
            coeff = 0.5 * complex(self.coefficient) * amp_i_minus * amp_j_plus
            actions.append(
                LocalUpdateAction(
                    coefficient=coeff,
                    variable_indices=self._variable_indices,
                    new_values=np.asarray([mi - 1, mj + 1], dtype=np.int64),
                )
            )

        return tuple(actions)
