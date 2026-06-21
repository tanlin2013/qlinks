from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from qlinks.lattice import LatticeGraph
from qlinks.operators.base import BaseLocalOperator, OperatorAction
from qlinks.operators.diagonal import PatternDiagonalOperator
from qlinks.variables import VariableLayout


@dataclass(frozen=True, slots=True)
class PlaquettePatternTransition:
    """
    One allowed plaquette pattern transition.

    initial:
        Values on plaquette links before the operator acts.

    final:
        Values on plaquette links after the operator acts.

    coefficient:
        Matrix element for this transition.
    """

    initial: npt.NDArray[np.int64]
    final: npt.NDArray[np.int64]
    coefficient: complex = 1.0

    def __post_init__(self) -> None:
        initial = np.asarray(self.initial, dtype=np.int64)
        final = np.asarray(self.final, dtype=np.int64)

        if initial.ndim != 1:
            raise ValueError("initial must be one-dimensional.")
        if final.ndim != 1:
            raise ValueError("final must be one-dimensional.")
        if initial.size != final.size:
            raise ValueError("initial and final must have the same length.")

        object.__setattr__(self, "initial", initial)
        object.__setattr__(self, "final", final)
        object.__setattr__(self, "coefficient", complex(self.coefficient))


@dataclass(frozen=True, slots=True)
class PlaquettePatternOperator(BaseLocalOperator):
    """
    General plaquette transition operator.

    It reads the plaquette's link variables in the lattice plaquette order.
    If the current values match one of the allowed transition patterns, it
    returns the corresponding new configuration.

    This is suitable for QDM plaquette flips, constrained ring exchanges,
    and other local loop moves.
    """

    layout: VariableLayout
    lattice: LatticeGraph
    plaquette_id: int
    transitions: tuple[PlaquettePatternTransition, ...]
    name: str = "plaquette_pattern"

    def __post_init__(self) -> None:
        link_ids = self.lattice.plaquette_links(self.plaquette_id)
        variable_indices = np.asarray(
            [self.layout.link_variable_index(int(link_id)) for link_id in link_ids],
            dtype=np.int64,
        )

        if len(self.transitions) == 0:
            raise ValueError("PlaquettePatternOperator needs at least one transition.")

        for transition in self.transitions:
            if transition.initial.size != variable_indices.size:
                raise ValueError("Transition initial pattern has wrong length.")
            if transition.final.size != variable_indices.size:
                raise ValueError("Transition final pattern has wrong length.")

            for variable_index, initial, final in zip(
                variable_indices,
                transition.initial,
                transition.final,
                strict=True,
            ):
                local_space = self.layout.local_space(int(variable_index))
                local_space.validate_value(int(initial))
                local_space.validate_value(int(final))

        object.__setattr__(self, "_link_ids", link_ids)
        object.__setattr__(self, "_variable_indices", variable_indices)

    @classmethod
    def qdm_flip(
        cls,
        layout: VariableLayout,
        lattice: LatticeGraph,
        plaquette_id: int,
        coefficient: complex = 1.0,
        reverse_coefficient: complex | None = None,
    ) -> PlaquettePatternOperator:
        """
        Standard binary dimer plaquette flip:

            1010 <-> 0101

        The order is the plaquette link order supplied by the lattice.
        """
        if reverse_coefficient is None:
            reverse_coefficient = complex(coefficient).conjugate()

        transitions = (
            PlaquettePatternTransition(
                initial=np.asarray([1, 0, 1, 0], dtype=np.int64),
                final=np.asarray([0, 1, 0, 1], dtype=np.int64),
                coefficient=complex(coefficient),
            ),
            PlaquettePatternTransition(
                initial=np.asarray([0, 1, 0, 1], dtype=np.int64),
                final=np.asarray([1, 0, 1, 0], dtype=np.int64),
                coefficient=complex(reverse_coefficient),
            ),
        )

        return cls(
            layout=layout,
            lattice=lattice,
            plaquette_id=plaquette_id,
            transitions=transitions,
            name="qdm_plaquette_flip",
        )

    @classmethod
    def alternating_binary_flip(
        cls,
        layout: VariableLayout,
        lattice: LatticeGraph,
        plaquette_id: int,
        coefficient: complex = 1.0,
        reverse_coefficient: complex | None = None,
    ) -> PlaquettePatternOperator:
        if reverse_coefficient is None:
            reverse_coefficient = complex(coefficient).conjugate()

        link_ids = lattice.plaquette_links(plaquette_id)
        p0, p1 = alternating_binary_patterns(len(link_ids))

        transitions = (
            PlaquettePatternTransition(initial=p0, final=p1, coefficient=complex(coefficient)),
            PlaquettePatternTransition(
                initial=p1, final=p0, coefficient=complex(reverse_coefficient)
            ),
        )

        return cls(
            layout=layout,
            lattice=lattice,
            plaquette_id=plaquette_id,
            transitions=transitions,
            name="alternating_binary_plaquette_flip",
        )

    @classmethod
    def alternating_flux_flip(
        cls,
        layout: VariableLayout,
        lattice: LatticeGraph,
        plaquette_id: int,
        coefficient: complex = 1.0,
    ) -> PlaquettePatternOperator:
        link_ids = lattice.plaquette_links(plaquette_id)
        p0, p1 = alternating_flux_patterns(len(link_ids))

        transitions = (
            PlaquettePatternTransition(initial=p0, final=p1, coefficient=coefficient),
            PlaquettePatternTransition(initial=p1, final=p0, coefficient=coefficient),
        )

        return cls(
            layout=layout,
            lattice=lattice,
            plaquette_id=plaquette_id,
            transitions=transitions,
            name="alternating_flux_plaquette_flip",
        )

    @property
    def link_ids(self) -> npt.NDArray[np.int64]:
        return self._link_ids.copy()

    @property
    def variable_indices(self) -> npt.NDArray[np.int64]:
        return self._variable_indices.copy()

    def affected_variables(self) -> npt.NDArray[np.int64]:
        return self._variable_indices.copy()

    def apply(self, config: npt.ArrayLike) -> tuple[OperatorAction, ...]:
        arr = self._as_config(config)
        local_values = arr[self._variable_indices]

        actions: list[OperatorAction] = []

        for transition in self.transitions:
            if np.array_equal(local_values, transition.initial):
                new = arr.copy()
                new[self._variable_indices] = transition.final
                actions.append(OperatorAction(transition.coefficient, new))

        return tuple(actions)


def qdm_flippability_projectors(
    layout: VariableLayout,
    lattice: LatticeGraph,
    plaquette_id: int,
    coefficient: complex = 1.0,
) -> tuple[PatternDiagonalOperator, PatternDiagonalOperator]:
    """
    Return diagonal projectors onto the two flippable QDM plaquette patterns:

        1010 and 0101

    The potential term V * P_p^2 in a QDM-like model can be represented using
    these diagonal projectors.
    """

    link_ids = lattice.plaquette_links(plaquette_id)
    variable_indices = np.asarray(
        [layout.link_variable_index(int(link_id)) for link_id in link_ids],
        dtype=np.int64,
    )

    return (
        PatternDiagonalOperator(
            layout=layout,
            variable_indices=variable_indices,
            pattern=np.asarray([1, 0, 1, 0], dtype=np.int64),
            coefficient=coefficient,
            name="qdm_flippability_1010",
        ),
        PatternDiagonalOperator(
            layout=layout,
            variable_indices=variable_indices,
            pattern=np.asarray([0, 1, 0, 1], dtype=np.int64),
            coefficient=coefficient,
            name="qdm_flippability_0101",
        ),
    )


def alternating_binary_patterns(length: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the two binary alternating patterns on an even plaquette.

    Args:
        length: Plaquette cycle length.

    Returns:
        Pair of arrays like ``1010...`` and ``0101...``.
    """
    if length <= 0:
        raise ValueError("length must be positive.")
    if length % 2 != 0:
        raise ValueError("alternating binary plaquette patterns require even length.")

    p0 = np.asarray([1 if i % 2 == 0 else 0 for i in range(length)], dtype=np.int64)
    p1 = 1 - p0
    return p0, p1


def alternating_flux_patterns(length: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the two ``{-1, +1}`` alternating flux patterns.

    Args:
        length: Plaquette cycle length.

    Returns:
        Pair of opposite alternating flux arrays.
    """
    if length <= 0:
        raise ValueError("length must be positive.")
    if length % 2 != 0:
        raise ValueError("alternating flux plaquette patterns require even length.")

    p0 = np.asarray([1 if i % 2 == 0 else -1 for i in range(length)], dtype=np.int64)
    p1 = -p0
    return p0, p1


def alternating_binary_flippability_projectors(
    layout: VariableLayout,
    lattice: LatticeGraph,
    plaquette_id: int,
    coefficient: complex = 1.0,
) -> tuple[PatternDiagonalOperator, PatternDiagonalOperator]:
    """Return binary flippability projectors for one plaquette.

    Args:
        layout: Variable layout.
        lattice: Lattice containing the plaquette.
        plaquette_id: Plaquette id.
        coefficient: Diagonal coefficient for each projector.

    Returns:
        Pair of diagonal projectors onto the two alternating binary patterns.
    """
    link_ids = lattice.plaquette_links(plaquette_id)
    variable_indices = np.asarray(
        [layout.link_variable_index(int(link_id)) for link_id in link_ids],
        dtype=np.int64,
    )

    p0, p1 = alternating_binary_patterns(len(link_ids))

    return (
        PatternDiagonalOperator(
            layout=layout,
            variable_indices=variable_indices,
            pattern=p0,
            coefficient=coefficient,
            name="alternating_binary_flippability_0",
        ),
        PatternDiagonalOperator(
            layout=layout,
            variable_indices=variable_indices,
            pattern=p1,
            coefficient=coefficient,
            name="alternating_binary_flippability_1",
        ),
    )


def alternating_flux_flippability_projectors(
    layout: VariableLayout,
    lattice: LatticeGraph,
    plaquette_id: int,
    coefficient: complex = 1.0,
) -> tuple[PatternDiagonalOperator, PatternDiagonalOperator]:
    """Return flux flippability projectors for one plaquette.

    Args:
        layout: Variable layout.
        lattice: Lattice containing the plaquette.
        plaquette_id: Plaquette id.
        coefficient: Diagonal coefficient for each projector.

    Returns:
        Pair of diagonal projectors onto the two alternating flux patterns.
    """
    link_ids = lattice.plaquette_links(plaquette_id)
    variable_indices = np.asarray(
        [layout.link_variable_index(int(link_id)) for link_id in link_ids],
        dtype=np.int64,
    )

    p0, p1 = alternating_flux_patterns(len(link_ids))

    return (
        PatternDiagonalOperator(
            layout=layout,
            variable_indices=variable_indices,
            pattern=p0,
            coefficient=coefficient,
            name="alternating_flux_flippability_0",
        ),
        PatternDiagonalOperator(
            layout=layout,
            variable_indices=variable_indices,
            pattern=p1,
            coefficient=coefficient,
            name="alternating_flux_flippability_1",
        ),
    )
