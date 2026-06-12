from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
import numpy.typing as npt

from qlinks.lattice import LatticeGraph
from qlinks.variables import VariableKind, VariableLayout


@dataclass(frozen=True, slots=True)
class LocalUpdateAction:
    """
    Compact operator action.

    Instead of storing the full output configuration, store only

        variable_indices
        new_values

    The optimized sparse builder will reuse a scratch array:

        scratch[:] = config
        scratch[variable_indices] = new_values
    """

    coefficient: complex
    variable_indices: npt.NDArray[np.int64]
    new_values: npt.NDArray[np.int64]

    def __post_init__(self) -> None:
        variable_indices = np.asarray(self.variable_indices, dtype=np.int64)
        new_values = np.asarray(self.new_values, dtype=np.int64)

        if variable_indices.ndim != 1:
            raise ValueError("variable_indices must be one-dimensional.")

        if new_values.ndim != 1:
            raise ValueError("new_values must be one-dimensional.")

        if variable_indices.size != new_values.size:
            raise ValueError("variable_indices and new_values must have the same length.")

        if variable_indices.size == 0:
            raise ValueError("LocalUpdateAction needs at least one updated variable.")

        object.__setattr__(self, "coefficient", complex(self.coefficient))
        object.__setattr__(self, "variable_indices", variable_indices)
        object.__setattr__(self, "new_values", new_values)


class LocalUpdateOperator(Protocol):
    """
    Optimized local-operator protocol.

    This protocol does not return a full new configuration. It returns only
    local updates.
    """

    layout: VariableLayout
    name: str

    def affected_variables(self) -> npt.NDArray[np.int64]: ...

    def apply_update(self, config: npt.ArrayLike) -> tuple[LocalUpdateAction, ...]: ...


class SingleLocalUpdateOperator(LocalUpdateOperator, Protocol):
    """Local update operator with at most one action per input config."""

    def single_update(
        self,
        config: npt.ArrayLike,
    ) -> tuple[complex, npt.NDArray[np.int64], npt.NDArray[np.int64]] | None: ...


class BaseLocalUpdateOperator:
    layout: VariableLayout
    name: str

    def _as_config(
        self,
        config: npt.ArrayLike,
        *,
        validate: bool = True,
    ) -> npt.NDArray[np.int64]:
        arr = np.asarray(config, dtype=np.int64)

        if validate:
            self.layout.validate_config(arr)
        elif arr.shape != self.layout.shape:
            raise ValueError(f"Expected config shape {self.layout.shape}, got {arr.shape}.")

        return arr

    def affected_variables(self) -> npt.NDArray[np.int64]:
        return np.arange(self.layout.n_variables, dtype=np.int64)


@dataclass(frozen=True, slots=True)
class UpdateOperatorSum:
    """
    Sum of update-level local operators.
    """

    terms: tuple[LocalUpdateOperator, ...]
    name: str = "update_operator_sum"

    @classmethod
    def from_terms(
        cls,
        terms: Sequence[LocalUpdateOperator],
        name: str = "update_operator_sum",
    ) -> UpdateOperatorSum:
        return cls(terms=tuple(terms), name=name)

    def affected_variables(self) -> npt.NDArray[np.int64]:
        affected: set[int] = set()
        for term in self.terms:
            affected.update(int(i) for i in term.affected_variables())
        return np.asarray(sorted(affected), dtype=np.int64)

    def apply_update(self, config: npt.ArrayLike) -> tuple[LocalUpdateAction, ...]:
        actions: list[LocalUpdateAction] = []
        for term in self.terms:
            actions.extend(term.apply_update(config))
        return tuple(actions)


@dataclass(frozen=True, slots=True)
class UpdateSetVariablesOperator(BaseLocalUpdateOperator):
    """
    Optimized version of SetVariablesOperator.

    If config[variable_indices] == initial_values, return a local update
    setting those variables to final_values.
    """

    layout: VariableLayout
    variable_indices: npt.NDArray[np.int64]
    initial_values: npt.NDArray[np.int64]
    final_values: npt.NDArray[np.int64]
    coefficient: complex = 1.0
    name: str = "update_set_variables"

    def __post_init__(self) -> None:
        variable_indices = np.asarray(self.variable_indices, dtype=np.int64)
        initial_values = np.asarray(self.initial_values, dtype=np.int64)
        final_values = np.asarray(self.final_values, dtype=np.int64)

        if variable_indices.ndim != 1:
            raise ValueError("variable_indices must be one-dimensional.")
        if initial_values.ndim != 1:
            raise ValueError("initial_values must be one-dimensional.")
        if final_values.ndim != 1:
            raise ValueError("final_values must be one-dimensional.")

        if not (variable_indices.size == initial_values.size == final_values.size):
            raise ValueError(
                "variable_indices, initial_values, and final_values must have the same length."
            )

        if variable_indices.size == 0:
            raise ValueError("At least one variable is required.")

        for variable_index, initial, final in zip(
            variable_indices,
            initial_values,
            final_values,
            strict=True,
        ):
            local_space = self.layout.local_space(int(variable_index))
            local_space.validate_value(int(initial))
            local_space.validate_value(int(final))

        object.__setattr__(self, "variable_indices", variable_indices)
        object.__setattr__(self, "initial_values", initial_values)
        object.__setattr__(self, "final_values", final_values)

    def affected_variables(self) -> npt.NDArray[np.int64]:
        return self.variable_indices.copy()

    def single_update(
        self,
        config: npt.ArrayLike,
    ) -> tuple[complex, npt.NDArray[np.int64], npt.NDArray[np.int64]] | None:
        arr = self._as_config(config)

        if not np.array_equal(arr[self.variable_indices], self.initial_values):
            return None

        return (self.coefficient, self.variable_indices, self.final_values)

    def apply_update(self, config: npt.ArrayLike) -> tuple[LocalUpdateAction, ...]:
        update = self.single_update(config)

        if update is None:
            return ()

        coefficient, variable_indices, new_values = update

        return (
            LocalUpdateAction(
                coefficient=coefficient,
                variable_indices=variable_indices,
                new_values=new_values,
            ),
        )


@dataclass(frozen=True, slots=True)
class UpdateBinaryFlipOperator(BaseLocalUpdateOperator):
    """
    Optimized binary flip 0 <-> 1.
    """

    layout: VariableLayout
    variable_index: int
    coefficient: complex = 1.0
    name: str = "update_binary_flip"

    def __post_init__(self) -> None:
        values = set(self.layout.local_space(self.variable_index).values.tolist())
        if values != {0, 1}:
            raise ValueError("UpdateBinaryFlipOperator requires local values {0, 1}.")

        object.__setattr__(
            self,
            "_variable_indices",
            np.asarray([self.variable_index], dtype=np.int64),
        )
        object.__setattr__(
            self,
            "_value_if_zero",
            np.asarray([1], dtype=np.int64),
        )
        object.__setattr__(
            self,
            "_value_if_one",
            np.asarray([0], dtype=np.int64),
        )

    @classmethod
    def on_site(
        cls,
        layout: VariableLayout,
        site_id: int,
        coefficient: complex = 1.0,
    ) -> UpdateBinaryFlipOperator:
        return cls(
            layout=layout,
            variable_index=layout.variable_index(VariableKind.SITE, site_id),
            coefficient=coefficient,
        )

    @classmethod
    def on_link(
        cls,
        layout: VariableLayout,
        link_id: int,
        coefficient: complex = 1.0,
    ) -> UpdateBinaryFlipOperator:
        return cls(
            layout=layout,
            variable_index=layout.variable_index(VariableKind.LINK, link_id),
            coefficient=coefficient,
        )

    def affected_variables(self) -> npt.NDArray[np.int64]:
        return self._variable_indices.copy()

    def single_update(
        self,
        config: npt.ArrayLike,
    ) -> tuple[complex, npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        arr = self._as_config(config)
        value = int(arr[self.variable_index])
        new_values = self._value_if_zero if value == 0 else self._value_if_one

        return (self.coefficient, self._variable_indices, new_values)

    def apply_update(self, config: npt.ArrayLike) -> tuple[LocalUpdateAction, ...]:
        coefficient, variable_indices, new_values = self.single_update(config)

        return (
            LocalUpdateAction(
                coefficient=coefficient,
                variable_indices=variable_indices,
                new_values=new_values,
            ),
        )


@dataclass(frozen=True, slots=True)
class UpdateNegationFlipOperator(BaseLocalUpdateOperator):
    """
    Optimized sign flip v -> -v.
    """

    layout: VariableLayout
    variable_index: int
    coefficient: complex = 1.0
    name: str = "update_negation_flip"

    def __post_init__(self) -> None:
        values = set(self.layout.local_space(self.variable_index).values.tolist())
        for value in values:
            if -value not in values:
                raise ValueError(
                    "UpdateNegationFlipOperator requires a local space closed under v -> -v."
                )

        object.__setattr__(
            self,
            "_variable_indices",
            np.asarray([self.variable_index], dtype=np.int64),
        )

    @classmethod
    def on_link(
        cls,
        layout: VariableLayout,
        link_id: int,
        coefficient: complex = 1.0,
    ) -> UpdateNegationFlipOperator:
        return cls(
            layout=layout,
            variable_index=layout.variable_index(VariableKind.LINK, link_id),
            coefficient=coefficient,
        )

    def affected_variables(self) -> npt.NDArray[np.int64]:
        return self._variable_indices.copy()

    def single_update(
        self,
        config: npt.ArrayLike,
    ) -> tuple[complex, npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        arr = self._as_config(config)
        new_value = -int(arr[self.variable_index])

        return (
            self.coefficient,
            self._variable_indices,
            np.asarray([new_value], dtype=np.int64),
        )

    def apply_update(self, config: npt.ArrayLike) -> tuple[LocalUpdateAction, ...]:
        coefficient, variable_indices, new_values = self.single_update(config)

        return (
            LocalUpdateAction(
                coefficient=coefficient,
                variable_indices=variable_indices,
                new_values=new_values,
            ),
        )


@dataclass(frozen=True, slots=True)
class UpdateMultiNegationFlipOperator(BaseLocalUpdateOperator):
    """
    Optimized simultaneous sign flip on several variables.
    """

    layout: VariableLayout
    variable_indices: npt.NDArray[np.int64]
    coefficient: complex = 1.0
    name: str = "update_multi_negation_flip"

    def __post_init__(self) -> None:
        variable_indices = np.asarray(self.variable_indices, dtype=np.int64)

        if variable_indices.ndim != 1:
            raise ValueError("variable_indices must be one-dimensional.")

        if variable_indices.size == 0:
            raise ValueError("At least one variable is required.")

        for variable_index in variable_indices:
            values = set(self.layout.local_space(int(variable_index)).values.tolist())
            for value in values:
                if -value not in values:
                    raise ValueError(
                        "UpdateMultiNegationFlipOperator requires local spaces closed "
                        "under v -> -v."
                    )

        object.__setattr__(self, "variable_indices", variable_indices)

    def affected_variables(self) -> npt.NDArray[np.int64]:
        return self.variable_indices.copy()

    def single_update(
        self,
        config: npt.ArrayLike,
    ) -> tuple[complex, npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        arr = self._as_config(config)
        new_values = -arr[self.variable_indices]

        return (self.coefficient, self.variable_indices, new_values)

    def apply_update(self, config: npt.ArrayLike) -> tuple[LocalUpdateAction, ...]:
        coefficient, variable_indices, new_values = self.single_update(config)

        return (
            LocalUpdateAction(
                coefficient=coefficient,
                variable_indices=variable_indices,
                new_values=new_values,
            ),
        )


@dataclass(frozen=True, slots=True)
class UpdatePlaquettePatternTransition:
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
class UpdatePlaquettePatternOperator(BaseLocalUpdateOperator):
    """
    Optimized plaquette-pattern transition operator.

    This is the update-action version of PlaquettePatternOperator.
    """

    layout: VariableLayout
    lattice: LatticeGraph
    plaquette_id: int
    transitions: tuple[UpdatePlaquettePatternTransition, ...]
    name: str = "update_plaquette_pattern"

    def __post_init__(self) -> None:
        link_ids = self.lattice.plaquette_links(self.plaquette_id)

        variable_indices = np.asarray(
            [self.layout.link_variable_index(int(link_id)) for link_id in link_ids],
            dtype=np.int64,
        )

        if len(self.transitions) == 0:
            raise ValueError("UpdatePlaquettePatternOperator needs at least one transition.")

        initial_patterns: set[bytes] = set()
        has_unique_initial_patterns = True

        for transition in self.transitions:
            if transition.initial.size != variable_indices.size:
                raise ValueError("Transition initial pattern has wrong length.")
            if transition.final.size != variable_indices.size:
                raise ValueError("Transition final pattern has wrong length.")

            initial_key = np.ascontiguousarray(transition.initial, dtype=np.int64).tobytes()
            if initial_key in initial_patterns:
                has_unique_initial_patterns = False
            initial_patterns.add(initial_key)

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
        object.__setattr__(self, "_supports_single_update", has_unique_initial_patterns)

    @classmethod
    def qdm_flip(
        cls,
        layout: VariableLayout,
        lattice: LatticeGraph,
        plaquette_id: int,
        coefficient: complex = 1.0,
        reverse_coefficient: complex | None = None,
    ) -> UpdatePlaquettePatternOperator:
        if reverse_coefficient is None:
            reverse_coefficient = complex(coefficient).conjugate()

        transitions = (
            UpdatePlaquettePatternTransition(
                initial=np.asarray([1, 0, 1, 0], dtype=np.int64),
                final=np.asarray([0, 1, 0, 1], dtype=np.int64),
                coefficient=complex(coefficient),
            ),
            UpdatePlaquettePatternTransition(
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
            name="update_qdm_plaquette_flip",
        )

    @property
    def link_ids(self) -> npt.NDArray[np.int64]:
        return self._link_ids.copy()

    @property
    def variable_indices(self) -> npt.NDArray[np.int64]:
        return self._variable_indices.copy()

    def affected_variables(self) -> npt.NDArray[np.int64]:
        return self._variable_indices.copy()

    @property
    def supports_single_update(self) -> bool:
        return bool(self._supports_single_update)

    def single_update(
        self,
        config: npt.ArrayLike,
    ) -> tuple[complex, npt.NDArray[np.int64], npt.NDArray[np.int64]] | None:
        if not self._supports_single_update:
            return None

        arr = self._as_config(config)
        local_values = arr[self._variable_indices]

        for transition in self.transitions:
            if np.array_equal(local_values, transition.initial):
                return (transition.coefficient, self._variable_indices, transition.final)

        return None

    def apply_update(self, config: npt.ArrayLike) -> tuple[LocalUpdateAction, ...]:
        arr = self._as_config(config)
        local_values = arr[self._variable_indices]

        actions: list[LocalUpdateAction] = []

        for transition in self.transitions:
            if np.array_equal(local_values, transition.initial):
                actions.append(
                    LocalUpdateAction(
                        coefficient=transition.coefficient,
                        variable_indices=self._variable_indices,
                        new_values=transition.final,
                    )
                )

        return tuple(actions)


@dataclass(frozen=True, slots=True)
class UpdatePXPSpinFlipOperator(BaseLocalUpdateOperator):
    """
    Optimized PXP constrained spin flip.

    Flip site_id only if all neighbors are not occupied.
    """

    layout: VariableLayout
    lattice: LatticeGraph
    site_id: int
    coefficient: complex = 1.0
    occupied_value: int = 1
    name: str = "update_pxp_spin_flip"

    def __post_init__(self) -> None:
        site_variable = self.layout.site_variable_index(self.site_id)
        neighbor_sites = self.lattice.neighbors(self.site_id)

        neighbor_variables = np.asarray(
            [self.layout.site_variable_index(int(site)) for site in neighbor_sites],
            dtype=np.int64,
        )

        values = set(self.layout.local_space(site_variable).values.tolist())
        if values != {0, 1}:
            raise ValueError("UpdatePXPSpinFlipOperator requires binary site variables {0, 1}.")

        self.layout.local_space(site_variable).validate_value(self.occupied_value)

        for variable_index in neighbor_variables:
            self.layout.local_space(int(variable_index)).validate_value(self.occupied_value)

        object.__setattr__(self, "_site_variable", site_variable)
        object.__setattr__(self, "_neighbor_sites", neighbor_sites)
        object.__setattr__(self, "_neighbor_variables", neighbor_variables)
        object.__setattr__(
            self,
            "_variable_indices",
            np.asarray([site_variable], dtype=np.int64),
        )
        object.__setattr__(
            self,
            "_value_if_zero",
            np.asarray([1], dtype=np.int64),
        )
        object.__setattr__(
            self,
            "_value_if_one",
            np.asarray([0], dtype=np.int64),
        )

    @property
    def site_variable(self) -> int:
        return int(self._site_variable)

    @property
    def neighbor_sites(self) -> npt.NDArray[np.int64]:
        return self._neighbor_sites.copy()

    @property
    def neighbor_variables(self) -> npt.NDArray[np.int64]:
        return self._neighbor_variables.copy()

    def affected_variables(self) -> npt.NDArray[np.int64]:
        return np.asarray(
            [self._site_variable, *self._neighbor_variables.tolist()],
            dtype=np.int64,
        )

    def single_update(
        self,
        config: npt.ArrayLike,
    ) -> tuple[complex, npt.NDArray[np.int64], npt.NDArray[np.int64]] | None:
        arr = self._as_config(config)

        if np.any(arr[self._neighbor_variables] == self.occupied_value):
            return None

        value = int(arr[self._site_variable])
        new_values = self._value_if_zero if value == 0 else self._value_if_one

        return (self.coefficient, self._variable_indices, new_values)

    def apply_update(self, config: npt.ArrayLike) -> tuple[LocalUpdateAction, ...]:
        update = self.single_update(config)

        if update is None:
            return ()

        coefficient, variable_indices, new_values = update

        return (
            LocalUpdateAction(
                coefficient=coefficient,
                variable_indices=variable_indices,
                new_values=new_values,
            ),
        )
