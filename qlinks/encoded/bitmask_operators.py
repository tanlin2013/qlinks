from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np
import numpy.typing as npt

from qlinks.encoded.binary_basis import bitmask_from_indices
from qlinks.lattice import LatticeGraph
from qlinks.variables import VariableLayout


@dataclass(frozen=True, slots=True)
class BitmaskAction:
    coefficient: complex
    code: int

    def __post_init__(self) -> None:
        if self.code < 0:
            raise ValueError("BitmaskAction.code must be non-negative.")

        object.__setattr__(self, "coefficient", complex(self.coefficient))
        object.__setattr__(self, "code", int(self.code))


class BitmaskOperator(Protocol):
    layout: VariableLayout
    name: str

    def affected_variables(self) -> npt.NDArray[np.int64]: ...

    def apply_code(self, code: int) -> tuple[BitmaskAction, ...]: ...


@dataclass(frozen=True, slots=True)
class BitmaskOperatorSum:
    terms: tuple[BitmaskOperator, ...]
    name: str = "bitmask_operator_sum"

    @classmethod
    def from_terms(
        cls,
        terms: Sequence[BitmaskOperator],
        name: str = "bitmask_operator_sum",
    ) -> BitmaskOperatorSum:
        return cls(terms=tuple(terms), name=name)

    def affected_variables(self) -> npt.NDArray[np.int64]:
        affected: set[int] = set()
        for term in self.terms:
            affected.update(int(i) for i in term.affected_variables())
        return np.asarray(sorted(affected), dtype=np.int64)

    def apply_code(self, code: int) -> tuple[BitmaskAction, ...]:
        actions: list[BitmaskAction] = []
        for term in self.terms:
            actions.extend(term.apply_code(code))
        return tuple(actions)


@dataclass(frozen=True, slots=True)
class BitmaskConstantDiagonalOperator:
    layout: VariableLayout
    coefficient: complex
    name: str = "bitmask_constant_diagonal"

    def affected_variables(self) -> npt.NDArray[np.int64]:
        return np.asarray([], dtype=np.int64)

    def apply_code(self, code: int) -> tuple[BitmaskAction, ...]:
        return (BitmaskAction(self.coefficient, code),)


@dataclass(frozen=True, slots=True)
class BitmaskBinaryFlipOperator:
    """
    Flip one binary variable using XOR:

        code -> code ^ (1 << variable_index)
    """

    layout: VariableLayout
    variable_index: int
    coefficient: complex = 1.0
    name: str = "bitmask_binary_flip"
    _flip_mask: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        values = set(self.layout.local_space(self.variable_index).values.tolist())
        if values != {0, 1}:
            raise ValueError("BitmaskBinaryFlipOperator requires local values {0, 1}.")

        object.__setattr__(self, "_flip_mask", 1 << int(self.variable_index))

    @property
    def flip_mask(self) -> int:
        return int(self._flip_mask)

    def affected_variables(self) -> npt.NDArray[np.int64]:
        return np.asarray([self.variable_index], dtype=np.int64)

    def apply_code(self, code: int) -> tuple[BitmaskAction, ...]:
        return (BitmaskAction(self.coefficient, int(code) ^ self._flip_mask),)


@dataclass(frozen=True, slots=True)
class BitmaskPXPSpinFlipOperator:
    """
    PXP constrained spin flip.

    Flip `site_id` only when all neighboring sites are zero.

    Since the encoded basis is binary, occupied_value is fixed to 1.
    """

    layout: VariableLayout
    lattice: LatticeGraph
    site_id: int
    coefficient: complex = 1.0
    name: str = "bitmask_pxp_spin_flip"

    _site_variable: int = field(init=False, repr=False)
    _neighbor_variables: npt.NDArray[np.int64] = field(init=False, repr=False)
    _flip_mask: int = field(init=False, repr=False)
    _neighbor_mask: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        site_variable = self.layout.site_variable_index(self.site_id)
        neighbor_sites = self.lattice.neighbors(self.site_id)

        neighbor_variables = np.asarray(
            [self.layout.site_variable_index(int(site)) for site in neighbor_sites],
            dtype=np.int64,
        )

        for variable_index in [site_variable, *neighbor_variables.tolist()]:
            values = set(self.layout.local_space(int(variable_index)).values.tolist())
            if values != {0, 1}:
                raise ValueError("BitmaskPXPSpinFlipOperator requires binary site variables.")

        flip_mask = 1 << int(site_variable)
        neighbor_mask = bitmask_from_indices(int(i) for i in neighbor_variables)

        object.__setattr__(self, "_site_variable", int(site_variable))
        object.__setattr__(self, "_neighbor_variables", neighbor_variables)
        object.__setattr__(self, "_flip_mask", flip_mask)
        object.__setattr__(self, "_neighbor_mask", neighbor_mask)

    @property
    def site_variable(self) -> int:
        return int(self._site_variable)

    @property
    def neighbor_mask(self) -> int:
        return int(self._neighbor_mask)

    @property
    def flip_mask(self) -> int:
        return int(self._flip_mask)

    def affected_variables(self) -> npt.NDArray[np.int64]:
        return np.asarray(
            [self._site_variable, *self._neighbor_variables.tolist()],
            dtype=np.int64,
        )

    def apply_code(self, code: int) -> tuple[BitmaskAction, ...]:
        code = int(code)

        if code & self._neighbor_mask:
            return ()

        return (BitmaskAction(self.coefficient, code ^ self._flip_mask),)


@dataclass(frozen=True, slots=True)
class BitmaskPatternFlipOperator:
    """
    General bitmask pattern transition.

    If

        code & mask == initial_bits

    then replace the masked region with final_bits:

        new_code = (code & ~mask) | final_bits
    """

    layout: VariableLayout
    variable_indices: npt.NDArray[np.int64]
    initial_values: npt.NDArray[np.int64]
    final_values: npt.NDArray[np.int64]
    coefficient: complex = 1.0
    name: str = "bitmask_pattern_flip"

    _mask: int = field(init=False, repr=False)
    _initial_bits: int = field(init=False, repr=False)
    _final_bits: int = field(init=False, repr=False)

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
            values = set(self.layout.local_space(int(variable_index)).values.tolist())
            if values != {0, 1}:
                raise ValueError("BitmaskPatternFlipOperator requires binary variables.")

            if int(initial) not in (0, 1):
                raise ValueError("initial_values must be binary.")
            if int(final) not in (0, 1):
                raise ValueError("final_values must be binary.")

        mask = bitmask_from_indices(int(i) for i in variable_indices)

        initial_bits = 0
        final_bits = 0

        for variable_index, initial, final in zip(
            variable_indices,
            initial_values,
            final_values,
            strict=True,
        ):
            bit = 1 << int(variable_index)

            if int(initial) == 1:
                initial_bits |= bit

            if int(final) == 1:
                final_bits |= bit

        object.__setattr__(self, "variable_indices", variable_indices)
        object.__setattr__(self, "initial_values", initial_values)
        object.__setattr__(self, "final_values", final_values)
        object.__setattr__(self, "_mask", mask)
        object.__setattr__(self, "_initial_bits", initial_bits)
        object.__setattr__(self, "_final_bits", final_bits)

    @property
    def mask(self) -> int:
        return int(self._mask)

    @property
    def initial_bits(self) -> int:
        return int(self._initial_bits)

    @property
    def final_bits(self) -> int:
        return int(self._final_bits)

    def affected_variables(self) -> npt.NDArray[np.int64]:
        return self.variable_indices.copy()

    def apply_code(self, code: int) -> tuple[BitmaskAction, ...]:
        code = int(code)

        if (code & self._mask) != self._initial_bits:
            return ()

        new_code = (code & ~self._mask) | self._final_bits

        return (BitmaskAction(self.coefficient, new_code),)


@dataclass(frozen=True, slots=True)
class BitmaskQDMFlipOperator:
    """
    Binary QDM plaquette flip:

        1010 <-> 0101

    The pattern order is the lattice plaquette link order.
    """

    layout: VariableLayout
    lattice: LatticeGraph
    plaquette_id: int
    coefficient: complex = 1.0
    name: str = "bitmask_qdm_flip"

    _variable_indices: npt.NDArray[np.int64] = field(init=False, repr=False)
    _op_1010_to_0101: BitmaskPatternFlipOperator = field(init=False, repr=False)
    _op_0101_to_1010: BitmaskPatternFlipOperator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        link_ids = self.lattice.plaquette_links(self.plaquette_id)

        variable_indices = np.asarray(
            [self.layout.link_variable_index(int(link_id)) for link_id in link_ids],
            dtype=np.int64,
        )

        if variable_indices.size != 4:
            raise ValueError("BitmaskQDMFlipOperator currently expects four-link plaquettes.")

        op_1010_to_0101 = BitmaskPatternFlipOperator(
            layout=self.layout,
            variable_indices=variable_indices,
            initial_values=np.asarray([1, 0, 1, 0], dtype=np.int64),
            final_values=np.asarray([0, 1, 0, 1], dtype=np.int64),
            coefficient=self.coefficient,
        )

        op_0101_to_1010 = BitmaskPatternFlipOperator(
            layout=self.layout,
            variable_indices=variable_indices,
            initial_values=np.asarray([0, 1, 0, 1], dtype=np.int64),
            final_values=np.asarray([1, 0, 1, 0], dtype=np.int64),
            coefficient=self.coefficient,
        )

        object.__setattr__(self, "_variable_indices", variable_indices)
        object.__setattr__(self, "_op_1010_to_0101", op_1010_to_0101)
        object.__setattr__(self, "_op_0101_to_1010", op_0101_to_1010)

    @property
    def variable_indices(self) -> npt.NDArray[np.int64]:
        return self._variable_indices.copy()

    def affected_variables(self) -> npt.NDArray[np.int64]:
        return self._variable_indices.copy()

    def apply_code(self, code: int) -> tuple[BitmaskAction, ...]:
        return self._op_1010_to_0101.apply_code(code) + self._op_0101_to_1010.apply_code(code)


@dataclass(frozen=True, slots=True)
class BitmaskPatternDiagonalOperator:
    """
    Diagonal projector onto a local binary pattern.

    If

        code & mask == pattern_bits

    then return

        coefficient * |code>

    Otherwise return no action.

    This is useful for QDM/QLM flippability potentials.
    """

    layout: VariableLayout
    variable_indices: npt.NDArray[np.int64]
    pattern: npt.NDArray[np.int64]
    coefficient: complex = 1.0
    name: str = "bitmask_pattern_diagonal"

    _mask: int = field(init=False, repr=False)
    _pattern_bits: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        variable_indices = np.asarray(self.variable_indices, dtype=np.int64)
        pattern = np.asarray(self.pattern, dtype=np.int64)

        if variable_indices.ndim != 1:
            raise ValueError("variable_indices must be one-dimensional.")

        if pattern.ndim != 1:
            raise ValueError("pattern must be one-dimensional.")

        if variable_indices.size != pattern.size:
            raise ValueError("variable_indices and pattern must have the same length.")

        if variable_indices.size == 0:
            raise ValueError("At least one variable is required.")

        for variable_index, value in zip(variable_indices, pattern, strict=True):
            values = set(self.layout.local_space(int(variable_index)).values.tolist())
            if values != {0, 1}:
                raise ValueError("BitmaskPatternDiagonalOperator requires binary variables.")

            if int(value) not in (0, 1):
                raise ValueError("pattern values must be binary.")

        mask = bitmask_from_indices(int(i) for i in variable_indices)

        pattern_bits = 0
        for variable_index, value in zip(variable_indices, pattern, strict=True):
            if int(value) == 1:
                pattern_bits |= 1 << int(variable_index)

        object.__setattr__(self, "variable_indices", variable_indices)
        object.__setattr__(self, "pattern", pattern)
        object.__setattr__(self, "_mask", mask)
        object.__setattr__(self, "_pattern_bits", pattern_bits)

    @property
    def mask(self) -> int:
        return int(self._mask)

    @property
    def pattern_bits(self) -> int:
        return int(self._pattern_bits)

    def affected_variables(self) -> npt.NDArray[np.int64]:
        return self.variable_indices.copy()

    def apply_code(self, code: int) -> tuple[BitmaskAction, ...]:
        code = int(code)

        if (code & self._mask) != self._pattern_bits:
            return ()

        return (BitmaskAction(self.coefficient, code),)


def bitmask_qdm_flippability_projectors(
    layout: VariableLayout,
    lattice: LatticeGraph,
    plaquette_id: int,
    coefficient: complex = 1.0,
) -> tuple[BitmaskPatternDiagonalOperator, BitmaskPatternDiagonalOperator]:
    """
    Bitmask projectors onto QDM flippable plaquette patterns:

        1010 and 0101

    in the lattice plaquette link order.
    """

    link_ids = lattice.plaquette_links(plaquette_id)

    variable_indices = np.asarray(
        [layout.link_variable_index(int(link_id)) for link_id in link_ids],
        dtype=np.int64,
    )

    if variable_indices.size != 4:
        raise ValueError("QDM flippability projectors currently expect four-link plaquettes.")

    return (
        BitmaskPatternDiagonalOperator(
            layout=layout,
            variable_indices=variable_indices,
            pattern=np.asarray([1, 0, 1, 0], dtype=np.int64),
            coefficient=coefficient,
            name="bitmask_qdm_flippability_1010",
        ),
        BitmaskPatternDiagonalOperator(
            layout=layout,
            variable_indices=variable_indices,
            pattern=np.asarray([0, 1, 0, 1], dtype=np.int64),
            coefficient=coefficient,
            name="bitmask_qdm_flippability_0101",
        ),
    )


def binary_pattern_from_flux_pattern(
    flux_pattern: npt.NDArray[np.int64],
) -> npt.NDArray[np.int64]:
    """Convert physical QLM flux values {-1, +1} to binary values {0, 1}."""
    flux_pattern = np.asarray(flux_pattern, dtype=np.int64)

    if not np.all(np.isin(flux_pattern, [-1, 1])):
        raise ValueError("flux_pattern must contain only -1 and +1.")

    return ((flux_pattern + 1) // 2).astype(np.int64)


def bitmask_qlm_flippability_projectors(
    layout: VariableLayout,
    lattice: LatticeGraph,
    plaquette_id: int,
    coefficient: complex = 1.0,
) -> tuple[BitmaskPatternDiagonalOperator, BitmaskPatternDiagonalOperator]:
    """Bitmask projectors for spin-1/2 QLM flippable flux plaquettes.

    The physical QLM flippable flux patterns are the oriented plaquette
    boundary pattern and its negative. With binary convention

        -1 -> 0
        +1 -> 1

    these become binary_pattern and 1 - binary_pattern.
    """
    link_ids = lattice.plaquette_links(plaquette_id)
    variable_indices = np.asarray(
        [
            layout.link_variable_index(int(link_id))
            for link_id in link_ids
        ],
        dtype=np.int64,
    )

    orientation_pattern = np.asarray(
        lattice.plaquette_orientations(int(plaquette_id)),
        dtype=np.int64,
    )

    if variable_indices.size != orientation_pattern.size:
        raise ValueError(
            "Plaquette links and orientations must have the same length."
        )

    binary_pattern = binary_pattern_from_flux_pattern(orientation_pattern)

    return (
        BitmaskPatternDiagonalOperator(
            layout=layout,
            variable_indices=variable_indices,
            pattern=binary_pattern,
            coefficient=coefficient,
            name="bitmask_qlm_flippability_oriented",
        ),
        BitmaskPatternDiagonalOperator(
            layout=layout,
            variable_indices=variable_indices,
            pattern=1 - binary_pattern,
            coefficient=coefficient,
            name="bitmask_qlm_flippability_reversed",
        ),
    )


@dataclass(frozen=True, slots=True)
class BitmaskQLMFluxFlipOperator:
    """Spin-1/2 QLM plaquette ring exchange in binary flux encoding.

    Physical convention:
        -1 -> 0
        +1 -> 1

    The flippable QLM flux patterns are given by the plaquette orientation
    signs and their negatives.
    """

    layout: VariableLayout
    lattice: LatticeGraph
    plaquette_id: int
    coefficient: complex = 1.0
    name: str = "bitmask_qlm_flux_flip"

    _variable_indices: npt.NDArray[np.int64] = field(init=False, repr=False)
    _op_oriented_to_reversed: BitmaskPatternFlipOperator = field(
        init=False,
        repr=False,
    )
    _op_reversed_to_oriented: BitmaskPatternFlipOperator = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        link_ids = self.lattice.plaquette_links(self.plaquette_id)
        variable_indices = np.asarray(
            [
                self.layout.link_variable_index(int(link_id))
                for link_id in link_ids
            ],
            dtype=np.int64,
        )

        orientation_pattern = np.asarray(
            self.lattice.plaquette_orientations(int(self.plaquette_id)),
            dtype=np.int64,
        )

        if variable_indices.size != orientation_pattern.size:
            raise ValueError(
                "Plaquette links and orientations must have the same length."
            )

        binary_pattern = binary_pattern_from_flux_pattern(orientation_pattern)
        reversed_binary_pattern = 1 - binary_pattern

        op_oriented_to_reversed = BitmaskPatternFlipOperator(
            layout=self.layout,
            variable_indices=variable_indices,
            initial_values=binary_pattern,
            final_values=reversed_binary_pattern,
            coefficient=self.coefficient,
            name="bitmask_qlm_flux_flip_oriented_to_reversed",
        )
        op_reversed_to_oriented = BitmaskPatternFlipOperator(
            layout=self.layout,
            variable_indices=variable_indices,
            initial_values=reversed_binary_pattern,
            final_values=binary_pattern,
            coefficient=self.coefficient,
            name="bitmask_qlm_flux_flip_reversed_to_oriented",
        )

        object.__setattr__(self, "_variable_indices", variable_indices)
        object.__setattr__(
            self,
            "_op_oriented_to_reversed",
            op_oriented_to_reversed,
        )
        object.__setattr__(
            self,
            "_op_reversed_to_oriented",
            op_reversed_to_oriented,
        )

    @property
    def variable_indices(self) -> npt.NDArray[np.int64]:
        return self._variable_indices.copy()

    def affected_variables(self) -> npt.NDArray[np.int64]:
        return self._variable_indices.copy()

    def apply_code(self, code: int) -> tuple[BitmaskAction, ...]:
        return (
            self._op_oriented_to_reversed.apply_code(code)
            + self._op_reversed_to_oriented.apply_code(code)
        )


def bitmask_alternating_flippability_projectors(
    layout: VariableLayout,
    lattice: LatticeGraph,
    plaquette_id: int,
    coefficient: complex = 1.0,
) -> tuple[BitmaskPatternDiagonalOperator, BitmaskPatternDiagonalOperator]:
    link_ids = lattice.plaquette_links(plaquette_id)

    variable_indices = np.asarray(
        [layout.link_variable_index(int(link_id)) for link_id in link_ids],
        dtype=np.int64,
    )

    length = variable_indices.size

    if length % 2 != 0:
        raise ValueError("Alternating bitmask projectors require even-length plaquettes.")

    p0 = np.asarray([1 if i % 2 == 0 else 0 for i in range(length)], dtype=np.int64)
    p1 = 1 - p0

    return (
        BitmaskPatternDiagonalOperator(
            layout=layout,
            variable_indices=variable_indices,
            pattern=p0,
            coefficient=coefficient,
            name="bitmask_alternating_flippability_0",
        ),
        BitmaskPatternDiagonalOperator(
            layout=layout,
            variable_indices=variable_indices,
            pattern=p1,
            coefficient=coefficient,
            name="bitmask_alternating_flippability_1",
        ),
    )


@dataclass(frozen=True, slots=True)
class BitmaskAlternatingPlaquetteFlipOperator:
    """
    Generic binary alternating plaquette flip:

        1010... <-> 0101...

    Works for any even-length plaquette.
    """

    layout: VariableLayout
    lattice: LatticeGraph
    plaquette_id: int
    coefficient: complex = 1.0
    name: str = "bitmask_alternating_plaquette_flip"

    _variable_indices: npt.NDArray[np.int64] = field(init=False, repr=False)
    _op_0_to_1: BitmaskPatternFlipOperator = field(init=False, repr=False)
    _op_1_to_0: BitmaskPatternFlipOperator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        link_ids = self.lattice.plaquette_links(self.plaquette_id)

        variable_indices = np.asarray(
            [self.layout.link_variable_index(int(link_id)) for link_id in link_ids],
            dtype=np.int64,
        )

        length = variable_indices.size

        if length % 2 != 0:
            raise ValueError("BitmaskAlternatingPlaquetteFlipOperator requires even length.")

        p0 = np.asarray([1 if i % 2 == 0 else 0 for i in range(length)], dtype=np.int64)
        p1 = 1 - p0

        op_0_to_1 = BitmaskPatternFlipOperator(
            layout=self.layout,
            variable_indices=variable_indices,
            initial_values=p0,
            final_values=p1,
            coefficient=self.coefficient,
        )

        op_1_to_0 = BitmaskPatternFlipOperator(
            layout=self.layout,
            variable_indices=variable_indices,
            initial_values=p1,
            final_values=p0,
            coefficient=self.coefficient,
        )

        object.__setattr__(self, "_variable_indices", variable_indices)
        object.__setattr__(self, "_op_0_to_1", op_0_to_1)
        object.__setattr__(self, "_op_1_to_0", op_1_to_0)

    @property
    def variable_indices(self) -> npt.NDArray[np.int64]:
        return self._variable_indices.copy()

    def affected_variables(self) -> npt.NDArray[np.int64]:
        return self._variable_indices.copy()

    def apply_code(self, code: int) -> tuple[BitmaskAction, ...]:
        return self._op_0_to_1.apply_code(code) + self._op_1_to_0.apply_code(code)
