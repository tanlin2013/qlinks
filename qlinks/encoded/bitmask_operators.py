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
    """One encoded-basis matrix action.

    Attributes:
        coefficient: Matrix element coefficient.
        code: Target encoded configuration.
    """

    coefficient: complex
    code: int

    def __post_init__(self) -> None:
        if self.code < 0:
            raise ValueError("BitmaskAction.code must be non-negative.")

        object.__setattr__(self, "coefficient", complex(self.coefficient))
        object.__setattr__(self, "code", int(self.code))


class BitmaskOperator(Protocol):
    """Protocol for operators acting on integer-encoded binary states."""

    layout: VariableLayout
    name: str

    def affected_variables(self) -> npt.NDArray[np.int64]: ...

    def apply_code(self, code: int) -> tuple[BitmaskAction, ...]: ...


class BitmaskDiagonalOperator(BitmaskOperator, Protocol):
    """Configuration-space diagonal operator in binary bitmask encoding.

    Returning ``None`` means the operator gives no diagonal contribution for
    this code. Returning a complex number means the operator contributes that
    diagonal matrix element.
    """

    def diagonal_value_code(self, code: int) -> complex | None: ...


class BitmaskSingleActionOperator(BitmaskOperator, Protocol):
    """Operator that produces at most one non-diagonal action per input code.

    Returning ``None`` means the operator has no action on this code. Returning
    ``(coefficient, new_code)`` means the operator contributes one matrix
    element without allocating a :class:`BitmaskAction`.
    """

    def single_action_code(self, code: int) -> tuple[complex, int] | None: ...


def _bits_from_binary_pattern(
    *,
    variable_indices: npt.NDArray[np.int64],
    pattern: npt.NDArray[np.int64],
) -> int:
    """Return encoded bits for a binary local pattern on variable_indices."""
    bits = 0

    for variable_index, value in zip(variable_indices, pattern, strict=True):
        if int(value) == 1:
            bits |= 1 << int(variable_index)

    return int(bits)


def _single_pattern_action_code(
    *,
    code: int,
    mask: int,
    initial_bits: int,
    final_bits: int,
    coefficient: complex,
) -> tuple[complex, int] | None:
    code = int(code)

    if (code & mask) != initial_bits:
        return None

    new_code = (code & ~mask) | final_bits

    return complex(coefficient), int(new_code)


@dataclass(frozen=True, slots=True)
class BitmaskOperatorSum:
    """Sum of bitmask operators presented as one operator.

    Attributes:
        terms: Bitmask operators to apply and concatenate.
        name: Human-readable operator name.
    """

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
    """Constant diagonal operator in bitmask representation.

    Attributes:
        layout: Binary variable layout.
        coefficient: Constant diagonal matrix element.
        name: Human-readable operator name.
    """

    layout: VariableLayout
    coefficient: complex
    name: str = "bitmask_constant_diagonal"

    def affected_variables(self) -> npt.NDArray[np.int64]:
        return np.asarray([], dtype=np.int64)

    def diagonal_value_code(self, code: int) -> complex | None:
        code = int(code)
        if code < 0:
            raise ValueError("code must be non-negative.")

        return complex(self.coefficient)

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

    def single_action_code(self, code: int) -> tuple[complex, int] | None:
        return complex(self.coefficient), int(code) ^ self._flip_mask

    def apply_code(self, code: int) -> tuple[BitmaskAction, ...]:
        action = self.single_action_code(code)
        assert action is not None
        coefficient, new_code = action
        return (BitmaskAction(coefficient, new_code),)


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

    def single_action_code(self, code: int) -> tuple[complex, int] | None:
        code = int(code)

        if code & self._neighbor_mask:
            return None

        return complex(self.coefficient), code ^ self._flip_mask

    def apply_code(self, code: int) -> tuple[BitmaskAction, ...]:
        action = self.single_action_code(code)

        if action is None:
            return ()

        coefficient, new_code = action
        return (BitmaskAction(coefficient, new_code),)


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

    def single_action_code(self, code: int) -> tuple[complex, int] | None:
        code = int(code)

        if (code & self._mask) != self._initial_bits:
            return None

        new_code = (code & ~self._mask) | self._final_bits

        return complex(self.coefficient), new_code

    def apply_code(self, code: int) -> tuple[BitmaskAction, ...]:
        action = self.single_action_code(code)

        if action is None:
            return ()

        coefficient, new_code = action
        return (BitmaskAction(coefficient, new_code),)


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
    reverse_coefficient: complex | None = None
    name: str = "bitmask_qdm_flip"

    _variable_indices: npt.NDArray[np.int64] = field(init=False, repr=False)
    _mask: int = field(init=False, repr=False)
    _forward_initial_bits: int = field(init=False, repr=False)
    _forward_final_bits: int = field(init=False, repr=False)
    _reverse_initial_bits: int = field(init=False, repr=False)
    _reverse_final_bits: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        link_ids = self.lattice.plaquette_links(self.plaquette_id)

        variable_indices = np.asarray(
            [self.layout.link_variable_index(int(link_id)) for link_id in link_ids],
            dtype=np.int64,
        )

        if variable_indices.size != 4:
            raise ValueError("BitmaskQDMFlipOperator currently expects four-link plaquettes.")

        if self.reverse_coefficient is None:
            object.__setattr__(
                self,
                "reverse_coefficient",
                complex(self.coefficient).conjugate(),
            )

        forward_initial = np.asarray([1, 0, 1, 0], dtype=np.int64)
        forward_final = np.asarray([0, 1, 0, 1], dtype=np.int64)

        object.__setattr__(self, "_variable_indices", variable_indices)
        object.__setattr__(self, "_mask", bitmask_from_indices(int(i) for i in variable_indices))
        object.__setattr__(
            self,
            "_forward_initial_bits",
            _bits_from_binary_pattern(
                variable_indices=variable_indices,
                pattern=forward_initial,
            ),
        )
        object.__setattr__(
            self,
            "_forward_final_bits",
            _bits_from_binary_pattern(
                variable_indices=variable_indices,
                pattern=forward_final,
            ),
        )
        object.__setattr__(self, "_reverse_initial_bits", self._forward_final_bits)
        object.__setattr__(self, "_reverse_final_bits", self._forward_initial_bits)

    @property
    def variable_indices(self) -> npt.NDArray[np.int64]:
        return self._variable_indices.copy()

    @property
    def mask(self) -> int:
        return int(self._mask)

    def affected_variables(self) -> npt.NDArray[np.int64]:
        return self._variable_indices.copy()

    def single_action_code(self, code: int) -> tuple[complex, int] | None:
        action = _single_pattern_action_code(
            code=code,
            mask=self._mask,
            initial_bits=self._forward_initial_bits,
            final_bits=self._forward_final_bits,
            coefficient=self.coefficient,
        )

        if action is not None:
            return action

        return _single_pattern_action_code(
            code=code,
            mask=self._mask,
            initial_bits=self._reverse_initial_bits,
            final_bits=self._reverse_final_bits,
            coefficient=complex(self.reverse_coefficient),
        )

    def apply_code(self, code: int) -> tuple[BitmaskAction, ...]:
        action = self.single_action_code(code)

        if action is None:
            return ()

        coefficient, new_code = action
        return (BitmaskAction(coefficient, new_code),)


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

    def diagonal_value_code(self, code: int) -> complex | None:
        code = int(code)

        if (code & self._mask) != self._pattern_bits:
            return None

        return complex(self.coefficient)

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
        [layout.link_variable_index(int(link_id)) for link_id in link_ids],
        dtype=np.int64,
    )

    orientation_pattern = np.asarray(
        lattice.plaquette_orientations(int(plaquette_id)),
        dtype=np.int64,
    )

    if variable_indices.size != orientation_pattern.size:
        raise ValueError("Plaquette links and orientations must have the same length.")

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
    reverse_coefficient: complex | None = None
    name: str = "bitmask_qlm_flux_flip"

    _variable_indices: npt.NDArray[np.int64] = field(init=False, repr=False)
    _mask: int = field(init=False, repr=False)
    _oriented_bits: int = field(init=False, repr=False)
    _reversed_bits: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        link_ids = self.lattice.plaquette_links(self.plaquette_id)
        variable_indices = np.asarray(
            [self.layout.link_variable_index(int(link_id)) for link_id in link_ids],
            dtype=np.int64,
        )

        orientation_pattern = np.asarray(
            self.lattice.plaquette_orientations(int(self.plaquette_id)),
            dtype=np.int64,
        )

        if variable_indices.size != orientation_pattern.size:
            raise ValueError("Plaquette links and orientations must have the same length.")

        if self.reverse_coefficient is None:
            object.__setattr__(
                self,
                "reverse_coefficient",
                complex(self.coefficient).conjugate(),
            )

        binary_pattern = binary_pattern_from_flux_pattern(orientation_pattern)
        reversed_binary_pattern = 1 - binary_pattern

        object.__setattr__(self, "_variable_indices", variable_indices)
        object.__setattr__(self, "_mask", bitmask_from_indices(int(i) for i in variable_indices))
        object.__setattr__(
            self,
            "_oriented_bits",
            _bits_from_binary_pattern(
                variable_indices=variable_indices,
                pattern=binary_pattern,
            ),
        )
        object.__setattr__(
            self,
            "_reversed_bits",
            _bits_from_binary_pattern(
                variable_indices=variable_indices,
                pattern=reversed_binary_pattern,
            ),
        )

    @property
    def variable_indices(self) -> npt.NDArray[np.int64]:
        return self._variable_indices.copy()

    @property
    def mask(self) -> int:
        return int(self._mask)

    def affected_variables(self) -> npt.NDArray[np.int64]:
        return self._variable_indices.copy()

    def single_action_code(self, code: int) -> tuple[complex, int] | None:
        action = _single_pattern_action_code(
            code=code,
            mask=self._mask,
            initial_bits=self._oriented_bits,
            final_bits=self._reversed_bits,
            coefficient=self.coefficient,
        )

        if action is not None:
            return action

        return _single_pattern_action_code(
            code=code,
            mask=self._mask,
            initial_bits=self._reversed_bits,
            final_bits=self._oriented_bits,
            coefficient=complex(self.reverse_coefficient),
        )

    def apply_code(self, code: int) -> tuple[BitmaskAction, ...]:
        action = self.single_action_code(code)

        if action is None:
            return ()

        coefficient, new_code = action
        return (BitmaskAction(coefficient, new_code),)


def bitmask_alternating_flippability_projectors(
    layout: VariableLayout,
    lattice: LatticeGraph,
    plaquette_id: int,
    coefficient: complex = 1.0,
) -> tuple[BitmaskPatternDiagonalOperator, BitmaskPatternDiagonalOperator]:
    """Return the two alternating-pattern projectors for a binary plaquette.

    Args:
        layout: Binary variable layout.
        lattice: Lattice containing the plaquette.
        plaquette_id: Plaquette id.
        coefficient: Diagonal coefficient for each projector.

    Returns:
        Pair of bitmask diagonal projectors onto the two alternating patterns.
    """
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
    reverse_coefficient: complex | None = None
    name: str = "bitmask_alternating_plaquette_flip"

    _variable_indices: npt.NDArray[np.int64] = field(init=False, repr=False)
    _mask: int = field(init=False, repr=False)
    _p0_bits: int = field(init=False, repr=False)
    _p1_bits: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        link_ids = self.lattice.plaquette_links(self.plaquette_id)

        variable_indices = np.asarray(
            [self.layout.link_variable_index(int(link_id)) for link_id in link_ids],
            dtype=np.int64,
        )

        length = variable_indices.size

        if length % 2 != 0:
            raise ValueError("BitmaskAlternatingPlaquetteFlipOperator requires even length.")

        if self.reverse_coefficient is None:
            object.__setattr__(
                self,
                "reverse_coefficient",
                complex(self.coefficient).conjugate(),
            )

        p0 = np.asarray([1 if i % 2 == 0 else 0 for i in range(length)], dtype=np.int64)
        p1 = 1 - p0

        object.__setattr__(self, "_variable_indices", variable_indices)
        object.__setattr__(self, "_mask", bitmask_from_indices(int(i) for i in variable_indices))
        object.__setattr__(
            self,
            "_p0_bits",
            _bits_from_binary_pattern(
                variable_indices=variable_indices,
                pattern=p0,
            ),
        )
        object.__setattr__(
            self,
            "_p1_bits",
            _bits_from_binary_pattern(
                variable_indices=variable_indices,
                pattern=p1,
            ),
        )

    @property
    def variable_indices(self) -> npt.NDArray[np.int64]:
        return self._variable_indices.copy()

    @property
    def mask(self) -> int:
        return int(self._mask)

    def affected_variables(self) -> npt.NDArray[np.int64]:
        return self._variable_indices.copy()

    def single_action_code(self, code: int) -> tuple[complex, int] | None:
        action = _single_pattern_action_code(
            code=code,
            mask=self._mask,
            initial_bits=self._p0_bits,
            final_bits=self._p1_bits,
            coefficient=self.coefficient,
        )

        if action is not None:
            return action

        return _single_pattern_action_code(
            code=code,
            mask=self._mask,
            initial_bits=self._p1_bits,
            final_bits=self._p0_bits,
            coefficient=complex(self.reverse_coefficient),
        )

    def apply_code(self, code: int) -> tuple[BitmaskAction, ...]:
        action = self.single_action_code(code)

        if action is None:
            return ()

        coefficient, new_code = action
        return (BitmaskAction(coefficient, new_code),)
