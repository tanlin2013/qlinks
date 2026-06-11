import numpy as np

from qlinks.encoded import (
    BitmaskAlternatingPlaquetteFlipOperator,
    BitmaskBinaryFlipOperator,
    BitmaskConstantDiagonalOperator,
    BitmaskOperatorSum,
    BitmaskPatternDiagonalOperator,
    BitmaskPatternFlipOperator,
    BitmaskPXPSpinFlipOperator,
    BitmaskQDMFlipOperator,
    BitmaskQLMFluxFlipOperator,
    bitmask_qdm_flippability_projectors,
    encode_binary_config,
)
from qlinks.lattice import (
    ChainLattice,
    HoneycombLattice,
    SquareLattice,
    TriangularLattice,
)
from qlinks.variables import LocalSpace, VariableLayout


def test_bitmask_constant_diagonal_operator() -> None:
    layout = VariableLayout.from_sites(2, LocalSpace.binary())
    op = BitmaskConstantDiagonalOperator(layout=layout, coefficient=3.0)

    actions = op.apply_code(2)

    assert len(actions) == 1
    assert actions[0].coefficient == 3.0 + 0j
    assert actions[0].code == 2


def test_bitmask_binary_flip_operator() -> None:
    layout = VariableLayout.from_sites(3, LocalSpace.binary())

    op = BitmaskBinaryFlipOperator(layout=layout, variable_index=1)

    # 101 -> flip bit 1 -> 111
    code = encode_binary_config(np.array([1, 0, 1]))

    actions = op.apply_code(code)

    assert len(actions) == 1
    assert actions[0].code == encode_binary_config(np.array([1, 1, 1]))


def test_bitmask_operator_sum() -> None:
    layout = VariableLayout.from_sites(2, LocalSpace.binary())

    op_sum = BitmaskOperatorSum.from_terms(
        [
            BitmaskBinaryFlipOperator(layout=layout, variable_index=0),
            BitmaskBinaryFlipOperator(layout=layout, variable_index=1),
        ]
    )

    actions = op_sum.apply_code(0)

    assert {a.code for a in actions} == {1, 2}


def test_bitmask_pxp_spin_flip_allowed() -> None:
    lattice = ChainLattice(3, boundary_condition="open")
    layout = VariableLayout.from_lattice_sites(lattice, LocalSpace.binary())

    op = BitmaskPXPSpinFlipOperator(layout=layout, lattice=lattice, site_id=1)

    code = encode_binary_config(np.array([0, 0, 0]))

    actions = op.apply_code(code)

    assert len(actions) == 1
    assert actions[0].code == encode_binary_config(np.array([0, 1, 0]))


def test_bitmask_pxp_spin_flip_blocked() -> None:
    lattice = ChainLattice(3, boundary_condition="open")
    layout = VariableLayout.from_lattice_sites(lattice, LocalSpace.binary())

    op = BitmaskPXPSpinFlipOperator(layout=layout, lattice=lattice, site_id=1)

    left_blocked = encode_binary_config(np.array([1, 0, 0]))
    right_blocked = encode_binary_config(np.array([0, 0, 1]))

    assert op.apply_code(left_blocked) == ()
    assert op.apply_code(right_blocked) == ()


def test_bitmask_pattern_flip() -> None:
    layout = VariableLayout.from_sites(4, LocalSpace.binary())

    op = BitmaskPatternFlipOperator(
        layout=layout,
        variable_indices=np.array([0, 2]),
        initial_values=np.array([1, 0]),
        final_values=np.array([0, 1]),
        coefficient=-1.0,
    )

    code = encode_binary_config(np.array([1, 1, 0, 1]))
    actions = op.apply_code(code)

    assert len(actions) == 1
    assert actions[0].coefficient == -1.0 + 0j
    assert actions[0].code == encode_binary_config(np.array([0, 1, 1, 1]))


def test_bitmask_pattern_flip_no_match() -> None:
    layout = VariableLayout.from_sites(4, LocalSpace.binary())

    op = BitmaskPatternFlipOperator(
        layout=layout,
        variable_indices=np.array([0, 2]),
        initial_values=np.array([1, 0]),
        final_values=np.array([0, 1]),
    )

    code = encode_binary_config(np.array([0, 1, 0, 1]))

    assert op.apply_code(code) == ()


def test_bitmask_qdm_flip_single_square() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.binary())

    op = BitmaskQDMFlipOperator(
        layout=layout,
        lattice=lattice,
        plaquette_id=0,
        coefficient=-1.0,
    )

    code_1010 = encode_binary_config(np.array([1, 0, 1, 0]))
    code_0101 = encode_binary_config(np.array([0, 1, 0, 1]))

    actions = op.apply_code(code_1010)

    assert len(actions) == 1
    assert actions[0].coefficient == -1.0 + 0j
    assert actions[0].code == code_0101

    reverse = op.apply_code(code_0101)

    assert len(reverse) == 1
    assert reverse[0].code == code_1010


def test_bitmask_pattern_diagonal_matches() -> None:
    layout = VariableLayout.from_sites(4, LocalSpace.binary())

    op = BitmaskPatternDiagonalOperator(
        layout=layout,
        variable_indices=np.array([0, 2]),
        pattern=np.array([1, 0]),
        coefficient=3.0,
    )

    code = encode_binary_config(np.array([1, 1, 0, 1]))

    actions = op.apply_code(code)

    assert len(actions) == 1
    assert actions[0].coefficient == 3.0 + 0j
    assert actions[0].code == code


def test_bitmask_pattern_diagonal_no_match() -> None:
    layout = VariableLayout.from_sites(4, LocalSpace.binary())

    op = BitmaskPatternDiagonalOperator(
        layout=layout,
        variable_indices=np.array([0, 2]),
        pattern=np.array([1, 0]),
    )

    code = encode_binary_config(np.array([0, 1, 0, 1]))

    assert op.apply_code(code) == ()


def test_bitmask_qdm_flippability_projectors() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.binary())

    projectors = bitmask_qdm_flippability_projectors(
        layout=layout,
        lattice=lattice,
        plaquette_id=0,
        coefficient=2.0,
    )

    code_1010 = encode_binary_config(np.array([1, 0, 1, 0]))
    code_0101 = encode_binary_config(np.array([0, 1, 0, 1]))
    code_other = encode_binary_config(np.array([1, 1, 0, 0]))

    assert len(projectors[0].apply_code(code_1010)) == 1
    assert projectors[0].apply_code(code_1010)[0].coefficient == 2.0 + 0j

    assert len(projectors[1].apply_code(code_0101)) == 1
    assert projectors[1].apply_code(code_0101)[0].coefficient == 2.0 + 0j

    assert projectors[0].apply_code(code_other) == ()
    assert projectors[1].apply_code(code_other) == ()


def _code_with_local_pattern(
    *,
    n_variables: int,
    variable_indices: np.ndarray,
    pattern: np.ndarray,
) -> int:
    config = np.zeros(n_variables, dtype=np.int64)
    config[variable_indices] = pattern
    return encode_binary_config(config)


def test_bitmask_alternating_plaquette_flip_single_action_matches_apply_code() -> None:
    lattice = TriangularLattice(2, 2, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.binary())
    plaquette_id = lattice.qdm_plaquette_ids()[0]

    op = BitmaskAlternatingPlaquetteFlipOperator(
        layout=layout,
        lattice=lattice,
        plaquette_id=int(plaquette_id),
        coefficient=-2.0,
    )

    pattern = np.asarray([1 if i % 2 == 0 else 0 for i in range(op.variable_indices.size)])
    code = _code_with_local_pattern(
        n_variables=layout.n_variables,
        variable_indices=op.variable_indices,
        pattern=pattern,
    )

    single_action = op.single_action_code(code)
    assert single_action is not None

    apply_action = op.apply_code(code)[0]
    coefficient, new_code = single_action

    assert coefficient == apply_action.coefficient
    assert new_code == apply_action.code


def test_bitmask_qlm_flux_flip_single_action_matches_apply_code_on_hexagon() -> None:
    lattice = HoneycombLattice(2, 2, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.binary())
    plaquette_id = lattice.qlm_plaquette_ids()[0]

    op = BitmaskQLMFluxFlipOperator(
        layout=layout,
        lattice=lattice,
        plaquette_id=int(plaquette_id),
        coefficient=-1.5,
    )

    orientation_pattern = np.asarray(lattice.plaquette_orientations(int(plaquette_id)))
    binary_pattern = ((orientation_pattern + 1) // 2).astype(np.int64)
    code = _code_with_local_pattern(
        n_variables=layout.n_variables,
        variable_indices=op.variable_indices,
        pattern=binary_pattern,
    )

    single_action = op.single_action_code(code)
    assert single_action is not None

    apply_action = op.apply_code(code)[0]
    coefficient, new_code = single_action

    assert coefficient == apply_action.coefficient
    assert new_code == apply_action.code
