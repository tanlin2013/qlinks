from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pytest

from qlinks.basis import Basis, full_basis_from_layout
from qlinks.models import ToricCodeModel
from qlinks.qec import (
    CodeSpace,
    ErrorOperator,
    LocalErrorSet,
    diagnose_knill_laflamme,
    diagnose_local_indistinguishability,
    diagnose_projected_error_algebra,
    search_projected_logical_operators,
)
from qlinks.variables import VariableLayout


@dataclass(frozen=True, slots=True)
class _DenseIdentity:
    """Small matrix-like identity used to keep support metadata empty."""

    name: str = "I"

    def __matmul__(self, vectors: npt.ArrayLike) -> npt.NDArray[np.complex128]:
        return np.asarray(vectors, dtype=np.complex128).copy()


@dataclass(frozen=True, slots=True)
class _DensePauliX:
    """Single-link X operator in the toric-code Z basis."""

    permutation: npt.NDArray[np.int64]
    variable_index: int
    name: str

    def __matmul__(self, vectors: npt.ArrayLike) -> npt.NDArray[np.complex128]:
        x = np.asarray(vectors, dtype=np.complex128)
        y = np.empty_like(x, dtype=np.complex128)
        y[self.permutation, ...] = x
        return y


@dataclass(frozen=True, slots=True)
class _DensePauliZ:
    """Single-link or string Z operator in the toric-code Z basis."""

    signs: npt.NDArray[np.int8]
    name: str

    def __matmul__(self, vectors: npt.ArrayLike) -> npt.NDArray[np.complex128]:
        x = np.asarray(vectors, dtype=np.complex128)
        return self.signs.reshape((-1,) + (1,) * (x.ndim - 1)) * x


@dataclass(frozen=True, slots=True)
class _ToricCodeFixture:
    model: ToricCodeModel
    layout: VariableLayout
    basis: Basis
    code: CodeSpace
    local_errors: LocalErrorSet
    logical_z_loop: ErrorOperator


def _basis_index_for_spin_half_config(config: npt.ArrayLike) -> int:
    arr = np.asarray(config, dtype=np.int64)
    bits = ((arr + 1) // 2).astype(np.int64)
    index = 0
    for bit in bits:
        index = (index << 1) | int(bit)
    return int(index)


def _star_variable_indices(model: ToricCodeModel) -> tuple[npt.NDArray[np.int64], ...]:
    layout = model.layout
    return tuple(
        np.asarray(
            [
                layout.link_variable_index(int(link_id))
                for link_id in model.lattice.incident_links(site_id)
            ],
            dtype=np.int64,
        )
        for site_id in model.lattice.site_ids
    )


def _noncontractible_seed(model: ToricCodeModel, *, z_string_a: bool, z_string_b: bool):
    """Return a B_p=+1 representative in a chosen toric-code topological sector."""
    lattice = model.lattice
    layout = model.layout
    site_cells = {site.id: tuple(int(x) for x in site.cell) for site in lattice.sites}
    config = np.ones(layout.n_variables, dtype=np.int64)

    # A column of x-oriented links and a row of y-oriented links are closed
    # noncontractible Z strings on the dual torus.  Each plaquette sees either
    # zero or two negative boundary links, so B_p remains +1.
    for link in lattice.links:
        x, y = site_cells[link.source]
        if z_string_a and link.kind == "x" and x == 0:
            config[layout.link_variable_index(link.id)] *= -1
        if z_string_b and link.kind == "y" and y == 0:
            config[layout.link_variable_index(link.id)] *= -1

    for plaquette in lattice.plaquettes:
        variables = [layout.link_variable_index(int(link_id)) for link_id in plaquette.links]
        assert int(np.prod(config[variables])) == 1

    return config


def _star_orbit_indices(model: ToricCodeModel, seed: npt.ArrayLike) -> tuple[int, ...]:
    config0 = np.asarray(seed, dtype=np.int64)
    stars = _star_variable_indices(model)
    # The product of all star flips is identity, so omit one dependent star.
    independent_stars = stars[:-1]
    indices: set[int] = set()

    for mask in range(1 << len(independent_stars)):
        config = config0.copy()
        for bit, star_variables in enumerate(independent_stars):
            if (mask >> bit) & 1:
                config[star_variables] *= -1
        indices.add(_basis_index_for_spin_half_config(config))

    return tuple(sorted(indices))


def _toric_code_ground_space(model: ToricCodeModel, basis: Basis) -> CodeSpace:
    sectors = ((False, False), (True, False), (False, True), (True, True))
    vectors = np.zeros((basis.n_states, len(sectors)), dtype=np.complex128)
    seen: set[int] = set()

    for column, (z_string_a, z_string_b) in enumerate(sectors):
        seed = _noncontractible_seed(model, z_string_a=z_string_a, z_string_b=z_string_b)
        support = _star_orbit_indices(model, seed)
        assert len(support) == 2 ** (model.lattice.num_sites - 1)
        assert seen.isdisjoint(support)
        seen.update(support)
        vectors[np.asarray(support, dtype=np.int64), column] = 1.0 / np.sqrt(len(support))

    return CodeSpace.from_vectors(
        basis,
        vectors,
        labels=("00", "10", "01", "11"),
        orthonormalize=False,
    )


def _flip_permutation(basis: Basis, variable_index: int) -> npt.NDArray[np.int64]:
    n_variables = basis.n_variables
    bit_mask = 1 << (n_variables - 1 - int(variable_index))
    return np.bitwise_xor(np.arange(basis.n_states, dtype=np.int64), bit_mask)


def _z_signs(basis: Basis, variable_indices: tuple[int, ...]) -> npt.NDArray[np.int8]:
    signs = np.prod(basis.states[:, variable_indices], axis=1)
    return np.asarray(signs, dtype=np.int8)


def _single_link_error_set(basis: Basis, variable_indices: tuple[int, ...]) -> LocalErrorSet:
    errors: list[ErrorOperator] = [
        ErrorOperator(
            name="I",
            operator=_DenseIdentity(),
            support_variables=(),
            kind="identity",
        )
    ]

    for variable_index in variable_indices:
        errors.append(
            ErrorOperator(
                name=f"X_{variable_index}",
                operator=_DensePauliX(
                    permutation=_flip_permutation(basis, variable_index),
                    variable_index=variable_index,
                    name=f"X_{variable_index}",
                ),
                support_variables=(variable_index,),
                kind="pauli_x",
            )
        )
        errors.append(
            ErrorOperator(
                name=f"Z_{variable_index}",
                operator=_DensePauliZ(
                    signs=_z_signs(basis, (variable_index,)),
                    name=f"Z_{variable_index}",
                ),
                support_variables=(variable_index,),
                kind="pauli_z",
            )
        )

    return LocalErrorSet(tuple(errors), name="toric_code_single_link_paulis")


def _logical_z_loop_operator(model: ToricCodeModel, basis: Basis) -> ErrorOperator:
    site_cells = {site.id: tuple(int(x) for x in site.cell) for site in model.lattice.sites}
    loop_variables = tuple(
        model.layout.link_variable_index(link.id)
        for link in model.lattice.links
        if link.kind == "x" and site_cells[link.source][1] == 0
    )
    assert len(loop_variables) == model.ly
    return ErrorOperator(
        name="logical_Z_a",
        operator=_DensePauliZ(
            signs=_z_signs(basis, loop_variables),
            name="logical_Z_a",
        ),
        support_variables=loop_variables,
        kind="logical_loop",
    )


@pytest.fixture(scope="module")
def toric_code_fixture() -> _ToricCodeFixture:
    model = ToricCodeModel(lx=3, ly=3, boundary_condition="periodic")
    layout = model.layout
    basis = full_basis_from_layout(layout, sort=True)
    code = _toric_code_ground_space(model, basis)
    # A representative subset keeps the sanity check fast while still
    # exercising nontrivial E_a^† E_b products below the distance-3 loop.
    errors = _single_link_error_set(basis, (0, 1, 2))
    logical_z_loop = _logical_z_loop_operator(model, basis)
    return _ToricCodeFixture(
        model=model,
        layout=layout,
        basis=basis,
        code=code,
        local_errors=errors,
        logical_z_loop=logical_z_loop,
    )


def test_toric_code_fixture_builds_four_orthonormal_ground_states(
    toric_code_fixture: _ToricCodeFixture,
) -> None:
    code = toric_code_fixture.code

    assert code.dimension == 4
    assert code.ambient_dimension == 2**toric_code_fixture.model.lattice.num_links
    np.testing.assert_allclose(code.vectors.conj().T @ code.vectors, np.eye(4), atol=1e-12)


def test_qec_kl_diagnostic_passes_for_toric_code_single_link_paulis(
    toric_code_fixture: _ToricCodeFixture,
) -> None:
    report = diagnose_knill_laflamme(
        toric_code_fixture.code,
        toric_code_fixture.local_errors,
        tolerance=1e-10,
    )

    assert report.passes_exact_kl
    assert report.max_frobenius_residual < 1e-10
    assert report.max_spectral_residual < 1e-10
    assert report.worst_error_image is not None
    assert report.worst_error_image.relative_leakage_frobenius_norm > 0.0


def test_qec_local_indistinguishability_profile_accepts_toric_code_ground_space(
    toric_code_fixture: _ToricCodeFixture,
) -> None:
    report = diagnose_local_indistinguishability(
        toric_code_fixture.code,
        toric_code_fixture.local_errors,
        max_weight=1,
        tolerance=1e-10,
    )

    assert report.passes_all_tested_weights
    assert report.first_violating_weight is None
    assert report.local_indistinguishability_weight == 1
    assert report.worst_summary.max_frobenius_residual < 1e-10


def test_qec_error_algebra_is_scalar_for_toric_code_single_link_paulis(
    toric_code_fixture: _ToricCodeFixture,
) -> None:
    report = diagnose_projected_error_algebra(
        toric_code_fixture.code,
        toric_code_fixture.local_errors,
        max_weight=1,
        tolerance=1e-10,
    )

    assert report.classification == "scalar_algebra_subspace_code"
    assert report.algebra_dimension == 1
    assert report.generator_span_dimension == 1
    assert report.center_dimension == 1
    assert report.commutant_dimension == 16


def test_qec_logical_operator_diagnostic_detects_nonlocal_toric_code_loop(
    toric_code_fixture: _ToricCodeFixture,
) -> None:
    report = search_projected_logical_operators(
        toric_code_fixture.code,
        [toric_code_fixture.logical_z_loop],
    )
    candidate = report.best_candidate

    assert candidate is not None
    assert candidate.name == "logical_Z_a"
    assert candidate.weight == toric_code_fixture.model.ly
    assert candidate.leakage_frobenius_norm < 1e-12
    assert candidate.traceless_frobenius_norm > 0.0
    assert candidate.relative_traceless_frobenius_norm > 0.0
