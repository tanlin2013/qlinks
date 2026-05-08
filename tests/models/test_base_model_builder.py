from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from qlinks.basis import Basis
from qlinks.builders import is_hermitian_sparse
from qlinks.encoded import BinaryEncodedBasis, BitmaskBinaryFlipOperator
from qlinks.lattice import ChainLattice
from qlinks.models.base import (
    GenericModelBuilder,
    HamiltonianModelBase,
    HamiltonianTermSpec,
    combine_hamiltonian_terms,
    validate_builder_name,
)
from qlinks.operators import BinaryFlipOperator, UpdateBinaryFlipOperator
from qlinks.variables import LocalSpace, VariableLayout


@dataclass(frozen=True)
class TinyFlipModel(HamiltonianModelBase):
    """
    Minimal test model with one binary site variable and one flip operator.
    """

    coefficient: complex = 2.0

    def _make_lattice(self) -> ChainLattice:
        return ChainLattice(1, boundary_condition="open")

    def _make_layout(self) -> VariableLayout:
        return VariableLayout.from_lattice_sites(
            self.lattice,
            LocalSpace.binary(),
        )

    def make_terms(
        self,
        layout: VariableLayout,
        *,
        builder: str = "sparse",
    ):
        if builder == "sparse":
            op = BinaryFlipOperator(
                layout=layout,
                variable_index=0,
                coefficient=self.coefficient,
            )
        elif builder == "optimized":
            op = UpdateBinaryFlipOperator(
                layout=layout,
                variable_index=0,
                coefficient=self.coefficient,
            )
        elif builder == "bitmask":
            op = BitmaskBinaryFlipOperator(
                layout=layout,
                variable_index=0,
                coefficient=self.coefficient,
            )
        else:
            raise ValueError(f"Unsupported builder: {builder}")

        return (
            HamiltonianTermSpec.from_operators(
                name="kinetic",
                operators=(op,),
                kind="kinetic",
            ),
        )


def test_validate_builder_name() -> None:
    validate_builder_name("sparse")
    validate_builder_name("optimized")
    validate_builder_name("bitmask")

    with pytest.raises(ValueError, match="builder must be"):
        validate_builder_name("bad")  # type: ignore[arg-type]


def test_combine_hamiltonian_terms_rejects_empty() -> None:
    with pytest.raises(ValueError, match="zero Hamiltonian terms"):
        combine_hamiltonian_terms([None, None])


def test_base_model_caches_lattice_and_layout() -> None:
    model = TinyFlipModel()

    assert model.lattice is model.lattice
    assert model.layout is model.layout
    assert model.model_builder is model.model_builder

    assert model.make_lattice() is model.lattice
    assert model.make_layout() is model.layout


def test_base_model_build_basis() -> None:
    model = TinyFlipModel()

    basis = model.build_basis(solver="dfs", sort=True)

    assert basis.n_states == 2
    np.testing.assert_array_equal(
        basis.states,
        np.array(
            [
                [0],
                [1],
            ],
            dtype=np.int64,
        ),
    )


def test_generic_builder_sparse_build() -> None:
    model = TinyFlipModel(coefficient=2.0)

    result = model.build(
        basis_solver="dfs",
        builder="sparse",
        sort_basis=True,
    )

    expected = np.array(
        [
            [0, 2],
            [2, 0],
        ],
        dtype=np.complex128,
    )

    assert isinstance(result.basis, Basis)
    assert result.kinetic is not None
    assert result.potential is None
    assert result.kinetic_operators
    assert result.potential_operators == ()
    assert result.operators == result.kinetic_operators
    assert is_hermitian_sparse(result.hamiltonian)

    np.testing.assert_allclose(result.hamiltonian.toarray(), expected)
    np.testing.assert_allclose(result.kinetic.toarray(), expected)


def test_generic_builder_optimized_matches_sparse() -> None:
    model = TinyFlipModel(coefficient=2.0)

    sparse_result = model.build(
        basis_solver="dfs",
        builder="sparse",
        sort_basis=True,
    )

    optimized_result = model.build(
        basis=sparse_result.basis,
        builder="optimized",
        sort_basis=True,
    )

    np.testing.assert_allclose(
        sparse_result.hamiltonian.toarray(),
        optimized_result.hamiltonian.toarray(),
    )


def test_generic_builder_bitmask_matches_sparse() -> None:
    model = TinyFlipModel(coefficient=2.0)

    sparse_result = model.build(
        basis_solver="dfs",
        builder="sparse",
        sort_basis=True,
    )

    bitmask_result = model.build(
        basis=sparse_result.basis,
        builder="bitmask",
        sort_basis=True,
    )

    assert isinstance(bitmask_result.basis, BinaryEncodedBasis)

    np.testing.assert_allclose(
        sparse_result.hamiltonian.toarray(),
        bitmask_result.hamiltonian.toarray(),
    )


def test_generic_builder_accepts_prebuilt_encoded_basis() -> None:
    model = TinyFlipModel(coefficient=2.0)

    array_basis = model.build_basis(solver="dfs", sort=True)
    encoded_basis = BinaryEncodedBasis.from_basis(array_basis, sort=False)

    result = model.build(
        basis=encoded_basis,
        builder="bitmask",
    )

    assert result.basis is encoded_basis
    assert result.hamiltonian.shape == (2, 2)


def test_build_hamiltonian_convenience() -> None:
    model = TinyFlipModel(coefficient=2.0)

    H = model.build_hamiltonian(
        basis_solver="dfs",
        builder="sparse",
        sort_basis=True,
    )

    expected = np.array(
        [
            [0, 2],
            [2, 0],
        ],
        dtype=np.complex128,
    )

    np.testing.assert_allclose(H.toarray(), expected)


def test_generic_model_builder_direct_usage() -> None:
    model = TinyFlipModel(coefficient=2.0)

    result = GenericModelBuilder(model).build(
        basis_solver="dfs",
        builder="sparse",
        sort_basis=True,
    )

    assert result.hamiltonian.shape == (2, 2)


@dataclass(frozen=True)
class EmptyTermModel(HamiltonianModelBase):
    def _make_lattice(self) -> ChainLattice:
        return ChainLattice(1, boundary_condition="open")

    def _make_layout(self) -> VariableLayout:
        return VariableLayout.from_lattice_sites(
            self.lattice,
            LocalSpace.binary(),
        )

    def make_terms(
        self,
        layout: VariableLayout,
        *,
        builder: str = "sparse",
    ):
        return (
            HamiltonianTermSpec.from_operators(
                name="kinetic",
                operators=(),
                kind="kinetic",
            ),
        )


@dataclass(frozen=True)
class EmptyHamiltonianModel(HamiltonianModelBase):
    def _make_lattice(self) -> ChainLattice:
        return ChainLattice(2, boundary_condition="open")

    def _make_layout(self) -> VariableLayout:
        return VariableLayout.from_lattice_sites(
            self.lattice,
            LocalSpace.binary(),
        )

    def make_terms(
        self,
        layout: VariableLayout,
        *,
        builder: str = "sparse",
    ):
        return (
            HamiltonianTermSpec.from_operators(
                name="kinetic",
                operators=(),
                kind="kinetic",
            ),
        )


def test_empty_hamiltonian_model_returns_zero_sparse_matrix() -> None:
    model = EmptyHamiltonianModel()

    result = model.build(
        basis_solver="dfs",
        builder="sparse",
        sort_basis=True,
    )

    assert result.basis.n_states == 4
    assert result.hamiltonian.shape == (4, 4)
    assert result.hamiltonian.nnz == 0
    assert result.kinetic is None
