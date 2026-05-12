import numpy as np
import pytest

from qlinks.builders import is_hermitian_sparse
from qlinks.models import ToricCodeModel


def test_toric_code_2_by_2_basis_count() -> None:
    model = ToricCodeModel(
        lx=2,
        ly=2,
        boundary_condition="periodic",
    )

    basis = model.build_basis(
        solver="dfs",
        sort=True,
    )

    assert model.lattice.num_links == 8
    assert basis.n_states == 2**8


def test_toric_code_2_by_2_build_smoke() -> None:
    model = ToricCodeModel(
        lx=2,
        ly=2,
        boundary_condition="periodic",
        electric=1.0,
        magnetic=1.0,
    )

    result = model.build(
        builder="sparse",
        basis_solver="dfs",
        sort_basis=True,
    )

    assert result.hamiltonian.shape == (2**8, 2**8)
    assert result.kinetic is not None
    assert result.potential is not None
    assert result.kinetic.nnz > 0
    assert result.potential.nnz > 0


def test_toric_code_hamiltonian_is_hermitian() -> None:
    model = ToricCodeModel(
        lx=2,
        ly=2,
        boundary_condition="periodic",
    )

    H = model.build_hamiltonian(builder="sparse")

    assert is_hermitian_sparse(H)


def test_toric_code_requires_periodic_boundary() -> None:
    model = ToricCodeModel(
        lx=2,
        ly=2,
        boundary_condition="open",
    )

    with pytest.raises(ValueError, match="periodic"):
        _ = model.lattice


def test_toric_code_2_by_2_ground_energy() -> None:
    from scipy.sparse.linalg import eigsh

    model = ToricCodeModel(
        lx=2,
        ly=2,
        boundary_condition="periodic",
        electric=1.0,
        magnetic=1.0,
    )

    H = model.build_hamiltonian(builder="sparse")

    eigenvalues = eigsh(H, k=4, which="SA", return_eigenvectors=False)
    eigenvalues = np.sort(eigenvalues)

    assert np.isclose(eigenvalues[0], -8.0, atol=1e-10)
