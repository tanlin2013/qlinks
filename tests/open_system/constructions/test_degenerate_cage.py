from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp

from qlinks.caging.candidate import CandidateSubgraph
from qlinks.caging.search import CageRecord
from qlinks.caging.solver import CageState
from qlinks.models import LocalTermDescriptor
from qlinks.open_system import (
    build_local_recycling_jumps_from_subspace_regions,
    local_reduced_density_matrix_from_state_matrix,
)
from qlinks.open_system.constructions import build_degenerate_cage_lindblad_construction


class _ArrayBasis:
    def __init__(self, states):
        self.states = np.asarray(states, dtype=np.int64)


class _FakeLocalTermModel:
    def local_term_descriptors(self, *, term_kind=None):
        if term_kind not in {None, "plaquette"}:
            return ()
        return (
            LocalTermDescriptor(
                term_id=0,
                term_kind="plaquette",
                operator_kind="kinetic",
                support_links=(0, 1),
                support_variables=(0, 1),
            ),
        )


def _two_bit_build_result():
    basis = _ArrayBasis(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ]
    )
    hamiltonian = sp.csr_array((4, 4), dtype=np.complex128)
    return SimpleNamespace(basis=basis, hamiltonian=hamiltonian)


def _two_state_manifold_rows():
    return np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.complex128,
    )


def test_local_reduced_density_matrix_from_state_matrix_uses_manifold_support():
    build_result = _two_bit_build_result()
    rdm = local_reduced_density_matrix_from_state_matrix(
        basis_configs=build_result.basis.states,
        states=_two_state_manifold_rows(),
        variable_indices=(0, 1),
    )

    assert rdm.support_rank == 2
    assert rdm.nullity == 2
    np.testing.assert_allclose(np.sort(rdm.eigenvalues), [0.0, 0.0, 0.5, 0.5])


def test_subspace_block_reset_recycling_annihilates_entire_manifold():
    build_result = _two_bit_build_result()
    states = _two_state_manifold_rows().T
    result = build_local_recycling_jumps_from_subspace_regions(
        basis_configs=build_result.basis.states,
        states=states,
        regions=((0, 1),),
        source="local_rdm_block_reset",
    )

    assert result.n_jumps == 1
    jump = result.jumps[0]
    np.testing.assert_allclose((jump @ states), np.zeros((4, 2)))
    assert result.selections[0].candidate.inflow_norm > 0.0


def test_degenerate_cage_lindblad_construction_from_states():
    build_result = _two_bit_build_result()
    construction = build_degenerate_cage_lindblad_construction(
        build_result=build_result,
        states=_two_state_manifold_rows(),
        local_regions=((0, 1),),
    )

    summary = construction.to_summary_dict()
    assert summary["manifold_dimension"] == 2
    assert summary["n_jumps"] == 1
    assert summary["local_regions"] == ((0, 1),)
    assert construction.inflow_norm > 0.0
    assert construction.max_jump_residual < 1e-12
    assert construction.liouvillian_residual is not None
    assert construction.liouvillian_residual < 1e-12
    np.testing.assert_allclose(
        construction.target_density_matrix,
        np.diag([0.5, 0.0, 0.0, 0.5]),
    )


def test_degenerate_cage_lindblad_construction_infers_regions_from_model():
    build_result = _two_bit_build_result()
    construction = build_degenerate_cage_lindblad_construction(
        build_result=build_result,
        model=_FakeLocalTermModel(),
        states=_two_state_manifold_rows(),
        local_term_kind="plaquette",
    )

    assert construction.local_regions == ((0, 1),)
    assert construction.n_jumps == 1


def test_degenerate_cage_lindblad_construction_from_records_validates_signature():
    build_result = _two_bit_build_result()
    records = []
    for basis_index in (0, 3):
        records.append(
            CageRecord(
                cage_state=CageState(
                    energy=0.0,
                    local_state=np.asarray([1.0], dtype=np.complex128),
                    support=np.asarray([basis_index], dtype=np.int64),
                    boundary_residual=0.0,
                    eigen_residual=0.0,
                ),
                signature=(0, 4),
                candidate=CandidateSubgraph(np.asarray([basis_index], dtype=np.int64)),
                full_state=None,
            )
        )

    construction = build_degenerate_cage_lindblad_construction(
        build_result=build_result,
        records=records,
        local_regions=((0, 1),),
    )

    assert construction.record_signature == (0, 4)
    assert construction.manifold_dimension == 2
    assert construction.n_jumps == 1
