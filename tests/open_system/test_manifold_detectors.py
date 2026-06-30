from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp

from qlinks.open_system import diagnose_manifold_dark_operator_basis
from qlinks.open_system.constructions import build_degenerate_cage_lindblad_construction


class _ArrayBasis:
    def __init__(self, states):
        self.states = np.asarray(states, dtype=np.int64)


def _two_qubit_build_result():
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


def _equal_bit_manifold_rows():
    return np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.complex128,
    )


def _single_site_z_operators():
    z0 = sp.diags([1.0, 1.0, -1.0, -1.0], format="csr", dtype=np.complex128)
    z1 = sp.diags([1.0, -1.0, 1.0, -1.0], format="csr", dtype=np.complex128)
    return z0, z1


def test_collective_dark_operator_can_exist_when_single_site_support_is_full():
    build_result = _two_qubit_build_result()
    construction = build_degenerate_cage_lindblad_construction(
        build_result=build_result,
        states=_equal_bit_manifold_rows(),
        local_regions=((0,), (1,)),
    )

    support_summary = construction.local_subspace_support_report.to_summary_dict()
    assert support_summary["all_regions_have_full_local_support"] is True
    assert support_summary["n_regions_with_nullity"] == 0

    report = construction.diagnose_dark_operator_basis(
        operators=_single_site_z_operators(),
        operator_names=("Z0", "Z1"),
    )

    summary = report.to_summary_dict()
    assert summary["detector_nullity"] == 1
    assert summary["has_dark_detectors"] is True
    assert len(summary["candidates"]) == 1
    assert summary["candidates"][0]["action_residual"] < 1e-12

    coefficients = report.candidates[0].coefficients
    assert abs(abs(coefficients[0]) - abs(coefficients[1])) < 1e-12
    assert abs(coefficients[0] + coefficients[1]) < 1e-12


def test_manifold_dark_operator_basis_direct_api_and_rich_render():
    report = diagnose_manifold_dark_operator_basis(
        states=_equal_bit_manifold_rows(),
        operators=_single_site_z_operators(),
        operator_names=("Z0", "Z1"),
    )

    assert report.detector_nullity == 1
    assert report.candidates[0].terms[0].operator_name in {"Z0", "Z1"}

    from rich.console import Console

    console = Console(record=True, width=120)
    console.print(report)
    rendered = console.export_text()
    assert "Manifold dark-operator basis report" in rendered
    assert "dark-detector nullity" in rendered
    assert "Z0" in rendered or "Z1" in rendered
