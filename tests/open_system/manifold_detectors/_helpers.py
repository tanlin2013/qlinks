from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp


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


def _single_site_x0_operator():
    rows = np.asarray([2, 3, 0, 1], dtype=np.int64)
    cols = np.asarray([0, 1, 2, 3], dtype=np.int64)
    data = np.ones(4, dtype=np.complex128)
    return sp.csr_array((data, (rows, cols)), shape=(4, 4))


def _single_qutrit_build_result():
    basis = _ArrayBasis([[0], [1], [2]])
    hamiltonian = sp.csr_array((3, 3), dtype=np.complex128)
    return SimpleNamespace(basis=basis, hamiltonian=hamiltonian)


def _single_qutrit_target_state():
    return np.asarray([1.0, 0.0, 0.0], dtype=np.complex128)


def _single_qutrit_detector():
    return sp.diags([0.0, 1.0, 0.0], format="csr", dtype=np.complex128)


def _single_qutrit_detector_pair():
    d1 = sp.diags([0.0, 1.0, 0.0], format="csr", dtype=np.complex128)
    d2 = sp.diags([0.0, 0.0, 1.0], format="csr", dtype=np.complex128)
    return d1, d2


def _single_qutrit_offdiagonal_detector():
    rows = np.asarray([1], dtype=np.int64)
    cols = np.asarray([2], dtype=np.int64)
    data = np.ones(1, dtype=np.complex128)
    return sp.csr_array((data, (rows, cols)), shape=(3, 3))
