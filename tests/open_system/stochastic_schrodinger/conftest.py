import numpy as np
import pytest


@pytest.fixture
def qubit_ops():
    sigma_minus = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
    sigma_plus = sigma_minus.conj().T
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    identity = np.eye(2, dtype=np.complex128)

    ket0 = np.array([1.0, 0.0], dtype=np.complex128)
    ket1 = np.array([0.0, 1.0], dtype=np.complex128)

    return {
        "sigma_minus": sigma_minus,
        "sigma_plus": sigma_plus,
        "sigma_x": sigma_x,
        "sigma_z": sigma_z,
        "identity": identity,
        "ket0": ket0,
        "ket1": ket1,
    }
