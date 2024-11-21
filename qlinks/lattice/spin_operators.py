import numpy as np


def spin_operators(s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate the spin ladder operators S+ (raising), S- (lowering),
    and Sz (z-component) for a given spin quantum number S.

    Parameters:
        s (float): Spin quantum number (e.g., 0.5, 1, 1.5, 2).

    Returns:
        tuple: (s_plus, s_minus, s_z)
               s_plus: Raising operator (matrix)
               s_minus: Lowering operator (matrix)
               s_z: Z-component operator (matrix)
    """
    # Number of states is 2S + 1
    dim = int(2 * s + 1)

    # Initialize matrices
    s_plus = np.zeros((dim, dim), dtype=float)
    s_minus = np.zeros((dim, dim), dtype=float)
    s_z = np.zeros((dim, dim), dtype=float)

    # Basis states |m> from m = s to m = -s
    m_values = np.arange(s, -s - 1, -1)

    for i, m in enumerate(m_values):
        # Sz operator is diagonal with eigenvalues m
        s_z[i, i] = m

        # S+ operator
        if i > 0:
            s_plus[i - 1, i] = np.sqrt(s * (s + 1) - m * (m + 1))

        # S- operator
        if i < dim - 1:
            s_minus[i + 1, i] = np.sqrt(s * (s + 1) - m * (m - 1))

    return s_plus, s_minus, s_z
