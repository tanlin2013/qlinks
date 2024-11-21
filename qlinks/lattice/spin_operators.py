from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True, frozen=True)
class SpinOperators:
    """
    Generate the spin ladder operators S+ (raising), S- (lowering),
    and Sz (z-component) for a given spin quantum number S.

    Parameters:
        s (float): Spin quantum number (e.g., 0.5, 1, 1.5, 2).

    Attributes:
        s_plus: Raising operator (matrix)
        s_minus: Lowering operator (matrix)
        s_x: X-component operator (matrix)
        s_y: Y-component operator (matrix)
        s_z: Z-component operator (matrix)
        idty: Identity operator (matrix)
    """
    s: float
    s_plus: np.ndarray = field(init=False, repr=False)
    s_minus: np.ndarray = field(init=False, repr=False)
    s_x: np.ndarray = field(init=False, repr=False)
    s_y: np.ndarray = field(init=False, repr=False)
    s_z: np.ndarray = field(init=False, repr=False)
    idty: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        dim = int(2 * self.s + 1)

        # Initialize matrices
        s_plus = np.zeros((dim, dim), dtype=float)
        s_minus = np.zeros((dim, dim), dtype=float)
        s_z = np.zeros((dim, dim), dtype=float)

        # Basis states |m> from m = s to m = -s
        m_values = np.arange(self.s, -self.s - 1, -1)

        for i, m in enumerate(m_values):
            # Sz operator is diagonal with eigenvalues m
            s_z[i, i] = m

            # S+ operator
            if i > 0:
                s_plus[i - 1, i] = np.sqrt(self.s * (self.s + 1) - m * (m + 1))

            # S- operator
            if i < dim - 1:
                s_minus[i + 1, i] = np.sqrt(self.s * (self.s + 1) - m * (m - 1))

        object.__setattr__(self, "s_plus", s_plus)
        object.__setattr__(self, "s_minus", s_minus)
        object.__setattr__(self, "s_x", 0.5 * (s_plus + s_minus))
        object.__setattr__(self, "s_y", -0.5j * (s_plus - s_minus))
        object.__setattr__(self, "s_z", s_z)
        object.__setattr__(self, "idty", np.eye(dim, dtype=float))
