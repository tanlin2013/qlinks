from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

PlaquetteCoupling: TypeAlias = complex | Mapping[int, complex] | Callable[[int], complex]


@dataclass(frozen=True, slots=True)
class DirectedPlaquetteCoupling:
    """Forward/backward coupling for an oriented plaquette transition.

    If backward is omitted, Hermiticity is imposed by using forward.conjugate().
    """

    forward: complex
    backward: complex | None = None

    def resolved_forward(self) -> complex:
        return complex(self.forward)

    def resolved_backward(self) -> complex:
        if self.backward is None:
            return complex(self.forward).conjugate()

        return complex(self.backward)


DirectedPlaquetteCouplingLike: TypeAlias = (
    DirectedPlaquetteCoupling
    | complex
    | Mapping[int, DirectedPlaquetteCoupling | complex]
    | Callable[[int], DirectedPlaquetteCoupling | complex]
)


def plaquette_coupling_value(
    coupling: PlaquetteCoupling,
    plaquette_id: int,
    *,
    name: str,
) -> complex:
    if callable(coupling):
        return complex(coupling(plaquette_id))

    if isinstance(coupling, Mapping):
        try:
            return complex(coupling[plaquette_id])
        except KeyError as error:
            raise KeyError(f"{name} has no value for plaquette_id={plaquette_id}.") from error

    return complex(coupling)


def directed_plaquette_coupling_value(
    coupling: DirectedPlaquetteCouplingLike,
    plaquette_id: int,
    *,
    name: str,
) -> DirectedPlaquetteCoupling:
    if callable(coupling):
        value = coupling(plaquette_id)
    elif isinstance(coupling, Mapping):
        try:
            value = coupling[plaquette_id]
        except KeyError as error:
            raise KeyError(f"{name} has no value for plaquette_id={plaquette_id}.") from error
    else:
        value = coupling

    if isinstance(value, DirectedPlaquetteCoupling):
        return value

    return DirectedPlaquetteCoupling(forward=complex(value))


def peierls_plaquette_coupling(
    amplitude: complex,
    phase: float,
) -> DirectedPlaquetteCoupling:
    """Return Hermitian Peierls forward/backward couplings."""
    forward = complex(amplitude) * np.exp(1j * phase)
    return DirectedPlaquetteCoupling(forward=forward)


def is_zero_coupling(
    coupling: PlaquetteCoupling,
    plaquette_ids: list[int],
) -> bool:
    return all(
        plaquette_coupling_value(
            coupling,
            int(plaquette_id),
            name="coupling",
        )
        == 0
        for plaquette_id in plaquette_ids
    )
