from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TypeAlias

PlaquetteCoupling: TypeAlias = complex | Mapping[int, complex] | Callable[[int], complex]


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
