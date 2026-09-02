"""Spin-1 exchange-normalization contract shared by Sec. VI evidence jobs.

The permanent convention is

    H_XY = J sum(Sx Sx + Sy Sy)
         = (J/2) sum(S+ S- + S- S+).

Historical Sec. VI evidence predating 2026-09-02 used manuscript-facing wrappers
with a ladder prefactor of one.  Those folders remain immutable and may only be
reused through an explicit rescaling step.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

CURRENT_EXCHANGE_CONVENTION = "J_over_2_ladder_v1"
LEGACY_EXCHANGE_CONVENTION = "ladder_prefactor_1"
EXCHANGE_CONVENTION_METADATA_KEY = "spin1_xy_exchange_convention"
RESCALED_FROM_METADATA_KEY = "rescaled_from_exchange_convention"

LEGACY_TO_CURRENT_ENERGY_SCALE = 0.5
LEGACY_TO_CURRENT_BETA_J_SCALE = 2.0

PRIMARY_WINDOW_EXPONENT = 0.25
PRIMARY_WINDOW_PREFACTOR = 0.5
FIXED_CONTROL_HALF_WIDTH = 0.5
PRIMARY_WINDOW_PROTOCOL = "quarter_power_c0p5"
FIXED_WINDOW_PROTOCOL = "fixed_width_0p5"

LEGACY_PROTOCOL_MAP = {
    "quarter_power_c1": PRIMARY_WINDOW_PROTOCOL,
    "fixed_width_1": FIXED_WINDOW_PROTOCOL,
}


def exchange_convention_from_metadata(metadata: Mapping[str, Any]) -> str:
    """Return the explicit convention, treating an absent field as historical legacy."""

    value = metadata.get(EXCHANGE_CONVENTION_METADATA_KEY)
    if value is None:
        return LEGACY_EXCHANGE_CONVENTION
    return str(value)


def require_current_exchange_convention(metadata: Mapping[str, Any]) -> None:
    """Reject metadata that is not explicitly on the permanent convention."""

    actual = exchange_convention_from_metadata(metadata)
    if actual != CURRENT_EXCHANGE_CONVENTION:
        raise ValueError(
            "spin-1 checkpoint exchange convention mismatch: "
            f"expected {CURRENT_EXCHANGE_CONVENTION!r}, got {actual!r}. "
            "Use the explicit legacy rescaling path instead of silent reuse."
        )


def current_window_half_width(length: int, protocol: str) -> float:
    """Return the canonical Sec. VI half-width in displayed J=1 units."""

    if protocol == PRIMARY_WINDOW_PROTOCOL:
        return PRIMARY_WINDOW_PREFACTOR * float(length) ** PRIMARY_WINDOW_EXPONENT
    if protocol == FIXED_WINDOW_PROTOCOL:
        return FIXED_CONTROL_HALF_WIDTH
    raise ValueError(f"unknown current spin-1 window protocol: {protocol}")


def map_legacy_window_protocol(protocol: str) -> str:
    """Map a historical protocol label into the permanent convention."""

    return LEGACY_PROTOCOL_MAP.get(str(protocol), str(protocol))


def current_metadata(*, rescaled_from: str | None = None) -> dict[str, str]:
    """Return convention metadata suitable for new checkpoints and derived products."""

    result = {EXCHANGE_CONVENTION_METADATA_KEY: CURRENT_EXCHANGE_CONVENTION}
    if rescaled_from is not None:
        result[RESCALED_FROM_METADATA_KEY] = str(rescaled_from)
    return result
