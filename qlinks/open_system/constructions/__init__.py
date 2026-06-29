"""Preset Lindblad/open-system problem constructors.

This namespace collects model-aware construction helpers that assemble jump
operators, targets, and diagnostics from higher-level qlinks model/caging data.
The low-level open-system solvers and operator utilities remain in
``qlinks.open_system``.
"""

from importlib import import_module

_CAGE_EXPORTS = {
    "CageLindbladConstruction",
    "JumpOperatorDesign",
    "JumpPlaquettePolicy",
    "KineticJumpGrouping",
    "LocalRecyclerReadout",
    "MonitorPlaquettePolicy",
    "MonitorRecyclerHamiltonianClosureSource",
    "MonitorRecyclerHamiltonianShift",
    "MonitorSource",
    "ReducedIZMonitorComponent",
    "ReducedIZMonitorContent",
    "build_type1_cage_lindblad_construction",
    "build_type1_local_cage_lindblad_construction",
}

_DEGENERATE_CAGE_EXPORTS = {
    "DegenerateCageLindbladConstruction",
    "LocalRegionSource",
    "build_degenerate_cage_lindblad_construction",
}

_SPIN_ONE_XY_EXPORTS = {
    "SpinOneXYLeftMultiplier",
    "SpinOneXYLindbladConstruction",
    "build_spin_one_xy_lindblad_construction",
}

__all__ = sorted(_CAGE_EXPORTS | _DEGENERATE_CAGE_EXPORTS | _SPIN_ONE_XY_EXPORTS)


def __getattr__(name: str) -> object:
    if name in _CAGE_EXPORTS:
        module = import_module("qlinks.open_system.constructions.cage")
        return getattr(module, name)
    if name in _DEGENERATE_CAGE_EXPORTS:
        module = import_module("qlinks.open_system.constructions.degenerate_cage")
        return getattr(module, name)
    if name in _SPIN_ONE_XY_EXPORTS:
        module = import_module("qlinks.open_system.constructions.spin_one_xy")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
