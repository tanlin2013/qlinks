"""Preset Lindblad/open-system problem constructors.

The preferred cage-state API is the unified workflow in
``build_cage_lindblad_problem``.  It treats a single cage state as a
one-dimensional dark manifold and a degenerate cage multiplet as a
higher-dimensional dark manifold, then uses the same detector/recycler design
workflow in both cases.

Legacy single-cage and old degenerate constructor import paths live under
``qlinks.open_system.constructions.deprecated`` for comparison only.
"""

from importlib import import_module

_CAGE_LINDBLAD_EXPORTS = {
    "CageLindbladDesignProblem",
    "CageLindbladDesignResult",
    "CageLindbladDetectorOperators",
    "CageLindbladWorkflowReport",
    "DetectorOperatorKind",
    "LocalRegionSource",
    "build_cage_lindblad_detector_operators",
    "build_cage_lindblad_problem",
}

_SPIN_ONE_XY_EXPORTS = {
    "SpinOneXYLeftMultiplier",
    "SpinOneXYLindbladConstruction",
    "build_spin_one_xy_lindblad_construction",
}

_LEGACY_EXPORTS = {
    "CageLindbladConstruction",
    "DegenerateCageLindbladConstruction",
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
    "build_degenerate_cage_lindblad_construction",
    "build_type1_cage_lindblad_construction",
    "build_type1_local_cage_lindblad_construction",
}

__all__ = sorted(_CAGE_LINDBLAD_EXPORTS | _SPIN_ONE_XY_EXPORTS)


def __getattr__(name: str) -> object:
    if name in _CAGE_LINDBLAD_EXPORTS:
        module = import_module("qlinks.open_system.constructions.cage_lindblad")
        return getattr(module, name)
    if name in _SPIN_ONE_XY_EXPORTS:
        module = import_module("qlinks.open_system.constructions.spin_one_xy")
        return getattr(module, name)
    if name in _LEGACY_EXPORTS:
        module = import_module("qlinks.open_system.constructions.deprecated")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
