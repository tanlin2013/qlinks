"""Deprecated cage Lindblad constructors kept for legacy comparison.

New code should use :func:`qlinks.open_system.constructions.build_cage_lindblad_problem`
and the returned problem's ``design_jumps`` method.  The legacy single-cage and
former degenerate module import paths are re-exported here only so notebooks/tests
can compare older physical ansatzes against the current workflow.
"""

from qlinks.open_system.constructions.deprecated.cage import (
    CageLindbladConstruction,
    JumpOperatorDesign,
    JumpPlaquettePolicy,
    KineticJumpGrouping,
    LocalRecyclerReadout,
    MonitorPlaquettePolicy,
    MonitorRecyclerHamiltonianClosureSource,
    MonitorRecyclerHamiltonianShift,
    MonitorSource,
    ReducedIZMonitorComponent,
    ReducedIZMonitorContent,
    build_type1_cage_lindblad_construction,
    build_type1_local_cage_lindblad_construction,
)
from qlinks.open_system.constructions.deprecated.degenerate_cage import (
    DegenerateCageLindbladConstruction,
    build_degenerate_cage_lindblad_construction,
)

__all__ = [
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
]
