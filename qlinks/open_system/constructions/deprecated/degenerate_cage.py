"""Deprecated compatibility imports for the former degenerate cage module.

The active implementation now lives in
:mod:`qlinks.open_system.constructions.cage_lindblad`.  This module is kept only
for legacy comparison imports.
"""

from qlinks.open_system.constructions.cage_lindblad import (
    DegenerateCageJumpDesignWorkflowReport,
    DegenerateCageLindbladConstruction,
    LocalRegionSource,
    build_degenerate_cage_lindblad_construction,
)

__all__ = [
    "DegenerateCageJumpDesignWorkflowReport",
    "DegenerateCageLindbladConstruction",
    "LocalRegionSource",
    "build_degenerate_cage_lindblad_construction",
]
