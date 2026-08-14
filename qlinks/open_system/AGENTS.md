# Open-system-layer agent rules

This file supplements the repository-root `AGENTS.md`.

- Keep solver kernels, jump construction, detector selection, diagnostics, and visualization
  as separate responsibilities.
- Prefer state/operator protocols and neutral local-structure primitives over concrete
  dependencies on caging search records.
- The temporary `manifold_detectors.py` facade has been removed. Import
  `manifold_dark`, `manifold_recycling`, `manifold_residual`, or
  `manifold_detector_types` directly.
- Do not add new responsibilities to the large cage-Lindblad construction modules. Extract
  focused modules instead.
- New Lindblad constructions must state the darkness, kernel-invariance, trace-preservation,
  and recycling/inflow conditions they rely on.
- Dense Liouvillian construction is allowed only for explicitly small validation systems.
- Deprecated constructors receive compatibility fixes only. New features belong in the
  current construction path, with a migration note when behaviour differs. Compatibility
  bridges should be removed after the replacement API stabilizes and first-party callers
  have migrated.
- Avoid expanding `qlinks.open_system.__init__` unless the interface is intentionally
  supported.
