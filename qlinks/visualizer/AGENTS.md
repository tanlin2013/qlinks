# Visualizer-layer agent rules

This file supplements the repository-root `AGENTS.md`.

- Visualization code may depend on scientific/model layers for read-only interpretation, but
  numerical kernels must not depend on `qlinks.visualizer`.
- `qlinks.visualizer.basis` is a responsibility-oriented subpackage, not a monolithic module.
  Keep these roles separate:
  - `styles`: public style/type contracts and named theme defaults;
  - `render_cache`: passive geometry/render-cache records;
  - `rendering`: generic node/link drawing backends;
  - `periodic`: periodic-image and coordinate geometry;
  - `plaquette_geometry`: visual plaquette construction/selection;
  - `plaquette_symbols`: QDM/QLM plaquette-symbol and vulnerable-link semantics;
  - `configuration`: the composed single-configuration visualizer;
  - `grid` and `local_grid`: multi-state and local-support orchestration;
  - `formatting`: pure labels/grid-shape/cage-support presentation helpers; and
  - `api`: functional wrappers that instantiate the supported visualizer classes.
- Do not add a new rendering responsibility directly to `configuration.py`; add or extend the
  focused role module and compose it through `BasisConfigurationVisualizer`.
- Keep `basis/__init__.py` curated. Private rendering/cache helpers must be imported from their
  defining child module by first-party implementation/tests rather than re-exported for convenience.
- Rendering refactors must preserve lattice-coordinate, periodic-image, plaquette-ordering, and
  symbol conventions unless a scientific/presentation change is explicitly requested and tested.
- Manual image-inspection tests remain `manual`; deterministic artist/geometry contracts should be
  protected in the fast lane without requiring display backends.
