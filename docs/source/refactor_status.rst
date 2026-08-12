Repository refactor status
==========================

This page tracks temporary architecture bridges introduced while the caging and open-system
research layers are being decomposed.  The bridges are migration aids, not long-term API
commitments.

Current decomposition
---------------------

Cage stability
~~~~~~~~~~~~~~

The former monolithic ``qlinks.caging.stability`` implementation is split into:

* ``stability_core`` for perturbative stability, continuation, Jacobian, and subspace helpers;
* ``stability_topology`` for chiral, locality, CLS-completeness, and cohomological diagnostics;
* ``stability_boundary`` for boundary-cancellation matroid and periodic-scaling diagnostics;
* ``stability_qdm`` for square-QDM compact-cage and transfer diagnostics;
* ``stability_laurent`` for Laurent-polynomial constraint-module diagnostics;
* ``stability_types`` for report/data contracts; and
* ``stability_symmetry`` for small shared symmetry linear-algebra helpers.

``qlinks.caging.stability`` currently re-exports these objects so historical imports continue
to work during migration.  Active package code must use the focused modules directly.

Local cage search
~~~~~~~~~~~~~~~~~

``local_search_types`` now owns passive configuration/report contracts, while
``local_search_geometry`` owns pure stripe, snake, plaquette/link-region, and local-index
geometry helpers.  Search orchestration, QDM local enumeration, and padding/certification
remain in ``local_search`` and are the next decomposition target.

Dark-manifold detectors
~~~~~~~~~~~~~~~~~~~~~~~

The former ``qlinks.open_system.manifold_detectors`` implementation is split into:

* ``manifold_dark`` for dark-operator basis construction, dressing, and shared linear algebra;
* ``manifold_recycling`` for local recycler construction and family-kernel diagnostics;
* ``manifold_residual`` for residual-kernel diagnostics and targeted jump selection; and
* ``manifold_detector_types`` for passive report/data contracts.

``qlinks.open_system.manifold_detectors`` is a temporary re-export facade.  Active package
code must import the focused modules instead.

Neutral local-structure migration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Local reduced-density matrices, matrix-unit expansion, and local pattern embedding belong to
``qlinks.local_structure``.  Historical re-exports through caging/open-system modules exist
only to let notebooks and deprecated constructors migrate.

Removal gates
-------------

A compatibility bridge should be deleted when:

#. the replacement names and module boundaries have survived the current review cycle;
#. first-party package code, tests, scripts, notebooks, and documentation use the replacement;
#. supported public API tests target the replacement interface rather than the bridge; and
#. the removal is recorded in the changelog or release/refactor milestone.

No new implementation may depend on a temporary compatibility facade.  The architecture tests
enforce this for the current stability and manifold-detector facades.
