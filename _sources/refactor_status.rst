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

The former ``qlinks.caging.local_search`` implementation is now split into:

* ``local_search_types`` for passive configuration/report contracts;
* ``local_search_geometry`` for pure stripe, snake, plaquette/link-region, and local-index
  geometry helpers;
* ``local_search_core`` for generic local type-1 search algebra and adapter registration;
* ``local_search_qdm`` for QDM local-region construction, basis enumeration, and local kinetic
  algebra;
* ``local_search_global`` for explicit global-QDM plaquette actions and limited global operators;
* ``local_search_padding`` for single/multi-block exterior-padding search and structural
  validation;
* ``local_search_factorized`` for exact factorized-product residual certification;
* ``local_search_certification`` for local/multi-block residual certification and result assembly;
* ``local_search_proposals`` for stripe/snake/adaptive proposal generation;
* ``local_search_scan`` for proposal execution and block collection; and
* ``local_search_workflows`` for robust multi-stage local-search orchestration.

``qlinks.caging.local_search`` is now a temporary compatibility facade. Active first-party code
uses the focused modules directly. Because qlinks is primarily group-internal software, this
facade is intentionally minimal and should be removed during the later API-cleanup pass once the
refactored interface has stabilized.

The focused local-search graph is required to remain free of both eager import cycles and
TYPE_CHECKING/function-local static cycles. Passive result containers therefore live in
``local_search_types`` rather than importing their implementation modules back into the contract
layer.

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
enforce this for the current local-search, stability, and manifold-detector facades.
