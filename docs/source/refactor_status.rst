Repository refactor status
==========================

This page records the stabilized module boundaries that remain after the temporary refactor
facades were removed. The focused modules below are now the supported import paths for active
package code.

Current decomposition
---------------------

Cage stability
~~~~~~~~~~~~~~

The cage-stability implementation now lives in the ``qlinks.caging.stability`` subpackage:

* ``core`` for perturbative stability, continuation, Jacobian, and subspace helpers;
* ``topology`` for chiral, locality, CLS-completeness, and cohomological diagnostics;
* ``boundary`` for boundary-cancellation matroid and periodic-scaling diagnostics;
* ``qdm`` for square-QDM compact-cage and transfer diagnostics;
* ``laurent`` for Laurent-polynomial constraint-module diagnostics;
* ``types`` for report/data contracts; and
* ``symmetry`` for small shared symmetry linear-algebra helpers.

``qlinks.caging.stability`` is now a real subpackage with a curated public API. Active
implementation code should import the defining child module directly.

Local cage search
~~~~~~~~~~~~~~~~~

Local cage search now lives in the ``qlinks.caging.local_search`` subpackage:

* ``types`` for passive configuration/report contracts;
* ``geometry`` for pure stripe, snake, plaquette/link-region, and local-index
  geometry helpers;
* ``core`` for generic local type-1 search algebra and adapter registration;
* ``qdm`` for QDM local-region construction, basis enumeration, and local kinetic
  algebra;
* ``global_ops`` for explicit global-QDM plaquette actions and limited global operators;
* ``padding`` for single/multi-block exterior-padding search and structural
  validation;
* ``factorized`` for exact factorized-product residual certification;
* ``certification`` for local/multi-block residual certification and result assembly;
* ``proposals`` for stripe/snake/adaptive proposal generation;
* ``scan`` for proposal execution and block collection; and
* ``workflows`` for robust multi-stage local-search orchestration.

``qlinks.caging.local_search`` is now a real subpackage with a curated public API. Active
implementation code should import the defining child module directly.

The focused local-search graph is required to remain free of both eager import cycles and
TYPE_CHECKING/function-local static cycles. Passive result containers therefore live in
``local_search.types`` rather than importing implementation modules back into the contract
layer.

Dark-manifold detectors
~~~~~~~~~~~~~~~~~~~~~~~

The former ``qlinks.open_system.manifold_detectors`` implementation is split into:

* ``manifold_dark`` for dark-operator basis construction, dressing, and shared linear algebra;
* ``manifold_recycling`` for local recycler construction and family-kernel diagnostics;
* ``manifold_residual`` for residual-kernel diagnostics and targeted jump selection; and
* ``manifold_detector_types`` for passive report/data contracts.

The temporary ``qlinks.open_system.manifold_detectors`` facade has been removed. Active package
code and tests must import the focused modules instead.

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

No new implementation may depend on the removed compatibility-module paths. The architecture
tests enforce this for the former local-search, stability, and manifold-detector facades.
