qlinks.caging package
=====================

The caging package contains the core cage search/solver API plus focused subpackages for local
search, post-search analysis, and stability. Shared local algebra lives in
:mod:`qlinks.local_structure`; open-system construction lives separately in
:mod:`qlinks.open_system`.

Subpackages
-----------

qlinks.caging.analysis package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``analysis.environment`` is a role-oriented subpackage that diagnoses whether the exterior environment can be safely removed when
constructing a bounded local caging operator.  It deliberately does **not** classify the caged
eigenstate.  Broader local structure and support morphology are separate analyses.

.. automodule:: qlinks.caging.analysis
   :members:
   :show-inheritance:

.. automodule:: qlinks.caging.analysis.environment
   :members:
   :show-inheritance:

.. automodule:: qlinks.caging.analysis.local_structure
   :members:
   :show-inheritance:

.. automodule:: qlinks.caging.analysis.thermodynamic
   :members:
   :show-inheritance:

qlinks.caging.local_search package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: qlinks.caging.local_search
   :members:
   :show-inheritance:

qlinks.caging.stability package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: qlinks.caging.stability
   :members:
   :show-inheritance:

Core modules
------------

.. automodule:: qlinks.caging.search
   :members:
   :show-inheritance:

.. automodule:: qlinks.caging.solver
   :members:
   :show-inheritance:

Module contents
---------------

.. automodule:: qlinks.caging
   :members:
   :show-inheritance:
