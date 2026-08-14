qlinks.open_system package
==========================

The open-system package provides Lindblad operators, Liouvillian solvers,
Monte-Carlo wavefunction sampling, random state helpers, and dark-state
diagnostics.  Shared local-RDM and local-operator algebra is owned by
:mod:`qlinks.local_structure`; ``local_recycling`` contains the Lindblad-specific
selection and recycling workflow built on those primitives.

Submodules
----------

qlinks.open_system.backend module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: qlinks.open_system.backend
   :members:
   :show-inheritance:

qlinks.open_system.diagnostics package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: qlinks.open_system.diagnostics
   :members:
   :show-inheritance:

The diagnostics implementation is split by responsibility into target-manifold observables,
jump diagnostics, evolution analysis, verification, monitor-kernel closure, dark-manifold
diagnostics, and absorbing-projector diagnostics. Active package code imports these focused
child modules directly.

qlinks.open_system.local_recycling module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: qlinks.open_system.local_recycling
   :members:
   :show-inheritance:

qlinks.open_system.operators module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: qlinks.open_system.operators
   :members:
   :show-inheritance:

qlinks.open_system.protocols module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: qlinks.open_system.protocols
   :members:
   :show-inheritance:

qlinks.open_system.solvers module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: qlinks.open_system.solvers
   :members:
   :show-inheritance:

qlinks.open_system.states module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: qlinks.open_system.states
   :members:
   :show-inheritance:

qlinks.open_system.stochastic_schrodinger module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: qlinks.open_system.stochastic_schrodinger
   :members:
   :show-inheritance:

Module contents
---------------

.. automodule:: qlinks.open_system
   :members:
   :show-inheritance:
