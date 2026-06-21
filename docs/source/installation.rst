Installation
============

Install from PyPI
-----------------

For the core package, use:

.. code-block:: bash

   pip install qlinks

Optional features are split into extras:

.. code-block:: bash

   pip install "qlinks[cpsat]"          # OR-Tools CP-SAT basis solver
   pip install "qlinks[automorphism]"  # pynauty graph automorphisms
   pip install "qlinks[drawing]"       # pyvis, plotly, igraph, pycairo
   pip install "qlinks[distributed]"   # Ray helpers
   pip install "qlinks[storage]"       # HDF5 and parquet-oriented IO

Install from source
-------------------

For development, clone the repository and install with Poetry:

.. code-block:: bash

   git clone https://github.com/tanlin2013/qlinks.git
   cd qlinks
   poetry install --all-extras --with docs
   poetry run pre-commit install

Useful development commands are:

.. code-block:: bash

   poetry run pytest
   poetry run pre-commit run --all-files
   poetry run make -C docs html

Docker
------

A feature-complete Docker image is available from the project repository:

.. code-block:: bash

   docker pull tanlin2013/qlinks:main

The image intentionally installs all extras so that visual, storage,
distributed, and optional solver workflows are available in the same runtime.
