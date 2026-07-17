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
   pip install "qlinks[tn]"            # quimb tensor-network backend

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

Docker tensor-network interpreter
---------------------------------

The tensor-network dependencies are intentionally optional.  A normal local
Poetry environment therefore remains usable on Python 3.14 without installing
``quimb``, ``numba``, or ``llvmlite``:

.. code-block:: bash

   poetry install

The Docker image installs only the ``tn`` extra.  Its Linux runtime is isolated
from the host macOS architecture and selects Linux wheels from the lock file.
The lock file records versions, markers, and hashes for multiple platforms; it
does not copy or reuse a wheel installed in the host Poetry environment.

On an Intel Mac, build and load the image with:

.. code-block:: bash

   ./scripts/docker_build_tn.sh

Docker uses the daemon's native Linux architecture by default. To request a
specific target explicitly, set, for example,
``QLINKS_DOCKER_PLATFORM=linux/amd64`` or ``linux/arm64``.

On an Intel Mac, the default is normally ``linux/amd64``. This is equivalent to:

.. code-block:: bash

   docker buildx build \
       --load \
       --platform linux/amd64 \
       --build-arg QLINKS_EXTRAS=tn \
       --tag qlinks:tn \
       .

The ``--load`` flag is important when Docker uses a non-default BuildKit
builder: it places the completed image in Docker's local image store, where
PyCharm can retrieve its image ID.

To verify the image outside the IDE, run:

.. code-block:: bash

   docker run --rm qlinks:tn python scripts/verify_tn_environment.py

For an interactive shell with the repository mounted at the same path used by
the image:

.. code-block:: bash

   ./scripts/docker_run.sh

The image build enforces wheel-only installation for ``numba`` and
``llvmlite``.  If a selected Python/Linux architecture has no compatible
wheel, the build fails immediately rather than attempting to compile LLVM.
Both Linux ``amd64`` and ``arm64`` are supported by the pinned TN stack.

PyCharm setup
~~~~~~~~~~~~~

Keep the existing local Poetry interpreter for the core package.  Add a second
interpreter from the prebuilt Docker image:

#. Open **Settings | Project | Python Interpreter**.
#. Choose **Add Interpreter | On Docker**.
#. Select **Docker Image**, not **Dockerfile**.
#. Choose ``qlinks:tn``.
#. Set the interpreter path to ``/usr/local/bin/python``.

Using the prebuilt image avoids PyCharm's generic
``Can't retrieve image ID from build stream`` wrapper.  If the terminal build
fails, its output contains the actual package or Docker error.  If the terminal
build succeeds but PyCharm still cannot see the image, check:

.. code-block:: bash

   docker context show
   docker image inspect qlinks:tn
   docker buildx ls

PyCharm and the terminal must use the same Docker context.  With a custom
``docker-container`` buildx driver, always build with ``--load`` before
selecting the image in PyCharm.
