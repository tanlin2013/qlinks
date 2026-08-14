Contributing
============

Development happens on GitHub.  The repository-root ``AGENTS.md`` defines the
architecture, API-maturity, testing, and scientific-review rules.  Additional
contributor notes live under ``docs/contributing``.

For local development, install the development group and only the optional
extras needed for the change:

.. code-block:: bash

   poetry install --with dev
   poetry install --with dev --extras "storage drawing"
   poetry run pre-commit install

The standard pull-request checks are split by purpose:

.. code-block:: bash

   scripts/test.sh fast
   scripts/test.sh integration
   scripts/lint_blocking.sh
   poetry run python tools/repository_health.py --check
   poetry run python tools/test_health.py --check

Expensive finite-size, optimization, and research-claim validation belongs in
the scientific lane:

.. code-block:: bash

   scripts/test.sh scientific

The scientific lane is scheduled or explicitly dispatched and is not part of
the fast compatibility matrix.  Tests protect behavioural contracts and
mathematical invariants; the project does not require one test object per code
object or direct tests of every private helper.

The project uses Black and isort for formatting, flake8 for blocking lint checks,
and mypy as advisory typing feedback.  Optional dependencies are grouped by
feature so core model-building workflows do not require every visualization,
storage, distributed, CP-SAT, or tensor-network dependency.

Repository-wide architecture/API/security budgets are described in
:doc:`repository_health`. Install both commit and pre-push hooks so the same constraints run
locally before CI.
