Contributing
============

Development happens on GitHub.  The repository includes contributing notes,
coding standards, code-of-conduct text, and acknowledgements under the
``docs/contributing`` directory.

For local development, install all extras and the documentation group:

.. code-block:: bash

   poetry install --all-extras --with docs
   poetry run pre-commit install

Common checks are:

.. code-block:: bash

   poetry run pytest
   poetry run pre-commit run --all-files
   poetry run make -C docs html

The project uses Black and isort for formatting, flake8 for blocking lint checks,
and mypy as advisory typing feedback.  Optional dependencies are grouped by
feature so that core model-building workflows do not require every visualization,
storage, distributed, or CP-SAT dependency.
