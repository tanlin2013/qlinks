Architecture diagnostics
========================

``qlinks`` includes a repository-level architecture report for inspecting
internal import structure without importing the scientific package.  The report
is generated from Python's abstract syntax tree, uses NetworkX for graph
analysis, and is emitted as a self-contained HTML page plus machine-readable
JSON.

The report is intended for diagnosis rather than as a new package API.  It
shows:

* the top-level package dependency graph and coupling weights;
* a filterable module-level explorer for each top-level package;
* fan-in, fan-out, source-size, and import-reference hotspots;
* strongly connected components (import-cycle candidates); and
* the broad dependency guardrails documented in ``AGENTS.md``.

Interactive report
------------------

The documentation build generates the current report automatically.

.. raw:: html

   <p><a class="reference external" href="_static/architecture/qlinks-architecture.html">Open the interactive qlinks architecture report</a>
   &nbsp;·&nbsp;
   <a class="reference external" href="_static/architecture/qlinks-architecture.json">Download the machine-readable JSON</a></p>
   <iframe
     src="_static/architecture/qlinks-architecture.html"
     title="Interactive qlinks architecture diagnosis"
     style="width:100%;height:760px;border:1px solid #d9d9d9;border-radius:6px;"
     loading="lazy">
   </iframe>

Local diagnosis
---------------

Generate the same HTML without rebuilding all Sphinx pages:

.. code-block:: bash

   uv run make -C docs architecture

The output is written to
``docs/build/html/_static/architecture/qlinks-architecture.html``.  To generate
and open it directly:

.. code-block:: bash

   uv run python tools/architecture_report.py --open

A normal documentation build also refreshes the report:

.. code-block:: bash

   uv run make -C docs html

CI and architecture enforcement
-------------------------------

The documentation workflow runs the normal ``html`` target, so every CI docs
build regenerates the architecture HTML and the deployed Sphinx site contains
the matching report.  The visualization itself is diagnostic: the dedicated
fast tests in ``tests/test_architecture_boundaries.py`` remain the blocking
architecture guardrails.

This separation is deliberate.  A high fan-out or a large module is a signal
for review, not automatically an error, whereas forbidden dependency direction
is an enforceable repository contract.
