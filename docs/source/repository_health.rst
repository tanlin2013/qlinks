Repository health and governance
================================

The repository treats architecture and test health as executable contracts rather than reviewer
memory. The authoritative behavioural rules live in the root ``AGENTS.md`` and its nested layer
files; this page summarizes the mechanical guardrails used by local hooks and CI.

Repository-health check
-----------------------

Run:

.. code-block:: bash

   poetry run python tools/repository_health.py --check

The budget in ``tools/repository_health_budget.json`` enforces:

* zero static/import-time module and package strongly connected components;
* zero reviewed architecture-boundary violations;
* the current top-level package dependency topology unless a new dependency is explicitly reviewed;
* a 1,500-line default ceiling for new source modules, with existing oversized modules
  grandfathered at fixed ceilings;
* reviewed ceilings for package-level ``__all__`` surfaces;
* direct child-module imports inside a package instead of back-importing an ancestor package API;
* no package imports from ``experimental``;
* high-confidence sensitive-file/secret checks; and
* least-privilege GitHub workflow baselines and rejection of obvious floating action refs.

Budgets are regression gates, not goals. When an oversized module or broad API shrinks, ratchet its
budget downward. Raising a budget requires the same review as the architecture/API change itself.

Test-health check
-----------------

Run:

.. code-block:: bash

   poetry run python tools/test_health.py --check

``tests/test_health_budget.json`` prevents known test debt from growing silently. The qualitative
history is maintained in ``tests/TEST_HEALTH_AUDIT.md``.

Local hooks and CI
------------------

Install both commit and push hooks:

.. code-block:: bash

   poetry run pre-commit install
   poetry run pre-commit install --hook-type pre-push

Commit-time hooks run formatting/lint hygiene plus repository health. Pre-push hooks additionally
run test health, Poetry/lock checks, and the fast test lane. Integration remains a pull-request CI
job; scientific validation remains scheduled or explicitly dispatched.

Security
--------

``SECURITY.md`` defines credential and vulnerability-reporting policy. Workflows use a read-only
``GITHUB_TOKEN`` baseline and grant write permissions only to jobs that need them. Private keys and
common secret-bearing file names are rejected before CI by both pre-commit and repository-health
checks.
