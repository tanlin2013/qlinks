# qlinks repository-health audit

Baseline: `qlinks-current-01483f5.zip`, audited 2026-08-14 after the caging/module and T1–T3
test-health refactors.

## Audit conclusion

The repository already had strong qualitative governance in `AGENTS.md`, explicit caging/open-system
boundaries, layered test lanes, CodeQL, blocking format/lint CI, and the T3 test-health budget. The
main remaining risk was that several important rules were still prose-only: a future change could
silently grow a giant module or package API, add a new top-level dependency, recreate an SCC, remove
a health check from CI, or weaken workflow/secret hygiene without a dedicated blocking signal.

This pass converts those rules into a repository-health gate.

## Blocking guardrails added

`python tools/repository_health.py --check` now requires:

- zero static and import-time module/package SCCs;
- zero broad/reviewed architecture-boundary violations;
- no unreviewed new top-level package dependency;
- a 1,500-line default ceiling for new source modules;
- existing oversized modules to stay at or below their grandfathered line ceilings;
- reviewed ceilings for every package `__all__` surface currently tracked;
- implementation modules to import defining child modules instead of an ancestor package API;
- no package import from `experimental/`;
- no common secret-bearing filenames, private-key material, or high-confidence token patterns;
- local/generated workspace directories such as `.venv`, `.tox`, caches, and build outputs are
  excluded from security scanning, while arbitrary `.gitignore` entries are intentionally still scanned;
- a top-level least-privilege `permissions` baseline in every GitHub Actions workflow;
- no `write-all` workflow permission and no obvious floating action refs (`@main`, `@master`, etc.);
- the core pre-commit/CI/test-lane guardrails themselves to remain wired.

The budget lives at `tools/repository_health_budget.json`. It is intentionally a regression budget:
existing oversized modules are grandfathered, but they cannot grow without an explicit reviewed
budget change. When debt shrinks, the corresponding limit should be ratcheted downward.

## Local and CI enforcement

Commit-time pre-commit now runs repository health in addition to formatting/lint hygiene and
private-key detection. Pre-push runs the lightweight test-health guardrail; the full fast test lane is intentionally CI-owned to keep local laptop development responsive. Blocking lint CI also
runs repository health, while the existing coverage-bearing fast job continues to run test health.

Every GitHub workflow now declares a read-only token baseline. Documentation deployment was split
from documentation building so write permission exists only in the main-branch deploy job.

## Current baseline

- Python modules: **192**
- source lines: **101,083**
- top-level package dependency edges: **48**
- static module SCCs: **0**
- static package SCCs: **0**
- import-time module SCCs: **0**
- architecture boundary violations: **0**
- grandfathered source modules above 1,500 lines: **12**
- tracked package-level public APIs: **27**
- sensitive filename findings: **0**
- high-confidence secret-pattern findings: **0**
- workflow permission findings: **0**
- floating action findings: **0**
- guardrail-wiring findings: **0**

Test-architecture debt is tracked separately in `tests/TEST_HEALTH_AUDIT.md` and
`tests/test_health_budget.json`.

The `qlinks.visualizer.basis` 6,500-line grandfathered module was removed in the first debt-
ratcheting pass and replaced by a role-oriented subpackage whose child modules all satisfy the
normal 1,500-line ceiling. Its grandfathered budget entry was deleted rather than transferred.

The former 3,825-line `qlinks.open_system.diagnostics` module was likewise removed from the
grandfathered list after decomposition into focused diagnostics modules below the normal ceiling.
Generic common-kernel/nullspace algebra was extracted to the lower internal
`qlinks.open_system._subspace` layer so manifold-detector code does not depend on diagnostics
internals.

The former 2,847-line `qlinks.caging.analysis.environment` module was removed from the
grandfathered list after decomposition into a role-oriented environment-reduction subpackage.
The corrected scientific boundary is now mechanically represented: transition signatures remain a
shared analysis leaf, reduced local-operator application is separated from mechanism annotation,
and the user-facing report/orchestrator sit above those primitives without static cycles.

## Remaining known debt / non-blocking checks

The repository still has 12 oversized legacy modules. Their current sizes are frozen rather than
being treated as acceptable targets; future decomposition should ratchet those limits downward.

Mypy and broad Bandit reporting remain advisory. Dependency-vulnerability scanning is not promoted
to a new blocking dependency in this pass because the current `safety` CLI is unsuitable for
non-interactive CI in this repository. CodeQL remains enabled. If the project begins handling live
service/broker credentials or private financial data, the next security step should be a pinned,
non-interactive dependency scanner and stronger immutable pinning for security-sensitive Actions.
