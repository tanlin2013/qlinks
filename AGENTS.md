# AGENTS.md

This file defines the repository-wide rules for human contributors and coding agents.
Its purpose is to keep rapid scientific development compatible with reviewable software,
reproducible numerical claims, and a bounded maintenance burden.

## Repository intent

`qlinks` is both a reusable Python package and an active research codebase. These roles
must be separated explicitly:

1. **Stable library code** has reviewed interfaces, bounded dependencies, and fast tests.
2. **Experimental package code** may evolve quickly, but must remain clearly labelled and
   must not silently acquire stable-API obligations.
3. **Provisioning and evidence workflows** belong under `experimental/` or dedicated job
   scripts. Expensive scientific validation is not part of the unit-test lane.

A research result does not become a supported package API merely because it is useful in a
notebook or has one successful numerical run.

## Change discipline

- Make the smallest coherent change that addresses the stated problem.
- Do not combine scientific changes, API redesign, broad formatting, and unrelated cleanup
  in one patch.
- Before adding a new class or workflow to a large module, identify whether it introduces a
  new responsibility. If it does, create a focused module instead.
- Modules above roughly 1,500 lines should not receive a new responsibility without an
  explicit decomposition plan.
- Reuse an existing implementation or extract a shared primitive instead of creating a
  second near-duplicate implementation.
- Do not preserve deprecated private internals solely because tests import them directly.
  Preserve public behaviour through a small compatibility facade and black-box tests.
- Compatibility facades are temporary migration scaffolding, not an architecture layer. New
  package code must import the focused replacement modules directly, not the facade.
- Every compatibility facade must name its replacement path and a removal gate. Do not add new
  functionality to a facade merely to keep old imports convenient.
- Do not add package-level re-exports without an explicit public-API decision.
- Once a coherent module family has stable responsibility boundaries, prefer a nested subpackage over a long flat filename prefix. Keep subpackage `__init__.py` files curated; implementation modules should import siblings directly rather than through the subpackage API.
- Do not silently change tolerances, convergence criteria, basis conventions, winding
  conventions, or normalization rules. Document the mathematical or numerical reason.
- Do not hide numerical failures by returning empty results, zero arrays, or partially valid
  reports. Fail with enough context to diagnose the scientific assumption that broke.

## Architecture boundaries

The intended dependency direction is:

```text
basis / constraints / lattice / models / operators
                      |
                      v
          neutral local-structure primitives
                 /                 \
                v                   v
             caging            open_system
```

Repository-level rules:

- `qlinks.caging` must not depend on `qlinks.open_system`.
- Shared local reduced-density-matrix, local matrix-unit, pattern-embedding, and local
  operator utilities belong in a neutral lower layer, not in Lindblad-specific modules.
- `qlinks.open_system` should consume compact state/operator protocols rather than concrete
  cage-search implementation objects where practical.
- Visualization, reporting, and serialization helpers must not become hidden dependencies
  of numerical kernels.
- New cross-layer imports require an architecture note in the pull request.
- In caging analysis, environment removability is a property of the local-operator construction,
  not a taxonomy of eigenstates. Do not use environment-removal mechanisms as cage-state classes.
  Same-pattern removal requires equality of the support-aware weighted local transition signature.

The active caging and open-system paths satisfy this boundary. The explicitly deprecated
open-system cage constructors retain compatibility imports from caging; do not expand that
exception or use it as precedent for new code.

## API maturity

Treat interfaces according to four maturity levels:

- **Supported**: documented, package-level only when justified, and covered by behavioural
  compatibility tests.
- **Experimental**: usable by research workflows but allowed to change with clear release
  notes.
- **Internal**: implementation detail; import from its defining module only and do not build
  downstream workflows around it.
- **Deprecated**: compatibility-only, with a replacement path and a planned removal point. Once
  first-party callers have migrated and the replacement API is declared stable, remove the
  bridge rather than carrying it indefinitely.

Package-level `__init__.py` exports are reviewed interfaces, not a convenient index of every
object in a subpackage. New research reports, intermediate data classes, and helper functions
should normally stay in their defining modules.

### Temporary compatibility lifecycle

A refactor compatibility layer may exist only while callers are migrating. It is ready for
removal when all of the following are true:

1. the replacement module boundaries and names have survived the current review/refactor pass;
2. first-party package code, tests, scripts, notebooks, and documentation use the replacement;
3. public compatibility tests have been rewritten against the supported replacement API; and
4. the removal is recorded in the changelog or the next release/refactor milestone.

Compatibility modules must not become dependencies of active implementation code. Architecture
tests should enforce this whenever a facade is introduced. Once the reviewed replacement API
stabilizes, delete the facade rather than retaining it as a permanent public layer.

## Testing policy

Tests are classified by purpose, not by directory size or implementation object count.

### Unit tests

- Exercise one behavioural contract or mathematical invariant on a tiny deterministic case.
- Must not perform production-size exact diagonalization, tensor-network optimization,
  scaling studies, distributed execution, or long stochastic sampling.
- Should normally finish well below one second individually.
- Run on every supported Python version.

### Integration tests

- Exercise a complete cross-module workflow on a deliberately small system.
- May build a model, run a small cage search, construct a Lindbladian, or validate an optional
  backend.
- Run on the default Python version in pull requests; optional backends may use a dedicated
  supported Python version.

### Scientific tests

- Validate a research claim, finite-size trend, optimization, cross-size construction, or
  expensive numerical regression.
- Must use `@pytest.mark.scientific` and should also be integration-level.
- Run on a schedule or by explicit workflow dispatch, not in the fast pull-request lane.
- Their docstrings or nearby comments must state the scientific claim being protected and
  why the chosen system size is needed.

### Manual and GPU tests

- `manual` requires human inspection or explicit local activation.
- `gpu` requires a real CUDA-capable runtime.
- These markers are orthogonal to the unit/integration/scientific taxonomy.

### Test design rules

- Test public behaviour and scientific invariants rather than every private helper.
- When a source family has stable nested responsibility boundaries, mirror those responsibilities
  in the test tree rather than keeping a single pre-refactor monolithic test module.
- If a private numerical kernel genuinely needs direct tests, isolate those contracts in an
  explicitly named internal-kernel test module; do not spread private imports through public
  behavioural tests or game health metrics by hiding the same coupling behind module attributes.
- Add regression tests for meaningful failure modes or contracts, not for line-by-line
  implementation mirroring.
- Prefer one parametrized behavioural test over many nearly identical tests.
- Keep random tests reproducible with explicit seeds.
- Keep broadly shared fixtures under `tests/fixtures/`; domain-specific fixtures belong in the
  nearest test-package `conftest.py`. Delete unused global fixtures instead of retaining them for
  hypothetical future tests.
- Run the narrowest relevant lane before broader lanes.
- Do not weaken a failing assertion merely to make CI pass; determine whether the code,
  invariant, fixture, or maturity classification is wrong.

Use:

```bash
scripts/test.sh fast
scripts/test.sh integration
scripts/test.sh scientific
scripts/test.sh all
```

See `docs/contributing/5.-testing.md` for lane details.

The maintained test-suite health cache lives at `tests/TEST_HEALTH_AUDIT.md`. Update its
remediation status after repository-wide test taxonomy or ownership changes so future
refactors can distinguish known debt from regressions. Run `python tools/test_health.py --check`
after broad test changes; `tests/test_health_budget.json` is a deliberate regression budget, not
a target to game by hiding imports or markers.

## Scientific review requirements

For caging changes, preserve or explicitly re-evaluate as applicable:

- eigenpair residuals;
- leakage or boundary-cancellation residuals;
- support, sector, and basis consistency;
- chiral grading and uniform-potential-shell assumptions for type-1 constructions;
- distinction between analytic certification, exact numerical verification, and empirical
  evidence;
- finite-size limitations of any thermodynamic claim.

For open-system changes, preserve or explicitly re-evaluate as applicable:

- target-state darkness residuals;
- Hamiltonian invariance of the selected common kernel;
- trace and Hermiticity preservation;
- positivity or complete-positivity conditions where numerically testable;
- inflow/recycling assumptions;
- avoidance of dense Liouvillian construction except for explicitly small systems.

A numerical result is not self-explanatory. New diagnostics must state the convention,
normalization, tolerance, and failure interpretation.

## Validation before handoff

Before presenting a change:

1. Run `flake8` on every touched Python file, or the repository-wide flake8 command when the
   change is broad. A Python patch is not lint-validated until flake8 passes with no errors.
2. Run Black and isort when feasible. For a heavy scientific/refactor task they may be deferred
   to the maintainer, but the handoff must say explicitly that they were not run. Do not defer
   flake8 for ordinary Python patches.
3. Run the narrowest relevant test lane.
4. Run `scripts/test.sh fast` for package changes unless the environment lacks a documented
   optional dependency.
5. For scientific changes, report the exact command, seed, system size, tolerance, and
   whether the result is a smoke test or production validation.
6. Summarize public API changes, dependency-direction changes, and tests moved between lanes.
7. Do not claim validation that was skipped or could not finish.

### Patch handoff rules

When delivering a git patch:

- Generate it relative to the repository root with ordinary git paths (`a/...` and `b/...`).
- The recipient must be able to apply it from the repository root without path rewriting,
  `--directory`, or manual prefix stripping.
- Before handoff, validate the exact artifact against a clean copy of the stated base with:

  ```bash
  git apply --check /path/to/change.patch
  ```

  Run that command from the repository root with no `-p`/prefix adjustment, then apply the
  patch to the clean copy and run the required lint/tests there.
- If the patch was generated from a reconstructed archive rather than the original git
  checkout, say which archive was used as the base.
