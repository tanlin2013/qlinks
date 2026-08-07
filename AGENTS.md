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
- Do not add package-level re-exports without an explicit public-API decision.
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

The current repository does not fully satisfy these boundaries. New work must move toward
this direction and must not deepen the existing cycle.

## API maturity

Treat interfaces according to four maturity levels:

- **Supported**: documented, package-level only when justified, and covered by behavioural
  compatibility tests.
- **Experimental**: usable by research workflows but allowed to change with clear release
  notes.
- **Internal**: implementation detail; import from its defining module only and do not build
  downstream workflows around it.
- **Deprecated**: compatibility-only, with a replacement path and a planned removal point.

Package-level `__init__.py` exports are reviewed interfaces, not a convenient index of every
object in a subpackage. New research reports, intermediate data classes, and helper functions
should normally stay in their defining modules.

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
- Add regression tests for meaningful failure modes or contracts, not for line-by-line
  implementation mirroring.
- Prefer one parametrized behavioural test over many nearly identical tests.
- Keep random tests reproducible with explicit seeds.
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

1. Run formatting/lint checks for touched Python files.
2. Run the narrowest relevant test lane.
3. Run `scripts/test.sh fast` for package changes unless the environment lacks a documented
   optional dependency.
4. For scientific changes, report the exact command, seed, system size, tolerance, and
   whether the result is a smoke test or production validation.
5. Summarize public API changes, dependency-direction changes, and tests moved between lanes.
6. Do not claim validation that was skipped or could not finish.
