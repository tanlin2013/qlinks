# qlinks test-suite health audit

## MCWF optimization-debt follow-up (2026-08-14)

The stochastic-Schrödinger simplification removed the private contracts that existed only to
protect unsupported micro-optimizations. The structural sparse/vectorized MCWF tests now cover
operator preservation, total-rate consistency, channel-rate correctness, serial chunking, and
``total-rate-first`` equivalence. Direct private-symbol imports fall from **48** to **26**, and the
regression budget is ratcheted to that new ceiling rather than retaining deleted debt.

The current collection after this cleanup is **1,443** cases with **1,370** selected by the fast
lane; integration/scientific/manual marker counts are unchanged.

Baseline audit: `qlinks-current-d2e041e.zip` after the caging analysis/environment refactor.

## Repository-health guardrail follow-up (2026-08-14)

The repository-health enforcement and subsequent architecture-decomposition passes add fast
regression tests for blocking architecture/API/security budgets and reviewed nested-module DAGs.
The current collection is:

- Python files under `tests/`: **191**;
- test LOC including fixtures/helpers: **38,296**;
- AST test functions: **1,406**;
- pytest collected cases: **1,459**;
- default fast selection: **1,386** cases;
- integration: **40** cases;
- scientific: **7** cases;
- manual: **29** cases;
- GPU: **4** cases;
- direct private-symbol imports: **48**;
- globally registered fixtures: **11**, all used;
- unmarked manual-visual cases: **0**;
- largest test file: **1,337** lines.

`python tools/test_health.py --check` remains **PASS**.


## T3 remediation status (2026-08-14)

The fixture/helper hygiene and CI-quality pass is implemented against
`qlinks-current-75fab76.zip`. The goal is to keep the T1/T2 improvements observable and prevent
known structural debt from growing silently.

Completed in T3:

- removed 13 unused globally registered lattice/layout/model fixtures;
- moved the three environment-reduction fixtures from the root plugin registry to
  `tests/caging/analysis/conftest.py`;
- fixed the three warning sources seen in the fast lane: parallel MCWF chunks now use an explicit
  spawn multiprocessing context, undirected NetworkX dimer edges no longer pass an inapplicable
  `connectionstyle`, and animation smoke tests initialize the returned Matplotlib animation before
  disposal;
- promoted warnings attributed to qlinks modules to pytest errors, so new package-originated
  warnings cannot accumulate unnoticed;
- added `tools/test_health.py` and `tests/test_health_budget.json`; the reporter measures test LOC,
  collected/fast/marker counts, largest files, direct private imports, globally registered fixture
  usage, and unmarked manual-visual cases;
- added the budgeted test-health check to the coverage-bearing fast CI job and publish its Markdown
  snapshot to the GitHub job summary.

The budget is a regression ceiling/floor, not a score to optimize mechanically. In particular, the
48 intentional private imports retained after T2 remain visible rather than being hidden behind
module attribute access.

Post-T3 snapshot:

- Python files under `tests/`: **190**;
- test LOC including fixtures/helpers: **38,051**;
- AST test functions: **1,397**;
- pytest collected cases: **1,450**;
- default fast selection: **1,377** cases;
- integration: **40** cases;
- scientific: **7** cases;
- manual: **29** cases;
- GPU: **4** cases;
- direct private-symbol imports: **48**;
- globally registered fixtures: **11**, all used;
- unmarked manual-visual cases: **0**;
- largest test file: **1,337** lines.

T3 validation in the audit environment:

- fixture-dependent focused suite: **104 passed, 2 skipped**;
- warning-source focused suite: **5 passed** with no warnings;
- default lane: **1357 passed, 20 skipped, 73 deselected** with **0 warnings**;
- `python tools/test_health.py --check`: **PASS**.

## T2 remediation status (2026-08-14)

The ownership/decomposition pass is implemented against `qlinks-current-8af7893.zip`. The goal
was to make the physical test layout follow the responsibility boundaries established in the
source refactor without changing scientific assertions.

Completed in T2:

- split the former 1,702-line `caging/test_local_search.py` into
  `tests/caging/local_search/` by core, QDM, proposal, scan, certification, factorized, and
  workflow responsibilities;
- split the former 1,078-line `caging/test_stability.py` into
  `tests/caging/stability/` by core, topology, boundary, QDM, and Laurent responsibilities;
- split the former `open_system/test_manifold_detectors.py` into dark, recycling, residual, and
  readout contracts under `tests/open_system/manifold_detectors/`;
- split the former 2,004-line stochastic-Schrödinger test into primitive, trajectory, ensemble,
  storage/streaming, and optimized sparse-kernel contracts; the 27 intentional private numerical
  kernel imports now live only in `test_sparse_kernels.py`;
- split environment-reduction coverage into public scenarios, internal mechanism contracts,
  collective-cancellation mechanisms, and support morphology; the 9 direct private environment
  imports now live only in `test_environment_mechanisms.py`;
- remove two duplicate `basis_configs_from_basis` tests from the environment suite because the
  same contracts are already owned by `tests/basis/test_configs.py`.

Test-body preservation check:

- **220** targeted pre-T2 test functions were compared by AST against the reorganized tree;
- **218** are AST-identical after the move;
- the only two removed tests are the deliberately deduplicated basis-config cases above.

Post-T2 snapshot:

- Python files under `tests/`: **190**;
- test LOC including fixtures/helpers: **37,959**;
- AST test functions: **1,397**;
- pytest collected cases: **1,450**;
- default fast selection: **1,377** cases;
- integration: **40** cases;
- scientific: **7** cases;
- manual: **29** cases;
- direct private-symbol imports: **48**.

The private-import count is intentionally not reduced by hiding attribute access. Instead, it is
now concentrated in explicit internal-contract modules: 27 MCWF optimized-kernel imports in
`stochastic_schrodinger/test_sparse_kernels.py`, 9 environment mechanism imports in
`analysis/test_environment_mechanisms.py`, and 3 collective-environment mechanism imports in
`test_environment_collective.py`; the remaining 9 are visualizer/distributed internal tests.

T2 validation in the audit environment:

- focused reorganized ownership suite: **218 passed, 7 deselected** in 8.10 s;
- default lane: **1357 passed, 20 skipped, 73 deselected** in 22.23 s;
- no targeted test body changed semantically according to the AST comparison described above.

T3 fixture/helper hygiene and CI-quality work is recorded above.

## T1 remediation status (2026-08-14)

The first remediation pass is implemented. This cache is intentionally retained so later test
refactors can be checked against the original findings rather than rediscovering them.

Completed in T1:

- the two visual-only modules now carry the `manual` marker in addition to their
  `QLINKS_SHOW_PLOTS` guard, moving 22 visual cases out of the default lane;
- the toric-code QEC sanity workflow and tensor-network visualizer tests are classified as
  `integration`;
- square-QDM singlet-product tests are classified as `integration`, with the 6x6 exact-cover/no-go
  claim additionally classified as `scientific`;
- physical QDM periodic-product scaling, the 8x4 collective product-kernel check, and the
  fixed-width compact-QDM reduced-winding check, and spin-1 XY imaginary-J2 tower-preservation
  check are classified as `integration` + `scientific`;
- deprecated single-cage Lindblad coverage was reduced from 32 white-box tests with 23 private
  imports to 3 black-box compatibility tests with no private imports;
- architecture-boundary AST imports are cached, and architecture-report analysis is reused across
  its tests;
- public-API contract tests now include `qlinks.caging.analysis`,
  `qlinks.caging.local_search`, and `qlinks.caging.stability`.

Post-T1 snapshot:

- Python files under `tests/`: **163**;
- test LOC including fixtures/helpers: **38,033**;
- AST test functions: **1,392**;
- pytest collected cases: **1,452**;
- default fast selection: **1,379** cases;
- integration: **40** cases;
- scientific: **7** cases;
- manual: **29** cases;
- direct private-symbol imports: **48**, down from **71**.

The remaining private-import debt is now concentrated in `test_stochastic_schrodinger.py` (27),
`caging/analysis/test_environment.py` (9), and a few visualizer/internal-kernel tests. These are
T2 ownership/decomposition work rather than blockers for T1.

T1 validation in the audit environment:

- default lane: **1359 passed, 20 skipped, 73 deselected** in 31.72 s;
- newly reclassified workflows outside the pre-existing caging tensor-network file: **18 passed**
  in 10.94 s;
- architecture boundary/report tests: **10 passed** in 2.31 s;
- the complete non-scientific integration lane still exceeds the local command window because of
  the pre-existing tensor-network integration workload.

## Executive diagnosis

The suite is substantially healthier than before the repository refactor: the default lane is mostly fast, random tests are seeded, there are no sleep-based tests, architecture guardrails exist, and expensive tensor-network claims are at least partly separated.

The remaining problem is no longer simply "tests are too fat". The main issue is that **test ownership and taxonomy still mirror the pre-refactor implementation**. A small number of very large test modules import many private helpers, several research/integration checks remain in the default lane, deprecated open-system code still carries a large white-box test burden, and global fixture/plugin infrastructure contains dead or overly broad fixtures.

## Inventory

- Python test files: **163**
- Test LOC: **38,981**
- AST test functions: **1,428**
- Pytest collected cases: **1,481**
- Default fast selection: **1,448** cases (33 deselected)
- Integration: **22** cases, all from `tests/caging/test_tensor_network.py`
- Scientific: **2** cases, both tensor-network tests
- Manual marker: **7** collected cases
- GPU marker: **4** cases

Largest test areas:

| Area | files | LOC | test functions |
|---|---:|---:|---:|
| caging | 31 | 10,479 | 302 |
| open_system | 15 | 6,883 | 252 |
| visualizer | 15 | 6,692 | 232 |
| models | 13 | 3,230 | 124 |
| constraints | 13 | 2,130 | 95 |
| basis | 8 | 1,917 | 79 |

Largest individual files:

| File | LOC | tests |
|---|---:|---:|
| `open_system/test_stochastic_schrodinger.py` | 2,004 | 80 |
| `caging/test_local_search.py` | 1,702 | 42 |
| `caging/analysis/test_environment.py` | 1,394 | 29 |
| `visualizer/test_hamiltonian_graph.py` | 1,337 | 54 |
| `visualizer/test_basis_symbols.py` | 1,303 | 47 |
| `basis/test_dfs.py` | 1,177 | 38 |
| `open_system/constructions/test_cage.py` | 1,102 | 32 |
| `caging/test_stability.py` | 1,071 | 37 |
| `open_system/test_manifold_detectors.py` | 1,069 | 32 |

## Runtime profile

Representative default-lane package timings in the audit environment:

- caging: **277 passed, 26 deselected in 12.97 s**
- open_system: **249 passed, 3 deselected in 1.32 s**
- visualizer: **203 passed, 33 skipped, 2 deselected in 7.84 s**
- models: **160 passed in 3.94 s**
- qec: **36 passed in 3.89 s**
- architecture/local-structure root checks: **26 passed, 1 skipped in 6.22 s**

Important default-lane hotspots:

- `qec/test_toric_code_sanity.py`: one test ~2.28 s, module ~3.9 s; cross-module exact-ground-space workflow.
- `visualizer/test_tensor_network.py::...type1_diagnostics`: ~2.15 s; cross-layer tensor-network visualization.
- architecture guardrails repeatedly reparse the repository; individual checks are ~0.7--1.3 s and report-generation tests ~1.1--1.3 s.
- several physical caging checks are 0.5--1.1 s and validate multi-size/QDM claims rather than isolated unit contracts.

The suite is therefore fast enough for ordinary development, but the **fast lane contains integration/scientific semantics that should not be there**.

## Major findings

### P0 — Taxonomy does not reflect test purpose

The integration lane contains only 22 tensor-network tests. Many other tests clearly exercise cross-module workflows but are unmarked. Examples include the toric-code QEC sanity workflow, tensor-network visualization, robust QDM cage searches, and physical periodic-product diagnostics.

Only two tests are marked scientific. Claim-bearing candidates that deserve explicit scientific review include:

- 6x6 singlet exact-cover/no-go evidence;
- physical periodic-product cancellation scaling from actual QDM flips;
- 8x4 collective QDM local-grammar/product-kernel check;
- fixed-width compact-QDM reduced-winding check;
- spin-1 XY imaginary-J2 family preserving the pi tower.

Not every scaling-named test should become scientific: toy-matrix algebraic unit tests can remain fast. The criterion should be whether the test protects a research claim or finite-size conclusion, not runtime alone.

### P0 — Manual visual tests are skipped, not classified

Two manual visualization modules use module-level `skipif(QLINKS_SHOW_PLOTS != 1)` but are not marked `manual`. This produces **22 manual visual skips inside the default lane**. They should carry both the `manual` marker and the environment guard, so `scripts/test.sh fast` deselects them rather than collecting/skipping them.

### P0 — Deprecated open-system code is over-tested white-box style

`tests/open_system/constructions/test_cage.py` is ~1,102 LOC / 32 tests and directly imports **23 private helpers** from the deprecated cage constructor. This contradicts the current repository policy: deprecated code should receive compatibility fixes, not permanent private-helper obligations.

Recommended target: retain a small set of black-box migration/compatibility tests for the deprecated constructor, then delete private-helper tests rather than preserving deprecated internals for them.

### P1 — Private implementation coupling remains high

There are **71 direct imports of private qlinks symbols** across eight test files:

- 27 — `open_system/test_stochastic_schrodinger.py`
- 23 — deprecated `open_system/constructions/test_cage.py`
- 9 — `caging/analysis/test_environment.py`
- 5 — `visualizer/test_hamiltonian_graph.py`
- 3 — `caging/analysis/test_environment_collective.py`
- 2 — `visualizer/test_basis_grid.py`
- 1 each — distributed and basis-symbol tests

Some numerical kernels merit direct low-level tests, but then their ownership should be explicit (for example a focused internal kernel module) rather than a 2,000-line behavioural file importing many `_...` functions from another 2,000+ line module.

### P1 — Test organization has not caught up with nested source packages

The source now has `caging.local_search`, `caging.stability`, and `caging.analysis`, but tests still retain large flat files such as `test_local_search.py` and `test_stability.py`.

Recommended physical structure:

```text
tests/caging/
  local_search/
    test_core.py
    test_qdm.py
    test_padding.py
    test_factorized.py
    test_certification.py
    test_proposals.py
    test_scan.py
    test_workflows.py
  stability/
    test_core.py
    test_boundary.py
    test_topology.py
    test_qdm.py
    test_laurent.py
  analysis/
    ...
```

This is responsibility mirroring, not the old one-test-object-per-code-object rule.

The same applies to the former `open_system/manifold_detectors` test: split it by `manifold_dark`, `manifold_recycling`, and `manifold_residual` contracts.

### P1 — A few tests are themselves mini-programs

- 53 test functions are >=50 lines.
- 4 are >=100 lines.
- The largest environment-reduction test is 116 lines with 62 assertions.

The environment tests in particular should factor reusable case construction into named toy-system builders and assert report-level invariants in smaller scenario tests. Avoid turning them into implementation-helper tests while doing this.

### P1 — Architecture tests are valuable but reparsing is wasteful

The architecture guardrails are excellent and should remain blocking. However, several tests independently walk/parse the repository, causing ~4--6 s of default-lane cost.

Recommended change: build the import/AST model once with a session-scoped test fixture (or call the architecture analyzer once), then have individual assertions query that immutable result. Keep one HTML/JSON rendering smoke test; the full rendering test can be integration if necessary.

### P2 — Fixture registry has dead/global fixtures

The dynamic global fixture plugin is compact, but many registered fixtures have no use outside the fixture registry. Examples include several 4-state layouts, PBC lattices, triangular/honeycomb model fixtures, and QLM fixtures.

Environment-reduction fixtures are useful but highly domain-specific and globally registered. They would be cleaner in `tests/caging/analysis/conftest.py` or local helper factories.

Prune dead fixtures before adding more. There are also at least two unused public test helpers (`as_csr`, `binary_product_states`).

### P2 — Public API contract tests lag the new package structure

`tests/test_public_api.py` validates the old top-level module set, but does not yet cover the new curated subpackage APIs such as:

- `qlinks.caging.analysis`
- `qlinks.caging.local_search`
- `qlinks.caging.stability`

It also does not encode a deliberate maturity decision for packages such as `qec` and `local_structure`. The test should be driven by an explicit supported/experimental API manifest rather than accumulating module names ad hoc.

### P2 — Warning noise should be reduced

Current default tests emit known warnings from:

- NetworkX drawing (`connectionstyle` with LineCollection);
- Matplotlib animations deleted without rendering;
- Python 3.13 multiprocessing `fork()` from a multithreaded process in MCWF tests.

After fixing/explicitly asserting expected warnings, enable stricter warning handling for qlinks-originated warnings so new regressions do not disappear in noise.

## Healthy aspects worth preserving

- Random/stochastic tests use explicit deterministic seeds.
- No sleep-based timing tests were found.
- No wildcard qlinks imports were found.
- No autouse fixture network was found.
- Most unit packages are extremely quick (<0.3 s per package).
- Caging is now ~13 s despite substantial scientific breadth.
- Architecture boundary tests exist and catch import-direction regressions.
- Tensor-network integration/scientific separation is already a good starting model.

## Recommended remediation sequence

### Pass T1 — taxonomy + obvious debt

1. Correct manual markers on visual-only modules.
2. Review claim-bearing/cross-module tests and move them to `integration` / `scientific` by purpose.
3. Demote deprecated open-system private-helper tests to a small black-box compatibility suite.
4. Optimize architecture tests to reuse one parsed repository model.
5. Extend public-API tests to the newly supported nested caging subpackages.

This pass should mostly move/relabel tests, not alter scientific algorithms.

### Pass T2 — test ownership / decomposition

1. Split `test_local_search.py` into `tests/caging/local_search/`.
2. Split `test_stability.py` into `tests/caging/stability/`.
3. Split `test_manifold_detectors.py` by the new open-system modules.
4. Split `test_stochastic_schrodinger.py` by behavioural responsibility; decide which private numerical kernels deserve explicit internal-module contracts.
5. Split `analysis/test_environment.py` into public environment-reduction scenarios vs transition-pattern/mechanism tests.

### Pass T3 — fixture/helper hygiene + CI quality

1. Remove unused fixtures/helpers and localize domain-specific fixtures.
2. Reduce warning noise and consider warning-as-error for qlinks warnings.
3. Add a lightweight test-health audit script/report (LOC, private imports, marker/lane counts, slowest tests, unused fixture detection) so the suite does not regress silently.
4. Reassess coverage only after taxonomy is correct; do not use a high global coverage number as a substitute for scientific/behavioural contracts.
