# Test-layer agent rules

This file supplements the repository-root `AGENTS.md`.

- Tests are owned by the behaviour or scientific claim they protect, not by one-to-one mirroring
  of implementation objects.
- Keep fast, integration, scientific, manual, and GPU markers semantically truthful. Do not move a
  test into the fast lane merely to reduce CI time, and do not mark a compact deterministic
  contract scientific merely to avoid fixing it.
- Mirror stable source responsibility boundaries in the test tree. When a source subpackage is
  decomposed, migrate the corresponding broad test module once ownership is clear.
- Prefer public behaviour and mathematical invariants. Direct private-symbol tests are allowed only
  for subtle numerical kernels or mechanisms that cannot be protected adequately through a public
  contract; isolate them in explicitly named internal-kernel/mechanism files.
- Broadly reusable fixtures live under `tests/fixtures/`; domain-specific fixtures belong in the
  nearest `conftest.py`. Delete unused global fixtures.
- Every randomized test uses an explicit seed. Every scientific test states the claim, necessary
  system size, and acceptance criterion in a docstring or nearby comment.
- Do not suppress new qlinks-originated warnings globally. Use `pytest.warns(...)` for expected
  warnings or fix the source.
- `tests/test_health_budget.json` is a regression budget. Existing debt may be ratcheted downward;
  increasing a ceiling requires a deliberate test-architecture decision and an update to
  `tests/TEST_HEALTH_AUDIT.md`.
- Run `python tools/test_health.py --check` after broad test changes and keep the fast lane clean.
