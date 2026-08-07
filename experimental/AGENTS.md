# Experimental-workflow agent rules

This file supplements the repository-root `AGENTS.md`.

- `experimental/` may contain expensive provisioning, notebooks, evidence jobs, and research
  prototypes that are not stable package APIs.
- Keep generated data, rendered figures, caches, and production outputs out of source control
  unless a small fixture is explicitly required for a regression test.
- Record exact parameters, random seeds, environment information, and output paths for
  scientific runs.
- Promote code into `qlinks/` only after its responsibility, dependency direction, public API,
  and test lane have been reviewed.
- A notebook result should be converted into a deterministic small test only when a compact
  invariant exists; do not copy the production computation into the unit suite.
