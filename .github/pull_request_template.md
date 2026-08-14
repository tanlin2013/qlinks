## Scope

<!-- State the responsibility/layer being changed and why. -->

## Health checklist

- [ ] The change stays within the documented responsibility/ownership boundary, or the architecture
      change is explained explicitly.
- [ ] `python tools/repository_health.py --check` passes for package/workflow/structure changes.
- [ ] `python tools/test_health.py --check` passes for broad test changes.
- [ ] Public API additions/removals are deliberate; `__all__`/API-budget changes are explained.
- [ ] New top-level dependencies or oversized-module budget changes are explained rather than added
      only to silence a guardrail.
- [ ] Tests are in the correct fast/integration/scientific/manual/GPU lane.
- [ ] Numerical conventions, tolerances, seeds, and scientific acceptance criteria are unchanged or
      the reason for changing them is documented.
- [ ] No credentials, private keys, `.env` files, sensitive runtime data, or generated production
      artifacts are included.
- [ ] Relevant lint/tests were run; skipped or unavailable checks are stated explicitly.
