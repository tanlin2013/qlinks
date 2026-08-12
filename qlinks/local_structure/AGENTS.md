# Local-structure agent rules

This file supplements the repository-root `AGENTS.md`.

- This package is a neutral lower layer. It must not import `qlinks.caging` or
  `qlinks.open_system`.
- Keep names and semantics model-agnostic: local reduced density matrices,
  constrained-basis pattern embedding, and local matrix units belong here;
  cage classification, detector selection, recycling scores, and Lindblad
  semantics do not.
- Public functions must operate on explicit basis/state/operator data rather
  than concrete search-result or open-system workflow objects.
- Preserve constrained-basis pattern ordering and embedding conventions when
  moving utilities into this package. Add a small behavioural test whenever a
  shared primitive changes convention.
- Do not use this package as a miscellaneous utility bucket. New responsibilities
  require a clear local-algebra or local-structure interpretation.
