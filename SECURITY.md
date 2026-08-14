# Security policy

`qlinks` is research software, but repository and CI hygiene should assume that future workflows
may use private infrastructure, unpublished data, or service credentials.

## Credentials and sensitive data

Do not commit API keys, passwords, private keys, `.env` files, service-account credentials,
production account identifiers, or unsanitized private datasets. Use local untracked configuration
and GitHub Actions secrets/environments for CI credentials.

Notebooks and evidence metadata are source-controlled artifacts too: outputs, command strings, and
captured environment variables must be inspected for credentials before commit.

If a credential is committed, **rotate or revoke it immediately**. Removing it in a later commit is
not sufficient because the value remains in history and clones. After rotation, remove the secret
from the repository/history as appropriate and review logs/artifacts that may contain the same
value.

## Reporting a vulnerability

Do not publish exploitable details or live credentials in a public issue. Use the repository's
private security-reporting/advisory mechanism when available, or contact the maintainers through a
private project channel. Include the affected revision, reproduction steps, impact, and a minimal
sanitized example.

## Automated checks

Repository health, pre-commit hooks, CodeQL, and Bandit provide defense in depth. They do not replace
review of authentication, input validation, dependency provenance, or scientific-data handling.
Security-sensitive changes should use least-privilege GitHub workflow permissions and avoid floating
action references.
