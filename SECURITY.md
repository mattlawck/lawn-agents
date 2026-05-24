# Security Policy

`lawn-agents` is a personal-use CLI advisor, not a hosted service.
This policy describes how to report vulnerabilities and what is in
scope.

## Supported versions

Only the latest commit on `main` is supported. There are no LTS
branches or back-ports.

## Reporting a vulnerability

Please **do not open a public issue** for security findings.

Use GitHub's [private vulnerability reporting][pvr] to send a draft
advisory:

> https://github.com/mattlawck/lawn-agents/security/advisories/new

I'll respond within roughly a week. Once a fix is available, the
advisory is published and credited (unless you prefer anonymity).

[pvr]: https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing/privately-reporting-a-security-vulnerability

## Scope

In-scope:

- Source code in this repository.
- The provider-abstraction layer (`src/lawn_agents/llm.py`) and how
  it handles API keys loaded from `.env` via Pydantic-Settings.
- The research subagent (ADR 0005) and its domain-allowlist
  enforcement.
- The corpus ingester (`scripts/ingest_corpus.py`) — particularly
  any path-traversal or arbitrary-file-read risk in PDF/URL handling.

Out-of-scope:

- Bugs in upstream dependencies — please report to the relevant
  project; I'll bump the pin once a fix ships.
- Issues that require pre-existing local-machine access (this is a
  CLI tool that runs on your laptop with your `.env`).
- Hallucinations or incorrect agronomic recommendations from the
  LLM — these are functional bugs, not security issues. See ADR 0003
  for the never-guess guardrail that minimizes the blast radius.
- Findings in third-party hosted scanners (SonarCloud, Snyk, etc.)
  that don't reproduce in CodeQL or pose no concrete exploit path.

## Hardening already in place

- `.env` and `config.yaml` are gitignored; `.env.example` and
  `config.example.yaml` ship placeholders.
- API keys live in `pydantic.SecretStr`; their values do not appear
  in `repr()` output or structured logs.
- The research subagent only fetches URLs whose host matches
  `research.domain_allowlist` in `config.yaml`.
- CI runs ruff, mypy `--strict`, and pytest on every push and PR.
- CodeQL (`.github/workflows/codeql.yml`) scans on push, PR, and
  weekly.
- Dependabot (`.github/dependabot.yml`) opens PRs for pip and
  GitHub Actions updates weekly.
