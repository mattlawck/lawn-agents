# Contributing

This is a personal-use repo, but contributions are welcome.

## Quickstart for contributors

```bash
# 1. Fork on GitHub, then clone your fork.
git clone https://github.com/<your-handle>/lawn-agents
cd lawn-agents

# 2. Install uv (https://docs.astral.sh/uv/) if you don't have it.
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Install dependencies.
uv sync --all-extras --dev

# 4. Install pre-commit hooks so you don't have to remember to run them.
uv run pre-commit install

# 5. Make your change on a feature branch.
git checkout -b feat/your-thing

# 6. Verify locally before pushing.
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest

# 7. Push and open a PR against `main`.
git push -u origin feat/your-thing
gh pr create
```

All PRs must pass:

- `ruff check` (lint) and `ruff format --check` (format)
- `mypy --strict` (types)
- `pytest` (≥60% coverage)
- CodeQL (security-and-quality)
- SonarCloud (no new issues)
- Dependency review (no new high-severity vulns, no AGPL/GPL deps)

Branch protection on `main` is solo-friendly — no human approval
required, but the checks gate the merge.

## Engineering norms

- Follow the module shapes in [`docs/architecture.md`](docs/architecture.md).
- Add an [ADR](docs/adr/) for non-obvious or load-bearing decisions.
- Document changes worth remembering in
  `docs/journal/YYYY-MM-DD-*.md` — these double as blog feedstock.
- **Do not loosen the never-guess guardrail** (ADR 0003) without an
  ADR amendment. Schema validation, prompt rules, and tests work
  together — weakening any layer requires explicit reasoning.
- Conventional commit prefixes: `feat:`, `fix:`, `chore:`, `docs:`,
  `refactor:`, `test:`.
- Bring-your-own-corpus is the model — never commit licensed PDFs or
  third-party content.

## Reporting issues

- **Security**: use [private vulnerability reporting][pvr]. See
  [SECURITY.md](SECURITY.md).
- **Bugs and features**: use the templates under "New Issue".

[pvr]: https://github.com/mattlawck/lawn-agents/security/advisories/new

## Code of conduct

This project follows the [Contributor Covenant 2.1](CODE_OF_CONDUCT.md).
By participating, you agree to abide by its terms.
