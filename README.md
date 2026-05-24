# lawn-agents

A multi-agent lawn-care advisory for **Zeon Zoysia** in coastal South Carolina.
It pulls live weather and soil temperature, retrieves cited passages from a
local knowledge base (Clemson HGIC, Super-Sod Lawn Academy, the Turfgrass
Group, your own paid guides), and recommends *when* to fertilize, treat for
grubs, apply pre-emergent, and put down micronutrients — with citations on
every chemical recommendation.

> **Safety rule.** This system never guesses on herbicide, fertilizer, or
> insecticide products. Every chemical recommendation must cite an
> authoritative passage from the retrieved knowledge base. If no source is
> found, it refuses and tells you where to look instead.

## Status

Phase 1 — Zeon Zoysia advisory (in development). Phase 2 will add live oaks,
palms, hydrangeas, and other shrubs.

## Architecture

```mermaid
flowchart LR
    user(["User<br/>--ask / --scheduled / --plan-year"]) --> main
    main[main.py] --> orch[orchestrator.py]

    orch -->|route intent<br/>Sonnet 4.6| router{Router}
    router --> weather[agents/weather.py<br/>NWS api.weather.gov]
    router --> soil[agents/soiltemp.py<br/>USDA-NRCS AWDB / SCAN]
    router --> drought[agents/drought.py<br/>US Drought Monitor]
    router --> kn[agents/knowledge.py<br/>LanceDB + bge-small]

    kn -.->|weak retrieval| research[agents/research.py<br/>web_search +<br/>allowlisted fetch]
    research -.->|new passages| kn

    weather --> synth
    soil --> synth
    drought --> synth
    kn --> synth[Synthesizer<br/>Opus 4.7]

    synth --> validate[Pydantic guardrail:<br/>citations required on<br/>chemical recommendations]
    validate --> notify[notify.py<br/>console / email / SMS]
```

External I/O is isolated to the `agents/*.py` modules. Each fails closed to
`None` so a flaky NWS endpoint can't kill the whole run — the synthesizer is
told what's missing and degrades gracefully ("soil temp unavailable today,
recommending based on 7-day air-temp trend").

For full module contracts and data flow, see [`docs/architecture.md`](docs/architecture.md).

## Quickstart

```bash
# 1. Install uv if you don't have it (https://docs.astral.sh/uv/)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone and sync
git clone https://github.com/mattlawck/lawn-agents
cd lawn-agents
uv sync --all-extras --dev

# 3. Configure
cp .env.example .env
# Edit .env and paste your GEMINI_API_KEY from aistudio.google.com
# (free-tier works). Or switch to Anthropic in config.yaml and use
# ANTHROPIC_API_KEY instead — see docs/adr/0006-provider-abstraction.md.

# 4. Copy the example config and edit for your location/cultivar/frost dates.
# Your config.yaml stays local (gitignored); the committed file is
# config.example.yaml.
cp config.example.yaml config.yaml

# 5. Bring your own corpus (see below) and ingest it
uv run python scripts/ingest_corpus.py

# 6. Ask away
uv run lawn-agents --ask "Is it too early to put down pre-emergent?"
uv run lawn-agents --scheduled
uv run lawn-agents --plan-year 2026
```

## Bring Your Own Corpus

The public repo deliberately ships **no** third-party content. Extension
publications, manufacturer guides, and paid PDFs you subscribe to remain on
your machine. The ingester (`scripts/ingest_corpus.py`) reads two sources:

1. **`data/corpus/*.pdf`** — drop any PDFs you've purchased or downloaded
   here. Gitignored.
2. **`seed_urls:` in `config.yaml`** — public pages we recommend pre-loading
   (Super-Sod Lawn Academy, NCSU TurfFiles, etc.). Fetched, chunked, and
   embedded into the local LanceDB index. The URL list is yours to edit.

Add a source any time — just append to `seed_urls` or drop a PDF into
`data/corpus/`, then re-run the ingester. Provenance metadata (source URL,
fetched-at timestamp, page number) is stored with every chunk so citations
are precise.

### When the corpus has a gap

If retrieval comes back weak on a question, the orchestrator can dispatch a
**research subagent** restricted to a domain allowlist (`config.yaml >
research.domain_allowlist`, defaults to `.edu`, `.gov`, and named
turf-industry sites). Found pages are chunked, embedded, and stored with
`requires_review: true` until you promote them via
`lawn-agents review-additions`. The RAG grows monotonically — your second
run is smarter than your first.

## Configuration

`config.yaml` holds non-secret settings: location, cultivar, frost dates,
soil-temp thresholds, knowledge/retrieval tuning, research allowlist, model
routing, notify sinks. Schema is defined in
[`src/lawn_agents/config.py`](src/lawn_agents/config.py) and validated by
Pydantic at startup.

`.env` holds secrets only (currently just `ANTHROPIC_API_KEY`).

## Development

```bash
# Install + activate
uv sync --all-extras --dev
source .venv/bin/activate  # optional; or prefix commands with `uv run`

# Install pre-commit hooks (runs ruff, mypy, basic checks on every commit)
uv run pre-commit install

# Lint + format
uv run ruff check .
uv run ruff format .

# Type-check (strict)
uv run mypy

# Tests (with coverage)
uv run pytest
```

CI runs the same checks on every push and PR — see
[`.github/workflows/ci.yml`](.github/workflows/ci.yml). Coverage floor is
60% and ratchets up over time.

### Decision records

Non-obvious architectural choices are documented as ADRs in
[`docs/adr/`](docs/adr/). Engineering journal entries (the blog feedstock)
live in [`docs/journal/`](docs/journal/).

## Roadmap

- **Phase 1 (now)** — Zeon Zoysia: weather + soil temp + RAG, scheduled
  weekly check, ad-hoc Q&A, monthly and annual planning with drought
  awareness, launchd scheduling.
- **Phase 2** — add subject modules for live oaks (young + mature), palms,
  hydrangeas, and other shrubs. Extend the router to pick the right
  subject from the question.
- **Phase 3** — additional notify sinks (email, SMS), photo-based disease
  triage, and a simple web UI.

## License

MIT — see [`LICENSE`](LICENSE).
