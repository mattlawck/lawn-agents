# ADR 0004 — USDA-NRCS AWDB (SCAN) as the soil-temperature source

- **Status**: Accepted — 2026-05-22
- **Deciders**: Matt

## Context

Zoysia green-up, dormancy, and pre-emergent timing are all driven by
soil temperature at 2"–4" depth. We need a programmatic, reliable
source close to the configured lat/lon (coastal SC in the original
deployment).

Candidates evaluated:

- **Syngenta GreenCast soil-temperature tool**
  (`greencastonline.com/tools/soil-temperature`). Web UI only, no
  public API, terms-of-use unclear, and the displayed depth is the
  2–5 cm layer (≈ 1–2 inches) — shallower than the conventional
  4-inch reference depth for warm-season turf decisions. Scraping a
  corporate tool from a public open-source repo is also legally
  awkward.
- **`soiltemps.com`** — third-party aggregator, rate-limits aggressive
  clients, terms unclear.
- **USDA-NRCS AWDB REST API**
  (`wcc.sc.egov.usda.gov/awdbRestApi/swagger-ui`). Documented Swagger
  API exposing the Soil Climate Analysis Network (SCAN). 380+
  stations, 2-inch and 4-inch soil-temperature sensors, federal data
  in the public domain, no key required.

## Decision

Use the USDA-NRCS AWDB REST API as the canonical soil-temperature
source. `agents/soiltemp.py`:

1. At startup, resolves the nearest SCAN station to the configured
   lat/lon.
2. Queries 2"/4" soil temperature (current + recent 7-day window).
3. If no SCAN station sits within a configurable radius (default
   ~75 miles), falls back to a Parton/Logan-style model that
   estimates 4-inch soil temperature from a rolling window of NWS
   air-temperature observations.

Both paths return a `SoilSnapshot` whose `provenance` field records
the station ID and depth (or "modeled-from-air-temp" when the fallback
ran).

## Rationale

- Federal data, public domain — no licensing concern when the repo is
  published.
- Actual 4-inch readings match the depth the turf literature uses.
- No third-party scraping risk.
- The modeled fallback gives graceful degradation in regions with thin
  SCAN coverage. Coastal SC may need it; we'll know after the first
  station lookup.

## Consequences

- We accept the engineering cost of implementing and testing the
  fallback air-temp model. ADR 0006 (TBD) may revisit if the model
  proves too noisy in coastal conditions.
- We give up the convenience of GreenCast's pre-rendered map UI,
  which is fine — this is an API-backed CLI, not a dashboard.
