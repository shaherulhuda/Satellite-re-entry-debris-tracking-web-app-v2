# Satellite-re-entry-debris-tracking-web-app-v2
Satellite re-entry debris tracking web app v2

## Optional AI assistant

The app runs fully offline. The **Assistant (AI)** tab (object briefing, risk explanation, "ask the data" chat) only works
when you supply an Anthropic API key, and the model only sees numbers computed by this repo's own code.

- Paste a key into the tab (kept in the browser session only), **or**
- set `ANTHROPIC_API_KEY` as an environment variable / in `.streamlit/secrets.toml`.

On a public deployment a key from secrets can be used by anyone with the link. Set `ASSISTANT_PASSWORD` to require an
access code and `ASSISTANT_MAX_REQUESTS` (default 30) to cap requests per session. Never commit a key.

## Debris data and the Triage desk

`data/fengyun1c_debris.tle` holds 1,983 Fengyun-1C breakup objects (3-line TLE format). `debris.py` parses them into the
same record format as `active.json`, so they flow through every feature (watchlist, tracker, conjunctions, chat tools).

The **Triage desk** tab has two views:

- **Protect a satellite:** closest passes of one satellite against the debris set (default 48 h), a first-order avoidance-burn
  cost (delta-v from Clohessy-Wiltshire, propellant from the rocket equation, `maneuver.py`), and an optional AI decision memo.
- **Follow a debris object:** its estimated descent through the crowded altitude bands (found from the data), which satellites
  it passes close to, and the latitudes it can come down within, with an optional AI end-of-life briefing.

Screening uses mean elements and gives no collision probability; avoidance costs assume a small impulsive burn. These are
triage estimates, not operational decisions.
