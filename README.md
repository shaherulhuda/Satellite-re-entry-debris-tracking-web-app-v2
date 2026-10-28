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

`data/debris/` holds the debris groups (Fengyun-1C, Iridium-33, Cosmos-2251; CelesTrak JSON or 3-line TLE). `debris.py` parses them into the
same record format as `active.json`, so they flow through every feature (watchlist, tracker, conjunctions, chat tools).

The **Conjunctions & Triage** tab (it replaces the separate Conjunctions and Triage desk tabs) has two views:

- **Screen an object for close approaches:** closest passes of any object against debris, active satellites or everything
  (default 48 h).
- **Follow a debris object:** its estimated descent through the crowded altitude bands (found from the data), which satellites
  it passes close to, and the latitudes it can come down within.

Screening uses mean elements and gives no collision probability. These are triage estimates, not operational decisions.
Avoidance-cost maths (`maneuver.py`) and memo prompts (`llm.py`) are kept in the code for the AI assistant to use later.

## Keeping every data set at the same snapshot

All data sets (the active catalogue and each debris group: Fengyun-1C, Iridium-33, Cosmos-2251) should come from one download
so their element epochs line up. On a machine with internet access run:

```
python tools/refresh_data.py          # downloads everything together; all-or-nothing
git add -A && git commit -m "Refresh data snapshot"
```

The script writes `active.json`, one file per group in `data/debris/` and `data/snapshot.json`, and appends the perigee history.
The app shows the data age and warns when a data set is out of step with the others or the newest element set is old.
Any extra `.tle` / `.json` file dropped into `data/debris/` is loaded as another debris group.
