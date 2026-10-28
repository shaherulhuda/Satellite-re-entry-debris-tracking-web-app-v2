# Satellite-re-entry-debris-tracking-web-app-v2
Satellite re-entry debris tracking web app v2

## Optional AI assistant

The app runs fully offline. The **Ask the assistant** popup (ask the data, object briefing, risk explanation, recommended
action) only works when you supply an Anthropic API key, and the model only sees numbers computed by this repo's own code.
It defaults to Claude Haiku 4.5 (as in the project proposal; Sonnet and Opus are in the dropdown), knows how the app and
risk score work, and the monitor / prepare / act suggestion comes from fixed rules in `llm.suggested_action`, which the model only
explains. To check it against your key: `ANTHROPIC_API_KEY=... python tools/check_assistant.py`.

- Paste a key into the tab (kept in the browser session only), **or**
- set `ANTHROPIC_API_KEY` as an environment variable / in `.streamlit/secrets.toml`.

On a public deployment a key from secrets can be used by anyone with the link. Set `ASSISTANT_PASSWORD` to require an
access code and `ASSISTANT_MAX_REQUESTS` (default 10) to cap questions per session. Never commit a key.

## Debris data and scope

The app is **low Earth orbit only**: the Watchlist, Re-entry Tracker and Conjunctions & Triage tabs (and the assistant) cover **active satellites from `active.json` with mean altitude and perigee below 2,000 km**. Higher orbits (GEO/MEO) are never listed but still count as screening partners. The Re-entry Tracker and screening pickers list those same 50 highest-risk satellites. The Watchlist draws on them; the **Rank by** dropdown orders them and **Objects** shows the first N of that order (10 to 50).
Debris (`data/debris/`: Fengyun-1C, Iridium-33, Cosmos-2251; CelesTrak JSON or 3-line TLE, parsed by `debris.py`) is used in two ways:

- **Orbital view:** the Watchlist globe can draw debris orbits (the **Debris** box; 0 = hidden).
- **Close approaches:** when an active satellite is screened (Conjunctions & Triage, and the close-approach terms of the risk score),
  every other satellite and every debris fragment is a possible partner.

**Conjunctions & Triage:** pick an active satellite, a window of 1 to 30 days and a distance threshold; the closest passes
against every satellite and debris fragment are listed. Screening uses mean elements and gives no collision probability, so
results are triage estimates, not operational decisions. Avoidance-cost maths (`maneuver.py`) and the memo prompt (`llm.py`)
are kept in the code for the AI assistant to use later.

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

## Risk score and colour levels

The Watchlist risk score blends seven terms (weights adjustable in the app): perigee, estimated decay time, inclination, object
type, and three close-approach terms: **miss distance**, **relative speed** and **time to closest approach**. The close-approach
terms use each object's closest pass to any satellite or debris in the 24 hours after the snapshot time. The listed objects are
coloured by overall risk rank: the 10 highest-risk active satellites are red, the next 30 yellow, all others green (so 50 listed satellites show 10 / 30 / 10, and a list of 10 is all red); debris is not on the watchlist.

Screening every object would be too slow for the web app, so `tools/compute_conjunctions.py` precomputes the closest approach for
the highest-risk objects (default 300) into `data/closest_approaches.json`. Re-run it after `tools/refresh_data.py`.


## Speed notes

- **Plotly is shared.** Every globe loads one copy of Plotly from `static/plotly.min.js` (served because `.streamlit/config.toml` sets
  `enableStaticServing`), which the browser caches. If that file cannot be served, the same version is fetched from the Plotly CDN.
  `test_debris_globe.py` checks the file matches the pinned `plotly` version, so update it when you bump Plotly:
  `cp $(python -c "import plotly,os;print(os.path.dirname(plotly.__file__))")/package_data/plotly.min.js static/`.
- **One section at a time.** The Watchlist, Re-entry Tracker and Conjunctions & Triage sections are separate fragments; only the one on
  screen is built, and interacting inside it reruns just that fragment. The assistant is its own fragment too.
- Finished globes are cached for 5 minutes per set of objects.
