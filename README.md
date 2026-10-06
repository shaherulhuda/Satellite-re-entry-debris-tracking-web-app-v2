# Satellite-re-entry-debris-tracking-web-app-v2
Satellite re-entry debris tracking web app v2

## Optional AI assistant

The app runs fully offline. The **Assistant (AI)** tab (object briefing, risk explanation, "ask the data" chat) only works
when you supply an Anthropic API key, and the model only sees numbers computed by this repo's own code.

- Paste a key into the tab (kept in the browser session only), **or**
- set `ANTHROPIC_API_KEY` as an environment variable / in `.streamlit/secrets.toml`.

On a public deployment a key from secrets can be used by anyone with the link. Set `ASSISTANT_PASSWORD` to require an
access code and `ASSISTANT_MAX_REQUESTS` (default 30) to cap requests per session. Never commit a key.
