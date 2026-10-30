# Validation results (final report, Section 7 and Appendix C)

Run every script from the repository root, with the app's packages installed (`pip install -r requirements.txt`).
The first argument is the repository folder (`.`). Outputs are written to this `results/` folder.

| Experiment | Script | Output files | Date run |
|---|---|---|---|
| Test suite | `pytest` | 82 / 82 passed in 106 s (pytest 9.2.0, Python 3.13.18) | 29 Oct 2026 |
| E1 Re-entry estimate | `python results/e1.py . active_2026-10-06.json` then `python results/e1b.py` | `e1_candidates.csv`, `e1_rate_check.csv`, `e1_reentry_check.csv` | 29 Oct 2026 |
| E2 Close-approach screening | `python results/e2c.py .` | `e2_watchlist_vs_socrates.csv` | 30 Oct 2026 |
| E3 Weight sensitivity | `python results/e3.py .` | `e3_weight_sensitivity.csv` | 29 Oct 2026 |
| E4 RAG AI assistant | `ANTHROPIC_API_KEY=... python results/e4.py .` and `python results/e4b.py .` | `e4_run.json` (6 questions), `e4b_run.json` (4 questions) | 30 Oct 2026 |

Inputs that are not produced by the scripts:

- `active_2026-10-06.json` (E1): the 6 Oct 2026 snapshot. Recreate it with `git show 99f8ce4:active.json > active_2026-10-06.json`.
- `e1_decay_dates.csv` (E1): actual re-entry dates of 48 objects, looked up one by one in the Orbital Register (orbitalradar.com/register/norad/<NORAD id>) on 29 Oct 2026.
- `e2_socrates_watchlist.csv` (E2): the 34 close approaches CelesTrak SOCRATES Plus listed for five watchlist satellites (window 29 Oct 08:00 – 5 Nov 08:00 UTC), copied from celestrak.org/SOCRATES on 30 Oct 2026.

All results use the data committed on 29 Oct 2026 (commit 4fb215c). No API key is stored in any file here.
