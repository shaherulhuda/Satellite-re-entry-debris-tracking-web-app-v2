# Validation results

Scripts and result files from the validation work (Appendix C). Nothing here is used by the app.

| File | What it is |
|---|---|
| `e1.py` | Decay estimate check. Part A compares predicted and observed altitude loss for objects present in both the early-October and the 29 Oct snapshots; part B lists objects whose predicted re-entry fell before 29 Oct. Writes `e1_candidates.csv` and `e1_rate_check.csv`. |
| `e1b.py` | Compares those candidates with actual decay dates (`e1_decay_dates.csv`). Writes `e1_reentry_check.csv`. |
| `e1_decay_dates.csv` | Actual decay dates used by `e1b.py` (NORAD id, date). |
| `e1_reentry_check.csv` | Predicted re-entry window vs actual decay date for each candidate. |
| `e1_rate_check.csv` | Predicted vs observed altitude drop for every object in both snapshots (about 10,000 rows). |
| `e2b.py` | Screens the ISS for one week with the app's close-approach code and compares the passes with five SOCRATES passes (reference values are written into the script). Writes `e2_iss_vs_socrates.csv`. |
| `e2_iss_vs_socrates.csv` | The comparison: reference vs app time, miss distance and relative speed. |
| `e3.py` | Risk-score weight sensitivity: re-ranks the low-Earth-orbit satellites with each weight halved, doubled or set to zero. Writes `e3_weight_sensitivity.csv`. |
| `e3_weight_sensitivity.csv` | Top-20 / top-50 overlap and rank correlation for each weight change. |

The scripts take the repository folder as their first argument and write to `/home/claude/results/`; change that path to run them elsewhere.
`e1_candidates.csv` (an intermediate file written by `e1.py` and read by `e1b.py`) was not part of the files supplied.
