"""Append a new active.json download to data/perigee_history.csv.

Usage: python tools/add_snapshot.py [path/to/active.json] [max_perigee_km]
Run it each time you replace active.json with fresh CelesTrak data, then commit the CSV.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import history  # noqa: E402
import orbits  # noqa: E402

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else orbits.DATA_PATH
    max_p = float(sys.argv[2]) if len(sys.argv) > 2 else history.DEFAULT_MAX_PERIGEE_KM
    df = orbits.load_catalog(src)
    added = history.append_snapshot(df, max_perigee_km=max_p)
    print(f"Added {added} rows (snapshot {orbits.snapshot_date(df)}) to {history.HISTORY_PATH}")
