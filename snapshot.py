"""Snapshot manifest and consistency check for the bundled data sets.

Every data set (active catalogue + each debris group) should come from one download so their element
epochs line up. `tools/refresh_data.py` fetches them together and writes `data/snapshot.json`;
`health()` flags data sets that have drifted apart or gone stale.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

MANIFEST_PATH = Path(__file__).with_name("data") / "snapshot.json"
MAX_GROUP_LAG_DAYS = 3.0  # a data set whose newest epoch trails the others by more than this is out of step
STALE_DAYS = 7.0  # element sets older than this (relative to the newest epoch) propagate poorly
MAX_STALE_SHARE = 0.05


def load_manifest(path: Path | str | None = None) -> dict | None:
    p = Path(path or MANIFEST_PATH)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def write_manifest(fetched_utc: str, sources: dict, path: Path | str | None = None) -> None:
    Path(path or MANIFEST_PATH).write_text(json.dumps({"fetched_utc": fetched_utc, "sources": sources}, indent=1))


def source_stats(records: list[dict]) -> dict:
    """Count and epoch range of one data set (for the manifest)."""
    ep = pd.to_datetime([r["EPOCH"] for r in records])
    return {"count": len(records), "epoch_min": ep.min().isoformat(), "epoch_median": ep.sort_values()[len(ep) // 2].isoformat(),
            "epoch_max": ep.max().isoformat()}


def health(df: pd.DataFrame, now: pd.Timestamp | None = None) -> dict:
    """Epoch spread per data set and a list of human-readable problems (empty = consistent).

    `df` needs EPOCH_DT and, for debris, a GROUP column (NaN = active catalogue). `newest_age_days` is how
    old the newest element set is compared with `now` (UTC clock by default): positions "now" are extrapolated by that much.
    """
    ref = df["EPOCH_DT"].max()
    now = now if now is not None else pd.Timestamp.now("UTC").tz_localize(None)
    group = df["GROUP"].fillna("Active catalogue") if "GROUP" in df else pd.Series("Active catalogue", index=df.index)
    rows = []
    for name, part in df.groupby(group):
        rows.append({"set": name, "objects": int(len(part)), "newest": part["EPOCH_DT"].max(),
                     "median_age_days": float((ref - part["EPOCH_DT"]).dt.total_seconds().median() / 86400.0)})
    table = pd.DataFrame(rows).sort_values("set").reset_index(drop=True)
    issues = []
    for r in table.itertuples():
        lag = (ref - r.newest).total_seconds() / 86400.0
        if lag > MAX_GROUP_LAG_DAYS:
            issues.append(f"{r.set}: newest element set is {lag:.1f} days older than the rest of the data")
    age = (ref - df["EPOCH_DT"]).dt.total_seconds() / 86400.0
    stale = float((age > STALE_DAYS).mean())
    if stale > MAX_STALE_SHARE:
        issues.append(f"{stale * 100:.0f}% of element sets are more than {STALE_DAYS:.0f} days older than the newest one")
    return {"reference": ref, "table": table, "stale_share": stale, "issues": issues,
            "newest_age_days": float((now - ref).total_seconds() / 86400.0)}
