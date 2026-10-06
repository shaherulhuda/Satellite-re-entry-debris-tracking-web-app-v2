"""Perigee history: one row per object per element set, kept in a small CSV in data/."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HISTORY_PATH = Path(__file__).with_name("data") / "perigee_history.csv"
COLUMNS = ["norad_id", "epoch", "perigee_km", "apogee_km"]
DEFAULT_MAX_PERIGEE_KM = 400.0  # only low objects are tracked, to keep the file small


def snapshot_rows(df: pd.DataFrame, max_perigee_km: float = DEFAULT_MAX_PERIGEE_KM) -> pd.DataFrame:
    low = df[df["perigee_km"] <= max_perigee_km]
    return pd.DataFrame({
        "norad_id": low["NORAD_CAT_ID"].astype(int).to_numpy(),
        "epoch": low["EPOCH"].astype(str).to_numpy(),
        "perigee_km": low["perigee_km"].round(3).to_numpy(),
        "apogee_km": low["apogee_km"].round(3).to_numpy(),
    })


def load_history(path: Path | str = HISTORY_PATH) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=COLUMNS).astype({"norad_id": int})
    h = pd.read_csv(p)
    h["epoch"] = pd.to_datetime(h["epoch"])
    return h.sort_values(["norad_id", "epoch"]).reset_index(drop=True)


def append_snapshot(df: pd.DataFrame, path: Path | str = HISTORY_PATH,
                    max_perigee_km: float = DEFAULT_MAX_PERIGEE_KM) -> int:
    """Add this snapshot's rows; (norad_id, epoch) pairs already stored are skipped. Returns rows added."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    new = snapshot_rows(df, max_perigee_km)
    if p.exists():
        old = pd.read_csv(p)
        key = set(zip(old["norad_id"], old["epoch"].astype(str)))
        new = new[[(n, e) not in key for n, e in zip(new["norad_id"], new["epoch"])]]
        out = pd.concat([old, new], ignore_index=True)
    else:
        out = new
    out.to_csv(p, index=False)
    return len(new)


def object_history(hist: pd.DataFrame, norad_id: int) -> pd.DataFrame:
    return hist[hist["norad_id"] == int(norad_id)].sort_values("epoch").reset_index(drop=True)


def observed_decay_rate(h: pd.DataFrame, min_days: float = 1.0):
    """Least-squares perigee trend in km/day (negative = falling), or None if the data span is too short."""
    if len(h) < 2:
        return None
    t = (h["epoch"] - h["epoch"].iloc[0]).dt.total_seconds().to_numpy() / 86400.0
    if t.max() - t.min() < min_days:
        return None
    return float(np.polyfit(t, h["perigee_km"].to_numpy(float), 1)[0])
