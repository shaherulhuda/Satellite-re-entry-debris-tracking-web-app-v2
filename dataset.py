"""Load the full catalogue (active satellites + debris groups) with decay columns, and the stored
closest-approach results that feed the risk score. No Streamlit imports, so tools and tests can use it."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import debris
import decay
import orbits

CA_PATH = Path(__file__).with_name("data") / "closest_approaches.json"
CA_COLUMNS = ["ca_miss_km", "ca_speed_km_s", "ca_hours", "ca_with"]
LEO_MAX_KM = 2000.0  # low Earth orbit: mean altitude and perigee below 2,000 km


def load_catalogue() -> tuple[pd.DataFrame, dict]:
    """(catalogue frame, records by NORAD id). Frame has decay columns, `type`, `source` ('active'/'debris')
    and `leo` (True when the orbit is low Earth orbit; the app lists and ranks LEO satellites only)."""
    active, deb = orbits.load_records(), debris.load_debris_records()
    recs = debris.merge_records(active, deb)
    cat = decay.estimate_decay(orbits.catalog_from_records(recs))
    cat["type"] = cat["OBJECT_NAME"].map(decay.classify_type)
    deb_ids = {r["NORAD_CAT_ID"] for r in deb}
    cat["source"] = np.where(cat["NORAD_CAT_ID"].isin(deb_ids), "debris", "active")
    cat["leo"] = (decay.mean_altitude_km(cat["MEAN_MOTION"]) < LEO_MAX_KM) & (cat["perigee_km"] < LEO_MAX_KM)
    return cat, {r["NORAD_CAT_ID"]: r for r in recs}


def load_closest_approaches(path: Path | str | None = None) -> dict | None:
    p = Path(path or CA_PATH)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def attach_closest_approaches(cat: pd.DataFrame, ca: dict | None) -> pd.DataFrame:
    """Add ca_* columns (NaN where an object was not screened or had no approach inside the threshold)
    and `ca_screened` (True where the stored run covered the object)."""
    out = cat.copy()
    for c in CA_COLUMNS:
        out[c] = np.nan if c != "ca_with" else None
    out["ca_screened"] = False
    if not ca:
        return out
    rows = pd.DataFrame(ca["rows"]).set_index("norad_id")
    idx = out["NORAD_CAT_ID"].isin(rows.index)
    out.loc[idx, "ca_screened"] = True
    sub = rows.loc[out.loc[idx, "NORAD_CAT_ID"]]
    out.loc[idx, "ca_miss_km"] = sub["miss_km"].to_numpy()
    out.loc[idx, "ca_speed_km_s"] = sub["rel_speed_km_s"].to_numpy()
    out.loc[idx, "ca_hours"] = sub["hours_to_ca"].to_numpy()
    out.loc[idx, "ca_with"] = sub["with_name"].to_numpy()
    return out
