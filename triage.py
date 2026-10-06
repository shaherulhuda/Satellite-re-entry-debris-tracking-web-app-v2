"""Triage desk: debris screening for one asset, avoidance cost, and an end-of-life view of one debris object.

Everything here is computed from the stored elements (SGP4 + the rough decay model). It supports a
manager's triage, not an operational collision-avoidance decision.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import conjunction
import decay
import footprint
import maneuver

EVENT_COLUMNS = ["name", "norad_id", "tca_jd", "miss_km", "rel_speed_km_s"]


def _enrich(events: pd.DataFrame, catalogue: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events
    cols = ["NORAD_CAT_ID", "perigee_km", "apogee_km", "INCLINATION", "decay_days", "decay_status", "type"]
    other = catalogue[cols].rename(columns={"NORAD_CAT_ID": "norad_id", "perigee_km": "other_perigee_km",
                                            "apogee_km": "other_apogee_km", "INCLINATION": "other_inclination_deg",
                                            "decay_days": "other_est_days_to_reentry",
                                            "decay_status": "other_decay_status", "type": "other_type"})
    return events.merge(other, on="norad_id", how="left")


def _screen(primary: dict, catalogue: pd.DataFrame, candidates: pd.DataFrame, records: dict, start_jd: float,
            hours: float, threshold_km: float):
    pid = int(primary["NORAD_CAT_ID"])
    row = catalogue[catalogue["NORAD_CAT_ID"] == pid].iloc[0]
    step = 10.0 if hours <= 12 else (20.0 if hours <= 36 else 30.0)  # keeps long windows affordable
    events, n = conjunction.screen(primary, candidates, records, start_jd, hours=hours, step_s=step,
                                   threshold_km=threshold_km, primary_band=(float(row["perigee_km"]), float(row["apogee_km"])))
    return _enrich(events, catalogue), n


def screen_asset_vs_debris(asset: dict, catalogue: pd.DataFrame, records: dict, start_jd: float,
                           hours: float = 48.0, threshold_km: float = 20.0):
    """Closest passes of one chosen satellite against the debris set. Returns (events, candidates_screened)."""
    return _screen(asset, catalogue, catalogue[catalogue["source"] == "debris"], records, start_jd, hours, threshold_km)


def screen_debris_vs_satellites(debris_obj: dict, catalogue: pd.DataFrame, records: dict, start_jd: float,
                                hours: float = 48.0, threshold_km: float = 20.0):
    """Closest passes of one debris object against every non-debris satellite (who it threatens now)."""
    return _screen(debris_obj, catalogue, catalogue[catalogue["source"] != "debris"], records, start_jd, hours, threshold_km)


# ------------------------------------------------------------ altitude bands for the end-of-life view

def altitude_bands(catalogue: pd.DataFrame, bin_km: float = 20.0, min_objects: int = 150) -> list[dict]:
    """Altitude ranges crowded with active satellites, found from the data (plus the ISS band if present).

    Bins of `bin_km` holding at least `min_objects` non-debris objects are merged when adjacent. Each
    band is labelled with the most common name prefix when it dominates (e.g. a constellation).
    """
    act = catalogue[catalogue["source"] != "debris"].copy()
    act["alt"] = decay.mean_altitude_km(act["MEAN_MOTION"])
    act = act[(act["alt"] >= 250) & (act["alt"] <= 1400) & (act["ECCENTRICITY"] < 0.02)]
    edges = np.arange(250, 1400 + bin_km, bin_km)
    counts, _ = np.histogram(act["alt"], bins=edges)
    bands, cur = [], None
    for i, c in enumerate(counts):
        if c >= min_objects:
            lo, hi = edges[i], edges[i + 1]
            if cur and abs(cur["hi"] - lo) < 1e-6:
                cur["hi"] = hi
            else:
                cur = {"lo": lo, "hi": hi}
                bands.append(cur)
        else:
            cur = None
    out = []
    for b in bands:
        sel = act[(act["alt"] >= b["lo"]) & (act["alt"] < b["hi"])]
        prefix = sel["OBJECT_NAME"].str.split(r"[-\s]", n=1, regex=True).str[0]
        top = prefix.value_counts()
        share = float(top.iloc[0]) / len(sel)
        label = f"{top.index[0].title()} shell" if share >= 0.5 else "crowded shell"
        out.append({"lo": float(b["lo"]), "hi": float(b["hi"]), "n_objects": int(len(sel)), "label": label,
                    "dominant_share": round(share, 2)})
    iss = catalogue[catalogue["NORAD_CAT_ID"] == 25544]
    if len(iss):
        r = iss.iloc[0]
        out.append({"lo": float(r["perigee_km"]) - 10, "hi": float(r["apogee_km"]) + 10, "n_objects": 1,
                    "label": "ISS orbit", "dominant_share": 1.0})
    return sorted(out, key=lambda b: -b["hi"])


def band_crossings(h0_km: float, rate_km_day: float, bands: list[dict]) -> pd.DataFrame:
    """When the model says an object sinks into / out of each band (days from the element epoch).

    Bands the object is already inside get enters_days = 0; bands wholly above it are skipped; bands it never
    reaches before the model's re-entry threshold are skipped too.
    """
    proj = decay.altitude_projection(h0_km, rate_km_day, n=600)
    if proj is None:
        return pd.DataFrame()
    t, h = proj  # h decreasing, t increasing

    def when(alt):
        return float(np.interp(alt, h[::-1], t[::-1])) if h[-1] <= alt <= h[0] else None

    rows = []
    for b in bands:
        if b["lo"] >= h0_km:
            continue
        inside = b["hi"] >= h0_km
        enter = 0.0 if inside else when(b["hi"])
        if enter is None:
            continue
        leave = when(b["lo"])
        rows.append({"band": f"{b['label']} ({b['lo']:.0f}-{b['hi']:.0f} km)", "n_objects": b["n_objects"],
                     "already_inside": inside, "enters_days": enter, "leaves_days": leave,
                     "enters_low": enter / decay.UNCERTAINTY_FACTOR, "enters_high": enter * decay.UNCERTAINTY_FACTOR})
    return pd.DataFrame(rows).sort_values("enters_days").reset_index(drop=True) if rows else pd.DataFrame()


# ------------------------------------------------------------ facts handed to the LLM

def _events_summary(events: pd.DataFrame, limit: int = 5) -> list[dict]:
    out = []
    for r in events.head(limit).itertuples():
        out.append({
            "object": r.name, "norad_id": int(r.norad_id),
            "closest_approach_utc": conjunction.jd_to_datetime(r.tca_jd).strftime("%Y-%m-%d %H:%M"),
            "miss_km": round(float(r.miss_km), 2), "relative_speed_km_s": round(float(r.rel_speed_km_s), 2),
            "its_perigee_km": None if pd.isna(r.other_perigee_km) else round(float(r.other_perigee_km), 0),
            "its_apogee_km": None if pd.isna(r.other_apogee_km) else round(float(r.other_apogee_km), 0),
            "its_type": r.other_type,
        })
    return out


def asset_facts(asset_facts_dict: dict, events: pd.DataFrame, n_candidates: int, hours: float, threshold_km: float,
                start_utc: str, plan: pd.DataFrame, plan_inputs: dict, chosen: dict | None) -> dict:
    """Decision-brief facts for 'protect this satellite from debris'. All numbers come from this app's code."""
    facts = {
        "situation": "Screening one satellite against the Fengyun-1C breakup debris set",
        "asset": asset_facts_dict,
        "screening": {
            "window_start_utc": start_utc, "window_hours": hours, "report_within_km": threshold_km,
            "debris_objects_in_similar_altitude_band": n_candidates,
            "passes_within_threshold": int(len(events)),
            "closest_passes": _events_summary(events) if len(events) else [],
        },
        "chosen_pass": chosen,
        "avoidance_assumptions": plan_inputs,
        "avoidance_options": [] if plan.empty else [
            {"burn_lead_hours": float(r.lead_h), "delta_v_m_s_best_case": round(float(r.dv_best_ms), 3),
             "delta_v_m_s_worst_case": round(float(r.dv_worst_ms), 3),
             "propellant_kg_worst_case": round(float(r.fuel_worst_kg), 4),
             "propellant_pct_of_mass_worst_case": round(float(r.fuel_worst_pct_mass), 5)} for r in plan.itertuples()],
        "limits": ["mean-element positions are good to roughly a kilometre or worse for old elements",
                   "no uncertainty or collision probability is available",
                   "avoidance cost is a first-order, impulsive estimate; electric propulsion needs long burns"],
    }
    return facts


def eol_facts(debris_facts: dict, crossings: pd.DataFrame, threats: pd.DataFrame, n_candidates: int, hours: float,
              threshold_km: float, start_utc: str, bands: list[dict]) -> dict:
    """Facts for 'follow this debris object through its end of life'."""
    return {
        "situation": "End-of-life view of one debris object: who it threatens while sinking, then where it can come down",
        "debris_object": debris_facts,
        "altitude_bands_below_it": [{"label": b["label"], "from_km": b["lo"], "to_km": b["hi"], "active_objects": b["n_objects"]}
                                    for b in bands if b["hi"] < debris_facts["orbit"]["perigee_km"] + 50],
        "band_crossings_days_from_element_epoch": [] if crossings.empty else [
            {"band": r.band, "already_inside_now": bool(r.already_inside), "enters_in_days": round(float(r.enters_days), 1),
             "rough_range_days": [round(float(r.enters_low), 1), round(float(r.enters_high), 1)]} for r in crossings.itertuples()],
        "threat_screening": {
            "window_start_utc": start_utc, "window_hours": hours, "report_within_km": threshold_km,
            "satellites_in_similar_altitude_band": n_candidates, "passes_within_threshold": int(len(threats)),
            "closest_passes": _events_summary(threats) if len(threats) else [],
        },
        "footprint": debris_facts.get("footprint"),
        "limits": ["decay timing is a rough estimate (factor 2 or worse) and ignores solar activity changes",
                   "the landing point cannot be predicted this far ahead; the footprint only shows where it can pass",
                   "screening uses mean elements, not collision probability"],
    }
