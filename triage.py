"""Triage desk: screen one active satellite against every satellite and debris fragment.

Everything here is computed from the stored elements (SGP4 + the rough decay model). It supports a
manager's triage, not an operational collision-avoidance decision.
"""
from __future__ import annotations

import pandas as pd

import conjunction

EVENT_COLUMNS = ["name", "norad_id", "tca_jd", "miss_km", "rel_speed_km_s"]


def _enrich(events: pd.DataFrame, catalogue: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events
    cols = ["NORAD_CAT_ID", "perigee_km", "apogee_km", "INCLINATION", "decay_days", "decay_status", "type"]
    if "GROUP" in catalogue:
        cols.append("GROUP")
    other = catalogue[cols].rename(columns={"NORAD_CAT_ID": "norad_id", "perigee_km": "other_perigee_km",
                                            "apogee_km": "other_apogee_km", "INCLINATION": "other_inclination_deg",
                                            "decay_days": "other_est_days_to_reentry",
                                            "decay_status": "other_decay_status", "type": "other_type",
                                            "GROUP": "other_group"})
    return events.merge(other, on="norad_id", how="left")


def _screen(primary: dict, catalogue: pd.DataFrame, candidates: pd.DataFrame, records: dict, start_jd: float,
            hours: float, threshold_km: float):
    pid = int(primary["NORAD_CAT_ID"])
    row = catalogue[catalogue["NORAD_CAT_ID"] == pid].iloc[0]
    events, n = conjunction.screen(primary, candidates, records, start_jd, hours=hours, threshold_km=threshold_km,
                                   primary_band=(float(row["perigee_km"]), float(row["apogee_km"])))
    out = _enrich(events, catalogue)
    out.attrs.update(events.attrs)  # keeps window_hours_effective
    return out, n


def screen_asset_vs_debris(asset: dict, catalogue: pd.DataFrame, records: dict, start_jd: float,
                           hours: float = 48.0, threshold_km: float = 20.0, groups: list[str] | None = None):
    """Closest passes of one chosen satellite against the debris set (optionally only some groups).

    Returns (events, candidates_screened).
    """
    deb = catalogue[catalogue["source"] == "debris"]
    if groups is not None and "GROUP" in deb:
        deb = deb[deb["GROUP"].isin(groups)]
    return _screen(asset, catalogue, deb, records, start_jd, hours, threshold_km)


AGAINST = {
    "debris": "Debris sets",
    "active": "Active satellites",
    "all": "Everything (satellites and debris)",
}


def screen_asset(asset: dict, catalogue: pd.DataFrame, records: dict, start_jd: float, hours: float = 48.0,
                 threshold_km: float = 20.0, against: str = "debris", groups: list[str] | None = None):
    """Closest passes of one object against debris, active satellites, or everything.

    `groups` limits which debris groups count (None = all). Returns (events, candidates_screened).
    """
    if against not in AGAINST:
        raise ValueError(f"against must be one of {sorted(AGAINST)}")
    is_deb = catalogue["source"] == "debris"
    if groups is not None and "GROUP" in catalogue:
        is_deb = is_deb & catalogue["GROUP"].isin(groups)
    if against == "debris":
        cand = catalogue[is_deb]
    elif against == "active":
        cand = catalogue[catalogue["source"] != "debris"]
    else:
        cand = catalogue[(catalogue["source"] != "debris") | is_deb]
    return _screen(asset, catalogue, cand, records, start_jd, hours, threshold_km)


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
                start_utc: str, plan: pd.DataFrame, plan_inputs: dict, chosen: dict | None,
                against: str = "the debris sets") -> dict:
    """Decision-brief facts for one object's close-approach screening. All numbers come from this app's code."""
    facts = {
        "situation": f"Screening one object for close approaches against {against}",
        "asset": asset_facts_dict,
        "screening": {
            "window_start_utc": start_utc, "window_hours": hours, "report_within_km": threshold_km,
            "objects_screened_in_similar_altitude_band": n_candidates,
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
