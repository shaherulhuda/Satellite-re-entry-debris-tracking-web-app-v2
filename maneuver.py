"""First-order cost of a collision-avoidance burn: delta-v from Clohessy-Wiltshire, fuel from the rocket equation.

Simplifications (all of them favour a quick manager's estimate, not flight dynamics):
  * circular orbit, one small impulsive burn along the velocity direction `lead` seconds before closest approach;
  * the burn's displacement at closest approach is the CW result
        radial  x = (2/n)(1 - cos nt) dv,   along-track  y = (4 sin nt - 3 nt)/n dv
  * the direction of the existing miss is not used. Best case: the shift points away from the debris
    (need target - miss). Worst case: it points toward it (need target + miss);
  * electric propulsion cannot burn impulsively; for those vehicles treat the result as a lower bound.
"""
from __future__ import annotations

import math

import pandas as pd

G0 = 9.80665  # m/s^2

PRESETS = {
    "Small satellite, electric thruster (260 kg, Isp 1500 s)": (260.0, 1500.0),
    "Large satellite, electric thruster (800 kg, Isp 1800 s)": (800.0, 1800.0),
    "Satellite with hydrazine thrusters (1000 kg, Isp 220 s)": (1000.0, 220.0),
    "Crewed station (420 000 kg, Isp 300 s)": (420000.0, 300.0),
}


def shift_per_unit_dv_km(mean_motion_rev_day: float, lead_s: float) -> float:
    """Displacement (km) at closest approach per 1 m/s of tangential burn made `lead_s` seconds earlier."""
    n = mean_motion_rev_day * 2.0 * math.pi / 86400.0
    nt = n * lead_s
    per_kms = math.hypot(2.0 / n * (1.0 - math.cos(nt)), (4.0 * math.sin(nt) - 3.0 * nt) / n)  # km per (km/s)
    return per_kms / 1000.0  # km per (m/s)


def required_dv_ms(shift_km: float, mean_motion_rev_day: float, lead_s: float) -> float:
    """Tangential delta-v (m/s) that moves the object `shift_km` at closest approach."""
    per_ms = shift_per_unit_dv_km(mean_motion_rev_day, lead_s)
    return float("inf") if per_ms <= 0 else shift_km / per_ms


def propellant_kg(mass_kg: float, dv_ms: float, isp_s: float) -> float:
    """Tsiolkovsky: propellant needed for dv_ms from a vehicle of wet mass mass_kg."""
    return mass_kg * (1.0 - math.exp(-dv_ms / (G0 * isp_s)))


def avoidance_plan(miss_km: float, target_km: float, mean_motion_rev_day: float, mass_kg: float, isp_s: float,
                   lead_hours=(1, 3, 6, 12, 24, 36)) -> pd.DataFrame:
    """Delta-v and propellant for a range of burn lead times. Empty frame if no burn is needed."""
    if miss_km >= target_km:
        return pd.DataFrame()
    rows = []
    for h in lead_hours:
        lead = h * 3600.0
        best = required_dv_ms(max(target_km - miss_km, 0.0), mean_motion_rev_day, lead)
        worst = required_dv_ms(target_km + miss_km, mean_motion_rev_day, lead)
        rows.append({
            "lead_h": h, "dv_best_ms": best, "dv_worst_ms": worst,
            "fuel_best_kg": propellant_kg(mass_kg, best, isp_s), "fuel_worst_kg": propellant_kg(mass_kg, worst, isp_s),
            "fuel_worst_pct_mass": 100.0 * propellant_kg(mass_kg, worst, isp_s) / mass_kg,
        })
    return pd.DataFrame(rows)
