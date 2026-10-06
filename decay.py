"""Rough atmospheric-decay estimate and a transparent re-entry risk score. Offline, vectorised.

DECAY MODEL (an estimate, not a forecast)
  1. The element set carries a measured drag effect: MEAN_MOTION_DOT. In OMM/TLE data this
     field is half of the true first derivative of mean motion, so n_dot = 2 * field.
  2. For a near-circular orbit n^2 a^3 = mu, so  da/dt = -(2/3) * a * n_dot / n   (km/day).
  3. Drag scales with air density, which falls roughly exponentially with height. We scale the
     measured rate by rho(h)/rho(h0) using a piecewise-exponential atmosphere (Vallado table),
     and integrate the altitude down to REENTRY_ALT_KM:
         days = rho(h0) / (da/dt)_0  *  integral_{REENTRY}^{h0} dh / rho(h)
  Because the rate is measured now, it already reflects current solar activity at h0; the model
  cannot know how solar activity or the object's orientation will change, and it ignores
  manoeuvres. Treat the result as order-of-magnitude, with an uncertainty of a factor of 2 or more.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from orbits import EARTH_RADIUS_KM, MU_EARTH_KM3_S2

REENTRY_ALT_KM = 120.0
MAX_YEARS = 50.0
UNCERTAINTY_FACTOR = 2.0

# Vallado exponential atmosphere: base altitude (km), base density (kg/m^3), scale height (km)
_BASE = np.array([0, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 180, 200, 250,
                  300, 350, 400, 450, 500, 600, 700, 800, 900, 1000], dtype=float)
_RHO0 = np.array([1.225, 3.899e-2, 1.774e-2, 3.972e-3, 1.057e-3, 3.206e-4, 8.770e-5, 1.905e-5,
                  3.396e-6, 5.297e-7, 9.661e-8, 2.438e-8, 8.484e-9, 3.845e-9, 2.070e-9, 5.464e-10,
                  2.789e-10, 7.248e-11, 2.418e-11, 9.158e-12, 3.725e-12, 1.585e-12, 6.967e-13,
                  1.454e-13, 3.614e-14, 1.170e-14, 5.245e-15, 3.019e-15])
_H = np.array([7.249, 6.349, 6.682, 7.554, 8.382, 7.714, 6.549, 5.799, 5.382, 5.877, 7.263, 9.473,
               12.636, 16.149, 22.523, 29.740, 37.105, 45.546, 53.628, 53.298, 58.515, 60.828,
               63.822, 71.835, 88.667, 124.64, 181.05, 268.0])


def density(h_km):
    """Air density (kg/m^3) at altitude h_km."""
    h = np.asarray(h_km, dtype=float)
    i = np.clip(np.searchsorted(_BASE, h, side="right") - 1, 0, len(_BASE) - 1)
    return _RHO0[i] * np.exp(-(h - _BASE[i]) / _H[i])


# cumulative integral of dh / rho(h) on a fine grid, for fast lookups
_GRID = np.arange(REENTRY_ALT_KM, 4000.0 + 0.25, 0.25)
_INV_RHO = 1.0 / density(_GRID)
_CUM = np.concatenate([[0.0], np.cumsum(0.5 * (_INV_RHO[1:] + _INV_RHO[:-1]) * np.diff(_GRID))])


def _integral_from_reentry(h_km):
    return np.interp(h_km, _GRID, _CUM)


def mean_altitude_km(mean_motion_rev_day):
    n = np.asarray(mean_motion_rev_day, dtype=float) * 2 * np.pi / 86400.0
    return (MU_EARTH_KM3_S2 / n**2) ** (1.0 / 3.0) - EARTH_RADIUS_KM


def decay_rate_km_day(mean_motion, mean_motion_dot_field):
    """Present rate of semi-major-axis loss in km/day (positive = decaying)."""
    n = np.asarray(mean_motion, dtype=float)
    a = mean_altitude_km(n) + EARTH_RADIUS_KM
    ndot = 2.0 * np.asarray(mean_motion_dot_field, dtype=float)  # field is ndot/2
    return (2.0 / 3.0) * a * ndot / n


def days_to_reentry(h0_km, rate_km_day):
    """Integrate decay from h0 to REENTRY_ALT_KM. NaN where no meaningful decay."""
    h0 = np.asarray(h0_km, dtype=float)
    rate = np.asarray(rate_km_day, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        days = density(h0) / rate * _integral_from_reentry(np.minimum(h0, _GRID[-1]))
    days = np.where(h0 <= REENTRY_ALT_KM, 0.0, days)
    bad = ~np.isfinite(days) | (rate <= 0) | (days > MAX_YEARS * 365.25)
    return np.where(bad & (h0 > REENTRY_ALT_KM), np.nan, days)


def estimate_decay(df: pd.DataFrame) -> pd.DataFrame:
    """Add decay columns to a catalogue frame (needs MEAN_MOTION, MEAN_MOTION_DOT, BSTAR, ECCENTRICITY)."""
    h0 = mean_altitude_km(df["MEAN_MOTION"])
    rate = decay_rate_km_day(df["MEAN_MOTION"], df["MEAN_MOTION_DOT"])
    # a positive BSTAR (drag term) must agree with the measured decay, otherwise ignore it
    rate = np.where(df["BSTAR"].to_numpy(float) > 0, rate, np.nan)
    days = days_to_reentry(h0, rate)
    status = np.where(np.isnan(days), "no measurable decay",
                      np.where(df["ECCENTRICITY"].to_numpy(float) > 0.02, "low confidence (eccentric)", "estimate"))
    out = df.copy()
    out["decay_km_day"] = rate
    out["decay_days"] = days
    out["decay_days_low"] = days / UNCERTAINTY_FACTOR
    out["decay_days_high"] = days * UNCERTAINTY_FACTOR
    out["decay_status"] = status
    return out


def format_days(days: float) -> str:
    if days is None or not np.isfinite(days):
        return "—"
    if days < 1:
        return f"{days * 24:.0f} h"
    if days < 120:
        return f"{days:.0f} d"
    return f"{days / 365.25:.1f} yr"


def format_range(low: float, high: float) -> str:
    if low is None or not np.isfinite(low):
        return "—"
    return f"{format_days(low)} – {format_days(high)}"


def altitude_projection(h0_km: float, rate_km_day: float, n: int = 200):
    """(days, altitude_km) from now to re-entry using the same model; None if not decaying."""
    if not np.isfinite(rate_km_day) or rate_km_day <= 0 or h0_km <= REENTRY_ALT_KM:
        return None
    total = float(days_to_reentry(h0_km, rate_km_day))
    if not np.isfinite(total):
        return None
    h = np.linspace(h0_km, REENTRY_ALT_KM, n)
    t = density(h0_km) / rate_km_day * (_integral_from_reentry(h0_km) - _integral_from_reentry(h))
    return t, h


# ------------------------------------------------------------------ risk score
DEFAULT_WEIGHTS = {"perigee": 0.35, "decay": 0.35, "inclination": 0.20, "type": 0.10}
TYPE_SCORE = {"debris": 1.0, "rocket body": 0.8, "payload": 0.3}
PERIGEE_HIGH_KM, PERIGEE_LOW_KM = 700.0, 150.0
DECAY_FAST_DAYS, DECAY_SLOW_DAYS = 10.0, 3650.0


def classify_type(name: str) -> str:
    """Heuristic from the catalogue name: DEB -> debris, R/B -> rocket body, else payload."""
    u = str(name).upper()
    if " DEB" in u or u.endswith("DEB"):
        return "debris"
    if "R/B" in u:
        return "rocket body"
    return "payload"


def risk_components(df: pd.DataFrame) -> pd.DataFrame:
    """Four 0-1 sub-scores (1 = riskier). df must already have decay_days and perigee_km."""
    s_perigee = ((PERIGEE_HIGH_KM - df["perigee_km"]) / (PERIGEE_HIGH_KM - PERIGEE_LOW_KM)).clip(0, 1)
    d = df["decay_days"].to_numpy(float)
    with np.errstate(invalid="ignore"):
        s_decay = 1 - np.log10(np.maximum(d, DECAY_FAST_DAYS) / DECAY_FAST_DAYS) / np.log10(DECAY_SLOW_DAYS / DECAY_FAST_DAYS)
    s_decay = np.where(np.isfinite(d), np.clip(s_decay, 0, 1), 0.0)
    inc = df["INCLINATION"].to_numpy(float)
    s_incl = np.sin(np.radians(np.minimum(inc, 180 - inc)))  # share of Earth's area overflown
    s_type = df["OBJECT_NAME"].map(classify_type).map(TYPE_SCORE).to_numpy(float)
    return pd.DataFrame({"s_perigee": s_perigee.to_numpy(float), "s_decay": s_decay,
                         "s_inclination": s_incl, "s_type": s_type}, index=df.index)


def risk_score(components: pd.DataFrame, weights: dict | None = None) -> pd.Series:
    """0-100 weighted average of the sub-scores."""
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    total = sum(w.values()) or 1.0
    score = (w["perigee"] * components["s_perigee"] + w["decay"] * components["s_decay"]
             + w["inclination"] * components["s_inclination"] + w["type"] * components["s_type"])
    return 100.0 * score / total
