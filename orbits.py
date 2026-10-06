"""Orbital physics helpers. No network access; no Streamlit imports."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from skyfield.api import EarthSatellite, load, wgs84

MU_EARTH_KM3_S2 = 398600.4418  # Earth gravitational parameter
EARTH_RADIUS_KM = 6378.137  # WGS-84 equatorial radius
DATA_PATH = Path(__file__).with_name("active.json")

# Built-in timescale: uses bundled data, so nothing is downloaded.
TS = load.timescale(builtin=True)


def load_records(path: Path | str = DATA_PATH) -> list[dict]:
    """Load raw OMM JSON records from disk."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compute_altitudes(df: pd.DataFrame) -> pd.DataFrame:
    """Add perigee_km / apogee_km columns from MEAN_MOTION and ECCENTRICITY."""
    n_rad_s = df["MEAN_MOTION"].astype(float) * 2 * math.pi / 86400.0  # rev/day -> rad/s
    a_km = (MU_EARTH_KM3_S2 / n_rad_s**2) ** (1.0 / 3.0)
    ecc = df["ECCENTRICITY"].astype(float)
    out = df.copy()
    out["perigee_km"] = a_km * (1 - ecc) - EARTH_RADIUS_KM
    out["apogee_km"] = a_km * (1 + ecc) - EARTH_RADIUS_KM
    return out


def load_catalog(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Load the file into a DataFrame with altitudes computed for every object."""
    df = pd.DataFrame(load_records(path))
    df = df[df["MEAN_MOTION"] > 0].reset_index(drop=True)
    df["EPOCH_DT"] = pd.to_datetime(df["EPOCH"])
    return compute_altitudes(df)


def snapshot_date(df: pd.DataFrame) -> str:
    """Date of the newest EPOCH in the data (YYYY-MM-DD)."""
    return df["EPOCH_DT"].max().strftime("%Y-%m-%d")


def watchlist(df: pd.DataFrame, n: int = 50) -> pd.DataFrame:
    """The n lowest-perigee objects with display columns."""
    cols = {
        "OBJECT_NAME": "Name",
        "NORAD_CAT_ID": "NORAD ID",
        "perigee_km": "Perigee (km)",
        "apogee_km": "Apogee (km)",
        "INCLINATION": "Inclination (°)",
    }
    out = df.nsmallest(n, "perigee_km")[list(cols)].rename(columns=cols)
    return out.reset_index(drop=True)


def make_satellite(record: dict) -> EarthSatellite:
    return EarthSatellite.from_omm(TS, record)


def current_position(record: dict, when=None) -> dict:
    """Sub-satellite point (lat, lon, altitude km) at `when` (default: now)."""
    sat = make_satellite(record)
    t = when if when is not None else TS.now()
    sp = wgs84.geographic_position_of(sat.at(t))
    return {
        "lat": sp.latitude.degrees,
        "lon": sp.longitude.degrees,
        "alt_km": sp.elevation.km,
        "time": t.utc_datetime(),
    }


def ground_track(record: dict, n_orbits: int = 3, step_s: float = 30.0, start=None) -> pd.DataFrame:
    """Ground track covering n_orbits starting at `start` (default: now).

    Longitude jumps across the antimeridian are replaced by NaN rows so a
    plotted line breaks instead of streaking across the map.
    """
    sat = make_satellite(record)
    period_s = 86400.0 / float(record["MEAN_MOTION"])
    t0 = start if start is not None else TS.now()
    offsets = np.arange(0.0, n_orbits * period_s + step_s, step_s)
    t = TS.tt_jd(t0.tt + offsets / 86400.0)
    sp = wgs84.subpoint_of(sat.at(t))
    lat = np.asarray(sp.latitude.degrees)
    lon = np.asarray(sp.longitude.degrees)
    jumps = np.where(np.abs(np.diff(lon)) > 180)[0] + 1
    lat = np.insert(lat, jumps, np.nan)
    lon = np.insert(lon, jumps, np.nan)
    return pd.DataFrame({"lat": lat, "lon": lon})


def orbit_path_xyz(record: dict, n_points: int = 180, start=None) -> dict:
    """One full orbit in the inertial (TEME) frame, in km, plus the current point.

    Returns arrays x, y, z for the path and `now` = (x, y, z) at `start`.
    """
    sat = make_satellite(record)
    period_s = 86400.0 / float(record["MEAN_MOTION"])
    t0 = start if start is not None else TS.now()
    offsets = np.linspace(0.0, period_s, n_points)
    t = TS.tt_jd(t0.tt + offsets / 86400.0)
    x, y, z = sat.at(t).position.km
    nx, ny, nz = sat.at(t0).position.km
    return {"x": np.asarray(x), "y": np.asarray(y), "z": np.asarray(z), "now": (float(nx), float(ny), float(nz))}
